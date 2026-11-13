"""The Image page's JSON API: the shared draft, each model's options, images to edit and runs."""

from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from h3_app import catalog
from h3_app.jobs import JOBS
from h3_app.workspace_store import WorkspaceStore, install_owner_cookie
from h3_ui import application as gradio_app
from h3_ui.image.page import EDIT_MODE, GENERATE_MODE, GROUP_OF, COMPOSE, install_image_page, private_image_root
from h3_ui.job_bindings import owned_generation

API = "/image/api"


class ImagePageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The page takes its options, their defaults and their choices from the built views.
        cls.demo = gradio_app.build_ui()
        cls.pages = next(install.keywords["pages"] for install in cls.demo.h3_routes
                         if getattr(install, "func", None) is install_image_page)
        cls.services = cls.pages.services

    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.addCleanup(setattr, JOBS, "store", JOBS.store)
        self.store = JOBS.store = WorkspaceStore(self.root / "state")
        self.generated, self.suggested = [], []
        names = ("engine", "mode", "prompt", "reference_images", *self.services.job_fields)
        previous = JOBS.callbacks.get("image")
        JOBS.callbacks["image"] = owned_generation(self.generate, "image", input_names=names)
        self.addCleanup(lambda: JOBS.callbacks.update(image=previous) if previous else JOBS.callbacks.pop("image", None))
        self.h3 = {"model_profile": "Quality"}
        self.export = SimpleNamespace(settings={})
        from dataclasses import replace
        from h3_ui.image.page import ImagePageState

        pages = ImagePageState(replace(
            self.services, root=lambda: self.root / "pages", suggest=self.suggest, h3_values=lambda owner: self.h3,
            interrupt=lambda request, family, job_id: "Stopping.", library=lambda: None, followers=(self.export,)))
        app = FastAPI()
        install_owner_cookie(app, self.store)
        install_image_page(app, pages)
        self.client = TestClient(app)
        self.boot = self.ok(self.client.get(f"{API}/boot"))
        self.initial_state = self.boot["state"]
        self.boot["state"] = self.change("engine", engine="qwen")

    def generate(self, engine, mode, prompt, references, *values):
        self.generated.append({"engine": engine, "mode": mode, "prompt": prompt, "references": references,
                               **dict(zip(self.services.job_fields, values, strict=True))})
        target = self.root / "out" / f"image-{len(self.generated)}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), "blue").save(target)
        yield {"__type__": "update"}, "Sampling"
        yield [str(target)], "Done"

    def suggest(self, engine, mode, prompt, references, options):
        self.suggested.append((engine, mode, prompt, references, options))
        return prompt + " In golden light.", "Enhanced."

    def ok(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def post(self, route, **body):
        return self.ok(self.client.post(f"{API}{route}", json=body))

    def change(self, action="edit", name=None, **changes):
        return self.post("/settings", action=action, changes=changes, name=name)["state"]

    def upload(self, *names, size=(640, 480)):
        files = []
        for name in names:
            path = self.root / name
            Image.new("RGB", size, "red").save(path)
            files.append(("files", (name, path.read_bytes(), "image/png")))
        return self.ok(self.client.post(f"{API}/references/upload", files=files))["state"]

    def finished(self, kind):
        for _ in range(200):
            answer = self.ok(self.client.get(f"{API}/run?state=true"))
            if not answer["runs"][kind]["active"]:
                return answer
            time.sleep(0.05)
        self.fail("The run never finished.")

    def field(self, state, name):
        return next(row for row in state["fields"] if row["name"] == name)

    def test_official_h3_defaults_and_backend_reference_limits(self):
        self.assertEqual(self.initial_state["engine"], "h3")
        self.assertEqual(self.initial_state["options"]["image__backend"], "MiniMax H3 Image (Official)")
        self.assertEqual((self.initial_state["width"], self.initial_state["height"]), (4096, 4096))
        self.change("engine", engine="h3")
        self.post("/prompt", prompt="combine")
        state = self.upload(*(f"ref{i}.png" for i in range(16)))
        self.assertTrue(state["readiness"]["ready"])
        self.assertIn("16 images", state["guidance"])
        state = self.change(image__backend="Pruna P-H3-Image")
        self.assertIn("8 references", state["readiness"]["message"])
        self.assertNotIn("image__expand_prompt", {row["name"] for row in state["fields"]})
        state = self.change(image__backend="MiniMax H3 Image (Official)")
        self.assertTrue(state["readiness"]["ready"])

    def test_every_model_exposes_and_saves_variations(self):
        for engine in ("h3", "qwen", "flux3", "moss_image", "fibo2"):
            with self.subTest(engine=engine):
                state = self.change("engine", engine=engine)
                batch = state["batch"]
                self.assertEqual((batch["minimum"], batch["maximum"]), (1, 4))
                state = self.change(**{batch["name"]: 4})
                self.assertEqual(state["batch"]["value"], 4)
                self.assertEqual(self.ok(self.client.get(f"{API}/boot"))["state"]["batch"]["value"], 4)
                self.assertEqual(self.post("/settings", action="engine", changes={"engine": engine})["state"]["batch"]["value"], 4)

    def test_empty_clears_shared_inputs_and_layouts_and_persists(self):
        self.post("/prompt", prompt="Edit this bird")
        self.upload("bird.png")
        self.change(flux3__layout="layout", moss_image__layout="layout", qwen__batch_count=3)
        state = self.post("/empty")["state"]
        self.assertEqual((state["prompt"], state["references"], state["mode"]), ("", [], GENERATE_MODE))
        self.assertEqual(state["options"]["flux3__layout"], "")
        self.assertEqual(state["options"]["moss_image__layout"], "")
        self.assertEqual(state["batch"]["value"], 3)
        restored = self.ok(self.client.get(f"{API}/boot"))["state"]
        self.assertEqual((restored["prompt"], restored["references"]), ("", []))
        self.assertFalse(restored["readiness"]["ready"])

    def test_h3_prompt_assistance_passes_image_settings_and_acceptance_skips_local_expansion(self):
        self.change("engine", engine="h3")
        self.change(image__width=4736, image__height=3552)
        self.post("/prompt", prompt="three bowls")
        self.post("/enhance")
        result = self.finished("enhance")
        self.assertEqual(self.suggested[0][:4], ("h3", GENERATE_MODE, "three bowls", []))
        options = self.suggested[0][4]
        self.assertEqual((options["h3_width"], options["h3_height"]), (4736, 3552))
        suggestion = result["runs"]["enhance"]["suggestion"]
        state = self.post("/prompt", prompt=suggestion, accepted=True)["state"]
        self.assertFalse(state["options"]["image__expand_prompt"])

    def test_every_job_option_has_a_place_on_the_page(self):
        own = [name for name in self.services.job_fields if not name.startswith("h3__")]
        self.assertTrue(set(own) <= set(self.services.fields))
        for name in own:
            self.assertIn(name.partition("__")[2], {*GROUP_OF, *COMPOSE}, name)
        self.assertEqual([engine["id"] for engine in self.boot["engines"]],
                         ["h3", "qwen", "flux3", "moss_image", "fibo2"])
        self.assertEqual({engine: [tier["label"] for tier in tiers] for engine, tiers in self.boot["sizes"].items()},
                         {"qwen": ["1K", "2K"], "h3": ["4M", "8M", "16M"], "flux3": ["1K", "2K"],
                          "moss_image": ["1M", "2M"], "fibo2": ["1M", "2M"]})

    def test_a_qwen_preset_sets_its_sampling_and_a_change_is_counted(self):
        state = self.change(qwen__preset="Quality")
        self.assertEqual(state["options"]["qwen__steps"], 40)
        self.assertEqual(state["options"]["qwen__turbo_variant"], "Off")
        self.assertEqual(state["options"]["qwen__accelerator"], "Spectrum (Quality)")
        self.assertEqual(state["changed"], {})
        state = self.change(qwen__steps=12)
        self.assertEqual(state["changed"], {"0": 1})
        self.assertEqual(self.field(state, "qwen__steps")["preset"], 40)
        self.assertEqual(self.change("restore")["changed"], {})
        fast = self.change(qwen__preset="Fast")["options"]
        self.assertEqual((fast["qwen__steps"], fast["qwen__cfg"], fast["qwen__accelerator"]), (8, 1.0, "Off"))

    def test_options_show_only_where_they_apply(self):
        state = self.boot["state"]
        names = lambda found: {row["name"] for row in found["fields"]}
        self.assertNotIn("qwen__negative_prompt", names(state))
        self.assertNotIn("qwen__lightning_strength", names(state))
        self.assertTrue(self.field(state, "qwen__cache_device")["disabled"])
        self.assertIn("qwen__negative_prompt", names(self.change(qwen__cfg=3)))
        self.upload("a.png")
        self.assertFalse(self.field(self.boot_state(), "qwen__cache_device")["disabled"])
        state = self.change("engine", engine="fibo2")
        self.assertFalse(self.field(state, "fibo2__steps")["disabled"])
        state = self.change(fibo2__model_choice="Edit Turbo")
        self.assertEqual((state["options"]["fibo2__steps"], state["options"]["fibo2__cfg"]), (4, 1))
        self.assertTrue(self.field(state, "fibo2__steps")["disabled"])
        state = self.change("engine", engine="h3")
        self.assertTrue({"image__width", "image__height", "image__seed",
                         "image__backend", "image__expand_prompt"} <= names(state))
        self.assertTrue(state["can_enhance"])

    def boot_state(self):
        return self.ok(self.client.get(f"{API}/boot"))["state"]

    def test_h3_quick_sizes_accept_every_preset_across_three_tiers(self):
        self.change("engine", engine="h3")
        for tier in self.boot["sizes"]["h3"]:
            for size in tier["sizes"]:
                with self.subTest(tier=tier["label"], name=size["name"]):
                    state = self.change("resolution", name=size["name"])
                    self.assertEqual((state["width"], state["height"]), (size["width"], size["height"]))
                    self.assertEqual(state["options"]["image__edit_size"], catalog.QWEN_EDIT_SIZE_MANUAL)

    def test_a_named_size_sets_the_dimensions_and_overrides_the_edit_size(self):
        name = self.boot["sizes"]["qwen"][1]["sizes"][0]["name"]
        state = self.change("resolution", name=name)
        size = self.boot["sizes"]["qwen"][1]["sizes"][0]
        self.assertEqual((state["width"], state["height"]), (size["width"], size["height"]))
        self.assertEqual(state["options"]["qwen__output_resolution_2k"], name)
        self.assertIsNone(state["options"]["qwen__output_resolution_1k"])
        self.assertEqual(state["options"]["qwen__edit_size"], catalog.QWEN_EDIT_SIZE_MANUAL)
        state = self.change(qwen__edit_size=catalog.QWEN_EDIT_SIZE_MATCH)
        self.assertIsNone(state["options"]["qwen__output_resolution_2k"])
        self.change("resolution", name=name)
        self.assertIsNone(self.change(qwen__width=1024)["options"]["qwen__output_resolution_2k"])
        response = self.client.post(f"{API}/settings", json={"action": "resolution", "name": "9K · 1:1 · 9×9"})
        self.assertEqual(response.status_code, 400)

    def test_values_are_checked_against_the_option(self):
        for changes in ({"qwen__steps": 500}, {"qwen__sampler_name": "nope"}, {"qwen__batch_edit_inputs": "yes"},
                        {"unknown": 1}, {"qwen__width": 1025}, {"qwen__height": 1023.9}):
            response = self.client.post(f"{API}/settings", json={"changes": changes})
            self.assertEqual(response.status_code, 400, changes)
        self.assertEqual(self.change(qwen__steps="12")["options"]["qwen__steps"], 12)
        self.assertEqual(self.change(qwen__width=1056)["options"]["qwen__width"], 1056)

    def test_prompt_and_layout_save_together_before_generation(self):
        self.change("engine", engine="flux3")
        self.post("/prompt", prompt="New prompt", layout={"name": "flux3__layout", "value": "New boxes"})
        self.post("/generate")
        self.finished("generate")
        self.assertEqual((self.generated[0]["prompt"], self.generated[0]["flux3__layout"]), ("New prompt", "New boxes"))
        stale = {"prompt": "Wrong draft", "layout": {"name": "moss_image__layout", "value": ""}}
        self.assertEqual(self.client.post(f"{API}/prompt", json=stale).status_code, 400)
        self.assertEqual(self.boot_state()["prompt"], "New prompt")

    def test_rejected_upload_batch_preserves_draft_and_files(self):
        state = self.upload("keep.png")
        folder = Path(state["references"][0]["path"]).parent
        before = set(folder.iterdir())
        def post(names):
            files = [("files", (name, b"data", "application/octet-stream")) for name in names]
            return self.client.post(f"{API}/references/upload", files=files)

        for names in (["valid.png", "invalid.txt"], ["overflow.png"] * 24):
            self.assertEqual(post(names).status_code, 400)
            self.assertEqual(set(folder.iterdir()), before)
        with patch.object(self.page_state(), "save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                post(["new.png"])
        self.assertEqual(set(folder.iterdir()), before)
        self.assertEqual(self.boot_state()["references"], state["references"])

    def test_existing_draft_moves_outside_both_public_output_roots(self):
        from dataclasses import replace
        from h3_ui.image.page import ImagePageState

        for output in (self.root / "public" / "nested", self.root / "separate"):
            runtime = SimpleNamespace(outputs_dir=output, output_dir=self.root / "public")
            old = output / ".image-page"
            pages = ImagePageState(replace(self.services, root=lambda: old, followers=()))
            reference = pages.uploads("owner") / "reference.png"
            reference.write_bytes(b"reference")
            state = pages.load("owner")
            state.update(prompt="Private draft", references=[str(reference)])
            pages.save("owner", state)
            workspace = output / "workspace" if output.name == "nested" else self.root / "workspace"
            new = private_image_root(runtime, workspace)
            self.assertFalse(old.exists())
            self.assertFalse(any(new.is_relative_to(root) for root in (output, runtime.output_dir)))
            self.assertEqual(private_image_root(runtime, workspace), new)
            moved = ImagePageState(replace(pages.services, root=lambda: new)).load("owner")
            self.assertEqual(moved["prompt"], "Private draft")
            self.assertEqual(Path(moved["references"][0]).read_bytes(), b"reference")

    def test_adding_an_image_means_editing_and_the_order_can_change(self):
        self.assertEqual(self.boot["state"]["mode"], GENERATE_MODE)
        state = self.upload("a.png", "b.png", "c.png")
        self.assertEqual(state["mode"], EDIT_MODE)
        self.assertEqual([item["tag"] for item in state["references"]], ["<image1>", "<image2>", "<image3>"])
        self.assertTrue(state["references"][0]["detail"].startswith("a.png"))
        state = self.post("/references/move", **{"from": 2, "to": 0})["state"]
        self.assertTrue(state["references"][0]["detail"].startswith("c.png"))
        state = self.post("/references/remove", index=1)["state"]
        self.assertEqual(len(state["references"]), 2)
        self.assertEqual(self.client.post(f"{API}/references/move", json={"from": 0, "to": 5}).status_code, 400)
        self.assertEqual(self.client.get(state["references"][0]["url"]).status_code, 200)
        self.assertEqual(self.client.get(f"{API}/file", params={"path": str(self.root / "a.png")}).status_code, 404)
        # Each model names the images its own way, and separate edits have no shared name.
        self.assertEqual(self.change(qwen__batch_edit_inputs=True)["references"][0]["tag"], None)
        self.assertEqual(self.change("engine", engine="h3")["references"][1]["tag"], "<Picture 2>")
        self.assertEqual(self.change("engine", engine="flux3")["references"][0]["tag"], "<ref_image_0>")
        self.assertEqual(self.post("/references/clear")["state"]["references"], [])

    def test_moss_generates_only_and_keeps_the_draft_for_other_models(self):
        self.upload("a.png")
        self.post("/prompt", prompt="A fox")
        state = self.change("engine", engine="moss_image")
        self.assertEqual((state["mode"], state["editing"], state["generate_only"]), (GENERATE_MODE, False, True))
        self.assertTrue(state["readiness"]["ready"])
        self.assertIsNone(state["edit_size"])
        self.assertEqual(state["preset"]["value"], catalog.DEFAULT_MOSS_IMAGE_PRESET)
        state = self.change(moss_image__preset="Quality")
        self.assertEqual((state["options"]["moss_image__steps"], state["options"]["moss_image__cfg"]), (50, 3.0))
        self.assertTrue(self.change("engine", engine="qwen")["editing"])

    def test_readiness_follows_the_draft(self):
        self.assertFalse(self.boot["state"]["readiness"]["ready"])
        self.assertTrue(self.post("/prompt", prompt="A fox")["state"]["readiness"]["ready"])
        state = self.change("mode", mode=EDIT_MODE)
        self.assertIn("at least one image", state["readiness"]["message"])
        self.assertEqual(self.client.post(f"{API}/generate").status_code, 400)
        state = self.change("engine", engine="fibo2")
        self.upload(*[f"{index}.png" for index in range(5)])
        self.assertIn("4 references", self.boot_state()["readiness"]["message"])

    def test_generation_takes_the_job_path_with_the_models_options(self):
        self.post("/prompt", prompt="A fox")
        self.change(qwen__steps=11, qwen__batch_count=2)
        self.post("/generate")
        answer = self.finished("generate")
        self.assertIsNone(answer["runs"]["generate"]["error"])
        self.assertIn("Qwen Image 2.1", answer["runs"]["generate"]["status"])
        sent = self.generated[0]
        self.assertEqual((sent["engine"], sent["mode"], sent["prompt"], sent["references"]),
                         ("qwen", GENERATE_MODE, "A fox", []))
        self.assertEqual((sent["qwen__steps"], sent["qwen__batch_count"]), (11, 2))
        # P-H3-Image runs on what the MiniMax H3 page saved.
        self.assertEqual(sent["h3__model_profile"], "Quality")
        images = answer["state"]["results"]["images"]
        self.assertEqual(len(images), 1)
        self.assertEqual(self.client.get(images[0]["url"]).status_code, 200)
        state = self.post("/references/use-result", path=images[0]["path"])["state"]
        self.assertEqual((state["mode"], len(state["references"])), (EDIT_MODE, 1))
        self.post("/generate")
        self.finished("generate")
        self.assertEqual(self.generated[1]["mode"], EDIT_MODE)
        self.assertEqual(len(self.generated[1]["references"]), 1)

    def test_inactive_images_are_not_sent_when_generating(self):
        self.upload("a.png")
        self.post("/prompt", prompt="A fox")
        self.change("mode", mode=GENERATE_MODE)
        self.post("/generate")
        self.finished("generate")
        self.assertEqual(self.generated[0]["references"], [])

    def test_completed_images_survive_an_interrupted_or_failed_batch(self):
        from h3_app.jobs import JobCancelled

        for failure in (JobCancelled(), RuntimeError("Later variation failed")):
            with self.subTest(failure=type(failure).__name__):
                def partial(*args):
                    yield from self.generate(*args)
                    # The user can keep editing while a later variation runs.
                    self.post("/prompt", prompt="The next draft")
                    raise failure

                names = ("engine", "mode", "prompt", "reference_images", *self.services.job_fields)
                JOBS.callbacks["image"] = owned_generation(partial, "image", input_names=names)
                self.post("/prompt", prompt="A fox")
                self.post("/generate")
                answer = self.finished("generate")
                run = answer["runs"]["generate"]
                if isinstance(failure, JobCancelled):
                    self.assertEqual(run["status"], "Interrupted.")
                else:
                    self.assertIn("Later variation failed", run["error"])
                results = answer["state"]["results"]["images"]
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0]["path"], answer["progress"][0]["path"])
                restored = self.boot_state()
                self.assertEqual(restored["results"]["images"], results)
                self.assertEqual(restored["prompt"], "The next draft")
                self.assertEqual(self.client.get(results[0]["url"]).status_code, 200)

    def test_a_suggestion_is_reviewed_and_accepting_it_settles_the_models_helpers(self):
        self.change("engine", engine="flux3")
        self.post("/prompt", prompt="A fox")
        self.post("/writer", changes={"backend": "OpenAI-compatible", "openai_api_key": "secret"})
        self.post("/enhance")
        run = self.finished("enhance")["runs"]["enhance"]
        self.assertEqual((run["suggestion"], run["original"]), ("A fox In golden light.", "A fox"))
        engine, mode, prompt, references, options = self.suggested[0]
        self.assertEqual((engine, mode, prompt, options["writer_backend"], options["writer_openai_key"]),
                         ("flux3", GENERATE_MODE, "A fox", "OpenAI-compatible", "secret"))
        self.assertTrue(self.boot_state()["options"]["flux3__expand_prompt"])
        state = self.post("/prompt", prompt=run["suggestion"], accepted=True)["state"]
        self.assertFalse(state["options"]["flux3__expand_prompt"])
        saved = (self.root / "pages").rglob("settings.json")
        self.assertNotIn("secret", next(saved).read_text(encoding="utf-8"))

    def test_another_page_hands_over_a_picture_and_the_export_follows_the_options(self):
        owner = next(iter(self.pages_owner()))
        source = self.root / "frame.png"
        Image.new("RGB", (32, 32), "green").save(source)
        self.page_state().receive(owner, "h3", str(source))
        state = self.boot_state()
        self.assertEqual((state["engine"], state["mode"], len(state["references"])), ("h3", EDIT_MODE, 1))
        self.change(qwen__steps=13)
        followed = self.export.settings.get(owner)
        self.assertEqual(set(followed), set(self.services.job_fields))
        self.assertEqual((followed["qwen__steps"], followed["h3__model_profile"]), (13, "Quality"))

    def page_state(self):
        return self.export.settings.load.__self__

    def pages_owner(self):
        return [path.name for path in (self.root / "pages").iterdir()]


if __name__ == "__main__":
    unittest.main()
