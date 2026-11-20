"""The LTX 2.5 page's JSON API: the draft, keyframes and references, the writer, runs and model downloads."""

from dataclasses import replace
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from h3_app.jobs import JOBS
from h3_app.workspace_store import WorkspaceStore, install_owner_cookie
from h3_ui import application as gradio_app
from h3_ui.job_bindings import owned_generation
from h3_ui.ltx.page import JOB_ORDER, LtxPageState, install_ltx_page, media_targets, private_ltx_root

API = "/ltx/api"


def png(name="frame.png"):
    buffer = io.BytesIO()
    Image.new("RGB", (32, 18), "teal").save(buffer, format="PNG")
    return (name, buffer.getvalue(), "image/png")


class LtxPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The page takes its options, their defaults and their limits from the built view.
        cls.demo = gradio_app.build_ui()
        cls.pages = next(install.keywords["pages"] for install in cls.demo.h3_routes
                         if getattr(install, "func", None) is install_ltx_page)
        cls.services = cls.pages.services

    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.addCleanup(setattr, JOBS, "store", JOBS.store)
        self.store = JOBS.store = WorkspaceStore(self.root / "state")
        self.generated, self.written, self.stopped, self.prepared = [], [], [], []
        self.hold = None
        previous = JOBS.callbacks.get("ltx")
        JOBS.callbacks["ltx"] = owned_generation(self.generator, "ltx", input_names=JOB_ORDER)
        self.addCleanup(lambda: JOBS.callbacks.update({"ltx": previous}) if previous else JOBS.callbacks.pop("ltx", None))
        self.missing = ["ltx25_iclora_ingredients.safetensors"]
        self.state = LtxPageState(replace(
            self.services, root=lambda: self.root / "pages", enhance=self.writer, workflows=self.workflows,
            prepare=self.prepare, prepare_all=lambda: self.prepare(None), interrupt=self.interrupt,
            library=lambda: None))
        app = FastAPI()
        install_owner_cookie(app, self.store)
        install_ltx_page(app, self.state)
        self.client = TestClient(app)
        self.boot = self.ok(self.client.get(f"{API}/boot"))

    # Fakes -----------------------------------------------------------------
    def generator(self, *values):
        self.generated.append(dict(zip(JOB_ORDER, values, strict=True)))
        target = self.root / "out" / f"ltx-{len(self.generated)}.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"\x00\x00\x00\x18ftypmp42")
        yield None, "Sampling"
        if self.hold is not None:
            self.hold.wait(5)
        yield str(target), "<b>Done</b> in 9 s"

    def writer(self, prompt, *arguments, **options):
        self.written.append((prompt, arguments, options))
        return prompt + " The camera pushes in.", "Prompt ready."

    def workflows(self):
        return [{"label": "Text / image to video — single stage", "id": "t2v", "description": "Official.",
                 "inputs": "Prompt.", "total": 5, "missing": [], "json": "/ltx25-workflows/t2v.json"},
                {"label": "Ingredients / reference sheet — IC-LoRA", "id": "ing", "description": "Sheet.",
                 "inputs": "Pictures.", "total": 6, "missing": list(self.missing), "json": "/ltx25-workflows/ing.json"}]

    def prepare(self, label):
        self.prepared.append(label)
        yield "Downloading models for **Ingredients**", ""
        self.missing = []
        yield "Ready: installed all models for **Ingredients**.", ""

    def interrupt(self, request, family, job_id):
        self.stopped.append((family, job_id))
        return "Stopping."

    # Helpers ---------------------------------------------------------------
    def ok(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def refused(self, response, words):
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn(words, response.json()["error"])

    def post(self, route, **body):
        return self.ok(self.client.post(f"{API}{route}", json=body))

    def change(self, action="edit", name=None, **changes):
        return self.post("/settings", action=action, changes=changes, name=name)["state"]

    def upload(self, slot, *files):
        return self.client.post(f"{API}/media/upload", data={"slot": slot},
                                files=[("files", file) for file in files or (png(),)])

    def settle(self, kind):
        for _ in range(250):
            answer = self.ok(self.client.get(f"{API}/run?state=true&models=true"))
            run = answer["runs"][kind]
            if run and not run["active"]:
                return answer
            time.sleep(0.02)
        self.fail(f"The {kind} run did not finish.")

    # Tests -----------------------------------------------------------------
    def test_boot_draws_the_view_defaults_and_the_workflows(self):
        state = self.boot["state"]
        self.assertEqual(state["mode"], "Text to video")
        self.assertEqual((state["width"], state["height"], state["ratio"], state["tier"]), (960, 544, "16:9", "Medium"))
        self.assertEqual(state["frames"], 121)
        self.assertEqual(state["compose"]["duration"]["maximum"], 20)
        self.assertEqual({row["name"] for row in state["fields"] if row["group"] < 3},
                         {"seed", "cfg", "sampler_name", "fps", "width", "height", "model_choice"})
        self.assertFalse(state["readiness"]["ready"])
        self.assertEqual(self.boot["missing"], 1)
        self.assertEqual(self.boot["modes"], ["Text to video", "Image to video", "Reference images"])
        self.assertEqual([engine["id"] for engine in self.boot["engines"]], ["h3", "ltx"])
        self.assertTrue(self.boot["gpu"])

    def test_modes_need_their_media_and_reference_locks_the_size(self):
        self.post("/prompt", prompt="A lighthouse at dusk.")
        state = self.change("mode", mode="Image to video")
        self.assertIn("start keyframe", state["readiness"]["message"])
        state = self.ok(self.upload("first_image"))["state"]
        self.assertTrue(state["readiness"]["ready"])
        self.assertTrue(state["keyframes"]["first_image"]["url"].startswith(f"{API}/file?path="))
        self.change(name="16:9 Large", action="size")
        state = self.change("mode", mode="Reference images")
        self.assertEqual((state["width"], state["height"], state["options"]["fps"], state["frames"]), (768, 448, 24, 121))
        self.assertTrue(state["locked"])
        self.assertIn("reference images", state["readiness"]["message"])
        self.refused(self.client.post(f"{API}/settings", json={"action": "size", "name": "1:1 Small", "changes": {}}),
                     "set by Reference images")
        self.refused(self.client.post(f"{API}/settings", json={"changes": {"duration": 8}}), "set by Reference images")

    def test_options_are_checked_and_sizes_snap(self):
        self.refused(self.client.post(f"{API}/settings", json={"changes": {"cfg": 9}}), "between")
        self.refused(self.client.post(f"{API}/settings", json={"changes": {"bogus": 1}}), "Unknown setting")
        state = self.change(width=1000)
        self.assertEqual(state["width"], 992)
        self.assertEqual((state["ratio"], state["tier"]), ("Custom", "Exact"))
        state = self.change(cfg=1.5, fps=30)
        self.assertEqual(state["changed"], {"0": 1, "1": 1})
        state = self.change("restore")
        self.assertEqual(state["changed"], {})
        self.assertEqual(state["width"], 992)

    def test_a_shorter_video_keeps_the_middle_keyframe_inside(self):
        state = self.change(middle_time=4.0)
        state = self.change(duration=3)
        self.assertLess(state["options"]["middle_time"], 3)

    def test_references_take_at_most_nine_and_can_be_removed(self):
        state = self.ok(self.upload("references", *(png(f"r{i}.png") for i in range(3))))["state"]
        self.assertEqual(len(state["references"]), 3)
        self.refused(self.upload("references", *(png(f"s{i}.png") for i in range(7))), "at most 9")
        state = self.post("/media/remove", slot="references", index=1)["state"]
        self.assertEqual(len(state["references"]), 2)
        self.refused(self.upload("first_image", png("a.png"), png("b.png")), "one picture")
        self.refused(self.upload("somewhere"), "Choose a keyframe")

    def test_generate_sends_only_the_media_the_mode_uses(self):
        self.post("/prompt", prompt="A lighthouse at dusk.")
        self.upload("references")
        self.change("mode", mode="Image to video")
        self.upload("first_image")
        self.upload("end_image")
        self.post("/generate")
        answer = self.settle("generate")
        values = self.generated[-1]
        self.assertEqual(values["mode"], "Image to video")
        self.assertTrue(values["first_image"] and values["end_image"])
        self.assertIsNone(values["middle_image"])
        self.assertIsNone(values["reference_images"])
        self.assertEqual((values["width"], values["height"], values["model_choice"]), (960, 544, "INT8 ConvRot"))
        videos = answer["state"]["results"]["videos"]
        self.assertEqual(len(videos), 1)
        self.assertIn("Done", answer["runs"]["generate"]["status"])
        self.assertEqual(self.client.get(videos[0]["url"]).status_code, 200)

        self.change("mode", mode="Text to video")
        self.post("/generate")
        answer = self.settle("generate")
        self.assertIsNone(self.generated[-1]["first_image"])
        self.assertEqual(answer["state"]["results"]["videos"][0]["name"], "ltx-2.mp4")

    def test_generate_refuses_an_unready_draft(self):
        self.refused(self.client.post(f"{API}/generate"), "Write a video prompt")

    def test_a_running_video_can_be_interrupted(self):
        self.hold = threading.Event()
        self.post("/prompt", prompt="Waves.")
        self.post("/generate")
        for _ in range(250):
            run = self.ok(self.client.get(f"{API}/run"))["runs"]["generate"]
            if run["status"] == "Sampling":
                break
            time.sleep(0.02)
        self.refused(self.client.post(f"{API}/generate"), "already being made")
        self.refused(self.client.post(f"{API}/models/prepare", json={}), "Wait for the video")
        answer = self.post("/interrupt")
        self.assertEqual(answer["message"], "Stopping.")
        self.assertEqual(self.stopped[0][0], "ltx")
        self.hold.set()
        self.settle("generate")

    def test_writer_previews_a_suggestion_with_the_keyframes(self):
        self.post("/prompt", prompt="A fox.")
        self.change("mode", mode="Image to video")
        self.upload("first_image")
        self.post("/enhance")
        answer = self.settle("enhance")
        run = answer["runs"]["enhance"]
        self.assertEqual(run["suggestion"], "A fox. The camera pushes in.")
        self.assertEqual(run["original"], "A fox.")
        prompt, arguments, options = self.written[0]
        # (model, key, mode, start, middle, end, duration, width, height)
        self.assertEqual(arguments[2], "Image to video")
        self.assertTrue(arguments[3])
        self.assertEqual(arguments[6:], (5, 960, 544))
        self.assertEqual(options["backend"], "Gemini")
        # The suggestion is a preview: the saved prompt is unchanged until accepted.
        self.assertEqual(answer["state"]["prompt"], "A fox.")

    def test_model_downloads_report_and_refresh_the_workflows(self):
        self.refused(self.client.post(f"{API}/models/prepare", json={"workflow": "Nope"}), "official workflows")
        self.post("/models/prepare", workflow="Ingredients / reference sheet — IC-LoRA")
        answer = self.settle("models")
        self.assertEqual(self.prepared, ["Ingredients / reference sheet — IC-LoRA"])
        self.assertIn("Ready", answer["runs"]["models"]["status"])
        self.assertEqual(answer["missing"], 0)

    def test_media_hands_pictures_to_keyframes_and_the_sheet(self):
        source = self.root / "library.png"
        Image.new("RGB", (16, 16)).save(source)
        from types import SimpleNamespace
        from h3_app.workspace_store import request_owner

        owner = request_owner(SimpleNamespace(cookies=dict(self.client.cookies), headers={}), self.store)
        targets = media_targets(self.state)
        self.assertEqual(set(targets), {"LTX · Start keyframe", "LTX · Middle keyframe", "LTX · End keyframe",
                                        "LTX · Reference images"})
        self.assertIn("Start keyframe", targets["LTX · Start keyframe"](owner, str(source)))
        state = self.ok(self.client.get(f"{API}/run?state=true"))["state"]
        self.assertEqual(state["mode"], "Image to video")
        self.assertIsNotNone(state["keyframes"]["first_image"])
        targets["LTX · Reference images"](owner, str(source))
        state = self.ok(self.client.get(f"{API}/run?state=true"))["state"]
        self.assertEqual((state["mode"], len(state["references"]), state["width"]), ("Reference images", 1, 768))

    def test_another_owner_cannot_read_a_keyframe(self):
        url = self.ok(self.upload("first_image"))["state"]["keyframes"]["first_image"]["url"]
        other = TestClient(self.client.app)
        self.assertEqual(other.get(url).status_code, 404)

    def test_keys_are_not_written_to_disk(self):
        self.post("/writer", changes={"gemini_api_key": "secret-key"})
        saved = next((self.root / "pages").rglob("settings.json")).read_text(encoding="utf-8")
        self.assertNotIn("secret-key", saved)

    def test_private_root_stays_outside_the_output_roots(self):
        class Runtime:
            outputs_dir = self.root / "workspace" / "outputs"
            output_dir = self.root / "workspace"
        root = private_ltx_root(Runtime(), self.root / "workspace" / "jobs")
        self.assertFalse(root.is_relative_to(Runtime.output_dir.resolve()))


if __name__ == "__main__":
    unittest.main()
