"""The MiniMax H3 page's JSON API: settings transitions, media, references and runs."""

from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from h3_app.contracts import GENERATION_FIELDS
from h3_app.jobs import JOBS
from h3_app.workspace_store import WorkspaceStore, install_owner_cookie
from h3_ui.h3.handlers import H3Handlers, H3Services
from h3_ui.h3.page import MEDIA_FIELDS, SETTING_FIELDS, H3PageServices, install_h3_page, setting_defaults
from h3_ui.job_bindings import owned_generation

API = "/h3/api"


class H3PageTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.addCleanup(setattr, JOBS, "store", JOBS.store)
        self.store = JOBS.store = WorkspaceStore(self.root / "state")
        self.generated = []
        self.enhanced = []
        runtime = SimpleNamespace(memory_profile="unknown", output_dir=self.root / "out", outputs_dir=self.root / "out")
        handlers = H3Handlers(H3Services(runtime_config=lambda: runtime, generate=self.generate, download_url=None))
        previous = JOBS.callbacks.get("h3")
        JOBS.callbacks["h3"] = owned_generation(
            handlers.generate_for_ui, "h3", ("batch_count", *GENERATION_FIELDS, "preset"), metadata_output=True)
        self.addCleanup(lambda: JOBS.callbacks.update(h3=previous) if previous else JOBS.callbacks.pop("h3", None))
        self.remake = SimpleNamespace(settings={})
        self.storyboard = SimpleNamespace(settings={}, capture_settings=None)
        self.services = H3PageServices(
            root=lambda: self.root / "pages", resolve=handlers.resolve_request_settings,
            describe=handlers.describe_settings, enhance=self.enhance,
            interrupt=lambda request, family, job_id: "Stopping.", followers=(self.remake, self.storyboard))
        app = FastAPI()
        install_owner_cookie(app, self.store)
        install_h3_page(app, self.services)
        self.client = TestClient(app)
        self.boot = self.ok(self.client.get(f"{API}/boot"))

    def generate(self, **values):
        self.generated.append(values)
        target = self.root / "out" / f"video-{len(self.generated)}.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"video")
        yield None, "Sampling"
        yield str(target), "Done"

    def enhance(self, prompt, backend, *rest, **options):
        self.enhanced.append((prompt, backend, rest, options))
        return prompt + " In golden light.", "Enhanced."

    def ok(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def change(self, action, **changes):
        return self.ok(self.client.post(f"{API}/settings", json={"action": action, "changes": changes}))["state"]

    def upload(self, slot, name="frame.png", size=(1920, 1080)):
        path = self.root / name
        if name.endswith(".png"):
            Image.new("RGB", size, "red").save(path)
        else:
            path.write_bytes(b"RIFF")
        with path.open("rb") as handle:
            return self.ok(self.client.post(f"{API}/media/upload?slot={slot}", files={"file": (name, handle)}))["state"]

    def finished(self, kind):
        for _ in range(200):
            answer = self.ok(self.client.get(f"{API}/run?state=true"))
            if not answer["runs"][kind]["active"]:
                return answer
            time.sleep(0.05)
        self.fail("The run never finished.")

    def test_defaults_cover_every_generation_setting(self):
        defaults = setting_defaults()
        self.assertEqual(set(defaults), set(SETTING_FIELDS))
        self.assertEqual({*SETTING_FIELDS, *MEDIA_FIELDS, "prompt"}, {*GENERATION_FIELDS, "preset", "batch_count"})
        self.assertEqual(self.boot["state"]["values"], defaults)
        self.assertEqual(self.boot["state"]["values"]["result_format"], "Video")

    def test_empty_clears_prompt_and_all_media_and_persists(self):
        self.change("mode", mode="Reference media")
        self.upload("ref_image_1")
        self.upload("first_image", "first.png")
        self.upload("ref_video_1", "reference.mp4")
        self.upload("ref_audio_1", "voice.wav")
        before = self.change("batch_count", batch_count=3)
        self.ok(self.client.post(f"{API}/prompt", json={"prompt": "A bird <Image 1>"}))
        state = self.ok(self.client.post(f"{API}/empty"))["state"]
        self.assertEqual(state["prompt"], "")
        self.assertEqual(state["media"], {})
        self.assertEqual(state["values"]["mode"], "Text to video")
        self.assertEqual(state["values"]["batch_count"], before["values"]["batch_count"])
        restored = self.ok(self.client.get(f"{API}/boot"))["state"]
        self.assertEqual((restored["prompt"], restored["media"]), ("", {}))

    def test_a_recipe_sets_its_sampling_and_a_change_is_counted(self):
        state = self.change("preset", preset="Quality")
        self.assertEqual(state["values"]["text_encoder"], "INT8 ConvRot")
        self.assertEqual(state["values"]["sla_preset"], "Quality")
        self.assertEqual(state["differences"], {})
        state = self.change("sla_preset", sla_preset="Fast")
        self.assertEqual(state["differences"]["sla_preset"], ["Quality", "Fast"])
        self.assertIn("Speed &amp; attention", state["summary"])
        state = self.change("restore")
        self.assertEqual(state["differences"], {})

    def test_generation_mode_keeps_each_modes_steps(self):
        self.assertEqual(self.boot["state"]["locked"]["steps"], 6)
        self.change("turbo_variant", turbo_variant="LightX2V v1.1 / 8-step 768p")
        state = self.change("steps", steps=9)
        self.assertEqual(state["locked"], {})
        normal = self.change("generation_mode", generation_mode="Normal")
        self.assertEqual(normal["values"]["steps"], 15)
        turbo = self.change("generation_mode", generation_mode="Turbo")
        self.assertEqual(turbo["values"]["steps"], 9)

    def test_bf16_requires_offload_and_gives_the_choice_back(self):
        state = self.change("text_encoder", text_encoder="BF16")
        self.assertTrue(state["values"]["stage_model_offload"])
        self.assertTrue(state["locked"]["stage_model_offload"])
        state = self.change("text_encoder", text_encoder="NVFP4 / AWQ")
        self.assertFalse(state["values"]["stage_model_offload"])

    def test_a_decoder_choice_sets_one_decoder(self):
        state = self.change("decoder:TensorRT")
        self.assertEqual(state["decoder"], "TensorRT")
        self.assertTrue(state["values"]["use_trt_vae"])
        self.assertFalse(state["values"]["use_sol_vae"])

    def test_sizes_come_from_the_tier_and_typed_sizes_snap(self):
        answer = self.ok(self.client.post(f"{API}/settings", json={
            "action": "resolution", "changes": {}, "size": {"tier": "fast", "name": "16:9 · 1920×1088"}}))
        self.assertEqual((answer["state"]["values"]["width"], answer["state"]["values"]["height"]), (1920, 1088))
        state = self.change("latent_upscale", latent_upscale=False)
        state = self.change("width", width=1000)
        self.assertEqual(state["values"]["width"], 992)

    def test_unknown_or_wrong_settings_are_refused(self):
        response = self.client.post(f"{API}/settings", json={"action": "edit", "changes": {"nonsense": 1}})
        self.assertEqual(response.status_code, 400)
        response = self.client.post(f"{API}/settings", json={"action": "steps", "changes": {"steps": "many"}})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.ok(self.client.get(f"{API}/boot"))["state"]["values"], setting_defaults())

    def test_readiness_follows_the_prompt_and_the_modes_media(self):
        self.assertFalse(self.boot["state"]["readiness"]["ready"])
        state = self.ok(self.client.post(f"{API}/prompt", json={"prompt": "A harbor at dawn."}))["state"]
        self.assertTrue(state["readiness"]["ready"])
        state = self.change("mode", mode="First / last frame")
        self.assertFalse(state["readiness"]["ready"])
        state = self.upload("first_image")
        self.assertTrue(state["readiness"]["ready"])
        # The first frame's ratio sets the size, under the start-frame cap.
        self.assertAlmostEqual(state["values"]["width"] / state["values"]["height"], 16 / 9, delta=0.1)
        self.assertEqual(self.client.get(state["media"]["first_image"]["url"]).status_code, 200)

    def test_a_replaced_reference_waits_for_confirmation(self):
        self.change("mode", mode="Reference media")
        state = self.upload("ref_image_2")
        self.assertEqual([(r["tag"], r["engine"], r["unresolved"]) for r in state["references"]],
                         [("<Picture 2>", "<Picture 1>", False)])
        state = self.upload("ref_image_2", "other.png")
        self.assertTrue(state["references"][0]["unresolved"])
        state = self.ok(self.client.post(f"{API}/references/confirm"))["state"]
        self.assertFalse(state["references"][0]["unresolved"])
        state = self.ok(self.client.post(f"{API}/media/clear", json={"slot": "ref_image_2"}))["state"]
        self.assertNotIn("ref_image_2", state["media"])

    def test_removing_the_last_reference_clears_its_tag_and_returns_to_text_generation(self):
        self.change("mode", mode="Reference media")
        self.upload("ref_image_1")
        self.ok(self.client.post(f"{API}/prompt", json={"prompt": "Animate <Picture 1>"}))
        state = self.ok(self.client.post(f"{API}/media/clear", json={"slot": "ref_image_1"}))["state"]
        self.assertEqual(state["prompt"], "Animate")
        self.assertEqual(state["references"], [])
        self.assertEqual(state["values"]["mode"], "Text to video")
        self.assertTrue(state["readiness"]["ready"])
        restored = self.ok(self.client.get(f"{API}/boot"))["state"]
        self.assertEqual(restored["references"], [])
        self.assertEqual(restored["prompt"], "Animate")
        state = self.upload("ref_image_1", "replacement.png")
        self.assertFalse(state["references"][0]["unresolved"])

    def test_confirm_clears_missing_reference_tags_in_existing_drafts(self):
        self.change("mode", mode="Reference media")
        self.upload("ref_image_1")
        self.ok(self.client.post(f"{API}/prompt", json={"prompt": "Animate <Picture 1>"}))
        # A missing file reproduces the stale binding left by older Remove behavior.
        for file in (self.root / "pages").rglob("*-frame.png"):
            file.unlink()
        state = self.ok(self.client.post(f"{API}/references/confirm"))["state"]
        self.assertEqual(state["prompt"], "Animate")
        self.assertEqual(state["references"], [])
        self.assertEqual(state["values"]["mode"], "Text to video")

    def test_first_frame_generation_ignores_unconfirmed_reference_media(self):
        self.change("mode", mode="Reference media")
        self.upload("ref_image_1")
        state = self.upload("ref_image_1", "replacement.png")
        self.assertTrue(state["references"][0]["unresolved"])
        self.upload("first_image", "first.png")
        self.change("mode", mode="First / last frame")
        self.ok(self.client.post(f"{API}/prompt", json={"prompt": "Animate <Picture 1>"}))
        self.ok(self.client.post(f"{API}/generate"))
        result = self.finished("generate")
        self.assertIsNone(result["runs"]["generate"]["error"])
        self.assertTrue(self.generated)
        captured = self.generated[-1]
        self.assertEqual(captured["mode"], "First / last frame")
        self.assertTrue(captured["first_image"])
        self.assertIsNone(captured["ref_image_1"])

    def test_upscale_updates_only_inputs_that_have_not_been_changed(self):
        from h3_ui.h3.page import UPSCALE_OUTPUTS

        for action in ("unchanged", "empty", "remove", "replace"):
            with self.subTest(action=action):
                self.upload("first_image", f"{action}.png")
                self.upload("last_image", f"{action}-last.png")
                output = self.root / f"upscaled-{action}.png"
                Image.new("RGB", (64, 64), "blue").save(output)
                started, release = Event(), Event()

                def updates(*args):
                    started.set()
                    if not release.wait(5):
                        raise RuntimeError("Upscale fixture timed out")
                    yield [str(output), str(output), *[{"__type__": "update"} for _ in UPSCALE_OUTPUTS[2:]], [], "Done"]

                with patch.dict(JOBS.callbacks, {"h3-input": lambda: None}), \
                        patch.object(JOBS, "accept", return_value=SimpleNamespace(id="fixture-upscale")), \
                        patch("h3_ui.h3.page.execute_accepted", updates):
                    try:
                        self.ok(self.client.post(f"{API}/input-upscale", json={"slots": ["First frame", "Last frame"]}))
                        self.assertTrue(started.wait(3))
                        if action == "empty":
                            self.ok(self.client.post(f"{API}/empty"))
                        elif action == "remove":
                            self.ok(self.client.post(f"{API}/media/clear", json={"slot": "first_image"}))
                        elif action == "replace":
                            replacement = self.upload("first_image", "new-first.png")["media"]["first_image"]["path"]
                        release.set()
                        answer = self.finished("input-upscale")
                        self.assertIsNone(answer["runs"]["input-upscale"]["error"])
                        media = answer["state"]["media"]
                        if action in ("empty", "remove"):
                            self.assertNotIn("first_image", media)
                        elif action == "replace":
                            self.assertEqual(media["first_image"]["path"], replacement)
                        else:
                            self.assertIn(output.name, media["first_image"]["path"])
                        if action == "empty":
                            self.assertEqual(media, {})
                        else:
                            self.assertIn(output.name, media["last_image"]["path"])
                    finally:
                        release.set()

    def test_wrong_files_and_foreign_paths_are_refused(self):
        path = self.root / "notes.txt"
        path.write_text("x")
        with path.open("rb") as handle:
            response = self.client.post(f"{API}/media/upload?slot=first_image", files={"file": ("notes.txt", handle)})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.get(f"{API}/file", params={"path": str(path)}).status_code, 404)

    def test_generate_runs_the_job_and_keeps_the_result(self):
        self.assertEqual(self.client.post(f"{API}/generate").status_code, 400)
        self.ok(self.client.post(f"{API}/prompt", json={"prompt": "A harbor at dawn."}))
        self.change("batch_count", batch_count=2)
        self.ok(self.client.post(f"{API}/generate"))
        answer = self.finished("generate")
        self.assertIsNone(answer["runs"]["generate"]["error"])
        self.assertEqual(len(self.generated), 2)
        self.assertEqual(self.generated[0]["prompt"], "A harbor at dawn.")
        self.assertEqual(self.generated[0]["result_format"], "Video")
        videos = answer["state"]["results"]["videos"]
        self.assertEqual(len(videos), 2)
        self.assertEqual(self.client.get(videos[0]["url"]).content, b"video")
        self.assertEqual(len(self.ok(self.client.get(f"{API}/boot"))["state"]["results"]["videos"]), 2)

    def test_reference_tags_reach_the_engine_as_dense_ordinals(self):
        self.change("mode", mode="Reference media")
        self.upload("ref_image_3")
        self.ok(self.client.post(f"{API}/prompt", json={"prompt": "<Picture 3> waves."}))
        self.ok(self.client.post(f"{API}/generate"))
        self.assertIsNone(self.finished("generate")["runs"]["generate"]["error"])
        sent = self.generated[0]
        self.assertEqual(sent["prompt"], "<Picture 1> waves.")
        self.assertTrue(sent["ref_image_1"])
        self.assertIsNone(sent["ref_image_3"])
        self.assertFalse(sent["semantic_bridge"] and False)

    def test_enhance_returns_a_suggestion_in_the_users_tags(self):
        self.change("mode", mode="Reference media")
        self.upload("ref_image_2")
        self.ok(self.client.post(f"{API}/prompt", json={"prompt": "<Picture 2> waves."}))
        self.ok(self.client.post(f"{API}/enhance"))
        run = self.finished("enhance")["runs"]["enhance"]
        self.assertEqual(self.enhanced[0][0], "<Picture 1> waves.")
        self.assertEqual(run["suggestion"], "<Picture 2> waves. In golden light.")
        self.assertEqual(run["original"], "<Picture 2> waves.")

    def test_settings_survive_a_restart_without_api_keys_and_feed_the_followers(self):
        self.change("preset", preset="Balanced")
        self.ok(self.client.post(f"{API}/writer", json={"changes": {"gemini_api_key": "secret",
                                                                     "prompt_writer_backend": "Gemini"}}))
        owner = next(iter(self.remake.settings))
        self.assertEqual(self.remake.settings[owner]["text_encoder"], "INT8 ConvRot")
        self.assertEqual(self.storyboard.settings[owner]["writer"]["gemini_api_key"], "secret")
        saved = next((self.root / "pages").glob("*/settings.json")).read_text(encoding="utf-8")
        self.assertNotIn("secret", saved)
        # A new process: followers read the saved settings before the page is opened again.
        remake = SimpleNamespace(settings={})
        app = FastAPI()
        install_owner_cookie(app, self.store)
        install_h3_page(app, H3PageServices(**{**self.services.__dict__, "followers": (remake,)}))
        self.assertEqual(remake.settings.get(owner, {})["text_encoder"], "INT8 ConvRot")
        self.assertEqual(remake.settings.get("stranger", {}), {})


if __name__ == "__main__":
    unittest.main()
