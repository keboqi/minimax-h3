"""The Soundstage page's Cast & sounds step: speakers, sounds, clips, the libraries and cast jobs.

The tab's own handlers run against a temporary workspace; only the GPU work is replaced.
"""

import dataclasses
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

from fastapi import FastAPI
from fastapi.testclient import TestClient
import gradio as gr

from h3_app.jobs import JOBS
from h3_app.prompt_writers.drama3 import DRAMA3_SCRIPT_ADVANCED
from h3_app.workspace_store import WorkspaceStore, install_owner_cookie
from h3_ui.drama3.bindings import Drama3Actions
from h3_ui.drama3.controls import DRAMA3_SOUND_SLOTS, DRAMA3_VOICE_SLOTS, UPLOAD_VOICE
from h3_ui.drama3.handlers import Drama3Handlers, Drama3Services
from h3_ui.drama3.page import install_soundstage_page

API = "/soundstage/api"
NAME = {"name": "Night Market"}


def write_wav(path: Path, seconds=0.5):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * int(16000 * seconds))
    return path


class Drama3PageCastTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.addCleanup(setattr, JOBS, "store", JOBS.store)
        self.store = JOBS.store = WorkspaceStore(self.root / "state")
        self.runtime = SimpleNamespace(outputs_dir=self.root / "outputs", output_dir=self.root / "outputs" / "h3",
                                       input_dir=self.root / "input", prompt_systems={})
        workspace = patch.dict("os.environ", {"H3_WORKSPACE_DIR": str(self.root / "workspace")})
        workspace.start()
        self.addCleanup(workspace.stop)
        self.made = self.root / "made"
        self.made.mkdir()
        self.calls = []
        self.handlers = Drama3Handlers(Drama3Services(runtime_config=lambda: self.runtime, stage_file=lambda *args: "",
                                                      generation_services=lambda: None, free_gpu=lambda: None))
        actions = dataclasses.replace(Drama3Actions.of(self.handlers, interrupt=lambda *args, **kwargs: ""),
                                      generate_drama3_voice=self.fake_voice, generate_drama3_sound=self.fake_sound,
                                      transcribe_drama3_voice=self.fake_transcribe)
        # Everything missing goes through the tab's prepare handler, which calls these.
        self.handlers.generate_drama3_voices = self.fake_voices
        self.handlers.generate_drama3_sounds = self.fake_sounds
        app = FastAPI()
        install_owner_cookie(app, self.store)
        install_soundstage_page(app, actions, store=self.store, output_dir=self.root / "outputs",
                                takes_dir=lambda: self.root / "takes", runtime_config=lambda: self.runtime)
        self.client = TestClient(app)
        self.client.get(f"{API}/boot")
        self.ok(self.client.post(f"{API}/projects", json=NAME))

    # Fake GPU work, in the shapes the tab's handlers answer with.
    def fake_voice(self, description, preview, language, *settings):
        self.calls.append(("voice", description, preview, language, settings[0]))
        yield gr.skip(), gr.skip(), "Preparing Qwen VoiceDesign."
        yield str(write_wav(self.made / "voice.wav")), gr.update(value=preview, interactive=True), "Voice generated."

    def fake_sound(self, description, duration):
        self.calls.append(("sound", description, duration))
        yield str(write_wav(self.made / "sound.wav")), "Sound generated with Stable Audio 3 Medium; ready to save."

    def fake_transcribe(self, audio, model):
        self.calls.append(("transcribe", Path(audio).parent.name, model))
        return "Fresh plums, two for one.", "Transcribed with Qwen3-ASR."

    def fake_voices(self, *values):
        self.calls.append(("voices", values))
        updates = [gr.skip()] * (3 * DRAMA3_VOICE_SLOTS)
        updates[0:3] = [str(write_wav(self.made / "mia.wav")), gr.update(value="Hello there."), "Voice generated."]
        yield (*updates, "Generated 1 of 1 speaker voices.")

    def fake_sounds(self, *values):
        updates = [gr.skip()] * (2 * DRAMA3_SOUND_SLOTS)
        updates[0:2] = [str(write_wav(self.made / "rain.wav")), "Sound generated."]
        yield (*updates, "Generated 1 of 1 sound clips.")

    def ok(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertNotIn("error", data)
        return data

    def wait(self):
        for _ in range(300):
            run = self.ok(self.client.get(f"{API}/project", params=NAME))["cast_run"]
            if run and not run["active"]:
                return run
            time.sleep(0.02)
        self.fail("The cast run did not finish.")

    def patch_slot(self, kind, slot, **body):
        return self.ok(self.client.patch(f"{API}/project/slot", params={**NAME, "kind": kind, "slot": slot}, json=body))

    def test_boot_offers_the_generators(self):
        boot = self.ok(self.client.get(f"{API}/boot"))
        self.assertIn("Qwen3-TTS VoiceDesign", boot["voice_engines"])
        self.assertIn("Auto", boot["languages"])
        self.assertEqual(boot["slots"], {"voices": DRAMA3_VOICE_SLOTS, "sounds": DRAMA3_SOUND_SLOTS})

    def test_speakers_are_edited_added_and_removed(self):
        project = self.patch_slot("voice", 0, save_name="Mia", description="A bright vendor.", preview_text="Plums!")
        mia = project["speakers"][0]
        self.assertEqual((mia["name"], mia["description"], mia["preview_text"], mia["ready"]), ("Mia", "A bright vendor.", "Plums!", False))
        self.assertEqual([speaker["name"] for speaker in project["cast"]], ["Mia"])
        project = self.ok(self.client.post(f"{API}/project/slots", params={**NAME, "kind": "voice"}, json={"add": True}))
        self.assertEqual([speaker["name"] for speaker in project["speakers"]], ["Mia", "Speaker 1"])
        self.assertEqual([speaker["name"] for speaker in project["cast"]], ["Mia"])  # An empty slot is no one yet.
        project = self.ok(self.client.post(f"{API}/project/slots", params={**NAME, "kind": "voice"}, json={"add": False}))
        self.assertEqual(len(project["speakers"]), 1)
        refused = self.client.patch(f"{API}/project/slot", params={**NAME, "kind": "voice", "slot": 99}, json={})
        self.assertIn("No such slot", refused.json()["error"])

    def test_an_uploaded_clip_is_kept_served_and_removed(self):
        self.patch_slot("voice", 0, save_name="Mia", transcript="old words")
        clip = write_wav(self.root / "mine.wav")
        with clip.open("rb") as handle:
            project = self.ok(self.client.post(f"{API}/project/slot/upload", params={**NAME, "kind": "voice", "slot": 0},
                                               files={"file": ("mine.wav", handle, "audio/wav")}))
        mia = project["speakers"][0]
        self.assertTrue(mia["ready"])
        self.assertEqual(mia["source"], "clip")
        self.assertEqual(mia["transcript"], "")  # A new clip needs its own transcript.
        self.assertEqual(self.client.get(mia["clip"]).content, clip.read_bytes())
        with (self.root / "notes.txt").open("wb") as handle:
            handle.write(b"text")
        with (self.root / "notes.txt").open("rb") as handle:
            refused = self.client.post(f"{API}/project/slot/upload", params={**NAME, "kind": "voice", "slot": 0},
                                       files={"file": ("notes.txt", handle, "text/plain")})
        self.assertIn("isn't an audio file", refused.json()["error"])
        project = self.ok(self.client.delete(f"{API}/project/slot/clip", params={**NAME, "kind": "voice", "slot": 0}))
        self.assertFalse(project["speakers"][0]["ready"])
        self.assertEqual(self.client.get(mia["clip"]).status_code, 404)

    def test_a_voice_saved_to_a_group_is_reused_in_another_slot(self):
        self.patch_slot("voice", 0, save_name="Mia", description="A bright vendor.")
        with write_wav(self.root / "mine.wav").open("rb") as handle:
            self.ok(self.client.post(f"{API}/project/slot/upload", params={**NAME, "kind": "voice", "slot": 0},
                                     files={"file": ("mine.wav", handle, "audio/wav")}))
        refused = self.client.post(f"{API}/project/slot/save", params={**NAME, "kind": "voice", "slot": 0}, json={})
        self.assertIn("transcript", refused.json()["error"])
        self.patch_slot("voice", 0, transcript="Fresh plums.")
        group = self.ok(self.client.post(f"{API}/library/groups", json={"name": "Lantern Lane"}))["group"]
        self.ok(self.client.patch(f"{API}/project/cast-settings", params=NAME, json={"speaker_group": group}))
        saved = self.ok(self.client.post(f"{API}/project/slot/save", params={**NAME, "kind": "voice", "slot": 0}, json={}))
        self.assertIn("Lantern Lane", saved["message"])
        self.assertEqual(saved["speakers"][0]["source"], "saved")
        library = self.ok(self.client.get(f"{API}/library", params={"group": group}))
        self.assertIn(group, library["groups"])
        self.assertEqual([speaker["name"] for speaker in library["speakers"]], ["Mia"])
        self.ok(self.client.post(f"{API}/project/slots", params={**NAME, "kind": "voice"}, json={"add": True}))
        project = self.ok(self.client.post(f"{API}/project/slot/use-saved", params={**NAME, "kind": "voice", "slot": 1},
                                           json={"key": library["speakers"][0]["key"]}))
        reused = project["speakers"][1]
        self.assertEqual((reused["name"], reused["source"], reused["ready"], reused["transcript"]),
                         ("Mia", "saved", True, "Fresh plums."))
        self.assertEqual(self.client.get(reused["clip"]).status_code, 200)
        project = self.patch_slot("voice", 1, source="upload")
        self.assertEqual(project["speakers"][1]["source"], "describe")

    def test_a_sound_is_described_saved_and_reused(self):
        self.ok(self.client.patch(f"{API}/project/script", params=NAME, json={
            "text": 'Mia: Plums!\n<sound name="rain" duration="8" loop="true">Steady rain on awnings</sound>',
            "script_mode": DRAMA3_SCRIPT_ADVANCED}))
        project = self.ok(self.client.get(f"{API}/project", params=NAME))
        rain = project["sounds"][0]
        self.assertEqual((rain["name"], rain["loop"], rain["ready"]), ("rain", True, False))
        with write_wav(self.root / "rain.wav").open("rb") as handle:
            self.ok(self.client.post(f"{API}/project/slot/upload", params={**NAME, "kind": "sound", "slot": 0},
                                     files={"file": ("rain.wav", handle, "audio/wav")}))
        saved = self.ok(self.client.post(f"{API}/project/slot/save", params={**NAME, "kind": "sound", "slot": 0}, json={}))
        self.assertEqual(saved["sounds"][0]["source"], "saved")
        library = self.ok(self.client.get(f"{API}/library"))
        self.assertEqual([sound["name"] for sound in library["sounds"]], ["rain"])
        project = self.ok(self.client.post(f"{API}/project/slots", params={**NAME, "kind": "sound"}, json={"add": True}))
        project = self.patch_slot("sound", 1, save_name="thunder", description="Distant thunder", duration=4)
        self.assertEqual([(sound["name"], sound["duration"], sound["loop"]) for sound in project["sounds"]],
                         [("rain", 8, True), ("thunder", 4, None)])

    def test_making_one_voice_keeps_its_clip_and_transcript_in_the_slot(self):
        refused = self.client.post(f"{API}/project/make", params=NAME, json={"target": "voice", "slot": 0})
        self.assertIn("voice description", refused.json()["error"])
        self.patch_slot("voice", 0, save_name="Mia", description="A bright vendor.", preview_text="Fresh plums!")
        self.ok(self.client.post(f"{API}/project/make", params=NAME, json={"target": "voice", "slot": 0}))
        run = self.wait()
        self.assertIsNone(run["error"])
        self.assertEqual(run["slots"]["voice:0"], "Voice generated.")
        self.assertEqual(self.calls[-1], ("voice", "A bright vendor.", "Fresh plums!", "Auto", "Qwen3-TTS VoiceDesign"))
        mia = self.ok(self.client.get(f"{API}/project", params=NAME))["speakers"][0]
        self.assertEqual((mia["ready"], mia["source"], mia["transcript"]), (True, "clip", "Fresh plums!"))
        saved, _ = self.store.soundstage_project("Night Market")
        self.assertEqual(saved["voice_0_preset"], UPLOAD_VOICE)
        self.assertTrue((self.root / "outputs" / "h3" / ".soundstage" / "clips" / saved["voice_0_audio"]).is_file())

    def test_transcribing_fills_the_clips_transcript(self):
        with write_wav(self.root / "mine.wav").open("rb") as handle:
            self.ok(self.client.post(f"{API}/project/slot/upload", params={**NAME, "kind": "voice", "slot": 0},
                                     files={"file": ("mine.wav", handle, "audio/wav")}))
        self.ok(self.client.post(f"{API}/project/make", params=NAME, json={"target": "transcribe", "slot": 0}))
        self.wait()
        self.assertEqual(self.calls[-1][0], "transcribe")
        self.assertEqual(self.ok(self.client.get(f"{API}/project", params=NAME))["speakers"][0]["transcript"],
                         "Fresh plums, two for one.")

    def test_making_one_sound_and_everything_missing(self):
        self.ok(self.client.patch(f"{API}/project/script", params=NAME, json={
            "text": 'Mia: Plums!\n<sound name="rain" duration="8" loop="true">Steady rain on awnings</sound>',
            "script_mode": DRAMA3_SCRIPT_ADVANCED}))
        self.ok(self.client.post(f"{API}/project/make", params=NAME, json={"target": "sound", "slot": 0}))
        self.wait()
        self.assertEqual(self.calls[-1], ("sound", "Steady rain on awnings", 8))
        self.assertTrue(self.ok(self.client.get(f"{API}/project", params=NAME))["sounds"][0]["ready"])
        self.ok(self.client.delete(f"{API}/project/slot/clip", params={**NAME, "kind": "sound", "slot": 0}))
        self.patch_slot("voice", 0, save_name="Mia", description="A bright vendor.", preview_text="Hello there.")
        self.ok(self.client.post(f"{API}/project/make", params=NAME, json={"target": "prepare"}))
        run = self.wait()
        self.assertIsNone(run["error"])
        self.assertIn("Preparation finished", run["message"])
        project = self.ok(self.client.get(f"{API}/project", params=NAME))
        self.assertEqual((project["speakers"][0]["ready"], project["speakers"][0]["transcript"]), (True, "Hello there."))
        self.assertTrue(project["sounds"][0]["ready"])
        voices = self.calls[-1][1]
        self.assertEqual(voices[0], "Qwen3-TTS VoiceDesign")  # The project's voice generator.

    def test_one_cast_run_at_a_time_and_no_gpu_no_run(self):
        self.assertIn("Unknown", self.client.post(f"{API}/project/make", params=NAME, json={"target": "all"}).json()["error"])
        app = FastAPI()
        install_owner_cookie(app, self.store)
        install_soundstage_page(app, Drama3Actions.of(self.handlers, interrupt=lambda *args, **kwargs: ""),
                                store=self.store, output_dir=self.root / "outputs", takes_dir=lambda: self.root,
                                runtime_config=lambda: self.runtime, gpu=False)
        client = TestClient(app)
        client.get(f"{API}/boot")
        self.assertIn("Studio", client.post(f"{API}/project/make", params=NAME, json={"target": "prepare"}).json()["error"])


if __name__ == "__main__":
    unittest.main()
