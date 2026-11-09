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


class CastFixture(unittest.TestCase):
    """A project on a page whose GPU work is faked, and helpers to drive it."""

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
                                      transcribe_drama3_voice=self.fake_transcribe, generate_drama3=self.fake_speak,
                                      repair_drama3_words=self.fake_words,
                                      generate_drama3_voice_takes=self.fake_takes)
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

    def fake_speak(self, *values):
        """Speech marks its project generated, as the real handler does with the project name it is given."""
        from h3_app.drama3.results import DRAMA3_FAMILY
        from h3_app.generation.results import GenerationUpdate
        from h3_app.provenance import write_snapshot

        self.calls.append(("speak", values))
        mix = write_wav(self.made / f"mix_{len(self.calls)}.wav")
        write_snapshot(mix, {"family": DRAMA3_FAMILY, "settings": {"duration": 0.5, "chunks": [], "sounds": []}})
        self.store.mark_soundstage_generated(values[-1], str(mix))
        yield GenerationUpdate(None, "Speaking…")
        yield GenerationUpdate(str(mix), "Done.")

    def fake_words(self, output):
        self.calls.append(("words", output))
        if getattr(self, "words_fail", False):
            yield gr.skip(), "Word alignment unavailable: the aligner is not installed"
            return
        yield gr.skip(), "Aligning 1 take…"
        yield "<div></div>", "Aligned 1 of 1 takes."

    def fake_takes(self, key, count, description, transcript, *settings):
        self.calls.append(("takes", key, count, settings[0]))
        clips = [None] * 4
        yield (*[gr.update(value=None) for _ in clips], gr.skip(), None, "Generating…")
        for index in range(count):
            clips[index] = str(write_wav(self.made / f"try_{index}.wav", seconds=0.2 + index / 10))
        brief = {"key": key, "description": description, "transcript": transcript}
        yield (*[gr.update(value=clip, visible=index < count) for index, clip in enumerate(clips)], gr.skip(), brief,
               f"Generated {count} of {count} new voices.")

    def voiced_mia(self, group="Global"):
        """Mia with an uploaded clip and transcript, saved to ``group``; returns her saved key."""
        self.patch_slot("voice", 0, save_name="Mia", description="A bright vendor.", transcript="Fresh plums.")
        with write_wav(self.root / "mine.wav").open("rb") as handle:
            self.ok(self.client.post(f"{API}/project/slot/upload", params={**NAME, "kind": "voice", "slot": 0},
                                     files={"file": ("mine.wav", handle, "audio/wav")}))
        self.patch_slot("voice", 0, transcript="Fresh plums.")
        if group != "Global":
            self.ok(self.client.post(f"{API}/library/groups", json={"name": group}))
            self.ok(self.client.patch(f"{API}/project/cast-settings", params=NAME, json={"speaker_group": group}))
        self.ok(self.client.post(f"{API}/project/slot/save", params={**NAME, "kind": "voice", "slot": 0}, json={}))
        return self.store.soundstage_project("Night Market")[0]["voice_0_preset"]

    def wait_speech(self):
        for _ in range(300):
            run = self.ok(self.client.get(f"{API}/run", params=NAME))["run"]
            if run and not run["active"]:
                return run
            time.sleep(0.02)
        self.fail("The run did not finish.")

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


class Drama3PageCastTests(CastFixture):
    def test_generation_does_not_replace_an_edited_slot(self):
        self.patch_slot("voice", 0, description="Original voice", preview_text="Hello there.")
        self.patch_slot("voice", 1, description="Other voice", preview_text="Good morning.")
        # Prepare calls this generator; edit slot zero before delivering its result.
        def voices(*values):
            self.patch_slot("voice", 0, description="New direction")
            yield from self.fake_voices(*values)
        self.handlers.generate_drama3_voices = voices
        self.ok(self.client.post(f"{API}/project/make", params=NAME, json={"target": "prepare"}))
        run = self.wait()
        self.assertIn("slot changed", run["error"])
        data, _ = self.store.soundstage_project("Night Market")
        self.assertEqual(data["voice_0_description"], "New direction")
        self.assertFalse(data.get("voice_0_audio"))
        self.assertIn("voice:0", data["_cast_proposals"])
        params = {**NAME, "kind": "voice", "slot": 0}
        reviewed = self.ok(self.client.get(f"{API}/project/slot/proposal", params=params))
        self.assertEqual(self.client.get(f"{API}/project/slot/proposal/audio", params={**params, "id": reviewed["id"]}).status_code, 200)
        # A newer result arriving while the review dialog is open invalidates both
        # its audio URL and its Apply action without consuming the replacement.
        data["_cast_proposals"]["voice:0"]["id"] = "replacement"
        data["_cast_proposals"]["voice:0"]["patch"]["voice_0_transcript"] = "Replacement transcript."
        self.store.save_soundstage_project("Night Market", data)
        self.assertEqual(self.client.get(f"{API}/project/slot/proposal/audio", params={**params, "id": reviewed["id"]}).status_code, 404)
        for body in ({}, {"id": reviewed["id"]}):
            self.assertEqual(self.client.post(f"{API}/project/slot/proposal", params=params, json=body).status_code, 400)
        current = self.ok(self.client.get(f"{API}/project/slot/proposal", params=params))
        self.ok(self.client.post(f"{API}/project/slot/proposal", params=params, json={"id": current["id"]}))
        data, _ = self.store.soundstage_project("Night Market")
        self.assertTrue(data["voice_0_audio"])
        self.assertEqual(data["voice_0_description"], "New direction")
        self.assertEqual(data["voice_0_transcript"], "Replacement transcript.")

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


class Drama3PageAutoTests(CastFixture):
    """Auto and the speech settings, on the same fake GPU work."""

    WRITER = ("<|speaker:0|>Fresh plums, two for one.", "Rewrote the script.",
              [{"speaker_id": 0, "name": "Mia", "description": "A bright vendor.",
                "preview_text": "Fresh plums, two for one, only while the rain holds off tonight."}], "Night Market")

    def test_auto_rewrites_makes_the_cast_and_speaks(self):
        self.ok(self.client.patch(f"{API}/project/script", params=NAME, json={"text": "Mia sells plums in the rain."}))
        with patch("h3_app.prompt_writers.drama3.enhance_drama3_text", return_value=self.WRITER) as writer:
            run = self.ok(self.client.post(f"{API}/auto", params=NAME, json={"writer": {"gemini_api_key": "k"}}))["run"]
            self.assertTrue(run["auto"])
            run = self.wait_speech()
        self.assertIsNone(run["error"])
        self.assertEqual((run["stage"], run["status"]), ("speak", "Done."))
        writer.assert_called_once()
        self.assertEqual([call[0] for call in self.calls], ["voices", "speak"])
        project = self.ok(self.client.get(f"{API}/project", params=NAME))
        self.assertEqual(project["script"]["text"], "Mia: Fresh plums, two for one.")
        self.assertTrue(project["speakers"][0]["ready"])
        self.assertIsNotNone(project["result"])
        # Speech went out with the voice Auto just made.
        speech = self.calls[-1][1]
        self.assertTrue(str(speech[2]).endswith(".wav"))

    def test_auto_keeps_a_prepared_script_and_can_save_the_cast(self):
        self.ok(self.client.patch(f"{API}/project/script", params=NAME, json={"text": "Mia sells plums."}))
        with patch("h3_app.prompt_writers.drama3.enhance_drama3_text", return_value=self.WRITER):
            self.ok(self.client.post(f"{API}/project/rewrite", params=NAME, json={}))
        self.ok(self.client.patch(f"{API}/project/settings", params=NAME, json={"auto_save_speakers": True}))
        with patch("h3_app.prompt_writers.drama3.enhance_drama3_text") as writer:
            self.ok(self.client.post(f"{API}/auto", params=NAME, json={}))
            run = self.wait_speech()
        writer.assert_not_called()
        self.assertIsNone(run["error"])
        self.assertIn('to "Global"', run["message"])
        self.assertEqual(self.ok(self.client.get(f"{API}/project", params=NAME))["speakers"][0]["source"], "saved")

    def test_auto_stops_with_the_writers_reason(self):
        self.ok(self.client.patch(f"{API}/project/script", params=NAME, json={"text": "Mia sells plums."}))
        with patch("h3_app.prompt_writers.drama3.enhance_drama3_text",
                   return_value=("", "Prompt enhancement failed: no key", [], "")):
            self.ok(self.client.post(f"{API}/auto", params=NAME, json={}))
            run = self.wait_speech()
        self.assertEqual(run["error"], "The writer stopped: no key")
        self.assertEqual(self.calls, [])
        empty = self.client.post(f"{API}/projects", json={"name": "Empty"})
        self.ok(empty)
        self.assertIn("Script first", self.client.post(f"{API}/auto", params={"name": "Empty"}, json={}).json()["error"])

    def test_interrupting_auto_during_the_rewrite_stops_it_before_the_gpu(self):
        self.ok(self.client.patch(f"{API}/project/script", params=NAME, json={"text": "Mia sells plums."}))
        app = FastAPI()
        install_owner_cookie(app, self.store)
        actions = dataclasses.replace(Drama3Actions.of(self.handlers, interrupt=lambda *args, **kwargs: ""),
                                      generate_drama3=self.fake_speak)
        install_soundstage_page(app, actions, store=self.store, output_dir=self.root / "outputs",
                                takes_dir=lambda: self.root / "takes", runtime_config=lambda: self.runtime,
                                interrupt=lambda request, family, job_id: "Stopping.")
        self.client = TestClient(app)
        self.client.get(f"{API}/boot")

        def slow_writer(*args, **kwargs):
            time.sleep(0.3)
            return self.WRITER

        with patch("h3_app.prompt_writers.drama3.enhance_drama3_text", side_effect=slow_writer):
            self.ok(self.client.post(f"{API}/auto", params=NAME, json={}))
            stopped = self.ok(self.client.post(f"{API}/interrupt", params=NAME))
            self.assertEqual(stopped["message"], "Auto stops when the writer answers.")
            run = self.wait_speech()
        self.assertEqual(run["status"], "Auto interrupted.")
        self.assertEqual(self.calls, [])  # No voices, sounds or speech.

    def test_speech_settings_are_kept_inside_their_ranges(self):
        boot = self.ok(self.client.get(f"{API}/boot"))
        self.assertEqual(boot["ranges"]["parallel"], [1, 32, 1])
        project = self.ok(self.client.patch(f"{API}/project/settings", params=NAME, json={
            "parallel": 99, "retries": 2, "max_frames": 4000, "temperature": 1.3, "top_k": -5}))
        settings = project["settings"]
        self.assertEqual((settings["parallel"], settings["retries"], settings["max_frames"], settings["top_k"]),
                         (32, 2, 4000, 0))
        self.assertIsNone(settings["preset"])  # Custom.
        saved, _ = self.store.soundstage_project("Night Market")
        self.assertEqual((saved["parallel"], saved["temperature"]), (32, 1.3))


class Drama3PageLibraryTests(CastFixture):
    """Word timing for a mix, managing saved speakers and sounds, and trying new voices."""

    def test_word_timing_is_repaired_from_the_saved_takes(self):
        refused = self.client.post(f"{API}/words", params=NAME)
        self.assertIn("Speak the project first", refused.json()["error"])
        mix = write_wav(self.made / "mix.wav")
        self.store.mark_soundstage_generated("Night Market", str(mix))
        run = self.ok(self.client.post(f"{API}/words", params=NAME))["run"]
        self.assertEqual(run["task"], "words")
        run = self.wait_speech()
        self.assertEqual((run["status"], run["error"]), ("Aligned 1 of 1 takes.", None))
        self.assertEqual(self.calls[-1], ("words", str(mix)))
        self.words_fail = True
        self.ok(self.client.post(f"{API}/words", params=NAME))
        self.assertEqual(self.wait_speech()["error"], "the aligner is not installed")

    def test_a_renamed_saved_speaker_stays_in_use_and_a_deleted_one_is_let_go(self):
        key = self.voiced_mia("Lantern Lane")
        listed = self.ok(self.client.get(f"{API}/library/manage", params={"group": "Lantern Lane"}))
        self.assertEqual([(speaker["name"], speaker["transcript"]) for speaker in listed["speakers"]], [("Mia", "Fresh plums.")])
        self.assertEqual(self.client.get(f"{API}/library/clip", params={"kind": "voice", "key": key}).status_code, 200)
        renamed = self.ok(self.client.post(f"{API}/library/speaker/update", json={
            "key": key, "name": "Mia Chen", "description": "A bright plum seller.", "transcript": "Fresh plums.",
            "project": "Night Market"}))
        self.assertIn("Mia Chen", renamed["message"])
        self.assertEqual(renamed["project"]["speakers"][0]["saved_name"], "Mia Chen")
        self.assertEqual(self.store.soundstage_project("Night Market")[0]["voice_0_preset"], renamed["key"])
        copied = self.ok(self.client.post(f"{API}/library/speaker/copy", json={"key": renamed["key"], "group": "Global"}))
        self.assertIn('to "Global"', copied["message"])
        self.ok(self.client.post(f"{API}/library/speaker/delete", json={"key": renamed["key"], "project": "Night Market"}))
        self.assertEqual(self.store.soundstage_project("Night Market")[0]["voice_0_preset"], UPLOAD_VOICE)
        self.assertEqual(self.ok(self.client.get(f"{API}/library/manage", params={"group": "Lantern Lane"}))["speakers"], [])

    def test_deleting_a_group_lets_go_of_its_voices(self):
        self.voiced_mia("Lantern Lane")
        refused = self.client.post(f"{API}/library/group/delete", json={"group": "Global"})
        self.assertIn("cannot be deleted", refused.json()["error"])
        data = self.ok(self.client.post(f"{API}/library/group/delete", json={"group": "Lantern Lane", "project": "Night Market"}))
        self.assertIn("1 saved speakers", data["message"])
        saved, _ = self.store.soundstage_project("Night Market")
        self.assertEqual((saved["voice_0_preset"], saved["speaker_group"]), (UPLOAD_VOICE, "Global"))
        self.assertNotIn("Lantern Lane", self.ok(self.client.get(f"{API}/library/manage"))["groups"])

    def test_a_saved_sound_is_deleted_and_let_go(self):
        with write_wav(self.root / "rain.wav").open("rb") as handle:
            self.ok(self.client.post(f"{API}/project/slot/upload", params={**NAME, "kind": "sound", "slot": 0},
                                     files={"file": ("rain.wav", handle, "audio/wav")}))
        self.patch_slot("sound", 0, save_name="rain", description="Steady rain")
        self.ok(self.client.post(f"{API}/project/slot/save", params={**NAME, "kind": "sound", "slot": 0}, json={}))
        data = self.ok(self.client.post(f"{API}/library/sound/delete", json={"name": "rain", "project": "Night Market"}))
        self.assertEqual(self.ok(self.client.get(f"{API}/library/manage"))["sounds"], [])
        # The slot lets go of the deleted saved sound and keeps its own clip.
        self.assertEqual(data["project"]["sounds"][0]["source"], "clip")

    def test_trying_new_voices_and_keeping_one(self):
        key = self.voiced_mia()
        before = self.client.get(f"{API}/library/clip", params={"kind": "voice", "key": key}).content
        refused = self.client.post(f"{API}/library/speaker/takes", json={"key": key, "count": 2, "description": "", "transcript": "x"})
        self.assertIn("voice description", refused.json()["error"])
        run = self.ok(self.client.post(f"{API}/library/speaker/takes", json={
            "key": key, "count": 2, "description": "Warmer, slower.", "transcript": "Fresh plums.", "project": "Night Market"}))["takes_run"]
        for _ in range(300):
            run = self.ok(self.client.get(f"{API}/library/takes", params={"run": run["id"]}))["takes_run"]
            if not run["active"]:
                break
            time.sleep(0.02)
        self.assertIsNone(run["error"])
        self.assertEqual(self.calls[-1], ("takes", key, 2, "Qwen3-TTS VoiceDesign"))
        self.assertEqual(len([take for take in run["takes"] if take]), 2)
        self.assertEqual(self.client.get(run["takes"][1]).status_code, 200)
        kept = self.ok(self.client.post(f"{API}/library/speaker/use-take", json={"run": run["id"], "index": 1, "project": "Night Market"}))
        self.assertIn("now uses voice 2", kept["message"])
        after = self.client.get(f"{API}/library/clip", params={"kind": "voice", "key": key}).content
        self.assertNotEqual(before, after)
        listed = self.ok(self.client.get(f"{API}/library/manage"))["speakers"][0]
        self.assertEqual(listed["description"], "Warmer, slower.")

    def test_a_typed_name_takes_the_saved_voice_of_that_name(self):
        self.voiced_mia()
        project = self.ok(self.client.post(f"{API}/project/slots", params={**NAME, "kind": "voice"}, json={"add": True}))
        project = self.patch_slot("voice", 1, save_name="mia")
        second = project["speakers"][1]
        self.assertEqual((second["source"], second["name"], second["ready"]), ("saved", "Mia", True))
        # A speaker with a clip of its own keeps it.
        self.ok(self.client.post(f"{API}/project/slots", params={**NAME, "kind": "voice"}, json={"add": True}))
        with write_wav(self.root / "own.wav").open("rb") as handle:
            self.ok(self.client.post(f"{API}/project/slot/upload", params={**NAME, "kind": "voice", "slot": 2},
                                     files={"file": ("own.wav", handle, "audio/wav")}))
        self.assertEqual(self.patch_slot("voice", 2, save_name="Mia")["speakers"][2]["source"], "clip")


if __name__ == "__main__":
    unittest.main()
