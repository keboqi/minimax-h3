"""The Soundstage page's JSON API: projects, the mix and its takes, settings and speech runs."""

import inspect
from pathlib import Path
from shutil import which
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

from fastapi import FastAPI
from fastapi.testclient import TestClient

from h3_app.drama3.results import DRAMA3_FAMILY, script_fingerprint
from h3_app.errors import H3Error
from h3_app.generation.results import GenerationUpdate
from h3_app.jobs import JOBS
from h3_app.provenance import write_snapshot
from h3_app.workspace_store import WorkspaceStore, install_owner_cookie
from h3_ui.drama3.controls import DRAMA3_SOUND_SLOTS, DRAMA3_VOICE_SLOTS, UPLOAD_VOICE
from h3_ui.drama3.handlers import Drama3Handlers
from h3_ui.drama3.page import generation_values, install_soundstage_page

API = "/soundstage/api"
TAKE_MIA, TAKE_LEO = "a" * 32, "b" * 32


class Drama3PageTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.addCleanup(setattr, JOBS, "store", JOBS.store)
        self.store = JOBS.store = WorkspaceStore(self.root / "state")
        self.output_dir = self.root / "output"
        self.clips = self.output_dir / "h3" / ".soundstage" / "clips"
        self.clips.mkdir(parents=True)
        (self.clips / "mia.wav").write_bytes(b"RIFF")
        self.takes = self.root / "takes"
        self.takes.mkdir()
        for key in (TAKE_MIA, TAKE_LEO):
            (self.takes / f"{key}.wav").write_bytes(b"RIFF")
        self.calls = []
        self.interrupted = []
        self.result = None
        actions = SimpleNamespace(generate_drama3=self.generate, remix_drama3=self.remix,
                                  regenerate_drama3_line=self.regenerate)
        app = FastAPI()
        install_owner_cookie(app, self.store)
        install_soundstage_page(
            app, actions, store=self.store, output_dir=self.output_dir, takes_dir=lambda: self.takes,
            music=SimpleNamespace(generate_music3=self.generate_music, enhance_music3_prompt=self.write_music),
            interrupt=lambda request, family, job_id: self.interrupted.append((family, job_id)) or "Stopping.")
        self.client = TestClient(app)
        self.client.get(f"{API}/boot")

    # Fake speech actions: each records its inputs and reports one mix.
    def _speak(self, kind, values):
        self.calls.append((kind, values))
        yield GenerationUpdate(None, "Speaking line 1…")
        result = str(self.result or self.mix())
        yield GenerationUpdate(result, "Done.")

    def generate(self, *values, publish):
        yield from self._speak("generate", values)

    def remix(self, *values, publish):
        yield from self._speak("remix", values)

    def regenerate(self, line, *values, publish):
        yield from self._speak("regenerate", (line, *values))

    def write_music(self, prompt, *values, **options):
        self.calls.append(("music brief", prompt, options))
        return "Soft piano and brushed drums", "Unexpected writer lyrics", "Written."

    def generate_music(self, *values):
        self.calls.append(("music", values))
        yield GenerationUpdate(str(self.music_clip), "Done.")

    def mix(self):
        path = self.output_dir / "drama3" / f"night_{uuid.uuid4().hex}.mp3"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"ID3")
        write_snapshot(path, {"family": DRAMA3_FAMILY, "settings": {
            "title": "Night Market", "duration": 6.0, "word_timing": True,
            "script_fingerprint": script_fingerprint("Mia: Leo! You kept the stall open?\nLeo: Someone has to."),
            "chunks": [
                {"name": "Mia", "text": "Leo! [surprised] You kept the stall open?", "take": TAKE_MIA, "reused": True,
                 "ok": True, "warnings": [], "seconds": 2.1, "start": 0.0, "end": 2.1,
                 "lines": [{"text": "Leo! [surprised] You kept the stall open?", "start": 0.0, "end": 2.1,
                            "words": [{"text": "Leo", "start": 0.0, "end": 0.4},
                                      {"text": "You", "start": 0.6, "end": 0.8},
                                      {"text": "kept", "start": 0.8, "end": 1.1},
                                      {"text": "the", "start": 1.1, "end": 1.2},
                                      {"text": "stall", "start": 1.2, "end": 1.6},
                                      {"text": "open", "start": 1.6, "end": 2.1}]}]},
                {"name": "Leo", "text": "Someone has to.", "take": TAKE_LEO, "reused": False, "ok": True,
                 "warnings": ["Ends early"], "seconds": 1.5, "start": 2.5, "end": 4.0},
            ],
            "sounds": [{"name": "rain", "loop": True, "start": 0.0, "end": 6.0}],
        }})
        return path

    def save_project(self, name="Night Market", generated=True, **fields):
        data = {"text": "Mia: Leo! You kept the stall open?\nLeo: Someone has to.", "title": "Night Market",
                "script_mode": "Advanced", "temperature": 2.0, "top_p": 1.0, "top_k": 30,
                "shown_slots": 2, "voice_0_save_name": "Mia", "voice_0_audio": "mia.wav",
                "voice_0_transcript": "Hello there.", "voice_1_save_name": "Leo", **fields}
        self.store.save_soundstage_project(name, data)
        if generated:
            self.result = self.mix()
            self.store.mark_soundstage_generated(name, str(self.result))
        return data

    def ok(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def wait(self, name="Night Market"):
        for _ in range(200):
            run = self.ok(self.client.get(f"{API}/run", params={"name": name}))["run"]
            if not run["active"]:
                return run
            time.sleep(0.02)
        self.fail("The run did not finish.")

    def test_page_and_assets_are_served(self):
        page = self.client.get("/soundstage/")
        self.assertEqual(page.status_code, 200)
        self.assertIn('src="app.js"', page.text)
        self.assertEqual(self.client.get("/soundstage/app.css").status_code, 200)
        self.assertEqual(self.client.get("/soundstage/other.txt").status_code, 404)

    @unittest.skipUnless(which("ffmpeg"), "Background music mixing needs FFmpeg")
    def test_remix_applies_background_music_and_reuses_it_after_toggling(self):
        import numpy as np
        from h3_app.drama3.serving import audio

        data = self.save_project()
        rate = 48000
        audio.save_mp3(np.zeros((6 * rate, 2), dtype=np.float32), rate, self.result)
        self.music_clip = self.root / "music.wav"
        cue = (0.6 * np.sin(np.arange(rate) * (2 * np.pi * 440 / rate))).astype(np.float32)
        self.music_clip.write_bytes(audio.encode_wav(cue, rate))
        params = {"name": "Night Market"}
        original = self.ok(self.client.get(f"{API}/project", params=params))["result"]

        pending = self.ok(self.client.patch(f"{API}/project/music", params=params, json={"music_enabled": True}))
        self.assertTrue(pending["music_pending"])
        self.assertEqual(pending["result"], original)
        self.assertEqual(self.calls, [])
        self.ok(self.client.post(f"{API}/speak", params=params, json={"action": "remix"}))
        self.assertIsNone(self.wait()["error"])
        self.assertIn(data["text"], self.calls[0][1])
        self.assertEqual(self.calls[0][2]["vocals"], "Instrumental")
        self.assertEqual([call[0] for call in self.calls], ["music brief", "music", "remix"])
        self.assertEqual(self.calls[1][1][2], "")
        project = self.ok(self.client.get(f"{API}/project", params=params))
        self.assertTrue(project["background_music"])
        self.assertEqual(project["result"]["music"], {"start": 0, "end": 6.0})
        self.assertEqual(project["result"]["lines"], original["lines"])
        self.assertEqual(project["result"]["takes"], original["takes"])
        self.assertNotEqual(project["result"]["version"], original["version"])

        scored = self.client.get(f"{API}/audio", params=params)
        self.assertEqual(scored.status_code, 200)
        path = self.root / "download.mp3"
        path.write_bytes(scored.content)
        samples = audio.load_audio_file(path, rate)
        self.assertEqual(len(samples), 6 * rate)
        self.assertGreater(np.max(np.abs(samples[5 * rate:5 * rate + rate // 2])), 0.05)

        pending = self.ok(self.client.patch(f"{API}/project/music", params=params, json={"music_enabled": False}))
        self.assertTrue(pending["music_pending"])
        self.assertEqual(self.client.get(f"{API}/audio", params=params).content, scored.content)
        self.ok(self.client.post(f"{API}/speak", params=params, json={"action": "remix"}))
        self.assertIsNone(self.wait()["error"])
        project = self.ok(self.client.get(f"{API}/project", params=params))
        self.assertTrue(project["background_music"])
        self.assertEqual(project["result"], original)
        self.assertEqual(self.client.get(f"{API}/audio", params=params).content, self.result.read_bytes())
        self.ok(self.client.patch(f"{API}/project/music", params=params, json={"music_enabled": True}))
        self.ok(self.client.post(f"{API}/speak", params=params, json={"action": "remix"}))
        self.assertIsNone(self.wait()["error"])
        self.assertEqual([call[0] for call in self.calls].count("music"), 1)
        self.assertEqual(self.client.get(f"{API}/audio", params=params).content, scored.content)

    @unittest.skipUnless(which("ffmpeg"), "Background music mixing needs FFmpeg")
    def test_background_music_can_be_added_before_speaking(self):
        import numpy as np
        from h3_app.drama3.serving import audio

        self.save_project(generated=False)
        self.result = self.mix()
        self.music_clip = self.root / "music.wav"
        recording = audio.encode_wav(np.zeros(48000, dtype=np.float32), 48000)
        self.result.write_bytes(recording)
        self.music_clip.write_bytes(recording)
        params = {"name": "Night Market"}
        draft = self.ok(self.client.get(f"{API}/project", params=params))
        self.assertFalse(draft["music_settings"]["music_enabled"])
        self.ok(self.client.patch(f"{API}/project/music", params=params, json={"music_enabled": True}))
        self.assertEqual(self.calls, [])
        self.ok(self.client.post(f"{API}/speak", params=params, json={"action": "generate"}))
        self.assertIsNone(self.wait()["error"])
        self.assertEqual([call[0] for call in self.calls], ["music brief", "music", "generate"])
        project = self.ok(self.client.get(f"{API}/project", params=params))
        self.assertTrue(project["background_music"])
        self.assertIn("music", project["result"])

    def test_failed_music_encode_preserves_the_previous_mix(self):
        self.save_project(background_music="bed.wav", music_enabled=False)
        params = {"name": "Night Market"}
        previous = self.result
        self.result = self.mix()
        (self.clips / "bed.wav").write_bytes(b"RIFF")
        enabled = self.ok(self.client.patch(f"{API}/project/music", params=params, json={"music_enabled": True}))
        self.assertTrue(enabled["music_pending"])

        with patch("h3_ui.drama3.page_music.mixed_audio", side_effect=H3Error("Music encoding failed.")):
            self.ok(self.client.post(f"{API}/speak", params=params, json={"action": "remix"}))
            self.assertEqual(self.wait()["error"], "Music encoding failed.")

        project = self.ok(self.client.get(f"{API}/project", params=params))
        self.assertTrue(project["music_pending"])
        self.assertNotIn("music", project["result"])
        self.assertEqual(project["result"]["file"], previous.name)
        self.assertEqual(self.client.get(f"{API}/audio", params=params).content, previous.read_bytes())

    @unittest.skipUnless(which("ffmpeg"), "Background music mixing needs FFmpeg")
    def test_music_volume_and_ducking_control_the_saved_audio(self):
        import numpy as np
        from h3_app.drama3.serving import audio

        self.save_project(background_music="bed.wav")
        rate = 48000
        time_axis = np.arange(6 * rate) / rate
        bed = 0.4 * np.sin(2 * np.pi * 440 * time_axis)
        foreground = 0.18 * np.sin(2 * np.pi * 997 * time_axis)
        base = np.zeros((len(time_axis), 2), dtype=np.float32)
        base[rate:2 * rate] = foreground[rate:2 * rate, None]  # centered dialogue
        base[3 * rate:4 * rate] = foreground[3 * rate:4 * rate, None] * [1, -1]  # wide effect
        audio.save_mp3(base, rate, self.result)
        (self.clips / "bed.wav").write_bytes(audio.encode_wav(bed, rate))
        params = {"name": "Night Market"}

        def patch_music(**body):
            self.ok(self.client.patch(f"{API}/project/music", params=params, json=body))
            self.ok(self.client.post(f"{API}/speak", params=params, json={"action": "remix"}))
            self.assertIsNone(self.wait()["error"])
            return self.ok(self.client.get(f"{API}/project", params=params))

        def download():
            response = self.client.get(f"{API}/audio", params=params)
            self.assertEqual(response.status_code, 200)
            path = self.root / "download.mp3"
            path.write_bytes(response.content)
            return audio.load_audio_file(path, rate, stereo=True)[:, 0]

        def level(samples, start):
            window = slice(round(start * rate), round((start + 0.1) * rate))
            carrier = np.exp(-2j * np.pi * 440 * time_axis[window])
            return float(2 * abs(np.mean(samples[window] * carrier)))

        quiet = patch_music(music_volume=20, music_ducking=False)
        quiet_audio = download()
        loud = patch_music(music_volume=40)
        loud_audio = download()
        self.assertAlmostEqual(level(loud_audio, 0.6) / level(quiet_audio, 0.6), 2, delta=0.06)
        self.assertNotEqual(quiet["result"]["version"], loud["result"]["version"])
        self.assertAlmostEqual(level(loud_audio, 1.6), level(loud_audio, 3.6), delta=0.01)

        ducked = patch_music(music_ducking=True)
        ducked_audio = download()
        self.assertNotEqual(loud["result"]["version"], ducked["result"]["version"])
        for start in (1.6, 3.6):
            self.assertAlmostEqual(level(ducked_audio, start) / level(loud_audio, start), 0.5, delta=0.06)
        self.assertGreater(level(ducked_audio, 3.0), level(ducked_audio, 3.6))
        self.assertGreater(level(ducked_audio, 5.5), level(ducked_audio, 4.5))
        self.assertEqual(self.ok(self.client.get(f"{API}/project", params=params))["music_settings"],
                         {"music_enabled": True, "music_volume": 40, "music_ducking": True})
        muted = patch_music(music_volume=0)
        self.assertTrue(muted["background_music"])
        self.assertEqual(self.client.get(f"{API}/audio", params=params).content, self.result.read_bytes())

    def test_generating_music_reenables_background_music_after_disabling(self):
        self.save_project(music_enabled=False)
        self.music_clip = self.root / "fresh.wav"
        self.music_clip.write_bytes(b"RIFF")
        params = {"name": "Night Market"}

        self.ok(self.client.post(f"{API}/project/music", params=params, json={}))
        self.assertIsNone(self.wait()["error"])
        project = self.ok(self.client.get(f"{API}/project", params=params))
        self.assertTrue(project["background_music"])
        self.assertTrue(project["music_settings"]["music_enabled"])
        self.assertTrue(project["music_pending"])

    def test_missing_music_clip_falls_back_to_speech_audio(self):
        self.save_project(background_music="missing.wav", music_enabled=True,
                          applied_music={"background_music": "missing.wav", "music_enabled": True,
                                         "music_volume": 18, "music_ducking": True})
        params = {"name": "Night Market"}
        response = self.client.get(f"{API}/audio", params=params)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, self.result.read_bytes())

    def test_music_unavailable_messages_distinguish_cpu_and_missing_engine(self):
        self.save_project()
        actions = SimpleNamespace(generate_drama3=self.generate, remix_drama3=self.remix,
                                  regenerate_drama3_line=self.regenerate)
        for gpu, expected in ((False, "Open Soundstage in Studio to generate background music."),
                              (True, "Music 3 is unavailable here.")):
            with self.subTest(gpu=gpu):
                app = FastAPI()
                install_owner_cookie(app, self.store)
                install_soundstage_page(app, actions, store=self.store, output_dir=self.output_dir,
                                        takes_dir=lambda: self.takes, gpu=gpu, music=None)
                client = TestClient(app)
                boot = self.ok(client.get(f"{API}/boot"))
                self.assertFalse(boot["background_music"])
                refused = client.post(f"{API}/project/music", params={"name": "Night Market"}, json={})
                self.assertEqual(refused.status_code, 400)
                self.assertEqual(refused.json()["error"], expected)

    def test_run_revision_waits_until_publish_mix_finishes(self):
        self.save_project(generated=False)
        params = {"name": "Night Market"}
        observed = []

        def publish_spy(store, name, result, soundtrack, clips, *, stopped):
            run = self.ok(self.client.get(f"{API}/run", params=params))["run"]
            observed.append(("before", run["revision"]))
            from h3_ui.drama3.page_music import publish_mix
            publish_mix(store, name, result, soundtrack, clips, stopped=stopped)

        with patch("h3_ui.drama3.page.publish_mix", side_effect=publish_spy):
            self.ok(self.client.post(f"{API}/speak", params=params, json={"action": "generate"}))
            done = self.wait()
        self.assertIsNone(done["error"])
        self.assertEqual(observed, [("before", 0)])
        self.assertGreater(done["revision"], 0)

    def test_generation_values_follow_the_speak_steps_input_order(self):
        data = self.save_project(generated=False)
        values = generation_values("Night Market", data, self.clips)
        names = [name for name in inspect.signature(Drama3Handlers.generate_drama3).parameters
                 if name not in ("self", "progress")]
        self.assertEqual(len(values), len(names))
        by_name = dict(zip(names, values))
        self.assertEqual(by_name["text"], data["text"])
        self.assertEqual(by_name["speaker_0_voice"], UPLOAD_VOICE)
        self.assertEqual(by_name["speaker_0_audio"], str(self.clips / "mia.wav"))
        self.assertEqual(by_name["speaker_0_transcript"], "Hello there.")
        self.assertIsNone(by_name["speaker_1_audio"])  # Leo has no clip yet.
        self.assertEqual(by_name["speaker_1_name"], "Leo")
        self.assertEqual(by_name["script_mode"], "Advanced")
        self.assertEqual(by_name["max_frames"], 8192)  # Saved before the field existed: the step's default.
        self.assertEqual(by_name["project_name"], "Night Market")
        self.assertEqual(len([name for name in names if name.startswith("sound_")]), 5 * DRAMA3_SOUND_SLOTS)
        self.assertEqual(len([name for name in names if name.endswith("_name") and name.startswith("speaker_")]),
                         DRAMA3_VOICE_SLOTS)

    def test_a_project_shows_its_cast_lines_takes_and_sounds(self):
        self.save_project()
        self.save_project("Draft only", generated=False)
        boot = self.ok(self.client.get(f"{API}/boot"))
        self.assertEqual({row["name"]: row["generated"] for row in boot["projects"]},
                         {"Night Market": True, "Draft only": False})
        self.assertEqual(boot["presets"]["Expressive"], [2.0, 1.0, 30])
        project = self.ok(self.client.get(f"{API}/project", params={"name": "Night Market"}))
        self.assertEqual([(speaker["name"], speaker["ready"]) for speaker in project["cast"]],
                         [("Mia", True), ("Leo", False)])
        self.assertEqual(project["settings"]["preset"], "Expressive")
        result = project["result"]
        self.assertEqual(result["duration"], 6.0)
        # Cues are left out of what is heard, and aligned words keep their timing.
        self.assertEqual(result["lines"][0]["text"], "Leo! You kept the stall open?")
        self.assertEqual([word["text"] for word in result["lines"][0]["words"]["items"]][:2], ["Leo! ", "You "])
        self.assertIsNone(result["lines"][1]["words"])
        self.assertEqual([(take["speaker"], take["status"]) for take in result["takes"]],
                         [("Mia", "reused"), ("Leo", "flagged")])
        self.assertEqual(result["sounds"], [{"name": "rain", "loop": True, "start": 0.0, "end": 6.0}])
        self.assertIsNone(self.ok(self.client.get(f"{API}/project", params={"name": "Draft only"}))["result"])
        self.assertIn("error", self.client.get(f"{API}/project", params={"name": "Missing"}).json())

    def test_the_script_timeline_uses_the_mixs_real_timings_while_the_script_matches(self):
        data = self.save_project()
        preview = self.ok(self.client.post(f"{API}/preview", params={"name": "Night Market"},
                                           json={"text": data["text"], "script_mode": "Basic"}))
        self.assertTrue(preview["actual"])
        self.assertEqual(preview["duration"], 6.0)
        self.assertEqual([(line["text"], line["line"], line["start"], line["end"]) for line in preview["lines"]],
                         [("Leo! You kept the stall open?", 1, 0.0, 2.1), ("Someone has to.", 2, 2.5, 4.0)])
        # Basic mode stages no sounds, so the mix's bed has no script line to point at.
        self.assertEqual([(sound["name"], sound["line"]) for sound in preview["sounds"]], [("rain", None)])
        # An edited line, a script mode the mix wasn't spoken in, or no project: back to the estimate.
        for params, body in (({"name": "Night Market"}, {"text": data["text"] + "\nMia: One more.", "script_mode": "Basic"}),
                             ({"name": "Night Market"}, {"text": data["text"], "script_mode": "Advanced"}),
                             ({}, {"text": data["text"], "script_mode": "Basic"})):
            preview = self.ok(self.client.post(f"{API}/preview", params=params, json=body))
            self.assertNotIn("actual", preview)

    def test_script_timeline_rejects_changed_staging_and_unverifiable_legacy_mixes(self):
        from h3_ui.drama3.page_script import preview_payload, spoken_payload
        from h3_ui.drama3.result_view import script_preview_settings

        source = 'Mia: First.\n<#1#>\nLeo: Second.\n<sound name="bell" duration="2">Bell rings</sound>'
        settings = {**script_preview_settings(source, "Advanced"), "script_mode": "Advanced",
                    "script_fingerprint": script_fingerprint(source)}
        self.assertTrue(spoken_payload(preview_payload(source, "Advanced"), settings, "Advanced", source)["actual"])
        changes = [source.replace("<#1#>", "<#10#>"),
                   source.replace("Mia: First.", "Leo: Second.").replace("Leo: Second.\n<sound", "Mia: First.\n<sound"),
                   source.replace('duration="2"', 'duration="20"'),
                   source.split('\n<sound')[0],
                   source + '\n<sound name="door">A door opens</sound>',
                   source.replace("Mia: First.", "Mia: [whispering] First."),
                   source.replace("<#1#>\n", "<parallel>\n") + '\n</parallel>']
        for changed in changes:
            with self.subTest(script=changed):
                self.assertIsNone(spoken_payload(preview_payload(changed, "Advanced"), settings, "Advanced", changed))
        settings.pop("script_fingerprint")
        self.assertIsNone(spoken_payload(preview_payload(source, "Advanced"), settings, "Advanced", source))

    def test_only_the_projects_mix_and_its_own_takes_are_served(self):
        self.save_project()
        self.assertEqual(self.client.get(f"{API}/audio", params={"name": "Night Market"}).content, b"ID3")
        self.assertEqual(self.client.get(f"{API}/audio", params={"name": "Missing"}).status_code, 404)
        take = self.client.get(f"{API}/take", params={"name": "Night Market", "key": TAKE_MIA})
        self.assertEqual(take.status_code, 200)
        (self.takes / f"{'c' * 32}.wav").write_bytes(b"RIFF")
        for key in ("c" * 32, "../state", ""):
            self.assertEqual(self.client.get(f"{API}/take", params={"name": "Night Market", "key": key}).status_code,
                             404, key)

    def test_settings_are_saved_with_the_project(self):
        self.save_project()
        project = self.ok(self.client.get(f"{API}/project", params={"name": "Night Market"}))
        self.assertTrue(project["settings"]["word_timing"])
        project = self.ok(self.client.patch(f"{API}/project/settings", params={"name": "Night Market"},
                                            json={"temperature": 0.7, "top_p": 0.7, "top_k": 30, "word_timing": False}))
        self.assertEqual(project["settings"]["preset"], "Strict")
        self.assertFalse(project["settings"]["word_timing"])
        data, _ = self.store.soundstage_project("Night Market")
        self.assertEqual((data["temperature"], data["word_timing"]), (0.7, False))
        self.assertEqual(data["voice_0_save_name"], "Mia")  # Everything else is kept.

    def test_speaking_runs_the_saved_project_as_a_job(self):
        self.save_project()
        run = self.ok(self.client.post(f"{API}/speak", params={"name": "Night Market"},
                                       json={"action": "generate"}))["run"]
        self.assertTrue(run["active"])
        run = self.wait()
        self.assertIsNone(run["error"])
        self.assertEqual(run["status"], "Done.")
        kind, values = self.calls[-1]
        self.assertEqual((kind, values[0], values[-1]), ("generate", self.store.soundstage_project("Night Market")[0]["text"],
                                                         "Night Market"))

    def test_regenerate_needs_a_take_of_this_mix_and_remix_needs_a_mix(self):
        self.save_project()
        refused = self.client.post(f"{API}/speak", params={"name": "Night Market"},
                                   json={"action": "regenerate", "line": "f" * 32})
        self.assertIn("Choose a line", refused.json()["error"])
        self.ok(self.client.post(f"{API}/speak", params={"name": "Night Market"},
                                 json={"action": "regenerate", "line": TAKE_LEO}))
        self.wait()
        self.assertEqual(self.calls[-1][0], "regenerate")
        self.assertEqual(self.calls[-1][1][0], TAKE_LEO)
        self.save_project("Draft only", generated=False)
        refused = self.client.post(f"{API}/speak", params={"name": "Draft only"}, json={"action": "remix"})
        self.assertIn("Speak the project once", refused.json()["error"])
        self.assertIn("Unknown", self.client.post(f"{API}/speak", params={"name": "Night Market"},
                                                  json={"action": "delete"}).json()["error"])

    def test_a_failed_run_reports_why(self):
        self.save_project()

        def failing(*values, publish):
            yield GenerationUpdate(None, "Error: Speaker 1 has no voice.\n\nTimings")

        app = FastAPI()
        install_owner_cookie(app, self.store)
        install_soundstage_page(
            app, SimpleNamespace(generate_drama3=failing, remix_drama3=self.remix, regenerate_drama3_line=self.regenerate),
            store=self.store, output_dir=self.output_dir, takes_dir=lambda: self.takes)
        self.client = TestClient(app)
        self.client.get(f"{API}/boot")
        self.ok(self.client.post(f"{API}/speak", params={"name": "Night Market"}, json={"action": "generate"}))
        run = self.wait()
        self.assertEqual(run["error"], "Speaker 1 has no voice.")
        self.assertEqual(run["status"], "Error: Speaker 1 has no voice.")

    def test_interrupt_stops_only_this_projects_run(self):
        self.save_project()
        self.assertEqual(self.ok(self.client.post(f"{API}/interrupt", params={"name": "Night Market"}))["message"],
                         "This project isn't speaking.")


if __name__ == "__main__":
    unittest.main()
