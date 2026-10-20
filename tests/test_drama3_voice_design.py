"""Cast/script pairing, designed voice references and the resident audio tools worker."""

import json
import os
from pathlib import Path
import tempfile
from contextlib import nullcontext
from pathlib import PurePosixPath, PureWindowsPath
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from h3_app import audio_tools, drama3_speakers, prompt_service, voice_design
from h3_app.config import RuntimeConfig
from h3_app.drama3_cast import parse_script_cast
from h3_app.errors import H3Error
from h3_app.jobs import JobCancelled
from h3_requirements import FLASH_ATTN_WHEELS, audio_tools_install_commands


def cast_entry(speaker, name="Alice"):
    return {"speaker_id": speaker, "name": name,
            "description": "A warm midrange voice with crisp articulation and a natural pace.",
            "preview_text": "The morning is quiet, and I am ready to begin our journey."}


def writer_reply(*ids):
    return json.dumps({"script": "\n".join(f"<|speaker:{i}|> Hello there." for i in ids),
                       "speakers": [cast_entry(i, f"Character {i}") for i in ids]})


class CastValidationTests(unittest.TestCase):
    def test_sparse_ids_stay_paired(self):
        script, cast = parse_script_cast(writer_reply(5, 2))
        self.assertTrue(script.startswith("<|speaker:5|>"))
        self.assertEqual([entry["speaker_id"] for entry in cast], [2, 5])

    def test_rejects_inconsistent_cast_and_unperformable_output(self):
        valid = json.loads(writer_reply(5))
        for mutation in (
            {"speakers": [cast_entry(0)]},
            {"speakers": [cast_entry(5), cast_entry(5)]},
            {"speakers": [cast_entry(True)]},
            {"speakers": [cast_entry("5")]},
            {"speakers": [cast_entry(16)]},
            {"speakers": [{**cast_entry(5), "description": ""}]},
            {"speakers": [{**cast_entry(5), "preview_text": "[laughs] Hello"}]},
            {"script": "Heading\n<|speaker:5|> Hello"},
            {"script": "<|speaker:5|> Hello <|speaker:0|> Bye"},
            {"script": "<|speaker:5|> <|audio_separator|>"},
        ):
            with self.subTest(mutation=mutation), self.assertRaises(H3Error):
                parse_script_cast(json.dumps({**valid, **mutation}))

    @patch("h3_app.prompt_service._enhance_prompt_from_media")
    def test_structured_writer_is_atomic_and_preserves_mode_and_existing_ids(self, enhance):
        runtime = SimpleNamespace(prompt_systems={"Drama 3": Path("prompt_drama3.txt")})
        def rewrite(reply, text="Original", **kwargs):
            enhance.return_value = reply, "Enhanced prompt with writer"
            return prompt_service.enhance_drama3_text(text, "model", "", runtime=runtime,
                                                     return_cast=True, **kwargs)
        script, status, cast = rewrite(writer_reply(5))
        self.assertEqual(cast[0]["speaker_id"], 5)
        self.assertIn("script", status)
        for reply, text, settings in (
            (writer_reply(0), "<|speaker:5|> Original", {}),
            (writer_reply(0), "Original", {"speaker_mode": "Adapt to voices", "voices": [(5, "Alice")]}),
            (writer_reply(0, 5), "Original", {"speaker_mode": "Single"}),
            ("<|speaker:0|> Legacy reply", "Original", {}),
        ):
            with self.subTest(reply=reply, settings=settings):
                script, status, cast = rewrite(reply, text, **settings)
                self.assertEqual(script, text)
                self.assertTrue(status.startswith("Prompt enhancement failed:"))
                self.assertEqual(cast, [])


class VoiceDesignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runtime = RuntimeConfig(self.root, "http://x", self.root / "ComfyUI",
                                     self.root / "models.json", self.root / "outputs")

    def test_invalid_preview_never_prepares_runtime(self):
        for description, text, language in (("", "Hello", "Auto"), ("Warm", "", "Auto"),
                                            ("Warm", "<|speaker:5|> Hi", "Auto"),
                                            ("Warm", "Hi", "unsupported")):
            with self.subTest(text=text), patch.object(audio_tools, "run") as run:
                with self.assertRaises(H3Error):
                    voice_design.generate_reference(description, text, language, self.runtime, free_gpu=Mock())
                run.assert_not_called()

    def test_runtime_recipe_targets_the_isolated_interpreter_on_windows_and_modal(self):
        for venv, expected, wheel in (
                (PureWindowsPath("C:/h3/audio-tools"), "C:\\h3\\audio-tools\\Scripts\\python.exe", "windows"),
                (PurePosixPath("/opt/h3/audio-tools"), "/opt/h3/audio-tools/bin/python", "linux")):
            with self.subTest(venv=venv):
                commands = audio_tools_install_commands(venv, ["python", "-m", "uv"])
                self.assertEqual(commands[1][commands[1].index("--python") + 1], expected)
                self.assertIn("qwen-tts==0.1.1", commands[1])
                self.assertIn("hf-xet>=1.1", commands[1])
                self.assertIn("https://download.pytorch.org/whl/cu130", commands[1])
                self.assertIn("--no-deps", commands[2])
                url, digest = FLASH_ATTN_WHEELS[wheel]
                self.assertIn(f"flash-attn @ {url}#sha256={digest}", commands[2])
                self.assertEqual(commands[-1][0], expected)

    def test_generated_reference_uses_the_exact_synthesis_text_and_keeps_drama3_loaded(self):
        requests = []
        def run(model, payload, runtime, **kwargs):
            requests.append((model, payload))
            Path(payload["output"]).write_bytes(b"RIFF" + b"x" * 80)
            return {"ok": True}
        with patch.object(audio_tools, "run", side_effect=run), \
                patch("h3_app.drama3_serving.server.release_all") as release:
            audio, transcript = voice_design.generate_reference(
                "Warm narrator", "你好，今天我们一起出发。", "Chinese", self.runtime, free_gpu=Mock())
        release.assert_not_called()
        self.assertEqual(requests[0][0], "voice_design")
        self.assertEqual(transcript, requests[0][1]["text"])
        self.assertEqual(requests[0][1]["description"], "Warm narrator")
        self.assertTrue(audio.is_file())
        with patch.dict(os.environ, {"H3_WORKSPACE_DIR": str(self.root / "workspace")}):
            saved = drama3_speakers.save_speaker("Narrator", audio, transcript, self.runtime, "Warm narrator")
            self.assertEqual((saved.transcript, saved.description), (transcript, "Warm narrator"))
            metadata = saved.audio.parent / "speaker.json"
            old = json.loads(metadata.read_text(encoding="utf-8"))
            del old["description"]
            metadata.write_text(json.dumps(old), encoding="utf-8")
            self.assertEqual(drama3_speakers.load_speaker("Narrator", self.runtime).description, "")

    def test_failed_worker_removes_partial_audio(self):
        def run(model, payload, runtime, **kwargs):
            Path(payload["output"]).write_bytes(b"partial")
            raise H3Error("failed")
        with patch.object(audio_tools, "run", side_effect=run):
            with self.assertRaises(H3Error):
                voice_design.generate_reference("Warm", "Hello.", "Auto", self.runtime, free_gpu=Mock())
        self.assertEqual(list(self.runtime.output_dir.rglob("*.wav")), [])

    def test_saved_preset_identity_and_description_reach_the_matching_writer_slot(self):
        from h3_ui.application import drama3_handlers
        clip = self.root / "uploaded.wav"
        clip.write_bytes(b"RIFF" + b"x" * 80)
        response = json.dumps({"script": "<|speaker:5|> Hello there.", "speakers": [cast_entry(5, "Narrator")]})
        with patch.dict(os.environ, {"H3_WORKSPACE_DIR": str(self.root / "workspace")}), \
                patch.object(drama3_handlers, "_runtime_config", return_value=self.runtime), \
                patch.object(prompt_service, "_enhance_prompt_from_media", return_value=(response, "Enhanced prompt")) as writer:
            saved = drama3_speakers.save_speaker("Narrator", clip, "Uploaded words.", self.runtime, "A measured baritone.")
            selection = drama3_handlers.select_drama3_speaker(saved.name)
            self.assertEqual(selection[1]["value"], "Uploaded words.")
            self.assertEqual(selection[5], saved.name)
            self.assertEqual(selection[6]["value"], saved.description)
            voices = ["Upload a clip", None, ""] * 16
            voices[15:18] = [saved.name, None, ""]
            result = drama3_handlers.enhance_drama3_text("Original", "Auto", "model", "", "Lightning AI", "key", "llm",
                                                    *voices, *([""] * 48), 4)
            self.assertEqual(result[17:19], (saved.name, saved.description))
            self.assertIn("A measured baritone.", writer.call_args.kwargs["context"])
            self.assertEqual(drama3_speakers.load_speaker(saved.name, self.runtime).transcript, "Uploaded words.")

    def test_cancelled_setup_step_is_terminated_and_reaped(self):
        process = Mock()
        process.poll.return_value = None
        with patch.object(audio_tools.subprocess, "Popen", return_value=process), \
                patch.object(audio_tools, "check_cancelled", side_effect=[None, JobCancelled("cancelled")]):
            with self.assertRaises(JobCancelled):
                audio_tools.run_process(["python"], self.root / "worker.log", timeout=10)
        process.terminate.assert_called_once()
        process.wait.assert_called_once()

    def test_environment_is_ready_only_after_successful_smoke_test(self):
        venv = self.root / "venv"
        interpreter = audio_tools.python_path(venv)
        interpreter.parent.mkdir(parents=True)
        interpreter.touch()
        log = self.root / "install.log"
        with patch.dict(os.environ, {"H3_AUDIO_TOOLS_VENV": str(venv)}), \
                patch.object(audio_tools, "uv_command", return_value=["uv"]), \
                patch.object(audio_tools, "run_process", side_effect=H3Error("install failed")):
            with self.assertRaises(H3Error):
                audio_tools.prepare(self.runtime, log)
        self.assertFalse((venv / audio_tools.READY_MARKER).exists())
        with patch.dict(os.environ, {"H3_AUDIO_TOOLS_VENV": str(venv)}), \
                patch.object(audio_tools, "uv_command", return_value=["uv"]), \
                patch.object(audio_tools, "run_process") as run:
            self.assertEqual(audio_tools.prepare(self.runtime, log), interpreter)
            self.assertEqual(run.call_count, 3)
            audio_tools.prepare(self.runtime, log)
            self.assertEqual(run.call_count, 3)


REAL_RESIDENT_WORKER = audio_tools.ResidentWorker


class FakeResident:
    instances = []

    def __init__(self, python, models, log):
        self.python, self.models, self.log_path = python, models, log
        self.loaded, self.requests, self.stopped = set(), [], False
        self.error = None
        FakeResident.instances.append(self)

    def running(self):
        return not self.stopped

    def request(self, payload, timeout):
        self.requests.append(payload)
        if self.error:
            raise self.error
        self.loaded.add(payload["op"])
        return {"ok": True, "loaded": sorted(self.loaded)}

    def stop(self):
        self.stopped = True


class ResidentAudioToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runtime = RuntimeConfig(self.root, "http://x", self.root / "ComfyUI",
                                     self.root / "models.json", self.root / "outputs")
        FakeResident.instances = []
        self.addCleanup(audio_tools.stop)
        for name, replacement in (("ResidentWorker", FakeResident), ("ensure_weights", Mock()),
                                  ("prepare", Mock(return_value=Path("python")))):
            patcher = patch.object(audio_tools, name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        for name in ("persist_caches", "release_all"):
            patcher = patch(f"h3_app.drama3_serving.server.{name}")
            setattr(self, name, patcher.start())
            self.addCleanup(patcher.stop)

    def gpu(self, *free_mib):
        values = iter(free_mib)
        return patch("h3_app.drama3_serving.server.gpu_memory",
                     side_effect=lambda: (next(values), 98304.0, ""))

    def test_one_worker_serves_both_models_without_freeing_a_roomy_gpu(self):
        free_gpu = Mock()
        with self.gpu(40000, 30000):
            audio_tools.run("voice_design", {"text": "Hi"}, self.runtime, free_gpu=free_gpu)
            audio_tools.run("stable_audio", {"items": []}, self.runtime, free_gpu=free_gpu)
            audio_tools.run("voice_design", {"text": "Again"}, self.runtime, free_gpu=free_gpu)
        self.assertEqual(len(FakeResident.instances), 1)
        self.assertEqual([request["op"] for request in FakeResident.instances[0].requests],
                         ["voice_design", "stable_audio", "voice_design"])
        free_gpu.assert_not_called()
        self.release_all.assert_not_called()
        self.assertEqual(audio_tools.loaded_models(), {"voice_design", "stable_audio"})

    def test_a_full_gpu_frees_comfyui_first_then_sleeps_sglang(self):
        free_gpu = Mock()
        with self.gpu(1000, 20000):
            audio_tools.run("voice_design", {}, self.runtime, free_gpu=free_gpu)
        free_gpu.assert_called_once()
        self.release_all.assert_not_called()
        audio_tools.stop()
        free_gpu.reset_mock()
        with self.gpu(1000, 1000):
            audio_tools.run("stable_audio", {}, self.runtime, free_gpu=free_gpu)
        free_gpu.assert_called_once()
        self.release_all.assert_called_once()

    def test_other_engines_and_unload_stop_the_worker_but_drama3_jobs_keep_it(self):
        with self.gpu(40000):
            audio_tools.run("voice_design", {}, self.runtime, free_gpu=Mock())
        worker = FakeResident.instances[0]
        for family in ("drama3", "drama3_voice", "drama3_sound"):
            audio_tools.release_for_job(family)
            self.assertFalse(worker.stopped)
        audio_tools.release_for_job("h3")
        self.assertTrue(worker.stopped)
        self.assertEqual(audio_tools.loaded_models(), set())

    def test_cancellation_reaps_the_worker_but_a_failed_request_keeps_it(self):
        with self.gpu(40000, 40000, 40000):
            audio_tools.run("voice_design", {}, self.runtime, free_gpu=Mock())
            worker = FakeResident.instances[0]
            worker.error = audio_tools.RequestFailed("CUDA out of memory")
            with self.assertRaises(H3Error):
                audio_tools.run("voice_design", {}, self.runtime, free_gpu=Mock())
            self.assertFalse(worker.stopped)
            worker.error = JobCancelled("cancelled")
            with self.assertRaises(JobCancelled):
                audio_tools.run("voice_design", {}, self.runtime, free_gpu=Mock())
            self.assertTrue(worker.stopped)
            audio_tools.run("voice_design", {}, self.runtime, free_gpu=Mock())
        self.assertEqual(len(FakeResident.instances), 2)

    def test_resident_worker_round_trips_requests_over_pipes(self):
        import sys
        script = self.root / "echo_worker.py"
        script.write_text(
            "import json, sys\n"
            "for line in sys.stdin:\n"
            "    request = json.loads(line)\n"
            "    print('library noise', file=sys.stderr, flush=True)\n"
            "    ok = request['op'] != 'fail'\n"
            "    reply = {'id': request['id'], 'ok': ok, 'loaded': [request['op']], 'error': 'boom'}\n"
            "    sys.stdout.write(json.dumps(reply) + '\\n'); sys.stdout.flush()\n",
            encoding="utf-8")
        log = self.root / "worker.log"
        with patch.object(audio_tools, "WORKER", script):
            worker = REAL_RESIDENT_WORKER(Path(sys.executable), self.root, log)
        try:
            self.assertTrue(worker.request({"op": "voice_design"}, timeout=30)["ok"])
            self.assertEqual(worker.loaded, {"voice_design"})
            with self.assertRaises(audio_tools.RequestFailed):
                worker.request({"op": "fail"}, timeout=30)
            self.assertTrue(worker.running())
        finally:
            worker.stop()
        self.assertFalse(worker.running())
        self.assertIn("library noise", log.read_text(encoding="utf-8"))


class CastSlotTests(unittest.TestCase):
    def rewrite(self, response, *, name="", description="", reference=False):
        from h3_ui.application import drama3_handlers
        voices = ["Upload a clip", None, ""] * 16
        if reference:
            voices[15:18] = ["Upload a clip", "uploaded.wav", "The exact uploaded words."]
        metadata = [""] * 48
        metadata[15:18] = [name, description, "Old preview"]
        runtime = SimpleNamespace(prompt_systems={"Drama 3": Path("prompt_drama3.txt")})
        with patch.object(drama3_handlers, "_runtime_config", return_value=runtime), \
                patch.object(prompt_service, "_enhance_prompt_from_media", return_value=(response, "Enhanced prompt")):
            return drama3_handlers.enhance_drama3_text("Original", "Auto", "model", "", "Lightning AI", "key", "llm",
                                                  *voices, *metadata, 4)

    def test_writer_populates_the_matching_slot_and_reveals_sparse_ids(self):
        result = self.rewrite(writer_reply(5), name="Character 5", description="Existing description", reference=True)
        self.assertEqual(len(result), 70)
        self.assertEqual(result[17:20], ("Character 5", "Existing description", cast_entry(5)["preview_text"]))
        self.assertEqual(result[50], 6)
        self.assertTrue(result[56]["visible"])
        # Slots already hidden are left alone; only slots that change visibility are sent.
        self.assertEqual(result[57], {"__type__": "update"})
        self.assertEqual(result[51:55], ({"__type__": "update"},) * 4)

    def test_invalid_or_reassigned_cast_keeps_all_slots_unchanged(self):
        for response in (writer_reply(5), "{broken"):
            result = self.rewrite(response, name="Alice", reference=True)
            self.assertEqual(result[0], "Original")
            self.assertIn("failed", result[1])
            self.assertTrue(all(value == {"__type__": "update"} for value in result[2:]))

    def test_copied_prompt_contains_system_prompt_slot_context_and_speech_text(self):
        from h3_ui.application import drama3_handlers
        voices = ["Upload a clip", None, ""] * 16
        metadata = [""] * 48
        metadata[6:9] = ["Mia", "Bright young voice", ""]
        runtime = SimpleNamespace(prompt_systems={"Drama 3": Path("prompt_drama3.txt")})
        with patch.object(drama3_handlers, "_runtime_config", return_value=runtime):
            prompt, status = drama3_handlers.copy_drama3_writer_prompt("MIA: Hello there.", "Auto",
                                                                   *voices, *metadata, 4)
            _skip, failed = drama3_handlers.copy_drama3_writer_prompt("  ", "Auto", *voices, *metadata, 4)
        text = prompt["value"]
        self.assertTrue(text.startswith(Path("prompt_drama3.txt").read_text(encoding="utf-8").strip()[:80]))
        self.assertIn('"speaker_id": 2, "name": "Mia"', text)
        self.assertTrue(text.rstrip().endswith("User text:\nMIA: Hello there."))
        self.assertTrue(prompt["visible"])
        self.assertIn("paste", status)
        self.assertIn("speech text", failed)

    def test_pasted_reply_fills_slots_like_the_built_in_writer(self):
        from h3_ui.application import drama3_handlers
        voices = ["Upload a clip", None, ""] * 16
        runtime = SimpleNamespace(prompt_systems={"Drama 3": Path("prompt_drama3.txt")})
        fenced = "Here is the result:\n```json\n" + writer_reply(5) + "\n```"
        with patch.object(drama3_handlers, "_runtime_config", return_value=runtime):
            result = drama3_handlers.apply_drama3_writer_reply(fenced, "Original", "Auto", *voices, *[""] * 48, 4)
            bad = drama3_handlers.apply_drama3_writer_reply("not json", "Original", "Auto", *voices, *[""] * 48, 4)
        self.assertEqual(result[0], "<|speaker:5|> Hello there.")
        self.assertIn("Applied the pasted", result[1])
        self.assertEqual(result[17:20], ("Character 5", cast_entry(5)["description"], cast_entry(5)["preview_text"]))
        self.assertEqual(result[50], 6)
        self.assertEqual(bad[0], "Original")
        self.assertIn("kept", bad[1])
        self.assertTrue(all(value == {"__type__": "update"} for value in bad[2:]))

    def test_unchecked_skip_regenerates_speakers_that_already_have_a_clip(self):
        from h3_ui.application import drama3_handlers
        from h3_ui.drama3_controls import UPLOAD_VOICE, drama3_voices_title
        calls = []

        def single(description, *args):
            calls.append(description)
            yield "new.wav", {"__type__": "update", "value": "Hey."}, "Voice generated."
        settings = ("Qwen3-TTS VoiceDesign", 1.0, 0.9, 0, 8192)
        slots = [[UPLOAD_VOICE, False, "", "", "Auto"] for _ in range(16)]
        slots[0] = [UPLOAD_VOICE, True, "Deep", "Hey.", "Auto"]
        slots[1] = ["Saved Alice", True, "Bright", "Hi.", "Auto"]
        with patch.object(drama3_handlers, "generate_drama3_voice", side_effect=single):
            list(drama3_handlers.generate_drama3_voices(*settings, 8, 2, False, *[v for slot in slots for v in slot]))
        self.assertEqual(calls, ["Deep"])  # saved speakers are never overwritten
        self.assertEqual(drama3_voices_title(3, ["Saved Alice", UPLOAD_VOICE, UPLOAD_VOICE], [False, True, False]),
                         "Speaker voices · 3 speakers · 2 with a voice")
        self.assertEqual(drama3_voices_title(1, [UPLOAD_VOICE], [True, True]),
                         "Speaker voices · 1 speaker · 1 with a voice")
        self.assertEqual(drama3_voices_title(2, [UPLOAD_VOICE], [True], group="Chapter 1"),
                         "Speaker voices · Chapter 1 group · 2 speakers · 1 with a voice")

    def test_generation_failure_preserves_current_audio_and_transcript(self):
        from h3_ui.application import drama3_handlers
        with patch.object(drama3_handlers, "_runtime_config"), \
                patch.object(voice_design, "generate_reference", side_effect=H3Error("broken")):
            updates = list(drama3_handlers.generate_drama3_voice("Warm", "Hello.", "Auto", "Qwen3-TTS VoiceDesign"))
        self.assertEqual(updates[-1][:2], ({"__type__": "update"}, {"__type__": "update"}))
        self.assertIn("failed", updates[-1][2])

    def test_drama3_generator_uses_the_tab_sampling(self):
        from h3_ui.application import drama3_handlers
        from h3_app import drama3_voice_design
        from h3_app.generation.results import GenerationUpdate
        seen = {}

        def design(description, text, request, runtime, services, **kwargs):
            seen.update(description=description, text=text, request=request)
            yield GenerationUpdate(None, "Generating")
            yield GenerationUpdate("voice.wav", "Hello.")
        with patch.object(drama3_handlers, "_runtime_config"), patch.object(drama3_handlers, "_generation_services"), \
                patch("h3_models.resolve_hf_token", return_value=None), \
                patch.object(voice_design, "generate_reference") as qwen, \
                patch.object(drama3_voice_design, "generate_drama3_reference", side_effect=design):
            updates = list(drama3_handlers.generate_drama3_voice(
                "Warm", "Hello.", "Auto", "Drama 3", temperature=0.7, top_k=30, max_frames=4096))
        qwen.assert_not_called()
        self.assertEqual((seen["request"].temperature, seen["request"].top_k, seen["request"].max_frames),
                         (0.7, 30, 4096))
        self.assertEqual(updates[-1][0], "voice.wav")
        self.assertEqual(updates[-1][1]["value"], "Hello.")


    def test_generate_all_fills_only_shown_upload_slots_with_a_brief_and_no_clip(self):
        from h3_ui.application import drama3_handlers
        from h3_ui.drama3_controls import UPLOAD_VOICE, drama3_slot_title
        calls = []

        def single(description, preview, language, *settings):
            calls.append((description, preview, settings))
            yield {"__type__": "update"}, {"__type__": "update"}, "Preparing"
            if description == "Broken":
                yield {"__type__": "update"}, {"__type__": "update"}, "Voice generation failed: no GPU"
                return
            yield f"{description}.wav", {"__type__": "update", "value": preview}, "Voice generated."
        settings = ("Qwen3-TTS VoiceDesign", 1.0, 0.9, 0, 8192)
        slots = [[UPLOAD_VOICE, False, "", "", "Auto"] for _ in range(16)]
        slots[0] = [UPLOAD_VOICE, False, "Warm", "Hello there.", "Auto"]
        slots[1] = ["Saved Alice", False, "Bright", "Hi.", "Auto"]
        slots[2] = [UPLOAD_VOICE, True, "Deep", "Hey.", "Auto"]
        slots[3] = [UPLOAD_VOICE, False, "Broken", "Oops.", "Auto"]
        slots[5] = [UPLOAD_VOICE, False, "Hidden", "Not shown.", "Auto"]
        with patch.object(drama3_handlers, "generate_drama3_voice", side_effect=single):
            updates = list(drama3_handlers.generate_drama3_voices(*settings, 8, 4, True, *[v for slot in slots for v in slot]))
        self.assertEqual([call[0] for call in calls], ["Warm", "Broken"])
        self.assertEqual(calls[0][2], settings)
        self.assertEqual(len(updates[-1]), 49)
        first = next(index for index, update in enumerate(updates) if update[0] == "Warm.wav")
        # A clip repeats in every later frame; a skip after it would corrupt the Audio in the browser.
        self.assertEqual({update[0] for update in updates[first:]}, {"Warm.wav"})
        self.assertFalse(any(isinstance(update[3 * 3], str) for update in updates))
        self.assertIn("already has a clip", updates[0][3 * 2 + 2])
        self.assertIn("Generated 1 of 2", updates[-1][-1])
        self.assertIn("speaker 3", updates[-1][-1])
        self.assertEqual(drama3_slot_title(2), "Speaker 2")
        self.assertEqual(drama3_slot_title(2, " Mia ", UPLOAD_VOICE, True), "Speaker 2 · Mia · voice ready")
        self.assertEqual(drama3_slot_title(3, "Alice", "Alice"), "Speaker 3 · Alice · saved voice")

    def test_generate_all_with_drama3_sends_every_brief_in_one_batch_with_the_parallel_setting(self):
        from h3_ui.application import drama3_handlers
        from h3_app import drama3_voice_design
        from h3_app.drama3_voice_design import DesignResult
        from h3_app.generation.results import GenerationUpdate
        from h3_ui.drama3_controls import UPLOAD_VOICE
        seen = {}

        def batch(briefs, request, *args, **kwargs):
            seen.update(briefs=briefs, request=request)
            yield GenerationUpdate(None, "Generated 1/2 voice(s)")
            yield DesignResult(1, "one.wav", "Hi there.")
            yield DesignResult(0, None, "Hello.", "silent take")
        settings = ("Drama 3", 0.7, 0.7, 30, 8192)
        slots = [[UPLOAD_VOICE, False, "", "", "Auto"] for _ in range(16)]
        slots[0] = [UPLOAD_VOICE, False, "Warm", "Hello.", "Auto"]
        slots[1] = [UPLOAD_VOICE, False, "Bright", "Hi there.", "Auto"]
        with patch.object(drama3_handlers, "_runtime_config"), patch.object(drama3_handlers, "_generation_services"), \
                patch("h3_models.resolve_hf_token", return_value=None), \
                patch.object(drama3_handlers, "generate_drama3_voice") as single, \
                patch.object(drama3_voice_design, "generate_drama3_references", side_effect=batch):
            updates = list(drama3_handlers.generate_drama3_voices(*settings, 6, 2, True, *[v for slot in slots for v in slot]))
        single.assert_not_called()
        self.assertEqual(seen["briefs"], {0: ("Warm", "Hello."), 1: ("Bright", "Hi there.")})
        self.assertEqual((seen["request"].parallel, seen["request"].top_k), (6, 30))
        clip = next(update for update in updates if update[3] == "one.wav")
        self.assertEqual(clip[4]["value"], "Hi there.")
        self.assertEqual(updates[-1][3], "one.wav")
        self.assertTrue(any("silent take" in str(update[2]) for update in updates))
        self.assertIn("Generated 1 of 2", updates[-1][-1])


class Drama3VoiceDesignTests(unittest.TestCase):
    def setUp(self):
        from h3_app.generation.requests import Drama3Request
        self.temp = tempfile.TemporaryDirectory()
        self.runtime = SimpleNamespace(output_dir=Path(self.temp.name))
        self.services = SimpleNamespace(models=Mock())
        self.request = Drama3Request("", (), 0.7, 0.7, 30, 8192)

    def tearDown(self):
        self.temp.cleanup()

    def test_description_cue_follows_the_opening_phrase(self):
        from h3_app.drama3_voice_design import design_script
        self.assertEqual(design_script(" Gravelly  old sailor ", "Well now, the  tide is turning."),
                         "<|speaker:0|> Well now, [Gravelly old sailor] the tide is turning.")
        self.assertEqual(design_script("Warm", "Hello world"), "<|speaker:0|> Hello [Warm] world")
        self.assertEqual(design_script("Warm", "你好，今天我们一起出发。"),
                         "<|speaker:0|> 你好， [Warm] 今天我们一起出发。")
        for description, text in (("A [bright] voice", "Hello there."), ("Warm", "Hello [laughs] there."),
                                  ("", "Hello."), ("Warm", "")):
            with self.subTest(description=description, text=text), self.assertRaises(H3Error):
                design_script(description, text)

    def test_sglang_batch_runs_takes_concurrently_and_keeps_failures_per_voice(self):
        import threading
        import numpy as np
        from dataclasses import replace
        from h3_app import drama3_voice_design
        from h3_app.drama3_serving import audio, generate, server as served
        clip = audio.encode_wav(np.full(24000, 0.2, dtype=np.float32), 24000)
        barrier = threading.Barrier(2, timeout=5)  # both requests must be in flight at once

        def speech(payload):
            barrier.wait()
            if "Broken" in payload["input"]:
                raise H3Error("HTTP 500")
            return clip
        fake = Mock()
        fake.payload.side_effect = lambda **kwargs: served.SglangOmniServer.payload(None, **kwargs)
        fake.speech.side_effect = speech
        with patch.object(generate, "ServerSettings"), patch.object(generate, "shared_server", return_value=fake), \
                patch.object(generate, "_ready_server", return_value=iter(())) as ready, \
                patch.object(generate, "persist_caches"):
            results = [update for update in drama3_voice_design.generate_drama3_references(
                {0: ("Warm", "Hello there."), 1: ("Broken", "Oh no, again."), 2: ("[bad]", "Hi.")},
                replace(self.request, parallel=4), self.runtime, self.services,
                free_gpu=Mock()) if isinstance(update, drama3_voice_design.DesignResult)]
        ready.assert_called_once()
        by_key = {result.key: result for result in results}
        self.assertTrue(Path(by_key[0].audio).is_file())
        self.assertEqual(by_key[0].transcript, "Hello there.")
        self.assertEqual((by_key[1].audio, by_key[1].error), (None, "HTTP 500"))
        self.assertIn("brackets", by_key[2].error)
        self.assertEqual(fake.speech.call_count, 2)

    def test_sglang_sends_no_reference_and_writes_the_clip(self):
        import numpy as np
        from h3_app import drama3_voice_design
        from h3_app.drama3_serving import audio, generate, server as served
        clip = audio.encode_wav(np.full(24000, 0.2, dtype=np.float32), 24000)
        fake = Mock()
        fake.payload.side_effect = lambda **kwargs: served.SglangOmniServer.payload(None, **kwargs)
        fake.speech.return_value = clip
        with patch.object(generate, "ServerSettings"), patch.object(generate, "shared_server", return_value=fake), \
                patch.object(generate, "_ready_server", return_value=iter(())), \
                patch.object(generate, "persist_caches"):
            updates = list(drama3_voice_design.generate_drama3_reference(
                "Warm", "Hello there.", self.request, self.runtime, self.services, free_gpu=Mock()))
        payload = fake.speech.call_args.args[0]
        self.assertEqual(payload["references"], [])
        self.assertEqual(payload["input"], "<|speaker:0|> Hello [Warm] there.")
        self.assertEqual((payload["temperature"], payload["top_k"]), (0.7, 30))
        path = Path(updates[-1].output)
        self.assertEqual((path.parent.name, updates[-1].status), ("drama3_voices", "Hello there."))
        self.assertTrue(path.is_file())
        # The voice-design job keeps the server it is about to use; other engines still release it.
        with patch.object(served, "release_all") as release:
            served.release_for_job("drama3_voice")
            release.assert_not_called()
            served.release_for_job("flux3")
            release.assert_called_once()


class BrowserSlotPresentationTests(unittest.TestCase):
    """Slot titles, panel titles and Add/Remove run in the browser and must match their Python twins."""

    def test_slot_titles_and_clip_flags_match_python(self):
        from tests.node_js import run_js
        from h3_ui.drama3_controls import (
            _SOUND_KIND, _VOICE_KIND, _slot_title_js, drama3_slot_title, drama3_sound_slot_title
        )
        audio = {"path": "/tmp/voice.wav", "url": "/file=voice.wav"}
        for name, preset, clip in ((None, None, None), (" Mia ", "Upload a clip", audio),
                                   ("Alice", "Alice", None), ("", "Upload a clip", None)):
            with self.subTest(name=name, preset=preset, clip=bool(clip)):
                label, ready = run_js(_slot_title_js(_VOICE_KIND, 3), name, preset, clip)
                self.assertEqual(label["label"], drama3_slot_title(3, name, preset, bool(clip)))
                self.assertIs(ready, bool(clip))
                label, _ready = run_js(_slot_title_js(_SOUND_KIND, 2), name, preset, clip)
                self.assertEqual(label["label"], drama3_sound_slot_title(2, name, preset, bool(clip)))

    def test_panel_titles_match_python(self):
        from tests.node_js import run_js
        from h3_ui.drama3_controls import (
            UPLOAD_VOICE, _VOICE_KIND, _SOUND_KIND, _panel_title_js, drama3_sounds_title, drama3_voices_title
        )
        presets = ["Saved Alice", UPLOAD_VOICE, UPLOAD_VOICE] + [UPLOAD_VOICE] * 13
        clips = [False, True, False, True] + [False] * 12
        for shown in (1, 3, 4, 16, 0, None):
            with self.subTest(shown=shown):
                self.assertEqual(run_js(_panel_title_js(_VOICE_KIND), shown, *presets, *clips)["label"],
                                 drama3_voices_title(shown, presets, clips))
                self.assertEqual(run_js(_panel_title_js(_SOUND_KIND), shown, *presets, *clips)["label"],
                                 drama3_sounds_title(shown, presets, clips))
        for group, size in (("Global", 3), ("Chapter 1", 0), ("Story", None), ("", 5)):
            with self.subTest(group=group, size=size):
                self.assertEqual(run_js(_panel_title_js(_VOICE_KIND, True), group, size, 4, *presets, *clips)["label"],
                                 drama3_voices_title(4, presets, clips, group, size))
        self.assertIn("Story group (2 saved) · 4 speakers", drama3_voices_title(4, group="Story", group_size=2))

    def test_add_and_remove_send_only_changed_slots_and_clear_the_removed_one(self):
        from tests.node_js import run_js
        from h3_ui.drama3_controls import _slot_count_js, _slot_visibility
        skip = {"__type__": "update"}
        added = run_js(_slot_count_js(16, 1), 4)
        self.assertEqual(added, [{**value} if isinstance(value, dict) else value
                                 for value in _slot_visibility(5, 4)])
        self.assertEqual(added[1 + 4], {"__type__": "update", "visible": True})
        self.assertEqual(added[1:5], [skip] * 4)
        self.assertEqual(run_js(_slot_count_js(16, 1), 16)[0], 16)
        removed = run_js(_slot_count_js(16, -1, [["Upload a clip", None, ""]]), 4)
        self.assertEqual(removed[:19], [{**value} if isinstance(value, dict) else value
                                        for value in _slot_visibility(3, 4)])
        cleared = removed[19:]
        self.assertEqual(len(cleared), 48)
        self.assertEqual(cleared[9:12], [{"__type__": "update", "value": "Upload a clip"},
                                         {"__type__": "update", "value": None},
                                         {"__type__": "update", "value": ""}])
        self.assertTrue(all(value == skip for index, value in enumerate(cleared) if not 9 <= index < 12))
        # The last slot is never removed or cleared.
        self.assertTrue(all(value == skip for value in run_js(_slot_count_js(16, -1, [["x"]]), 1)[19:]))


if __name__ == "__main__":
    unittest.main()
