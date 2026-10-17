"""SGLang-Omni Drama 3 route: environment, server lifecycle, chunking, checks and joining."""

from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from h3_app.drama3_serving import audio, environment, generate, server
from h3_app.drama3_serving.planner import expected_seconds, parse_turns, plan_chunks, spoken_text
from h3_app.errors import H3Error
from h3_app.generation.optional_engines import validate_request
from h3_app.generation.requests import Drama3Request
from h3_requirements import drama3_sglang_install_commands

RATE = 44100


def tone(seconds, rate=RATE, silent_tail=0.0):
    t = np.arange(int(seconds * rate)) / rate
    voiced = 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    return np.concatenate([voiced, np.zeros(int(silent_tail * rate), dtype=np.float32)])


def float_wav(samples, rate=RATE):
    payload = samples.astype("<f4").tobytes()
    fmt = struct.pack("<HHIIHH", 3, 1, rate, rate * 4, 4, 32)
    return (b"RIFF" + struct.pack("<I", 36 + len(payload)) + b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt
            + b"data" + struct.pack("<I", len(payload)) + payload)


class PlannerTests(unittest.TestCase):
    def test_turns_merge_consecutive_lines_of_one_speaker(self):
        script = "Intro line.\n<|speaker:1|> Hi there.\n<|speaker:1|> Still me.\n<|speaker:0|> Back to zero."
        self.assertEqual(parse_turns(script),
                         [(0, "Intro line."), (1, "Hi there."), (1, "Still me."), (0, "Back to zero.")])
        chunks = plan_chunks(script)
        self.assertEqual([(c.speaker, c.text) for c in chunks],
                         [(0, "Intro line."), (1, "Hi there. Still me."), (0, "Back to zero.")])
        self.assertEqual([c.index for c in chunks], [0, 1, 2])
    def test_long_turns_split_at_sentence_ends_without_breaking_cues(self):
        sentence = "This sentence has [softly, almost to herself. Then louder] a cue with a period inside."
        chunks = plan_chunks("<|speaker:2|> " + " ".join([sentence] * 8), limit=200)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(chunk.speaker, 2)
            self.assertEqual(chunk.text.count("["), chunk.text.count("]"))
            self.assertLessEqual(len(chunk.text.encode()), 200 + len(sentence))
    def test_spoken_text_and_expected_length(self):
        self.assertEqual(spoken_text("<|speaker:0|> I got [laughing] <emphasis>the job</emphasis>."), "I got the job .")
        self.assertAlmostEqual(expected_seconds("one two three four five [sighs]"), 2.5)
        self.assertAlmostEqual(expected_seconds("等我一下咱们继续看这只包"), 12 / 4.5)


class AudioTests(unittest.TestCase):
    def test_decodes_float_and_pcm16_wav(self):
        samples = tone(0.5)
        decoded, rate = audio.decode_wav(float_wav(samples))
        self.assertEqual(rate, RATE)
        np.testing.assert_allclose(decoded, samples, atol=1e-6)
        pcm, rate = audio.decode_wav(audio.encode_wav(samples, RATE))
        np.testing.assert_allclose(pcm, samples, atol=1e-4)
        with self.assertRaises(ValueError):
            audio.decode_wav(b"not a wav")
    def test_take_checks(self):
        self.assertTrue(audio.check_take(tone(4.0), RATE, 4.0).ok)
        self.assertEqual(audio.check_take(np.zeros(RATE * 10, np.float32), RATE, 4.0).reason, "no speech")
        self.assertIn("too short", audio.check_take(tone(1.0, silent_tail=9.0), RATE, 10.0).reason)
        self.assertIn("runs on", audio.check_take(tone(30.0), RATE, 4.0).reason)
    def test_quiet_short_speech_and_edge_padding_are_not_failures(self):
        quiet = tone(0.12) * 0.001
        padded = np.concatenate([np.zeros(2 * RATE), quiet, np.zeros(3 * RATE)]).astype(np.float32)
        check = audio.check_take(padded, RATE, 0.8)
        self.assertTrue(check.ok)
        self.assertFalse(check.warnings)
        self.assertGreater(len(audio.tidy_take(padded, RATE)), 0)
    def test_duration_and_internal_pause_heuristics_are_advisory(self):
        for samples, expected, warning in (
            (tone(1.0), 10.0, "too short"),
            (tone(30.0), 4.0, "runs on"),
            (np.concatenate([tone(1.0), np.zeros(5 * RATE), tone(1.0)]), 2.0, "internal pauses"),
        ):
            with self.subTest(warning=warning):
                check = audio.check_take(samples, RATE, expected)
                self.assertTrue(check.ok)
                self.assertIn(warning, " ".join(check.warnings))
    def test_invalid_audio_is_not_scored_as_a_good_take(self):
        for samples, rate in ((tone(1.0), 0), (np.array([np.nan], dtype=np.float32), RATE)):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                audio.check_take(samples, rate, 1.0)
    def test_tidy_trims_edges_and_shortens_long_silences(self):
        take = np.concatenate([np.zeros(RATE), tone(1.0), np.zeros(4 * RATE), tone(1.0), np.zeros(RATE)])
        tidied = audio.tidy_take(take.astype(np.float32), RATE)
        self.assertLess(len(tidied) / RATE, 2.0 + audio.KEPT_GAP + 2 * audio.EDGE_PAD_SECONDS + 0.1)
        self.assertGreater(len(tidied) / RATE, 2.0)
    def test_join_uses_turn_and_same_speaker_gaps(self):
        joined = audio.join_takes([(0, tone(1.0)), (0, tone(1.0)), (1, tone(1.0))], RATE)
        expected = 3.0 + audio.SAME_SPEAKER_GAP + audio.TURN_GAP
        self.assertAlmostEqual(len(joined) / RATE, expected, places=2)


class EnvironmentTests(unittest.TestCase):
    def test_uv_runs_as_a_module_when_no_executable_is_on_path(self):
        with patch.object(environment.shutil, "which", return_value=None), \
                patch("importlib.util.find_spec", return_value=object()):
            uv = environment.uv_command()
        self.assertEqual(uv[1:], ["-m", "uv"])
        commands = drama3_sglang_install_commands(PurePosixPath("/v"), uv)
        self.assertEqual(commands[0][:4], [uv[0], "-m", "uv", "venv"])
        self.assertEqual(commands[2][:4], [uv[0], "-m", "uv", "pip"])
        with patch.object(environment.shutil, "which", return_value=None), \
                patch("importlib.util.find_spec", return_value=None):
            self.assertIsNone(environment.uv_command())
    def test_sglang_install_pins_the_release_with_fishaudio_extras(self):
        commands = drama3_sglang_install_commands(PurePosixPath("/opt/h3/drama3-sglang-venv"))
        self.assertIn("protobuf>=6.31.1,<7.0.0", commands[1][-1])
        self.assertEqual(commands[2][commands[2].index("--overrides") + 1], "/opt/h3/drama3-sglang-venv/h3-overrides.txt")
        self.assertIn("--prerelease=allow", commands[2])
        self.assertIn("--torch-backend=cu130", commands[2])
        self.assertTrue(commands[2][-1].startswith("sglang-omni[fishaudio] @ git+https://github.com/sgl-project/sglang-omni.git@caae7eb"))
        self.assertEqual(environment.default_venv(Path("/app")), Path("/app/h3/drama3-sglang-venv"))
    def test_readiness_requires_the_current_fingerprint(self):
        with tempfile.TemporaryDirectory() as root:
            venv = Path(root) / "venv"
            (venv / "bin").mkdir(parents=True)
            (venv / "bin" / "python").write_text("")
            self.assertFalse(environment.is_ready(venv))
            (venv / environment.READY_MARKER).write_text("stale")
            self.assertFalse(environment.is_ready(venv))
            (venv / environment.READY_MARKER).write_text(environment.fingerprint())
            self.assertTrue(environment.is_ready(venv))
    def test_install_env_scrubs_inherited_python_settings(self):
        with patch.dict("os.environ", {"PYTHONPATH": "/comfy", "VIRTUAL_ENV": "/main"}):
            env = environment.install_env(Path("/venv"))
        self.assertNotIn("PYTHONPATH", env)
        self.assertNotIn("VIRTUAL_ENV", env)
        self.assertEqual(env["PYTHONNOUSERSITE"], "1")
    def test_prepare_refuses_non_linux(self):
        with patch.object(environment.os, "name", "nt"), self.assertRaisesRegex(H3Error, "Linux only"):
            environment.prepare(Path("venv"))


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.settings = server.ServerSettings(venv=root / "venv", data_dir=root / "data", model_root=root / "models")
        no_gpu_query = patch.object(server, "gpu_memory", return_value=None)  # never call the real nvidia-smi
        no_gpu_query.start()
        self.addCleanup(no_gpu_query.stop)

    def settings_for(self, **changes):
        return server.ServerSettings(**{**self.settings.__dict__, **changes})

    def test_sglang_environment_overrides_and_defaults(self):
        runtime = SimpleNamespace(script_dir=Path("/app"), comfy_dir=Path("/app/ComfyUI"))
        with patch.dict("os.environ", {}, clear=True):
            defaults = server.ServerSettings.from_environment(runtime)
        self.assertEqual(defaults.port, 8096)
        self.assertEqual(defaults.sleep_mode, "auto")
        with patch.dict("os.environ", {"H3_DRAMA3_SGLANG_SLEEP": "off"}):
            self.assertEqual(server.ServerSettings.from_environment(runtime).sleep_mode, "off")
        with self.assertRaisesRegex(ValueError, "auto or off"):
            self.settings_for(sleep_mode="invalid")
        self.assertEqual(defaults.data_dir, Path("/app/drama3-serving"))
        self.assertEqual(defaults.model_root, Path("/app/ComfyUI/models/drama3_serving"))
        with patch.dict("os.environ", {"H3_DRAMA3_SGLANG_DATA": "/data/d3",
                                       "H3_DRAMA3_SGLANG_ENGINE_MEMORY": "0.6",
                                       "H3_DRAMA3_SGLANG_CODEC_MEMORY": "0.1",
                                       "H3_DRAMA3_SGLANG_MAX_SEQS": "8",
                                       "H3_DRAMA3_SGLANG_PORT": "9000"}, clear=True):
            custom = server.ServerSettings.from_environment(runtime)
        self.assertEqual(custom.data_dir, Path("/data/d3"))
        self.assertEqual((custom.engine_memory, custom.codec_memory, custom.max_seqs, custom.port),
                         (0.6, 0.1, 8, 9000))

    def test_existing_pinned_checkpoint_is_reused_after_storage_rename(self):
        settings = self.settings_for(model_root=Path(self.temp.name) / "models" / "serving")
        target = settings.model_dir
        previous = settings.model_root.parent / "earlier-serving" / target.name
        previous.mkdir(parents=True)
        marker = previous / ".h3-complete"
        marker.write_text("stale", encoding="utf-8")
        self.assertEqual(settings.model_dir, target)
        marker.write_text(server.DRAMA3_HF_REVISION, encoding="utf-8")
        self.assertEqual(settings.model_dir, previous)
        self.assertEqual(server.ensure_model(settings), previous)

    def test_existing_sglang_kernel_cache_is_reused(self):
        previous = self.settings.data_dir.parent / "earlier-serving" / "sglang"
        (previous / "cache").mkdir(parents=True)
        self.assertEqual(self.settings.runtime_dir, previous)
        (self.settings.data_dir / "cache").mkdir(parents=True)
        self.assertEqual(self.settings.runtime_dir, self.settings.data_dir)

    def test_single_server_is_reused_and_replaced_when_settings_change(self):
        with patch.object(server, "_SERVER", None):
            first = server.shared_server(self.settings)
            self.assertIs(server.shared_server(self.settings), first)
            with patch.object(first, "stop") as stop:
                second = server.shared_server(self.settings_for(port=9000))
                stop.assert_called_once()
            self.assertIsNot(first, second)
            with patch.object(second, "release") as release:
                server.release_all()
                release.assert_called_once()

    def test_compile_and_jit_caches_live_with_the_persistent_data(self):
        env = server.SglangOmniServer(self.settings).server_env()
        cache = self.settings.runtime_dir / "cache"
        for key, folder in (("FLASHINFER_WORKSPACE_BASE", "flashinfer"), ("XDG_CACHE_HOME", "xdg"),
                            ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"), ("TRITON_CACHE_DIR", "triton"),
                            ("CUDA_CACHE_PATH", "cuda")):
            self.assertEqual(env[key], str(cache / folder))
    def test_volume_commit_runs_only_on_modal(self):
        with patch.dict("os.environ", {}, clear=True), patch.object(server.threading, "Thread") as thread:
            server.persist_caches()
            thread.assert_not_called()
        with patch.dict("os.environ", {"H3_MODAL_COMMIT_VOLUME": "vol"}), \
                patch.object(server.threading, "Thread") as thread:
            server.persist_caches()
            thread.return_value.start.assert_called_once()

    def test_startup_progress_is_not_written_back_into_the_server_log(self):
        instance = server.SglangOmniServer(self.settings)
        instance._log = MagicMock()
        instance.log_line("Starting SGLang-Omni (30s): loading", to_file=False)
        instance._log.write.assert_not_called()
        instance.log_line("SGLang-Omni ready after 40s")
        instance._log.write.assert_called_once()

    def test_sglang_command_and_payload(self):
        settings = self.settings_for(port=8096)
        instance = server.SglangOmniServer(settings)
        command = instance.command()
        self.assertEqual(command[1:3], ["serve", "--model-path"])
        self.assertEqual(command[command.index("--tts_engine.gpu_memory_fraction") + 1], "0.55")
        self.assertEqual(command[command.index("--vocoder.gpu_memory_fraction") + 1], "0.08")
        self.assertEqual(command[command.index("--tts_engine.factory.max_running_requests") + 1], "32")
        self.assertEqual(command[command.index("--tts_engine.engine.cuda_graph_backend_prefill") + 1], "disabled")
        self.assertEqual(command[command.index("--tts_engine.factory.first_frame_cuda_graph_max_requests") + 1], "8")
        self.assertEqual(command[command.index("--vocoder.factory.window_cuda_graphs") + 1], "false")
        self.assertEqual(command[command.index("--vocoder.factory.codec_path") + 1],
                         str(settings.model_dir / "codec.pth"))  # offline: no S2-Pro Hub download
        self.assertEqual(settings.runtime_dir, settings.data_dir)
        payload = instance.payload(text="Hi.", reference=Path(self.temp.name) / "a.wav", transcript="Ref.",
                                   max_new_tokens=300, sampling={"temperature": 1.0, "top_p": 0.9, "top_k": 30})
        self.assertEqual(payload["references"], [{"audio_path": str((Path(self.temp.name) / "a.wav").resolve()),
                                                  "text": "Ref."}])
        self.assertNotIn("seed", payload)
        self.assertEqual((payload["top_k"], payload["max_new_tokens"]), (30, 300))
        sampling = {"temperature": 1.0, "top_p": 0.9, "top_k": 0}
        uncapped = instance.payload(text="Hi.", reference=Path(self.temp.name) / "a.wav",
                                   transcript="Ref.", max_new_tokens=300, sampling=sampling)
        self.assertEqual(uncapped["top_k"], 1 << 30)
        self.assertEqual(sampling["top_k"], 0)

    def test_a_loaded_server_is_reused_after_releasing_comfyui(self):
        instance = server.SglangOmniServer(self.settings)
        instance.running = lambda: True
        instance.healthy = lambda timeout=2.0: True
        free = MagicMock()
        instance.ensure_ready(free)
        free.assert_called_once()
    def test_cold_start_frees_gpu_then_waits_for_health(self):
        instance = server.SglangOmniServer(self.settings)
        order = []
        instance._start = lambda: order.append("start")
        instance._wait_healthy = lambda progress: order.append("ready")
        instance.ensure_ready(lambda: order.append("free"))
        self.assertEqual(order, ["free", "start", "ready"])

    def test_failed_comfyui_release_prevents_server_start(self):
        instance = server.SglangOmniServer(self.settings)
        instance._start = MagicMock()
        with self.assertRaisesRegex(H3Error, "still reserved"):
            instance.ensure_ready(MagicMock(side_effect=H3Error("still reserved")))
        instance._start.assert_not_called()

    def test_start_is_refused_when_the_gpu_is_still_occupied(self):
        instance = server.SglangOmniServer(self.settings)
        instance._start = MagicMock()
        with patch.object(server, "gpu_memory", return_value=(1024.0, 97280.0, "1, python, 94000 MiB")), \
                self.assertRaisesRegex(H3Error, "Still holding VRAM: 1, python, 94000 MiB"):
            instance.ensure_ready(lambda: None)
        instance._start.assert_not_called()
        instance._wait_healthy = MagicMock()
        with patch.object(server, "gpu_memory", return_value=(90000.0, 97280.0, "")):
            instance.ensure_ready(lambda: None)
        instance._start.assert_called_once()
    def test_other_engines_release_the_servers_when_they_take_the_gpu(self):
        with patch.object(server, "release_all") as release_all:
            server.release_for_job("drama3")
            release_all.assert_not_called()
            server.release_for_job("qwen_image21")
            release_all.assert_called_once_with()
    def test_the_job_lock_runs_claim_hooks_before_the_job(self):
        from h3_app.jobs import JobCoordinator

        coordinator = JobCoordinator()
        seen = []
        coordinator.add_gpu_claim_hook(seen.append)
        coordinator.add_gpu_claim_hook(seen.append)  # registered once
        with coordinator.run("owner", "flux3"):
            self.assertEqual(seen, ["flux3"])


class GenerateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.runtime = SimpleNamespace(script_dir=root, comfy_dir=root / "ComfyUI", output_dir=root / "output")
        self.clips = {}
        for speaker in (0, 1):
            clip = root / f"voice{speaker}.wav"
            clip.write_bytes(audio.encode_wav(tone(1.0), RATE))
            self.clips[speaker] = str(clip)

    def run_route(self, request, speech, server_class=server.SglangOmniServer):
        fake = MagicMock()
        fake.speech.side_effect = speech
        fake.payload.side_effect = server_class(None).payload
        with patch.dict("os.environ", {"H3_DRAMA3_SGLANG_DATA": str(Path(self.temp.name) / "data")}), \
                patch.object(generate, "shared_server", return_value=fake), \
                patch.object(generate.environment, "prepare"), patch.object(generate, "ensure_model"), \
                patch.object(generate, "write_snapshot") as snapshot:
            updates = list(generate.generate_drama3_served(request, runtime=self.runtime, free_gpu=lambda: None))
        return fake, updates, snapshot

    def request(self, text, backend="SGLang-Omni", **overrides):
        voices = tuple((speaker, path, f"<|speaker:{speaker}|> Reference {speaker}.") for speaker, path in self.clips.items())
        return Drama3Request(text, voices=voices, backend=backend, parallel=4, retries=2,
                             voice_names=((0, "Alice"), (1, "")), **overrides)
    def test_each_chunk_uses_its_speakers_voice_and_the_server_stays_loaded(self):
        script = "<|speaker:0|> Hello there, friend.\n<|speaker:1|> [laughs] Hi Alice.\n<|speaker:0|> Ready to go?"
        fake, updates, snapshot = self.run_route(self.request(script),
                                                 lambda payload: float_wav(tone(expected_seconds(payload["input"]))))
        payloads = [call.args[0] for call in fake.speech.call_args_list]
        self.assertEqual(sorted(p["input"] for p in payloads), sorted(["Hello there, friend.", "[laughs] Hi Alice.", "Ready to go?"]))
        for payload in payloads:
            speaker = 1 if "Alice." in payload["input"] else 0
            self.assertTrue(Path(payload["references"][0]["audio_path"]).is_file())
            self.assertEqual(payload["references"][0]["text"], f"Reference {speaker}.")
            self.assertEqual({k: payload[k] for k in ("temperature", "top_p", "top_k")},
                             {"temperature": 1.0, "top_p": 0.9, "top_k": 1 << 30})
            self.assertLessEqual(payload["max_new_tokens"], 8192)
        fake.ensure_ready.assert_called_once()
        fake.release.assert_not_called()  # kept loaded for the next Drama 3 job
        self.assertIn("Alice: 2, Speaker 1: 1", updates[0].status)
        result = Path(updates[-1].output["value"])
        self.assertEqual(updates[-1].output["playback_position"], 0)
        self.assertTrue(result.name.startswith("Drama3_sglang_") and result.is_file())
        chunks = snapshot.call_args.args[1]["settings"]["chunks"]
        self.assertEqual([c["speaker"] for c in chunks], [0, 1, 0])
        self.assertEqual(chunks[0]["name"], "Alice")
    def test_sglang_route_sends_references_inline(self):
        def speech(payload):
            if "seed" in payload:
                raise H3Error("seed is not supported by this model")
            return float_wav(tone(2.0))

        fake, updates, snapshot = self.run_route(
            self.request("<|speaker:1|> Hi there.", backend="SGLang-Omni"),
            speech, server_class=server.SglangOmniServer)
        payload = fake.speech.call_args.args[0]
        self.assertEqual(payload["references"][0]["text"], "Reference 1.")
        self.assertEqual(payload["temperature"], 1.0)
        self.assertEqual(snapshot.call_args.args[1]["settings"]["backend"], "SGLang-Omni")
        self.assertIsNone(snapshot.call_args.args[1]["settings"]["seed"])
        self.assertIsNone(snapshot.call_args.args[1]["settings"]["chunks"][0]["seed"])
        self.assertIn("seed control unavailable", updates[-1].status)
    def test_silent_takes_are_retried_without_seed_control(self):
        attempts = []

        def speech(payload):
            self.assertNotIn("seed", payload)
            attempts.append(payload)
            return float_wav(np.zeros(RATE * 5, np.float32) if len(attempts) == 1 else tone(2.0))

        fake, updates, snapshot = self.run_route(self.request("<|speaker:0|> One two three four five."), speech)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(snapshot.call_args.args[1]["settings"]["chunks"][0]["attempts"], 2)
    def test_request_errors_surface(self):
        def speech(payload):
            raise H3Error("boom")

        with self.assertRaisesRegex(H3Error, "boom"):
            self.run_route(self.request("<|speaker:0|> Hello."), speech)
    def test_duration_advisory_keeps_the_first_take_even_when_retries_enabled(self):
        fake, updates, snapshot = self.run_route(
            self.request("<|speaker:0|> One two three four five."),
            lambda payload: float_wav(tone(12.0)))
        self.assertEqual(fake.speech.call_count, 1)
        self.assertIn("Audio advisories (takes kept)", updates[-1].status)
        self.assertTrue(snapshot.call_args.args[1]["settings"]["chunks"][0]["warnings"])
    def test_default_retries_are_disabled(self):
        request = Drama3Request("Hello.", voices=((0, self.clips[0], "Reference 0."),), backend="SGLang-Omni")
        self.assertEqual(request.retries, 0)
        with patch.object(audio, "check_take", side_effect=AssertionError("Zero retries must skip take checks")):
            fake, updates, snapshot = self.run_route(request, lambda payload: float_wav(tone(1.0)),
                                                     server_class=server.SglangOmniServer)
        self.assertEqual(fake.speech.call_count, 1)
        self.assertEqual(snapshot.call_args.args[1]["settings"]["retries"], 0)
        chunk = snapshot.call_args.args[1]["settings"]["chunks"][0]
        self.assertEqual(chunk["check"], "skipped")
        self.assertEqual(chunk["warnings"], [])
        self.assertAlmostEqual(chunk["seconds"], 1.0)
        self.assertNotIn("failed a check", updates[-1].status)
        self.assertNotIn("Audio advisories", updates[-1].status)
    def test_every_speaker_needs_a_reference_voice(self):
        for backend in ("SGLang-Omni",):
            request = self.request("<|speaker:0|> Hi.\n<|speaker:3|> Who am I?", backend=backend)
            with self.subTest(backend=backend), self.assertRaisesRegex(H3Error, r"<\|speaker:3\|>"):
                validate_request(request)
        validate_request(Drama3Request("<|speaker:0|> Hi.\n<|speaker:3|> Who am I?"))  # ComfyUI invents voices


if __name__ == "__main__":
    unittest.main()
