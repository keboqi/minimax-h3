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
from h3_app.drama3_serving.generate import frame_budget
from h3_app.drama3_serving.planner import CHUNK_BYTES, expected_seconds, parse_turns, plan_chunks, spoken_text
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
    def test_cjk_turns_split_without_spaces_and_stay_under_the_server_budget(self):
        paragraph = ("今天的 AI 基础设施热潮，得益于当今顶尖科技公司空前惊人的利润。正如塞姆巴莱斯特所解释的，"
                     "这些公司的盈利能力如此之强。“它们正在产生前所未有的自由现金流，”塞姆巴莱斯特告诉我。"
                     "“它们赚得盆满钵满，这就是为什么它们有能力每年投入数千亿美元的资本支出。”") * 3
        chunks = plan_chunks("<|speaker:0|> " + paragraph)
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(c.text for c in chunks), paragraph.replace("。 ", "。"))
        for chunk in chunks:
            self.assertFalse(chunk.text.startswith("”"))
            self.assertLessEqual(len(chunk.text.encode()), CHUNK_BYTES * 2)
            self.assertLessEqual(frame_budget(chunk.text, 8192), 2048)
        self.assertEqual(frame_budget("字" * 2000, 8192), 2048)
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
        with patch.object(environment.os, "name", "nt"), self.assertRaisesRegex(H3Error, "requires Linux"):
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

    def test_speech_requests_share_one_pooled_session(self):
        instance = server.SglangOmniServer(self.settings_for(max_seqs=12))
        session = instance.http()
        self.assertIs(instance.http(), session)
        self.assertEqual(session.get_adapter("http://127.0.0.1:8096").poolmanager.connection_pool_kw["maxsize"], 12)
        with patch.object(session, "post", return_value=MagicMock(ok=True, content=b"wav")) as post:
            self.assertEqual(instance.speech({}), b"wav")
            self.assertEqual(instance.speech({}), b"wav")
        self.assertEqual(post.call_count, 2)
        instance.stop()
        self.assertIsNot(instance.http(), session)  # a stopped server drops its pooled connections

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

    def request(self, text, **overrides):
        voices = tuple((speaker, path, f"<|speaker:{speaker}|> Reference {speaker}.") for speaker, path in self.clips.items())
        return Drama3Request(text, voices=voices, parallel=4, retries=2,
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
            self.assertLessEqual(payload["max_new_tokens"], 2048)
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
            self.request("<|speaker:1|> Hi there."),
            speech, server_class=server.SglangOmniServer)
        payload = fake.speech.call_args.args[0]
        self.assertEqual(payload["references"][0]["text"], "Reference 1.")
        self.assertEqual(payload["temperature"], 1.0)
        self.assertEqual(snapshot.call_args.args[1]["settings"]["backend"], "SGLang-Omni")
        self.assertNotIn("seed", snapshot.call_args.args[1]["settings"])
        self.assertNotIn("seed", snapshot.call_args.args[1]["settings"]["chunks"][0])
        self.assertNotIn("seed", updates[-1].status)
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
    def test_a_failed_chunk_returns_without_waiting_for_running_requests(self):
        import threading
        import time

        release = threading.Event()
        self.addCleanup(release.set)

        def speech(payload):
            if "fail" in payload["input"]:
                raise H3Error("boom")
            release.wait(30)  # an in-flight request the job must not wait for
            return float_wav(tone(1.0))

        started = time.monotonic()
        with self.assertRaisesRegex(H3Error, "boom"):
            self.run_route(self.request("<|speaker:0|> Slow line here.\n<|speaker:1|> This will fail."), speech)
        self.assertLess(time.monotonic() - started, 5)

    def test_dropped_connections_and_server_errors_are_retried(self):
        import requests

        failures = [requests.ConnectionError("reset"), server.SpeechRequestError("busy", 503)]

        def speech(payload):
            if failures:
                raise failures.pop(0)
            return float_wav(tone(1.0))

        request = Drama3Request("Hello.", voices=((0, self.clips[0], "Reference 0."),))
        with patch.object(generate, "TRANSPORT_BACKOFF_SECONDS", 0):
            fake, updates, _snapshot = self.run_route(request, speech)
        self.assertEqual(fake.speech.call_count, 3)
        self.assertIsNotNone(updates[-1].output)

    def test_client_errors_are_not_retried(self):
        def speech(payload):
            raise server.SpeechRequestError("bad request", 400)

        request = Drama3Request("Hello.", voices=((0, self.clips[0], "Reference 0."),))
        with patch.object(generate, "TRANSPORT_BACKOFF_SECONDS", 0), \
                self.assertRaisesRegex(H3Error, "bad request"):
            fake, _updates, _snapshot = self.run_route(request, speech)

    def test_failed_sound_generation_stops_before_any_speech(self):
        req = self.request('<|speaker:0|> Hello.\n<sound name="rain">Steady rain, 4 seconds</sound>',
                           script_mode="Advanced (sound enhanced)")
        fake = MagicMock()
        with patch("h3_app.stable_audio.generate_sound_references", side_effect=H3Error("no stable audio")), \
                patch.dict("os.environ", {"H3_DRAMA3_SGLANG_DATA": str(Path(self.temp.name) / "data")}), \
                patch.object(generate, "shared_server", return_value=fake), \
                self.assertRaisesRegex(H3Error, "no stable audio"):
            list(generate.generate_drama3_served(req, runtime=self.runtime, free_gpu=lambda: None))
        fake.ensure_ready.assert_not_called()
        fake.speech.assert_not_called()

    def test_unchanged_references_are_hashed_once(self):
        import os

        media = Path(self.temp.name) / "media"
        clip = Path(self.clips[0])
        with patch.object(generate.hashlib, "sha256", wraps=generate.hashlib.sha256) as sha:
            first = generate._staged_reference(str(clip), media)
            self.assertEqual(generate._staged_reference(str(clip), media), first)
            self.assertEqual(sha.call_count, 1)
            clip.write_bytes(audio.encode_wav(tone(2.0), RATE))
            os.utime(clip, ns=(1, 1))  # a changed clip is hashed again
            self.assertNotEqual(generate._staged_reference(str(clip), media), first)
            self.assertEqual(sha.call_count, 2)

    def test_a_repeated_sound_is_decoded_once(self):
        rain = Path(self.temp.name) / "rain.wav"
        rain.write_bytes(audio.encode_wav(tone(0.5), RATE))
        script = ('<sound name="rain">Rain, 1 second</sound>\n<|speaker:0|> Hello.\n'
                  '<sound name="rain">Rain, 1 second</sound>\n<|speaker:0|> Again.\n'
                  '<sound name="rain">Rain, 1 second</sound>')
        req = self.request(script, script_mode="Advanced (sound enhanced)",
                           sounds=(("rain", str(rain), "Rain, 1 second", 1),))
        with patch.object(audio, "load_audio_file", wraps=audio.load_audio_file) as load:
            fake, updates, _snapshot = self.run_route(req, lambda payload: float_wav(tone(0.6)))
        self.assertEqual(load.call_count, 1)
        self.assertTrue(Path(updates[-1].output["value"]).is_file())

    def test_duration_advisory_keeps_the_first_take_even_when_retries_enabled(self):
        fake, updates, snapshot = self.run_route(
            self.request("<|speaker:0|> One two three four five."),
            lambda payload: float_wav(tone(12.0)))
        self.assertEqual(fake.speech.call_count, 1)
        self.assertIn("Audio advisories (takes kept)", updates[-1].status)
        self.assertTrue(snapshot.call_args.args[1]["settings"]["chunks"][0]["warnings"])
    def test_default_retries_are_disabled(self):
        request = Drama3Request("Hello.", voices=((0, self.clips[0], "Reference 0."),))
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
        request = self.request("<|speaker:0|> Hi.\n<|speaker:3|> Who am I?")
        with self.assertRaisesRegex(H3Error, r"<\|speaker:3\|>"):
            validate_request(request)

    def test_served_generation_renders_nested_sound_tree_pauses_and_speaker_effects(self):
        sfx_clip = Path(self.temp.name) / "thunder.wav"
        sfx_clip.write_bytes(audio.encode_wav(tone(0.5), RATE))
        auto_bgm = Path(self.temp.name) / "rain_bgm.wav"
        auto_bgm.write_bytes(audio.encode_wav(tone(1.0), RATE))

        script = (
            "<parallel>\n"
            '<sound loop="true" volume="0.2" name="rain_bgm">Steady rain, 4 seconds</sound>\n'
            "<sequential>\n"
            "<|speaker:0|> <phone_call> Hello? <#0.5#> Is anyone there?\n"
            "<parallel>\n"
            '<sound name="thunder">Loud thunderclap, 2 seconds</sound>\n'
            "<|speaker:1|> <robotic_voice> System online.\n"
            "</parallel>\n"
            "</sequential>\n"
            "</parallel>"
        )
        req = self.request(
            script,
            script_mode="Advanced (sound enhanced)",
            sounds=(("thunder", str(sfx_clip), "Loud thunderclap, 2 seconds", 2),),
        )
        with patch("h3_app.stable_audio.generate_sound_references", return_value={"rain_bgm": auto_bgm}) as gen_sounds:
            fake, updates, snapshot = self.run_route(req, lambda payload: float_wav(tone(0.6)))
        gen_sounds.assert_called_once()
        # 3 speech chunks: "Hello?", "Is anyone there?" (split across <#0.5#>), and "System online."
        self.assertEqual(fake.speech.call_count, 3)
        result_path = Path(updates[-1].output["value"])
        self.assertTrue(result_path.is_file())
        mixed = audio.load_audio_file(result_path, RATE)
        self.assertGreater(len(mixed) / RATE, 1.5)


class TreeAndSpeakerEffectsTests(unittest.TestCase):
    def test_nested_parallel_sequential_tree_and_pause_splitting(self):
        from h3_app.drama3_serving.planner import plan_turns_chunks
        from h3_app.drama3_serving.tree import (
            ParallelNode,
            PauseNode,
            SequentialNode,
            SoundNode,
            SpeechNode,
            extract_script_sounds,
            iter_speech_nodes,
            parse_drama3_tree,
        )

        script = (
            "<parallel>\n"
            '<sound loop="true" volume="0.2" name="tavern_bgm">Acoustic lute and tavern murmur, 15 seconds</sound>\n'
            "<sequential>\n"
            "<|speaker:0|> Welcome, traveler.\n"
            "<parallel>\n"
            '<sound name="mug_slam">Heavy wooden mug slamming on table, 2 seconds</sound>\n'
            "<|speaker:1|> <phone_call> Can you hear me? <#1.25#> <robotic_voice> Over and out.\n"
            "</parallel>\n"
            "</sequential>\n"
            "</parallel>"
        )
        root = parse_drama3_tree(script)
        self.assertEqual(len(root.children), 1)
        outer = root.children[0]
        self.assertIsInstance(outer, ParallelNode)
        self.assertIsInstance(outer.children[0], SoundNode)
        self.assertTrue(outer.children[0].loop)
        self.assertAlmostEqual(outer.children[0].volume, 0.2)
        self.assertEqual(outer.children[0].duration, 15)

        seq = outer.children[1]
        self.assertIsInstance(seq, SequentialNode)
        self.assertIsInstance(seq.children[0], SpeechNode)
        inner_par = seq.children[1]
        self.assertIsInstance(inner_par, ParallelNode)
        self.assertIsInstance(inner_par.children[0], SoundNode)
        inner_seq = inner_par.children[1]
        self.assertIsInstance(inner_seq, SequentialNode)
        self.assertIsInstance(inner_seq.children[0], SpeechNode)
        self.assertIsInstance(inner_seq.children[1], PauseNode)
        self.assertAlmostEqual(inner_seq.children[1].seconds, 1.25)
        self.assertIsInstance(inner_seq.children[2], SpeechNode)

        speech_nodes = iter_speech_nodes(root)
        self.assertEqual(len(speech_nodes), 3)
        self.assertEqual(speech_nodes[1].turns[0].speaker_effects, ("phone_call",))
        self.assertEqual(speech_nodes[2].turns[0].speaker_effects, ("phone_call", "robotic_voice"))

        # Planner does not merge turns with different speaker_effects
        chunks = plan_turns_chunks([t for node in speech_nodes for t in node.turns])
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[1].speaker_effects, ("phone_call",))
        self.assertEqual(chunks[2].speaker_effects, ("phone_call", "robotic_voice"))

        sounds = extract_script_sounds(script)
        self.assertEqual([s["name"] for s in sounds], ["tavern_bgm", "mug_slam"])
        self.assertEqual([s["duration"] for s in sounds], [15, 2])

    def test_all_twenty_speaker_effects_transform_audio_cleanly(self):
        from h3_app.drama3_serving.speaker_effects import (
            SPEAKER_EFFECTS,
            apply_speaker_effects,
            extract_speaker_effects,
        )

        self.assertEqual(len(SPEAKER_EFFECTS), 20)
        sample = tone(0.4)
        for effect in SPEAKER_EFFECTS:
            with self.subTest(effect=effect):
                cleaned, extracted = extract_speaker_effects(f"<{effect}> [calm] Testing {effect}.")
                self.assertEqual(extracted, (effect,))
                self.assertEqual(cleaned, f"[calm] Testing {effect}.")
                processed = apply_speaker_effects(sample, RATE, (effect,))
                self.assertEqual(processed.dtype, np.float32)
                self.assertGreater(len(processed), 0)
                self.assertTrue(np.all(np.isfinite(processed)))
                self.assertGreater(float(np.max(np.abs(processed))), 0.01)

    def test_fallback_pitch_shift_keeps_length_and_shifts_cleanly_to_the_end(self):
        from h3_app.drama3_serving.speaker_effects import _pitch_shift_preserve_len

        def spectrum(samples):
            magnitude = np.abs(np.fft.rfft(samples * np.hanning(samples.size))) ** 2
            return np.fft.rfftfreq(samples.size, 1 / RATE), magnitude

        for semitones in (6.0, -8.0):
            with self.subTest(semitones=semitones):
                source = tone(1.0)
                shifted = _pitch_shift_preserve_len(source, semitones)
                self.assertEqual(shifted.size, source.size)
                expected = 220 * 2 ** (semitones / 12)
                # A wrapped read position jumps the waveform; a clean shift never steps faster than the tone.
                max_slope = 0.3 * 2 * np.pi * expected / RATE
                self.assertLess(float(np.max(np.abs(np.diff(shifted[RATE // 10:-RATE // 10])))), 1.2 * max_slope)
                for segment in (shifted[RATE // 4:3 * RATE // 4], shifted[-4096:]):  # body and tail
                    freqs, power = spectrum(segment)
                    self.assertAlmostEqual(float(freqs[np.argmax(power)]), expected, delta=12)
                    # A wrapped read position spreads energy into buzz far from the shifted tone.
                    near = power[np.abs(freqs - expected) < 30].sum()
                    self.assertGreater(near / power.sum(), 0.9)

    def test_pedalboard_output_is_peak_limited(self):
        from h3_app.drama3_serving import speaker_effects

        with patch.object(speaker_effects, "_try_apply_pedalboard", lambda samples, rate, effects: samples * 6):
            processed = speaker_effects.apply_speaker_effects(tone(0.2), RATE, ("radio_broadcast",))
        self.assertAlmostEqual(float(np.max(np.abs(processed))), 0.95, places=3)

    def test_mix_parallel_tracks_loops_bgm_ducks_under_speech_and_trims_sfx(self):
        speech_track = tone(2.0)
        bgm_track = tone(0.6)  # shorter than speech, with loop=True loops to speech + 1.0s fade-out
        mixed = audio.mix_parallel_tracks(
            [(speech_track, False, False, 1.0), (bgm_track, True, True, 0.2)],
            RATE,
        )
        self.assertEqual(len(mixed), len(speech_track) + RATE)
        self.assertLessEqual(float(np.max(np.abs(mixed))), 0.901)

    def test_sound_tracks_duck_while_speech_is_audible(self):
        speech = np.concatenate([np.zeros(RATE, dtype=np.float32), tone(1.0), np.zeros(RATE, dtype=np.float32)])
        gain = audio.speech_duck_gain(speech, RATE)

        def level(start, end):
            return float(np.mean(gain[int(start * RATE):int(end * RATE)]))

        self.assertAlmostEqual(level(0.0, 0.9), 1.0, places=3)  # before speech
        self.assertAlmostEqual(level(1.3, 1.9), 10 ** (-audio.DUCK_DEPTH_DB / 20), places=2)  # under speech
        self.assertGreater(level(2.0, 2.1), level(1.3, 1.9))  # recovering after speech
        self.assertAlmostEqual(level(2.8, 3.0), 1.0, places=1)  # released

        noise = np.random.default_rng(0).uniform(-0.1, 0.1, 3 * RATE).astype(np.float32)

        def under_speech_ratio(loop):
            mixed = audio.mix_parallel_tracks(
                [{"audio": speech, "is_sound": False}, {"audio": noise, "is_sound": True, "loop": loop}], RATE)
            bed = mixed[:3 * RATE] - speech
            quiet = float(np.sqrt(np.mean(bed[:int(0.9 * RATE)] ** 2)))
            return float(np.sqrt(np.mean(bed[int(1.3 * RATE):int(1.9 * RATE)] ** 2))) / quiet

        self.assertLess(under_speech_ratio(loop=True), 1 / 3)  # a looped bed ducks
        self.assertAlmostEqual(under_speech_ratio(loop=False), 1.0, delta=0.1)  # a one-shot SFX stays in front

    def test_loops_crossfade_their_seams(self):
        clip = np.linspace(0.0, 0.5, RATE, dtype=np.float32)  # ends far from where it starts
        looped = audio.loop_to_length(clip, 5 * RATE + 123, RATE)
        self.assertEqual(looped.size, 5 * RATE + 123)
        self.assertGreater(float(np.max(np.abs(np.diff(np.tile(clip, 2))))), 0.4)  # a hard cut jumps
        self.assertLess(float(np.max(np.abs(np.diff(looped)))), 0.01)
        np.testing.assert_allclose(looped[:RATE // 2], clip[:RATE // 2])  # first pass starts untouched


if __name__ == "__main__":
    unittest.main()

