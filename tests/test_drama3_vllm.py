"""Optional vLLM-Omni Drama 3 route: environment, server lifecycle, chunking, checks and joining."""

from __future__ import annotations

import io
import struct
import tempfile
import unittest
import wave
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from h3_app.drama3_vllm import audio, environment, generate, server
from h3_app.drama3_vllm.planner import expected_seconds, parse_turns, plan_chunks, spoken_text
from h3_app.errors import H3Error
from h3_app.generation.optional_engines import validate_request
from h3_app.generation.requests import Drama3Request
from h3_requirements import drama3_vllm_install_commands

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
    def test_install_commands_pin_versions_and_keep_posix_paths(self):
        commands = drama3_vllm_install_commands(PurePosixPath("/opt/h3/drama3-vllm-venv"))
        self.assertEqual(commands[0][:2], ["uv", "venv"])
        self.assertIn("vllm==0.31.0", commands[1])
        self.assertIn("--torch-backend=cu130", commands[1])
        self.assertTrue(commands[2][-1].startswith("vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@"))
        self.assertEqual(commands[3][0], "/opt/h3/drama3-vllm-venv/bin/python")

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
        self.assertEqual(env["VLLM_OMNI_TARGET_DEVICE"], "cuda")

    def test_prepare_refuses_non_linux(self):
        with patch.object(environment.os, "name", "nt"), self.assertRaisesRegex(H3Error, "Linux only"):
            environment.prepare(Path("venv"))


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.settings = server.ServerSettings(venv=root / "venv", data_dir=root / "data", model_root=root / "models")

    def test_command_enables_sleep_and_local_reference_media(self):
        command = server.Drama3VllmServer(self.settings).command(Path("deploy.json"))
        self.assertIn("--omni", command)
        self.assertIn("--enable-sleep-mode", command)
        self.assertEqual(command[command.index("--allowed-local-media-path") + 1], str(self.settings.media_dir))
        self.assertEqual(command[command.index("--served-model-name") + 1], server.SERVED_MODEL)
        self.assertTrue(str(self.settings.model_dir).endswith("fishaudio--drama-3--e0a9218aa488"))

    def test_settings_come_from_environment(self):
        runtime = SimpleNamespace(script_dir=Path("/app"), comfy_dir=Path("/app/ComfyUI"))
        with patch.dict("os.environ", {"H3_DRAMA3_VLLM_DATA": "/data/d3", "H3_DRAMA3_VLLM_ENGINE_MEMORY": "0.6",
                                       "H3_DRAMA3_VLLM_CUDA_MPS": "true"}):
            settings = server.ServerSettings.from_environment(runtime)
        self.assertEqual(settings.data_dir, Path("/data/d3"))
        self.assertEqual((settings.engine_memory, settings.cuda_mps), (0.6, True))
        self.assertEqual(settings.model_root, Path("/app/ComfyUI/models/drama3_vllm"))

    def test_wake_frees_the_gpu_first_and_sleep_parks_both_stages(self):
        instance = server.Drama3VllmServer(self.settings)
        calls = []
        instance.running = lambda: True
        instance.healthy = lambda timeout=2.0: True
        instance.sleeping = True
        instance._post = lambda path, payload, timeout=120: calls.append((path, payload)) or {"status": "SUCCESS"}
        instance.ensure_awake(lambda: calls.append("free"))
        self.assertEqual(calls, ["free", ("/v1/omni/wakeup", {"stage_ids": [0, 1]})])
        self.assertFalse(instance.sleeping)
        instance.sleep()
        self.assertEqual(calls[-1], ("/v1/omni/sleep", {"stage_ids": [0, 1], "level": 1}))
        self.assertTrue(instance.sleeping)

    def test_cold_start_frees_gpu_then_waits_for_health(self):
        instance = server.Drama3VllmServer(self.settings)
        order = []
        instance._start = lambda: order.append("start")
        instance._wait_healthy = lambda progress: order.append("ready")
        instance.ensure_awake(lambda: order.append("free"))
        self.assertEqual(order, ["free", "start", "ready"])

    def test_failed_sleep_stops_the_server_so_the_gpu_is_released(self):
        instance = server.Drama3VllmServer(self.settings)
        instance.running = lambda: True
        instance.stop = MagicMock()
        instance._post = MagicMock(return_value=[{"status": "ERROR"}])
        instance.sleep()
        instance.stop.assert_called_once()


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

    def run_route(self, request, speech):
        fake = MagicMock()
        fake.speech.side_effect = speech
        with patch.dict("os.environ", {"H3_DRAMA3_VLLM_DATA": str(Path(self.temp.name) / "data")}), \
                patch.object(generate, "shared_server", return_value=fake), \
                patch.object(generate.environment, "prepare"), patch.object(generate, "ensure_model"), \
                patch.object(generate, "write_snapshot") as snapshot:
            updates = list(generate.generate_drama3_vllm(request, runtime=self.runtime, free_gpu=lambda: None,
                                                          seed=42))
        return fake, updates, snapshot

    def request(self, text, **overrides):
        voices = tuple((speaker, path, f"<|speaker:{speaker}|> Reference {speaker}.") for speaker, path in self.clips.items())
        return Drama3Request(text, voices=voices, backend="vLLM-Omni", parallel=4, retries=2,
                             voice_names=((0, "Alice"), (1, "")), **overrides)

    def test_each_chunk_uses_its_speakers_voice_and_the_result_is_joined_in_order(self):
        script = "<|speaker:0|> Hello there, friend.\n<|speaker:1|> [laughs] Hi Alice.\n<|speaker:0|> Ready to go?"
        fake, updates, snapshot = self.run_route(self.request(script),
                                                 lambda payload: float_wav(tone(expected_seconds(payload["input"]))))
        payloads = [call.args[0] for call in fake.speech.call_args_list]
        self.assertEqual(sorted(p["input"] for p in payloads), sorted(["Hello there, friend.", "[laughs] Hi Alice.", "Ready to go?"]))
        for payload in payloads:
            speaker = 1 if "Alice." in payload["input"] else 0
            self.assertTrue(payload["ref_audio"].startswith("file:"))
            self.assertEqual(payload["ref_text"], f"Reference {speaker}.")
            self.assertEqual(payload["extra_params"], {"temperature": 1.0, "top_p": 0.9, "top_k": 30})
            self.assertLessEqual(payload["max_new_tokens"], 8192)
        fake.ensure_awake.assert_called_once()
        fake.sleep.assert_called_once()
        self.assertIn("Alice: 2, Speaker 1: 1", updates[0].status)
        result = Path(updates[-1].output["value"])
        self.assertEqual(updates[-1].output["playback_position"], 0)
        self.assertTrue(result.name.startswith("Drama3_vllm_") and result.is_file())
        chunks = snapshot.call_args.args[1]["settings"]["chunks"]
        self.assertEqual([c["speaker"] for c in chunks], [0, 1, 0])
        self.assertEqual(chunks[0]["name"], "Alice")

    def test_silent_takes_are_retried_with_new_seeds(self):
        seeds = []

        def speech(payload):
            seeds.append(payload["seed"])
            return float_wav(np.zeros(RATE * 5, np.float32) if len(seeds) == 1 else tone(2.0))

        fake, updates, snapshot = self.run_route(self.request("<|speaker:0|> One two three four five."), speech)
        self.assertEqual(len(seeds), 2)
        self.assertNotEqual(seeds[0], seeds[1])
        self.assertEqual(snapshot.call_args.args[1]["settings"]["chunks"][0]["attempts"], 2)

    def test_server_sleeps_even_when_a_request_fails(self):
        fake = None

        def speech(payload):
            raise H3Error("boom")

        with self.assertRaisesRegex(H3Error, "boom"):
            fake, _updates, _snapshot = self.run_route(self.request("<|speaker:0|> Hello."), speech)

    def test_every_speaker_needs_a_reference_voice(self):
        request = self.request("<|speaker:0|> Hi.\n<|speaker:3|> Who am I?")
        with self.assertRaisesRegex(H3Error, r"<\|speaker:3\|>"):
            validate_request(request)
        validate_request(Drama3Request("<|speaker:0|> Hi.\n<|speaker:3|> Who am I?"))  # ComfyUI invents voices


if __name__ == "__main__":
    unittest.main()
