"""Encoder availability must include hardware initialization, not just a name."""

import ast
from fractions import Fraction
import json
import logging
import math
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from h3_app.media_tools import has_encoder

try:
    import av
except ImportError:
    av = None


def load_saver(av_module):
    path = Path(__file__).resolve().parents[1] / "custom_nodes/H3Acceleration/__init__.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "H3SaveVideoNVENC")
    namespace = dict(av=av_module, Fraction=Fraction, json=json, logging=logging,
                     math=math, os=os, torch=SimpleNamespace(uint8=np.uint8))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["H3SaveVideoNVENC"], namespace


class EncoderSelectionTests(unittest.TestCase):
    def test_working_nvenc_is_preferred(self):
        context = Mock()
        module = SimpleNamespace(CodecContext=SimpleNamespace(create=Mock(return_value=context)))
        node, _ = load_saver(module)
        encoder, options = node._select_encoder(1280, 720, Fraction(24), "p4", 23)
        self.assertEqual(encoder, "h264_nvenc")
        self.assertEqual(options, {"preset": "p4", "tune": "hq", "rc": "vbr", "cq": "23"})
        module.CodecContext.create.assert_called_once_with("h264_nvenc", "w")
        context.open.assert_called_once()
        self.assertEqual((context.width, context.height, context.pix_fmt), (1280, 720, "yuv420p"))
        self.assertEqual(context.time_base, Fraction(1, 24))

    def test_registered_but_unusable_nvenc_uses_software(self):
        hardware, software = Mock(), Mock()
        hardware.open.side_effect = RuntimeError("No NVENC capable devices found")
        module = SimpleNamespace(CodecContext=SimpleNamespace(create=Mock(side_effect=[hardware, software])))
        node, _ = load_saver(module)
        with self.assertLogs(level="WARNING") as logs:
            encoder, options = node._select_encoder(1280, 720, Fraction(24), "p7", 19)
        self.assertEqual(encoder, "libx264")
        self.assertEqual(options, {"preset": "medium", "crf": "19"})
        software.open.assert_called_once()
        self.assertIn("using CPU libx264", logs.output[0])

    def test_missing_both_encoders_reports_both_errors(self):
        module = SimpleNamespace(CodecContext=SimpleNamespace(create=Mock(side_effect=ValueError("codec missing"))))
        node, _ = load_saver(module)
        with self.assertLogs(level="WARNING"), self.assertRaisesRegex(RuntimeError, "NVENC:.*libx264:"):
            node._select_encoder(1280, 720, Fraction(24), "p4", 23)

    def test_ffmpeg_probe_requires_successful_encode(self):
        for returncode in (0, 1):
            with self.subTest(returncode=returncode), patch(
                "h3_app.media_tools.run_media_process",
                return_value=subprocess.CompletedProcess([], returncode, "h264_nvenc", "unsupported device"),
            ) as run:
                self.assertEqual(has_encoder("h264_nvenc"), returncode == 0)
                command = run.call_args.args[0]
                self.assertIn("lavfi", command)
                self.assertEqual(command[command.index("-c:v") + 1], "h264_nvenc")
                self.assertEqual(run.call_args.kwargs["timeout"], 15)

    def test_encoder_probe_timeout_falls_back_but_cancellation_propagates(self):
        with patch("h3_app.media_tools.run_media_process", side_effect=subprocess.TimeoutExpired("ffmpeg", 15)):
            self.assertFalse(has_encoder("h264_nvenc"))
        with patch("h3_app.media_tools.run_media_process", side_effect=RuntimeError("cancelled")):
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                has_encoder("h264_nvenc")


class CpuTensor:
    """Small NumPy-backed adapter for the node's tensor-to-PyAV boundary."""

    def __init__(self, array):
        self.array = np.asarray(array)

    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self):
        return self.array.ndim

    def __getitem__(self, key):
        return CpuTensor(self.array[key])

    def mul(self, value):
        return CpuTensor(self.array * value)

    def clamp(self, low, high):
        return CpuTensor(self.array.clip(low, high))

    def to(self, *, device, dtype):
        return CpuTensor(self.array.astype(dtype))

    def float(self):
        return CpuTensor(self.array.astype(np.float32))

    def cpu(self):
        return self

    def contiguous(self):
        return CpuTensor(np.ascontiguousarray(self.array))

    def numpy(self):
        return self.array


@unittest.skipIf(av is None, "PyAV is required for real software encoding tests")
class SoftwareVideoSaveTests(unittest.TestCase):
    def test_fallback_mp4_keeps_frames_fps_audio_and_metadata(self):
        for with_audio in (False, True):
            with self.subTest(audio=with_audio), TemporaryDirectory() as directory:
                node, namespace = load_saver(av)
                namespace["folder_paths"] = SimpleNamespace(
                    get_output_directory=lambda: directory,
                    get_save_image_path=lambda *args: (directory, "video", 1, "", None),
                )
                original_probe = node._probe_encoder

                def unavailable_hardware(encoder, *args):
                    if encoder == "h264_nvenc":
                        raise RuntimeError("H200 has no NVENC")
                    return original_probe(encoder, *args)

                frames = np.zeros((16, 48, 64, 3), dtype=np.float32)
                frames[..., 0] = np.linspace(0.1, 0.9, 16)[:, None, None]
                waveform = np.sin(np.arange(48000) * (2 * np.pi * 440 / 48000)).astype(np.float32)
                audio = {"sample_rate": 48000, "waveform": CpuTensor(waveform[None, None])} if with_audio else None
                components = SimpleNamespace(images=CpuTensor(frames), audio=audio, frame_rate=23.976)
                video = SimpleNamespace(get_components=lambda: components)
                prompt = {"description": "original prompt"}
                workflow = {"nodes": ["original workflow"]}
                with patch.object(node, "_probe_encoder", side_effect=unavailable_hardware), self.assertLogs(level="WARNING"):
                    result = node().save(video, "h3/test", "p4", 23, prompt=prompt,
                                         extra_pnginfo={"workflow": workflow})
                info = result["ui"]["videos"][0]
                path = Path(directory) / info["filename"]
                with av.open(str(path)) as output:
                    stream = output.streams.video[0]
                    self.assertEqual(stream.codec_context.name, "h264")
                    self.assertEqual((stream.width, stream.height), (64, 48))
                    self.assertEqual(stream.average_rate, Fraction(2997, 125))
                    decoded = list(output.decode(video=0))
                    self.assertEqual(len(decoded), 16)
                    self.assertLess(decoded[0].to_ndarray(format="rgb24")[..., 0].mean(),
                                    decoded[-1].to_ndarray(format="rgb24")[..., 0].mean())
                    self.assertEqual(json.loads(output.metadata["prompt"]), prompt)
                    self.assertEqual(json.loads(output.metadata["workflow"]), workflow)
                    self.assertEqual(len(output.streams.audio), int(with_audio))
                    if with_audio:
                        self.assertEqual(output.streams.audio[0].codec_context.name, "aac")
                        duration = float(output.streams.audio[0].duration * output.streams.audio[0].time_base)
                        self.assertAlmostEqual(duration, 16 / 23.976, delta=0.025)
                np.testing.assert_array_equal(components.images.numpy(), frames)
                if with_audio:
                    with av.open(str(path)) as output:
                        self.assertGreater(sum(frame.samples for frame in output.decode(audio=0)), 0)

    def test_unrelated_output_error_is_not_hidden_by_fallback(self):
        node, namespace = load_saver(av)
        with TemporaryDirectory() as directory:
            namespace["folder_paths"] = SimpleNamespace(
                get_output_directory=lambda: directory,
                get_save_image_path=lambda *args: (directory, "video", 1, "", None),
            )
            video = SimpleNamespace(get_components=lambda: SimpleNamespace(
                images=CpuTensor(np.zeros((1, 48, 64, 3))), frame_rate=24, audio=None,
            ))
            with (
                patch.object(node, "_select_encoder", return_value=("libx264", {"crf": "23"})) as select,
                patch.object(av, "open", side_effect=PermissionError("disk read-only")),
                self.assertRaisesRegex(PermissionError, "disk read-only"),
            ):
                node().save(video, "h3/test", "p4", 23)
            select.assert_called_once()
            self.assertEqual(list(Path(directory).glob("*.mp4")), [])


if __name__ == "__main__":
    unittest.main()
