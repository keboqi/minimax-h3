"""CPU acceptance for NVIDIA streaming, immutable requests and failure behavior."""

from contextlib import ExitStack, nullcontext
import inspect
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from h3_app.catalog import DLSS5_NR, NVIDIA_VSR, POSTPROCESS_OPTIONS, GENERATION_POSTPROCESS_OPTIONS
from h3_app.config import RuntimeConfig
from h3_app.errors import H3Error
from h3_app.jobs import JobCancelled
from h3_app import nvidia_postprocess as service
from h3_app.nvidia_pipeline import stream_pipeline
from h3_app.provenance import read_snapshot, write_snapshot
from h3_ui.nvidia_media import with_nvidia_settings, can_enhance
from h3_ui.job_bindings import owned_generation


class NvidiaSettingsTests(unittest.TestCase):
    def test_settings_reject_unknown_nonfinite_and_invalid_values(self):
        for value in ({"strength": float("nan")}, {"intensity": 2}, {"quality": "fake"},
                      {"original_size": "yes"}, {"arbitrary_exe": "bad"}, []):
            with self.subTest(value=value), self.assertRaises(H3Error):
                service.NvidiaSettings.parse(value)

    def test_original_vsr_requires_restoration_and_upscale_cannot_shrink(self):
        with self.assertRaisesRegex(H3Error, "Denoise or Deblur"):
            service.target_dimensions(NVIDIA_VSR, 1280, 720, "1920 × 1920",
                                      service.NvidiaSettings(original_size=True))
        with self.assertRaisesRegex(H3Error, "1× to 4×"):
            service.target_dimensions(NVIDIA_VSR, 3840, 2160, "1920 × 1920", service.NvidiaSettings())
        self.assertEqual(service.target_dimensions(DLSS5_NR, 1280, 720, "1920 × 1920",
            service.NvidiaSettings(original_size=True)), (1280, 720))
        self.assertEqual(service.target_dimensions(NVIDIA_VSR, 1280, 720, "3840 × 3840",
            service.NvidiaSettings()), (3840, 2160))

    def test_gallery_only_methods_and_old_job_defaults(self):
        for method in (NVIDIA_VSR, DLSS5_NR):
            self.assertIn(method, POSTPROCESS_OPTIONS)
            self.assertNotIn(method, GENERATION_POSTPROCESS_OPTIONS)
        callback = Mock(__name__="postprocess")
        capture = with_nvidia_settings(callback)
        values = ["Video", "clip.mp4", "SeedVR2 2x", 42, "model", "ltx", "", False, False, 5, "1920 × 1920"]
        capture(*values, request="owner")
        callback.assert_called_once_with(*values, request="owner")
        wrapped = owned_generation(capture, "gallery")
        self.assertEqual(inspect.signature(wrapped).parameters["dlss_style"].default, "Natural")
        values[2] = DLSS5_NR
        capture(*values, True, "Ultra", 0.7, "High bitrate", "Deblur", 0.2, "Cinematic", 0.4, request="owner")
        settings = callback.call_args.kwargs["nvidia_settings"]
        self.assertEqual(settings["intensity"], 0.4)
        self.assertTrue(settings["original_size"])

    def test_unavailable_methods_cannot_run_and_browsing_never_probes_gpu(self):
        with patch("h3_ui.nvidia_media.availability", return_value=(False, "missing")):
            self.assertFalse(can_enhance("Video", "clip.mp4", DLSS5_NR))
        with patch.object(service, "dlss_config", side_effect=H3Error("missing model")), patch.object(service, "probe_runtime") as probe:
            self.assertEqual(service.availability(DLSS5_NR), (False, "missing model"))
            probe.assert_not_called()
        self.assertFalse(can_enhance("Audio", "sound.wav", NVIDIA_VSR))

    def test_hdr_rotation_and_non_square_pixels_are_rejected(self):
        for metadata in ({"color_transfer": "smpte2084"}, {"color_transfer": "arib-std-b67"},
                         {"side_data_list": [{"rotation": 90}]}, {"sample_aspect_ratio": "4:3"}):
            with self.subTest(metadata=metadata), self.assertRaises(H3Error):
                service.validate_source(metadata)

    def test_wine_model_paths_and_helper_flags(self):
        with patch.object(service, "dlss_config", return_value=(PurePosixPath("/opt/helper/video2dlssnr.exe"), PurePosixPath("/opt/my models"), PurePosixPath("model"), "/usr/bin/wine")), patch.object(service.os, "name", "posix"):
            command, cwd = service.worker_command(DLSS5_NR, 640, 360, 1280, 720, service.NvidiaSettings())
        self.assertEqual(command[:2], ["/usr/bin/wine", "/opt/helper/video2dlssnr.exe"])
        self.assertEqual(command[command.index("--dll-dir") + 1], "Z:\\opt\\my models")
        self.assertEqual(command[command.index("--nr-motion") + 1], "1")


class BinaryPipeTests(unittest.TestCase):
    def command(self, code):
        return [sys.executable, "-u", "-c", code]

    def test_binary_relay_handles_partial_reads_and_large_frames(self):
        frame = bytes(range(256)) * 4096
        output = bytearray()
        command = self.command("import sys; data=sys.stdin.buffer.read(); sys.stdout.buffer.write(data)")
        updates = list(stream_pipeline([command], input_size=len(frame), output_size=len(frame),
            timeout=15, source_bytes=frame * 2, capture=output))
        self.assertEqual(output, frame * 2)
        self.assertEqual(updates[-1].frames, 2)

    def test_truncated_frames_and_nonzero_exits_fail(self):
        for code in ("import sys; sys.stdin.buffer.read(); sys.stdout.buffer.write(b'x')", "raise SystemExit(3)"):
            with self.subTest(code=code), self.assertRaises(H3Error):
                list(stream_pipeline([self.command(code)], input_size=8, output_size=8,
                    timeout=5, source_bytes=b"abcdefgh", capture=bytearray()))

    def test_timeout_and_cancellation_stop_workers(self):
        command = self.command("import time; time.sleep(30)")
        with self.assertRaisesRegex(H3Error, "timed out"):
            list(stream_pipeline([command], input_size=8, output_size=8, timeout=0.3,
                source_bytes=b"abcdefgh", capture=bytearray()))
        with patch("h3_app.nvidia_pipeline.check_cancelled", side_effect=[None, JobCancelled("stop")]):
            with self.assertRaises(JobCancelled):
                list(stream_pipeline([command], input_size=8, output_size=8, timeout=5,
                    source_bytes=b"abcdefgh", capture=bytearray()))

    def test_runtime_probe_rejects_bypass(self):
        command = self.command("import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())")
        with patch.object(service, "availability", return_value=(True, "")), patch.object(service, "worker_command", return_value=(command, None)):
            with self.assertRaisesRegex(H3Error, "unchanged frames"):
                service.probe_runtime(NVIDIA_VSR, Mock(generation_timeout=15))

    def test_runtime_evidence_survives_long_progress_logs(self):
        command = self.command("import sys; print('H3VFX {\"gpu\":\"fixture\"}', file=sys.stderr); "
                               "[print('H3PROG '+str(i), file=sys.stderr) for i in range(1000)]; "
                               "sys.stdout.buffer.write(sys.stdin.buffer.read())")
        updates = list(stream_pipeline([command], input_size=8, output_size=8, timeout=5,
                                       source_bytes=b"abcdefgh", capture=bytearray()))
        self.assertTrue(any("H3VFX " in line for line in updates[-1].diagnostics))
        self.assertLessEqual(len(updates[-1].diagnostics), 99)

    def test_dlss_probe_requires_rgb_changes_and_feature_initialization(self):
        for change_rgb, evidence, succeeds in ((False, True, False), (True, False, False), (True, True, True)):
            command = self.command(
                "import sys; data=bytearray(sys.stdin.buffer.read()); "
                + ("data[0]^=1; " if change_rgb else "data[3]=0; ")
                + ("print('feature ready; streaming', file=sys.stderr); " if evidence else "")
                + "sys.stdout.buffer.write(data)"
            )
            with self.subTest(rgb=change_rgb, evidence=evidence), ExitStack() as stack:
                stack.enter_context(patch.object(service, "availability", return_value=(True, "")))
                stack.enter_context(patch.object(service, "worker_command", return_value=(command, None)))
                stack.enter_context(patch.object(service, "worker_context", return_value=nullcontext()))
                if succeeds:
                    self.assertIn("passed", service.probe_runtime(DLSS5_NR, Mock(generation_timeout=15)))
                else:
                    with self.assertRaises(H3Error):
                        service.probe_runtime(DLSS5_NR, Mock(generation_timeout=15))


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class NvidiaMediaTests(unittest.TestCase):
    def test_dlss_rgba_video_without_audio_records_runtime_evidence(self):
        with TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            source = root / "silent.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                "testsrc2=s=64x48:r=24:d=0.25", "-c:v", "libx264", str(source)], check=True)
            runtime = RuntimeConfig(root, "", root / "comfy", root / "models.json", root / "outputs", generation_timeout=30)
            worker = [sys.executable, "-u", "-c",
                "import sys; print('feature ready; streaming', file=sys.stderr); "
                "print('optical flow: fixture', file=sys.stderr); "
                "data=sys.stdin.buffer.read(); sys.stdout.buffer.write(bytes(255-x for x in data))"]
            stack.enter_context(patch.object(service, "availability", return_value=(True, "")))
            stack.enter_context(patch.object(service, "worker_command", return_value=(worker, None)))
            stack.enter_context(patch.object(service, "worker_context", side_effect=lambda *args: nullcontext()))
            stack.enter_context(patch.object(service, "dlss_config", return_value=(source, root, source, None)))
            stack.enter_context(patch.object(service, "_encoder", return_value=("libx265", ["-c:v", "libx265", "-preset", "ultrafast"])))
            probe = stack.enter_context(patch.object(service, "probe_runtime", return_value="fixture passed"))
            result = list(service.postprocess(source, DLSS5_NR, "1920 × 1920",
                {"original_size": True}, runtime=runtime))[-1].result
            probe.assert_called_once_with(DLSS5_NR, runtime)
            payload, metadata, rate = service.probe_source(result, runtime, count=True)
            self.assertFalse(any(s["codec_type"] == "audio" for s in payload["streams"]))
            self.assertEqual(int(metadata["nb_read_frames"]), 6)
            self.assertEqual(rate, 24)
            settings = read_snapshot(result)["settings"]
            self.assertEqual(settings["stage_order"], "NR")
            self.assertEqual(settings["motion_backend"], "fixture")
            self.assertEqual(settings["model_sha256"], service.file_hash(source))

    def test_video_audio_timing_provenance_and_failure_cleanup(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=64x48:r=24000/1001:d=0.5",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5", "-c:v", "libx264", "-c:a", "aac", "-shortest", str(source)], check=True)
            runtime = RuntimeConfig(root, "", root / "comfy", root / "models.json", root / "outputs", generation_timeout=30)
            write_snapshot(source, {"settings": {"fixture": True}})
            worker = [sys.executable, "-u", "-c", "import sys;\nwhile True:\n b=sys.stdin.buffer.read(64*48*3)\n if not b: break\n sys.stdout.buffer.write(bytes(255-x for x in b)); sys.stdout.buffer.flush()"]
            with ExitStack() as stack:
                stack.enter_context(patch.object(service, "availability", return_value=(True, "")))
                stack.enter_context(patch.object(service, "worker_command", return_value=(worker, None)))
                stack.enter_context(patch.object(service, "_encoder", return_value=("libx265", ["-c:v", "libx265", "-preset", "ultrafast", "-crf", "19"])))
                updates = list(service.postprocess(source, NVIDIA_VSR, "1920 × 1920",
                    {"original_size": True, "preprocess": "Denoise"}, runtime=runtime))
                result = updates[-1].result
                self.assertTrue(result.is_file())
                payload, metadata, rate = service.probe_source(result, runtime, count=True)
                self.assertEqual(str(rate), "24000/1001")
                self.assertEqual(int(metadata["nb_read_frames"]), updates[-1].frames)
                self.assertTrue(any(s["codec_type"] == "audio" for s in payload["streams"]))
                snapshot = read_snapshot(result)
                self.assertEqual(snapshot["source_asset_ids"], [read_snapshot(source)["asset_id"]])
                self.assertTrue(snapshot["settings"]["encoder_fallback"])
                stack.enter_context(patch.object(service, "worker_command", return_value=([sys.executable, "-c", "raise SystemExit(4)"], None)))
                with self.assertRaises(H3Error):
                    list(service.postprocess(source, NVIDIA_VSR, "1920 × 1920",
                        {"original_size": True, "preprocess": "Denoise"}, runtime=runtime))
                self.assertEqual(list((root / "outputs/nvidia_vsr").glob(".processing-*")), [])
                self.assertEqual(len(list((root / "outputs/nvidia_vsr").glob("*.mp4"))), 1)


if __name__ == "__main__":
    unittest.main()
