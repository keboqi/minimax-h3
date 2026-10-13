"""CPU image acceptance: color, transparency, real pipes and gallery dispatch."""

from contextlib import ExitStack, nullcontext
from io import BytesIO
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from PIL import Image, ImageCms

from h3_app import nvidia_images as images
from h3_app import nvidia_postprocess as backend
from h3_app.catalog import DLSS5_NR
from h3_app.config import RuntimeConfig
from h3_app.errors import H3Error
from h3_app.jobs import JobCancelled
from h3_app.provenance import read_snapshot, write_snapshot


class NvidiaImageTests(unittest.TestCase):
    def runtime(self, root):
        return RuntimeConfig(root, "", root / "comfy", root / "models.json",
                             root / "outputs", generation_timeout=15)

    def mock_runtime(self, stack, source, command):
        stack.enter_context(patch.object(backend, "availability", return_value=(True, "")))
        stack.enter_context(patch.object(backend, "runtime_ready", return_value=True))
        stack.enter_context(patch.object(backend, "worker_command", return_value=(command, None)))
        stack.enter_context(patch.object(backend, "worker_context", side_effect=lambda *args: nullcontext()))
        stack.enter_context(patch.object(backend, "dlss_config", return_value=(source, source.parent, source, None)))
        return stack.enter_context(patch.object(backend, "probe_runtime", return_value="fixture passed"))

    def test_exif_orientation_and_color_profile_normalized(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "rotated.jpg"
            image = Image.new("RGB", (64, 48), (20, 50, 100))
            exif = image.getexif()
            exif[274] = 6
            icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
            image.save(source, exif=exif, icc_profile=icc)
            normalized, alpha, profile = images.load_image(source)
            self.assertEqual(normalized.size, (48, 64))
            self.assertIsNone(alpha)
            self.assertIn("sRGB", ImageCms.getProfileName(ImageCms.ImageCmsProfile(BytesIO(profile))))

    def test_dlss_native_png_preserves_alpha_source_and_lineage(self):
        with TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            source = root / "transparent.png"
            original = Image.new("RGBA", (64, 48), (20, 50, 100, 87))
            original.save(source)
            unchanged = source.read_bytes()
            write_snapshot(source, {"family": "fixture"})
            command = [sys.executable, "-u", "-c",
                "import sys; print('feature ready; streaming', file=sys.stderr); "
                "sys.stdout.buffer.write(bytes(255-x for x in sys.stdin.buffer.read()))"]
            probe = self.mock_runtime(stack, source, command)
            result = list(images.postprocess_image(source, DLSS5_NR, "1920 × 1920",
                {"original_size": True}, runtime=self.runtime(root)))[-1].result
            probe.assert_called_once()
            with Image.open(result) as enhanced:
                self.assertEqual(enhanced.mode, "RGBA")
                self.assertEqual(enhanced.size, original.size)
                self.assertEqual(enhanced.getpixel((0, 0)), (235, 205, 155, 87))
                self.assertTrue(enhanced.info["icc_profile"])
            snapshot = read_snapshot(result)
            self.assertEqual(snapshot["source_asset_ids"], [read_snapshot(source)["asset_id"]])
            self.assertEqual(snapshot["settings"]["backend"], "RunningHub DNR2")
            self.assertEqual(snapshot["settings"]["alpha"], "preserved")
            self.assertEqual(source.read_bytes(), unchanged)

    def test_dlss_upscaled_png_uses_no_motion_and_preserves_alpha(self):
        with TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            source = root / "transparent.png"
            Image.new("RGBA", (65, 49), (20, 50, 100, 87)).save(source)
            # Default upscaler: Lanczos to the target, so the worker runs DLSS 5 at 1×.
            command = [sys.executable, "-u", "-c",
                "import sys; print('feature ready; streaming', file=sys.stderr); "
                "data=sys.stdin.buffer.read(); assert len(data)==130*98*4; sys.stdout.buffer.write(data)",
                "--motion", "0"]
            probe = self.mock_runtime(stack, source, command)
            stack.enter_context(patch.dict(backend.UPSCALE_RESOLUTION_PRESETS, {"fixture": (130, 98)}))
            runtime = self.runtime(root)
            result = list(images.postprocess_image(source, DLSS5_NR, "fixture", {}, runtime=runtime))[-1].result
            probe.assert_called_once_with(DLSS5_NR, runtime, ngx_core="default")
            self.assertEqual(backend.worker_command.call_args.args[1:5], (130, 98, 130, 98))
            self.assertEqual(backend.worker_command.call_args.kwargs["ngx_core"], "default")
            self.assertEqual(command[-1], "0")
            with Image.open(result) as enhanced:
                self.assertEqual(enhanced.size, (130, 98))
                self.assertEqual(enhanced.getpixel((0, 0)), (20, 50, 100, 87))
            settings = read_snapshot(result)["settings"]
            self.assertEqual(settings["stage_order"], "Lanczos → NR")
            self.assertEqual(settings["upscaler"], "resize")
            self.assertEqual(settings["motion"], "disabled")
            self.assertEqual(settings["model_sha256"], backend.file_hash(source))
            self.assertEqual(backend.target_dimensions(DLSS5_NR, 65, 49, "fixture",
                backend.NvidiaSettings(original_size=True), image=True), (65, 49))

    def test_experimental_dlss_sr_uses_r580_core_only_on_linux(self):
        settings = backend.NvidiaSettings(upscaler="dlss_sr")
        info = {"driver": "581.57", "core_sha256": "ab"}
        with patch.object(backend, "ensure_r580_core", return_value=info) as ensure:
            with patch.object(backend.os, "name", "posix"):
                plan = backend.plan_upscale(settings, 64, 48, 128, 96)
            self.assertEqual((plan.width, plan.height, plan.resize, plan.ngx_core), (64, 48, False, "r580"))
            self.assertEqual((plan.core_info, plan.stage_order), (info, "DLSS SR → NR"))
            # prepare_upscale reports the one-time download, then returns the same plan.
            with patch.object(backend.os, "name", "posix"):
                steps = backend.prepare_upscale(settings, 64, 48, 128, 96, total=3)
                self.assertIn("R580", next(steps).message)
                with self.assertRaises(StopIteration) as done:
                    next(steps)
            self.assertEqual(done.exception.value, plan)
            self.assertEqual(backend.plan_upscale(backend.NvidiaSettings(), 64, 48, 128, 96).stage_order,
                             "Lanczos → NR")
            # Native size never needs a carrier, so nothing is downloaded.
            with patch.object(backend.os, "name", "posix"):
                self.assertEqual(backend.plan_upscale(settings, 64, 48, 64, 48).ngx_core, "default")
            self.assertEqual(ensure.call_count, 2)
        with self.assertRaises(H3Error):
            backend.NvidiaSettings.parse({"upscaler": "bicubic"})

    def test_bad_worker_and_cancellation_publish_no_image(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel), TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory)
                source = root / "source.png"
                Image.new("RGB", (64, 48)).save(source)
                command = [sys.executable, "-u", "-c", "import time; time.sleep(30)" if cancel else
                    "import sys; sys.stdin.buffer.read(); sys.stdout.buffer.write(b'x')"]
                self.mock_runtime(stack, source, command)
                if cancel:
                    stack.enter_context(patch("h3_app.nvidia_pipeline.check_cancelled",
                                              side_effect=[None, JobCancelled("stop")]))
                with self.assertRaises(JobCancelled if cancel else H3Error):
                    list(images.postprocess_image(source, DLSS5_NR, "1920 × 1920",
                        {"original_size": True}, runtime=self.runtime(root)))
                self.assertEqual(list((root / "outputs").rglob("*.png")), [])
                self.assertEqual(list((root / "outputs").rglob(".processing-*")), [])
                self.assertTrue(source.is_file())

    def test_animated_high_depth_and_invalid_profile_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("I;16", (64, 48)).save(root / "deep.png")
            Image.new("RGB", (64, 48)).save(root / "profile.png", icc_profile=b"invalid")
            Image.new("RGB", (64, 48), "red").save(root / "animated.png", save_all=True,
                append_images=[Image.new("RGB", (64, 48), "blue")], duration=10, loop=0)
            for source in root.iterdir():
                with self.subTest(source=source), self.assertRaises(H3Error):
                    images.load_image(source)

    def test_gallery_dispatch_forwards_settings_without_comfy_graph(self):
        from h3_ui import application as app
        from h3_ui import media_controller
        settings = {"original_size": True, "intensity": 0.4}
        with ExitStack() as stack:
            stack.enter_context(patch.object(app, "managed_gallery_image_path", return_value=Path("source.png")))
            stack.enter_context(patch.object(app, "gallery_media_progress_result", side_effect=lambda message: (message,)))
            complete = stack.enter_context(patch.object(app, "gallery_media_processed_result", return_value=("complete",)))
            offload = stack.enter_context(patch.object(app, "unload_comfy_models"))
            graph = stack.enter_context(patch.object(app, "object_info"))
            service = stack.enter_context(patch.object(media_controller, "nvidia_postprocess_image", return_value=iter([
                backend.NvidiaUpdate("processing"), backend.NvidiaUpdate("done", 1, 1, Path("result.png")),
            ])))
            updates = list(app.postprocess_selected_gallery_media("Image", "source.png", DLSS5_NR,
                42, "unused", "unused", "", True, False, 5, "1920 × 1920", None,
                progress=Mock(), nvidia_settings=settings))
            self.assertEqual(updates[-1], ("complete",))
            self.assertEqual(service.call_args.args[-1], settings)
            offload.assert_called_once()
            graph.assert_not_called()
            self.assertEqual(complete.call_args.args[:3], ("Image", Path("result.png"), DLSS5_NR))


if __name__ == "__main__":
    unittest.main()
