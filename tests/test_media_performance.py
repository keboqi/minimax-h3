"""Responsiveness contracts for slow previews and large selected media."""

from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import gradio as gr
from gradio import processing_utils
from h3_app.config import RuntimeConfig
from h3_app import gallery_store
from h3_ui.media_controller import MediaController
from h3_ui.media_previews import preview_results, browser_preview_updates
from dataclasses import replace


class MediaPerformanceTests(TestCase):
    def test_thumbnails_persist_inside_outputs_with_optional_override(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = RuntimeConfig(root, "", root, root / "models.json", root / "drive" / "outputs")
            self.assertTrue(runtime.gallery_thumbnails_dir.is_relative_to(runtime.outputs_dir))
            custom = RuntimeConfig.from_environment(root, {"H3_GALLERY_CACHE_DIR": str(root / "local-cache")})
            self.assertEqual(custom.gallery_thumbnails_dir, root / "local-cache")

    def test_existing_thumbnail_is_reused_without_reading_original_pixels(self):
        from PIL import Image
        with TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = replace(RuntimeConfig(root, "", root, root / "models.json", root / "drive"), thumbnail_root=root / "local-cache")
            source = runtime.outputs_dir / "original.png"
            source.parent.mkdir()
            source.write_bytes(b"original pixels must never be opened")
            thumbnail = gallery_store.gallery_image_thumbnail_path(source, runtime=runtime)
            thumbnail.parent.mkdir()
            Image.new("RGB", (100, 80)).save(thumbnail)
            with patch.object(Image, "open", side_effect=AssertionError("original image read")):
                self.assertEqual(gallery_store.gallery_image_thumbnail(source, runtime=runtime), thumbnail)

    def test_image_selection_serves_full_original_without_decoding_pixels(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = replace(RuntimeConfig(root, "", root, root / "models.json", root / "drive"), thumbnail_root=root / "local-cache")
            source = runtime.outputs_dir / "large.png"
            source.parent.mkdir()
            source.write_bytes(b"original")
            file_stat = source.stat()
            thumbnail = gallery_store.gallery_image_thumbnail_path(source, runtime=runtime)
            thumbnail.parent.mkdir()
            thumbnail.touch()
            controller = MediaController(SimpleNamespace(_runtime_config=lambda: runtime))
            controller.managed_gallery_image_path = Mock(return_value=source)
            controller.gallery_image_resolution_text = Mock(side_effect=AssertionError("original image read"))
            controller.absolute_gallery_media_download_url = Mock(return_value="https://example.com/downloads/gradio/large.png?download=1")
            request = SimpleNamespace(request=SimpleNamespace(base_url="https://example.com/"))
            updates = controller.select_gallery_media("Image", [str(source)], request, SimpleNamespace(index=0))
            self.assertEqual(updates[1]["value"], f"https://example.com/downloads/gradio/large.png?v={file_stat.st_mtime_ns}-{file_stat.st_size}")
            self.assertEqual(updates[-1], str(source))
            self.assertIn("/downloads/gradio/large.png?download=1", updates[3])
            controller.managed_gallery_image_path.assert_called_once_with(str(source), require_file=False)
    def test_image_dimensions_honor_orientation_without_loading_pixels(self):
        from PIL import Image
        with TemporaryDirectory() as directory:
            image = Path(directory) / "rotated.jpg"
            exif = Image.Exif()
            exif[274] = 6
            Image.new("RGB", (40, 20)).save(image, exif=exif)
            with patch.object(Image.Image, "load", side_effect=AssertionError("pixel decode")):
                self.assertEqual(gallery_store.gallery_image_resolution(image), (20, 40))

    def test_slow_preview_is_pending_without_blocking_and_work_is_shared(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "slow.mp4"
            source.touch()
            release = Event()
            entered = Event()

            def create(path):
                entered.set()
                release.wait(5)
                return path.with_suffix(".jpg")

            create = Mock(side_effect=create)
            try:
                start = monotonic()
                values = preview_results([source], create, kind="Video", cache_root=directory, timeout=0)
                self.assertLess(monotonic() - start, 0.5)
                self.assertEqual(values, [(None, True)])
                self.assertTrue(entered.wait(1))
                self.assertEqual(preview_results([source], create, kind="Video", cache_root=directory, timeout=0), values)
                create.assert_called_once_with(source)
            finally:
                release.set()
            self.assertEqual(preview_results([source], create, kind="Video", cache_root=directory), [(source.with_suffix(".jpg"), False)])
            source.write_bytes(b"replacement")
            preview_results([source], create, kind="Video", cache_root=directory)
            self.assertEqual(create.call_count, 2)

    def test_video_selection_does_not_probe_hash_copy_or_transcode_original(self):
        source = Path("original.mp4")
        url = "https://example.com/downloads/comfy/original.mp4"
        controller = MediaController(SimpleNamespace())
        controller.managed_video_path = Mock(return_value=source)
        controller.absolute_video_url = Mock(return_value=url)
        controller.absolute_video_download_url = Mock(return_value=url + "?download=1")
        controller.select_gallery_video = Mock(side_effect=AssertionError("metadata probe"))
        updates = controller.select_gallery_media("Video", [str(source)], Mock(), SimpleNamespace(index=0))
        self.assertEqual(updates[-1], str(source))
        with patch.object(processing_utils, "hash_file", side_effect=AssertionError("whole-file hash")), patch.object(processing_utils, "video_is_playable", side_effect=AssertionError("codec probe")):
            component = gr.Video()
            output = browser_preview_updates(updates)[0]["value"]
            result = processing_utils.move_files_to_cache(output, component, postprocess=True)
        self.assertEqual(result["url"], url)
