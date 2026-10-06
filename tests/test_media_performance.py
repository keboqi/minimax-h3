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


class MediaPerformanceTests(TestCase):
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
        services = SimpleNamespace(
            gallery_media_mode=lambda mode: mode,
            managed_video_path=Mock(return_value=source),
            absolute_video_url=Mock(return_value=url),
            absolute_video_download_url=Mock(return_value=url + "?download=1"),
            gallery_preview_updates=lambda mode, **values: tuple(gr.update(value=values.get(kind), visible=mode == kind.title()) for kind in ("video", "image", "audio")),
            select_gallery_video=Mock(side_effect=AssertionError("metadata probe")),
        )
        controller = MediaController(services)
        updates = controller.select_gallery_media("Video", [str(source)], Mock(), SimpleNamespace(index=0))
        self.assertEqual(updates[-1], str(source))
        with patch.object(processing_utils, "hash_file", side_effect=AssertionError("whole-file hash")), patch.object(processing_utils, "video_is_playable", side_effect=AssertionError("codec probe")):
            component = gr.Video()
            output = browser_preview_updates(updates)[0]["value"]
            result = processing_utils.move_files_to_cache(output, component, postprocess=True)
        self.assertEqual(result["url"], url)

    def test_progressive_video_page_preserves_selection_until_poster_is_ready(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "original.mp4"
            source.touch()
            runtime = RuntimeConfig(root, "", root, root / "models.json", root)
            release = Event()
            def poster(path):
                release.wait(5)
                return root / "poster.jpg"
            services = SimpleNamespace(
                Path=Path, gallery_thumbnail=poster, gallery_store=gallery_store,
                GALLERY_THUMBNAILS_DIR=runtime.gallery_thumbnails_dir,
                _runtime_config=lambda: runtime,
                generated_video_family=lambda path: "Generated", time=__import__("time"),
            )
            controller = MediaController(services)
            inventory = gallery_store.AssetInventory("Video", (source,))
            try:
                first = controller.refresh_gallery_page(paths=inventory, preview_timeout=0)
                self.assertEqual(first.preparing, 1)
                self.assertEqual(first.unavailable, 0)
                self.assertEqual(first.paths, (str(source),))
            finally:
                release.set()
            final = controller.refresh_gallery_page(paths=inventory)
            self.assertEqual(final.preparing, 0)
            self.assertEqual(final.paths, first.paths)
            self.assertEqual(final.items[0][0], str(root / "poster.jpg"))
