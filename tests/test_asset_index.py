"""Creation-time catalog and explicit historical discovery for every media type."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from PIL import Image
from h3_app.config import RuntimeConfig
from h3_app.gallery_store import gallery_image_paths
from h3_app.media_catalog import MediaCatalog
from h3_app.provenance import copy_media, read_snapshot, write_snapshot
from h3_app.workspace_store import WorkspaceStore, default_store
from h3_ui.asset_index import AssetIndex
from h3_ui.media_controller import MediaController


class AssetIndexTests(TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runtime = replace(RuntimeConfig(self.root, "", self.root, self.root / "models.json", self.root / "outputs"), output_root=self.root / "comfy-output")
        self.runtime.outputs_dir.mkdir()
        self.store = WorkspaceStore(self.runtime.outputs_dir / ".h3-workspace")
        self.catalog = MediaCatalog(self.store, lambda: self.runtime)
        self.scan = Mock(side_effect=lambda mode: gallery_image_paths(runtime=self.runtime) if mode == "Image" else [])
        self.index = AssetIndex(self.store, self.scan, catalog=self.catalog)
        configured = patch("h3_app.media_catalog._CATALOG", self.catalog)
        configured.start()
        self.addCleanup(configured.stop)

    def create_image(self, name="alpha.png", metadata=None):
        path = self.runtime.outputs_dir / name
        Image.new("RGB", (32, 24), "red").save(path)
        write_snapshot(path, metadata or {"seed": 7, "prompt": "private draft"})
        return path

    def test_creation_prepares_thumbnails_and_metadata_for_all_media_types(self):
        image = self.create_image()
        video = self.runtime.outputs_dir / "clip.mp4"
        video.write_bytes(b"video fixture")
        poster = self.runtime.gallery_thumbnails_dir / "video.jpg"
        Image.new("RGB", (32, 24)).save(poster)
        with patch("h3_app.gallery_store.gallery_thumbnail", return_value=poster):
            write_snapshot(video, {"family": "Video test", "seed": 4})
        audio = self.runtime.outputs_dir / "sound.wav"
        audio.write_bytes(b"audio fixture")
        write_snapshot(audio, {"family": "Audio test", "seed": 5})
        for kind, source in (("Image", image), ("Video", video), ("Audio", audio)):
            page = self.index.inventory(kind)
            self.assertEqual(page.paths, (source,))
            metadata = page.records[0]["metadata"]
            self.assertTrue((self.runtime.gallery_thumbnails_dir / metadata["_media"]["thumbnail"]).is_file())
            self.assertNotIn("prompt", metadata)
        self.scan.assert_not_called()

    def test_restart_browsing_and_search_do_not_scan_stat_or_read_sidecars(self):
        image = self.create_image()
        reopened = WorkspaceStore(self.store.root)
        index = AssetIndex(reopened, self.scan, catalog=self.catalog)
        controller = MediaController(SimpleNamespace(_runtime_config=lambda: self.runtime, gallery_media_mode=lambda mode: mode))
        with patch.object(Path, "stat", side_effect=AssertionError("filesystem stat")), patch("h3_ui.asset_index.read_snapshot", side_effect=AssertionError("sidecar read")):
            inventory = index.inventory("Image", "alpha", limit=48)
            page = controller.refresh_media_page("Image", paths=inventory)
        self.assertEqual(page.paths, (str(image),))
        self.assertEqual(len(page.items), 1)
        self.assertEqual(page.total, 1)
        self.scan.assert_not_called()

    def test_history_scan_is_explicit_preserves_annotations_and_reconciles_missing_files(self):
        known = self.create_image()
        row = self.store.search_assets()[0]
        self.store.annotate(row["id"], tags=["scene"], favorite=True)
        old = self.runtime.outputs_dir / "historical.png"
        Image.new("RGB", (32, 24)).save(old)
        self.assertEqual(self.index.paths("Image"), {known})
        self.scan.assert_not_called()
        list(self.index.rebuild())
        self.assertEqual(self.index.paths("Image"), {known, old})
        self.assertEqual(self.index.paths("Image", "scene", True), {known})
        known.unlink()
        list(self.index.rebuild())
        self.assertEqual(self.index.paths("Image", favorite=True), set())
        self.assertEqual(self.store.search_assets()[0]["path"], str(old))

    def test_catalog_pagination_is_bounded_and_search_finds_history_beyond_200(self):
        rows = []
        for number in range(205):
            source = self.runtime.outputs_dir / f"image-{number:03}.png"
            source.touch()
            rows.append((source, "Image", {"_media": {"mtime_ns": number, "registered_ns": number, "thumbnail": "thumb.jpg", "caption": "Image"}}))
        self.store.index_assets(rows)
        page = self.index.inventory("Image", "image-", limit=48)
        self.assertEqual(len(page.paths), 48)
        self.assertEqual(page.total, 205)
        self.assertEqual(len(self.index.paths("Image", "image-")), 205)
        self.scan.assert_not_called()

    def test_database_lives_in_output_folder_and_previous_state_is_preserved(self):
        outputs = self.root / "migrated-outputs"
        previous = WorkspaceStore(outputs.parent / "h3-workspace")
        previous_secret = previous.secret
        with patch.dict("os.environ", {"H3_WORKSPACE_DIR": ""}):
            current = default_store(outputs)
        self.assertTrue(current.database.is_relative_to(outputs))
        self.assertEqual(current.secret, previous_secret)

    def test_unreadable_history_folder_aborts_without_removing_catalog_records(self):
        image = self.create_image()
        with patch("h3_app.gallery_store.os.scandir", side_effect=PermissionError("Drive unavailable")):
            with self.assertRaises(PermissionError):
                gallery_image_paths(runtime=self.runtime, strict=True)
        self.scan.side_effect = PermissionError("Drive unavailable")
        with self.assertRaises(PermissionError):
            list(self.index.rebuild())
        self.assertEqual(self.index.paths("Image"), {image})

    def test_registration_during_history_scan_survives_reconciliation(self):
        old = self.create_image("old.png")
        scan = self.index.rebuild()
        next(scan)
        # This file arrives after the scanner's snapshot of that media type.
        self.scan.side_effect = lambda mode: [old] if mode == "Image" else []
        next(scan)
        next(scan)
        created = self.create_image("concurrent.png")
        list(scan)
        self.assertEqual(self.index.paths("Image"), {old, created})

    def test_generic_metadata_replacement_does_not_leave_invalid_catalog_membership(self):
        image = self.create_image()
        self.store.index_asset(image, "Image", {"seed": 8})
        self.assertEqual(self.index.inventory("Image").total, 0)
        write_snapshot(image, {"seed": 9})
        self.store.index_assets([(image, "Image", {})])
        self.assertEqual(self.index.inventory("Image").total, 0)
        write_snapshot(image, {"seed": 10})
        self.assertEqual(self.index.paths("Image"), {image})

    def test_complete_copy_is_registered_with_and_without_source_metadata(self):
        source = self.root / "uploaded.png"
        Image.new("RGB", (32, 24)).save(source)
        plain = self.runtime.outputs_dir / "plain-copy.png"
        copy_media(source, plain)
        write_snapshot(source, {"seed": 42})
        annotated = self.runtime.outputs_dir / "annotated-copy.png"
        copy_media(source, annotated)
        self.assertEqual(self.index.paths("Image"), {plain, annotated})
        self.assertEqual(read_snapshot(annotated)["seed"], 42)
        self.assertTrue(read_snapshot(plain)["source_asset_ids"])

    def test_processed_video_is_registered_after_final_output_exists(self):
        import subprocess
        from h3_app.media_tools import postprocess_video
        source = self.root / "uploaded.mp4"
        source.write_bytes(b"source video")

        def process(command, **kwargs):
            Path(command[-1]).write_bytes(b"complete processed video")
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch("h3_app.media_tools.has_encoder", return_value=False), patch(
            "h3_app.media_tools.run_media_process", side_effect=process,
        ), patch("h3_app.gallery_store.gallery_thumbnail", return_value=None):
            result = postprocess_video(source, "48 fps interpolation", runtime=self.runtime)
        self.assertEqual(result.read_bytes(), b"complete processed video")
        self.assertEqual(self.index.paths("Video"), {result})
        self.assertTrue(read_snapshot(result)["source_asset_ids"])
