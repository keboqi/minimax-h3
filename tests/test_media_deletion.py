"""Confirmed media deletion removes originals, previews and catalog listings."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from PIL import Image
from h3_app import gallery_store
from h3_app.config import RuntimeConfig
from h3_app.media_catalog import MediaCatalog
from h3_app.provenance import snapshot_path, write_snapshot
from h3_app.workspace_store import WorkspaceStore
from h3_ui import application as app
from h3_ui.asset_index import AssetIndex


class MediaDeletionTests(TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runtime = RuntimeConfig(
            self.root, "", self.root / "comfy", self.root / "models.json",
            self.root / "outputs",
        )
        self.runtime.outputs_dir.mkdir()
        self.store = WorkspaceStore(self.root / "workspace")
        self.catalog = MediaCatalog(self.store, lambda: self.runtime)
        for target, value in (
            ("h3_app.media_catalog._CATALOG", self.catalog),
            ("h3_ui.application._runtime_config", lambda: self.runtime),
        ):
            replacement = patch(target, value)
            replacement.start()
            self.addCleanup(replacement.stop)
        self.index = AssetIndex(self.store, lambda mode: [], catalog=self.catalog)

    def create_media(self, mode, name="selected"):
        extension = {"Video": ".mp4", "Image": ".png", "Audio": ".wav"}[mode]
        source = self.runtime.outputs_dir / (name + extension)
        if mode == "Image":
            Image.new("RGB", (32, 24), "red").save(source)
        else:
            source.write_bytes(b"media fixture")
        with patch.object(gallery_store, "gallery_thumbnail", return_value=None):
            write_snapshot(source, {"family": "Deletion test"})
        record = self.index.inventory(mode).records[0]
        preview = self.runtime.gallery_thumbnails_dir / record["metadata"]["_media"]["thumbnail"]
        return source, preview

    def test_confirmed_selection_disappears_from_catalog_and_restart(self):
        for mode in ("Video", "Image", "Audio"):
            with self.subTest(mode=mode):
                source, preview = self.create_media(mode)
                result = app.delete_selected_gallery_media(mode, str(source), False)
                self.assertIn("Confirm permanent deletion", result[2])
                self.assertTrue(source.exists())
                self.assertTrue(preview.exists())
                self.assertTrue(snapshot_path(source).exists())
                result = app.delete_selected_gallery_media(mode, str(source), True)
                self.assertIn(f"Deleted `{source.name}`", result[2])
                self.assertFalse(source.exists())
                self.assertFalse(preview.exists())
                self.assertFalse(snapshot_path(source).exists())
                self.assertIsNone(result[7])
                self.assertEqual(result[1], [])
                reopened = WorkspaceStore(self.store.root)
                self.assertEqual(reopened.catalog_page(kind=mode)[1], 0)

    def test_empty_library_removes_only_the_active_media_type(self):
        for mode in ("Video", "Image", "Audio"):
            with self.subTest(mode=mode):
                source, preview = self.create_media(mode)
                other_mode = "Image" if mode != "Image" else "Audio"
                retained, retained_preview = self.create_media(other_mode, "retained")
                app.empty_generated_media_gallery(mode, str(source), False)
                self.assertTrue(source.exists())
                result = app.empty_generated_media_gallery(mode, str(source), True)
                self.assertIn("Deleted 1 generated", result[2])
                self.assertFalse(source.exists())
                self.assertFalse(preview.exists())
                self.assertFalse(snapshot_path(source).exists())
                self.assertTrue(retained.exists())
                self.assertTrue(retained_preview.exists())
                self.assertEqual(self.index.inventory(mode).total, 0)
                app.delete_selected_gallery_media(other_mode, str(retained), True)

    def test_cleanup_failure_still_retires_deleted_original(self):
        source, _ = self.create_media("Image")
        original_unlink = Path.unlink

        def unlink(path, *args, **kwargs):
            if path == snapshot_path(source):
                raise PermissionError("sidecar is locked")
            return original_unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", unlink):
            result = app.delete_selected_gallery_media("Image", str(source), True)
        self.assertIn("Delete failed: sidecar is locked", result[2])
        self.assertFalse(source.exists())
        self.assertEqual(self.index.inventory("Image").total, 0)

    def test_unmanaged_paths_are_rejected(self):
        outside = self.root / "outside.png"
        Image.new("RGB", (32, 24)).save(outside)
        result = app.delete_selected_gallery_media("Image", str(outside), True)
        self.assertIn("Delete failed", result[2])
        self.assertTrue(outside.exists())

