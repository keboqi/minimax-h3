from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from h3_app.provenance import write_snapshot
from h3_app.workspace_store import WorkspaceStore
from h3_ui.asset_index import AssetIndex


class AssetIndexTests(unittest.TestCase):
    def test_browsing_reuses_recent_inventory_but_refresh_finds_new_files(self):
        image = self.root / "alpha.png"
        image.write_bytes(b"first")
        self.index.inventory("Image")
        with patch.object(self.index, "list_paths", side_effect=AssertionError("rescan")):
            self.assertEqual(self.index.inventory("Image", force=False).paths, (image,))
        other = self.root / "beta.png"
        other.write_bytes(b"new")
        self.assertEqual(set(self.index.inventory("Image").paths), {image, other})
        with patch("h3_ui.asset_index.monotonic", return_value=self.index.scanned_at["Image"] + 6):
            with patch.object(self.index, "list_paths", wraps=self.index.list_paths) as scan:
                self.index.inventory("Image", force=False)
                scan.assert_called_once()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = WorkspaceStore(self.root / "workspace")
        self.index = AssetIndex(
            self.store,
            lambda mode: list(self.root.glob("*.png")) if mode == "Image" else [],
        )

    def test_changed_files_and_sidecars_refresh_without_reindexing_unchanged_files(
        self,
    ):
        image = self.root / "alpha.png"
        image.write_bytes(b"first")
        with patch.object(self.store, "connect", wraps=self.store.connect) as connect:
            self.assertEqual(self.index.paths("Image"), {image})
            self.index.paths("Image")
            self.assertEqual(connect.call_count, 1)
            write_snapshot(
                image,
                {
                    "seed": 7,
                    "prompt": "private draft",
                    "nested": {"api_key": "secret", "steps": 4},
                },
            )
            self.index.paths("Image")
            self.assertEqual(connect.call_count, 2)
        row = self.store.search_assets()[0]
        self.assertEqual(row["metadata"]["seed"], 7)
        self.assertEqual(row["metadata"]["nested"], {"steps": 4})
        self.assertNotIn("prompt", row["metadata"])
        self.assertEqual(self.index.paths("Image", "private draft"), set())

    def test_large_scan_commits_once_and_cached_inventory_never_rescans(self):
        for number in range(50):
            (self.root / f"image-{number:03}.png").write_bytes(b"image")
        with patch.object(self.store, "connect", wraps=self.store.connect) as connect:
            allowed = self.index.paths("Image")
            self.assertEqual(connect.call_count, 1)
        with patch.object(
            self.index, "list_paths", side_effect=AssertionError("rescan")
        ):
            self.assertEqual(self.index.cached_paths("Image"), allowed)
            self.assertEqual(self.index.cached_paths("Video"), set())

    def test_filters_and_rebuild_preserve_annotations_and_exclude_missing_files(self):
        image = self.root / "alpha.png"
        other = self.root / "beta.png"
        image.write_bytes(b"alpha")
        other.write_bytes(b"beta")
        self.index.paths("Image")
        row = next(
            row for row in self.store.search_assets() if row["path"] == str(image)
        )
        self.store.annotate(row["id"], tags=["favorite scene"], favorite=True)
        self.assertEqual(self.index.paths("Image", "favorite scene", True), {image})
        self.assertEqual(self.index.paths("Image", "beta", True), set())
        list(self.index.rebuild())
        self.assertEqual(self.index.paths("Image", favorite=True), {image})
        image.unlink()
        self.assertEqual(self.index.paths("Image", favorite=True), set())
        list(self.index.rebuild())
        self.assertEqual(len(self.store.search_assets()), 1)

    def test_filtered_pagination_can_find_assets_beyond_the_search_dropdown_limit(self):
        for number in range(205):
            (self.root / f"image-{number:03}.png").write_bytes(b"image")
        allowed = self.index.paths("Image")
        self.assertEqual(len(allowed), 205)
        self.assertEqual(len(self.store.search_assets()), 200)
        self.assertEqual(self.index.paths("Image", "image-"), allowed)
