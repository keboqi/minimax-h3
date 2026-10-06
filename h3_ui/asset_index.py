"""Read the persisted catalog; filesystem discovery is an explicit history scan."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
from threading import RLock
from time import time_ns

from h3_app.provenance import read_snapshot
from h3_app.gallery_store import AssetInventory
from h3_app.media_catalog import get_media_catalog


class AssetIndex:
    def __init__(self, store, list_paths, *, catalog=None):
        self.store = store
        self.list_paths = list_paths
        self.catalog = catalog or get_media_catalog()
        self.inventories = {}
        self.versions = {}
        self.lock = RLock()

    def inventory(self, mode, query="", favorite=False, *, limit=None):
        rows, total = self.store.catalog_page(
            kind=mode, query=query.strip(), favorite=favorite, limit=limit,
            roots=self.catalog.roots if self.catalog else None,
        )
        paths = tuple(Path(row["path"]) for row in rows)
        with self.lock:
            self.inventories[mode] = set(paths)
            self.versions[mode] = max((row["metadata"]["_media"]["registered_ns"] for row in rows), default=0)
        return AssetInventory(mode, paths, tuple(rows), total)

    def paths(self, mode, query="", favorite=False):
        return set(self.inventory(mode, query, favorite).paths)

    def cached_paths(self, mode):
        with self.lock:
            return set(self.inventories.get(mode, ()))

    def rebuild(self):
        """Only this explicit action scans output folders and prepares old media."""
        if self.catalog is None:
            raise RuntimeError("Media catalog is not configured.")
        seen = []
        started_ns = time_ns()
        count = 0
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="h3-history") as workers:
            for mode in ("Video", "Image", "Audio"):
                yield f"Scanning historical {mode.lower()} files…"
                sources = self.list_paths(mode)
                for offset in range(0, len(sources), 48):
                    futures = [workers.submit(copy_context().run, self._prepare_historical, source)
                               for source in sources[offset:offset + 48]]
                    records = [record for future in futures if (record := future.result()) is not None]
                    self.store.index_assets(records)
                    seen.extend(str(record[0]) for record in records)
                    count += len(records)
                    yield f"Prepared thumbnails and metadata for {count} historical files."
        self.store.reconcile_catalog(seen, self.catalog.roots, started_ns=started_ns)
        yield f"Historical scan complete: {count} files. Tags and favorites are preserved."

    def _prepare_historical(self, source):
        try:
            return self.catalog.prepare(source, read_snapshot(source) or {})
        except FileNotFoundError:
            # A file removed during the scan will be reconciled as missing.
            return None
