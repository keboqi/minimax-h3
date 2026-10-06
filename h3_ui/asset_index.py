"""Incremental technical metadata indexing for the managed Media library."""

from pathlib import Path
from threading import RLock
from time import monotonic

from h3_app.provenance import read_snapshot, snapshot_path
from h3_app.gallery_store import AssetInventory


def technical_metadata(value):
    if isinstance(value, dict):
        return {
            key: technical_metadata(item)
            for key, item in value.items()
            if key == "prompt_id"
            or not any(
                word in key.casefold()
                for word in (
                    "prompt",
                    "caption",
                    "lyrics",
                    "api_key",
                    "secret",
                    "token",
                    "abc",
                )
            )
        }
    if isinstance(value, list):
        return [technical_metadata(item) for item in value]
    return value


class AssetIndex:
    def __init__(self, store, list_paths):
        self.store = store
        self.list_paths = list_paths
        self.fingerprints = {}
        self.inventories = {}
        self.ordered = {}
        self.revision = 0
        self.lock = RLock()
        self.scanned_at = {}

    def sync(self, mode):
        with self.lock:
            allowed = set()
            ordered = []
            changed = {}
            assets = []
            for source in self.list_paths(mode):
                path = Path(source).resolve()
                try:
                    stat = path.stat()
                    sidecar = snapshot_path(path)
                    metadata_stat = sidecar.stat() if sidecar.is_file() else None
                    fingerprint = (
                        stat.st_mtime_ns,
                        stat.st_size,
                        metadata_stat.st_mtime_ns if metadata_stat else None,
                    )
                    allowed.add(path)
                    ordered.append(path)
                    if self.fingerprints.get(path) != fingerprint:
                        assets.append(
                            (
                                path,
                                mode,
                                technical_metadata(read_snapshot(path) or {})
                                if metadata_stat
                                else {},
                            )
                        )
                        changed[path] = fingerprint
                except (OSError, ValueError):
                    continue
            indexed = self.store.index_assets(assets)
            self.fingerprints.update({path: changed[path] for path in indexed})
            if indexed or self.inventories.get(mode) != allowed:
                self.revision += 1
            self.inventories[mode] = allowed
            self.ordered[mode] = ordered
            self.scanned_at[mode] = monotonic()
            return allowed

    def cached_paths(self, mode):
        """Reuse a completed scan for comparison choices; actions validate live paths."""
        with self.lock:
            return set(self.inventories.get(mode, ()))

    def paths(self, mode, query="", favorite=False, *, force=True):
        with self.lock:
            allowed = (
                self.sync(mode)
                if force or monotonic() - self.scanned_at.get(mode, float("-inf")) >= 5
                else set(self.inventories[mode])
            )
        if not query.strip() and not favorite:
            return allowed
        rows = self.store.search_assets(
            query=query.strip(), kind=mode, favorite=favorite, limit=None
        )
        return {Path(row["path"]).resolve() for row in rows} & allowed

    def inventory(self, mode, query="", favorite=False, *, force=True):
        with self.lock:
            matching = self.paths(mode, query, favorite, force=force)
            return AssetInventory(
                mode, tuple(path for path in self.ordered[mode] if path in matching)
            )

    def rebuild(self):
        with self.lock:
            self.fingerprints.clear()
            self.inventories.clear()
            self.ordered.clear()
            self.scanned_at.clear()
            self.revision += 1
            with self.store.connect() as db:
                db.execute("UPDATE assets SET available=0")
            total = 0
            for mode in ("Video", "Image", "Audio"):
                total += len(self.sync(mode))
                yield f"Indexed {total} files. Tags and favorites are preserved."
