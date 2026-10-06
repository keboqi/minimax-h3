"""Publish media records and previews at creation; scan history only on request."""

from pathlib import Path
import time

from .catalog import AUDIO_EXTENSIONS, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from . import gallery_store

MEDIA_EXTENSIONS = {
    "Image": IMAGE_EXTENSIONS,
    "Video": VIDEO_EXTENSIONS,
    "Audio": AUDIO_EXTENSIONS,
}


def technical_metadata(value):
    if isinstance(value, dict):
        return {
            key: technical_metadata(item)
            for key, item in value.items()
            if key == "prompt_id" or not any(
                word in key.casefold()
                for word in ("prompt", "caption", "lyrics", "api_key", "secret", "token", "abc")
            )
        }
    if isinstance(value, list):
        return [technical_metadata(item) for item in value]
    return value


class MediaCatalog:
    def __init__(self, store, runtime):
        self.store = store
        self.runtime = runtime
        config = runtime()
        self.roots = (config.output_dir.resolve(), config.outputs_dir.resolve())

    def prepare(self, source, metadata):
        runtime = self.runtime()
        path = Path(source).resolve()
        kind = next((name for name, extensions in MEDIA_EXTENSIONS.items() if path.suffix.lower() in extensions), None)
        if (
            kind is None
            or ".processing" in path.parts
            or path.is_relative_to(runtime.gallery_thumbnails_dir.resolve())
            or not any(path.is_relative_to(root) for root in self.roots)
        ):
            return None
        info = path.stat()
        create = {
            "Image": gallery_store.gallery_image_thumbnail,
            "Video": gallery_store.gallery_thumbnail,
            "Audio": gallery_store.gallery_audio_thumbnail,
        }[kind]
        thumbnail = create(path, runtime=runtime)
        available = thumbnail is not None
        if thumbnail is None:
            thumbnail = gallery_store.gallery_placeholder(path, kind=kind, runtime=runtime)
        family = metadata.get("family") or {
            "Image": gallery_store.generated_image_family,
            "Video": gallery_store.generated_video_family,
            "Audio": gallery_store.generated_audio_family,
        }[kind](path, runtime=runtime)
        timestamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(info.st_mtime))
        details = technical_metadata(metadata)
        details["_media"] = {
            "version": 1, "size": info.st_size, "mtime_ns": info.st_mtime_ns,
            "registered_ns": time.time_ns(),
            "thumbnail": thumbnail.name if thumbnail else None,
            "preview_available": available,
            "caption": f"{family} · {path.name} · {timestamp} · {info.st_size / (1024 * 1024):.1f} MB",
        }
        return path, kind, details

    def register(self, source, metadata):
        record = self.prepare(source, metadata)
        if record:
            self.store.index_assets([record])


_CATALOG = None


def configure_media_catalog(store, runtime):
    global _CATALOG
    _CATALOG = MediaCatalog(store, runtime)
    return _CATALOG


def get_media_catalog():
    return _CATALOG


def register_created_media(source, metadata):
    if _CATALOG is not None:
        _CATALOG.register(source, metadata)
