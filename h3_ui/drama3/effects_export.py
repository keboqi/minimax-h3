"""Capture the shared Soundstage stage on the export's frame clock."""

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
from urllib.parse import urlsplit

from h3_app.drama3.visuals import FPS
from h3_app.errors import H3Error
from .effect_depth import DEPTH_FOLDER
from .effect_regions import MASK_FOLDER

PAGE = Path(__file__).with_name("page").resolve()
ORIGIN = "https://soundstage-export.invalid"
BATCH_FRAMES = 240


@lru_cache(maxsize=1)
def _renderer_revision():
    digest = sha256()
    # Use the renderer's served asset tree so new modules and textures are included automatically.
    for path in sorted(PAGE.rglob("*")):
        if path.is_file() and path.resolve().is_relative_to(PAGE):
            digest.update(path.relative_to(PAGE).as_posix().encode() + b"\0")
            digest.update(sha256(path.read_bytes()).digest())
    return digest.hexdigest()


@dataclass(frozen=True)
class EffectScene:
    stills_revision: str
    shots: list[dict]
    sounds: list[dict]

    @property
    def revision(self):
        shots = [{key: value for key, value in shot.items() if key not in ("url", "depth_url")} for shot in self.shots]
        data = [shots, self.sounds, _renderer_revision()]
        return sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


_WEBGPU_ARGS = [
    "--enable-unsafe-webgpu",
    "--use-webgpu-adapter=swiftshader",
    "--enable-unsafe-swiftshader",
]


def _launch(chromium):
    from playwright.sync_api import Error

    primary_error = None
    attempts = (
        {"args": _WEBGPU_ARGS},
        {},
        {"channel": "chrome", "args": _WEBGPU_ARGS},
        {"channel": "chrome"},
        {"channel": "msedge", "args": _WEBGPU_ARGS},
        {"channel": "msedge"},
    )
    for options in attempts:
        try:
            return chromium.launch(**options)
        except Error as exc:
            if not options or primary_error is None:
                primary_error = exc
            continue
    raise H3Error("Install Chromium on the server for visual effects export with "
                  "`python -m playwright install chromium`. "
                  f"{str(primary_error).splitlines()[0]}") from primary_error


def capture_frames(folder, shots, sounds, images, size, total_frames, progress=None):
    """Yield bounded JPEG streams at 24 fps, including held shots and dissolves."""
    try:
        from playwright.sync_api import Error, sync_playwright
    except ImportError as exc:
        raise H3Error("Install Playwright and Chromium on the server to export visual effects.") from exc

    urls = {image: f"/stills/{i}" for i, image in enumerate(dict.fromkeys(images))}
    images_by_url = {url: image for image, url in urls.items()}
    masks_by_url = {}
    depths_by_url = {}
    timeline = []
    for shot, image in zip(shots, images, strict=True):
        entry = {**shot, "url": urls[image]}
        if depth_file := shot.get("depth_file"):
            depth_url = f"/depth/{depth_file}"
            depths_by_url[depth_url] = Path(image).parent / DEPTH_FOLDER / depth_file
            entry["depth_url"] = depth_url
        else:
            entry.pop("depth_url", None)
        if "effect_layers" in shot:
            entry["effect_layers"] = []
            for layer in shot["effect_layers"]:
                if files := layer.get("mask_files"):
                    paths = []
                    for mask in files:
                        path = f"/masks/{mask}"
                        masks_by_url[path] = Path(image).parent / MASK_FOLDER / mask
                        paths.append(path)
                    layer = {**layer, "mask_urls": paths}
                elif mask := layer.get("mask_file"):
                    path = f"/masks/{mask}"
                    masks_by_url[path] = Path(image).parent / MASK_FOLDER / mask
                    layer = {**layer, "mask_url": path}
                entry["effect_layers"].append(layer)
        timeline.append(entry)

    def serve(route):
        url = route.request.url
        if not url.startswith(f"{ORIGIN}/"):
            route.abort()
            return
        path = urlsplit(url).path
        if image := images_by_url.get(path):
            route.fulfill(path=image, content_type="image/png" if image.suffix == ".png" else "image/jpeg")
        elif mask := masks_by_url.get(path):
            if mask.is_file():
                route.fulfill(path=mask, content_type="image/png")
            else:
                route.abort()
        elif depth := depths_by_url.get(path):
            if depth.is_file():
                route.fulfill(path=depth, content_type="image/png")
            else:
                route.abort()
        else:
            asset = (PAGE / path.lstrip("/")).resolve()
            if asset.is_relative_to(PAGE) and asset.is_file():
                route.fulfill(path=asset, content_type="application/javascript" if asset.suffix == ".js" else None)
            else:
                route.abort()

    try:
        with sync_playwright() as playwright:
            browser = _launch(playwright.chromium)
            try:
                width, height = size
                page = browser.new_page(viewport={"width": width, "height": height}, reduced_motion="no-preference")
                page.route("**/*", serve)
                page.goto(f"{ORIGIN}/export_stage.html")
                page.evaluate("scene => window.prepareStage(scene)", {"shots": timeline, "sounds": sounds})
                stream = folder / "stage-frames.mjpeg"
                for start in range(0, total_frames, BATCH_FRAMES):
                    end = min(total_frames, start + BATCH_FRAMES)
                    if progress:
                        progress(start, total_frames)
                    with stream.open("wb") as frames:
                        for frame in range(start, end):
                            page.evaluate("time => window.paintFrame(time)", frame / FPS)
                            frames.write(page.screenshot(type="jpeg", quality=95))
                    yield stream, end - start
            finally:
                browser.close()
    except Error as exc:
        raise H3Error(f"Visual effects rendering failed: {str(exc).splitlines()[0]}") from exc
