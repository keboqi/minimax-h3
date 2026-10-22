"""yt-dlp video downloads into the Media library, with saved per-site cookies.

Downloads land in ``<outputs>/downloads`` and are registered like any other
output. Cookies live in the private workspace, which the server never serves:
one Netscape ``<domain>_cookies.txt`` per site, applied automatically to URLs
on that site.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any, Callable
import uuid

from h3_app.catalog import VIDEO_EXTENSIONS
from h3_app.errors import H3Error
from h3_app.jobs import check_cancelled

QUALITIES = ("Best available", "2160p", "1440p", "1080p", "720p", "480p", "360p", "Smallest")
DEFAULT_QUALITY = "1080p"

FORMATS = {
    "Best available": "bestvideo*+bestaudio/bestvideo+bestaudio/best",
    "Smallest": "worstvideo*+worstaudio/worst",
    **{
        quality: (f"bestvideo*[height<={quality[:-1]}]+bestaudio/bestvideo[height<={quality[:-1]}]+bestaudio/"
                  f"best[height<={quality[:-1]}]/best")
        for quality in ("2160p", "1440p", "1080p", "720p", "480p", "360p")
    },
}

MAX_TITLE_CHARS = 120


def workspace_dir(runtime) -> Path:
    """Private workspace folder, outside the served gallery outputs."""
    workspace = os.getenv("H3_WORKSPACE_DIR")
    base = Path(workspace) if workspace else Path(runtime.outputs_dir) / ".h3-workspace"
    return base.resolve()


def downloads_dir(runtime) -> Path:
    return Path(runtime.outputs_dir).resolve() / "downloads"


def cookies_dir(runtime) -> Path:
    return workspace_dir(runtime) / "cookies"


def clean_error(exc: BaseException) -> str:
    message = re.sub(r"\x1b\[[0-9;]*m", "", str(exc))
    return message.replace("ERROR:", "").strip()


def format_fallbacks(quality: str) -> list[str | None]:
    """Format selectors from most specific to safest; ``None`` lets yt-dlp choose."""
    candidates: list[str | None] = [FORMATS.get(quality, FORMATS["Best available"])]
    if quality.endswith("p") and quality[:-1].isdigit():
        height = quality[:-1]
        candidates += [f"bestvideo[height<={height}]+bestaudio/best[height<={height}]/best",
                       f"bestvideo*[height<={height}]/best[height<={height}]/best"]
    elif quality == "Smallest":
        candidates += ["worstvideo+worstaudio/worst", "worstvideo*/worst"]
    else:
        candidates += ["bestvideo+bestaudio/best", "bestvideo*/best"]
    candidates += ["best", None]
    return list(dict.fromkeys(candidates))


def is_format_unavailable(exc: BaseException) -> bool:
    return "requested format is not available" in clean_error(exc).lower()


def js_runtime_options() -> dict[str, Any]:
    """YouTube's challenge solver needs a JavaScript runtime; prefer Node, as the CLI does.

    ``YTDLP_JS_RUNTIMES`` (``node``, ``deno:/path/to/deno``, comma separated) overrides discovery.
    """
    configured = (os.getenv("YTDLP_JS_RUNTIMES") or "").strip()
    # Setup installs Deno from PyPI beside this interpreter, which an unactivated venv leaves off PATH.
    scripts = os.pathsep.join((str(Path(sys.executable).parent), os.environ.get("PATH", "")))
    found = {name: shutil.which(name, path=scripts) for name in ("node", "deno", "bun")}
    if configured:
        items = []
        for spec in configured.split(","):
            name, _, path = spec.strip().partition(":")
            if name.strip():
                items.append((name.strip(), path.strip() or found.get(name.strip())))
    elif found["node"]:
        items = [("node", found["node"])]
    else:
        items = list(found.items())
    runtimes = {name: {"path": path} for name, path in items if path}
    return {"js_runtimes": runtimes} if runtimes else {}


def setup_hint(message: str) -> str:
    lowered = message.lower()
    if "sign in" in lowered or "confirm you" in lowered or "cookies" in lowered:
        return (" The site asked this server to sign in: add cookies for it under "
                "**Cookies for sign-in downloads**, then try again.")
    if "challenge" in lowered or "n challenge" in lowered or "po token" in lowered:
        return (" YouTube needs yt-dlp's JavaScript challenge solver: install Node or Deno on the "
                "server and upgrade yt-dlp (`pip install -U \"yt-dlp[default]\"`).")
    return ""


# ---------------------------------------------------------------------------
# Cookies
# ---------------------------------------------------------------------------


def site_domain(url: str) -> str:
    """The registrable part of a URL's host (``www.youtube.com`` → ``youtube.com``)."""
    match = re.search(r"https?://(?:www\.)?([^/:?#\s'\"]+)", url or "")
    if not match:
        return ""
    host = match.group(1).lower()
    parts = host.split(".")
    domain = ".".join(parts[-2:]) if len(parts) > 2 else host
    return "youtube.com" if domain == "youtu.be" else domain


def normalize_domain(domain: str) -> str:
    value = (domain or "").strip().lower()
    value = re.sub(r"^https?://", "", value).split("/")[0]
    value = value.removeprefix("www.").strip(".")
    if not value:
        raise H3Error("Enter the site's domain, such as youtube.com.")
    if not re.fullmatch(r"[a-z0-9.-]+", value) or "." not in value:
        raise H3Error(f"{domain!r} is not a valid domain.")
    return value


def parse_curl_cookies(curl_text: str) -> dict[str, str]:
    """Cookie pairs from a "Copy as cURL" command (``-H 'Cookie: …'``, ``-b`` or ``--cookie``)."""
    cookies: dict[str, str] = {}
    patterns = (
        r"(?:-H|--header)\s+\$?['\"]cookie:\s*([^'\"]+)['\"]",
        r"(?:-b|--cookie)\s+\$?['\"]([^'\"]+)['\"]",
    )
    for pattern in patterns:
        for match in re.findall(pattern, curl_text or "", re.IGNORECASE):
            for pair in match.split(";"):
                name, sep, value = pair.strip().partition("=")
                if sep and name.strip():
                    cookies[name.strip()] = value.strip()
    return cookies


def curl_url(curl_text: str) -> str:
    match = re.search(r"https?://[^\s'\"^]+", curl_text or "")
    return match.group(0) if match else ""


def netscape_cookies(cookies: dict[str, str], domain: str) -> str:
    """A Netscape cookies.txt that applies ``cookies`` to ``domain`` and its subdomains."""
    expires = int(time.time()) + 365 * 24 * 3600
    lines = ["# Netscape HTTP Cookie File", "# Imported from a cURL command by Prism Studio", ""]
    for name, value in cookies.items():
        lines.append("\t".join((f".{domain}", "TRUE", "/", "TRUE", str(expires), name, value)))
    return "\n".join(lines) + "\n"


def _cookie_lines(text: str) -> list[str]:
    return [line for line in text.splitlines()
            if line.strip() and (not line.startswith("#") or line.startswith("#HttpOnly_"))]


class CookieStore:
    """Saved ``<domain>_cookies.txt`` files in one private directory."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def path_for(self, domain: str) -> Path:
        return self.directory / (normalize_domain(domain).replace(".", "_") + "_cookies.txt")

    def sites(self) -> dict[str, int]:
        """Saved sites and how many cookies each holds."""
        if not self.directory.is_dir():
            return {}
        sites = {}
        for path in sorted(self.directory.glob("*_cookies.txt")):
            domain = path.name.removesuffix("_cookies.txt").replace("_", ".")
            try:
                sites[domain] = len(_cookie_lines(path.read_text(encoding="utf-8", errors="replace")))
            except OSError:
                sites[domain] = 0
        return sites

    def save(self, domain: str, text: str) -> tuple[str, int]:
        domain = normalize_domain(domain)
        count = len(_cookie_lines(text))
        if not count:
            raise H3Error("No cookies found. Export a Netscape cookies.txt or copy a request as cURL.")
        target = self.path_for(domain)
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + f".{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return domain, count

    def delete(self, domain: str) -> bool:
        target = self.path_for(domain)
        if not target.is_file():
            return False
        target.unlink()
        return True

    def file_for_url(self, url: str) -> Path | None:
        """The cookie file for a URL's site, falling back to a related saved domain."""
        domain = site_domain(url)
        if not domain:
            return None
        exact = self.path_for(domain)
        if exact.is_file():
            return exact
        for saved in self.sites():
            if saved.endswith("." + domain) or domain.endswith("." + saved):
                return self.path_for(saved)
        return None


def import_curl(store: CookieStore, curl_text: str, domain: str = "") -> tuple[str, int]:
    cookies = parse_curl_cookies(curl_text)
    if not cookies:
        raise H3Error("That cURL command has no Cookie header. In DevTools use Copy → Copy as cURL (bash).")
    domain = normalize_domain(domain or site_domain(curl_url(curl_text)))
    return store.save(domain, netscape_cookies(cookies, domain))


def import_cookie_file(store: CookieStore, path: str | Path, domain: str) -> tuple[str, int]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    if "\t" not in text:
        raise H3Error("That file is not a Netscape cookies.txt export (tab-separated lines).")
    return store.save(domain, text)


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------


def _youtube_dl():
    try:
        import yt_dlp
    except ImportError as exc:
        raise H3Error("yt-dlp is not installed. Rerun setup, or `pip install -U \"yt-dlp[default]\"`.") from exc
    return yt_dlp


def _base_options(url: str, cookies: CookieStore) -> dict[str, Any]:
    options: dict[str, Any] = {"quiet": True, "no_warnings": True, "noplaylist": True,
                               "noprogress": True, **js_runtime_options()}
    cookie_file = cookies.file_for_url(url)
    if cookie_file is not None:
        options["cookiefile"] = str(cookie_file)
    return options


def _validate_url(url: str) -> str:
    url = str(url or "").strip()
    if not re.match(r"https?://\S+$", url):
        raise H3Error("Enter a video page URL starting with http:// or https://.")
    return url


def format_duration(seconds: float | None) -> str:
    if not seconds or seconds <= 0:
        return "unknown length"
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def video_info(url: str, runtime) -> str:
    """A Markdown summary of the page's video: title, uploader, length and best heights."""
    url = _validate_url(url)
    yt_dlp = _youtube_dl()
    store = CookieStore(cookies_dir(runtime))
    try:
        with yt_dlp.YoutubeDL({**_base_options(url, store), "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        message = clean_error(exc)
        raise H3Error(f"Could not read that page: {message}.{setup_hint(message)}") from exc
    heights = sorted({int(f.get("height") or 0) for f in info.get("formats") or [] if f.get("height")},
                     reverse=True)
    lines = [f"**{info.get('title') or 'Untitled'}**",
             f"{info.get('uploader') or info.get('channel') or 'Unknown uploader'} · "
             f"{format_duration(info.get('duration'))}"]
    if heights:
        lines.append("Available: " + ", ".join(f"{height}p" for height in heights[:8]))
    if store.file_for_url(url) is not None:
        lines.append(f"Using saved cookies for {site_domain(url)}.")
    return "\n\n".join(lines)


def _safe_title(info: dict) -> str:
    title = str(info.get("title") or info.get("id") or "video")
    title = re.sub(r"[\\/:*?\"<>|\x00-\x1f]+", " ", title)
    title = re.sub(r"\s+", " ", title).strip(" .")[:MAX_TITLE_CHARS].strip(" .")
    return title or "video"


def _progress_message(status: dict) -> str | None:
    if status.get("status") != "downloading":
        return "Finishing the download" if status.get("status") == "finished" else None
    done = status.get("downloaded_bytes") or 0
    total = status.get("total_bytes") or status.get("total_bytes_estimate") or 0
    parts = [f"{done / total:.0%}" if total else f"{done / 2**20:.1f} MiB"]
    if status.get("speed"):
        parts.append(f"{status['speed'] / 2**20:.1f} MiB/s")
    if status.get("eta") is not None:
        parts.append(f"ETA {int(status['eta'])}s")
    return "Downloading " + " · ".join(parts)


def download_video(url: str, quality: str, runtime, *,
                   report: Callable[[str], None] = lambda _message: None,
                   stopped: Callable[[], bool] = lambda: False) -> Path:
    """Download one video into the Media library and return its path.

    The download runs in a hidden staging folder and is published only when
    complete, so an interrupted download never appears in the library.
    ``stopped`` returning True interrupts it at the next progress update.
    """
    url = _validate_url(url)
    yt_dlp = _youtube_dl()
    store = CookieStore(cookies_dir(runtime))
    target_dir = downloads_dir(runtime)
    staging = target_dir / ".processing" / uuid.uuid4().hex
    staging.mkdir(parents=True, exist_ok=True)
    interrupted = []

    def hook(status: dict) -> None:
        try:
            check_cancelled()
            if stopped():
                raise H3Error("Download interrupted.")
        except BaseException as exc:
            interrupted.append(exc)
            raise
        message = _progress_message(status)
        if message:
            report(message)

    try:
        info = None
        last_error: Exception | None = None
        for selector in format_fallbacks(quality):
            options = {**_base_options(url, store), "outtmpl": str(staging / "%(id)s.%(ext)s"),
                       "merge_output_format": "mp4", "progress_hooks": [hook], "restrictfilenames": True}
            if selector:
                options["format"] = selector
            try:
                with yt_dlp.YoutubeDL(options) as ydl:
                    info = ydl.extract_info(url, download=True)
                break
            except Exception as exc:
                if interrupted:
                    raise interrupted[0] from None
                last_error = exc
                if not is_format_unavailable(exc):
                    break
                report("That quality is unavailable; trying a fallback format")
        if info is None:
            message = clean_error(last_error) if last_error else "no video was returned"
            raise H3Error(f"Download failed: {message}.{setup_hint(message)}")
        if info.get("_type") == "playlist":
            raise H3Error("That URL is a playlist. Open one video and copy its URL.")
        files = [path for path in staging.iterdir()
                 if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS | {".m4v"}]
        if not files:
            raise H3Error("yt-dlp finished without a video file (the page may only offer audio or images).")
        source = max(files, key=lambda path: path.stat().st_size)
        suffix = ".mp4" if source.suffix.lower() == ".m4v" else source.suffix.lower()
        stem = f"{_safe_title(info)} [{re.sub(r'[^A-Za-z0-9_-]', '', str(info.get('id') or ''))[:32] or uuid.uuid4().hex[:8]}]"
        target = target_dir / f"{stem}{suffix}"
        for index in range(2, 1000):
            if not target.exists():
                break
            target = target_dir / f"{stem} ({index}){suffix}"
        os.replace(source, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        try:
            staging.parent.rmdir()  # only when no other download is staging
        except OSError:
            pass
    from h3_app.provenance import write_snapshot

    write_snapshot(target, {"family": "Downloaded", "settings": {
        "site": site_domain(url), "title": str(info.get("title") or ""), "video_id": str(info.get("id") or ""),
        "uploader": str(info.get("uploader") or info.get("channel") or ""),
        "duration": info.get("duration"), "quality": quality,
        "height": info.get("height"), "fps": info.get("fps"),
    }})
    return target
