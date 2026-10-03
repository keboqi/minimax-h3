"""Bounded local metadata for uploaded reference cards."""

from functools import lru_cache
import json
from pathlib import Path
import shutil
import subprocess

from PIL import Image


@lru_cache(maxsize=128)
def _inspect(path, kind, modified, size):
    source = Path(path)
    if kind == "Picture":
        with Image.open(source) as image:
            return f"{image.width}×{image.height} · {size / 1024:.0f} KB"
    executable = shutil.which("ffprobe")
    if not executable:
        return f"{size / 1024**2:.1f} MB · duration unavailable"
    command = [
        executable,
        "-v",
        "error",
        "-show_entries",
        "stream=width,height:format=duration",
        "-of",
        "json",
        str(source),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=2,
        creationflags=(
            subprocess.CREATE_NO_WINDOW
            if hasattr(subprocess, "CREATE_NO_WINDOW")
            else 0
        ),
    )
    data = json.loads(result.stdout)
    duration = data.get("format", {}).get("duration")
    dimensions = next(
        (
            f"{stream['width']}×{stream['height']}"
            for stream in data.get("streams", [])
            if stream.get("width")
        ),
        "",
    )
    return " · ".join(
        part
        for part in (
            dimensions,
            f"{float(duration):.1f}s" if duration else "",
            f"{size / 1024**2:.1f} MB",
        )
        if part
    )


def describe_reference(path, kind):
    source = Path(path)
    try:
        stat = source.stat()
        detail = _inspect(str(source), kind, stat.st_mtime_ns, stat.st_size)
    except (OSError, ValueError, subprocess.SubprocessError):
        detail = "Metadata unavailable"
    return f"{source.name} · {detail}"
