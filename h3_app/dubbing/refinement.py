"""Optional Desert Ant Align refinement; retain acoustic inputs for real-case comparisons."""

from __future__ import annotations

import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

from h3_app.errors import H3Error
from h3_app.processes import run_media_process

from .tools_worker import aligned_line_bounds

SDK_VERSION = "4.2.1"
REVISION = f"desert-ant-align-{SDK_VERSION}-v1"
LANGUAGES = {"english": "en", "chinese": "zh", "japanese": "ja", "korean": "ko", "french": "fr",
             "german": "de", "spanish": "es", "portuguese": "pt", "italian": "it"}
RATE = 16000
MAX_SHIFT = 1.2


def language_code(language: str) -> str:
    value = str(language or "").strip().lower()
    code = LANGUAGES.get(value, value.split("-")[0])
    return code if code in LANGUAGES.values() else ""


def check_runtime() -> tuple[str, str]:
    """The published Node native SDK has Linux and Apple-silicon builds, but no Windows build."""
    machine = platform.machine().lower()
    if not ((platform.system() == "Linux" and machine in {"x86_64", "aarch64", "arm64"})
            or (platform.system() == "Darwin" and machine == "arm64")):
        raise H3Error("Align's Node runtime requires Linux x64/arm64 or Apple silicon; use Linux/WSL here.")
    node, npm = shutil.which("node"), shutil.which("npm")
    if not node or not npm:
        raise H3Error("Install Node.js 22 or newer with npm to enable Align timestamp refinement.")
    return node, npm


def _run(command, *, timeout):
    try:
        result = run_media_process(command, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        raise H3Error(f"Align could not run: {exc}") from exc
    if result.returncode:
        raise H3Error(f"Align failed: {result.stderr[-1500:].strip()}")
    return result


def refine(audio: Path, groups: list[dict], root: Path, *, timeout: float, report=lambda _message: None) -> list[dict]:
    """One Node process per analysis; process audio windows rather than loading the whole recording in JS."""
    node, npm = check_runtime()
    version = _run([node, "--version"], timeout=15).stdout.strip().lstrip("v")
    if int(version.split(".")[0]) < 22:
        raise H3Error("Align requires Node.js 22 or newer.")
    runtime = root / "align-runtime"
    package = runtime / "node_modules" / "@desert-ant-labs" / "align" / "package.json"
    ready = False
    try:
        ready = json.loads(package.read_text(encoding="utf-8"))["version"] == SDK_VERSION
    except (OSError, ValueError, KeyError):
        pass
    if not ready:
        report("Installing the optional Align runtime (first use only)")
        runtime.mkdir(parents=True, exist_ok=True)
        _run([npm, "install", "--prefix", str(runtime), "--no-audit", "--no-fund", "--save-exact",
              f"@desert-ant-labs/align@{SDK_VERSION}", f"@desert-ant-labs/core@{SDK_VERSION}"], timeout=600)
    report("Refining acoustic word timestamps with Align")
    with tempfile.TemporaryDirectory(prefix="align-", dir=root) as directory:
        folder = Path(directory)
        pcm, job, result = folder / "audio.f32", folder / "job.json", folder / "result.json"
        _run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(audio),
              "-ar", str(RATE), "-ac", "1", "-f", "f32le", str(pcm)], timeout=timeout)
        job.write_text(json.dumps({"audio": str(pcm), "groups": groups, "output": str(result),
                                   "module": str(package.with_name("node.js")),
                                   "cache": str(root / "align-models")}, ensure_ascii=False), encoding="utf-8")
        _run([node, str(Path(__file__).with_name("refine_timestamps.mjs")), str(job)], timeout=timeout)
        from h3_app.drama3.serving.server import persist_caches

        persist_caches()
        try:
            return json.loads(result.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise H3Error("Align returned an unreadable result.") from exc


def apply_group(group: dict, result: dict, duration: float) -> tuple[list[dict], dict]:
    """Accept only matching, finite acoustic word times and map them back to the unchanged line text."""
    originals = group.get("words") or []
    if not isinstance(result, dict):
        raise ValueError("Align returned an invalid word-timing result")
    words = result.get("words") or []
    if result.get("error"):
        raise ValueError(result["error"])
    if not isinstance(words, list) or not originals or len(words) != len(originals):
        raise ValueError("Align returned a different word count or no acoustic word timing")
    for before, after in zip(originals, words):
        if not isinstance(after, dict):
            raise ValueError("Align returned an invalid word")
        if before["text"] != after.get("text"):
            raise ValueError("Align changed the transcript")
        for field in ("start", "end"):
            value = float(after[field])
            if not math.isfinite(value) or value < 0 or value > duration:
                raise ValueError("Align returned a timestamp outside the recording")
            if abs(value - float(before[field])) > MAX_SHIFT:
                raise ValueError("Align moved a word beyond its search window")
    bounds = aligned_line_bounds([line["text"] for line in group["lines"]], words)
    lines = [{**line, "start": round(start, 3), "end": round(end, 3)}
             for line, (start, end) in zip(group["lines"], bounds)]
    if any(line["end"] <= line["start"] for line in lines):
        raise ValueError("Align collapsed a line's duration")
    changed = any(abs(float(after[field]) - float(before[field])) >= 0.0005
                  for before, after in zip(originals, words) for field in ("start", "end"))
    return lines, {"original_words": originals, "words": words, "status": "refined" if changed else "unchanged"}
