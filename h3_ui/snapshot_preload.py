"""Load a workflow's default models before a Modal memory snapshot, then wait to serve.

Modal's snapshot-enabled endpoints start Prism Studio with ``H3_SNAPSHOT_PRELOAD``
set. The models load before the workspace is opened, so the snapshot holds them on
the GPU but no workspace database, owner secret or job state. The process then
reports readiness through ``H3_SNAPSHOT_READY_FILE`` and blocks until a line
arrives on stdin, which Modal sends after each restore; only then does the UI start
against the container's current volume.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PRELOAD_ENV = "H3_SNAPSHOT_PRELOAD"
READY_FILE_ENV = "H3_SNAPSHOT_READY_FILE"
# The warmup clip is deleted afterwards; it only has to run the default workflow.
VIDEO_WARMUP_PROMPT = "A calm ocean at sunrise, waves rolling onto a sandy beach, soft wind and distant gulls."


def _log(message: str) -> None:
    print(f"[snapshot-preload] {message}", flush=True)


def _remove_result(path: Path) -> None:
    from h3_app.provenance import score_path, snapshot_path

    for item in (path, snapshot_path(path), score_path(path)):
        item.unlink(missing_ok=True)


def preload_video(app) -> None:
    """Run the MiniMax H3 tab's default Video request once, so ComfyUI keeps its models."""
    from h3_app.progress import no_progress
    from h3_ui.h3.handlers import default_request

    results, last_status, started = [], "", time.monotonic()
    try:
        for output, status in app.generate(**default_request(VIDEO_WARMUP_PROMPT), progress=no_progress):
            if isinstance(output, str):
                results.append(Path(output))
            if status != last_status:
                _log(status.splitlines()[0] if status else "")
                last_status = status
    finally:
        for path in results:
            _remove_result(path)
    if not results:
        raise RuntimeError(f"The default H3 video warmup returned no result: {last_status}")
    _log(f"Default H3 video workflow ran in {time.monotonic() - started:.0f}s; its models stay loaded")


def preload_soundstage(app) -> None:
    """Load what Soundstage's Auto uses by default: Qwen VoiceDesign, Stable Audio 3, then SGLang-Omni."""
    from h3_app import audio_tools
    from h3_app.drama3.serving import server
    from h3_app.drama3.voice_design import ENGINE_QWEN
    from h3_app.prompt_writers.drama3 import DRAMA3_SCRIPT_ADVANCED

    status = ""
    for status in app.drama3_handlers.preload_drama3_models(ENGINE_QWEN, DRAMA3_SCRIPT_ADVANCED):
        _log(status)
    # The preload leaves a model that did not load to its step; a snapshot must not.
    missing = sorted({"voice_design", "stable_audio"} - audio_tools.loaded_models())
    if not server.server_ready():
        missing.append("SGLang-Omni")
    if missing:
        raise RuntimeError(f"Soundstage preload did not load {', '.join(missing)}: {status}")


PRELOADERS = {"video": preload_video, "soundstage": preload_soundstage}


def run_before_serving(app=None) -> None:
    """Preload the requested workflow and wait for Modal's restore; no-op without ``H3_SNAPSHOT_PRELOAD``.

    ``app`` is the application module, whose generator and handlers do the loading.
    """
    target = os.environ.pop(PRELOAD_ENV, "").strip().lower()
    if not target:
        return
    if target not in PRELOADERS:
        raise SystemExit(f"{PRELOAD_ENV} must be one of {', '.join(PRELOADERS)}; got {target!r}")
    from h3_app.drama3.serving import server

    if app is None:
        from h3_ui import application as app

    # Volume commits run in background threads with a live Modal client; neither may
    # be captured. Modal commits the volume itself before taking the snapshot.
    commit_volume = os.environ.pop("H3_MODAL_COMMIT_VOLUME", None)
    server.SNAPSHOT_CAPTURE.set()
    try:
        _log(f"Loading the default {target} models")
        PRELOADERS[target](app)
    finally:
        server.SNAPSHOT_CAPTURE.clear()
        if commit_volume is not None:
            os.environ["H3_MODAL_COMMIT_VOLUME"] = commit_volume
    # Pooled ComfyUI connections would be reused after the restore; open fresh ones.
    app.HTTP.close()
    ready = os.environ.pop(READY_FILE_ENV, "")
    if ready:
        Path(ready).write_text(target, encoding="utf-8")
    _log("Models loaded; waiting for the snapshot restore to start the UI")
    if not sys.stdin.readline():
        raise SystemExit("Snapshot preload: stdin closed before the restore signal")
    _log("Restored; starting the UI")
