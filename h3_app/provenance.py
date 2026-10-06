"""Versioned, credential-free execution snapshots next to generated media."""

from __future__ import annotations

import json
import os
import uuid
from contextvars import ContextVar
from html import escape
from pathlib import Path
from typing import Any, Mapping

RUN_CONTEXT: ContextVar[dict] = ContextVar("h3_run_context", default={})


def snapshot_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".settings.json")


def write_snapshot(path: str | Path, settings: Mapping[str, Any]) -> None:
    from .jobs import CURRENT_JOB

    destination = snapshot_path(path)
    temporary = destination.with_name(destination.name + f".{uuid.uuid4().hex}.tmp")
    payload = {"schema_version": 1, "output": Path(path).name, **settings}
    payload["asset_id"] = uuid.uuid5(
        uuid.NAMESPACE_URL, Path(path).resolve().as_uri()
    ).hex
    job = CURRENT_JOB.get()
    if job is not None:
        payload.update(
            application_job_id=job.id, variant=job.variant, retry_of=job.retry_of
        )
        if job.source_asset_ids:
            payload["source_asset_ids"] = job.source_asset_ids
        if job.canvas_request:
            payload["image_canvas"] = job.canvas_request
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(temporary, destination)
        from .media_catalog import register_created_media
        register_created_media(path, payload)
    finally:
        temporary.unlink(missing_ok=True)


def read_snapshot(path: str | Path | None) -> dict | None:
    if not path:
        return None
    try:
        payload = json.loads(snapshot_path(path).read_text(encoding="utf-8"))
        return payload if payload.get("schema_version") == 1 else None
    except (OSError, ValueError, AttributeError):
        return None


def render_snapshot(paths: Any) -> str:
    if not isinstance(paths, (tuple, list)):
        paths = [paths]
    sections = []
    for path in paths:
        if isinstance(path, dict):
            path = path.get("path") or path.get("name")
        if not path:
            continue
        payload = read_snapshot(path)
        content = (
            escape(json.dumps(payload, indent=2, ensure_ascii=False))
            if payload
            else "Settings unavailable for this output."
        )
        sections.append(
            f"<details><summary>Settings used · {escape(Path(path).name)}</summary><pre>{content}</pre></details>"
        )
    return "".join(sections) or "Settings used will appear with the generated result."


def copy_snapshot(source: Path, destination: Path) -> None:
    """Retain settings while recording the copied output's actual name."""
    payload = read_snapshot(source) or {}
    write_snapshot(
        destination,
        {
            **{
                key: value
                for key, value in payload.items()
                if key not in {"output", "asset_id"}
            },
            "source_asset_ids": [
                payload.get("asset_id")
                or uuid.uuid5(uuid.NAMESPACE_URL, source.resolve().as_uri()).hex
            ],
        },
    )


def copy_media(source: Path, destination: Path) -> None:
    """Publish a complete managed copy and preserve its execution settings."""
    import shutil

    from .jobs import check_cancelled

    temporary = destination.with_name(destination.name + f".{uuid.uuid4().hex}.partial")
    try:
        shutil.copy2(source, temporary)
        check_cancelled()
        temporary.replace(destination)
        copy_snapshot(source, destination)
    finally:
        temporary.unlink(missing_ok=True)
