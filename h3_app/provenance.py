"""Versioned, credential-free execution snapshots next to generated media."""

from __future__ import annotations

from contextvars import ContextVar
from html import escape
from pathlib import Path
from typing import Any, Mapping
import json
import os
import uuid

RUN_CONTEXT: ContextVar[dict] = ContextVar("h3_run_context", default={})


def snapshot_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".settings.json")


def write_snapshot(path: str | Path, settings: Mapping[str, Any]) -> None:
    destination = snapshot_path(path)
    temporary = destination.with_name(destination.name + f".{uuid.uuid4().hex}.tmp")
    payload = {"schema_version": 1, "output": Path(path).name, **settings}
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(temporary, destination)
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
