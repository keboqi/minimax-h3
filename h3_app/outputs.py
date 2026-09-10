"""Extracted outputs boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from h3_app import media as media_store
from h3_app.catalog import AUDIO_EXTENSIONS, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from h3_app.errors import H3Error
from h3_app.policy import validate_image_frame_count

from .config import RuntimeConfig


@dataclass(frozen=True)
class OutputContext:
    config: RuntimeConfig
    output_token: str | None = None


def walk_saved_refs(value):
    yield from media_store.walk_saved_refs(value)


def _history_output_candidates(
    history, extensions, *, directory=None, context: OutputContext
):
    return media_store.history_output_candidates(
        context.config.output_dir, history, extensions, directory=directory
    )


def _recent_output_candidates(
    directory, extensions, queued_at, *, context: OutputContext
):
    return media_store.recent_output_candidates(
        directory, extensions, queued_at, output_token=context.output_token
    )


def resolve_output(
    history: dict[str, Any], queued_at: float, *, context: OutputContext
) -> Path:
    videos = _history_output_candidates(history, VIDEO_EXTENSIONS, context=context)
    if videos:
        return max(videos, key=lambda p: p.stat().st_mtime)

    # Newer SaveVideo UI payloads can be omitted from the public history shape.
    # Fall back only to files created after this job was queued.
    recent = _recent_output_candidates(
        context.config.output_dir, VIDEO_EXTENSIONS, queued_at, context=context
    )
    if recent:
        return max(recent, key=lambda p: p.stat().st_mtime)
    raise H3Error("Generation completed, but no saved video could be located.")


def resolve_audio_output(
    history: dict[str, Any], queued_at: float, *, context: OutputContext
) -> Path:
    candidates = _history_output_candidates(history, AUDIO_EXTENSIONS, context=context)
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)
    recent = _recent_output_candidates(
        context.config.output_dir, AUDIO_EXTENSIONS, queued_at, context=context
    )
    if recent:
        return max(recent, key=lambda path: path.stat().st_mtime)
    raise H3Error("Generation completed, but no saved audio could be located.")


def resolve_image_outputs(
    history: dict[str, Any],
    queued_at: float,
    expected_count: int,
    *,
    context: OutputContext,
) -> list[Path]:
    expected_count = validate_image_frame_count(expected_count)
    staging_root = (context.config.output_dir / "h3" / "image_staging").resolve()
    candidates = _history_output_candidates(
        history, IMAGE_EXTENSIONS, directory=staging_root, context=context
    )
    unique = sorted(set(candidates), key=lambda path: path.name)
    if len(unique) >= expected_count:
        return unique[:expected_count]

    recent = sorted(
        _recent_output_candidates(
            staging_root, IMAGE_EXTENSIONS, queued_at, context=context
        ),
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    if len(recent) >= expected_count:
        return recent[-expected_count:]
    raise H3Error(
        "Generation completed, but the decoded image frame batch could not be located."
    )


def resolve_seedvr2_input_upscale_outputs(
    history: dict[str, Any],
    queued_at: float,
    output_token: str,
    slot_keys: Iterable[str],
    *,
    context: OutputContext,
) -> dict[str, Path]:
    """Resolve each named still written by a SeedVR2 input-upscale graph."""
    output_root = (context.config.output_dir / "h3" / "input_upscale").resolve()
    candidates = _history_output_candidates(
        history, IMAGE_EXTENSIONS, directory=output_root, context=context
    )
    if not candidates:
        candidates = _recent_output_candidates(
            output_root, IMAGE_EXTENSIONS, queued_at, context=context
        )
    resolved: dict[str, Path] = {}
    for slot_key in slot_keys:
        prefix = f"{output_token}_{slot_key}_"
        matches = [path for path in candidates if path.name.startswith(prefix)]
        if matches:
            resolved[slot_key] = max(matches, key=lambda path: path.stat().st_mtime)
    missing = [slot_key for slot_key in slot_keys if slot_key not in resolved]
    if missing:
        raise H3Error(
            "SeedVR2 completed, but these upscaled inputs could not be located: "
            + ", ".join(missing)
        )
    return resolved
