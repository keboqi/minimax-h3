"""Contained, submission-scoped media discovery, independent of Gradio."""

from pathlib import Path
from typing import Any, Iterable
from .jobs import CURRENT_JOB


def walk_saved_refs(value: Any) -> Iterable[dict[str, str]]:
    if isinstance(value, dict):
        if "filename" in value:
            yield {
                "filename": str(value["filename"]),
                "subfolder": str(value.get("subfolder", "")),
                "type": str(value.get("type", "output")),
            }
        for child in value.values():
            yield from walk_saved_refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_saved_refs(child)


def history_output_candidates(
    output_dir: Path,
    history: dict[str, Any],
    extensions: frozenset[str],
    *,
    directory: Path | None = None,
) -> list[Path]:
    """Return existing history outputs contained by the requested directory."""
    output_root = output_dir.resolve()
    resolved_directory = (output_dir if directory is None else directory).resolve()
    if not resolved_directory.is_relative_to(output_root):
        raise ValueError("Output candidate directory must be inside output_dir")

    candidates: list[Path] = []
    for ref in walk_saved_refs(history.get("outputs", {})):
        if ref["type"] != "output":
            continue
        try:
            path = (output_dir / ref["subfolder"] / ref["filename"]).resolve()
            if (
                path.is_relative_to(resolved_directory)
                and path.is_file()
                and path.suffix.lower() in extensions
            ):
                candidates.append(path)
        except OSError:
            continue
    return candidates


def recent_output_candidates(
    directory: Path,
    extensions: frozenset[str],
    queued_at: float,
) -> list[Path]:
    """Return recent files without following outputs outside the scan root."""
    if not directory.is_dir():
        return []
    resolved_directory = directory.resolve()
    candidates: dict[Path, Path] = {}
    for candidate in directory.rglob("*"):
        try:
            path = candidate.resolve()
            if (
                path.is_relative_to(resolved_directory)
                and path.is_file()
                and path.suffix.lower() in extensions
                and CURRENT_JOB.get() is not None
                and CURRENT_JOB.get().output_token is not None
                and CURRENT_JOB.get().output_token in path.name
                and path.stat().st_mtime >= queued_at - 2
            ):
                candidates[path] = path
        except OSError:
            continue
    return list(candidates.values())
