"""Per-project and batch cleanup of historical takes across all four project kinds."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from h3_app.errors import H3Error

PROJECT_KINDS = ("soundstage", "storyboard", "dubbing", "remake")
KIND_LABELS = {
    "soundstage": "Soundstage",
    "storyboard": "Storyboard",
    "dubbing": "Dubbing",
    "remake": "Remake",
}
PROJECT_TRANSFER_LOCK = threading.RLock()


def _format_bytes(num_bytes: int) -> str:
    if num_bytes >= (1 << 30):
        return f"{num_bytes / (1 << 30):.2f} GB"
    if num_bytes >= (1 << 20):
        return f"{num_bytes / (1 << 20):.1f} MB"
    if num_bytes >= (1 << 10):
        return f"{num_bytes / (1 << 10):.0f} KB"
    return f"{num_bytes} B"


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _unlink_tracked(path: Path) -> tuple[int, int]:
    """Delete a regular non-symlink file and return ``(removed_count, freed_bytes)``."""
    try:
        if not path.is_file() or path.is_symlink():
            return 0, 0
        size = path.stat().st_size
        path.unlink(missing_ok=True)
        return 1, size
    except OSError:
        return 0, 0


def _clear_directory_files(directory: Path, *, remove_root: bool = False) -> tuple[int, int]:
    """Remove all regular files under ``directory`` and prune emptied subdirectories."""
    if not directory.is_dir() or directory.is_symlink():
        return 0, 0
    removed_files = 0
    freed_bytes = 0
    root_resolved = directory.resolve()
    for dirpath, _dirnames, filenames in os.walk(directory, topdown=False):
        current_dir = Path(dirpath)
        for filename in filenames:
            count, size = _unlink_tracked(current_dir / filename)
            removed_files += count
            freed_bytes += size
        try:
            if current_dir.resolve() == root_resolved and not remove_root:
                continue
            current_dir.rmdir()
        except OSError:
            continue
    return removed_files, freed_bytes


def _soundstage_plan_audio(runtime, project_name: str) -> Path:
    digest = hashlib.sha256(project_name.encode()).hexdigest()
    return Path(runtime.output_dir) / ".soundstage" / "plans" / f"{digest}.wav"


def _resolve_soundstage_result(result: Any, runtime) -> Path | None:
    result_path = Path(result) if result else None
    if result_path is not None and not result_path.is_file():
        output_dir = Path(runtime.output_dir)
        result_path = next(
            (
                candidate
                for candidate in (
                    output_dir / "drama3" / result_path.name,
                    output_dir / "h3" / result_path.name,
                )
                if candidate.is_file()
            ),
            None,
        )
    return result_path


def _active_drama3_take_keys(store, runtime) -> set[str]:
    """Return every Drama 3 take key still referenced by any Soundstage mix or Dubbing session."""
    from h3_app.drama3.results import result_settings
    from h3_app.dubbing import projects as dubbing_projects, session as dub_sessions

    protected: set[str] = set()
    if store is not None:
        with store.connect() as db:
            rows = db.execute("SELECT result FROM soundstage_projects WHERE result IS NOT NULL").fetchall()
        for row in rows:
            res_path = _resolve_soundstage_result(row["result"], runtime)
            if res_path is None:
                continue
            for chunk in result_settings(res_path).get("chunks") or ():
                if isinstance(chunk, dict) and chunk.get("take"):
                    protected.add(str(chunk["take"]))

    sess_root = dub_sessions.sessions_dir(runtime)
    for entry in dubbing_projects.summaries(runtime):
        session_id = str(entry.get("session") or "")
        if not session_id:
            continue
        manifest = sess_root / session_id / dub_sessions.MANIFEST
        try:
            sess_data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if isinstance(sess_data, dict):
            for key in sess_data.get("last_take_keys") or ():
                if key:
                    protected.add(str(key))
    return protected


def _prune_unreferenced_drama3_takes(
    store,
    runtime,
    *,
    takes_dir: Path | None = None,
) -> tuple[int, int, int]:
    from h3_app.drama3.serving.takes import takes_root

    root = Path(takes_dir) if takes_dir is not None else takes_root(runtime)
    if not root.is_dir():
        return 0, 0, 0
    protected = _active_drama3_take_keys(store, runtime)
    removed_takes = 0
    removed_files = 0
    freed_bytes = 0
    try:
        entries = list(root.iterdir())
    except OSError:
        return 0, 0, 0
    for entry in entries:
        try:
            if not entry.is_file() or entry.is_symlink():
                continue
            if entry.suffix.lower() == ".wav" and not entry.name.startswith("."):
                if entry.stem not in protected:
                    count, size = _unlink_tracked(entry)
                    removed_takes += count
                    removed_files += count
                    freed_bytes += size
            elif entry.name.startswith(".") and entry.name.endswith(".tmp"):
                count, size = _unlink_tracked(entry)
                removed_files += count
                freed_bytes += size
        except OSError:
            continue
    return removed_takes, removed_files, freed_bytes


def _prune_soundstage_stills(
    payload: Any,
    result_path: Path | None,
    plan_audio: Path,
    output_dir: Path,
) -> tuple[int, int]:
    """Delete unreferenced stills images, effect masks/depth caches, and superseded plan dirs."""
    from h3_ui.drama3 import visuals

    removed_files = 0
    freed_bytes = 0
    active_audio = result_path if result_path is not None else plan_audio
    stills_folder = visuals.stills_dir(active_audio, output_dir)
    if stills_folder.is_dir() and (stills_folder / visuals.STILLS_RECORD).is_file():
        try:
            record = json.loads((stills_folder / visuals.STILLS_RECORD).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            record = None
        if isinstance(record, dict):
            keep_names: set[str] = {visuals.STILLS_RECORD}
            for shot in record.get("shots") or ():
                if isinstance(shot, dict):
                    if shot.get("image"):
                        keep_names.add(str(shot["image"]))
                    if shot.get("draft"):
                        keep_names.add(str(shot["draft"]))
            for portrait in (record.get("portraits") or {}).values():
                if isinstance(portrait, dict) and portrait.get("image"):
                    keep_names.add(str(portrait["image"]))

            try:
                entries = list(stills_folder.iterdir())
            except OSError:
                entries = []
            for entry in entries:
                if (
                    entry.is_file()
                    and not entry.is_symlink()
                    and entry.suffix.lower() in {".jpg", ".png"}
                    and entry.name not in keep_names
                ):
                    count, size = _unlink_tracked(entry)
                    removed_files += count
                    freed_bytes += size

            active_prefixes = tuple(f"{name}." for name in keep_names)
            protected_masks = {
                region["mask"]
                for fix in (payload.get("video_effect_regions") or {}).values()
                if isinstance(payload, dict) and isinstance(fix, dict)
                for region in (fix.get("regions") or {}).values()
                if isinstance(region, dict) and isinstance(region.get("mask"), str)
            }
            for sub in (stills_folder / ".effect-masks", stills_folder / ".effect-depth"):
                if not sub.is_dir() or sub.is_symlink():
                    continue
                try:
                    artifacts = list(sub.iterdir())
                except OSError:
                    artifacts = []
                for artifact in artifacts:
                    if (
                        artifact.is_file()
                        and not artifact.is_symlink()
                        and not artifact.name.startswith(active_prefixes)
                        and artifact.name not in protected_masks
                    ):
                        count, size = _unlink_tracked(artifact)
                        removed_files += count
                        freed_bytes += size

    if result_path is not None:
        count, size = _unlink_tracked(visuals.plan_path(plan_audio))
        removed_files += count
        freed_bytes += size
        c2, s2 = _clear_directory_files(visuals.stills_dir(plan_audio, output_dir), remove_root=True)
        removed_files += c2
        freed_bytes += s2

    return removed_files, freed_bytes


def _prune_soundstage_music_mixes(store, runtime) -> tuple[int, int]:
    """Delete cached background-music mixes not referenced by any Soundstage project."""
    from h3_ui.drama3.page_music import applied_music, desired_music, music_mix_id

    music_mixes_dir = Path(runtime.output_dir) / "h3" / ".soundstage" / "music-mixes"
    if not music_mixes_dir.is_dir() or music_mixes_dir.is_symlink():
        return 0, 0

    active_mix_files: set[str] = set()
    with store.connect() as db:
        all_rows = db.execute(
            "SELECT payload, result FROM soundstage_projects WHERE result IS NOT NULL"
        ).fetchall()
    for row in all_rows:
        res_p = _resolve_soundstage_result(row["result"], runtime)
        if res_p is None:
            continue
        try:
            res_stat = res_p.stat()
            pdata = json.loads(row["payload"]) if row["payload"] else {}
        except (OSError, TypeError, ValueError):
            continue
        if not isinstance(pdata, dict):
            continue
        for mcfg in (applied_music(pdata), desired_music(pdata)):
            mid = music_mix_id(mcfg)
            if mid:
                key = hashlib.sha256(
                    f"{res_p.resolve()}:{res_stat.st_mtime_ns}:{mid}".encode()
                ).hexdigest()
                active_mix_files.add(f"{key}.mp3")

    removed_files = 0
    freed_bytes = 0
    try:
        mix_entries = list(music_mixes_dir.iterdir())
    except OSError:
        mix_entries = []
    for entry in mix_entries:
        if (
            entry.is_file()
            and not entry.is_symlink()
            and entry.suffix.lower() == ".mp3"
            and entry.name not in active_mix_files
        ):
            count, size = _unlink_tracked(entry)
            removed_files += count
            freed_bytes += size
    return removed_files, freed_bytes


def _clear_soundstage_history_takes(
    project_id: str,
    *,
    store,
    runtime,
    takes_dir: Path | None = None,
) -> dict[str, Any]:
    if store is None:
        raise H3Error("Workspace store is unavailable.")
    from h3_app.drama3.results import result_settings

    try:
        payload, result = store.soundstage_project(project_id)
    except ValueError as exc:
        raise H3Error(f'Soundstage project "{project_id}" no longer exists.') from exc

    result_path = _resolve_soundstage_result(result, runtime)
    kept_takes = 0
    if result_path is not None:
        kept_takes = sum(
            1
            for chunk in result_settings(result_path).get("chunks") or ()
            if isinstance(chunk, dict) and chunk.get("take")
        )

    removed_takes, removed_files, freed_bytes = _prune_unreferenced_drama3_takes(
        store, runtime, takes_dir=takes_dir,
    )
    stills_files, stills_bytes = _prune_soundstage_stills(
        payload, result_path, _soundstage_plan_audio(runtime, project_id), Path(runtime.output_dir),
    )
    mix_files, mix_bytes = _prune_soundstage_music_mixes(store, runtime)

    return {
        "name": project_id,
        "kept_takes": kept_takes,
        "removed_takes": removed_takes,
        "removed_files": removed_files + stills_files + mix_files,
        "freed_bytes": freed_bytes + stills_bytes + mix_bytes,
    }


def _clear_dubbing_history_takes(project_id: str, *, store, runtime) -> dict[str, Any]:
    from h3_app.dubbing import lip_sync, projects as dubbing_projects

    _data, session = dubbing_projects.load(runtime, project_id)
    kept_takes = len([k for k in (session.data.get("last_take_keys") or ()) if k])
    removed_takes = 0
    removed_files = 0
    freed_bytes = 0

    last_output = session.data.get("last_output")
    if last_output:
        ls_outcome = lip_sync.clear_lip_sync_history_takes(Path(str(last_output)))
        kept_takes += int(ls_outcome["kept_takes"])
        removed_takes += int(ls_outcome["removed_takes"])
        removed_files += int(ls_outcome["removed_files"])
        freed_bytes += int(ls_outcome["freed_bytes"])

    for cache_dir in (session.root / "timed", session.root / "stems"):
        c, s = _clear_directory_files(cache_dir, remove_root=True)
        removed_files += c
        freed_bytes += s

    t_rm, f_rm, b_rm = _prune_unreferenced_drama3_takes(store, runtime)
    return {
        "name": project_id,
        "kept_takes": kept_takes,
        "removed_takes": removed_takes + t_rm,
        "removed_files": removed_files + f_rm,
        "freed_bytes": freed_bytes + b_rm,
    }


def format_clear_takes_message(
    kind: str,
    name: str,
    *,
    kept_takes: int,
    removed_takes: int,
    removed_files: int,
    freed_bytes: int,
) -> str:
    """Format a user-facing summary for one project's history take cleanup."""
    label = KIND_LABELS.get(kind, kind.title())
    if removed_takes > 0:
        return (
            f'Cleared {_plural(removed_takes, "history take")} '
            f'({_plural(removed_files, "file")}, {_format_bytes(freed_bytes)} freed) '
            f'from {label} "{name}"; kept {_plural(kept_takes, "take")} in use.'
        )
    if removed_files > 0:
        return (
            f'Cleared {_plural(removed_files, "stale take file")} '
            f'({_format_bytes(freed_bytes)} freed) '
            f'from {label} "{name}"; kept {_plural(kept_takes, "take")} in use.'
        )
    return f'{label} "{name}" has no history takes to clear ({_plural(kept_takes, "take")} in use).'


def build_clear_takes_outcome(
    kind: str,
    project_id: str,
    stats: Mapping[str, Any],
    *,
    default_name: str | None = None,
) -> dict[str, Any]:
    """Normalize per-project clear statistics into the API outcome dictionary."""
    kept_takes = int(stats.get("kept_takes", 0))
    removed_takes = int(stats.get("removed_takes", 0))
    removed_files = int(stats.get("removed_files", 0))
    freed_bytes = int(stats.get("freed_bytes", 0))
    name = str(stats.get("name") or default_name or project_id)
    return {
        "kind": kind,
        "id": project_id,
        "name": name,
        "kept_takes": kept_takes,
        "removed_takes": removed_takes,
        "removed_files": removed_files,
        "freed_bytes": freed_bytes,
        "message": format_clear_takes_message(
            kind,
            name,
            kept_takes=kept_takes,
            removed_takes=removed_takes,
            removed_files=removed_files,
            freed_bytes=freed_bytes,
        ),
    }


def parse_clear_takes_request(body: Any) -> tuple[str, list[tuple[str, str]]]:
    """Validate a single or batch clear-takes JSON body and return ``(mode, [(kind, id), ...])``."""
    if not isinstance(body, dict):
        raise H3Error("Invalid clear history takes request.")
    if "items" in body:
        raw_items = body.get("items")
        if not isinstance(raw_items, list) or not raw_items:
            raise H3Error("Select at least one project to clear history takes.")
        selections: list[tuple[str, str]] = []
        for entry in raw_items:
            if not isinstance(entry, dict) or not entry.get("kind") or not entry.get("id"):
                raise H3Error("Each item must specify kind and id.")
            selections.append((str(entry["kind"]), str(entry["id"])))
        return "batch", selections
    kind = str(body.get("kind") or "")
    project_id = str(body.get("id") or "")
    if not kind or not project_id:
        raise H3Error("Specify project kind and id.")
    return "single", [(kind, project_id)]


def clear_local_project_history_takes(
    kind: str,
    project_id: str,
    *,
    store,
    runtime,
    on_changed: Callable[[], None] | None = None,
    takes_dir: Path | None = None,
) -> dict[str, Any]:
    """Clear history takes for one local Soundstage, Storyboard, Dubbing, or Remake project, keeping takes in use."""
    if kind not in PROJECT_KINDS:
        raise H3Error(f"Unsupported project kind: {kind!r}")
    if not str(project_id or "").strip():
        raise H3Error("Specify a project id.")

    with PROJECT_TRANSFER_LOCK:
        if kind == "soundstage":
            stats = _clear_soundstage_history_takes(
                project_id, store=store, runtime=runtime, takes_dir=takes_dir,
            )
        elif kind == "storyboard":
            if store is None:
                raise H3Error("Workspace store is unavailable.")
            from h3_app.storyboard.service import StoryboardService

            try:
                stats = StoryboardService(store, lambda: runtime).clear_history_takes(None, project_id)
            except ValueError as exc:
                raise H3Error(str(exc)) from exc
        elif kind == "dubbing":
            stats = _clear_dubbing_history_takes(project_id, store=store, runtime=runtime)
        else:
            from h3_app.remake.project import Projects as RemakeProjects

            projects = RemakeProjects(runtime.output_dir)
            project = projects.load(None, project_id)
            if project is None:
                raise H3Error(f'Remake project "{project_id}" no longer exists.')
            stats = projects.clear_history_takes(project)

        outcome = build_clear_takes_outcome(kind, project_id, stats)
        if on_changed is not None and (outcome["removed_takes"] > 0 or outcome["removed_files"] > 0):
            on_changed()
        return outcome


def clear_local_projects_history_takes(
    items: Sequence[tuple[str, str]],
    *,
    store,
    runtime,
    on_changed: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Batch clear history takes across multiple selected local projects, keeping takes in use."""
    if not items:
        raise H3Error("Select at least one project to clear history takes.")
    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw_kind, raw_id in items:
        kind = str(raw_kind or "")
        pid = str(raw_id or "")
        if kind not in PROJECT_KINDS:
            raise H3Error(f"Unsupported project kind: {kind!r}")
        if not pid:
            raise H3Error("Each item must specify kind and id.")
        key = (kind, pid)
        if key not in seen:
            seen.add(key)
            deduped.append(key)

    with PROJECT_TRANSFER_LOCK:
        results: list[dict[str, Any]] = []
        cleared_projects = 0
        unchanged_projects = 0
        failed_projects = 0
        total_kept = 0
        total_removed_takes = 0
        total_removed_files = 0
        total_freed_bytes = 0

        for kind, pid in deduped:
            try:
                res = clear_local_project_history_takes(
                    kind,
                    pid,
                    store=store,
                    runtime=runtime,
                    on_changed=None,
                )
            except H3Error as exc:
                failed_projects += 1
                results.append(
                    {
                        "kind": kind,
                        "id": pid,
                        "name": pid,
                        "error": str(exc),
                        "kept_takes": 0,
                        "removed_takes": 0,
                        "removed_files": 0,
                        "freed_bytes": 0,
                        "message": str(exc),
                    }
                )
                continue
            results.append(res)
            total_kept += res["kept_takes"]
            total_removed_takes += res["removed_takes"]
            total_removed_files += res["removed_files"]
            total_freed_bytes += res["freed_bytes"]
            if res["removed_takes"] > 0 or res["removed_files"] > 0:
                cleared_projects += 1
            else:
                unchanged_projects += 1

        if on_changed is not None and (total_removed_takes > 0 or total_removed_files > 0):
            on_changed()

        processed = len(deduped)
        if total_removed_takes > 0:
            headline = (
                f"Cleared {_plural(total_removed_takes, 'history take')} "
                f"({_plural(total_removed_files, 'file')}, {_format_bytes(total_freed_bytes)} freed) "
                f"across {cleared_projects} of {_plural(processed, 'selected project')}."
            )
        elif total_removed_files > 0:
            headline = (
                f"Cleared {_plural(total_removed_files, 'stale take file')} "
                f"({_format_bytes(total_freed_bytes)} freed) "
                f"across {cleared_projects} of {_plural(processed, 'selected project')}."
            )
        elif failed_projects == processed:
            headline = f"Could not clear history takes for {_plural(processed, 'selected project')}."
        else:
            headline = (
                f"No history takes to clear across {_plural(processed - failed_projects, 'selected project')} "
                f"({_plural(total_kept, 'take')} in use)."
            )
        if failed_projects > 0 and failed_projects < processed:
            headline += f" {_plural(failed_projects, 'project')} failed."

        return {
            "processed": processed,
            "cleared_projects": cleared_projects,
            "unchanged_projects": unchanged_projects,
            "failed_projects": failed_projects,
            "kept_takes": total_kept,
            "removed_takes": total_removed_takes,
            "removed_files": total_removed_files,
            "freed_bytes": total_freed_bytes,
            "message": headline,
            "results": results,
        }
