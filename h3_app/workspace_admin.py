"""Offline workspace backup/restore. Stop H3 before restoring a backup.

Usage: python -m h3_app.workspace_admin --state DIR backup BACKUP.zip
       python -m h3_app.workspace_admin --state EMPTY_DIR restore BACKUP.zip
Generated media is external to workspace state and must be backed up separately.
"""

from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import zipfile

from .workspace_store import WorkspaceStore, SCHEMA_VERSION


def backup_workspace(store, destination):
    destination = Path(destination).resolve()
    if destination.is_relative_to(store.root):
        raise ValueError("Write the backup outside the workspace state directory.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory, store.connect() as db:
        database = Path(directory) / "workspace.sqlite3"
        with closing(sqlite3.connect(database)) as target:
            db.backup(target)
        temporary = destination.with_name(destination.name + ".partial")
        try:
            with zipfile.ZipFile(
                temporary, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                archive.writestr(
                    "manifest.json",
                    json.dumps(
                        {"schema_version": SCHEMA_VERSION, "root": str(store.root)}
                    ),
                )
                archive.write(database, "workspace.sqlite3")
                # Storyboard cast pictures; takes live with generated media.
                for path in sorted((store.root / "storyboards").glob("*/cast/*")):
                    if path.is_file() and not path.is_symlink():
                        archive.write(path, path.relative_to(store.root).as_posix())
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


def restore_workspace(source, destination):
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Restore into an empty state directory with H3 stopped.")
    with zipfile.ZipFile(source) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest["schema_version"] > SCHEMA_VERSION:
            raise ValueError("This backup requires a newer H3 version.")
        for entry in archive.infolist():
            path = Path(entry.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or ":" in entry.filename
                or "\\" in entry.filename
            ):
                raise ValueError("Unsafe backup path.")
            # Older backups also hold retained saved-project files, which are no longer used.
            cast = len(path.parts) == 4 and path.parts[0] == "storyboards" and path.parts[2] == "cast"
            if entry.filename not in {"workspace.sqlite3", "manifest.json"} and not cast and (
                len(path.parts) != 3 or path.parts[0] != "projects"
            ):
                raise ValueError("Unexpected backup content.")
            if not (destination / path).resolve().is_relative_to(destination):
                raise ValueError("Unsafe backup path.")
        destination.mkdir(parents=True, exist_ok=True)
        archive.extract("workspace.sqlite3", destination)
        for entry in archive.namelist():
            if entry.startswith("storyboards/"):
                archive.extract(entry, destination)
    with closing(sqlite3.connect(destination / "workspace.sqlite3")) as db:
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("The restored database failed its integrity check.")
        old_root = str(Path(manifest["root"]) / "storyboards").replace("\\", "/") + "/"
        has_table = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='storyboards'"
        ).fetchone()
        for row_id, payload in db.execute("SELECT id, payload FROM storyboards").fetchall() if has_table else ():
            board = json.loads(payload)
            for member in board.get("cast", []):
                image = member["image"].replace("\\", "/")
                if image.startswith(old_root):
                    member["image"] = str(destination / "storyboards" / image[len(old_root):])
            db.execute("UPDATE storyboards SET payload=? WHERE id=?", (json.dumps(board), row_id))
        db.commit()
    return WorkspaceStore(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("action", choices=("backup", "restore"))
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    if args.action == "backup":
        print(backup_workspace(WorkspaceStore(args.state), args.archive))
    else:
        print(restore_workspace(args.archive, args.state).root)


if __name__ == "__main__":
    main()
