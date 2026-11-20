"""Per-owner JSON draft storage, upload directories, and file access grants for standalone pages."""

from __future__ import annotations

from contextlib import ExitStack
import errno
import json
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from threading import RLock
from typing import Any, Mapping
import uuid

from h3_app.errors import H3Error

from .page_fields import SECRET_FIELDS, coerce, writer_defaults


def private_page_root(runtime, workspace: Path, slug: str, *, migrate_from: str | None = None) -> Path:
    """Keep owner drafts outside the served output roots, migrating legacy folders when needed."""
    outputs = runtime.outputs_dir.resolve()
    comfy = runtime.output_dir.resolve()
    publics = (outputs, comfy)
    root = (workspace / f"{slug}-page").resolve()
    holding = [public for public in publics if root.is_relative_to(public)]
    if holding:
        outer = min(holding, key=lambda public: len(public.parts))
        relative = (
            outputs.relative_to(outer)
            if slug == "image" and outputs.is_relative_to(outer)
            else root.relative_to(outer)
        )
        root = (outer.with_name(f".{outer.name}-{slug}-drafts") / relative).resolve()
        if any(root.is_relative_to(public) for public in publics):
            raise ValueError(f"Choose an H3_WORKSPACE_DIR outside the output directories for private {slug} drafts.")
    if migrate_from:
        previous = outputs / migrate_from
        if previous.is_dir() and previous.resolve() != root:
            root.parent.mkdir(parents=True, exist_ok=True)
            try:
                previous.rename(root)
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                with TemporaryDirectory(dir=root.parent) as staging:
                    copy = Path(staging) / f"{slug}-page"
                    shutil.copytree(previous, copy)
                    copy.rename(root)
                shutil.rmtree(previous)
    return root


class OwnerPageState:
    """Base per-owner JSON state store with in-memory API key secrets and scoped file access."""

    def __init__(self, services: Any):
        self.services = services
        self.lock = RLock()
        self.secrets: dict[str, dict[str, str]] = {}
        self.granted: dict[str, set[str]] = {}

    def owner_root(self, owner: str) -> Path:
        folder = self.services.root() / re.sub(r"[^\w-]", "_", owner)
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def uploads(self, owner: str) -> Path:
        folder = self.owner_root(owner) / "uploads"
        folder.mkdir(exist_ok=True)
        return folder

    def defaults(self) -> dict[str, Any]:
        return {name: spec.default for name, spec in self.services.fields.items()}

    def read_saved(self, owner: str) -> dict[str, Any]:
        try:
            saved = json.loads((self.owner_root(owner) / "settings.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return saved if isinstance(saved, dict) else {}

    def load_options(self, saved: Mapping[str, Any]) -> dict[str, Any]:
        options = self.defaults()
        for name, value in (saved.get("options") or {}).items():
            if name in options:
                try:
                    options[name] = coerce(self.services.fields[name], value)
                except H3Error:
                    continue
        return options

    def load_writer(self, owner: str, saved: Mapping[str, Any], defaults: Mapping[str, Any] | None = None) -> dict[str, Any]:
        writer = dict(defaults) if defaults is not None else writer_defaults()
        writer.update({name: value for name, value in (saved.get("writer") or {}).items()
                       if name in writer and name not in SECRET_FIELDS})
        writer.update(self.secrets.get(owner, {}))
        return writer

    def write_saved(self, owner: str, state: Mapping[str, Any], *, merge_options: bool = False) -> None:
        writer = state.get("writer") or {}
        self.secrets[owner] = {name: writer.get(name) or "" for name in SECRET_FIELDS}
        kept = {**state, "writer": {name: value for name, value in writer.items()
                                    if name not in SECRET_FIELDS}}
        target = self.owner_root(owner) / "settings.json"
        if merge_options:
            try:
                previous = json.loads(target.read_text(encoding="utf-8"))
                kept["options"] = {**(previous.get("options") or {}), **kept["options"]}
            except (OSError, ValueError, AttributeError, TypeError):
                pass
        scratch = target.with_suffix(".tmp")
        scratch.write_text(json.dumps(kept, default=str), encoding="utf-8")
        scratch.replace(target)

    def save(self, owner: str, state: dict[str, Any]) -> None:
        self.write_saved(owner, state)

    def may_read(self, owner: str, path: Any) -> bool:
        if str(path) in self.granted.get(owner, ()):
            return True
        return self.owner_root(owner).resolve() in Path(path).resolve().parents

    def grant(self, owner: str, path: Any) -> str:
        self.granted.setdefault(owner, set()).add(str(path))
        return str(path)

    def keep(self, owner: str, source: Any, rollback: ExitStack | None = None) -> str:
        """A copy of ``source`` among the owner's uploads, so the draft outlives where it came from."""
        target = self.uploads(owner) / f"{uuid.uuid4().hex[:12]}-{Path(source).name}"
        if rollback is not None:
            rollback.callback(target.unlink, missing_ok=True)
        shutil.copy2(source, target)
        return str(target)
