"""Owner-scoped Sequence runs, immutable request versions and an operation ledger.

SQLite is authoritative for ownership and operation intent. Upstream Continuum
remains authoritative for chunk and take validity; this module only reads a
bounded projection of its `project.json` for display and take selection.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator

from h3_app.generation.continuum_requests import ContinuumAction, ContinuumRequest

CONTINUUM_SCHEMA = """
CREATE TABLE IF NOT EXISTS continuum_runs (
    id TEXT PRIMARY KEY, owner TEXT NOT NULL, name TEXT NOT NULL,
    project_uuid TEXT UNIQUE NOT NULL, request_version INTEGER NOT NULL,
    epoch INTEGER NOT NULL DEFAULT 0, state TEXT NOT NULL,
    head_revision TEXT, output TEXT, report TEXT NOT NULL DEFAULT '{}',
    created REAL NOT NULL, updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS continuum_requests (
    run_id TEXT NOT NULL REFERENCES continuum_runs(id) ON DELETE CASCADE,
    version INTEGER NOT NULL, digest TEXT NOT NULL, payload TEXT NOT NULL,
    created REAL NOT NULL, PRIMARY KEY(run_id, version));
CREATE TABLE IF NOT EXISTS continuum_operations (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES continuum_runs(id) ON DELETE CASCADE,
    owner TEXT NOT NULL, request_key TEXT, parent_id TEXT, action TEXT NOT NULL,
    request_version INTEGER NOT NULL, expected_epoch INTEGER NOT NULL,
    state TEXT NOT NULL, prompt_id TEXT, receipt TEXT NOT NULL DEFAULT '{}',
    created REAL NOT NULL, updated REAL NOT NULL, UNIQUE(owner, request_key));
CREATE INDEX IF NOT EXISTS continuum_runs_owner ON continuum_runs(owner, updated DESC);
CREATE INDEX IF NOT EXISTS continuum_operations_run ON continuum_operations(run_id, created);
"""

OPEN_OPERATION_STATES = ("reserved", "submitting", "submitted")
MAX_PROJECT_BYTES = 8 * 1024**2
_HEX = frozenset("0123456789abcdef")


class StaleRunError(ValueError):
    """Another tab or job changed the sequence after this view was loaded."""


class RunBusyError(ValueError):
    pass


_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


@contextmanager
def run_lock(run_id: str) -> Iterator[None]:
    """One dispatcher per run, in addition to the GPU lease and upstream locks."""
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(run_id, threading.Lock())
    if not lock.acquire(blocking=False):
        raise RunBusyError("This sequence already has an operation running.")
    try:
        yield
    finally:
        lock.release()


def _is_hex(value: Any, limit: int = 64) -> bool:
    text = str(value or "")
    return 0 < len(text) <= limit and set(text) <= _HEX


@dataclass(frozen=True)
class Take:
    group: int
    start: int
    end: int
    revision_id: str
    created: str
    active: bool

    @property
    def label(self) -> str:
        chunks = f"Chunk {self.start}" if self.start == self.end else f"Chunks {self.start}–{self.end}"
        marker = " · in sequence" if self.active else ""
        return f"{chunks} · take {self.revision_id[:8]} · {self.created[:19]}{marker}"


@dataclass(frozen=True)
class UpstreamProject:
    head_revision: str | None
    chain: tuple[str, ...]
    takes: tuple[Take, ...]
    accepted_chunks: int
    status: str | None
    review_unit: tuple[int, int] | None

    def takes_for(self, group: int) -> tuple[Take, ...]:
        return tuple(take for take in self.takes if take.group == group)

    def active_take(self, group: int) -> Take | None:
        return next((t for t in self.takes if t.group == group and t.active), None)


def parse_project(payload: Any) -> UpstreamProject:
    """Project upstream's catalog without trusting it for reuse decisions."""
    if not isinstance(payload, dict) or not isinstance(
        payload.get("run_storage_schema_version"), int
    ):
        raise ValueError("Unsupported Continuum project catalog.")
    chain = tuple(str(item) for item in payload.get("canonical_chain") or () if _is_hex(item))
    active = set(chain)
    takes = []
    for item in payload.get("group_revisions") or ():
        if not isinstance(item, dict):
            continue
        group = item.get("group") or {}
        revision_id = str(item.get("revision_id", ""))
        try:
            start, end = int(group["start"]), int(group["end"])
            physical = int(group["physical_group"])
        except (KeyError, TypeError, ValueError):
            continue
        if not _is_hex(revision_id) or not 1 <= start <= end <= 16 or physical != start:
            continue
        takes.append(
            Take(physical, start, end, revision_id, str(item.get("created_utc", ""))[:32], revision_id in active)
        )
    takes.sort(key=lambda take: (take.group, take.created, take.revision_id))
    accepted = max((take.end for take in takes if take.active), default=0)
    status = None
    review_unit = None
    head_storage = str(payload.get("canonical_storage_revision_id") or "")
    for revision in payload.get("revisions") or ():
        if isinstance(revision, dict) and str(revision.get("revision_id")) == head_storage:
            status = str(revision.get("status") or "") or None
            unit = revision.get("review_unit")
            if isinstance(unit, dict):
                try:
                    review_unit = (int(unit["start"]), int(unit["end"]))
                except (KeyError, TypeError, ValueError):
                    review_unit = None
    head = payload.get("canonical_head_revision_id")
    return UpstreamProject(
        head_revision=str(head) if _is_hex(head) else None,
        chain=chain,
        takes=tuple(takes),
        accepted_chunks=accepted,
        status=status,
        review_unit=review_unit,
    )


class ContinuumStore:
    def __init__(self, workspace, output_dir: Path):
        self.workspace = workspace
        self.output_dir = Path(output_dir).resolve()
        self.root = (Path(workspace.root) / "continuum").resolve()

    # Runs ---------------------------------------------------------------

    def _sources(self, run_id: str) -> Path:
        if not _is_hex(run_id, 32):
            raise ValueError("Invalid sequence id.")
        return self.root / run_id / "sources"

    def _retain(self, run_id: str, request: ContinuumRequest) -> ContinuumRequest:
        """Copy sources into the run before any GPU work; reuse identical files."""
        directory = self._sources(run_id)
        directory.mkdir(parents=True, exist_ok=True)

        def keep(value: str | None) -> str | None:
            if not value:
                return None
            source = Path(value)
            if source.resolve().parent == directory.resolve():
                return str(source.resolve())
            if not source.is_file():
                raise ValueError("A Sequence source file expired. Upload it again.")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            target = directory / f"{digest}{source.suffix.lower() or '.bin'}"
            if not target.is_file():
                temporary = target.with_suffix(target.suffix + f".{uuid.uuid4().hex}.tmp")
                shutil.copy2(source, temporary)
                temporary.replace(target)
            return str(target)

        return replace(
            request,
            first_image=keep(request.first_image),
            reference_images=tuple(keep(item) for item in request.reference_images),
        )

    def create_run(self, owner: str, name: str, request: ContinuumRequest) -> str:
        run_id = uuid.uuid4().hex
        retained = self._retain(run_id, request)
        now = time.time()
        try:
            with self.workspace.connect() as db:
                db.execute(
                    "INSERT INTO continuum_runs VALUES (?, ?, ?, ?, 1, 0, 'draft', NULL, NULL, '{}', ?, ?)",
                    (run_id, owner, (name.strip() or "Untitled sequence")[:120], str(uuid.uuid4()), now, now),
                )
                db.execute(
                    "INSERT INTO continuum_requests VALUES (?, 1, ?, ?, ?)",
                    (run_id, retained.digest(), json.dumps(retained.payload()), now),
                )
        except BaseException:
            shutil.rmtree(self.root / run_id, ignore_errors=True)
            raise
        return run_id

    def runs(self, owner: str) -> list[dict[str, Any]]:
        with self.workspace.connect() as db:
            rows = db.execute(
                "SELECT id, name, state, request_version, epoch, updated FROM continuum_runs "
                "WHERE owner=? ORDER BY updated DESC LIMIT 200",
                (owner,),
            ).fetchall()
        return [dict(row) for row in rows]

    def run(self, owner: str, run_id: str) -> dict[str, Any]:
        with self.workspace.connect() as db:
            row = db.execute(
                "SELECT * FROM continuum_runs WHERE id=? AND owner=?", (run_id, owner)
            ).fetchone()
        if row is None:
            raise ValueError("This sequence is unavailable for this browser owner.")
        run = dict(row)
        run["report"] = json.loads(run["report"] or "{}")
        return run

    def request(self, owner: str, run_id: str, version: int | None = None) -> tuple[int, ContinuumRequest]:
        run = self.run(owner, run_id)
        version = int(version or run["request_version"])
        with self.workspace.connect() as db:
            row = db.execute(
                "SELECT payload FROM continuum_requests WHERE run_id=? AND version=?",
                (run_id, version),
            ).fetchone()
        if row is None:
            raise ValueError("This sequence request version is unavailable.")
        return version, ContinuumRequest.from_payload(json.loads(row["payload"]))

    def revise(self, owner: str, run_id: str, request: ContinuumRequest, *, expected_epoch: int) -> int:
        """Save an edited request as a new immutable version (CAS on the epoch)."""
        current_version, current = self.request(owner, run_id)
        retained = self._retain(run_id, request)
        if retained.digest() == current.digest():
            return current_version
        now = time.time()
        with self.workspace.connect() as db:
            changed = db.execute(
                "UPDATE continuum_runs SET epoch=epoch+1, request_version=request_version+1, updated=? "
                "WHERE id=? AND owner=? AND epoch=?",
                (now, run_id, owner, int(expected_epoch)),
            ).rowcount
            if not changed:
                raise StaleRunError("This sequence changed in another tab. Reload it first.")
            version = db.execute(
                "SELECT request_version FROM continuum_runs WHERE id=?", (run_id,)
            ).fetchone()[0]
            db.execute(
                "INSERT INTO continuum_requests VALUES (?, ?, ?, ?, ?)",
                (run_id, version, retained.digest(), json.dumps(retained.payload()), now),
            )
        return int(version)

    def rename(self, owner: str, run_id: str, name: str) -> None:
        self.run(owner, run_id)
        with self.workspace.connect() as db:
            db.execute(
                "UPDATE continuum_runs SET name=?, updated=? WHERE id=? AND owner=?",
                ((name.strip() or "Untitled sequence")[:120], time.time(), run_id, owner),
            )

    def set_state(self, run_id: str, state: str, **fields: Any) -> None:
        assignments = ["state=?", "updated=?"]
        values: list[Any] = [state, time.time()]
        for key in ("head_revision", "output"):
            if key in fields:
                assignments.append(f"{key}=?")
                values.append(fields[key])
        if "report" in fields:
            assignments.append("report=?")
            values.append(json.dumps(fields["report"]))
        with self.workspace.connect() as db:
            db.execute(
                f"UPDATE continuum_runs SET {', '.join(assignments)} WHERE id=?",
                (*values, run_id),
            )

    # Operations ---------------------------------------------------------

    def reserve(
        self,
        owner: str,
        run_id: str,
        action: ContinuumAction,
        *,
        expected_epoch: int,
        key: str | None = None,
        parent_id: str | None = None,
        request_version: int | None = None,
    ) -> dict[str, Any]:
        """Record intent before queueing. A transport retry reuses `key`."""
        encoded = json.dumps(action.__dict__, sort_keys=True)
        now = time.time()
        with self.workspace.connect() as db:
            if key:
                row = db.execute(
                    "SELECT * FROM continuum_operations WHERE owner=? AND request_key=?",
                    (owner, key),
                ).fetchone()
                if row is not None:
                    if row["run_id"] != run_id or row["action"] != encoded:
                        raise ValueError("This request id already belongs to another action.")
                    return dict(row)
            run = db.execute(
                "SELECT epoch, request_version FROM continuum_runs WHERE id=? AND owner=?",
                (run_id, owner),
            ).fetchone()
            if run is None:
                raise ValueError("This sequence is unavailable for this browser owner.")
            if parent_id is None:
                changed = db.execute(
                    "UPDATE continuum_runs SET epoch=epoch+1, state='queued', updated=? "
                    "WHERE id=? AND owner=? AND epoch=?",
                    (now, run_id, owner, int(expected_epoch)),
                ).rowcount
                if not changed:
                    raise StaleRunError(
                        "This sequence changed in another tab or job. Reload it before continuing."
                    )
            operation = {
                "id": uuid.uuid4().hex,
                "run_id": run_id,
                "owner": owner,
                "request_key": key,
                "parent_id": parent_id,
                "action": encoded,
                "request_version": int(request_version or run["request_version"]),
                "expected_epoch": int(expected_epoch),
                "state": "reserved",
                "prompt_id": None,
                "receipt": "{}",
                "created": now,
                "updated": now,
            }
            db.execute(
                "INSERT INTO continuum_operations VALUES "
                "(:id, :run_id, :owner, :request_key, :parent_id, :action, :request_version, "
                ":expected_epoch, :state, :prompt_id, :receipt, :created, :updated)",
                operation,
            )
        return operation

    def update_operation(self, operation_id: str, *, state: str, prompt_id: str | None = None, receipt: dict | None = None) -> None:
        with self.workspace.connect() as db:
            row = db.execute(
                "SELECT receipt FROM continuum_operations WHERE id=?", (operation_id,)
            ).fetchone()
            if row is None:
                raise ValueError("Unknown Sequence operation.")
            merged = {**json.loads(row["receipt"] or "{}"), **(receipt or {})}
            db.execute(
                "UPDATE continuum_operations SET state=?, prompt_id=COALESCE(?, prompt_id), "
                "receipt=?, updated=? WHERE id=?",
                (state, prompt_id, json.dumps(merged), time.time(), operation_id),
            )

    def operations(self, owner: str, run_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self.workspace.connect() as db:
            rows = db.execute(
                "SELECT * FROM continuum_operations WHERE run_id=? AND owner=? "
                "ORDER BY created DESC LIMIT ?",
                (run_id, owner, int(limit)),
            ).fetchall()
        return [{**dict(row), "receipt": json.loads(row["receipt"] or "{}")} for row in rows]

    def open_operations(self) -> list[dict[str, Any]]:
        placeholders = ", ".join("?" for _ in OPEN_OPERATION_STATES)
        with self.workspace.connect() as db:
            rows = db.execute(
                f"SELECT * FROM continuum_operations WHERE state IN ({placeholders})",
                OPEN_OPERATION_STATES,
            ).fetchall()
        return [dict(row) for row in rows]

    def reconcile(self, history: Any) -> int:
        """After a restart, settle operations whose outcome is unknown.

        `history(prompt_id)` returns ComfyUI's history item or None. Nothing is
        resubmitted: an unfinished operation is marked interrupted and the run
        is refreshed from upstream's committed prefix on next view.
        """
        settled = 0
        for operation in self.open_operations():
            state = "interrupted"
            if operation["prompt_id"]:
                try:
                    item = history(operation["prompt_id"])
                except Exception:
                    item = None
                status = (item or {}).get("status") or {}
                if status.get("status_str") == "error":
                    state = "failed"
                elif status.get("completed") or (item or {}).get("outputs"):
                    state = "completed_unrecorded"
            self.update_operation(operation["id"], state=state, receipt={"reconciled": True})
            self.set_state(operation["run_id"], "interrupted" if state != "completed_unrecorded" else "ready")
            settled += 1
        return settled

    # Upstream projection and deletion -----------------------------------

    def upstream_directory(self, run: dict[str, Any]) -> Path:
        project = uuid.UUID(str(run["project_uuid"]))
        runs = (self.output_dir / "h3_continuum" / "runs").resolve()
        directory = (runs / f"run_{project.hex}").resolve()
        if directory.parent != runs:
            raise ValueError("Invalid Continuum run directory.")
        return directory

    def project(self, run: dict[str, Any]) -> UpstreamProject | None:
        path = self.upstream_directory(run) / "project.json"
        try:
            if path.stat().st_size > MAX_PROJECT_BYTES:
                raise ValueError("The Continuum project catalog is unexpectedly large.")
            return parse_project(json.loads(path.read_text(encoding="utf-8")))
        except FileNotFoundError:
            return None

    def delete_run(self, owner: str, run_id: str) -> None:
        run = self.run(owner, run_id)
        with run_lock(run_id):
            with self.workspace.connect() as db:
                busy = db.execute(
                    "SELECT COUNT(*) FROM continuum_operations WHERE run_id=? AND state IN ('reserved','submitting','submitted')",
                    (run_id,),
                ).fetchone()[0]
                if busy:
                    raise RunBusyError("Stop the running operation before deleting this sequence.")
                db.execute("DELETE FROM continuum_runs WHERE id=? AND owner=?", (run_id, owner))
            shutil.rmtree(self.upstream_directory(run), ignore_errors=True)
            previews = (self.output_dir / "h3_continuum" / "previews" / run_id).resolve()
            if previews.parent == (self.output_dir / "h3_continuum" / "previews").resolve():
                shutil.rmtree(previews, ignore_errors=True)
            directory = (self.root / run_id).resolve()
            if directory.parent == self.root:
                shutil.rmtree(directory, ignore_errors=True)

    def owns_path(self, path: str | Path) -> bool:
        """Whether a file belongs to retained Sequence state (excluded from Media)."""
        try:
            resolved = Path(path).resolve()
        except OSError:
            return False
        return resolved.is_relative_to(self.root) or resolved.is_relative_to(
            self.output_dir / "h3_continuum"
        )
