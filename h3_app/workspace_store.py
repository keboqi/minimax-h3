"""Durable workspace metadata. Request content is saved only by explicit consent.

SQLite is the authority for jobs, project ownership and asset annotations. The
media listing is a rebuildable index; rebuilding it never discards annotations.
All connections are short lived so backup and threaded Gradio callbacks agree.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import shutil
import sqlite3
import time
import uuid
from contextlib import contextmanager, closing
from starlette.requests import Request

SCHEMA_VERSION = 1
COOKIE = "h3_workspace_owner"
OWNER_TTL = 30 * 86400


class WorkspaceStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "workspace.sqlite3"
        with self.connect() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise RuntimeError(
                    "The workspace database requires a newer H3 version."
                )
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, created REAL NOT NULL,
                    request_key TEXT, digest TEXT, payload TEXT NOT NULL,
                    UNIQUE(owner, request_key));
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, name TEXT NOT NULL,
                    job_id TEXT NOT NULL, created REAL NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS assets (
                    id TEXT PRIMARY KEY, path TEXT UNIQUE NOT NULL, kind TEXT NOT NULL,
                    modified REAL NOT NULL, available INTEGER NOT NULL, metadata TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS annotations (
                    asset_id TEXT PRIMARY KEY, tags TEXT NOT NULL DEFAULT '[]',
                    favorite INTEGER NOT NULL DEFAULT 0);
                CREATE INDEX IF NOT EXISTS jobs_owner ON jobs(owner, created);
                CREATE INDEX IF NOT EXISTS projects_owner ON projects(owner, created);
            """
            )
            db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            db.execute(
                "INSERT OR IGNORE INTO settings VALUES ('owner_secret', ?)",
                (secrets.token_hex(32),),
            )
            self.secret = (
                db.execute("SELECT value FROM settings WHERE key='owner_secret'")
                .fetchone()[0]
                .encode()
            )

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=15)
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=FULL")
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def issue_owner(self, owner=None):
        owner = owner or secrets.token_hex(24)
        payload = f"{owner}.{int(time.time()) + OWNER_TTL}"
        return (
            payload
            + "."
            + hmac.new(self.secret, payload.encode(), hashlib.sha256).hexdigest()
        )

    def verify_owner(self, token):
        try:
            owner, expiry, signature = token.split(".")
            if len(owner) != 48 or any(c not in "0123456789abcdef" for c in owner):
                return None
            payload = f"{owner}.{expiry}"
            expected = hmac.new(
                self.secret, payload.encode(), hashlib.sha256
            ).hexdigest()
            return (
                owner
                if int(expiry) > time.time()
                and hmac.compare_digest(signature, expected)
                else None
            )
        except (AttributeError, ValueError):
            return None

    def recovery_key(self, owner):
        return (
            owner
            + "."
            + hmac.new(
                self.secret, ("recovery:" + owner).encode(), hashlib.sha256
            ).hexdigest()
        )

    def recovery_owner(self, key):
        try:
            owner, signature = key.split(".")
            if len(owner) != 48 or any(c not in "0123456789abcdef" for c in owner):
                return None
            return owner if hmac.compare_digest(self.recovery_key(owner), key) else None
        except (ValueError, AttributeError):
            return None

    def save_job(self, job):
        # No prompt text, graph text, uploaded inputs or provider credentials.
        fields = (
            "id",
            "owner",
            "family",
            "prompt_id",
            "output_token",
            "state",
            "stage",
            "created_at",
            "finished_at",
            "outputs",
            "variant",
            "variant_seeds",
            "retry_of",
            "media_indices",
            "output_indices",
            "recoverable_sources",
            "finishing_offsets",
            "idempotency_key",
            "request_digest",
            "canvas_request",
            "source_asset_ids",
        )
        payload = {key: getattr(job, key) for key in fields}
        # Progress/error strings can include backend text. Keep unsaved history
        # technical even when a callback includes user content in its status.
        payload["stage"] = f"{job.family}: {job.state}"
        payload["failed_variants"] = sorted(job.failed_variants)
        payload["ledger"] = [
            {
                k: v
                for k, v in entry.items()
                if k not in {"graph_json", "error", "stage"}
            }
            for entry in job.ledger
        ]
        payload["error"] = (
            "Execution failed; inspect backend history." if job.error else None
        )
        with self.connect() as db:
            db.execute(
                "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                (
                    job.id,
                    job.owner,
                    job.created_at,
                    job.idempotency_key,
                    job.request_digest,
                    json.dumps(payload),
                ),
            )
            # Only an explicitly saved project may retain workflow/request content.
            for row in db.execute(
                "SELECT id, payload FROM projects WHERE job_id=? AND owner=?",
                (job.id, job.owner),
            ).fetchall():
                project = json.loads(row["payload"])
                project["ledger"] = job.ledger
                project["variant_seeds"] = job.variant_seeds
                project["failed_variants"] = sorted(job.failed_variants)
                if job.finishing_request:
                    project["finishing"] = job.finishing_request
                db.execute(
                    "UPDATE projects SET payload=? WHERE id=?",
                    (json.dumps(project), row["id"]),
                )

    def jobs(self, owner=None, limit=128):
        with self.connect() as db:
            rows = db.execute(
                "SELECT payload FROM jobs"
                + (" WHERE owner=?" if owner else "")
                + " ORDER BY created DESC LIMIT ?",
                ((owner, limit) if owner else (limit,)),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def job(self, owner, job_id):
        with self.connect() as db:
            row = db.execute(
                "SELECT payload FROM jobs WHERE id=? AND owner=?", (job_id, owner)
            ).fetchone()
        if row is None:
            raise ValueError("This job is unavailable for this browser owner.")
        return json.loads(row[0])

    def job_for_key(self, owner, key):
        with self.connect() as db:
            row = db.execute(
                "SELECT payload FROM jobs WHERE owner=? AND request_key=?", (owner, key)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def save_project(self, owner, job, name):
        if job.owner != owner:
            raise ValueError("Project ownership mismatch.")
        values = job.values()
        project_id = uuid.uuid4().hex
        directory = self.root / "projects" / project_id
        directory.mkdir(parents=True)
        copied = {}

        def retain(value):
            if isinstance(value, list):
                return [retain(item) for item in value]
            if isinstance(value, dict):
                return {
                    k: retain(v) if k in {"path", "name", "video", "audio"} else v
                    for k, v in value.items()
                }
            if value is None:
                return None
            path = Path(value)
            if not path.is_file():
                raise ValueError(
                    "A source file expired. Upload it again before saving this project."
                )
            if str(path) not in copied:
                destination = directory / (uuid.uuid4().hex + path.suffix)
                shutil.copy2(path, destination)
                copied[str(path)] = str(destination)
            return copied[str(path)]

        try:
            for index in job.media_indices:
                values[index] = retain(values[index])
            payload = {
                "family": job.family,
                "values": values,
                "media_indices": job.media_indices,
                "output_indices": job.output_indices,
                "ledger": job.ledger,
                "variant_seeds": job.variant_seeds,
                "failed_variants": sorted(job.failed_variants),
                "finishing": job.finishing_request,
                "canvas_request": job.canvas_request,
                "source_asset_ids": job.source_asset_ids,
            }
            with self.connect() as db:
                db.execute(
                    "INSERT INTO projects VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        project_id,
                        owner,
                        (name.strip() or "Untitled project")[:120],
                        job.id,
                        time.time(),
                        json.dumps(payload),
                    ),
                )
            return project_id
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise

    def projects(self, owner):
        with self.connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT id, name, job_id, created FROM projects WHERE owner=? ORDER BY created DESC",
                    (owner,),
                )
            ]

    def project(self, owner, project_id):
        with self.connect() as db:
            row = db.execute(
                "SELECT payload FROM projects WHERE id=? AND owner=?",
                (project_id, owner),
            ).fetchone()
        if row is None:
            raise ValueError("This project is unavailable for this browser owner.")
        return json.loads(row[0])

    def project_for_job(self, owner, job_id):
        projects = [p for p in self.projects(owner) if p["job_id"] == job_id]
        return (
            {**self.project(owner, projects[0]["id"]), "id": projects[0]["id"]}
            if projects
            else None
        )

    def delete_project(self, owner, project_id):
        self.project(owner, project_id)  # Validate ownership before changing files.
        with self.connect() as db:
            db.execute(
                "DELETE FROM projects WHERE id=? AND owner=?", (project_id, owner)
            )
        directory = (self.root / "projects" / project_id).resolve()
        if directory.parent != self.root / "projects":
            raise ValueError("Invalid project directory.")
        shutil.rmtree(directory, ignore_errors=True)

    def index_asset(self, path, kind, metadata=None):
        path = Path(path).resolve()
        asset_id = uuid.uuid5(uuid.NAMESPACE_URL, path.as_uri()).hex
        with self.connect() as db:
            db.execute(
                "INSERT INTO assets VALUES (?, ?, ?, ?, 1, ?) ON CONFLICT(path) DO UPDATE SET modified=excluded.modified, available=1, metadata=excluded.metadata",
                (
                    asset_id,
                    str(path),
                    kind,
                    path.stat().st_mtime,
                    json.dumps(metadata or {}),
                ),
            )
        return asset_id

    def annotate(self, asset_id, *, tags, favorite):
        tags = sorted({str(tag).strip()[:40] for tag in tags if str(tag).strip()})[:32]
        with self.connect() as db:
            if not db.execute(
                "SELECT id FROM assets WHERE id=?", (asset_id,)
            ).fetchone():
                raise ValueError("The selected asset is unavailable.")
            db.execute(
                "INSERT INTO annotations VALUES (?, ?, ?) ON CONFLICT(asset_id) DO UPDATE SET tags=excluded.tags, favorite=excluded.favorite",
                (asset_id, json.dumps(tags), int(bool(favorite))),
            )

    def search_assets(self, *, query="", kind=None, favorite=False, limit=200):
        pattern = (
            "%"
            + query.casefold()
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
            + "%"
        )
        with self.connect() as db:
            rows = db.execute(
                "SELECT a.*, COALESCE(n.tags, '[]') AS tags, COALESCE(n.favorite, 0) AS favorite "
                "FROM assets a LEFT JOIN annotations n ON a.id=n.asset_id "
                "WHERE available=1 AND (? IS NULL OR kind=?) AND (?=0 OR n.favorite=1) "
                "AND lower(path || ' ' || COALESCE(n.tags,'[]') || ' ' || metadata) LIKE ? ESCAPE '\\' "
                "ORDER BY modified DESC LIMIT ?",
                (kind, kind, int(favorite), pattern, min(max(1, limit), 200)),
            ).fetchall()
        results = []
        for row in rows:
            data = dict(row)
            data["metadata"] = json.loads(data["metadata"])
            data["tags"] = json.loads(data["tags"])
            results.append(data)
        return results

    def rebuild_index(self, paths):
        # Keep identity and annotation rows authoritative, even for missing files.
        with self.connect() as db:
            db.execute("UPDATE assets SET available=0")
        for path, kind, metadata in paths:
            if Path(path).is_file():
                self.index_asset(path, kind, metadata)

    def backup(self, destination):
        destination = Path(destination).resolve()
        if destination == self.database:
            raise ValueError("The backup must use a different path.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as source, closing(sqlite3.connect(destination)) as target:
            source.backup(target)
            if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("Workspace backup failed its integrity check.")
        return destination


def request_owner(request, store):
    token = getattr(request, "cookies", {}).get(COOKIE)
    owner = store.verify_owner(token) if store else None
    if owner:
        return owner
    # Unit fixtures and direct API clients still receive isolated session scopes.
    session = getattr(request, "session_hash", None)
    if session:
        return "api:" + session if store else session
    raise ValueError("A browser owner or live API session is required.")


def install_owner_cookie(app, store):
    from fastapi import HTTPException
    from fastapi.responses import JSONResponse

    @app.post("/workspace/owner/recover")
    async def recover_owner(request: Request):
        payload = await request.json()
        owner = store.recovery_owner(payload.get("key"))
        if not owner:
            raise HTTPException(403, "Invalid browser recovery key")
        response = JSONResponse({"restored": True})
        response.set_cookie(
            COOKIE,
            store.issue_owner(owner),
            max_age=OWNER_TTL,
            httponly=True,
            samesite="lax",
            secure=request.url.scheme == "https",
        )
        return response

    @app.middleware("http")
    async def owner_cookie(request, call_next):
        token = request.cookies.get(COOKIE)
        owner = store.verify_owner(token)
        issued = None
        if not owner:
            issued = store.issue_owner()
            request.scope["headers"] = [
                (k, v) for k, v in request.scope["headers"] if k != b"cookie"
            ] + [
                (
                    b"cookie",
                    (
                        request.headers.get("cookie", "") + f"; {COOKIE}={issued}"
                    ).encode(),
                )
            ]
        elif int(token.split(".")[1]) - time.time() < 7 * 86400:
            issued = store.issue_owner(owner)
        response = await call_next(request)
        if issued and request.url.path != "/workspace/owner/recover":
            response.set_cookie(
                COOKIE,
                issued,
                max_age=OWNER_TTL,
                httponly=True,
                samesite="lax",
                secure=request.url.scheme == "https",
            )
        return response


def default_store(outputs_dir):
    return WorkspaceStore(
        os.getenv(
            "H3_WORKSPACE_DIR", str(Path(outputs_dir).resolve().parent / "h3-workspace")
        )
    )
