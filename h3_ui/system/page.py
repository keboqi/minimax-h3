"""The System page and its JSON API, outside Gradio, as Media, Image and Song are.

Provides GPU backend & VRAM monitoring, VRAM unloading, global Drama 3 serving settings,
session-scoped temporary Gemini API key management, local project management, remote
project import, 30s-cached local model readiness, engine capabilities, and maintenance.
The Gradio view stays built but unseen for published API and contract tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from threading import RLock
import time
from typing import Any, Callable, Mapping

from fastapi import Body, FastAPI, Request

from h3_app.capabilities import ENGINES
from h3_app.drama3.serving.config import (
    BACKEND_CHOICES,
    BACKEND_COMFYUI,
    BACKEND_SGLANG,
    COMFY_EXPERTS_CHOICES,
    apply_drama3_settings_update,
    describe_drama3_settings,
    load_drama3_settings,
    model_choices_for_backend,
)
from h3_app.errors import H3Error
from h3_app.llm import gemini
from h3_app.project_transfer import (
    CONFLICT_MODES,
    KIND_LABELS,
    PROJECT_KINDS,
    build_auth_headers,
    delete_local_project,
    fetch_remote_projects,
    format_import_summary_headline,
    import_remote_projects,
    list_local_projects,
    rename_local_project,
)
from h3_gpu import CPU_ONLY_MESSAGE, is_a100, is_cpu_only

from ..page_fields import plain
from ..page_routes import install_static_page, owner_of
from ..page_runs import Run, Runs, guarded

STATIC = Path(__file__).with_name("page")
STATIC_TYPES = {"app.js": "application/javascript", "app.css": "text/css"}
ROUTE = "/system"
API = f"{ROUTE}/api"

UNLOAD, SCAN, IMPORT = "unload", "scan", "import"
BUSY = {
    UNLOAD: "A VRAM release is already running.",
    SCAN: "A historical media scan is already running.",
    IMPORT: "A project import is already running.",
}
MODEL_CACHE_TTL = 30.0
_TASK_DISPLAY = {"TTS": "Soundstage", "Dub": "Dubbing"}
_KIND_ORDER = {kind: index for index, kind in enumerate(PROJECT_KINDS)}
_FAMILY_PREFIXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Drama 3", ("drama3_",)),
    ("LTX 2.5", ("ltx25_",)),
    ("Qwen Image 2.1", ("qwen_image21_",)),
    ("FLUX 3", ("flux3_",)),
    ("MOSS-Image", ("moss_image_",)),
    ("Fibo 2.0", ("fibo2_",)),
    ("Music 3", ("music3_",)),
    ("YuE2", ("yue2_",)),
    ("SeedVR2", ("seedvr2_",)),
    ("Soundstage / Remake", ("soundstage_", "remake_", "face_")),
    ("RelayVSR", ("relayvsr_",)),
)
_FAMILY_ORDER = (
    "MiniMax H3",
    "Drama 3",
    "LTX 2.5",
    "Qwen Image 2.1",
    "FLUX 3",
    "MOSS-Image",
    "Fibo 2.0",
    "Music 3",
    "YuE2",
    "SeedVR2",
    "Soundstage / Remake",
    "RelayVSR",
)
_VRAM_FREE_RE = re.compile(r"([\d.]+)\s*/\s*([\d.]+)\s*GiB VRAM free")
_VRAM_TOTAL_RE = re.compile(r"([\d.]+)\s*GiB VRAM")
_PROFILE_RE = re.compile(r"^\*\*(.+?)\*\*\s*(?:[·|:]\s*)?(.*)$")
_DRAMA3_ERROR_MARKER = "\n\n**Could not apply change:** "
_BACKEND_DESC = {
    BACKEND_SGLANG: ("Parallel", "Parallel batch generation across Soundstage, Voice Design, and Dubbing."),
    BACKEND_COMFYUI: ("Fast start", "Sequential chunk generation with optional expert offloading to RAM."),
}


def model_family(key: str) -> str:
    """Classify a ``MODEL_SPECS`` key under its user-facing engine family."""
    for family, prefixes in _FAMILY_PREFIXES:
        if key.startswith(prefixes):
            return family
    return "MiniMax H3"


def engine_entries() -> list[dict[str, Any]]:
    """Return structured engine capability descriptors matching ``h3_app.capabilities.ENGINES``."""
    items: list[dict[str, Any]] = []
    for engine in ENGINES:
        tasks = [_TASK_DISPLAY.get(task, task) for task in engine.tasks]
        if "Video" in engine.tasks:
            group = "video"
        elif "Image" in engine.tasks:
            group = "image"
        elif "Audio / Music" in engine.tasks:
            group = "audio"
        else:
            group = "production"
        items.append(
            {
                "id": engine.id,
                "label": engine.label,
                "tasks": tasks,
                "tasks_display": " · ".join(tasks),
                "group": group,
                "description": engine.description,
            }
        )
    return items


def _cap_value(raw: str | None) -> str:
    cleaned = str(raw or "").strip()
    if not cleaned:
        return "—"
    if cleaned.lower() in {"available", "unavailable"}:
        return cleaned.capitalize()
    return cleaned


def parse_backend_status(detail: str, *, cpu_only: bool | None = None) -> dict[str, Any]:
    """Parse ``backend_status()`` text into structured GPU, VRAM, capability and profile fields."""
    raw = str(detail or "").strip()
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    first = lines[0] if lines else ""
    is_cpu = (
        first.startswith(("CPU Workshop", "CPU UI preview"))
        or (is_cpu_only() if cpu_only is None else bool(cpu_only))
    )

    if is_cpu:
        return {
            "mode": "workshop",
            "status_label": "CPU Workshop",
            "headline": "CPU Workshop · No GPU backend attached",
            "message": raw or CPU_ONLY_MESSAGE,
            "gpu": None,
            "vram": None,
            "can_open_comfyui": False,
            "can_unload_vram": False,
            "capabilities": [
                {"label": "Sparse attention", "value": "—"},
                {"label": "Dense attention", "value": "—"},
                {"label": "Memory profile", "value": "CPU only"},
                {"label": "FirstBlockCache", "value": "Unavailable"},
                {"label": "EasyCache", "value": "Unavailable"},
                {"label": "Spectrum accelerator", "value": "Unavailable"},
            ],
            "profiles": [],
            "raw": raw,
        }

    if first.startswith("Connected"):
        parts = [part.strip() for part in first.split(" · ") if part.strip()]
        gpu = "unknown GPU"
        for part in parts[1:]:
            if "GiB VRAM" not in part and ":" not in part:
                gpu = part
                break

        vram: dict[str, Any] | None = None
        free_match = _VRAM_FREE_RE.search(first)
        if free_match:
            free_gib = float(free_match.group(1))
            total_gib = float(free_match.group(2))
            used_gib = max(0.0, round(total_gib - free_gib, 1))
            used_pct = round((used_gib / total_gib) * 100.0, 1) if total_gib > 0 else 0.0
            vram = {
                "free_gib": free_gib,
                "total_gib": total_gib,
                "used_gib": used_gib,
                "used_pct": used_pct,
            }
        else:
            total_match = _VRAM_TOTAL_RE.search(first)
            if total_match:
                vram = {
                    "free_gib": None,
                    "total_gib": float(total_match.group(1)),
                    "used_gib": None,
                    "used_pct": None,
                }

        kv: dict[str, str] = {}
        for part in parts[1:]:
            if ":" in part:
                k, v = part.split(":", 1)
                kv[k.strip().lower()] = v.strip()

        spectrum = "Unavailable"
        profiles: list[dict[str, str]] = []
        for line in lines[1:]:
            match = _PROFILE_RE.match(line)
            if match:
                name = match.group(1).strip()
                info = match.group(2).replace("`", "").strip()
                if name.lower() == "spectrum accelerator":
                    spectrum = _cap_value(info)
                else:
                    profiles.append({"name": name, "detail": info})
            else:
                profiles.append({"name": "Diagnostic", "detail": line.replace("`", "")})

        capabilities = [
            {"label": "Sparse attention", "value": _cap_value(kv.get("sparse"))},
            {"label": "Dense attention", "value": _cap_value(kv.get("dense"))},
            {"label": "Memory profile", "value": _cap_value(kv.get("memory"))},
            {"label": "FirstBlockCache", "value": _cap_value(kv.get("firstblockcache"))},
            {"label": "EasyCache", "value": _cap_value(kv.get("easycache"))},
            {"label": "Spectrum accelerator", "value": spectrum},
        ]
        return {
            "mode": "studio",
            "status_label": "Ready",
            "headline": f"Connected · {gpu}",
            "message": first,
            "gpu": gpu,
            "vram": vram,
            "can_open_comfyui": True,
            "can_unload_vram": True,
            "capabilities": capabilities,
            "profiles": profiles,
            "raw": raw,
        }

    return {
        "mode": "attention",
        "status_label": "Backend needs attention",
        "headline": first or "Backend unavailable",
        "message": raw or "Backend unavailable",
        "gpu": None,
        "vram": None,
        "can_open_comfyui": True,
        "can_unload_vram": False,
        "capabilities": [
            {"label": "Sparse attention", "value": "—"},
            {"label": "Dense attention", "value": "—"},
            {"label": "Memory profile", "value": "—"},
            {"label": "FirstBlockCache", "value": "Unavailable"},
            {"label": "EasyCache", "value": "Unavailable"},
            {"label": "Spectrum accelerator", "value": "Unavailable"},
        ],
        "profiles": [],
        "raw": raw,
    }


def _format_import_summary(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No projects were imported."
    return f"{format_import_summary_headline(results)}."


@dataclass(frozen=True)
class SystemPageServices:
    """Application services consumed by the standalone System page."""

    backend_status: Callable[[], str]
    unload_all_models: Callable[[], tuple[str, str]]
    store: Any
    runtime_config: Callable[[], Any]
    index: Callable[[], Any] | None = None
    model_specs: Mapping[str, Any] = ()  # type: ignore[assignment]
    models_root: Callable[[], Path] = lambda: Path("models")
    on_changed: Callable[[], None] | None = None


@dataclass
class SystemRun(Run):
    """Long System operation (unload, scan, project import) exposed to polling."""

    results: list[dict[str, Any]] | None = None
    backend: dict[str, Any] | None = None

    def public(self, scene_id=None) -> dict[str, Any]:
        return {
            **super().public(scene_id),
            "results": self.results,
            "backend": self.backend,
        }


class SystemPageState:
    """Server-side state and actions for ``/system/``."""

    @classmethod
    def from_bootstrap(
        cls,
        catalog,
        services,
        *,
        store,
        index: Callable[[], Any] | None,
        model_specs: Mapping[str, Any],
        on_changed: Callable[[], None] | None = None,
    ) -> SystemPageState:
        runtime_fn = getattr(services.engine_deps, "runtime_config", None)
        if runtime_fn is None:
            def runtime_fn():
                return None
        return cls(
            SystemPageServices(
                backend_status=services.backend_status,
                unload_all_models=services.unload_all_models,
                store=store,
                runtime_config=runtime_fn,
                index=index,
                model_specs=model_specs,
                models_root=lambda: catalog.COMFY_DIR / "models",
                on_changed=on_changed,
            )
        )

    def __init__(self, services: SystemPageServices):
        self.services = services
        self.lock = RLock()
        self._models_cache: dict[str, Any] | None = None
        self._models_checked_at = 0.0

    def backend(self) -> dict[str, Any]:
        return parse_backend_status(self.services.backend_status())

    def drama3(self) -> dict[str, Any]:
        cfg = load_drama3_settings(self.services.store)
        models = [
            {"id": value, "label": label}
            for label, value in model_choices_for_backend(cfg.backend)
        ]
        backends = [
            {
                "id": value,
                "label": label,
                "badge": _BACKEND_DESC.get(value, ("", ""))[0],
                "description": _BACKEND_DESC.get(value, ("", ""))[1],
            }
            for label, value in BACKEND_CHOICES
        ]
        experts_choices = [
            {"id": value, "label": label}
            for label, value in COMFY_EXPERTS_CHOICES
        ]
        summary_md = describe_drama3_settings(cfg)
        return {
            "backend": cfg.backend,
            "backend_label": cfg.backend_label,
            "model": cfg.active_model_key,
            "model_label": cfg.active_model_label,
            "comfy_experts": cfg.comfy_experts,
            "show_comfy_experts": cfg.backend == BACKEND_COMFYUI,
            "backends": backends,
            "models": models,
            "experts_choices": experts_choices,
            "summary": summary_md,
            "summary_plain": plain(summary_md),
            "is_a100": is_a100(),
        }

    def update_drama3(self, backend: str, model: str, comfy_experts: str) -> dict[str, Any]:
        _saved, status_md = apply_drama3_settings_update(
            backend, model, comfy_experts, store=self.services.store
        )
        error: str | None = None
        if _DRAMA3_ERROR_MARKER in status_md:
            _, detail = status_md.split(_DRAMA3_ERROR_MARKER, 1)
            error = plain(detail)
        elif self.services.on_changed is not None:
            self.services.on_changed()
        return {"drama3": self.drama3(), "error": error}

    def local_projects(self) -> dict[str, Any]:
        items = list_local_projects(self.services.store, self.services.runtime_config())
        counts: dict[str, int] = {"all": len(items), **{kind: 0 for kind in PROJECT_KINDS}}
        enriched: list[dict[str, Any]] = []
        for item in items:
            kind = str(item.get("kind") or "")
            if kind in counts:
                counts[kind] += 1
            enriched.append(
                {
                    **item,
                    "kind_label": KIND_LABELS.get(kind, kind.title()),
                }
            )
        return {"projects": enriched, "counts": counts}

    def rename_project(self, kind: str, project_id: str, name: str) -> dict[str, Any]:
        outcome = rename_local_project(
            kind,
            project_id,
            name,
            store=self.services.store,
            runtime=self.services.runtime_config(),
            on_changed=self.services.on_changed,
        )
        return {"outcome": outcome, **self.local_projects()}

    def delete_project(self, kind: str, project_id: str) -> dict[str, Any]:
        outcome = delete_local_project(
            kind,
            project_id,
            store=self.services.store,
            runtime=self.services.runtime_config(),
            on_changed=self.services.on_changed,
        )
        return {"outcome": outcome, **self.local_projects()}

    def remote_projects(
        self, url: str, modal_key: str = "", modal_secret: str = ""
    ) -> dict[str, Any]:
        headers = build_auth_headers(modal_key, modal_secret)
        data = fetch_remote_projects(url, headers=headers)
        projects: list[dict[str, Any]] = []
        counts: dict[str, int] = {"all": 0, **{kind: 0 for kind in PROJECT_KINDS}}
        for item in data.get("projects", ()):
            if not isinstance(item, dict) or not item.get("kind") or not item.get("id"):
                continue
            kind = str(item.get("kind") or "")
            counts["all"] += 1
            counts[kind] = counts.get(kind, 0) + 1
            projects.append(
                {
                    **item,
                    "kind_label": KIND_LABELS.get(kind, kind.title()),
                }
            )
        projects.sort(
            key=lambda item: (
                _KIND_ORDER.get(str(item.get("kind") or ""), len(_KIND_ORDER)),
                str(item.get("kind") or ""),
            )
        )
        return {
            "app": data.get("app") or "remote server",
            "projects": projects,
            "counts": counts,
        }

    def models_inventory(self, *, refresh: bool = False) -> dict[str, Any]:
        with self.lock:
            now_mono = time.monotonic()
            if (
                self._models_cache is not None
                and not refresh
                and now_mono - self._models_checked_at < MODEL_CACHE_TTL
            ):
                return self._models_cache

            root = Path(self.services.models_root())
            items: list[dict[str, Any]] = []
            seen_families: set[str] = set()
            present_count = 0
            for key, spec in dict(self.services.model_specs).items():
                path = root / spec.folder / spec.local_name
                try:
                    present = path.is_file() and path.stat().st_size > 0
                except OSError:
                    present = False
                if present:
                    present_count += 1
                family = model_family(key)
                seen_families.add(family)
                items.append(
                    {
                        "key": key,
                        "family": family,
                        "folder": spec.folder,
                        "file": spec.local_name,
                        "path": f"{spec.folder}/{spec.local_name}",
                        "present": present,
                    }
                )
            total_count = len(items)
            ordered_families = [f for f in _FAMILY_ORDER if f in seen_families] + sorted(
                seen_families - set(_FAMILY_ORDER)
            )
            payload = {
                "items": items,
                "present_count": present_count,
                "missing_count": total_count - present_count,
                "total_count": total_count,
                "families": ordered_families,
                "cached_seconds": int(MODEL_CACHE_TTL),
                "checked_at": time.time(),
            }
            self._models_cache = payload
            self._models_checked_at = now_mono
            return payload

    def engines(self) -> list[dict[str, Any]]:
        return engine_entries()


def install_system_page(app: FastAPI, *, pages: SystemPageState) -> SystemPageState:
    """Mount ``/system/`` and ``/system/api/*`` on ``app``."""
    install_static_page(app, ROUTE, STATIC, STATIC_TYPES)
    import_runs: Runs | None = None
    import_runs = Runs(
        BUSY[IMPORT],
        blocked=lambda _run: (
            BUSY[IMPORT]
            if import_runs is not None and any(r.finished_at is None for r in import_runs.items.values())
            else None
        ),
    )
    runs = {
        UNLOAD: Runs(BUSY[UNLOAD]),
        SCAN: Runs(BUSY[SCAN]),
        IMPORT: import_runs,
    }

    def _owner(request: Request) -> str:
        try:
            return owner_of(request)
        except (ValueError, AttributeError):
            return "local"

    def run_payloads(owner: str) -> dict[str, Any]:
        return {
            kind: run.public(kind) if (run := runs[kind].latest(owner, kind)) else None
            for kind in runs
        }

    def start(owner: str, kind: str, work: Callable[[SystemRun], None], status: str) -> dict[str, Any]:
        run = SystemRun(owner, kind, kind)
        run.status = status
        runs[kind].start(run, work)
        return {"runs": run_payloads(owner)}

    @app.get(f"{API}/boot", include_in_schema=False)
    @guarded
    def boot(request: Request):
        owner = _owner(request)
        with gemini.owner_context(owner):
            return {
                "backend": pages.backend(),
                "drama3": pages.drama3(),
                "gemini_key": gemini.session_key_status(owner),
                "projects": pages.local_projects(),
                "models": pages.models_inventory(),
                "engines": pages.engines(),
                "runs": run_payloads(owner),
                "cpu_only": is_cpu_only(),
            }

    @app.get(f"{API}/backend", include_in_schema=False)
    @guarded
    def backend(request: Request):
        with gemini.owner_context(_owner(request)):
            return {"backend": pages.backend()}

    @app.post(f"{API}/unload", include_in_schema=False)
    @guarded
    def unload(request: Request):
        owner = _owner(request)
        if is_cpu_only():
            raise H3Error(CPU_ONLY_MESSAGE)

        def work(run: SystemRun):
            message, detail = pages.services.unload_all_models()
            cleaned = plain(message)
            run.status = cleaned
            run.message = cleaned
            run.backend = parse_backend_status(detail)
            run.revision += 1

        return start(owner, UNLOAD, work, "Unloading models and freeing VRAM…")

    @app.post(f"{API}/drama3", include_in_schema=False)
    @guarded
    def update_drama3(request: Request, body: dict = Body(...)):
        if not isinstance(body, dict):
            raise H3Error("Invalid Drama 3 request body.")
        with gemini.owner_context(_owner(request)):
            return pages.update_drama3(
                str(body.get("backend") or ""),
                str(body.get("model") or ""),
                str(body.get("comfy_experts") or ""),
            )

    @app.get(f"{API}/gemini-key", include_in_schema=False)
    @guarded
    def get_gemini_key(request: Request):
        owner = _owner(request)
        return {"gemini_key": gemini.session_key_status(owner)}

    @app.post(f"{API}/gemini-key", include_in_schema=False)
    @guarded
    def save_gemini_key(request: Request, body: dict = Body(...)):
        if not isinstance(body, dict):
            raise H3Error("Invalid Gemini API key request.")
        owner = _owner(request)
        api_key = str(body.get("api_key") or "").strip()
        gemini.set_session_key(owner, api_key)
        return {
            "gemini_key": gemini.session_key_status(owner),
            "message": (
                "Session Gemini API key saved in server memory for this browser session."
                if api_key
                else "Session Gemini API key cleared."
            ),
        }

    @app.delete(f"{API}/gemini-key", include_in_schema=False)
    @guarded
    def clear_gemini_key(request: Request):
        owner = _owner(request)
        gemini.clear_session_key(owner)
        return {
            "gemini_key": gemini.session_key_status(owner),
            "message": "Session Gemini API key cleared.",
        }

    @app.get(f"{API}/projects", include_in_schema=False)
    @guarded
    def projects(request: Request):
        _owner(request)
        return pages.local_projects()

    @app.post(f"{API}/projects/rename", include_in_schema=False)
    @guarded
    def project_rename(request: Request, body: dict = Body(...)):
        if not isinstance(body, dict):
            raise H3Error("Invalid rename request.")
        _owner(request)
        kind = str(body.get("kind") or "")
        project_id = str(body.get("id") or "")
        new_name = str(body.get("name") or "")
        if not kind or not project_id:
            raise H3Error("Specify project kind and id.")
        return pages.rename_project(kind, project_id, new_name)

    @app.post(f"{API}/projects/delete", include_in_schema=False)
    @guarded
    def project_delete(request: Request, body: dict = Body(...)):
        if not isinstance(body, dict):
            raise H3Error("Invalid delete request.")
        _owner(request)
        kind = str(body.get("kind") or "")
        project_id = str(body.get("id") or "")
        if not kind or not project_id:
            raise H3Error("Specify project kind and id.")
        return pages.delete_project(kind, project_id)

    @app.post(f"{API}/projects/remote", include_in_schema=False)
    @guarded
    def project_remote(request: Request, body: dict = Body(...)):
        if not isinstance(body, dict):
            raise H3Error("Invalid remote server request.")
        _owner(request)
        return pages.remote_projects(
            str(body.get("url") or ""),
            str(body.get("modal_key") or ""),
            str(body.get("modal_secret") or ""),
        )

    @app.post(f"{API}/projects/import", include_in_schema=False)
    @guarded
    def project_import(request: Request, body: dict = Body(...)):
        if not isinstance(body, dict):
            raise H3Error("Invalid import request.")
        owner = _owner(request)
        url = str(body.get("url") or "")
        raw_items = body.get("items")
        if not isinstance(raw_items, list) or not raw_items:
            raise H3Error("Select at least one project to import.")
        selections: list[tuple[str, str] | tuple[str, str, str]] = []
        for entry in raw_items:
            if not isinstance(entry, dict) or not entry.get("kind") or not entry.get("id"):
                raise H3Error("Each item must specify kind and id.")
            if entry.get("name"):
                selections.append((str(entry["kind"]), str(entry["id"]), str(entry["name"])))
            else:
                selections.append((str(entry["kind"]), str(entry["id"])))
        conflict = str(body.get("conflict") or "rename")
        if conflict not in CONFLICT_MODES:
            raise H3Error(f"Unsupported conflict mode: {conflict!r}")
        headers = build_auth_headers(
            str(body.get("modal_key") or ""),
            str(body.get("modal_secret") or ""),
        )

        def work(run: SystemRun):
            def on_progress(_fraction: float, desc: str) -> None:
                run.status = str(desc or "Importing projects…")
                run.revision += 1

            results = import_remote_projects(
                url,
                selections,
                store=pages.services.store,
                runtime=pages.services.runtime_config(),
                conflict=conflict,
                headers=headers,
                progress=on_progress,
                owner=owner,
                on_imported=pages.services.on_changed,
            )
            summary = _format_import_summary(results)
            run.results = results
            run.message = summary
            run.status = summary
            run.revision += 1

        return start(owner, IMPORT, work, "Connecting to remote server…")

    @app.get(f"{API}/models", include_in_schema=False)
    @guarded
    def models(request: Request, refresh: bool = False):
        _owner(request)
        return pages.models_inventory(refresh=bool(refresh))

    @app.post(f"{API}/scan", include_in_schema=False)
    @guarded
    def scan(request: Request):
        owner = _owner(request)
        if pages.services.index is None or pages.services.index() is None:
            raise H3Error("Media index maintenance is unavailable.")

        def work(run: SystemRun):
            for message in pages.services.index().rebuild():
                cleaned = plain(message)
                run.status = cleaned
                run.message = cleaned
                run.revision += 1

        return start(owner, SCAN, work, "Scanning historical media…")

    @app.get(f"{API}/run", include_in_schema=False)
    @guarded
    def read_runs(request: Request):
        return {"runs": run_payloads(_owner(request))}

    return pages
