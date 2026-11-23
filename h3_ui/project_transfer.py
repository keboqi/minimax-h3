"""Cross-server project transfer routes and System tab import UI."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

import gradio as gr
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from h3_app.errors import H3Error
from h3_app.project_transfer import (
    CONFLICT_MODES,
    KIND_LABELS,
    TRANSFER_SCHEMA_VERSION,
    build_auth_headers,
    export_project_archive,
    fetch_remote_projects,
    import_remote_projects,
    list_local_projects,
)
from h3_app.workspace_store import request_owner
from h3_gpu import is_cpu_only


def _is_not_found_error(exc: Exception) -> bool:
    text = str(exc)
    return any(
        phrase in text
        for phrase in ("no longer exists", "Select a saved", "Choose a saved")
    )


def _archive_filename(kind: str, project_id: str) -> str:
    slug = re.sub(r"[^\w.-]+", "_", f"{kind}_{project_id}").strip("._")[:80] or "project"
    return f"{slug}.zip"


def _format_import_summary(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No projects were imported."
    counts: dict[str, int] = {}
    for item in results:
        status = str(item.get("status") or "imported")
        counts[status] = counts.get(status, 0) + 1
    parts = [
        f"{count} {status}"
        for status, count in counts.items()
        if count
    ]
    lines = "\n".join(f"- {item['message']}" for item in results)
    return f"Import finished ({', '.join(parts)}):\n\n{lines}"


def install_project_transfer_routes(
    app: FastAPI,
    *,
    store,
    runtime_config: Callable[[], Any],
    on_imported: Callable[[], None] | None = None,
) -> None:
    """Mount `/api/projects/transfer/{list,export,import}` on Studio or Workshop."""

    @app.get("/api/projects/transfer/list", include_in_schema=False)
    def transfer_list():
        return {
            "app": "Prism Workshop" if is_cpu_only() else "Prism Studio",
            "schema_version": TRANSFER_SCHEMA_VERSION,
            "projects": list_local_projects(store, runtime_config()),
        }

    @app.get("/api/projects/transfer/export", include_in_schema=False)
    def transfer_export(kind: str, id: str):
        tmp_dir = tempfile.mkdtemp(prefix="h3-project-export-")
        try:
            archive_path = export_project_archive(
                kind,
                id,
                Path(tmp_dir) / "project.zip",
                store=store,
                runtime=runtime_config(),
            )
        except (H3Error, ValueError) as exc:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            status_code = 404 if _is_not_found_error(exc) else 400
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        return FileResponse(
            archive_path,
            media_type="application/zip",
            filename=_archive_filename(kind, id),
            background=BackgroundTask(shutil.rmtree, tmp_dir, ignore_errors=True),
        )

    @app.post("/api/projects/transfer/import", include_in_schema=False)
    def transfer_import(request: Request, body: dict = Body(...)):
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Invalid request body.")
        url = str(body.get("url") or "")
        items = body.get("items")
        if not isinstance(items, list) or not items:
            raise HTTPException(status_code=400, detail="Select at least one project to import.")
        selections: list[tuple[str, str]] = []
        for entry in items:
            if not isinstance(entry, dict) or not entry.get("kind") or not entry.get("id"):
                raise HTTPException(status_code=400, detail="Each item must specify kind and id.")
            selections.append((str(entry["kind"]), str(entry["id"])))
        conflict = str(body.get("conflict") or "rename")
        try:
            headers = build_auth_headers(
                str(body.get("modal_key") or ""),
                str(body.get("modal_secret") or ""),
            )
            try:
                owner = request_owner(request, store)
            except ValueError:
                owner = "imported"
            results = import_remote_projects(
                url,
                selections,
                store=store,
                runtime=runtime_config(),
                conflict=conflict,
                headers=headers,
                owner=owner,
                on_imported=on_imported,
            )
        except H3Error as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"results": results}


def build_project_import_view(
    root,
    *,
    store,
    runtime_config: Callable[[], Any],
    on_imported: Callable[[], None] | None = None,
) -> None:
    """Build the remote project import panel inside the System tab."""
    if root is None:
        return
    with root, gr.Column(elem_classes=["h3-ops-panel"]):
        gr.Markdown(
            "### Import projects from another server\n"
            "Works with both Prism Studio and Prism Workshop URLs. Imports Soundstage, "
            "Storyboard, Dubbing, and Remake projects along with their clips, cast, takes, and results.",
            elem_classes=["h3-ops-section-title"],
        )
        with gr.Row():
            remote_url = gr.Textbox(
                label="Remote Studio or Workshop URL",
                placeholder="https://your-server.modal.run or http://host:7860",
                scale=1,
            )
            load_btn = gr.Button("Load projects", scale=0)
        with gr.Accordion("Authentication (optional)", open=False):
            with gr.Row():
                modal_key = gr.Textbox(label="Modal-Key")
                modal_secret = gr.Textbox(label="Modal-Secret", type="password")
        projects = gr.CheckboxGroup(
            label="Projects on remote server",
            choices=[],
            value=[],
        )
        with gr.Row():
            select_all = gr.Button("Select all", scale=0)
            clear_sel = gr.Button("Clear selection", scale=0)
            conflict = gr.Radio(
                [
                    ("Import as copy (rename)", "rename"),
                    ("Skip existing", "skip"),
                    ("Overwrite existing", "overwrite"),
                ],
                value="rename",
                label="If a project already exists locally",
            )
        import_btn = gr.Button("Import selected projects", variant="primary", scale=0)
        status = gr.Markdown(elem_classes=["h3-ops-note"])
        loaded_choices = gr.State([])

    def load_projects(url: str, key: str, secret: str):
        try:
            headers = build_auth_headers(key, secret)
            data = fetch_remote_projects(url, headers=headers)
        except H3Error as exc:
            return gr.update(choices=[], value=[]), [], f"**Error:** {exc}"
        choices = [
            (
                f"[{KIND_LABELS.get(item['kind'], item['kind'])}] {item['name']}"
                + (f" · {item['summary']}" if item.get("summary") else ""),
                json.dumps([item["kind"], item["id"]]),
            )
            for item in data["projects"]
            if isinstance(item, dict) and item.get("kind") and item.get("id")
        ]
        server_name = data.get("app") or "remote server"
        if not choices:
            return (
                gr.update(choices=[], value=[]),
                [],
                f"Connected to {server_name}, but no saved projects were found.",
            )
        return (
            gr.update(choices=choices, value=[]),
            choices,
            f"Loaded {len(choices)} project(s) from {server_name}. Select projects to import.",
        )

    def select_all_projects(choices):
        return [value for _label, value in (choices or [])]

    def clear_selection():
        return []

    def run_import(
        url: str,
        key: str,
        secret: str,
        selected: list[str] | None,
        conflict_mode: str,
        progress=gr.Progress(),
    ) -> str:
        try:
            headers = build_auth_headers(key, secret)
            if conflict_mode not in CONFLICT_MODES:
                raise H3Error(f"Unsupported conflict mode: {conflict_mode!r}")
            selections: list[tuple[str, str]] = []
            for entry in selected or []:
                parsed = json.loads(entry)
                if isinstance(parsed, list) and len(parsed) == 2:
                    selections.append((str(parsed[0]), str(parsed[1])))
            results = import_remote_projects(
                url,
                selections,
                store=store,
                runtime=runtime_config(),
                conflict=conflict_mode,
                headers=headers,
                progress=lambda fraction, desc: progress(fraction, desc=desc),
                on_imported=on_imported,
            )
        except (H3Error, ValueError, TypeError) as exc:
            return f"**Error:** {exc}"
        return _format_import_summary(results)

    load_btn.click(
        load_projects,
        inputs=[remote_url, modal_key, modal_secret],
        outputs=[projects, loaded_choices, status],
        api_name=False,
    )
    select_all.click(
        select_all_projects,
        inputs=[loaded_choices],
        outputs=[projects],
        queue=False,
        api_name=False,
    )
    clear_sel.click(
        clear_selection,
        outputs=[projects],
        queue=False,
        api_name=False,
    )
    import_btn.click(
        run_import,
        inputs=[remote_url, modal_key, modal_secret, projects, conflict],
        outputs=[status],
        api_name=False,
    )
