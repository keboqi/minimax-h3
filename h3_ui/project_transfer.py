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
    delete_local_project,
    export_project_archive,
    fetch_remote_projects,
    import_remote_projects,
    list_local_projects,
    rename_local_project,
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


def _project_key(kind: str, project_id: str) -> str:
    return json.dumps([kind, project_id])


def _parse_project_key(raw: str | None) -> tuple[str, str]:
    if not raw:
        raise H3Error("Select a project first.")
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise H3Error("Select a valid project.") from exc
    if not isinstance(parsed, list) or len(parsed) != 2 or not parsed[0] or not parsed[1]:
        raise H3Error("Select a valid project.")
    return str(parsed[0]), str(parsed[1])


def _project_choice_label(item: dict[str, Any]) -> str:
    label = f"[{KIND_LABELS.get(item['kind'], item['kind'])}] {item['name']}"
    if item.get("summary"):
        label += f" · {item['summary']}"
    return label


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

    @app.post("/api/projects/manage/rename", include_in_schema=False)
    def manage_rename(body: dict = Body(...)):
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Invalid request body.")
        kind = str(body.get("kind") or "")
        project_id = str(body.get("id") or "")
        new_name = str(body.get("name") or "")
        if not kind or not project_id:
            raise HTTPException(status_code=400, detail="Specify project kind and id.")
        try:
            return rename_local_project(
                kind,
                project_id,
                new_name,
                store=store,
                runtime=runtime_config(),
                on_changed=on_imported,
            )
        except H3Error as exc:
            status_code = 404 if _is_not_found_error(exc) else 400
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    @app.post("/api/projects/manage/delete", include_in_schema=False)
    def manage_delete(body: dict = Body(...)):
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Invalid request body.")
        kind = str(body.get("kind") or "")
        project_id = str(body.get("id") or "")
        if not kind or not project_id:
            raise HTTPException(status_code=400, detail="Specify project kind and id.")
        try:
            return delete_local_project(
                kind,
                project_id,
                store=store,
                runtime=runtime_config(),
                on_changed=on_imported,
            )
        except H3Error as exc:
            status_code = 404 if _is_not_found_error(exc) else 400
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc


def build_project_manage_view(
    root,
    *,
    store,
    runtime_config: Callable[[], Any],
    on_changed: Callable[[], None] | None = None,
) -> None:
    """Build the local project rename/remove panel inside the System tab."""
    if root is None:
        return

    def load_local_choices(selected_kind: str = "all", preferred_key: str | None = None):
        items = list_local_projects(store, runtime_config())
        if selected_kind in KIND_LABELS:
            items = [item for item in items if item.get("kind") == selected_kind]
        choices: list[tuple[str, str]] = []
        names_map: dict[str, str] = {}
        for item in items:
            kind = item.get("kind")
            pid = item.get("id")
            if not kind or not pid:
                continue
            key = _project_key(str(kind), str(pid))
            choices.append((_project_choice_label(item), key))
            names_map[key] = str(item.get("name") or pid)
        values = [val for _label, val in choices]
        selected = preferred_key if preferred_key in values else (values[0] if values else None)
        selected_name = names_map.get(selected, "") if selected else ""
        return choices, selected, selected_name, names_map

    initial_choices, initial_value, initial_name, initial_names = load_local_choices()

    with root, gr.Column(elem_classes=["h3-ops-panel"]):
        gr.Markdown(
            "### Manage local projects\n"
            "Rename or remove saved Soundstage, Storyboard, Dubbing, and Remake projects on this server.",
            elem_classes=["h3-ops-section-title"],
        )
        with gr.Row():
            kind_filter = gr.Radio(
                [
                    ("All", "all"),
                    ("Soundstage", "soundstage"),
                    ("Storyboard", "storyboard"),
                    ("Dubbing", "dubbing"),
                    ("Remake", "remake"),
                ],
                value="all",
                label="Project type",
                scale=1,
            )
            refresh_btn = gr.Button("Refresh list", scale=0)
        project_select = gr.Dropdown(
            label="Local project",
            choices=initial_choices,
            value=initial_value,
            interactive=True,
        )
        with gr.Row():
            new_name = gr.Textbox(
                label="New name",
                value=initial_name,
                placeholder="Enter new project name",
                scale=1,
            )
            rename_btn = gr.Button("Rename project", scale=0)
            delete_btn = gr.Button("Remove project", variant="stop", scale=0)
        manage_status = gr.Markdown(elem_classes=["h3-ops-note"])
        names_state = gr.State(initial_names)

    def refresh_projects(selected_kind: str, current_key: str | None = None):
        choices, selected, selected_name, names_map = load_local_choices(selected_kind, preferred_key=current_key)
        msg = f"Showing {len(choices)} project(s)." if choices else "No saved projects match this filter."
        return gr.update(choices=choices, value=selected), selected_name, names_map, msg

    def sync_selected_name(selected_key: str | None, current_names: dict[str, str] | None) -> str:
        if not selected_key or not isinstance(current_names, dict):
            return ""
        return str(current_names.get(selected_key) or "")

    def rename_selected(
        selected_kind: str,
        selected_key: str | None,
        target_name: str,
        current_names: dict[str, str] | None,
    ):
        try:
            kind, project_id = _parse_project_key(selected_key)
            outcome = rename_local_project(
                kind,
                project_id,
                target_name,
                store=store,
                runtime=runtime_config(),
                on_changed=on_changed,
            )
        except H3Error as exc:
            return gr.update(), target_name, current_names or {}, f"**Error:** {exc}"
        new_key = _project_key(kind, str(outcome["id"]))
        choices, selected, selected_name, names_map = load_local_choices(selected_kind, preferred_key=new_key)
        return gr.update(choices=choices, value=selected), selected_name, names_map, outcome["message"]

    def delete_selected(
        selected_kind: str,
        selected_key: str | None,
        current_names: dict[str, str] | None,
    ):
        try:
            kind, project_id = _parse_project_key(selected_key)
            outcome = delete_local_project(
                kind,
                project_id,
                store=store,
                runtime=runtime_config(),
                on_changed=on_changed,
            )
        except H3Error as exc:
            fallback_name = current_names.get(selected_key or "", "") if isinstance(current_names, dict) else ""
            return gr.update(), fallback_name, current_names or {}, f"**Error:** {exc}"
        choices, selected, selected_name, names_map = load_local_choices(selected_kind)
        return gr.update(choices=choices, value=selected), selected_name, names_map, outcome["message"]

    kind_filter.change(
        refresh_projects,
        inputs=[kind_filter, project_select],
        outputs=[project_select, new_name, names_state, manage_status],
        api_name=False,
    )
    refresh_btn.click(
        refresh_projects,
        inputs=[kind_filter, project_select],
        outputs=[project_select, new_name, names_state, manage_status],
        api_name=False,
    )
    project_select.change(
        sync_selected_name,
        inputs=[project_select, names_state],
        outputs=[new_name],
        queue=False,
        api_name=False,
    )
    rename_btn.click(
        rename_selected,
        inputs=[kind_filter, project_select, new_name, names_state],
        outputs=[project_select, new_name, names_state, manage_status],
        api_name=False,
    )
    delete_btn.click(
        delete_selected,
        inputs=[kind_filter, project_select, names_state],
        outputs=[project_select, new_name, names_state, manage_status],
        api_name=False,
    )


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
                _project_choice_label(item),
                _project_key(str(item["kind"]), str(item["id"])),
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
