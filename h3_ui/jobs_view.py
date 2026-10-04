"""An owner-scoped monitor; Gradio remains the only dispatcher."""

from html import escape
import json
from pathlib import Path
import time
import uuid

import gradio as gr

from h3_app.jobs import JOBS, JobCancelled
from .job_admission import execute_accepted, require_owner, dispatch_failed
from .job_bindings import GPU_QUEUE


def table(jobs):
    if not jobs:
        return '<p role="status">No jobs for this browser owner yet.</p>'
    rows = []
    for job in jobs:
        cells = (
            job.id[:8],
            job.family,
            job.state,
            job.stage,
            f"{max(0, int((job.finished_at or time.time()) - job.created_at))}s",
            len(job.outputs),
        )
        rows.append(
            "<tr>"
            + "".join(f"<td>{escape(str(cell))}</td>" for cell in cells)
            + "</tr>"
        )
    return (
        '<div class="h3-job-table"><table><caption>Jobs for this browser owner</caption>'
        "<thead><tr><th>Job</th><th>Engine</th><th>State</th><th>Stage</th><th>Elapsed</th><th>Outputs</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def build_jobs_view(root, *, get, post, summary):
    with root:
        gr.Markdown(
            "## Jobs\nTrack requests, cancel active work, or retry failed variants. "
            "Select a job to inspect its outputs and recorded settings."
        )
        listing = gr.HTML(table(()))
        refresh = gr.Button("Refresh jobs")
        selected = gr.Dropdown([], label="Job", value=None)
        details = gr.HTML(elem_classes=["h3-job-details"])
        variants = gr.CheckboxGroup([], label="Failed variants to retry", value=[])
        with gr.Row():
            cancel = gr.Button("Cancel selected job", interactive=False)
            retry = gr.Button("Retry failed variants", interactive=False)
            finishing = gr.Button("Retry finishing only", interactive=False)
        finishing_variant = gr.Dropdown(
            [], label="Retained H3 source variant", value=None
        )
        files = gr.File(
            label="Retained outputs", file_count="multiple", interactive=False
        )
        status = gr.Markdown()
        ticket = gr.Textbox(visible=False)
        timer = gr.Timer(2)
        presentation = gr.State({})
        with gr.Accordion("Browser ownership and recovery", open=False):
            gr.Markdown(
                "Technical job history survives server restarts. Unsaved prompts and uploaded "
                "input copies expire after 30 minutes or a restart. Save a project to retain them. "
                "This browser's signed ownership cookie expires after 30 days. "
                "Recovery never automatically repeats inference. "
                "Keep a recovery key to reopen jobs and projects after losing browser cookies. "
                "The key grants access to your saved content; keep it private."
            )
            recovery_key = gr.Textbox(
                label="Browser recovery key", type="password", interactive=False
            )
            export_owner = gr.Button("Show my recovery key")
            recovery_input = gr.Textbox(
                label="Restore browser recovery key", type="password"
            )
            recover_owner = gr.Button("Restore owner and reload")
        reconcile = gr.Button("Reconnect to backend history")
        with gr.Accordion("Saved projects", open=False):
            gr.Markdown(
                "**Saving retains the selected job's prompt and source files on this server.** "
                "Projects belong to this browser owner. Provider keys are never saved."
            )
            project_name = gr.Textbox(label="Project name", max_lines=1)
            save_project = gr.Button("Save prompt, settings & sources")
            project = gr.Dropdown([], label="Saved project", value=None)
            saved_prompt = gr.Textbox(
                label="Saved prompt / caption / lyrics", interactive=False, lines=5
            )
            saved_settings = gr.JSON(label="Saved request settings")
            with gr.Row():
                refresh_projects = gr.Button("Refresh projects")
                run_project = gr.Button("Run saved request", variant="primary")
            confirm_project_delete = gr.Checkbox(
                label="Delete the selected project and its retained inputs"
            )
            delete_project = gr.Button("Delete project")

    def owner_key(request: gr.Request):
        owner = require_owner(request)
        if owner.startswith("api:"):
            raise gr.Error(
                "Open the workspace in a browser to create a durable recovery key."
            )
        return JOBS.store.recovery_key(owner)

    export_owner.click(owner_key, outputs=recovery_key, queue=False, api_name=False)
    recover_owner.click(
        None,
        inputs=recovery_input,
        outputs=None,
        queue=False,
        api_name=False,
        js="async (key) => { const r = await fetch('/workspace/owner/recover', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({key})}); if (!r.ok) { alert('Invalid browser recovery key'); return; } location.reload(); }",
    )

    def refresh_jobs(
        selected_id, failed_selection, source_selection, previous, request: gr.Request
    ):
        jobs = JOBS.list_owned(require_owner(request))
        choices = [
            (f"{job.id[:8]} · {job.family} · {job.state}", job.id) for job in jobs
        ]
        value = selected_id if any(job.id == selected_id for job in jobs) else None
        queued = sum(job.state == "queued" for job in jobs)
        active = sum(job.finished_at is None and job.state != "queued" for job in jobs)
        inspected = list(inspect_job(value, request))
        signature = json.dumps(inspected, sort_keys=True)
        failed_choices = inspected[1].get("choices")
        previous = previous or {}
        if previous.get("selected") == value:
            if previous.get("failed") == inspected[1].get("choices"):
                inspected[1]["value"] = failed_selection
            sources = [item[1] for item in inspected[5].get("choices", [])]
            if source_selection in sources:
                inspected[5]["value"] = source_selection
        if previous.get("detail") == signature and previous.get("selected") == value:
            inspected = [gr.skip() for _ in inspected]
        next_presentation = {
            "selected": value,
            "detail": signature,
            "choices": choices,
            "failed": failed_choices,
            "table": table(jobs),
            "counts": (queued, active),
        }
        return (
            (
                gr.skip()
                if previous.get("table") == next_presentation["table"]
                else next_presentation["table"]
            ),
            (
                gr.skip()
                if previous.get("choices") == choices and value == selected_id
                else gr.update(choices=choices, value=value)
            ),
            (
                gr.skip()
                if previous.get("counts") == (queued, active)
                else f'<p class="h3-system-status" role="status">{queued} queued · {active} active in this session</p>'
            ),
            *inspected,
            next_presentation,
        )

    for event in (timer.tick, refresh.click):
        event(
            refresh_jobs,
            inputs=[selected, variants, finishing_variant, presentation],
            outputs=[
                listing,
                selected,
                summary,
                details,
                variants,
                cancel,
                retry,
                finishing,
                finishing_variant,
                files,
                presentation,
            ],
            queue=False,
            api_name=False,
            show_progress="hidden",
        )

    def reconnect(request: gr.Request):
        try:
            JOBS.reconcile(require_owner(request), get)
            return "Backend history reconciled. Unknown submissions require manual review before a new request."
        except Exception as exc:
            return f"Backend history is unavailable: {exc}. No work was replayed."

    reconcile.click(reconnect, outputs=status, queue=False, api_name=False)

    def project_choices(request: gr.Request):
        rows = JOBS.store.projects(require_owner(request))
        return gr.update(choices=[(p["name"], p["id"]) for p in rows], value=None)

    refresh_projects.click(
        project_choices, outputs=project, queue=False, api_name=False
    )

    def save(job_id, name, request: gr.Request):
        owner = require_owner(request)
        job = JOBS.owned(owner, job_id)
        try:
            project_id = JOBS.store.save_project(owner, job, name)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        choices = project_choices(request)
        return {
            **choices,
            "value": project_id,
        }, "Project saved, including prompt and source media."

    save_project.click(
        save,
        inputs=[selected, project_name],
        outputs=[project, status],
        queue=False,
        api_name=False,
    )

    def inspect_project(project_id, request: gr.Request):
        if not project_id:
            return "", None
        saved = JOBS.store.project(require_owner(request), project_id)
        callback = JOBS.callbacks.get(saved["family"])
        names = getattr(callback, "job_input_names", ())
        if len(names) != len(saved["values"]):
            return "", {
                "engine": saved["family"],
                "status": "Engine unavailable for inspecting this saved request.",
            }
        values = dict(zip(names, saved["values"], strict=True))
        content = "\n\n".join(
            str(values.pop(name))
            for name in ("prompt", "caption", "lyrics")
            if values.get(name)
        )
        for index in saved["media_indices"]:
            value = values.get(names[index])
            values[names[index]] = (
                [Path(item).name for item in value]
                if isinstance(value, list)
                else Path(value).name
                if isinstance(value, str)
                else value
            )
        return content, {
            "engine": saved["family"],
            "settings": values,
            "canvas": saved.get("canvas_request"),
        }

    project.change(
        inspect_project,
        inputs=project,
        outputs=[saved_prompt, saved_settings],
        queue=False,
        api_name=False,
    )

    def remove_project(project_id, confirmed, request: gr.Request):
        if not confirmed:
            raise gr.Error("Select the project deletion confirmation first.")
        owner = require_owner(request)
        JOBS.store.project(owner, project_id)
        if any(
            job.finished_at is None and job.retry_of == project_id
            for job in JOBS.list_owned(owner)
        ):
            raise gr.Error(
                "Wait for this project's active run to finish before deleting it."
            )
        JOBS.store.delete_project(owner, project_id)
        JOBS.forget_project(owner, project_id)
        return (
            project_choices(request),
            False,
            "Project and retained inputs deleted. Generated outputs remain in Media.",
        )

    delete_project.click(
        remove_project,
        inputs=[project, confirm_project_delete],
        outputs=[project, confirm_project_delete, status],
        queue=False,
        api_name=False,
    )

    def accept_project(project_id, request: gr.Request):
        owner = require_owner(request)
        saved = JOBS.store.project(owner, project_id)
        callback = JOBS.callbacks.get(saved["family"])
        if callback is None:
            raise gr.Error(
                "This project's engine is unavailable in the current installation."
            )
        job = JOBS.accept(
            owner,
            saved["family"],
            saved["values"],
            callback=callback,
            media_indices=saved["media_indices"],
            output_indices=saved["output_indices"],
            key=uuid.uuid4().hex,
            retry_of=project_id,
        )
        job.variant_seeds = {int(k): v for k, v in saved["variant_seeds"].items()}
        job.replay_indices = tuple(sorted(job.variant_seeds))
        job.replay_seeds = (
            tuple(job.variant_seeds[i] for i in job.replay_indices)
            if job.family == "h3"
            else ()
        )
        job.replay_ledger = saved["ledger"]
        job.canvas_request = saved.get("canvas_request")
        job.source_asset_ids = saved.get("source_asset_ids", job.source_asset_ids)
        job.persist()
        return job.id

    def inspect_job(job_id, request: gr.Request):
        if not job_id:
            return (
                "",
                gr.update(choices=[], value=[]),
                *(gr.update(interactive=False) for _ in range(3)),
                gr.update(choices=[], value=None),
                [],
            )
        job = JOBS.owned(require_owner(request), job_id)
        safe_ledger = [
            {key: value for key, value in entry.items() if key != "graph_json"}
            for entry in job.ledger
        ]
        detail = (
            '<p role="status">' + escape(job.error or job.stage) + "</p>"
            "<details><summary>Recorded execution and seeds</summary><pre>"
            + escape(json.dumps(safe_ledger, indent=2))
            + "</pre></details>"
        )
        failed = sorted(job.failed_variants)
        retryable = bool(
            job.finished_at
            and job.callback
            and failed
            and all(i in job.variant_seeds for i in failed)
            and not any(entry["state"] == "submission_unknown" for entry in job.ledger)
        )
        sources = [
            (f"Variant {i + 1}", i)
            for i, path in job.recoverable_sources.items()
            if Path(path).is_file()
        ]
        return (
            detail,
            gr.update(choices=[(f"Variant {i + 1}", i) for i in failed], value=failed),
            gr.update(interactive=job.finished_at is None),
            gr.update(interactive=retryable),
            gr.update(
                interactive=bool(sources)
                and bool(job.finishing_callbacks)
                and job.finished_at is not None
            ),
            gr.update(choices=sources, value=sources[0][1] if sources else None),
            [path for path in job.outputs if Path(path).is_file()],
        )

    selected.change(
        inspect_job,
        inputs=selected,
        outputs=[details, variants, cancel, retry, finishing, finishing_variant, files],
        queue=False,
        api_name=False,
        show_progress="hidden",
    )

    def cancel_job(job_id, request: gr.Request):
        owner = require_owner(request)
        job = JOBS.owned(owner, job_id)
        return JOBS.cancel(owner, job.family, get, post, job_id=job.id)

    cancel.click(
        cancel_job, inputs=selected, outputs=status, queue=False, api_name=False
    )

    def accept_retry(job_id, failed, request: gr.Request):
        try:
            return JOBS.retry(require_owner(request), job_id, variants=failed).id
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc

    def accept_finishing(job_id, index, request: gr.Request):
        owner = require_owner(request)
        source_job = JOBS.owned(owner, job_id)
        if source_job.finished_at is None:
            raise gr.Error("Wait for generation to finish before retrying finishing.")
        callback, path = source_job.retry_finishing(index)
        job = JOBS.accept(
            owner,
            "h3-finishing",
            [path],
            callback=callback,
            media_indices=(0,),
            output_indices=(0,),
            key=uuid.uuid4().hex,
            retry_of=job_id,
        )
        job.variant = index
        if index in source_job.variant_seeds:
            job.variant_seeds[index] = source_job.variant_seeds[index]
        entries = [entry for entry in source_job.ledger if entry["variant"] == index]
        job.replay_ledger = entries[
            source_job.finishing_offsets.get(index, len(entries)) :
        ]
        return job.id

    def execute_retry(job_id, request: gr.Request):
        job = JOBS.owned(require_owner(request), job_id)
        try:
            for _ in execute_accepted(job_id, request):
                yield gr.skip(), f"Job `{job.id[:8]}` · {job.state} · {job.stage}"
        except JobCancelled:
            pass
        yield (
            [path for path in job.outputs if Path(path).is_file()],
            f"Job `{job.id[:8]}` · {job.state}. " + (job.error or ""),
        )

    for accepted in (
        retry.click(
            accept_retry,
            inputs=[selected, variants],
            outputs=ticket,
            queue=False,
            api_name=False,
        ),
        finishing.click(
            accept_finishing,
            inputs=[selected, finishing_variant],
            outputs=ticket,
            queue=False,
            api_name=False,
        ),
        run_project.click(
            accept_project, inputs=project, outputs=ticket, queue=False, api_name=False
        ),
    ):
        dispatched = accepted.success(
            execute_retry,
            inputs=ticket,
            outputs=[files, status],
            api_name=False,
            trigger_mode="multiple",
            **GPU_QUEUE,
        )
        dispatched.failure(
            dispatch_failed,
            inputs=ticket,
            outputs=None,
            queue=False,
            api_name=False,
            show_progress="hidden",
        )
