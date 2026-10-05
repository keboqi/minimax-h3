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

JOB_SELECTION_JS = """
const select = (row) => {
  if (!row) return;
  element.querySelectorAll('[data-job-id]').forEach(item => item.setAttribute('aria-selected', String(item === row)));
  trigger('click', {job_id: row.dataset.jobId});
};
element.addEventListener('click', event => select(event.target.closest('[data-job-id]')));
element.addEventListener('keydown', event => {
  if (event.key === 'Enter' || event.key === ' ') {
    const row = event.target.closest('[data-job-id]');
    if (row) { event.preventDefault(); select(row); }
  }
});
"""


def state_badge(state):
    tone = {
        "completed": "success",
        "failed": "danger",
        "cancelled": "muted",
        "queued": "waiting",
        "submission_unknown": "danger",
    }.get(state, "active")
    return f'<span class="h3-job-badge h3-job-{tone}">{escape(state.replace("_", " "))}</span>'


def empty_details():
    return (
        '<div class="h3-empty-state"><strong>Select a job</strong>'
        "<p>Click a request in the history to review its progress, outputs and retry options.</p></div>"
    )


def table(jobs, selected=None):
    if not jobs:
        return (
            '<div class="h3-empty-state" role="status"><strong>No jobs yet</strong>'
            "<p>Start a request in Create. Its progress and outputs will appear here.</p></div>"
        )
    counts = (
        ("Queued", sum(job.state == "queued" for job in jobs)),
        (
            "Active",
            sum(job.finished_at is None and job.state != "queued" for job in jobs),
        ),
        ("Completed", sum(job.state == "completed" for job in jobs)),
        (
            "Needs attention",
            sum(
                job.state in {"failed", "submission_unknown", "recovering"}
                for job in jobs
            ),
        ),
    )
    metrics = (
        '<dl class="h3-job-metrics">'
        + "".join(
            f"<div><dt>{label}</dt><dd>{count}</dd></div>" for label, count in counts
        )
        + "</dl>"
    )
    rows = []
    for job in jobs:
        cells = (
            job.id[:8],
            job.family,
            job.stage,
            f"{max(0, int((job.finished_at or time.time()) - job.created_at))}s",
            len(job.outputs),
        )
        rows.append(
            f'<tr data-job-id="{escape(job.id, quote=True)}" tabindex="0" aria-selected="{str(job.id == selected).lower()}" aria-label="Select job {escape(job.id[:8], quote=True)}">'
            + f"<td><code>{escape(str(cells[0]))}</code><small>{escape(str(cells[1]))}</small></td>"
            + f"<td>{state_badge(job.state)}</td>"
            + "".join(f"<td>{escape(str(cell))}</td>" for cell in cells[2:])
            + "</tr>"
        )
    return (
        metrics
        + '<div class="h3-job-table"><table><caption>Jobs for this browser owner</caption>'
        '<thead><tr><th scope="col">Request / engine</th><th scope="col">Status</th><th scope="col">Current stage</th><th scope="col">Elapsed</th><th scope="col">Outputs</th></tr></thead>'
        "<tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def build_jobs_view(root, *, get, post, summary):
    with root:
        with gr.Row(equal_height=False, elem_classes=["h3-view-heading"]):
            gr.Markdown(
                "## Jobs\nFollow your requests from queue to output. Review a job to download results or recover unfinished work.",
                scale=4,
            )
            refresh = gr.Button("Refresh jobs", scale=0, min_width=140)
        with gr.Row(equal_height=False, elem_classes=["h3-jobs-workspace"]):
            with gr.Column(scale=5, min_width=340, elem_classes=["h3-job-history"]):
                gr.Markdown(
                    "### Request history\nClick a row to inspect it. Updates automatically every 2 seconds.",
                    elem_classes=["h3-gallery-section-title"],
                )
                listing = gr.HTML(table(()), js_on_load=JOB_SELECTION_JS)
            with gr.Column(
                scale=3,
                min_width=320,
                elem_classes=["h3-preview-panel", "h3-job-inspector"],
            ):
                gr.Markdown(
                    "### Job details", elem_classes=["h3-gallery-section-title"]
                )
                selected = gr.Dropdown(
                    [],
                    label="Job",
                    value=None,
                    info="Choose a request to inspect or manage.",
                )
                details = gr.HTML(empty_details(), elem_classes=["h3-job-details"])
                files = gr.File(
                    label="Retained outputs",
                    file_count="multiple",
                    interactive=False,
                    visible=False,
                )
                cancel = gr.Button(
                    "Cancel selected job", variant="stop", interactive=False
                )
                with gr.Accordion(
                    "Retry & recovery", open=False, elem_classes=["h3-gallery-card"]
                ):
                    gr.Markdown(
                        "Retry failed variants with their recorded seeds, or finish a retained source without repeating generation."
                    )
                    variants = gr.CheckboxGroup(
                        [], label="Failed variants to retry", value=[]
                    )
                    retry = gr.Button(
                        "Retry failed variants", variant="primary", interactive=False
                    )
                    finishing_variant = gr.Dropdown(
                        [], label="Retained H3 source variant", value=None
                    )
                    finishing = gr.Button("Retry finishing only", interactive=False)
                status = gr.Markdown(elem_classes=["h3-job-action-status"])
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
            reconcile = gr.Button("Reconnect to backend history")
            recovery_key = gr.Textbox(
                label="Browser recovery key", type="password", interactive=False
            )
            export_owner = gr.Button("Show my recovery key")
            recovery_input = gr.Textbox(
                label="Restore browser recovery key", type="password"
            )
            recover_owner = gr.Button("Restore owner and reload")
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
            "table": table(jobs, value),
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

    def select_row(evt: gr.EventData, request: gr.Request):
        job = JOBS.owned(require_owner(request), evt.job_id)
        return job.id

    listing.click(
        select_row,
        outputs=selected,
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
                empty_details(),
                gr.update(choices=[], value=[]),
                *(gr.update(interactive=False) for _ in range(3)),
                gr.update(choices=[], value=None),
                gr.update(value=[], visible=False),
            )
        job = JOBS.owned(require_owner(request), job_id)
        safe_ledger = [
            {key: value for key, value in entry.items() if key != "graph_json"}
            for entry in job.ledger
        ]
        detail = (
            '<div class="h3-job-detail-heading">'
            + state_badge(job.state)
            + f"<code>{escape(job.id[:8])}</code></div>"
            + '<p role="status">'
            + escape(job.error or job.stage)
            + "</p>"
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
            gr.update(
                value=[path for path in job.outputs if Path(path).is_file()],
                visible=any(Path(path).is_file() for path in job.outputs),
            ),
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
            gr.update(
                value=[path for path in job.outputs if Path(path).is_file()],
                visible=any(Path(path).is_file() for path in job.outputs),
            ),
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
