"""An owner-scoped monitor; Gradio remains the only dispatcher."""

from html import escape
import json
from pathlib import Path
import time
import uuid

import gradio as gr

from h3_app.jobs import JOBS, JobCancelled
from .job_admission import (
    START_POLLING_JS,
    dispatch_failed,
    execute_accepted,
    poll_after_acceptance,
    require_owner,
    watch_acceptance,
)
from .job_bindings import GPU_QUEUE
from .operations_view import page_heading

JOB_SELECTION_JS = """
element.addEventListener('click', event => {
  const row = event.target.closest('[data-job-id]');
  if (!row) return;
  element.querySelectorAll('[data-job-id]').forEach(item => {
    item.querySelector('button').setAttribute('aria-current', String(item === row));
  });
  trigger('click', {job_id: row.dataset.jobId});
});
"""

STATUS_FILTERS = {
    "Queued": lambda job: job.state == "queued",
    "Active": lambda job: job.finished_at is None and job.state != "queued",
    "Completed": lambda job: job.state == "completed",
    "Needs attention": lambda job: job.state in {"failed", "submission_unknown", "recovering"},
    "Cancelled": lambda job: job.state == "cancelled",
}
METRIC_LABELS = ("Queued", "Active", "Completed", "Needs attention")


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
        "<p>Review progress, outputs, and retry options.</p></div>"
    )


def job_metrics(counts):
    return (
        '<dl class="h3-job-metrics" aria-label="Job counts">'
        + "".join(
            f"<div><dt>{label}</dt><dd>{count}</dd></div>"
            for label, count in zip(METRIC_LABELS, counts, strict=True)
        )
        + "</dl>"
    )


def table(jobs, selected=None, status_filter="All jobs"):
    if not jobs:
        return (
            '<div class="h3-empty-state" role="status"><strong>No jobs yet</strong>'
            "<p>Start a request in Create. Its progress and outputs will appear here.</p></div>"
        )
    matches = STATUS_FILTERS.get(status_filter)
    rows = []
    for job in jobs:
        if matches and not matches(job):
            continue
        job_label = escape(job.id[:8])
        current = str(job.id == selected).lower()
        elapsed = max(0, int((job.finished_at or time.time()) - job.created_at))
        rows.append(
            f'<tr data-job-id="{escape(job.id, quote=True)}">'
            f'<td><button type="button" aria-current="{current}" aria-label="Inspect job {job_label}">'
            f"<code>{job_label}</code><small>{escape(job.family)}</small></button></td>"
            f"<td>{state_badge(job.state)}</td><td>{escape(str(job.stage))}</td>"
            f"<td>{elapsed}s</td><td>{len(job.outputs)}</td></tr>"
        )
    if not rows:
        return (
            '<div class="h3-empty-state" role="status"><strong>No matching jobs</strong>'
            "<p>Choose another status to see more requests.</p></div>"
        )
    return (
        '<div class="h3-ops-table h3-job-table" tabindex="0" role="region" aria-label="Request history">'
        '<table aria-label="Jobs for this browser owner"><thead><tr>'
        '<th scope="col">Request / engine</th><th scope="col">Status</th>'
        '<th scope="col">Current stage</th><th scope="col">Elapsed</th><th scope="col">Outputs</th></tr></thead>'
        "<tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def build_jobs_view(root, *, get, post, summary, jobs_tab=None, other_tabs=()):
    """Jobs in ``root``.

    Jobs polls every 2 seconds only while this session has unfinished jobs or
    ``jobs_tab`` is open, so an idle page sends nothing and the server can scale down.
    """
    with root:
        with gr.Row(elem_classes=["h3-ops-hero"]):
            page_heading(
                "Jobs",
                "Track requests, download outputs, and recover unfinished work.",
                "M7 4h10a3 3 0 0 1 3 3v10a3 3 0 0 1-3 3H7a3 3 0 0 1-3-3V7a3 3 0 0 1 3-3Z "
                "M8 8h8m-8 4h8m-8 4h5",
            )
            refresh = gr.Button("Refresh jobs", variant="primary", scale=0)
        metrics = gr.HTML(job_metrics((0,) * len(METRIC_LABELS)))
        with gr.Row(equal_height=False, elem_classes=["h3-ops-split"]):
            with gr.Column(min_width=0, elem_classes=["h3-ops-panel", "h3-ops-main"]):
                with gr.Row(elem_classes=["h3-ops-panel-header"]):
                    gr.Markdown(
                        "### Request history\nUpdates every 2 seconds while this tab is open.",
                        elem_classes=["h3-ops-section-title"],
                    )
                    status_filter = gr.Dropdown(
                        ["All jobs", *STATUS_FILTERS],
                        value="All jobs",
                        label="Status",
                        container=False,
                        scale=0,
                        min_width=160,
                        elem_classes=["h3-job-filter"],
                    )
                listing = gr.HTML(table(()), js_on_load=JOB_SELECTION_JS)
            with gr.Column(
                min_width=0,
                elem_classes=["h3-ops-panel", "h3-ops-aside", "h3-job-inspector"],
            ):
                gr.Markdown("### Job details", elem_classes=["h3-ops-section-title"])
                selected = gr.Dropdown(
                    [],
                    label="Job",
                    value=None,
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
                    "Retry & recovery", open=False,
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
        # Active at load so the first tick finds unfinished jobs; it then stops when idle.
        timer = gr.Timer(2)
        watching = gr.Checkbox(False, visible=False)
        presentation = gr.State({})
        with gr.Accordion("Browser ownership and recovery", open=False):
            gr.Markdown(
                "Technical job history survives server restarts. Unsaved prompts and uploaded "
                "input copies expire after 30 minutes or a restart. "
                "This browser's signed ownership cookie expires after 30 days. "
                "Recovery never automatically repeats inference. "
                "Keep a recovery key to reopen jobs after losing browser cookies. "
                "The key grants access to your job history; keep it private.",
                elem_classes=["h3-ops-note"],
            )
            with gr.Row(equal_height=False, elem_classes=["h3-ops-split"]):
                with gr.Column(min_width=0, elem_classes=["h3-ops-recovery-field"]):
                    recovery_key = gr.Textbox(
                        label="Browser recovery key", type="password", interactive=False
                    )
                    export_owner = gr.Button("Show my recovery key")
                with gr.Column(min_width=0, elem_classes=["h3-ops-recovery-field"]):
                    recovery_input = gr.Textbox(
                        label="Restore browser recovery key", type="password"
                    )
                    recover_owner = gr.Button("Restore owner and reload")
            with gr.Row(elem_classes=["h3-ops-actions"]):
                reconcile = gr.Button("Reconnect to backend history", scale=0)
            recovery_status = gr.Markdown()

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
        selected_id, failed_selection, source_selection, previous, watching, filter_value,
        request: gr.Request,
    ):
        jobs = JOBS.list_owned(require_owner(request))
        choices = [
            (f"{job.id[:8]} · {job.family} · {job.state}", job.id) for job in jobs
        ]
        value = selected_id if any(job.id == selected_id for job in jobs) else None
        counts = tuple(sum(STATUS_FILTERS[label](job) for job in jobs) for label in METRIC_LABELS)
        queued, active = counts[:2]
        inspected = list(inspect_job(value, request))
        signature = json.dumps(inspected, sort_keys=True)
        failed_choices = inspected[1].get("choices")
        previous = previous or {}
        if previous.get("selected") == value:
            if previous.get("failed") == failed_choices:
                inspected[1]["value"] = failed_selection
            sources = [item[1] for item in inspected[5].get("choices", [])]
            if source_selection in sources:
                inspected[5]["value"] = source_selection
        if previous.get("detail") == signature and previous.get("selected") == value:
            inspected = [gr.skip() for _ in inspected]
        history = table(jobs, value, filter_value)
        next_presentation = {
            "selected": value,
            "detail": signature,
            "choices": choices,
            "failed": failed_choices,
            "table": history,
            "counts": counts,
        }
        return (
            gr.skip() if previous.get("table") == history else history,
            (
                gr.skip()
                if previous.get("choices") == choices and value == selected_id
                else gr.update(choices=choices, value=value)
            ),
            (
                gr.skip()
                if previous.get("counts", ())[:2] == (queued, active)
                else f'<p class="h3-system-status" role="status">{queued} queued · {active} active in this session</p>'
            ),
            gr.skip() if previous.get("counts") == counts else job_metrics(counts),
            *inspected,
            next_presentation,
            gr.Timer(active=bool(watching or queued or active)),
        )

    # Share a single request slot so a poll and a filter change cannot finish out of order.
    gr.on(
        triggers=[timer.tick, refresh.click, status_filter.input],
        fn=refresh_jobs,
        inputs=[selected, variants, finishing_variant, presentation, watching, status_filter],
        outputs=[
            listing,
            selected,
            summary,
            metrics,
            details,
            variants,
            cancel,
            retry,
            finishing,
            finishing_variant,
            files,
            presentation,
            timer,
        ],
        queue=False,
        api_name=False,
        show_progress="hidden",
        trigger_mode="always_last",
    )

    poll_after_acceptance(timer)
    if jobs_tab is not None:
        # Browser-side, so switching tabs costs no request.
        jobs_tab.select(None, js=f"() => [true, ({START_POLLING_JS})()]", outputs=[watching, timer],
                        queue=False, api_name=False, show_progress="hidden")
    for tab in other_tabs:
        tab.select(None, js="() => false", outputs=watching, queue=False, api_name=False,
                   show_progress="hidden")

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

    reconcile.click(reconnect, outputs=recovery_status, queue=False, api_name=False)

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

    for accepted, outputs in (
        (retry.click(
            accept_retry,
            inputs=[selected, variants],
            outputs=ticket,
            queue=False,
            api_name=False,
        ), [files, status]),
        (finishing.click(
            accept_finishing,
            inputs=[selected, finishing_variant],
            outputs=ticket,
            queue=False,
            api_name=False,
        ), [files, status]),
    ):
        watch_acceptance(accepted)
        dispatched = accepted.success(
            execute_retry,
            inputs=ticket,
            outputs=outputs,
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
