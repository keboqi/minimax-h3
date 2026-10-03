"""A bounded session-owned monitor; Gradio remains the only dispatcher."""

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
        return '<p role="status">No jobs in this session yet.</p>'
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
        '<div class="h3-job-table"><table><caption>Jobs in this browser session</caption>'
        "<thead><tr><th>Job</th><th>Engine</th><th>State</th><th>Stage</th><th>Elapsed</th><th>Outputs</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def build_jobs_view(root, *, get, post, summary):
    with root:
        gr.Markdown(
            "## Jobs\nAccepted requests keep their own settings and input files. "
            "Jobs and retries belong to this browser session and expire 30 minutes after finishing "
            "or when the server restarts; capacity limits can evict finished jobs earlier. "
            "Prompts and input copies are held locally for replay."
        )
        listing = gr.HTML(table(()))
        refresh = gr.Button("Refresh jobs")
        selected = gr.Dropdown([], label="Job", value=None)
        details = gr.HTML()
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

    def refresh_jobs(selected_id, request: gr.Request):
        jobs = JOBS.list_owned(require_owner(request))
        choices = [
            (f"{job.id[:8]} · {job.family} · {job.state}", job.id) for job in jobs
        ]
        value = selected_id if any(job.id == selected_id for job in jobs) else None
        queued = sum(job.state == "queued" for job in jobs)
        active = sum(job.finished_at is None and job.state != "queued" for job in jobs)
        return (
            table(jobs),
            gr.update(choices=choices, value=value),
            (
                f'<p class="h3-system-status" role="status">{queued} queued · {active} active in this session</p>'
            ),
        )

    for event in (timer.tick, refresh.click):
        event(
            refresh_jobs,
            inputs=selected,
            outputs=[listing, selected, summary],
            queue=False,
            api_name=False,
            show_progress="hidden",
        )

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
            gr.update(interactive=bool(sources) and job.finished_at is not None),
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
        job.replay_ledger = entries[source_job.finishing_offsets.get(index, len(entries)):]
        return job.id

    def execute_retry(job_id, request: gr.Request):
        job = JOBS.owned(require_owner(request), job_id)
        try:
            for _ in execute_accepted(job_id, request):
                yield gr.skip(), f"Job `{job.id[:8]}` · {job.state} · {job.stage}"
        except JobCancelled:
            pass
        yield [
            path for path in job.outputs if Path(path).is_file()
        ], f"Job `{job.id[:8]}` · {job.state}. " + (job.error or "")

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
