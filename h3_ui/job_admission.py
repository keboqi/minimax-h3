"""Accept immutable requests before Gradio's existing GPU queue runs them."""

from collections.abc import Iterator
import inspect
from pathlib import Path
import uuid
import time

import gradio as gr

from h3_app.jobs import CURRENT_JOB, JOBS, JobCancelled
from h3_app.reference_bindings import ReferenceMap
from h3_app.errors import H3Error


def require_owner(request):
    if request is None or not request.session_hash:
        raise gr.Error("A live session is required for this job action.")
    return request.session_hash


def dispatch_failed(job_id, request: gr.Request):
    job = JOBS.owned(require_owner(request), job_id)
    with JOBS.lock:
        if not job.claimed and job.finished_at is None:
            job.fail(
                RuntimeError(
                    "The execution queue rejected this request. Submit again when capacity is available."
                )
            )
            job.finished_at = time.time()
            JOBS.active.pop(job.id, None)


def _output_paths(value):
    if isinstance(value, dict):
        if "value" in value:
            yield from _output_paths(value["value"])
        elif "path" in value:
            yield from _output_paths(value["path"])
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _output_paths(item)
    elif isinstance(value, str) and Path(value).is_file():
        yield value


def execute_accepted(job_id: str, request: gr.Request):
    owner = require_owner(request)
    job = JOBS.owned(owner, job_id)
    iterator = None
    try:
        with JOBS.run(owner, job.family, accepted=job):
            values = job.values()
            while True:
                token = CURRENT_JOB.set(job)
                try:
                    job.check()
                    if iterator is None:
                        result = job.callback(*values, request)
                        iterator = (
                            result if isinstance(result, Iterator) else iter((result,))
                        )
                    update = next(iterator)
                    for index in job.output_indices:
                        if index < len(update):
                            for path in _output_paths(update[index]):
                                if path not in job.outputs:
                                    job.outputs.append(path)
                except StopIteration:
                    return
                finally:
                    CURRENT_JOB.reset(token)
                yield update
    finally:
        if iterator is not None:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()


def bind_accepted_action(
    trigger, callback, options, gpu_queue, *, reference_map=None, resolver=None
):
    inputs = options.get("inputs", [])
    inputs = list(inputs) if isinstance(inputs, (tuple, list)) else [inputs]
    outputs = options.get("outputs", [])
    outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
    # Existing named APIs retain their original parameter/order/stream contract.
    api_name = options.get("api_name", False)
    if api_name:
        gr.Button(visible=False).click(callback, **options, **gpu_queue)
    names = tuple(callback.job_input_names)
    media_types = (gr.Image, gr.Video, gr.Audio, gr.File)
    media_indices = tuple(
        i for i, component in enumerate(inputs) if isinstance(component, media_types)
    )
    output_indices = tuple(
        i
        for i, component in enumerate(outputs)
        if isinstance(component, (*media_types, gr.Gallery))
    )
    # A regular string component is captured in each queued event payload.
    # gr.State would be read at execution and could hold a later ticket.
    ticket = gr.Textbox(visible=False)
    request_key = gr.Textbox(visible=False)

    def capture(*args):
        request = args[-1]
        values = list(args[:-1])
        client_key = uuid.UUID(values.pop()).hex
        mapping = values.pop() if reference_map is not None else None
        captured = dict(zip(names, values, strict=True))
        active_media = media_indices
        if callback.job_family == "h3":
            mode = captured["mode"]
            prefixes = (
                ("ref_image_", "ref_video_", "ref_audio_")
                if mode == "Reference media"
                else ()
            )
            active_media = tuple(
                i
                for i in media_indices
                if names[i].startswith(prefixes)
                or mode == "First / last frame"
                and names[i]
                in (
                    "first_image",
                    "last_image",
                    "fl2va_audio_1",
                    "fl2va_audio_2",
                    "fl2va_audio_3",
                )
            )
            if mode == "Reference media":
                mapping = (mapping or ReferenceMap()).update(captured)
                captured = mapping.dense_values(captured)
                values = [captured[name] for name in names]
            for i in media_indices:
                if i not in active_media:
                    values[i] = None
                    captured[names[i]] = None
            if resolver is not None:
                plan = resolver(captured)
                if plan.issues:
                    raise gr.Error(" ".join(plan.issues))
        job = JOBS.accept(
            require_owner(request),
            callback.job_family,
            values,
            callback=callback,
            media_indices=active_media,
            output_indices=output_indices,
            key=client_key,
        )
        return job.id

    def accept(*args):
        try:
            return capture(*args)
        except (H3Error, ValueError) as exc:
            raise gr.Error(str(exc)) from exc

    # Preserve request injection even though acceptance has a variadic wrapper.
    parameters = [
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in (
            *names,
            *(("reference_map",) if reference_map is not None else ()),
            "client_request_id",
        )
    ]
    parameters.append(
        inspect.Parameter(
            "request",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=None,
            annotation=gr.Request,
        )
    )
    accept.__signature__ = inspect.Signature(parameters)
    accept.__annotations__ = {"request": gr.Request}
    accepted = trigger(
        accept,
        inputs=[
            *inputs,
            *((reference_map,) if reference_map is not None else ()),
            request_key,
        ],
        outputs=ticket,
        queue=False,
        api_name=False,
        trigger_mode="multiple",
        show_progress="hidden",
        js="(...args) => { const bytes = new Uint8Array(16); crypto.getRandomValues(bytes); args[args.length - 1] = Array.from(bytes, b => b.toString(16).padStart(2, '0')).join(''); return args; }",
    )

    def run_ticket(job_id, request: gr.Request):
        try:
            yield from execute_accepted(job_id, request)
        except JobCancelled:
            return

    event = accepted.success(
        run_ticket,
        inputs=ticket,
        outputs=outputs,
        api_name=False,
        show_progress=options.get("show_progress", "minimal"),
        trigger_mode="multiple",
        **gpu_queue
    )

    event.failure(
        dispatch_failed,
        inputs=ticket,
        outputs=None,
        queue=False,
        api_name=False,
        show_progress="hidden",
    )
    event.acceptance = accepted
    event.ticket = ticket
    return event
