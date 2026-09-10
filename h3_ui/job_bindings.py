"""Gradio request injection and job lifetime, isolated from generation code."""

from __future__ import annotations
import inspect
import uuid
import gradio as gr
from h3_app.jobs import JOBS, CURRENT_JOB
from h3_app.provenance import RUN_CONTEXT, render_snapshot

GPU_QUEUE = {"concurrency_id": "h3-gpu", "concurrency_limit": 1}


def bind_gpu_action(trigger, callback=None, **options):
    """Register generation or maintenance with the same GPU queue policy."""
    return trigger(callback, **options, **GPU_QUEUE)


def owned_generation(callback, family: str, input_names=None, *, metadata_output=False):
    """Advance under the job context even when Gradio switches worker threads."""
    signature = inspect.signature(callback)
    names = input_names or tuple(
        name for name in signature.parameters if name not in {"request", "progress"}
    )

    def run(*args):
        values, request = args[:-1], args[-1]
        owner = request.session_hash or uuid.uuid4().hex
        context = {}
        if family == "h3":
            has_preset = names[-1] == "preset"
            context = {
                "preset": values[-1] if has_preset else None,
                "batch_count": values[0],
            }
            if has_preset:
                values = values[:-1]
        iterator = None
        result_paths = {}
        with JOBS.run(owner, family) as job:
            try:
                while True:
                    job_token = CURRENT_JOB.set(job)
                    run_token = RUN_CONTEXT.set(context)
                    try:
                        job.check()
                        if iterator is None:
                            kwargs = (
                                {"request": request}
                                if "request" in signature.parameters
                                else {}
                            )
                            iterator = iter(callback(*values, **kwargs))
                        update = next(iterator)
                    except StopIteration:
                        return
                    finally:
                        RUN_CONTEXT.reset(run_token)
                        CURRENT_JOB.reset(job_token)
                    if metadata_output:
                        for index in (0, 1, 2, 3, 7, 9):
                            value = update[index]
                            if isinstance(value, dict):
                                if "value" not in value:
                                    continue
                                value = value["value"]
                            result_paths[index] = (
                                value
                                if isinstance(value, list)
                                else [value]
                                if value
                                else []
                            )
                        paths = [
                            path for values in result_paths.values() for path in values
                        ]
                        update = (*update, render_snapshot(paths))
                    yield update
            finally:
                if iterator is not None:
                    iterator.close()

    # Gradio treats an empty upload as required unless the callback signature
    # also supplies a default. Keep appended voice inputs optional for old clients.
    optional_defaults = (
        {
            "fl2va_audio_1": None,
            "fl2va_audio_2": None,
            "fl2va_audio_3": None,
            "preset": None,
        }
        if family == "h3"
        else {}
    )
    parameters = [
        inspect.Parameter(
            name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=optional_defaults.get(name, inspect.Parameter.empty),
        )
        for name in names
    ]
    parameters.append(
        inspect.Parameter(
            "request",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=gr.Request,
            default=None,
        )
    )
    run.__signature__ = inspect.Signature(parameters)
    run.__annotations__ = {"request": gr.Request}
    run.__name__ = callback.__name__
    return run


def owned_interrupt(callback, family):
    def stop(request: gr.Request):
        return callback(request, family=family)

    return stop
