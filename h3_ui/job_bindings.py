"""Gradio request injection and job lifetime, isolated from generation code."""

from __future__ import annotations
from collections.abc import Iterator
from contextlib import nullcontext
import inspect
import gradio as gr
from h3_app.jobs import JOBS, CURRENT_JOB
from h3_app.provenance import RUN_CONTEXT, render_snapshot

GPU_QUEUE = {"concurrency_id": "h3-gpu", "concurrency_limit": 1}
PROMPT_QUEUE = {"concurrency_id": "h3-prompt", "concurrency_limit": 4}


def bind_gpu_action(trigger, callback=None, **options):
    """Register generation or maintenance with the same GPU queue policy."""
    reference_map = options.pop("reference_map", None)
    resolver = options.pop("resolver", None)
    component = getattr(trigger, "__self__", None)
    if getattr(callback, "job_family", None) and (
        getattr(component, "visible", True) is not False
    ):
        from .job_admission import bind_accepted_action

        return bind_accepted_action(
            trigger,
            callback,
            options,
            GPU_QUEUE,
            reference_map=reference_map,
            resolver=resolver,
        )
    return trigger(callback, **options, **GPU_QUEUE)


def bind_prompt_action(trigger, callback=None, **options):
    """Allow remote prompt requests while a GPU job prepares or downloads models."""
    review = getattr(getattr(trigger, "__self__", None), "h3_review", None)
    if review is not None:
        from .prompt_review import bind_review_action

        return bind_review_action(trigger, callback, options, review, PROMPT_QUEUE)
    return trigger(callback, **options, **PROMPT_QUEUE)


def owned_generation(callback, family: str, input_names=None, *, metadata_output=False):
    """Advance under the job context even when Gradio switches worker threads."""
    signature = inspect.signature(callback)
    names = input_names or tuple(
        (
            name
            for name, param in signature.parameters.items()
            if name not in {"request", "progress"}
            and param.kind
            not in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD)
        )
    )

    def run(*args):
        values, request = (args[:-1], args[-1])
        from .job_admission import require_owner

        owner = require_owner(request)
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
        current = CURRENT_JOB.get()
        scope = (
            nullcontext(current)
            if current is not None
            and current.owner == owner
            and (current.family == family)
            and current.has_gpu
            else JOBS.run(owner, family)
        )
        with scope as job:
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
                            result = callback(*values, **kwargs)
                            iterator = (
                                result
                                if isinstance(result, Iterator)
                                else iter((result,))
                            )
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
                                else [value] if value else []
                            )
                        paths = [
                            path for values in result_paths.values() for path in values
                        ]
                        update = (*update, render_snapshot(paths))
                    yield update
            finally:
                if iterator is not None:
                    close = getattr(iterator, "close", None)
                    if close is not None:
                        close()

    optional_defaults = (
        {
            "fl2va_audio_1": None,
            "fl2va_audio_2": None,
            "fl2va_audio_3": None,
            "encoder_small_input": False,
            "use_lynnreal_vae": False,
            "latent_upscale_refine_lora": "Same as generation",
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
    run.job_family = family
    run.job_input_names = names
    return run


def owned_interrupt(callback, family):

    def stop(request: gr.Request):
        return callback(request, family=family)

    return stop
