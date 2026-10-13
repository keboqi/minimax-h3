"""NVIDIA gallery request capture and inexpensive UI availability checks."""

from __future__ import annotations

import gradio as gr

from h3_app.catalog import NVIDIA_POSTPROCESS_OPTIONS
from h3_app.jobs import record_failure
from h3_app.nvidia_postprocess import availability, probe_runtime
from .presentation import error_markdown


def can_enhance(mode, selected, method):
    if not selected or mode == "Audio":
        return False
    if method in NVIDIA_POSTPROCESS_OPTIONS:
        return mode in {"Video", "Image"} and availability(method)[0]
    return True


def with_nvidia_settings(callback):
    """Append optional values so retained pre-NVIDIA gallery jobs still replay."""
    def enhance(
        mode, selected_media, option, seed, seedvr2_model, ltx25_model,
        ltx25_prompt, force_offload, split_upscale, split_seconds, upscale_resolution,
        nvidia_original=False,
        dlss_style="Natural", dlss_intensity=1.0, dlss_upscaler="resize", request: gr.Request = None,
    ):
        kwargs = {}
        if option in NVIDIA_POSTPROCESS_OPTIONS:
            kwargs["nvidia_settings"] = {
                "original_size": nvidia_original,
                "style": dlss_style, "intensity": dlss_intensity,
                "upscaler": dlss_upscaler or "resize",
            }
        return callback(
            mode, selected_media, option, seed, seedvr2_model, ltx25_model,
            ltx25_prompt, force_offload, split_upscale, split_seconds,
            upscale_resolution, request=request, **kwargs,
        )
    enhance.__name__ = callback.__name__
    return enhance


def check_runtime(method: str, request: gr.Request):
    from .application import _runtime_config
    yield "Checking NVIDIA runtime on the processing host…"
    try:
        yield probe_runtime(method, _runtime_config())
    except Exception as exc:
        record_failure(exc)
        yield error_markdown("Runtime check failed", exc)
