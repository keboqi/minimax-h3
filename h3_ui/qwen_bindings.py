"""Reusable Gradio event-binding helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gradio as gr
from h3_models import QWEN_IMAGE21_TURBO_MODES
from .job_bindings import bind_gpu_action, bind_prompt_action, owned_generation


from .qwen_view import QwenImage21View

QWEN_IMAGE21_PRESETS = {
    "Fast": ("INT8 ConvRot (lower VRAM)", "Viggle Turbo v0.2.1 (6-step)", 6, "Off"),
    "Normal": ("BF16", "Off", 25, "Spectrum (Quality)"),
    "Quality": ("BF16", "Off", 40, "Spectrum (Quality)"),
}


def qwen_resolution_preset_values(name: str) -> tuple[int, int]:
    """Extract the aligned width and height from a Qwen native-size preset."""

    dimensions = str(name).rsplit("·", 1)[-1].strip()
    width, height = dimensions.split("×", 1)
    return int(width), int(height)


def qwen_preset_values(preset: str):
    """Apply a Qwen preset without locking its individual controls."""
    return QWEN_IMAGE21_PRESETS[preset]


def qwen_turbo_defaults(variant: str, preset: str = "Quality"):
    """Set Turbo defaults, preserving the selected base preset's step count."""
    if variant in QWEN_IMAGE21_TURBO_MODES:
        return QWEN_IMAGE21_TURBO_MODES[variant][1], 1.0, "euler", "Off"
    steps = 25 if preset == "Normal" else 40
    return steps, 1.0, "euler", "Spectrum (Quality)"


def bind_qwen_image21_view(
    view: QwenImage21View,
    *,
    enhance_prompt: Callable[..., Any],
    generate: Callable[..., Any],
) -> Any:
    view.mode.change(
        lambda mode: tuple(gr.update(visible=mode == "Image edit") for _ in range(3))
        + tuple(gr.update(interactive=mode == "Image edit") for _ in range(2)),
        inputs=view.mode,
        outputs=[
            view.batch_edit_inputs,
            view.edit_size,
            view.reference_resolution,
            view.cache_device,
            view.cache_dtype,
        ],
        queue=False,
        show_progress="hidden",
        api_name=False,
    )
    view.cfg.change(
        lambda cfg: gr.update(visible=cfg > 1),
        inputs=view.cfg,
        outputs=view.negative_prompt,
        queue=False,
        show_progress="hidden",
        api_name=False,
    )
    view.preset.change(
        qwen_preset_values,
        inputs=view.preset,
        outputs=[view.model, view.turbo_variant, view.steps, view.accelerator],
        queue=False,
        show_progress="hidden",
        api_name=False,
    )
    view.turbo_variant.change(
        qwen_turbo_defaults,
        inputs=[view.turbo_variant, view.preset],
        outputs=[view.steps, view.cfg, view.sampler, view.accelerator],
        queue=False,
        show_progress="hidden",
        api_name=False,
    )
    view.output_resolution_1k.input(
        lambda name: (
            (*qwen_resolution_preset_values(name), None)
            if name
            else (gr.update(), gr.update(), gr.update())
        ),
        inputs=view.output_resolution_1k,
        outputs=[view.width, view.height, view.output_resolution_2k],
        queue=False,
        show_progress="hidden",
        api_name=False,
    )
    view.output_resolution_2k.input(
        lambda name: (
            (*qwen_resolution_preset_values(name), None)
            if name
            else (gr.update(), gr.update(), gr.update())
        ),
        inputs=view.output_resolution_2k,
        outputs=[view.width, view.height, view.output_resolution_1k],
        queue=False,
        show_progress="hidden",
        api_name=False,
    )
    for dimension in (view.width, view.height):
        dimension.input(
            lambda: (None, None),
            outputs=[view.output_resolution_1k, view.output_resolution_2k],
            queue=False,
            show_progress="hidden",
            api_name=False,
        )
    bind_prompt_action(
        view.enhance.click,
        enhance_prompt,
        inputs=[
            view.prompt,
            view.prompt_model,
            view.api_key,
            view.mode,
            view.reference_images,
            view.width,
            view.height,
            view.prompt_backend,
            view.lightning_api_key,
            view.edit_size,
        ],
        outputs=[view.prompt, view.enhance_status],
        show_progress="minimal",
        api_name="enhance_qwen_image21_prompt",
    )
    return bind_gpu_action(
        view.run.click,
        owned_generation(generate, "qwen_image21"),
        inputs=[
            view.mode,
            view.model,
            view.text_encoder,
            view.prompt,
            view.negative_prompt,
            view.reference_images,
            view.width,
            view.height,
            view.reference_resolution,
            view.edit_size,
            view.seed,
            view.steps,
            view.cfg,
            view.sampler,
            view.scheduler,
            view.cache_device,
            view.cache_dtype,
            view.attention_backend,
            view.accelerator,
            view.turbo_variant,
            view.output_resolution_1k,
            view.output_resolution_2k,
            view.batch_count,
            view.batch_edit_inputs,
        ],
        outputs=[view.output, view.status],
        show_progress="minimal",
        api_name="generate_qwen_image21",
    )
