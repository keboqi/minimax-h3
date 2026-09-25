"""Reusable Gradio event-binding helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import gradio as gr
from h3_app.catalog import (
    LTX25_CQ_ENHANCER,
    LTX25_DEBLUR,
    LTX25_RESTORATION_OPTIONS,
)
from .job_bindings import bind_gpu_action, bind_prompt_action, owned_generation, owned_interrupt


from .ltx_view import LtxView
from .views import ApiView, GalleryView, MusicView, QwenImage21View, YuE2View


def qwen_resolution_preset_values(name: str) -> tuple[int, int]:
    """Extract the aligned width and height from a Qwen native-size preset."""

    dimensions = str(name).rsplit("·", 1)[-1].strip()
    width, height = dimensions.split("×", 1)
    return int(width), int(height)


def bind_preflight(
    controls: Sequence[gr.components.Component],
    *,
    prompt: gr.Textbox,
    callback: Callable[..., Any],
    readiness: gr.HTML,
    primary_action: gr.Button,
) -> None:
    """Keep readiness synchronized for both committed and live prompt input."""

    outputs = [readiness, primary_action]
    for control in controls:
        control.change(
            callback,
            inputs=list(controls),
            outputs=outputs,
            queue=False,
            show_progress="hidden",
            api_name=False,
        )
    prompt.input(
        callback,
        inputs=list(controls),
        outputs=outputs,
        queue=False,
        show_progress="hidden",
        api_name=False,
    )


def bind_summary(
    controls: Iterable[gr.components.Component],
    *,
    callback: Callable[..., Any],
    output: gr.Markdown,
    skip: Iterable[gr.components.Component] = (),
) -> None:
    """Refresh a derived summary when any non-specialized control changes."""

    inputs = list(controls)
    skipped = set(skip)
    for control in inputs:
        if control in skipped:
            continue
        control.change(
            callback,
            inputs=inputs,
            outputs=output,
            queue=False,
            show_progress="hidden",
            api_name=False,
        )


def bind_interrupts(
    callback: Callable[..., Any],
    bindings: Iterable[tuple[gr.Button, gr.components.Component, Sequence[Any]]],
) -> None:
    """Apply identical cancellation semantics across generation views."""

    for button, output, events in bindings:
        button.click(callback, outputs=output, cancels=list(events))


def bind_ltx_view(
    view: LtxView,
    *,
    render_workflow: Callable[..., Any],
    prepare_workflow: Callable[..., Any],
    prepare_all_models: Callable[..., Any],
    render_inventory: Callable[..., Any],
    enhance_prompt: Callable[..., Any],
    generate: Callable[..., Any],
) -> Any:
    view.mode.change(
        lambda value: gr.update(visible=value == "Image to video"),
        inputs=view.mode,
        outputs=view.image_group,
        queue=False,
        show_progress="hidden",
    )
    view.workflow.change(
        render_workflow,
        inputs=view.workflow,
        outputs=view.workflow_details,
        queue=False,
        show_progress="hidden",
    )
    bind_gpu_action(
        view.prepare_workflow.click,
        prepare_workflow,
        inputs=view.workflow,
        outputs=[view.workflow_status, view.model_inventory],
        show_progress="minimal",
    )
    bind_gpu_action(
        view.prepare_all_models.click,
        prepare_all_models,
        outputs=[view.workflow_status, view.model_inventory],
        show_progress="minimal",
    )
    view.refresh_models.click(
        render_inventory,
        outputs=view.model_inventory,
        queue=False,
        show_progress="hidden",
    )
    bind_prompt_action(
        view.enhance.click,
        enhance_prompt,
        inputs=[
            view.prompt,
            view.prompt_model,
            view.api_key,
            view.mode,
            view.image,
            view.middle_image,
            view.end_image,
            view.duration,
            view.width,
            view.height,
            view.prompt_backend,
            view.lightning_api_key,
        ],
        outputs=[view.prompt, view.enhance_status],
        show_progress="minimal",
        api_name="enhance_ltx25_prompt",
    )
    return bind_gpu_action(
        view.run.click,
        owned_generation(generate, "ltx"),
        inputs=[
            view.mode,
            view.model,
            view.prompt,
            view.negative,
            view.image,
            view.duration,
            view.fps,
            view.width,
            view.height,
            view.seed,
            view.cfg,
            view.sampler,
            view.image_strength,
            view.middle_image,
            view.middle_time,
            view.middle_strength,
            view.end_image,
            view.end_strength,
        ],
        outputs=[view.output, view.status],
        show_progress="minimal",
        api_name="generate_ltx25_video",
    )


def bind_music_view(
    view: MusicView,
    *,
    enhance_prompt: Callable[..., Any],
    generate: Callable[..., Any],
) -> Any:
    bind_prompt_action(
        view.enhance.click,
        enhance_prompt,
        inputs=[
            view.caption,
            view.prompt_model,
            view.api_key,
            view.lyrics,
            *view.reference_images,
            view.prompt_backend,
            view.lightning_api_key,
        ],
        outputs=[view.caption, view.lyrics, view.enhance_status],
        show_progress="minimal",
        api_name="enhance_music3_prompt",
    )
    return bind_gpu_action(
        view.run.click,
        owned_generation(generate, "music"),
        inputs=[
            view.model,
            view.caption,
            view.lyrics,
            view.duration,
            view.seed,
            view.steps,
            view.cfg,
            view.ar_cfg,
            view.top_k,
            view.tiled,
        ],
        outputs=[view.output, view.status],
        show_progress="minimal",
        api_name="generate_music3",
    )


QWEN_IMAGE21_PRESETS = {
    "Fast": ("INT8 ConvRot (lower VRAM)", "Viggle Turbo v0.2.1 (6-step)", 6, "Off"),
    "Normal": ("BF16", "Off", 25, "Spectrum (Quality)"),
    "Quality": ("BF16", "Off", 40, "Spectrum (Quality)"),
}


def qwen_preset_values(preset: str):
    """Apply a Qwen preset without locking its individual controls."""
    return QWEN_IMAGE21_PRESETS[preset]


def qwen_turbo_defaults(variant: str, preset: str = "Quality"):
    """Set Turbo defaults, preserving the selected base preset's step count."""
    if variant == "Viggle Turbo v0.2.1 (6-step)":
        return 6, 1.0, "euler", "Off"
    if variant == "Viggle 3-pass (configurable)":
        return 6, 1.0, "euler", "Off"
    if variant == "Alibaba PAI PDD 4-step":
        return 4, 1.0, "euler", "Off"
    if variant == "Pruna 8-step":
        return 8, 1.0, "euler", "Off"
    if variant == "Pruna 5-step":
        return 5, 1.0, "euler", "Off"
    steps = 25 if preset == "Normal" else 40
    return steps, 1.0, "euler", "Spectrum (Quality)"


def bind_qwen_image21_view(
    view: QwenImage21View,
    *,
    enhance_prompt: Callable[..., Any],
    generate: Callable[..., Any],
) -> Any:
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
    for preset in (
        view.square_resolution,
        view.landscape_resolution,
        view.portrait_resolution,
    ):
        preset.change(
            qwen_resolution_preset_values,
            inputs=preset,
            outputs=[view.width, view.height],
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
            view.viggle_pass2_steps,
            view.viggle_pass3_steps,
            view.viggle_pass2_denoise,
            view.viggle_pass3_denoise,
        ],
        outputs=[view.output, view.status],
        show_progress="minimal",
        api_name="generate_qwen_image21",
    )


def bind_yue2_view(
    view: YuE2View,
    *,
    enhance_prompt: Callable[..., Any],
    generate: Callable[..., Any],
) -> Any:
    bind_prompt_action(
        view.enhance.click,
        enhance_prompt,
        inputs=[
            view.style,
            view.prompt_model,
            view.api_key,
            view.lyrics,
            view.mode,
            view.duration,
            view.prompt_backend,
            view.lightning_api_key,
        ],
        outputs=[view.style, view.lyrics, view.enhance_status],
        show_progress="minimal",
        api_name="enhance_yue2_prompt",
    )
    return bind_gpu_action(
        view.run.click,
        owned_generation(generate, "yue2"),
        inputs=[
            view.model, view.style, view.lyrics, view.abc, view.mode, view.duration,
            view.seed, view.steps, view.cfg, view.temperature, view.top_p, view.top_k,
            view.repetition_penalty, view.max_abc_tokens, view.abc_temperature,
            view.abc_top_p, view.abc_top_k, view.abc_repetition_penalty,
            view.abc_penalty_window, view.tiled,
        ],
        outputs=[view.output, view.status],
        show_progress="minimal",
        api_name="generate_yue2",
    )


def bind_api_view(view: ApiView, *, generate: Callable[..., Any]) -> Any:
    return bind_gpu_action(
        view.run.click,
        owned_generation(generate, "api"),
        inputs=view.prompt,
        outputs=[view.download_url, view.status],
        show_progress="minimal",
        api_name="generate_video",
    )


def bind_gallery_view(
    view: GalleryView,
    *,
    tab: gr.Tab,
    selected_ltx_model: gr.Dropdown,
    ai_options: set[str],
    seedvr_option: str,
    ltx_option: str,
    refresh: Callable[..., Any],
    select: Callable[..., Any],
    import_video: Callable[..., Any],
    postprocess: Callable[..., Any],
    interrupt: Callable[..., Any],
    delete: Callable[..., Any],
    empty: Callable[..., Any],
) -> None:
    ltx_options = {ltx_option} | LTX25_RESTORATION_OPTIONS
    video_postprocess_options = [
        choice[1] if isinstance(choice, (tuple, list)) else choice
        for choice in view.postprocess.choices
    ]
    mode_changed = view.mode.change(
        lambda value: (
            gr.update(visible=value == "Video", value=None),
            gr.update(visible=value == "Image", value=None),
            gr.update(visible=value == "Audio", value=None),
            None,
            "",
            gr.update(visible=value == "Video"),
            gr.update(value=False),
            gr.update(
                choices=(
                    [seedvr_option]
                    if value == "Image"
                    else video_postprocess_options
                ),
                value=seedvr_option,
            ),
            gr.update(
                value=(
                    "Upscale selected image"
                    if value == "Image"
                    else "Enhance selected media"
                )
            ),
            gr.update(visible=value != "Audio"),
        ),
        inputs=view.mode,
        outputs=[
            view.player,
            view.image,
            view.audio,
            view.selected,
            view.download,
            view.manage,
            view.confirm_delete,
            view.postprocess,
            view.post_run,
            view.enhance,
        ],
        queue=False,
        show_progress="hidden",
    )
    mode_changed.then(
        refresh,
        inputs=view.mode,
        outputs=[view.grid, view.paths, view.status],
        queue=False,
        show_progress="hidden",
    )
    view.postprocess.change(
        lambda value: (
            gr.update(visible=value in ai_options),
            gr.update(visible=value == seedvr_option),
            gr.update(
                visible=value in ltx_options and value != LTX25_CQ_ENHANCER,
                info=(
                    "Describe the source scene; focus restoration instructions are added automatically. "
                    "Preserves source resolution. Uses the model selected in the LTX 2.5 tab."
                    if value == LTX25_DEBLUR
                    else "Describe the source scene; compression artifact removal instructions are added automatically. "
                    "Preserves source resolution. Uses the model selected in the LTX 2.5 tab."
                    if value in LTX25_RESTORATION_OPTIONS
                    else "Optional but recommended. Uses the transformer selected in the LTX 2.5 tab."
                ),
            ),
            gr.update(visible=value in ltx_options),
            gr.update(visible=value in ltx_options),
            gr.update(
                visible=value in ai_options and value not in LTX25_RESTORATION_OPTIONS
            ),
        ),
        inputs=view.postprocess,
        outputs=[
            view.ai_settings,
            view.seedvr2_model,
            view.ltx25_prompt,
            view.split_upscale,
            view.split_seconds,
            view.upscale_resolution,
        ],
        queue=False,
        show_progress="hidden",
    )
    opened = tab.select(
        lambda value: (
            gr.update(value=None, visible=value == "Video"),
            gr.update(value=None, visible=value == "Image"),
            gr.update(value=None, visible=value == "Audio"),
            "",
            None,
            False,
        ),
        inputs=view.mode,
        outputs=[
            view.player,
            view.image,
            view.audio,
            view.download,
            view.selected,
            view.confirm_delete,
        ],
        queue=False,
        show_progress="hidden",
    )
    opened.then(
        refresh,
        inputs=view.mode,
        outputs=[view.grid, view.paths, view.status],
        queue=False,
        show_progress="hidden",
    )
    view.refresh.click(
        refresh,
        inputs=view.mode,
        outputs=[view.grid, view.paths, view.status],
        queue=False,
        show_progress="hidden",
    )
    view.grid.select(
        select,
        inputs=[view.mode, view.paths],
        outputs=[
            view.player,
            view.image,
            view.audio,
            view.download,
            view.selected,
        ],
        queue=False,
        show_progress="hidden",
    )
    mutation_outputs = [
        view.grid,
        view.paths,
        view.status,
        view.player,
        view.image,
        view.audio,
        view.download,
        view.selected,
        view.confirm_delete,
    ]
    view.import_video.click(
        import_video,
        inputs=[view.mode, view.upload_video],
        outputs=mutation_outputs,
        queue=False,
        show_progress="minimal",
        api_name=False,
    )
    post_event = bind_gpu_action(
        view.post_run.click,
        owned_generation(postprocess, "gallery"),
        inputs=[
            view.mode,
            view.selected,
            view.postprocess,
            view.post_seed,
            view.seedvr2_model,
            selected_ltx_model,
            view.ltx25_prompt,
            view.force_offload,
            view.split_upscale,
            view.split_seconds,
            view.upscale_resolution,
        ],
        outputs=mutation_outputs + [view.post_status],
        show_progress="minimal",
        api_name=False,
    )
    stopped = view.post_stop.click(
        owned_interrupt(interrupt, "gallery"),
        outputs=view.post_status,
        api_name=False,
        queue=False,
    )
    stopped.then(fn=None, cancels=[post_event], queue=False, api_name=False)
    view.delete.click(
        delete,
        inputs=[view.mode, view.selected, view.confirm_delete],
        outputs=mutation_outputs,
        queue=False,
        show_progress="minimal",
        api_name=False,
    )
    view.empty.click(
        empty,
        inputs=[view.mode, view.selected, view.confirm_delete],
        outputs=mutation_outputs,
        queue=False,
        show_progress="minimal",
        api_name=False,
    )
