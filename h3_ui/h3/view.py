"""Typed MiniMax H3 view builder."""

from __future__ import annotations
from dataclasses import dataclass, fields
from typing import Any, Mapping
import gradio as gr
from .services import H3ViewServices
from .sections.finishing import build_finishing_section
from .sections.input_upscale import build_input_upscale_section
from .sections.model import build_model_section
from .sections.output import build_output_section
from .sections.performance import build_performance_section
from .sections.prompt import build_prompt_section
from .sections.references import build_references_section
from .sections.results import build_results_section


@dataclass(frozen=True)
class H3View:
    aspect_ratio: gr.components.Component | None
    size_tier: gr.components.Component | None
    prompt_preview: gr.components.Component
    prompt_proposal: gr.components.Component
    prompt_previous: gr.components.Component
    accept_prompt: gr.components.Component
    keep_prompt: gr.components.Component
    undo_prompt: gr.components.Component
    reference_map: gr.components.Component
    reference_status: gr.components.Component
    repair_references: gr.components.Component
    reference_tag: gr.components.Component
    insert_reference_tag: gr.components.Component
    ltx25_model: gr.components.Component
    video_decoder: gr.components.Component
    settings_used: gr.HTML
    restore_preset: gr.components.Component
    sla_settings: gr.components.Component
    sol_settings: gr.components.Component
    sol_quality_settings: gr.components.Component
    fbcache_settings: gr.components.Component
    easycache_settings: gr.components.Component
    finishing_section: gr.components.Component
    attention_mode: gr.components.Component
    audio_output: gr.components.Component
    auto_megapixels: gr.components.Component
    batch_count: gr.components.Component
    cache_mode: gr.components.Component
    draft_resolution: gr.components.Component
    duration: gr.components.Component
    easycache_end: gr.components.Component
    easycache_start: gr.components.Component
    easycache_threshold: gr.components.Component
    easycache_verbose: gr.components.Component
    enhance_prompt_button: gr.components.Component
    enhance_prompt_status: gr.components.Component
    fast_resolution: gr.components.Component
    fbcache_end: gr.components.Component
    fbcache_max_hits: gr.components.Component
    fbcache_preset: gr.components.Component
    fbcache_start: gr.components.Component
    fbcache_temporal_guard: gr.components.Component
    fbcache_threshold: gr.components.Component
    first: gr.components.Component
    frame_group: gr.components.Component
    gemini_api_key: gr.components.Component
    gemini_prompt_model: gr.components.Component
    gemini_prompt_writer_group: gr.components.Component
    generation_force_offload: gr.components.Component
    generation_ltx25_note: gr.components.Component
    generation_mode: gr.components.Component
    generation_postprocess: gr.components.Component
    generation_postprocess_settings: gr.components.Component
    generation_readiness: gr.components.Component
    generation_seedvr2_model: gr.components.Component
    generation_split_seconds: gr.components.Component
    generation_split_upscale: gr.components.Component
    generation_upscale_resolution: gr.components.Component
    height: gr.components.Component
    help_text: gr.components.Component
    image_clear_selection: gr.components.Component
    image_frame_paths: gr.components.Component
    image_frames: gr.components.Component
    image_output: gr.components.Component
    image_output_group: gr.components.Component
    image_save_selected: gr.components.Component
    image_save_status: gr.components.Component
    image_saved_files: gr.components.Component
    image_select_all: gr.components.Component
    image_selection: gr.components.Component
    image_vae: gr.components.Component
    input_upscale_downloads: gr.components.Component
    input_upscale_force_offload: gr.components.Component
    input_upscale_frame_height: gr.components.Component
    input_upscale_frame_preset: gr.components.Component
    input_upscale_frame_width: gr.components.Component
    input_upscale_model: gr.components.Component
    input_upscale_run: gr.components.Component
    input_upscale_seed: gr.components.Component
    input_upscale_slots: gr.components.Component
    input_upscale_status: gr.components.Component
    large_resolution: gr.components.Component
    last: gr.components.Component
    latent_split_chunk_frames: gr.components.Component
    latent_split_fade_ratio: gr.components.Component
    latent_split_overlap_ratio: gr.components.Component
    latent_split_seam_denoise: gr.components.Component
    latent_split_seam_polish: gr.components.Component
    latent_split_settings: gr.components.Component
    latent_split_temporal_overlap_frames: gr.components.Component
    latent_split_tile_height: gr.components.Component
    latent_split_tile_width: gr.components.Component
    latent_upscale: gr.components.Component
    latent_upscale_method: gr.components.Component
    latent_upscale_refine_lora: gr.components.Component
    latent_upscale_refine_steps: gr.components.Component
    latent_upscale_settings: gr.components.Component
    latent_upscaler_model: gr.components.Component
    openai_api_key: gr.components.Component
    openai_model: gr.components.Component
    openai_base_url: gr.components.Component
    openai_prompt_writer_group: gr.components.Component
    local_prompt_base_model: gr.components.Component
    local_prompt_greedy: gr.components.Component
    local_prompt_max_tokens: gr.components.Component
    local_prompt_seed: gr.components.Component
    local_prompt_temperature: gr.components.Component
    local_prompt_top_p: gr.components.Component
    local_prompt_writer_group: gr.components.Component
    mode: gr.components.Component
    model_profile: gr.components.Component
    output: gr.components.Component
    output_2: gr.components.Component
    output_3: gr.components.Component
    output_4: gr.components.Component
    preset: gr.components.Component
    prompt: gr.components.Component
    prompt_writer_backend: gr.components.Component
    fl2va_audio_1: gr.components.Component
    fl2va_audio_2: gr.components.Component
    fl2va_audio_3: gr.components.Component
    ref_audio_1: gr.components.Component
    ref_audio_2: gr.components.Component
    ref_audio_3: gr.components.Component
    ref_image_1: gr.components.Component
    ref_image_2: gr.components.Component
    ref_image_3: gr.components.Component
    ref_image_4: gr.components.Component
    ref_image_5: gr.components.Component
    ref_image_6: gr.components.Component
    ref_image_7: gr.components.Component
    ref_image_8: gr.components.Component
    ref_image_9: gr.components.Component
    ref_size: gr.components.Component
    ref_video_1: gr.components.Component
    ref_video_2: gr.components.Component
    ref_video_3: gr.components.Component
    reference_group: gr.components.Component
    refresh: gr.components.Component
    resolution_info: gr.components.Component
    result_format: gr.components.Component
    semantic_bridge: gr.components.Component
    semantic_bridge_alpha: gr.components.Component
    reuse_unchanged_inputs: gr.components.Component
    run: gr.components.Component
    scheduler: gr.components.Component
    seed: gr.components.Component
    settings_overview: gr.components.Component
    sla_preset: gr.components.Component
    sol_dense_steps: gr.components.Component
    sol_exact_mode: gr.components.Component
    sol_sink_tokens: gr.components.Component
    sol_step_off: gr.components.Component
    sol_tau: gr.components.Component
    sol_thresh_type: gr.components.Component
    stage_model_offload: gr.components.Component
    status: gr.components.Component
    steps: gr.components.Component
    stop: gr.components.Component
    text_encoder: gr.components.Component
    encoder_small_input: gr.components.Component
    turbo_variant: gr.components.Component
    trt_vae_compile: gr.components.Component
    use_int8_vae: gr.components.Component
    use_lynnreal_vae: gr.components.Component
    video_encoder: gr.components.Component
    lightvae_decoder: gr.components.Component
    use_trt_vae: gr.components.Component
    width: gr.components.Component

    @property
    def values(self) -> Mapping[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def unpack(self) -> tuple[Any, ...]:
        return tuple((getattr(self, name) for name in H3_COMPONENT_ORDER))


H3_COMPONENT_ORDER = (
    "attention_mode",
    "audio_output",
    "auto_megapixels",
    "batch_count",
    "cache_mode",
    "draft_resolution",
    "duration",
    "easycache_end",
    "easycache_start",
    "easycache_threshold",
    "easycache_verbose",
    "enhance_prompt_button",
    "enhance_prompt_status",
    "fast_resolution",
    "fbcache_end",
    "fbcache_max_hits",
    "fbcache_preset",
    "fbcache_start",
    "fbcache_temporal_guard",
    "fbcache_threshold",
    "first",
    "frame_group",
    "gemini_api_key",
    "gemini_prompt_model",
    "gemini_prompt_writer_group",
    "generation_force_offload",
    "generation_ltx25_note",
    "generation_mode",
    "generation_postprocess",
    "generation_postprocess_settings",
    "generation_readiness",
    "generation_seedvr2_model",
    "generation_split_seconds",
    "generation_split_upscale",
    "generation_upscale_resolution",
    "height",
    "help_text",
    "image_clear_selection",
    "image_frame_paths",
    "image_frames",
    "image_output",
    "image_output_group",
    "image_save_selected",
    "image_save_status",
    "image_saved_files",
    "image_select_all",
    "image_selection",
    "image_vae",
    "input_upscale_downloads",
    "input_upscale_force_offload",
    "input_upscale_frame_height",
    "input_upscale_frame_preset",
    "input_upscale_frame_width",
    "input_upscale_model",
    "input_upscale_run",
    "input_upscale_seed",
    "input_upscale_slots",
    "input_upscale_status",
    "large_resolution",
    "last",
    "latent_split_chunk_frames",
    "latent_split_fade_ratio",
    "latent_split_overlap_ratio",
    "latent_split_seam_denoise",
    "latent_split_seam_polish",
    "latent_split_settings",
    "latent_split_temporal_overlap_frames",
    "latent_split_tile_height",
    "latent_split_tile_width",
    "latent_upscale",
    "latent_upscale_method",
    "latent_upscale_refine_lora",
    "latent_upscale_refine_steps",
    "latent_upscale_settings",
    "latent_upscaler_model",
    "openai_api_key",
    "openai_model",
    "openai_base_url",
    "openai_prompt_writer_group",
    "local_prompt_base_model",
    "local_prompt_greedy",
    "local_prompt_max_tokens",
    "local_prompt_seed",
    "local_prompt_temperature",
    "local_prompt_top_p",
    "local_prompt_writer_group",
    "mode",
    "model_profile",
    "output",
    "output_2",
    "output_3",
    "output_4",
    "preset",
    "prompt",
    "prompt_writer_backend",
    "fl2va_audio_1",
    "fl2va_audio_2",
    "fl2va_audio_3",
    "ref_audio_1",
    "ref_audio_2",
    "ref_audio_3",
    "ref_image_1",
    "ref_image_2",
    "ref_image_3",
    "ref_image_4",
    "ref_image_5",
    "ref_image_6",
    "ref_image_7",
    "ref_image_8",
    "ref_image_9",
    "ref_size",
    "ref_video_1",
    "ref_video_2",
    "ref_video_3",
    "reference_group",
    "refresh",
    "resolution_info",
    "result_format",
    "semantic_bridge",
    "semantic_bridge_alpha",
    "reuse_unchanged_inputs",
    "run",
    "scheduler",
    "seed",
    "settings_overview",
    "sla_preset",
    "sol_dense_steps",
    "sol_exact_mode",
    "sol_sink_tokens",
    "sol_step_off",
    "sol_tau",
    "sol_thresh_type",
    "stage_model_offload",
    "status",
    "steps",
    "stop",
    "text_encoder",
    "encoder_small_input",
    "turbo_variant",
    "trt_vae_compile",
    "use_int8_vae",
    "use_lynnreal_vae",
    "video_encoder",
    "lightvae_decoder",
    "use_trt_vae",
    "width",
)


def build_h3_view(
    generation_view: gr.Row,
    defaults: Mapping[str, Any],
    services: H3ViewServices,
    *,
    advanced_root=None,
) -> H3View:
    (
        model_section,
        prompt_section,
        references_section,
        input_upscale_section,
        preset,
        restore_preset,
        settings_overview,
        results_section,
        output_section,
        performance_controls,
        finishing_controls,
        finishing_section,
    ) = _build_workspace_h3(
        generation_view,
        defaults,
        services,
        advanced_root=advanced_root,
    )
    results_section.run.h3_canvas = output_section.canvas_controls
    from ..canvas_controls import bind_image_canvas_preview

    bind_image_canvas_preview(
        output_section,
        references_section.first,
        model_section.result_format,
        model_section.mode,
    )
    # Each control keeps its section field's name; the recipe controls live here.
    controls = {
        "preset": preset,
        "restore_preset": restore_preset,
        "settings_overview": settings_overview,
        "finishing_section": finishing_section,
    }
    for section in (
        model_section, prompt_section, references_section, input_upscale_section,
        results_section, output_section, performance_controls, finishing_controls,
    ):
        for field in fields(section):
            name = field.name
            if name in controls:
                raise ValueError(f"H3 sections define {name!r} twice.")
            controls[name] = getattr(section, name)
    return H3View(**{field.name: controls[field.name] for field in fields(H3View)})


def _build_workspace_h3(generation_view, defaults, services, *, advanced_root=None):
    if advanced_root is None:
        raise ValueError("Workspace composition requires an advanced settings root.")
    with advanced_root:
        with gr.Tabs():
            with gr.Tab("Model & memory"):
                model_root = gr.Group()
            with gr.Tab("Sampling & performance"):
                performance_section = gr.Group()
            with gr.Tab("Upscaling & finishing"):
                finishing_section = gr.Group()
            with gr.Tab("Input images"):
                input_root = gr.Group()
    with generation_view:
        with gr.Column(
            scale=6,
            min_width=320,
            elem_id="h3-composer",
            elem_classes=["h3-composer", "h3-run-panel"],
        ):
            gr.HTML(
                '<div class="h3-panel-heading"><span class="h3-step" aria-hidden="true">01</span>'
                '<div><h3>Compose</h3></div></div>'
            )
            conditioning_root = gr.Group()
            references_section = build_references_section(defaults, services)
            prompt_section = build_prompt_section(defaults, services)
            with gr.Column(elem_id="h3-output-settings"):
                gr.HTML(
                    '<div class="h3-panel-heading"><span class="h3-step" aria-hidden="true">02</span>'
                    '<div><h3>Make it fit</h3><p>Choose a recipe and your output settings.</p></div></div>'
                )
                preset = gr.Radio(
                    ["Singularity", "Quality", "Balanced", "Fast"],
                    value="Singularity",
                    label="Recipe",
                    interactive=True,
                    info="Applies generation defaults. Keeps your prompt, media and output size.",
                )
                with gr.Accordion("Changes these settings", open=False):
                    gr.Markdown(
                        "Recipes change sampling, text encoding, memory, attention, refinement and decoding. Singularity also selects its checkpoint; other recipes keep the selected checkpoint. Prompt, media and output intent are preserved. Explicit edits appear in the next-run summary."
                    )
                    restore_preset = gr.Button("Restore preset settings", size="sm")
                model_section = build_model_section(
                    defaults,
                    services,
                    advanced_parent=model_root,
                    conditioning_parent=conditioning_root,
                )
                output_settings_section = gr.Column(elem_classes=["h3-essentials"])
                output_section = build_output_section(
                    defaults, services, output_settings_section
                )
        with gr.Column(
            scale=6,
            min_width=320,
            elem_id="h3-preview",
            elem_classes=["h3-preview-panel"],
        ):
            results_root = gr.Column()
            action_root = gr.Column(
                elem_id="h3-generate-actions", elem_classes=["h3-actions-slot"]
            )
            settings_overview = gr.HTML(
                services.compact_settings_summary(
                    defaults["mode"],
                    defaults["model_profile"],
                    defaults["text_encoder"],
                    defaults["stage_model_offload"],
                    defaults["reuse_unchanged_inputs"],
                    defaults["use_int8_vae"],
                    defaults["generation_mode"],
                    defaults["turbo_variant"],
                    defaults["duration"],
                    defaults["width"],
                    defaults["height"],
                    defaults["steps"],
                    defaults["scheduler"],
                    defaults["attention_mode"],
                    defaults["sla_preset"],
                    defaults["cache_mode"],
                    defaults["latent_upscale"],
                    defaults["latent_upscaler_model"],
                    defaults["latent_upscale_refine_steps"],
                    defaults["postprocess"],
                    defaults["seedvr2_model"],
                    services.DEFAULT_LTX25_MODEL,
                    defaults["upscale_force_offload"],
                    defaults["upscale_split_enabled"],
                    defaults["upscale_split_seconds"],
                    defaults["result_format"],
                    defaults["image_frames"],
                    defaults["image_vae"],
                    defaults["latent_upscale_method"],
                    defaults["latent_split_tile_width"],
                    defaults["latent_split_tile_height"],
                    defaults["latent_split_overlap_ratio"],
                    defaults["latent_split_fade_ratio"],
                    defaults["latent_split_chunk_frames"],
                    defaults["latent_split_temporal_overlap_frames"],
                    defaults["latent_split_seam_denoise"],
                    defaults["latent_split_seam_polish"],
                    use_trt_vae=defaults["use_trt_vae"],
                    use_lynnreal_vae=defaults["use_lynnreal_vae"],
                    video_encoder=defaults["video_encoder"],
                    lightvae_decoder=defaults["lightvae_decoder"],
                ),
                elem_classes=["h3-settings-summary"],
            )
            with results_root:
                results_section = build_results_section(
                    defaults, services, action_root=action_root
                )
    performance_controls = build_performance_section(
        defaults, services, performance_section
    )
    finishing_controls = build_finishing_section(defaults, services, finishing_section)
    with input_root:
        input_upscale_section = build_input_upscale_section(defaults, services)
    return (
        model_section,
        prompt_section,
        references_section,
        input_upscale_section,
        preset,
        restore_preset,
        settings_overview,
        results_section,
        output_section,
        performance_controls,
        finishing_controls,
        finishing_section,
    )
