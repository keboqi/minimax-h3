"""Typed MiniMax H3 view builder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import gradio as gr

from .sections.finishing import build_finishing_section
from .sections.input_upscale import build_input_upscale_section
from .sections.model import build_model_section
from .sections.output import build_output_section
from .sections.performance import build_performance_section
from .sections.prompt import build_prompt_section
from .sections.references import build_references_section
from .sections.results import build_results_section


@dataclass(frozen=True)
class H3ViewServices:
    RESULT_FORMATS: Sequence[str]
    MODEL_PROFILE_CHOICES: Sequence[str]
    H3_TEXT_ENCODER_CHOICES: Mapping[str, Any]
    IMAGE_VAE_CHOICES: Sequence[str]
    mode_help: Callable[[str], str]
    PROMPT_WRITER_BACKENDS: Sequence[str]
    DEFAULT_PROMPT_WRITER_BACKEND: str
    LOCAL_PROMPT_BASE_MODELS: Mapping[str, Any]
    DEFAULT_LOCAL_PROMPT_BASE_MODEL: str
    GEMINI_PROMPT_MODELS: Sequence[str]
    DEFAULT_GEMINI_PROMPT_MODEL: str
    LIGHTNING_PROMPT_MODEL: str
    reference_prompt_help: Callable[[], str]
    INPUT_IMAGE_UPSCALE_SLOTS: Sequence[str]
    INPUT_IMAGE_FRAME_PRESETS: Mapping[str, Any]
    DEFAULT_INPUT_IMAGE_FRAME_PRESET: str
    SEEDVR2_MODEL_CHOICES: Mapping[str, Any]
    compact_settings_summary: Callable[..., Any]
    DEFAULT_LTX25_MODEL: str
    generation_readiness_state: Callable[..., Any]
    MIN_IMAGE_FRAMES: int
    MAX_IMAGE_FRAMES: int
    FAST_RESOLUTIONS: Mapping[str, Any]
    DRAFT_RESOLUTIONS: Mapping[str, Any]
    LARGE_RESOLUTIONS: Mapping[str, Any]
    AUTO_RESOLUTION_MEGAPIXEL_PRESETS: Mapping[str, Any]
    DEFAULT_AUTO_RESOLUTION_MEGAPIXELS: str
    resolution_summary: Callable[..., Any]
    MIN_VIDEO_BATCH_COUNT: int
    MAX_VIDEO_BATCH_COUNT: int
    DEFAULT_VIDEO_BATCH_COUNT: int
    TURBO_SETTINGS: Mapping[str, Any]
    SERVER_ATTENTION_BACKEND: str
    AUTO_SOL_TOKEN_THRESHOLD: int
    SERVER_DENSE_ATTENTION_BACKEND: str
    SLA_PRESET_INPUTS: Mapping[str, Any]
    H3_LATENT_UPSCALER_MODEL_CHOICES: Mapping[str, Any]
    H3_LATENT_UPSCALE_METHODS: Sequence[str]
    H3_LATENT_UPSCALE_SPLIT: str
    GENERATION_POSTPROCESS_OPTIONS: Sequence[str]
    UPSCALE_RESOLUTION_PRESETS: Mapping[str, Any]
    DEFAULT_UPSCALE_RESOLUTION: str


@dataclass(frozen=True)
class H3View:
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
    latent_upscale_refine_steps: gr.components.Component
    latent_upscale_settings: gr.components.Component
    latent_upscaler_model: gr.components.Component
    lightning_api_key: gr.components.Component
    lightning_prompt_writer_group: gr.components.Component
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
    turbo_variant: gr.components.Component
    trt_vae_compile: gr.components.Component
    use_int8_vae: gr.components.Component
    use_trt_vae: gr.components.Component
    width: gr.components.Component

    @property
    def values(self) -> Mapping[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def unpack(self) -> tuple[Any, ...]:
        return tuple(getattr(self, name) for name in H3_COMPONENT_ORDER)


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
    "latent_upscale_refine_steps",
    "latent_upscale_settings",
    "latent_upscaler_model",
    "lightning_api_key",
    "lightning_prompt_writer_group",
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
    "turbo_variant",
    "trt_vae_compile",
    "use_int8_vae",
    "use_trt_vae",
    "width",
)


def build_h3_view(
    generation_view: gr.Row,
    defaults: Mapping[str, Any],
    services: H3ViewServices,
) -> H3View:
    with generation_view:
        with gr.Column(scale=3, elem_classes=["h3-composer"]):
            gr.HTML(
                '<div class="h3-section-intro"><h2>Create</h2>'
                "<p>Choose the output, describe the result, then add media only when needed.</p></div>"
            )
            model_section = build_model_section(defaults, services)
            prompt_section = build_prompt_section(defaults, services)
            references_section = build_references_section(defaults, services)
            input_upscale_section = build_input_upscale_section(defaults, services)
        with gr.Column(
            scale=2,
            min_width=420,
            elem_classes=["h3-settings-panel", "h3-run-panel"],
        ):
            gr.HTML(
                '<div class="h3-section-intro"><h2>Output</h2>'
                "<p>Start with a preset. Advanced controls stay collapsed.</p></div>"
            )
            preset = gr.Radio(
                ["Singularity", "Quality", "Balanced", "Fast"],
                value="Singularity",
                label="Generation preset",
                interactive=True,
                info=(
                    "Sets sampling, text encoding, memory, attention and refinement defaults. "
                    "Singularity selects its base model with Fast settings. Other presets keep your base model. "
                    "Keeps your prompt, media and output size."
                ),
            )
            restore_preset = gr.Button("Restore preset settings", size="sm")
            output_settings_section = gr.Accordion(
                "Output essentials",
                open=False,
                elem_classes=["h3-settings-section"],
            )
            performance_section = gr.Accordion(
                "Performance & sampling (advanced)",
                open=False,
                elem_classes=["h3-settings-section"],
            )
            finishing_section = gr.Accordion(
                "Upscaling & finishing (advanced)",
                open=False,
                elem_classes=["h3-settings-section"],
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
                ),
                elem_classes=["h3-settings-summary"],
            )
            results_section = build_results_section(defaults, services)
            output_section = build_output_section(
                defaults, services, output_settings_section
            )
            performance_controls = build_performance_section(
                defaults, services, performance_section
            )

            finishing_controls = build_finishing_section(
                defaults, services, finishing_section
            )

    return H3View(
        **{
            "settings_used": results_section.settings_used,
            "restore_preset": restore_preset,
            "sla_settings": performance_controls.sla_settings,
            "sol_settings": performance_controls.sol_settings,
            "sol_quality_settings": performance_controls.sol_quality_settings,
            "fbcache_settings": performance_controls.fbcache_settings,
            "easycache_settings": performance_controls.easycache_settings,
            "finishing_section": finishing_section,
            "attention_mode": performance_controls.attention_mode,
            "audio_output": results_section.audio_output,
            "auto_megapixels": output_section.auto_megapixels,
            "batch_count": output_section.batch_count,
            "cache_mode": performance_controls.cache_mode,
            "draft_resolution": output_section.draft_resolution,
            "duration": output_section.duration,
            "easycache_end": performance_controls.easycache_end,
            "easycache_start": performance_controls.easycache_start,
            "easycache_threshold": performance_controls.easycache_threshold,
            "easycache_verbose": performance_controls.easycache_verbose,
            "enhance_prompt_button": prompt_section.enhance_prompt_button,
            "enhance_prompt_status": prompt_section.enhance_prompt_status,
            "fast_resolution": output_section.fast_resolution,
            "fbcache_end": performance_controls.fbcache_end,
            "fbcache_max_hits": performance_controls.fbcache_max_hits,
            "fbcache_preset": performance_controls.fbcache_preset,
            "fbcache_start": performance_controls.fbcache_start,
            "fbcache_temporal_guard": performance_controls.fbcache_temporal_guard,
            "fbcache_threshold": performance_controls.fbcache_threshold,
            "first": references_section.first,
            "frame_group": references_section.frame_group,
            "gemini_api_key": prompt_section.gemini_api_key,
            "gemini_prompt_model": prompt_section.gemini_prompt_model,
            "gemini_prompt_writer_group": prompt_section.gemini_prompt_writer_group,
            "generation_force_offload": finishing_controls.generation_force_offload,
            "generation_ltx25_note": finishing_controls.generation_ltx25_note,
            "generation_mode": model_section.generation_mode,
            "generation_postprocess": finishing_controls.generation_postprocess,
            "generation_postprocess_settings": finishing_controls.generation_postprocess_settings,
            "generation_readiness": results_section.generation_readiness,
            "generation_seedvr2_model": finishing_controls.generation_seedvr2_model,
            "generation_split_seconds": finishing_controls.generation_split_seconds,
            "generation_split_upscale": finishing_controls.generation_split_upscale,
            "generation_upscale_resolution": finishing_controls.generation_upscale_resolution,
            "height": output_section.height,
            "help_text": prompt_section.help_text,
            "image_clear_selection": results_section.image_clear_selection,
            "image_frame_paths": results_section.image_frame_paths,
            "image_frames": output_section.image_frames,
            "image_output": results_section.image_output,
            "image_output_group": results_section.image_output_group,
            "image_save_selected": results_section.image_save_selected,
            "image_save_status": results_section.image_save_status,
            "image_saved_files": results_section.image_saved_files,
            "image_select_all": results_section.image_select_all,
            "image_selection": results_section.image_selection,
            "image_vae": model_section.image_vae,
            "input_upscale_downloads": input_upscale_section.input_upscale_downloads,
            "input_upscale_force_offload": input_upscale_section.input_upscale_force_offload,
            "input_upscale_frame_height": input_upscale_section.input_upscale_frame_height,
            "input_upscale_frame_preset": input_upscale_section.input_upscale_frame_preset,
            "input_upscale_frame_width": input_upscale_section.input_upscale_frame_width,
            "input_upscale_model": input_upscale_section.input_upscale_model,
            "input_upscale_run": input_upscale_section.input_upscale_run,
            "input_upscale_seed": input_upscale_section.input_upscale_seed,
            "input_upscale_slots": input_upscale_section.input_upscale_slots,
            "input_upscale_status": input_upscale_section.input_upscale_status,
            "large_resolution": output_section.large_resolution,
            "last": references_section.last,
            "latent_split_chunk_frames": finishing_controls.latent_split_chunk_frames,
            "latent_split_fade_ratio": finishing_controls.latent_split_fade_ratio,
            "latent_split_overlap_ratio": finishing_controls.latent_split_overlap_ratio,
            "latent_split_seam_denoise": finishing_controls.latent_split_seam_denoise,
            "latent_split_seam_polish": finishing_controls.latent_split_seam_polish,
            "latent_split_settings": finishing_controls.latent_split_settings,
            "latent_split_temporal_overlap_frames": (
                finishing_controls.latent_split_temporal_overlap_frames
            ),
            "latent_split_tile_height": finishing_controls.latent_split_tile_height,
            "latent_split_tile_width": finishing_controls.latent_split_tile_width,
            "latent_upscale": finishing_controls.latent_upscale,
            "latent_upscale_method": finishing_controls.latent_upscale_method,
            "latent_upscale_refine_steps": finishing_controls.latent_upscale_refine_steps,
            "latent_upscale_settings": finishing_controls.latent_upscale_settings,
            "latent_upscaler_model": finishing_controls.latent_upscaler_model,
            "lightning_api_key": prompt_section.lightning_api_key,
            "lightning_prompt_writer_group": prompt_section.lightning_prompt_writer_group,
            "local_prompt_base_model": prompt_section.local_prompt_base_model,
            "local_prompt_greedy": prompt_section.local_prompt_greedy,
            "local_prompt_max_tokens": prompt_section.local_prompt_max_tokens,
            "local_prompt_seed": prompt_section.local_prompt_seed,
            "local_prompt_temperature": prompt_section.local_prompt_temperature,
            "local_prompt_top_p": prompt_section.local_prompt_top_p,
            "local_prompt_writer_group": prompt_section.local_prompt_writer_group,
            "mode": model_section.mode,
            "model_profile": model_section.model_profile,
            "output": results_section.output,
            "output_2": results_section.output_2,
            "output_3": results_section.output_3,
            "output_4": results_section.output_4,
            "preset": preset,
            "prompt": prompt_section.prompt,
            "prompt_writer_backend": prompt_section.prompt_writer_backend,
            "fl2va_audio_1": references_section.fl2va_audio_1,
            "fl2va_audio_2": references_section.fl2va_audio_2,
            "fl2va_audio_3": references_section.fl2va_audio_3,
            "ref_audio_1": references_section.ref_audio_1,
            "ref_audio_2": references_section.ref_audio_2,
            "ref_audio_3": references_section.ref_audio_3,
            "ref_image_1": references_section.ref_image_1,
            "ref_image_2": references_section.ref_image_2,
            "ref_image_3": references_section.ref_image_3,
            "ref_image_4": references_section.ref_image_4,
            "ref_image_5": references_section.ref_image_5,
            "ref_image_6": references_section.ref_image_6,
            "ref_image_7": references_section.ref_image_7,
            "ref_image_8": references_section.ref_image_8,
            "ref_image_9": references_section.ref_image_9,
            "ref_size": references_section.ref_size,
            "ref_video_1": references_section.ref_video_1,
            "ref_video_2": references_section.ref_video_2,
            "ref_video_3": references_section.ref_video_3,
            "reference_group": references_section.reference_group,
            "refresh": results_section.refresh,
            "resolution_info": output_section.resolution_info,
            "result_format": model_section.result_format,
            "semantic_bridge": model_section.semantic_bridge,
            "semantic_bridge_alpha": model_section.semantic_bridge_alpha,
            "reuse_unchanged_inputs": model_section.reuse_unchanged_inputs,
            "run": results_section.run,
            "scheduler": performance_controls.scheduler,
            "seed": output_section.seed,
            "settings_overview": settings_overview,
            "sla_preset": performance_controls.sla_preset,
            "sol_dense_steps": performance_controls.sol_dense_steps,
            "sol_exact_mode": performance_controls.sol_exact_mode,
            "sol_sink_tokens": performance_controls.sol_sink_tokens,
            "sol_step_off": performance_controls.sol_step_off,
            "sol_tau": performance_controls.sol_tau,
            "sol_thresh_type": performance_controls.sol_thresh_type,
            "stage_model_offload": model_section.stage_model_offload,
            "status": results_section.status,
            "steps": output_section.steps,
            "stop": results_section.stop,
            "text_encoder": model_section.text_encoder,
            "turbo_variant": performance_controls.turbo_variant,
            "trt_vae_compile": model_section.trt_vae_compile,
            "use_int8_vae": model_section.use_int8_vae,
            "use_trt_vae": model_section.use_trt_vae,
            "width": output_section.width,
        }
    )
