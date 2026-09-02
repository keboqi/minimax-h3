"""Typed MiniMax H3 view builder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import gradio as gr


@dataclass(frozen=True)
class H3View:
    values: Mapping[str, Any]

    def unpack(self) -> tuple[Any, ...]:
        return tuple(self.values[name] for name in H3_COMPONENT_ORDER)


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
    services: Mapping[str, Any],
) -> H3View:
    with generation_view:
        with gr.Column(scale=3, elem_classes=["h3-composer"]):
            gr.HTML(
                '<div class="h3-section-intro"><h2>Create</h2>'
                "<p>Choose the output, describe the result, then add media only when needed.</p></div>"
            )
            with gr.Row(elem_classes=["h3-mode-row"]):
                mode = gr.Radio(
                    ["Text to video", "First / last frame", "Reference media"],
                    value=defaults["mode"],
                    label="Conditioning mode",
                )
                result_format = gr.Radio(
                    services["RESULT_FORMATS"],
                    value=defaults["result_format"],
                    label="Result format",
                    info="H3 always samples vision and audio; this selects what is decoded and shown.",
                )
            with gr.Row():
                model_profile = gr.Radio(
                    services["MODEL_PROFILE_CHOICES"],
                    value=defaults["model_profile"],
                    label="Base model",
                    info=(
                        "Speed uses the rebuilt single-pass NVFP4 files. "
                        "Quality uses the mixed NVFP4/FP8/INT8 ConvRot files. "
                        "Original uses the official BF16 files. Speed and Original "
                        "download when first selected."
                    ),
                )
                generation_mode = gr.Radio(
                    ["Normal", "Turbo"],
                    value=defaults["generation_mode"],
                    label="Generation",
                    info=(
                        "Turbo uses the implementation selected in Generation settings. "
                        "Reference media temporarily uses the corresponding FL2VA-trained "
                        "LoRA and is experimental."
                    ),
                )
            with gr.Accordion(
                "Model and memory (advanced)",
                open=False,
                elem_classes=["h3-advanced-block"],
            ):
                with gr.Row():
                    text_encoder = gr.Dropdown(
                        choices=list(services["H3_TEXT_ENCODER_CHOICES"]),
                        value=defaults["text_encoder"],
                        label="Text encoder",
                        info=(
                            "BF16 is approximately 51.5 GB. "
                            "NVFP4/AWQ and INT8 ConvRot download on first use."
                        ),
                    )
                    stage_model_offload = gr.Checkbox(
                        value=defaults["stage_model_offload"],
                        interactive=defaults["text_encoder"] != "BF16",
                        label="Offload models between H3 stages",
                        info=(
                            "Unload resident models between text encoding, diffusion, "
                            "latent upscaling, and VAE decoding."
                        ),
                    )
                    reuse_unchanged_inputs = gr.Checkbox(
                        value=defaults["reuse_unchanged_inputs"],
                        label="Reuse unchanged prompt and media",
                        info=(
                            "Use content-addressed staged inputs so ComfyUI can skip "
                            "unchanged loading and conditioning work. Sampling still "
                            "reruns when the seed changes."
                        ),
                    )
                use_int8_vae = gr.Checkbox(
                    value=defaults["use_int8_vae"],
                    label="Experimental INT8 ConvRot video VAE",
                    info=(
                        "Default off. Downloads on first use and accelerates H3 video "
                        "encode/decode; switch off for the reviewed FP16 path."
                    ),
                )
                with gr.Row():
                    use_trt_vae = gr.Checkbox(
                        value=defaults["use_trt_vae"],
                        label="Experimental TensorRT video VAE",
                        info="Uses a local TensorRT engine for final H3 video decoding.",
                        scale=2,
                    )
                    trt_vae_compile = gr.Button(
                        "Compile TensorRT VAE engine",
                        scale=1,
                    )
                image_vae = gr.Radio(
                    services["IMAGE_VAE_CHOICES"],
                    value=defaults["image_vae"],
                    label="Image VAE",
                    visible=False,
                    info=(
                        "Official is the default and remains the only video decoder. "
                        "The experimental 500K option downloads 9.69 GB on first use "
                        "and decodes one image from temporal latent slice 0."
                    ),
                )
            help_text = gr.Markdown(services["mode_help"]("Text to video"))
            prompt = gr.Textbox(
                label="Prompt",
                lines=12,
                placeholder="Describe shots, camera motion, dialogue, sound effects, ambience, music, and any tagged references.",
            )
            with gr.Accordion("Prompt writer / enhancer", open=False):
                gr.Markdown(
                    "The local MiniMax-H3 8B writer supports T2VA, I2VA, "
                    "L2VA, and FL2VA. Gemini supports all Reference media; "
                    "Lightning AI supports text and image enhancement."
                )
                prompt_writer_backend = gr.Radio(
                    services["PROMPT_WRITER_BACKENDS"],
                    value=services["DEFAULT_PROMPT_WRITER_BACKEND"],
                    label="Prompt writer",
                )
                with gr.Group(visible=False) as local_prompt_writer_group:
                    local_prompt_base_model = gr.Dropdown(
                        choices=list(services["LOCAL_PROMPT_BASE_MODELS"]),
                        value=services["DEFAULT_LOCAL_PROMPT_BASE_MODEL"],
                        label="Local base model",
                        info=(
                            "BF16 is the default full-precision checkpoint. FP8 is "
                            "available as the lower-memory alternative."
                        ),
                    )
                    with gr.Accordion("Local decoding settings", open=False):
                        local_prompt_greedy = gr.Checkbox(
                            value=True, label="Greedy decoding"
                        )
                        local_prompt_max_tokens = gr.Slider(
                            256,
                            8192,
                            value=4096,
                            step=256,
                            label="Max new tokens",
                        )
                        with gr.Row():
                            local_prompt_temperature = gr.Slider(
                                0.1,
                                2.0,
                                value=0.7,
                                step=0.1,
                                label="Temperature (sampling)",
                            )
                            local_prompt_top_p = gr.Slider(
                                0.05,
                                1.0,
                                value=0.8,
                                step=0.05,
                                label="Top-p (sampling)",
                            )
                        local_prompt_seed = gr.Number(
                            value=42, precision=0, label="Seed"
                        )
                with gr.Group(visible=True) as gemini_prompt_writer_group:
                    gr.Markdown(
                        "Uses the active inputs with `prompt.txt`. Set "
                        "`GEMINI_API_KEY` on the server or enter a temporary key; "
                        "the server does not store UI keys."
                    )
                    with gr.Row():
                        gemini_prompt_model = gr.Dropdown(
                            choices=list(services["GEMINI_PROMPT_MODELS"]),
                            value=services["DEFAULT_GEMINI_PROMPT_MODEL"],
                            label="Gemini model",
                        )
                        gemini_api_key = gr.Textbox(
                            label="Temporary Gemini API key",
                            type="password",
                            placeholder="Uses GEMINI_API_KEY when blank",
                        )
                with gr.Group(visible=False) as lightning_prompt_writer_group:
                    gr.Markdown(
                        f"Uses `{services['LIGHTNING_PROMPT_MODEL']}` with the active text "
                        "and images plus `prompt.txt`. Video and audio references "
                        "require Gemini. Set `LIGHTNING_API_KEY` on the server or "
                        "enter a temporary key; the server does not store UI keys."
                    )
                    lightning_api_key = gr.Textbox(
                        label="Temporary Lightning API key",
                        type="password",
                        placeholder="Uses LIGHTNING_API_KEY when blank",
                    )
                enhance_prompt_button = gr.Button("Generate / enhance prompt")
                enhance_prompt_status = gr.Textbox(
                    label="Prompt enhancer status", lines=2, interactive=False
                )
            with gr.Group(visible=False) as frame_group:
                gr.Markdown("### First / last frame inputs")
                with gr.Row():
                    first = gr.Image(
                        type="filepath",
                        label="First frame (auto resolution)",
                        elem_id="first-frame-image",
                    )
                    last = gr.Image(type="filepath", label="Last frame")
            with gr.Group(visible=False) as reference_group:
                gr.Markdown("### Reference media")
                gr.Markdown(services["reference_prompt_help"]())
                with gr.Accordion("Reference images · up to 9", open=True):
                    with gr.Row():
                        ref_image_1 = gr.Image(type="filepath", label="Picture 1")
                        ref_image_2 = gr.Image(type="filepath", label="Picture 2")
                        ref_image_3 = gr.Image(type="filepath", label="Picture 3")
                    with gr.Row():
                        ref_image_4 = gr.Image(type="filepath", label="Picture 4")
                        ref_image_5 = gr.Image(type="filepath", label="Picture 5")
                        ref_image_6 = gr.Image(type="filepath", label="Picture 6")
                    with gr.Row():
                        ref_image_7 = gr.Image(type="filepath", label="Picture 7")
                        ref_image_8 = gr.Image(type="filepath", label="Picture 8")
                        ref_image_9 = gr.Image(type="filepath", label="Picture 9")
                with gr.Accordion("Reference videos · up to 3", open=False):
                    with gr.Row():
                        ref_video_1 = gr.Video(label="Video 1")
                        ref_video_2 = gr.Video(label="Video 2")
                        ref_video_3 = gr.Video(label="Video 3")
                with gr.Accordion("Reference audio · up to 3", open=False):
                    with gr.Row():
                        ref_audio_1 = gr.Audio(type="filepath", label="Audio 1")
                        ref_audio_2 = gr.Audio(type="filepath", label="Audio 2")
                        ref_audio_3 = gr.Audio(type="filepath", label="Audio 3")
                ref_size = gr.Radio(
                    ["match", "max"],
                    value=defaults["ref_image_size"],
                    label="Reference image size",
                )
            with gr.Accordion(
                "Upscale input images with SeedVR2",
                open=False,
                elem_classes=["h3-advanced-block"],
            ):
                gr.Markdown(
                    "Select uploaded start/end frames or reference pictures, then "
                    "fit smaller images upward into a target frame before generation. "
                    "Aspect ratio is preserved and images already at or beyond the "
                    "frame are not downscaled."
                )
                input_upscale_slots = gr.CheckboxGroup(
                    choices=list(services["INPUT_IMAGE_UPSCALE_SLOTS"]),
                    value=[],
                    label="Images to upscale",
                )
                input_upscale_frame_preset = gr.Dropdown(
                    choices=list(services["INPUT_IMAGE_FRAME_PRESETS"]),
                    value=services["DEFAULT_INPUT_IMAGE_FRAME_PRESET"],
                    label="Target frame preset",
                )
                with gr.Row():
                    input_upscale_frame_width = gr.Number(
                        value=1920,
                        precision=0,
                        label="Frame width",
                    )
                    input_upscale_frame_height = gr.Number(
                        value=1920,
                        precision=0,
                        label="Frame height",
                    )
                with gr.Row():
                    input_upscale_model = gr.Dropdown(
                        choices=list(services["SEEDVR2_MODEL_CHOICES"]),
                        value=defaults["seedvr2_model"],
                        label="SeedVR2 model",
                    )
                    input_upscale_seed = gr.Number(
                        value=-1,
                        precision=0,
                        label="Seed (-1 random)",
                    )
                input_upscale_force_offload = gr.Checkbox(
                    value=False,
                    label="Unload resident models before input upscaling",
                    info="Useful when H3 or another large model is already resident in VRAM.",
                )
                input_upscale_run = gr.Button(
                    "Upscale selected inputs to frame", variant="secondary"
                )
                input_upscale_downloads = gr.File(
                    label="Selected input image files",
                    file_count="multiple",
                    interactive=False,
                )
                input_upscale_status = gr.Markdown()
        with gr.Column(
            scale=2,
            min_width=420,
            elem_classes=["h3-settings-panel", "h3-run-panel"],
        ):
            gr.HTML(
                '<div class="h3-section-intro"><h2>Output</h2>'
                "<p>Start with a preset. Advanced controls stay collapsed.</p></div>"
            )
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
                services["compact_settings_summary"](
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
                    services["DEFAULT_LTX25_MODEL"],
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
            with gr.Group(elem_classes=["h3-action-dock"]):
                generation_readiness = gr.HTML(
                    services["generation_readiness_state"](
                        defaults["mode"], "", None, None
                    ).html
                )
                with gr.Row():
                    run = gr.Button(
                        "Generate video",
                        variant="primary",
                        scale=2,
                        interactive=False,
                        elem_classes=["h3-primary-action"],
                    )
                    stop = gr.Button("Interrupt", scale=1)
                    refresh = gr.Button("Refresh status", scale=1)
                status = gr.Textbox(
                    label="Generation progress",
                    lines=2,
                    interactive=False,
                    elem_classes=["h3-status"],
                )
            gr.HTML(
                '<div class="h3-section-intro"><h3>Results</h3>'
                "<p>Your latest output remains here while you adjust settings.</p></div>"
            )
            with gr.Row():
                output = gr.Video(label="Generated video 1")
                output_2 = gr.Video(label="Generated video 2", visible=False)
            with gr.Row():
                output_3 = gr.Video(label="Generated video 3", visible=False)
                output_4 = gr.Video(label="Generated video 4", visible=False)
            with gr.Group(visible=False) as image_output_group:
                image_output = gr.Gallery(
                    value=[],
                    label="Generated image frames",
                    columns=4,
                    object_fit="contain",
                    allow_preview=True,
                    height=520,
                )
                image_frame_paths = gr.State([])
                image_selection = gr.CheckboxGroup(
                    choices=[],
                    value=[],
                    label="Frames to save",
                )
                with gr.Row():
                    image_select_all = gr.Button("Select all")
                    image_clear_selection = gr.Button("Clear selection")
                    image_save_selected = gr.Button(
                        "Save selected frames", variant="primary"
                    )
                image_saved_files = gr.File(
                    label="Saved image files",
                    file_count="multiple",
                    interactive=False,
                )
                image_save_status = gr.Markdown()
            audio_output = gr.Audio(
                label="Generated audio", type="filepath", visible=False
            )
            with output_settings_section:
                gr.Markdown(
                    "Choose the result length, quality target, canvas, and seed."
                )
                preset = gr.Radio(
                    ["Quality", "Balanced", "Fast"],
                    value="Fast",
                    label="Sampling preset",
                    interactive=True,
                    info=(
                        "Sets sampling, text encoder, model offload, start-frame cap, "
                        "Turbo LoRA, attention, SLA, and high-resolution refinement defaults."
                    ),
                )
                with gr.Row():
                    duration = gr.Slider(
                        2, 15, value=defaults["duration"], step=0.5, label="Seconds"
                    )
                    image_frames = gr.Slider(
                        services["MIN_IMAGE_FRAMES"],
                        services["MAX_IMAGE_FRAMES"],
                        value=defaults["image_frames"],
                        step=1,
                        label="Image frames",
                        visible=False,
                        info=(
                            "The official VAE returns 1–20 decoded video frames. "
                            "Selecting the 500K decoder fixes this to one image."
                        ),
                    )
                    steps = gr.Slider(
                        4,
                        30,
                        value=defaults["steps"],
                        step=1,
                        label="Steps",
                        info=(
                            "LightX2V 4-step is the default Turbo variant; Larry and the "
                            "8-step LightX2V variant keep their trained step counts. Increase Turbo "
                            "steps when a clip benefits from extra refinement; Normal H3 "
                            "presets normally use 15–20."
                        ),
                    )
                fast_resolution = gr.Dropdown(
                    choices=list(services["FAST_RESOLUTIONS"]),
                    value="16:9 · 864×480",
                    label="Recommended size",
                    info="Recommended working resolutions by aspect ratio.",
                )
                with gr.Accordion("More resolution presets", open=False):
                    with gr.Row():
                        draft_resolution = gr.Dropdown(
                            choices=list(services["DRAFT_RESOLUTIONS"]),
                            value=None,
                            label="Draft preview",
                            info="Small sizes for quick composition tests.",
                        )
                        large_resolution = gr.Dropdown(
                            choices=list(services["LARGE_RESOLUTIONS"]),
                            value=None,
                            label="Large output",
                            info="Higher-resolution sizes that need more time and VRAM.",
                        )
                with gr.Row():
                    width = gr.Number(
                        value=defaults["width"], precision=0, label="Width"
                    )
                    height = gr.Number(
                        value=defaults["height"], precision=0, label="Height"
                    )
                    auto_megapixels = gr.Dropdown(
                        choices=list(services["AUTO_RESOLUTION_MEGAPIXEL_PRESETS"]),
                        value=services["DEFAULT_AUTO_RESOLUTION_MEGAPIXELS"],
                        label="Start-frame auto cap",
                        info=(
                            "Maximum automatic resolution from the first frame; "
                            "manual sizes are unchanged."
                        ),
                    )
                resolution_info = gr.Markdown(
                    services["resolution_summary"](
                        defaults["width"], defaults["height"]
                    )
                )
                with gr.Row():
                    seed = gr.Number(
                        value=defaults["seed"],
                        precision=0,
                        label="Seed",
                        info=(
                            "Used for a single video. Batch videos always use "
                            "independent random seeds."
                        ),
                    )
                    batch_count = gr.Slider(
                        services["MIN_VIDEO_BATCH_COUNT"],
                        services["MAX_VIDEO_BATCH_COUNT"],
                        value=services["DEFAULT_VIDEO_BATCH_COUNT"],
                        step=1,
                        label="Videos per batch",
                        info="Generate up to four random-seed variants in one run.",
                    )
            with performance_section:
                gr.Markdown(
                    "Tune Turbo, attention, and caching. Defaults are recommended for most jobs."
                )
                turbo_variant = gr.Radio(
                    list(services["TURBO_SETTINGS"]),
                    value=defaults["turbo_variant"],
                    label="Turbo implementation",
                    info=(
                        "All base profiles use the sharper runtime LoRA path. Quality's "
                        "fused INT8 FC2 projections use a post-dequantization weight-cast "
                        "exception. "
                        "Larry also uses its adaptive sampler; strength stays at 1.0."
                    ),
                )
                scheduler = gr.Radio(
                    ["simple", "beta", "normal"],
                    value=defaults["scheduler"],
                    label="Scheduler",
                    info="Controls how sampling steps are distributed across denoising.",
                )
                attention_mode = gr.Radio(
                    ["Sage 2", "Kitchen", "SLA", "Sol-Attn", "Auto"],
                    value=defaults["attention_mode"],
                    label="Attention",
                    interactive=services["SERVER_ATTENTION_BACKEND"] == "sol",
                    info=(
                        f"Auto enables Sol-Attn for Reference mode or when estimated "
                        f"packed target tokens reach {services['AUTO_SOL_TOKEN_THRESHOLD']:,}; "
                        "Sage 2 applies the pinned KJNodes model override. Kitchen "
                        "selects the global ComfyUI backend. SLA is the default, uses "
                        "the Balanced audio-safe block-sparse preset, and automatically "
                        "keeps short sequences dense; it is intended for SLA-distilled "
                        "H3 LoRAs. Auto uses Kitchen for "
                        "smaller jobs. Sol "
                        f"dense/fallback calls use {services['SERVER_DENSE_ATTENTION_BACKEND']}."
                    ),
                )
                with gr.Accordion("SLA quality controls", open=False):
                    sla_preset = gr.Radio(
                        list(services["SLA_PRESET_INPUTS"]),
                        value=defaults["sla_preset"],
                        label="SLA preset",
                        info=(
                            "Fast uses validated 0.90 sparsity. Balanced uses the "
                            "LoRA-distilled 0.85 sparsity. Quality also runs the final "
                            "sampling step dense to recover fine detail; in a two-stage "
                            "latent-upscale workflow this applies to both stages. All "
                            "presets use 64-token blocks and protect audio."
                        ),
                    )

                with gr.Row():
                    sol_tau = gr.Slider(
                        0.5,
                        1.5,
                        value=defaults["sol_tau"],
                        step=0.1,
                        label="Sol-Attn tau",
                    )
                    sol_thresh_type = gr.Radio(
                        ["diag", "exact"],
                        value=defaults["sol_thresh_type"],
                        label="Sol threshold",
                        info="diag is faster; exact calculates a more precise routing threshold.",
                    )
                with gr.Accordion("Zero-copy Sol-Attn quality controls", open=False):
                    sol_exact_mode = gr.Radio(
                        ["off", "exact_kv", "exact_kv_and_rows"],
                        value=defaults["sol_exact_mode"],
                        label="Exact H3 prefix mode",
                        info=(
                            "exact_kv preserves text/condition/reference/audio KV "
                            "rows at low cost. exact_kv_and_rows also keeps prefix "
                            "query rows dense for maximum audio/conditioning fidelity."
                        ),
                    )
                    with gr.Row():
                        sol_dense_steps = gr.Slider(
                            0,
                            4,
                            value=defaults["sol_dense_steps"],
                            step=1,
                            label="Dense final transformer blocks",
                            info=(
                                "Keep the final N H3 transformer blocks dense. "
                                "The final block is the most approximation-sensitive."
                            ),
                        )
                    sol_step_off = gr.State(0.0)
                    sol_sink_tokens = gr.State(0)
                with gr.Accordion("Sampling acceleration", open=False):
                    cache_mode = gr.Radio(
                        ["Spectrum", "FirstBlockCache", "EasyCache", "Off"],
                        value=defaults["cache_mode"],
                        label="Acceleration mode",
                        info=(
                            "Spectrum is the normal H3 default based on broader community "
                            "speed testing. It forecasts selected transformer steps and uses "
                            "audio-isolated offline replay. FirstBlockCache is the lower-memory "
                            "fallback. Modes are mutually exclusive. Turbo defaults to Spectrum; "
                            "EasyCache and FirstBlockCache are opt-in experimental Turbo options."
                        ),
                    )
                    fbcache_preset = gr.Radio(
                        ["Safe", "Fast", "Aggressive", "Custom"],
                        value=defaults["fbcache_preset"],
                        label="FirstBlockCache preset",
                        info=(
                            "Fast is the recommended default. Named presets use "
                            "a protected 10–95% denoising window and at most two "
                            "consecutive cache hits."
                        ),
                    )
                    with gr.Row():
                        fbcache_threshold = gr.Slider(
                            0.0,
                            0.25,
                            value=defaults["fbcache_threshold"],
                            step=0.005,
                            label="FirstBlock threshold",
                            interactive=False,
                        )
                        fbcache_max_hits = gr.Slider(
                            1,
                            8,
                            value=defaults["fbcache_max_hits"],
                            step=1,
                            label="Max consecutive cache hits",
                            interactive=False,
                        )
                    with gr.Row():
                        fbcache_start = gr.Slider(
                            0.0,
                            0.90,
                            value=defaults["fbcache_start"],
                            step=0.01,
                            label="Cache start percent",
                            interactive=False,
                        )
                        fbcache_end = gr.Slider(
                            0.10,
                            1.0,
                            value=defaults["fbcache_end"],
                            step=0.01,
                            label="Cache end percent",
                            interactive=False,
                        )
                    fbcache_temporal_guard = gr.Checkbox(
                        value=defaults["fbcache_temporal_guard"],
                        label="Temporal frame guard",
                        info=(
                            "Checks the most-changed target-video latent frame "
                            "in addition to the global residual average."
                        ),
                    )
                    gr.Markdown("**EasyCache fallback settings**")
                    easycache_threshold = gr.Slider(
                        0.0,
                        0.5,
                        value=defaults["easycache_threshold"],
                        step=0.01,
                        label="Reuse threshold",
                        info=(
                            "Higher skips more steps. Start at 0.10 for H3; "
                            "ComfyUI's generic default is 0.20."
                        ),
                    )
                    with gr.Row():
                        easycache_start = gr.Slider(
                            0.0,
                            0.9,
                            value=defaults["easycache_start"],
                            step=0.01,
                            label="Start percent",
                        )
                        easycache_end = gr.Slider(
                            0.1,
                            1.0,
                            value=defaults["easycache_end"],
                            step=0.01,
                            label="End percent",
                        )
                    easycache_verbose = gr.Checkbox(
                        value=defaults["easycache_verbose"],
                        label="Log EasyCache decisions",
                        info="Logs skipped-step counts and estimated speedup in ComfyUI.",
                    )

            with finishing_section:
                gr.Markdown(
                    "Increase resolution during sampling or run an optional finishing pass."
                )
                gr.Markdown("**Native latent upscale**")
                latent_upscale = gr.Checkbox(
                    value=defaults["latent_upscale"],
                    label="Generate at half resolution, then latent upscale 2x",
                    info=(
                        "Runs inside H3 sampling, not after video generation. Width and "
                        "height remain the final output resolution. Disabled by default."
                    ),
                )
                with gr.Group(
                    visible=defaults["latent_upscale"]
                ) as latent_upscale_settings:
                    latent_upscaler_model = gr.Dropdown(
                        choices=list(services["H3_LATENT_UPSCALER_MODEL_CHOICES"]),
                        value=defaults["latent_upscaler_model"],
                        label="Latent upscaler model",
                        info=(
                            "Balanced uses BF16 and is the default. Fast uses FP16; "
                            "Quality uses FP32 and needs more memory. Downloaded on first use."
                        ),
                    )
                    latent_upscale_refine_steps = gr.Slider(
                        1,
                        6,
                        value=defaults["latent_upscale_refine_steps"],
                        step=1,
                        label="High-resolution refinement steps",
                        info=(
                            "H3 first finishes all generation steps at half resolution. "
                            "The clean 2x latent is then lightly re-noised and refined. "
                            "Two expensive high-resolution steps is the default."
                        ),
                    )
                    latent_upscale_method = gr.Dropdown(
                        choices=list(services["H3_LATENT_UPSCALE_METHODS"]),
                        value=defaults["latent_upscale_method"],
                        label="High-resolution refinement method",
                        info=(
                            "Full-frame is the normal path. MMH3 Split Upscale "
                            "re-samples temporal chunks and spatial tiles for jobs "
                            "that cannot fit a full target-resolution pass."
                        ),
                    )
                    with gr.Group(
                        visible=(
                            defaults["latent_upscale_method"]
                            == services["H3_LATENT_UPSCALE_SPLIT"]
                        )
                    ) as latent_split_settings:
                        gr.Markdown(
                            "**Experimental MMH3 Split Upscale** · Trades additional "
                            "sampling work for lower target-resolution VRAM pressure. "
                            "Fixed seeds are recommended when comparing settings."
                        )
                        with gr.Row():
                            latent_split_tile_width = gr.Slider(
                                256,
                                2048,
                                value=defaults["latent_split_tile_width"],
                                step=32,
                                label="Tile width (pixels)",
                            )
                            latent_split_tile_height = gr.Slider(
                                256,
                                2048,
                                value=defaults["latent_split_tile_height"],
                                step=32,
                                label="Tile height (pixels)",
                            )
                        with gr.Row():
                            latent_split_overlap_ratio = gr.Slider(
                                0.0,
                                0.90,
                                value=defaults["latent_split_overlap_ratio"],
                                step=0.05,
                                label="Spatial overlap",
                                info="Larger overlap reduces seams but repeats more work.",
                            )
                            latent_split_fade_ratio = gr.Slider(
                                0.0,
                                1.0,
                                value=defaults["latent_split_fade_ratio"],
                                step=0.05,
                                label="Overlap fade",
                                info="Controls how much of each overlap is cross-faded.",
                            )
                        with gr.Row():
                            latent_split_chunk_frames = gr.Slider(
                                5,
                                1000,
                                value=defaults["latent_split_chunk_frames"],
                                step=1,
                                label="Temporal chunk length (frames)",
                                info="Upstream snaps this to H3's native temporal grid.",
                            )
                            latent_split_temporal_overlap_frames = gr.Slider(
                                0,
                                240,
                                value=defaults["latent_split_temporal_overlap_frames"],
                                step=1,
                                label="Temporal overlap (frames)",
                            )
                        with gr.Row():
                            latent_split_seam_denoise = gr.Slider(
                                0.1,
                                1.0,
                                value=defaults["latent_split_seam_denoise"],
                                step=0.05,
                                label="Seam denoise cap",
                                info=(
                                    "0.5–0.8 can reduce motion breaks at tile seams; "
                                    "1.0 disables the cap."
                                ),
                            )
                            latent_split_seam_polish = gr.Dropdown(
                                choices=["off", "auto", "all"],
                                value=defaults["latent_split_seam_polish"],
                                label="Seam polish",
                                info=(
                                    "Auto re-samples only seams that fail the upstream "
                                    "probe. All is the slowest option."
                                ),
                            )
                    gr.Markdown(
                        "Final width and height must both be divisible by 64. For example, "
                        "1024×1024 generates the first stage at 512×512 and finishes at "
                        "1024×1024. Only the video latent is upscaled; H3 audio is preserved."
                    )

                gr.Markdown("**After generation**")
                generation_postprocess = gr.Dropdown(
                    choices=services["GENERATION_POSTPROCESS_OPTIONS"],
                    value=defaults["postprocess"],
                    label="After generation",
                    info=(
                        "Optionally run SeedVR2 or LTX-2.5 2x immediately after the base H3 "
                        "video finishes. The source video remains in the gallery."
                    ),
                )
                with gr.Group(visible=False) as generation_postprocess_settings:
                    generation_upscale_resolution = gr.Dropdown(
                        choices=list(services["UPSCALE_RESOLUTION_PRESETS"]),
                        value=services["DEFAULT_UPSCALE_RESOLUTION"],
                        label="Output resolution",
                        info="Fits the source inside the selected square while preserving aspect ratio.",
                    )
                    generation_seedvr2_model = gr.Dropdown(
                        choices=list(services["SEEDVR2_MODEL_CHOICES"]),
                        value=defaults["seedvr2_model"],
                        label="SeedVR2 model",
                        info=(
                            "Downloaded on first use. 7B Sharp favors stronger "
                            "detail; NVFP4 variants are optimized for Blackwell GPUs."
                        ),
                    )
                    generation_ltx25_note = gr.Markdown(
                        "Uses the transformer selected in the **LTX 2.5** tab and "
                        "the H3 generation prompt. The gated 2x IC-LoRA downloads "
                        "on first use.",
                        visible=False,
                    )
                    generation_force_offload = gr.Checkbox(
                        value=defaults["upscale_force_offload"],
                        label="Unload H3 models before upscaling",
                        info=(
                            "Reduces peak VRAM at the cost of reloading H3 for "
                            "the next generation."
                        ),
                    )
                    generation_split_upscale = gr.Checkbox(
                        value=defaults["upscale_split_enabled"],
                        label="Split source into clips before LTX upscaling",
                        info=(
                            "Opt in after an out-of-VRAM error. Each clip is "
                            "upscaled independently and concatenated afterward."
                        ),
                        visible=False,
                    )
                    generation_split_seconds = gr.Slider(
                        1.0,
                        15.0,
                        value=defaults["upscale_split_seconds"],
                        step=0.5,
                        label="Target clip length (seconds)",
                        info=(
                            "5 seconds is the recommended starting point. The "
                            "actual cut is adjusted to an LTX-valid frame count."
                        ),
                        visible=False,
                    )

    return H3View(
        {
            "attention_mode": attention_mode,
            "audio_output": audio_output,
            "auto_megapixels": auto_megapixels,
            "batch_count": batch_count,
            "cache_mode": cache_mode,
            "draft_resolution": draft_resolution,
            "duration": duration,
            "easycache_end": easycache_end,
            "easycache_start": easycache_start,
            "easycache_threshold": easycache_threshold,
            "easycache_verbose": easycache_verbose,
            "enhance_prompt_button": enhance_prompt_button,
            "enhance_prompt_status": enhance_prompt_status,
            "fast_resolution": fast_resolution,
            "fbcache_end": fbcache_end,
            "fbcache_max_hits": fbcache_max_hits,
            "fbcache_preset": fbcache_preset,
            "fbcache_start": fbcache_start,
            "fbcache_temporal_guard": fbcache_temporal_guard,
            "fbcache_threshold": fbcache_threshold,
            "first": first,
            "frame_group": frame_group,
            "gemini_api_key": gemini_api_key,
            "gemini_prompt_model": gemini_prompt_model,
            "gemini_prompt_writer_group": gemini_prompt_writer_group,
            "generation_force_offload": generation_force_offload,
            "generation_ltx25_note": generation_ltx25_note,
            "generation_mode": generation_mode,
            "generation_postprocess": generation_postprocess,
            "generation_postprocess_settings": generation_postprocess_settings,
            "generation_readiness": generation_readiness,
            "generation_seedvr2_model": generation_seedvr2_model,
            "generation_split_seconds": generation_split_seconds,
            "generation_split_upscale": generation_split_upscale,
            "generation_upscale_resolution": generation_upscale_resolution,
            "height": height,
            "help_text": help_text,
            "image_clear_selection": image_clear_selection,
            "image_frame_paths": image_frame_paths,
            "image_frames": image_frames,
            "image_output": image_output,
            "image_output_group": image_output_group,
            "image_save_selected": image_save_selected,
            "image_save_status": image_save_status,
            "image_saved_files": image_saved_files,
            "image_select_all": image_select_all,
            "image_selection": image_selection,
            "image_vae": image_vae,
            "input_upscale_downloads": input_upscale_downloads,
            "input_upscale_force_offload": input_upscale_force_offload,
            "input_upscale_frame_height": input_upscale_frame_height,
            "input_upscale_frame_preset": input_upscale_frame_preset,
            "input_upscale_frame_width": input_upscale_frame_width,
            "input_upscale_model": input_upscale_model,
            "input_upscale_run": input_upscale_run,
            "input_upscale_seed": input_upscale_seed,
            "input_upscale_slots": input_upscale_slots,
            "input_upscale_status": input_upscale_status,
            "large_resolution": large_resolution,
            "last": last,
            "latent_split_chunk_frames": latent_split_chunk_frames,
            "latent_split_fade_ratio": latent_split_fade_ratio,
            "latent_split_overlap_ratio": latent_split_overlap_ratio,
            "latent_split_seam_denoise": latent_split_seam_denoise,
            "latent_split_seam_polish": latent_split_seam_polish,
            "latent_split_settings": latent_split_settings,
            "latent_split_temporal_overlap_frames": (
                latent_split_temporal_overlap_frames
            ),
            "latent_split_tile_height": latent_split_tile_height,
            "latent_split_tile_width": latent_split_tile_width,
            "latent_upscale": latent_upscale,
            "latent_upscale_method": latent_upscale_method,
            "latent_upscale_refine_steps": latent_upscale_refine_steps,
            "latent_upscale_settings": latent_upscale_settings,
            "latent_upscaler_model": latent_upscaler_model,
            "lightning_api_key": lightning_api_key,
            "lightning_prompt_writer_group": lightning_prompt_writer_group,
            "local_prompt_base_model": local_prompt_base_model,
            "local_prompt_greedy": local_prompt_greedy,
            "local_prompt_max_tokens": local_prompt_max_tokens,
            "local_prompt_seed": local_prompt_seed,
            "local_prompt_temperature": local_prompt_temperature,
            "local_prompt_top_p": local_prompt_top_p,
            "local_prompt_writer_group": local_prompt_writer_group,
            "mode": mode,
            "model_profile": model_profile,
            "output": output,
            "output_2": output_2,
            "output_3": output_3,
            "output_4": output_4,
            "preset": preset,
            "prompt": prompt,
            "prompt_writer_backend": prompt_writer_backend,
            "ref_audio_1": ref_audio_1,
            "ref_audio_2": ref_audio_2,
            "ref_audio_3": ref_audio_3,
            "ref_image_1": ref_image_1,
            "ref_image_2": ref_image_2,
            "ref_image_3": ref_image_3,
            "ref_image_4": ref_image_4,
            "ref_image_5": ref_image_5,
            "ref_image_6": ref_image_6,
            "ref_image_7": ref_image_7,
            "ref_image_8": ref_image_8,
            "ref_image_9": ref_image_9,
            "ref_size": ref_size,
            "ref_video_1": ref_video_1,
            "ref_video_2": ref_video_2,
            "ref_video_3": ref_video_3,
            "reference_group": reference_group,
            "refresh": refresh,
            "resolution_info": resolution_info,
            "result_format": result_format,
            "reuse_unchanged_inputs": reuse_unchanged_inputs,
            "run": run,
            "scheduler": scheduler,
            "seed": seed,
            "settings_overview": settings_overview,
            "sla_preset": sla_preset,
            "sol_dense_steps": sol_dense_steps,
            "sol_exact_mode": sol_exact_mode,
            "sol_sink_tokens": sol_sink_tokens,
            "sol_step_off": sol_step_off,
            "sol_tau": sol_tau,
            "sol_thresh_type": sol_thresh_type,
            "stage_model_offload": stage_model_offload,
            "status": status,
            "steps": steps,
            "stop": stop,
            "text_encoder": text_encoder,
            "turbo_variant": turbo_variant,
            "trt_vae_compile": trt_vae_compile,
            "use_int8_vae": use_int8_vae,
            "use_trt_vae": use_trt_vae,
            "width": width,
        }
    )
