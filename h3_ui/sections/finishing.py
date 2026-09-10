"""Build the finishing section in its existing parent container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import gradio as gr

if TYPE_CHECKING:
    from ..h3_view import H3ViewServices


@dataclass(frozen=True)
class FinishingSection:
    generation_force_offload: gr.components.Component
    generation_ltx25_note: gr.components.Component
    generation_postprocess: gr.components.Component
    generation_postprocess_settings: gr.components.Component
    generation_seedvr2_model: gr.components.Component
    generation_split_seconds: gr.components.Component
    generation_split_upscale: gr.components.Component
    generation_upscale_resolution: gr.components.Component
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


def build_finishing_section(
    defaults: Mapping[str, Any],
    services: H3ViewServices,
    finishing_section: gr.blocks.BlockContext,
) -> FinishingSection:
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
                "height remain the H3 output resolution. Acceleration is disabled while this is enabled."
            ),
        )
        with gr.Group(visible=defaults["latent_upscale"]) as latent_upscale_settings:
            latent_upscaler_model = gr.Dropdown(
                choices=list(services.H3_LATENT_UPSCALER_MODEL_CHOICES),
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
                choices=list(services.H3_LATENT_UPSCALE_METHODS),
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
                    == services.H3_LATENT_UPSCALE_SPLIT
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
            choices=services.GENERATION_POSTPROCESS_OPTIONS,
            value=defaults["postprocess"],
            label="After generation",
            info=(
                "Optionally run SeedVR2 or LTX-2.5 2x immediately after the base H3 "
                "video finishes. The source video remains in the gallery."
            ),
        )
        with gr.Group(visible=False) as generation_postprocess_settings:
            generation_upscale_resolution = gr.Dropdown(
                choices=list(services.UPSCALE_RESOLUTION_PRESETS),
                value=services.DEFAULT_UPSCALE_RESOLUTION,
                label="Output resolution",
                info="Fits the source inside the selected square while preserving aspect ratio.",
            )
            generation_seedvr2_model = gr.Dropdown(
                choices=list(services.SEEDVR2_MODEL_CHOICES),
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

    return FinishingSection(
        generation_force_offload=generation_force_offload,
        generation_ltx25_note=generation_ltx25_note,
        generation_postprocess=generation_postprocess,
        generation_postprocess_settings=generation_postprocess_settings,
        generation_seedvr2_model=generation_seedvr2_model,
        generation_split_seconds=generation_split_seconds,
        generation_split_upscale=generation_split_upscale,
        generation_upscale_resolution=generation_upscale_resolution,
        latent_split_chunk_frames=latent_split_chunk_frames,
        latent_split_fade_ratio=latent_split_fade_ratio,
        latent_split_overlap_ratio=latent_split_overlap_ratio,
        latent_split_seam_denoise=latent_split_seam_denoise,
        latent_split_seam_polish=latent_split_seam_polish,
        latent_split_settings=latent_split_settings,
        latent_split_temporal_overlap_frames=latent_split_temporal_overlap_frames,
        latent_split_tile_height=latent_split_tile_height,
        latent_split_tile_width=latent_split_tile_width,
        latent_upscale=latent_upscale,
        latent_upscale_method=latent_upscale_method,
        latent_upscale_refine_steps=latent_upscale_refine_steps,
        latent_upscale_settings=latent_upscale_settings,
        latent_upscaler_model=latent_upscaler_model,
    )
