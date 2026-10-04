"""Build the model section in its existing parent container."""

from __future__ import annotations
from dataclasses import dataclass
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Mapping
import gradio as gr
from h3_app.decoder_intent import VideoDecoder

if TYPE_CHECKING:
    from ..h3_view import H3ViewServices


@dataclass(frozen=True)
class ModelSection:
    video_decoder: gr.components.Component
    generation_mode: gr.components.Component
    image_vae: gr.components.Component
    mode: gr.components.Component
    model_profile: gr.components.Component
    result_format: gr.components.Component
    reuse_unchanged_inputs: gr.components.Component
    semantic_bridge: gr.components.Component
    semantic_bridge_alpha: gr.components.Component
    stage_model_offload: gr.components.Component
    text_encoder: gr.components.Component
    encoder_small_input: gr.components.Component
    trt_vae_compile: gr.components.Component
    use_int8_vae: gr.components.Component
    use_lynnreal_vae: gr.components.Component
    use_trt_vae: gr.components.Component


def build_model_section(
    defaults: Mapping[str, Any],
    services: H3ViewServices,
    *,
    advanced_parent=None,
    conditioning_parent=None,
) -> ModelSection:
    with (
        conditioning_parent if conditioning_parent is not None else nullcontext(),
        gr.Row(elem_classes=["h3-mode-row"]),
    ):
        mode = gr.Radio(
            ["Text to video", "First / last frame", "Reference media"],
            value=defaults["mode"],
            label="Conditioning mode",
        )
        result_format = gr.Radio(
            services.RESULT_FORMATS,
            value=defaults["result_format"],
            label="Result format",
            info="H3 always samples vision and audio; this selects what is decoded and shown.",
            visible=False,
        )
    with advanced_parent if advanced_parent is not None else nullcontext(), gr.Row():
        model_profile = gr.Radio(
            services.MODEL_PROFILE_CHOICES,
            value=defaults["model_profile"],
            label="Base model",
            info="Speed uses the rebuilt single-pass NVFP4 files. Quality uses the mixed NVFP4/FP8/INT8 ConvRot files. Original uses the official BF16 files. Singularity uses the fine-tuned pruned v1.3 INT8 checkpoint. Speed, Original and Singularity download when first selected.",
        )
    with (
        advanced_parent if advanced_parent is not None else nullcontext(),
        gr.Accordion("Model & generation (advanced)", open=advanced_parent is not None),
        gr.Row(),
    ):
        generation_mode = gr.Radio(
            ["Normal", "Turbo"],
            value=defaults["generation_mode"],
            label="Generation",
            info="Turbo uses the implementation selected in Performance & sampling. LightX2V uses the matching reference adapter; Larry reference mode is experimental.",
        )
    with (
        advanced_parent if advanced_parent is not None else nullcontext(),
        gr.Accordion(
            "Model and memory (advanced)",
            open=advanced_parent is not None,
            elem_classes=["h3-advanced-block"],
        ),
    ):
        with gr.Row():
            text_encoder = gr.Dropdown(
                choices=list(services.H3_TEXT_ENCODER_CHOICES),
                value=defaults["text_encoder"],
                label="Text encoder",
                info="BF16 is approximately 51.5 GB. NVFP4/AWQ and INT8 ConvRot download on first use.",
            )
            stage_model_offload = gr.Checkbox(
                value=defaults["stage_model_offload"],
                interactive=defaults["text_encoder"] != "BF16",
                label="Offload models between H3 stages",
                info="Unload resident models between text encoding, diffusion, latent upscaling, and VAE decoding.",
            )
            reuse_unchanged_inputs = gr.Checkbox(
                value=defaults["reuse_unchanged_inputs"],
                label="Reuse unchanged prompt and media",
                info="Reuse matching prompt/media encoding and unchanged workflow nodes. Off forces fresh conditioning at every stage, including refinement. Sampling still reruns when the seed changes.",
            )
        encoder_small_input = gr.Checkbox(
            value=defaults["encoder_small_input"],
            label="Qwen small input attention",
            info="On: upstream PyTorch/basic attention. Off: the server attention backend (Kitchen in this app). Applies to Qwen text and vision encoding. Changing this rebuilds conditioning on the next generation. The diffusion Sage 2 selection does not change the encoder backend.",
        )
        semantic_bridge = gr.Checkbox(
            value=defaults["semantic_bridge"],
            label="Semantic Bridge (experimental)",
            interactive=defaults["mode"] != "Reference media",
            info="May improve prompt adherence. Downloads an 11 MB adapter on first use. FL2VA only; disabled for reference media.",
        )
        semantic_bridge_alpha = gr.Slider(
            0.0,
            1.0,
            step=0.01,
            value=defaults["semantic_bridge_alpha"],
            visible=defaults["semantic_bridge"]
            and defaults["mode"] != "Reference media",
            label="Semantic Bridge strength",
            info="Start at 0.10; try 0.15 for a stronger effect. Higher values can reduce quality. Uses per-token magnitude matching.",
        )
        video_decoder = gr.Dropdown(
            choices=[item.value for item in VideoDecoder],
            value=VideoDecoder.from_flags(
                defaults["use_int8_vae"],
                defaults["use_lynnreal_vae"],
                defaults["use_trt_vae"],
            ).value,
            label="Video decoder",
            visible=True,
            info="One decoder per run. Image decoding is configured separately. TensorRT prepares a local engine when needed.",
        )
        use_int8_vae = gr.Checkbox(
            visible=False,
            value=defaults["use_int8_vae"],
            label="INT8 ConvRot video VAE",
            info="On for Balanced and Quality presets. Downloads the official Comfy-Org checkpoint on first use for faster H3 video decoding.",
        )
        use_lynnreal_vae = gr.Checkbox(
            visible=False,
            value=defaults["use_lynnreal_vae"],
            label="LynnReal Light INT8 video VAE",
            info="On for Fast and Singularity presets. Experimental distilled decoder; downloads 2.14 GB on first use and may change fine detail.",
        )
        with gr.Row():
            use_trt_vae = gr.Checkbox(
                visible=False,
                value=defaults["use_trt_vae"],
                label="Experimental TensorRT video VAE",
                info="Default off. Uses a local TensorRT engine for final H3 video decoding and compiles it automatically when needed.",
                scale=2,
            )
            trt_vae_compile = gr.Button("Compile TensorRT VAE engine", scale=1)
        image_vae = gr.Radio(
            services.IMAGE_VAE_CHOICES,
            value=defaults["image_vae"],
            label="Image VAE",
            visible=False,
            info="Official is the default and remains the only video decoder. The experimental 500K option downloads 9.69 GB on first use and decodes one image from temporal latent slice 0.",
        )
    return ModelSection(
        video_decoder=video_decoder,
        generation_mode=generation_mode,
        image_vae=image_vae,
        mode=mode,
        model_profile=model_profile,
        result_format=result_format,
        reuse_unchanged_inputs=reuse_unchanged_inputs,
        semantic_bridge=semantic_bridge,
        semantic_bridge_alpha=semantic_bridge_alpha,
        stage_model_offload=stage_model_offload,
        text_encoder=text_encoder,
        encoder_small_input=encoder_small_input,
        trt_vae_compile=trt_vae_compile,
        use_int8_vae=use_int8_vae,
        use_lynnreal_vae=use_lynnreal_vae,
        use_trt_vae=use_trt_vae,
    )
