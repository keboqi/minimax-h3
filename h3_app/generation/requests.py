"""Named generation inputs; positional compatibility belongs to the UI adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping


@dataclass(frozen=True)
class Flux3Request:
    prompt: str
    model_choice: str = "NVFP4 (Blackwell)"
    reference_images: tuple[str, ...] = ()
    width: int = 1024
    height: int = 1024
    edit_size: str = "Match first image size"
    seed: int = -1
    steps: int = 16
    expand_prompt: bool = True
    upsampler_choice: str = "INT8 ConvRot"
    cfg: float | None = None
    layout: str = ""


@dataclass(frozen=True)
class MossImageRequest:
    prompt: str
    preset: str = "Fast"
    turbo_variant: str = "Lodestones DMD (8-step)"
    model_choice: str = "INT8 ConvRot (lower VRAM)"
    text_encoder_choice: str = "INT8 ConvRot (recommended)"
    vae_choice: str = "Small FLUX.2 VAE (faster decode)"
    width: int = 1024
    height: int = 1024
    seed: int = -1
    steps: int = 8
    cfg: float = 1.0
    layout: str = ""
    accelerate: bool = True
    reference_images: tuple[str, ...] = ()
    edit_size: str = "Match first image size"


@dataclass(frozen=True)
class Fibo2Request:
    prompt: str
    model_choice: str = "Dev"
    precision: str = "BF16"
    reference_images: tuple[str, ...] = ()
    width: int = 1024
    height: int = 1024
    edit_size: str = "Trained size (~1 MP, recommended)"
    seed: int = -1
    steps: int = 30
    cfg: float = 5.0


@dataclass(frozen=True)
class Drama3Request:
    text: str
    # (speaker number, audio path, exact transcript) per cloned voice.
    voices: tuple[tuple[int, str, str], ...] = ()
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 0
    max_frames: int = 8192
    # Speaker chunks generated in parallel on SGLang-Omni.
    parallel: int = 16
    retries: int = 0
    # (speaker number, saved speaker name or "") for status and provenance.
    voice_names: tuple[tuple[int, str], ...] = ()
    script_mode: str = "Basic"
    # (sound name, audio path or "", description, duration_seconds)
    sounds: tuple[tuple[str, str, str, int], ...] = ()
    # Saved-take keys to generate again; remix_only rebuilds the mix from saved takes alone.
    regenerate: tuple[str, ...] = ()
    remix_only: bool = False
    # Time every word with Qwen3's aligner for word-by-word subtitles.
    word_timing: bool = False
    # Names the output file; "" keeps the default name.
    title: str = ""



@dataclass
class H3MediaInputs:
    mode: str
    prompt: str
    first_image: str | None
    last_image: str | None
    ref_image_1: Any
    ref_image_2: Any
    ref_image_3: Any
    ref_image_4: Any
    ref_image_5: Any
    ref_image_6: Any
    ref_image_7: Any
    ref_image_8: Any
    ref_image_9: Any
    ref_video_1: Any
    ref_video_2: Any
    ref_video_3: Any
    ref_audio_1: Any
    ref_audio_2: Any
    ref_audio_3: Any
    ref_image_size: str
    reuse_unchanged_inputs: bool
    fl2va_audio_1: Any
    fl2va_audio_2: Any
    fl2va_audio_3: Any


@dataclass
class H3SamplingInputs:
    model_profile: str
    text_encoder: str
    stage_model_offload: bool
    generation_mode: str
    turbo_variant: str
    steps: int
    scheduler: str
    attention_mode: str
    sla_preset: str
    sol_tau: float
    sol_thresh_type: str
    sol_exact_mode: str
    sol_dense_steps: int
    sol_step_off: float
    sol_sink_tokens: int
    cache_mode: str
    fbcache_preset: str
    fbcache_threshold: float
    fbcache_start: float
    fbcache_end: float
    fbcache_max_hits: int
    fbcache_temporal_guard: bool
    easycache_threshold: float
    easycache_start: float
    easycache_end: float
    easycache_verbose: bool
    semantic_bridge: bool
    semantic_bridge_alpha: float
    encoder_small_input: bool = False


@dataclass
class H3OutputInputs:
    duration: float
    width: int
    height: int
    seed: int
    use_int8_vae: bool
    use_trt_vae: bool
    image_vae: str
    result_format: str
    image_frames: int
    use_lynnreal_vae: bool = False
    use_sol_vae: bool = False
    video_encoder: str = "Original"
    lightvae_decoder: str = "none"


@dataclass
class H3FinishingInputs:
    postprocess: str
    latent_upscale: bool
    latent_upscaler_model: str
    latent_upscale_refine_steps: int
    latent_upscale_method: str
    upscale_force_offload: bool
    upscale_resolution: str
    seedvr2_model: str
    ltx25_model: str
    latent_upscale_refine_lora: str = "Same as generation"


@dataclass
class H3Request:
    media: H3MediaInputs
    sampling: H3SamplingInputs
    output: H3OutputInputs
    finishing: H3FinishingInputs

    @classmethod
    def from_values(cls, values: Mapping[str, Any]) -> H3Request:
        return cls(
            media=H3MediaInputs(
                **{key: values[key] for key in H3MediaInputs.__dataclass_fields__}
            ),
            sampling=H3SamplingInputs(
                **{key: values[key] for key in H3SamplingInputs.__dataclass_fields__ if key in values}
            ),
            output=H3OutputInputs(
                **{key: values[key] for key in H3OutputInputs.__dataclass_fields__ if key in values}
            ),
            finishing=H3FinishingInputs(
                **{key: values[key] for key in H3FinishingInputs.__dataclass_fields__ if key in values}
            ),
        )

    def values(self) -> dict[str, Any]:
        return {
            key: value
            for group in asdict(self).values()
            for key, value in group.items()
        }

    def copy(self) -> H3Request:
        return replace(
            self,
            media=replace(self.media),
            sampling=replace(self.sampling),
            output=replace(self.output),
            finishing=replace(self.finishing),
        )


@dataclass(frozen=True)
class LtxRequest:
    mode: str
    model_choice: str
    prompt: str
    negative_prompt: str
    first_image: str | None
    duration: float
    fps: float
    width: int
    height: int
    seed: int
    cfg: float
    sampler_name: str
    image_strength: float
    middle_image: str | None
    middle_time: float
    middle_strength: float
    end_image: str | None
    end_strength: float
    reference_images: tuple[str, ...] = ()


@dataclass(frozen=True)
class MusicRequest:
    model_choice: str
    caption: str
    lyrics: str
    max_duration: float
    seed: int
    steps: int
    cfg: float
    ar_cfg: float
    top_k: int
    tiled_decode: bool


@dataclass(frozen=True)
class QwenImage21Request:
    mode: str
    model_choice: str
    text_encoder_choice: str
    prompt: str
    negative_prompt: str
    reference_images: tuple[str, ...]
    width: int
    height: int
    reference_resolution: int
    match_input_size: bool
    seed: int
    steps: int
    cfg: float
    sampler_name: str
    scheduler: str
    cache_device: str
    cache_dtype: str
    attention_backend: str
    batch_count: int = 1
    batch_edit_inputs: bool = False
    accelerator: str = "Off"
    turbo_variant: str = "Off"
    max_resolution: bool = False
    prompt_enhancer: str = "Off"
    lightning_strength: float = 0.5


@dataclass(frozen=True)
class YuE2Request:
    model_choice: str
    style: str
    lyrics: str
    abc: str
    mode: str
    max_duration: float
    seed: int
    steps: int
    cfg: float
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    max_abc_tokens: int
    abc_temperature: float
    abc_top_p: float
    abc_top_k: int
    abc_repetition_penalty: float
    abc_penalty_window: int
    tiled_decode: bool
