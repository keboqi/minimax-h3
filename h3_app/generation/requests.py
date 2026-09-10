"""Named generation inputs; positional compatibility belongs to the UI adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping


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


@dataclass
class H3FinishingInputs:
    postprocess: str
    latent_upscale: bool
    latent_upscaler_model: str
    latent_upscale_refine_steps: int
    latent_upscale_method: str
    latent_split_tile_width: int
    latent_split_tile_height: int
    latent_split_overlap_ratio: float
    latent_split_fade_ratio: float
    latent_split_chunk_frames: int
    latent_split_temporal_overlap_frames: int
    latent_split_seam_denoise: float
    latent_split_seam_polish: str
    upscale_force_offload: bool
    upscale_split_enabled: bool
    upscale_split_seconds: float
    upscale_resolution: str
    seedvr2_model: str
    ltx25_model: str


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
                **{key: values[key] for key in H3SamplingInputs.__dataclass_fields__}
            ),
            output=H3OutputInputs(
                **{key: values[key] for key in H3OutputInputs.__dataclass_fields__}
            ),
            finishing=H3FinishingInputs(
                **{key: values[key] for key in H3FinishingInputs.__dataclass_fields__}
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
