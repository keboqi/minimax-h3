"""Typed effects required by the generation orchestrators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol

from h3_app.media_types import UpscaleClipBatch, VideoMetadata
from h3_app.model_types import ModelConfig, ModelProfile
from h3_app.policy import H3SplitUpscaleConfig
from h3_app.progress import ProgressUpdate
from h3_app.settings import ResolvedSettings


class BuildFl2VaGraph(Protocol):
    def __call__(
        self,
        *,
        prompt: str,
        first_image: str | None,
        last_image: str | None,
        width: int,
        height: int,
        duration: float,
        steps: int,
        seed: int,
        scheduler: str,
        turbo_lora_name: str | None,
        turbo_variant: str,
        turbo_strength: float,
        use_sol: bool,
        sol_tau: float,
        sol_thresh_type: str,
        sol_exact_mode: str,
        sol_dense_steps: int,
        sol_step_off: float,
        sol_sink_tokens: int,
        cache_mode: str,
        fbcache_preset: str,
        fbcache_threshold: float,
        fbcache_start: float,
        fbcache_end: float,
        fbcache_max_hits: int,
        fbcache_temporal_guard: bool,
        easycache_threshold: float,
        easycache_start: float,
        easycache_end: float,
        easycache_verbose: bool,
        model_name: str,
        models: ModelConfig,
        available_nodes: set[str],
        use_int8_vae: bool = ...,
        use_trt_vae: bool = ...,
        use_sage: bool = ...,
        use_sla: bool = ...,
        sla_preset: str = ...,
        latent_upscale_model_name: str | None = ...,
        latent_upscale_precision: str = ...,
        latent_upscale_refine_steps: int = ...,
        latent_split_config: H3SplitUpscaleConfig | None = ...,
        result_format: str = ...,
        image_frames: int = ...,
        image_vae: str = ...,
        text_encoder_name: str | None = ...,
        encoder_small_input: bool = ...,
        reuse_unchanged_inputs: bool = ...,
        stage_model_offload: bool = ...,
        smart_stage_offload: bool = ...,
        semantic_bridge: bool = ...,
        semantic_bridge_alpha: float = ...,
        voice_reference_audios: list[str] | None = ...,
    ) -> dict[str, Any]: ...


class BuildLtx25Graph(Protocol):
    def __call__(
        self,
        *,
        model_choice: str = ...,
        prompt: str,
        negative_prompt: str,
        first_image: str | None,
        width: int,
        height: int,
        duration: float,
        fps: float,
        seed: int,
        cfg: float,
        sampler_name: str,
        image_strength: float,
        middle_image: str | None = ...,
        middle_time: float = ...,
        middle_strength: float = ...,
        end_image: str | None = ...,
        end_strength: float = ...,
    ) -> dict[str, Any]: ...


class BuildRef2VaGraph(Protocol):
    def __call__(
        self,
        *,
        prompt: str,
        reference_images: list[str],
        reference_videos: list[str],
        reference_audios: list[str],
        width: int,
        height: int,
        duration: float,
        steps: int,
        seed: int,
        scheduler: str,
        ref_image_size: str,
        turbo_lora_name: str | None,
        turbo_variant: str,
        turbo_strength: float,
        use_sol: bool,
        sol_tau: float,
        sol_thresh_type: str,
        sol_exact_mode: str,
        sol_dense_steps: int,
        sol_step_off: float,
        sol_sink_tokens: int,
        cache_mode: str,
        fbcache_preset: str,
        fbcache_threshold: float,
        fbcache_start: float,
        fbcache_end: float,
        fbcache_max_hits: int,
        fbcache_temporal_guard: bool,
        easycache_threshold: float,
        easycache_start: float,
        easycache_end: float,
        easycache_verbose: bool,
        model_name: str,
        models: ModelConfig,
        available_nodes: set[str],
        use_int8_vae: bool = ...,
        use_trt_vae: bool = ...,
        use_sage: bool = ...,
        use_sla: bool = ...,
        sla_preset: str = ...,
        latent_upscale_model_name: str | None = ...,
        latent_upscale_precision: str = ...,
        latent_upscale_refine_steps: int = ...,
        latent_split_config: H3SplitUpscaleConfig | None = ...,
        result_format: str = ...,
        image_frames: int = ...,
        image_vae: str = ...,
        text_encoder_name: str | None = ...,
        encoder_small_input: bool = ...,
        smart_stage_offload: bool = ...,
        reuse_unchanged_inputs: bool = ...,
        stage_model_offload: bool = ...,
    ) -> dict[str, Any]: ...


class BuildUpscaleGraph(Protocol):
    def __call__(
        self,
        *,
        option: str,
        source_video: str,
        seed: int,
        models: ModelConfig,
        seedvr2_model: str = ...,
        ltx25_model: str = ...,
        prompt: str = ...,
        width: int | None = ...,
        height: int | None = ...,
        target_width: int | None = ...,
        target_height: int | None = ...,
        fps: float = ...,
    ) -> tuple[dict[str, Any], int]: ...


class CleanupUpscaleClipBatch(Protocol):
    def __call__(
        self, batch: UpscaleClipBatch | None, outputs: Iterable[Path] = ...
    ) -> None: ...


class ConcatUpscaledClips(Protocol):
    def __call__(
        self,
        source: Path,
        clips: list[Path],
        *,
        option: str,
        duration: float,
        frame_count: int,
    ) -> Path: ...


class PostprocessSwiftvrVideo(Protocol):
    def __call__(
        self, source: Path, *, fps: float, target_width: int, target_height: int
    ) -> Path: ...


class PostprocessVideo(Protocol):
    def __call__(self, source: Path, option: str) -> Path: ...


class PrepareUpscaleClipBatch(Protocol):
    def __call__(
        self,
        source: Path,
        *,
        category: str,
        split_enabled: bool,
        split_seconds: float,
        metadata: VideoMetadata,
    ) -> UpscaleClipBatch: ...


class ProbeVideoMetadata(Protocol):
    def __call__(self, source: Path) -> VideoMetadata: ...


class ResolveAudioOutput(Protocol):
    def __call__(self, history: dict[str, Any], queued_at: float) -> Path: ...


class ResolveImageOutputs(Protocol):
    def __call__(
        self, history: dict[str, Any], queued_at: float, expected_count: int
    ) -> list[Path]: ...


class ResolveOutput(Protocol):
    def __call__(self, history: dict[str, Any], queued_at: float) -> Path: ...


class EnsureH3LatentUpscalerModel(Protocol):
    def __call__(self, model_choice: str) -> bool: ...


class EnsureH3SemanticBridge(Protocol):
    def __call__(self) -> None: ...


class EnsureH3TextEncoder(Protocol):
    def __call__(self, models: ModelConfig, model_choice: str) -> tuple[str, bool]: ...


class EnsureInt8VideoVae(Protocol):
    def __call__(self, models: ModelConfig) -> bool: ...


class EnsureLtx25Models(Protocol):
    def __call__(self, model_choice: str = ...) -> bool: ...


class EnsureLtx25UpscaleModels(Protocol):
    def __call__(self, model_choice: str = ...) -> bool: ...


class EnsureMusic3Models(Protocol):
    def __call__(self, model_choice: str) -> bool: ...


class EnsureProfileModel(Protocol):
    def __call__(self, profile_key: str, profile: ModelProfile, mode: str) -> bool: ...


class EnsureSeedvr2UpscaleModels(Protocol):
    def __call__(self, models: ModelConfig, model_choice: str) -> bool: ...


class EnsureSingleFrameImageVae(Protocol):
    def __call__(self, models: ModelConfig) -> bool: ...


class EnsureTrtVideoVaeEngine(Protocol):
    def __call__(
        self, models: ModelConfig, *, force: bool = ..., progress=...
    ) -> bool: ...


class EnsureTurboLora(Protocol):
    def __call__(self, models: ModelConfig, turbo_variant: str, mode: str) -> bool: ...


class LoadModelConfig(Protocol):
    def __call__(self) -> ModelConfig: ...


class MissingLtx25ModelNames(Protocol):
    def __call__(self, model_choice: str = ...) -> list[str]: ...


class MissingMusic3ModelNames(Protocol):
    def __call__(self, model_choice: str) -> list[str]: ...


class TrtVaeDecoderPaths(Protocol):
    def __call__(self, models: ModelConfig) -> tuple[Path, Path, Path]: ...


class UnloadComfyModels(Protocol):
    def __call__(self) -> None: ...


class H3TextEncoderSettings(Protocol):
    def __call__(
        self, models: ModelConfig, model_choice: str
    ) -> tuple[str, str, bool]: ...


class ModelFileIsReady(Protocol):
    def __call__(self, path: Path) -> bool: ...


class UnloadPromptRewriter(Protocol):
    def __call__(self) -> None: ...


class ObjectInfo(Protocol):
    def __call__(self) -> dict[str, Any]: ...


class PollComfyProgress(Protocol):
    def __call__(self, prompt_id, graph) -> Iterator[ProgressUpdate]: ...


class StreamComfyProgress(Protocol):
    def __call__(self, ws, prompt_id, graph, started) -> Iterator[ProgressUpdate]: ...


class SubmitPrompt(Protocol):
    def __call__(self, graph: dict[str, Any], client_id: str) -> str: ...


class WaitForHistory(Protocol):
    def __call__(self, prompt_id) -> dict[str, Any]: ...


class ResolveRequestSettings(Protocol):
    def __call__(self, values: dict) -> ResolvedSettings: ...


class ResolveSolPolicy(Protocol):
    def __call__(
        self,
        attention_mode: str,
        mode: str,
        width: int,
        height: int,
        duration: float,
        first_image: str | None,
        last_image: str | None,
        use_turbo: bool = ...,
    ) -> tuple[bool, int, str]: ...


@dataclass(frozen=True)
class WorkflowsServices:
    build_fl2va_graph: BuildFl2VaGraph
    build_ltx25_graph: BuildLtx25Graph
    build_ref2va_graph: BuildRef2VaGraph
    build_upscale_graph: BuildUpscaleGraph


@dataclass(frozen=True)
class MediaServices:
    cleanup_upscale_clip_batch: CleanupUpscaleClipBatch
    concat_upscaled_clips: ConcatUpscaledClips
    postprocess_swiftvr_video: PostprocessSwiftvrVideo
    postprocess_video: PostprocessVideo
    prepare_upscale_clip_batch: PrepareUpscaleClipBatch
    probe_video_metadata: ProbeVideoMetadata
    resolve_audio_output: ResolveAudioOutput
    resolve_image_outputs: ResolveImageOutputs
    resolve_output: ResolveOutput


@dataclass(frozen=True)
class ModelsServices:
    ensure_h3_latent_upscaler_model: EnsureH3LatentUpscalerModel
    ensure_h3_semantic_bridge: EnsureH3SemanticBridge
    ensure_h3_text_encoder: EnsureH3TextEncoder
    ensure_int8_video_vae: EnsureInt8VideoVae
    ensure_ltx25_models: EnsureLtx25Models
    ensure_ltx25_upscale_models: EnsureLtx25UpscaleModels
    ensure_music3_models: EnsureMusic3Models
    ensure_profile_model: EnsureProfileModel
    ensure_seedvr2_upscale_models: EnsureSeedvr2UpscaleModels
    ensure_single_frame_image_vae: EnsureSingleFrameImageVae
    ensure_trt_video_vae_engine: EnsureTrtVideoVaeEngine
    ensure_turbo_lora: EnsureTurboLora
    load_model_config: LoadModelConfig
    missing_ltx25_model_names: MissingLtx25ModelNames
    missing_music3_model_names: MissingMusic3ModelNames
    trt_vae_decoder_paths: TrtVaeDecoderPaths
    unload_comfy_models: UnloadComfyModels
    h3_text_encoder_settings: H3TextEncoderSettings
    model_file_is_ready: ModelFileIsReady
    unload_prompt_rewriter: UnloadPromptRewriter


@dataclass(frozen=True)
class ExecutionServices:
    object_info: ObjectInfo
    poll_comfy_progress: PollComfyProgress
    stream_comfy_progress: StreamComfyProgress
    submit_prompt: SubmitPrompt
    wait_for_history: WaitForHistory


@dataclass(frozen=True)
class PolicyServices:
    resolve_request_settings: ResolveRequestSettings
    resolve_sol_policy: ResolveSolPolicy


@dataclass(frozen=True)
class GenerationServices:
    workflows: WorkflowsServices
    media: MediaServices
    models: ModelsServices
    execution: ExecutionServices
    policy: PolicyServices
