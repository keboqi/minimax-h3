"""Typed effects required by the generation orchestrators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Protocol

from h3_app.media_types import UpscaleClipBatch, VideoMetadata
from h3_app.model_types import ModelConfig, ModelProfile
from h3_app.progress import ProgressUpdate
from h3_app.settings import ResolvedSettings


# Keyword arguments of h3_app.workflows.h3.build_fl2va_graph: its media plus H3GraphOptions fields.
BuildFl2VaGraph = Callable[..., dict[str, Any]]


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


# Keyword arguments of h3_app.workflows.h3.build_ref2va_graph: its media plus H3GraphOptions fields.
BuildRef2VaGraph = Callable[..., dict[str, Any]]


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


class EnsureH3VedaPredictor(Protocol):
    def __call__(self, mode: str) -> None: ...


class EnsureH3TextEncoder(Protocol):
    def __call__(self, models: ModelConfig, model_choice: str) -> tuple[str, bool]: ...


class EnsureBaseVideoVae(Protocol):
    def __call__(self, models: ModelConfig) -> bool: ...


class EnsureAudioVae(Protocol):
    def __call__(self, models: ModelConfig) -> bool: ...


class EnsureInt8VideoVae(Protocol):
    def __call__(self, models: ModelConfig, *, lynnreal: bool = ..., sol: bool = ...) -> bool: ...


class EnsureLtx25Models(Protocol):
    def __call__(self, model_choice: str = ...) -> bool: ...


class EnsureLtx25IngredientsModel(Protocol):
    def __call__(self) -> bool: ...


class EnsureLtx25UpscaleModels(Protocol):
    def __call__(self, model_choice: str = ...) -> bool: ...


class EnsureMusic3Models(Protocol):
    def __call__(self, model_choice: str) -> bool: ...


class EnsureQwenImage21Models(Protocol):
    def __call__(self, model_choice: str, text_encoder_choice: str, turbo_variant: str = ...) -> bool: ...


class EnsureYuE2Models(Protocol):
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


class MissingQwenImage21ModelNames(Protocol):
    def __call__(
        self, model_choice: str, text_encoder_choice: str, turbo_variant: str = ...
    ) -> list[str]: ...


class MissingYuE2ModelNames(Protocol):
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
    postprocess_video: PostprocessVideo
    prepare_upscale_clip_batch: PrepareUpscaleClipBatch
    probe_video_metadata: ProbeVideoMetadata
    resolve_audio_output: ResolveAudioOutput
    resolve_image_outputs: ResolveImageOutputs
    resolve_output: ResolveOutput


@dataclass(frozen=True)
class ModelsServices:
    ensure_audio_vae: EnsureAudioVae
    ensure_base_video_vae: EnsureBaseVideoVae
    ensure_h3_latent_upscaler_model: EnsureH3LatentUpscalerModel
    ensure_h3_semantic_bridge: EnsureH3SemanticBridge
    ensure_h3_veda_predictor: EnsureH3VedaPredictor
    ensure_h3_text_encoder: EnsureH3TextEncoder
    ensure_int8_video_vae: EnsureInt8VideoVae
    ensure_lightvae: Callable[[str, str], bool]
    ensure_ltx25_models: EnsureLtx25Models
    ensure_ltx25_ingredients_model: EnsureLtx25IngredientsModel
    ensure_ltx25_upscale_models: EnsureLtx25UpscaleModels
    ensure_music3_models: EnsureMusic3Models
    ensure_qwen_image21_models: EnsureQwenImage21Models
    ensure_yue2_models: EnsureYuE2Models
    ensure_profile_model: EnsureProfileModel
    ensure_seedvr2_upscale_models: EnsureSeedvr2UpscaleModels
    ensure_single_frame_image_vae: EnsureSingleFrameImageVae
    ensure_trt_video_vae_engine: EnsureTrtVideoVaeEngine
    ensure_turbo_lora: EnsureTurboLora
    load_model_config: LoadModelConfig
    missing_ltx25_model_names: MissingLtx25ModelNames
    missing_music3_model_names: MissingMusic3ModelNames
    missing_qwen_image21_model_names: MissingQwenImage21ModelNames
    missing_yue2_model_names: MissingYuE2ModelNames
    trt_vae_decoder_paths: TrtVaeDecoderPaths
    unload_comfy_models: UnloadComfyModels
    h3_text_encoder_settings: H3TextEncoderSettings
    model_file_is_ready: ModelFileIsReady
    unload_prompt_rewriter: UnloadPromptRewriter
    ensure_model_keys: Callable[[tuple[str, ...]], bool]
    ensure_h3_regenerate_models: Callable[[ModelConfig, str], bool]


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
