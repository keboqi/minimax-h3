"""Extracted h3 app/status boundary."""

from __future__ import annotations

import time
from typing import Any

from h3_app.catalog import (
    CHUNK_FEED_FORWARD_NODE,
    CORE_LORA_LOADER_NODE,
    CORE_SAMPLER_NODE,
    FUSED_MODULATION_NODE,
    H3_COMBINE_AV_LATENT_NODE,
    H3_LATENT_UPSCALER_NODE,
    H3_NVENC_SAVE_NODE,
    H3_SEPARATE_AV_LATENT_NODE,
    H3_SPLIT_SPATIAL_PARAMS_NODE,
    H3_SPLIT_TEMPORAL_PARAMS_NODE,
    H3_SPLIT_UPSCALE_NODE,
    LARRY_TURBO_LORA_NODE,
    LIGHTX2V_BYPASS_LORA_NODE,
    SAGE_ATTENTION_NODE,
    SOL_ATTENTION_NODE,
)


def graph_class_types(graph: dict[str, Any]) -> set[str]:
    return {str(node.get("class_type", "")) for node in graph.values()}


def node_stage(class_type: str, workflow_classes: set[str] | None = None) -> str:
    """Turn ComfyUI implementation node names into useful user-facing stages."""
    name = str(class_type)
    workflow_classes = workflow_classes or set()
    if name in {
        "UNETLoader",
        "CLIPLoader",
        "VAELoader",
        "CheckpointLoaderSimple",
        "LatentUpscaleModelLoader",
        CORE_LORA_LOADER_NODE,
        "LTXICLoRALoaderModelOnly",
        LARRY_TURBO_LORA_NODE,
        LIGHTX2V_BYPASS_LORA_NODE,
    }:
        return "Loading models"
    if name.startswith("Load") or name == "GetVideoComponents":
        return "Preparing reference media"
    if name == "LTXVAddGuide":
        return "Applying LTX-2.5 keyframes"
    if name == "LTXAddVideoICLoRAGuide":
        return "Encoding source video for LTX-2.5 2x upscaling"
    if name == "LTXVCropGuides":
        return "Removing LTX-2.5 reference tokens"
    if name in {"CLIPTextEncode", "LTXVConditioning"}:
        return "Encoding LTX-2.5 prompt"
    if name == "MiniMaxMusic3TextEncode":
        return "Composing song structure and acoustic conditioning"
    if name == "EmptyMiniMaxMusic3LatentAudio":
        return "Preparing Music 3 audio latents"
    if name in {
        "EmptyLTXVLatentVideo",
        "LTXVEmptyLatentAudio",
        "LTXVConcatAVLatent",
    }:
        return "Preparing LTX-2.5 audio-video latents"
    if name == "VAEEncodeTiled":
        if "SeedVR2Preprocess" in workflow_classes:
            return "Encoding H3 video for SeedVR2"
        return "Encoding video"
    if name == "SeedVR2Preprocess":
        return "Preparing SeedVR2 input"
    if name == "SeedVR2TemporalChunk":
        return "Splitting SeedVR2 video into VRAM-safe chunks"
    if name == "SeedVR2Conditioning":
        return "Preparing SeedVR2 conditioning"
    if name == "SeedVR2TemporalMerge":
        return "Merging SeedVR2 chunks"
    if name == "SeedVR2PostProcessing":
        return "Restoring SeedVR2 output"
    if "ImageToVideo" in name or "ReferenceToVideo" in name:
        return "Encoding prompt and conditioning"
    if name in {
        SOL_ATTENTION_NODE,
        SAGE_ATTENTION_NODE,
        FUSED_MODULATION_NODE,
        CHUNK_FEED_FORWARD_NODE,
        "SpectrumApplyMiniMaxH3",
        "H3FirstBlockCache",
        "EasyCache",
    }:
        return "Configuring generation model"
    if name in {
        "RandomNoise",
        "BasicGuider",
        "CFGGuider",
        CORE_SAMPLER_NODE,
        "BasicScheduler",
        "ManualSigmas",
        "SplitSigmas",
    }:
        return "Preparing sampler"
    if name == H3_SEPARATE_AV_LATENT_NODE:
        return "Separating H3 video and audio latents"
    if name == H3_LATENT_UPSCALER_NODE:
        return "Upscaling H3 video latent 2x"
    if name == H3_COMBINE_AV_LATENT_NODE:
        return "Recombining H3 video and audio latents"
    if name in {H3_SPLIT_TEMPORAL_PARAMS_NODE, H3_SPLIT_SPATIAL_PARAMS_NODE}:
        return "Preparing MMH3 split-upscale tiles and chunks"
    if name == H3_SPLIT_UPSCALE_NODE:
        return "Refining MMH3 temporal chunks and spatial tiles"
    if name == "KSampler" and "MiniMaxMusic3TextEncode" in workflow_classes:
        return "Generating music"
    if name == "SamplerCustomAdvanced" or "Sampler" in name:
        return "Generating video and audio"
    if name in {
        "VAEDecode",
        "VAEDecodeAudio",
        "VAEDecodeTiled",
        "LTXVSeparateAVLatent",
        "LTXVAudioVAEDecode",
    }:
        return "Decoding output"
    if name == "CreateVideo":
        return "Assembling video"
    if name == H3_NVENC_SAVE_NODE:
        return "Saving video with NVENC"
    if name.startswith("Save"):
        return "Saving audio" if "Audio" in name else "Saving video"
    return name


def progress_status(
    stage: str,
    *,
    started: float,
    completed_nodes: int = 0,
    total_nodes: int = 0,
    step: int | None = None,
    step_total: int | None = None,
    configured_steps: int | None = None,
    detail: str | None = None,
) -> str:
    elapsed = time.monotonic() - started
    lines = [f"{stage} · elapsed {elapsed:.1f}s"]
    if step is not None and step_total:
        percent = 100 * step / step_total
        if configured_steps and step_total != configured_steps:
            lines.append(
                f"Overall generation progress {step}/{step_total} ({percent:.0f}%)"
            )
            lines.append(f"Sampling schedule {configured_steps} steps (UI setting)")
        elif configured_steps:
            lines.append(f"Sampler step {step}/{configured_steps} ({percent:.0f}%)")
        else:
            lines.append(f"Stage progress {step}/{step_total} ({percent:.0f}%)")
    if total_nodes:
        lines.append(f"Workflow nodes {completed_nodes}/{total_nodes}")
    if detail:
        lines.append(detail)
    return "\n".join(lines)


class StageTimings:
    """Accumulate user-facing pipeline stage durations and log transitions."""

    def __init__(self, label: str, started: float, initial_stage: str) -> None:
        self.label = label
        self.started = started
        self.current_stage: str | None = initial_stage
        self.current_started = started
        self.durations: dict[str, float] = {}
        self.finished = False

    def transition(self, stage: str, *, now: float | None = None) -> None:
        if self.finished or stage == self.current_stage:
            return
        timestamp = time.monotonic() if now is None else now
        self._close_current(timestamp)
        self.current_stage = stage
        self.current_started = timestamp

    def _close_current(self, now: float) -> None:
        if self.current_stage is None:
            return
        elapsed = max(0.0, now - self.current_started)
        self.durations[self.current_stage] = (
            self.durations.get(self.current_stage, 0.0) + elapsed
        )
        print(
            f"[h3-timing] {self.label} · {self.current_stage}: {elapsed:.2f}s",
            flush=True,
        )

    def finish(self, *, now: float | None = None) -> float:
        timestamp = time.monotonic() if now is None else now
        if not self.finished:
            self._close_current(timestamp)
            self.current_stage = None
            self.finished = True
            print(
                f"[h3-timing] {self.label} · total: "
                f"{max(0.0, timestamp - self.started):.2f}s",
                flush=True,
            )
        return max(0.0, timestamp - self.started)

    def summary(self, *, now: float | None = None) -> str:
        self.finish(now=now)
        details = " · ".join(
            f"{stage} {elapsed:.1f}s" for stage, elapsed in self.durations.items()
        )
        return f"Step times: {details}" if details else "Step times: unavailable"
