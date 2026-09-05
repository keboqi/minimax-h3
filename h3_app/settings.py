"""Pure requested/effective settings and preset transitions.

No UI, provider SDK, filesystem or model downloads belong in this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping
import math

LIGHTX2V_4STEP = "LightX2V / 4-step (FL2V 768p · Ref2V 544p)"
LIGHTX2V_8STEP = "LightX2V v1.0 / 8-step 768p"
LARRY = "Larry v4-600 EMA"
TURBO_STEPS = {LIGHTX2V_4STEP: 4, LIGHTX2V_8STEP: 8, LARRY: 6}


@dataclass(frozen=True)
class SamplingSettings:
    steps: int = 4
    sol_tau: float = 1.2
    sol_thresh_type: str = "diag"
    scheduler: str = "simple"
    sol_exact_mode: str = "off"
    sol_dense_steps: int = 1
    auto_megapixels: str = "1 MP"
    turbo_variant: str = LIGHTX2V_4STEP
    attention_mode: str = "SLA"
    sla_preset: str = "Fast"
    latent_upscale_refine_steps: int = 2
    text_encoder: str = "NVFP4 / AWQ"
    stage_model_offload: bool = False


PRESET_FIELDS = tuple(SamplingSettings.__dataclass_fields__)
PRESETS = {
    "Fast": SamplingSettings(steps=15),
    "Balanced": SamplingSettings(
        steps=18,
        sol_tau=1.0,
        sol_exact_mode="exact_kv",
        auto_megapixels="2 MP",
        turbo_variant=LARRY,
        sla_preset="Balanced",
        text_encoder="INT8 ConvRot",
    ),
    "Quality": SamplingSettings(
        steps=20,
        sol_tau=0.8,
        sol_thresh_type="exact",
        scheduler="beta",
        sol_exact_mode="exact_kv_and_rows",
        auto_megapixels="4 MP",
        turbo_variant=LIGHTX2V_8STEP,
        sla_preset="Quality",
        text_encoder="BF16",
        stage_model_offload=True,
    ),
}


def preset_settings(name: str, mode: str) -> SamplingSettings:
    baseline = PRESETS.get(name, PRESETS["Balanced"])
    if mode == "Turbo":
        return replace(
            baseline, steps=TURBO_STEPS[baseline.turbo_variant], scheduler="simple"
        )
    return baseline


@dataclass(frozen=True)
class OutputSettings:
    result_format: str = "Video"
    width: int = 864
    height: int = 480
    duration: float = 5
    image_frames: int = 1
    image_vae: str = ""
    batch_count: int = 1
    seed: int = -1


@dataclass(frozen=True)
class FinishingSettings:
    latent_upscale: bool = True
    postprocess: str = "None"


@dataclass(frozen=True)
class GenerationRequest:
    sampling: SamplingSettings = field(default_factory=SamplingSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    finishing: FinishingSettings = field(default_factory=FinishingSettings)
    preset: str = "Fast"
    generation_mode: str = "Turbo"
    mode: str = "Text to video"
    model_profile: str = "Speed"
    cache_mode: str = "Spectrum"
    use_trt_vae: bool = True
    use_int8_vae: bool = False

    @classmethod
    def from_values(cls, values: Mapping[str, Any]) -> GenerationRequest:
        def group(kind):
            return kind(
                **{
                    key: values[key]
                    for key in kind.__dataclass_fields__
                    if key in values
                }
            )

        return cls(
            sampling=group(SamplingSettings),
            output=group(OutputSettings),
            finishing=group(FinishingSettings),
            **{
                key: values[key]
                for key in cls.__dataclass_fields__
                if key not in {"sampling", "output", "finishing"} and key in values
            },
        )


@dataclass(frozen=True)
class Adjustment:
    field: str
    requested: Any
    effective: Any
    reason: str


@dataclass(frozen=True)
class ResolutionContext:
    # The boundary reads image metadata using the existing aspect-ratio policy.
    start_dimensions: tuple[int, int] | None = None
    memory_profile: str = "unknown"


@dataclass(frozen=True)
class ResolvedSettings:
    requested: GenerationRequest
    effective: GenerationRequest
    adjustments: tuple[Adjustment, ...]
    inactive: frozenset[str]
    issues: tuple[str, ...]

    def differences(self) -> dict[str, tuple[Any, Any]]:
        baseline = asdict(
            preset_settings(self.requested.preset, self.requested.generation_mode)
        )
        requested = asdict(self.requested.sampling)
        return {
            key: (baseline[key], value)
            for key, value in requested.items()
            if key not in self.inactive and value != baseline[key]
        }


def resolve_settings(
    request: GenerationRequest, context: ResolutionContext = ResolutionContext()
) -> ResolvedSettings:
    adjustments: list[Adjustment] = []
    inactive: set[str] = set()
    issues: list[str] = []
    sampling, output, finishing = request.sampling, request.output, request.finishing

    def adjusted(key, before, after, reason):
        if before != after:
            adjustments.append(Adjustment(key, before, after, reason))
        return after

    fmt = str(output.result_format).title()
    if fmt not in {"Video", "Image", "Audio"}:
        issues.append("Choose Video, Image, or Audio output.")
    if fmt != "Video":
        inactive.update({"postprocess", "batch_count"})
        finishing = replace(finishing, postprocess="None")
        output = replace(output, batch_count=1)
    if fmt == "Audio":
        inactive.update(
            {"width", "height", "latent_upscale", "latent_upscale_refine_steps"}
        )
        finishing = replace(finishing, latent_upscale=False)
        output = replace(output, width=32, height=32)
    else:
        grid = 64 if finishing.latent_upscale else 32
        dimensions = (
            context.start_dimensions
            if fmt == "Image" and request.mode == "First / last frame"
            else None
        )
        source = dimensions or (output.width, output.height)
        snapped = tuple(
            max(grid, int(round(int(value) / grid)) * grid) for value in source
        )
        reason = (
            "From the start image, aligned for generation."
            if dimensions
            else f"Aligned to {grid}-pixel boundaries."
        )
        adjusted("dimensions", (output.width, output.height), snapped, reason)
        output = replace(output, width=snapped[0], height=snapped[1])
    if fmt == "Image":
        inactive.add("duration")
        if "500K" in output.image_vae:
            frames = adjusted(
                "image_frames",
                output.image_frames,
                1,
                "The 500K image decoder produces one frame.",
            )
            output = replace(output, image_frames=frames)
    else:
        inactive.update({"image_frames", "image_vae"})
    if sampling.text_encoder == "BF16":
        offload = adjusted(
            "stage_model_offload",
            sampling.stage_model_offload,
            True,
            "Required by the BF16 text encoder.",
        )
        sampling = replace(sampling, stage_model_offload=offload)
        inactive.add("stage_model_offload")
    if sampling.stage_model_offload and context.memory_profile == "gpu-only":
        issues.append("Stage offload requires restarting ComfyUI in Dynamic VRAM mode.")
    cache = request.cache_mode
    if finishing.latent_upscale:
        cache = adjusted(
            "cache_mode",
            cache,
            "Off",
            "Native refinement uses two samplers; acceleration cannot share cache state across them.",
        )
    if not finishing.latent_upscale:
        inactive.add("latent_upscale_refine_steps")
    if request.generation_mode != "Turbo":
        inactive.add("turbo_variant")
    if sampling.attention_mode not in {"Sol-Attn", "Auto"}:
        inactive.update(
            {"sol_tau", "sol_thresh_type", "sol_exact_mode", "sol_dense_steps"}
        )
    if sampling.attention_mode != "SLA":
        inactive.add("sla_preset")
    if request.mode != "First / last frame":
        inactive.add("auto_megapixels")
    minimum = 4 if request.generation_mode == "Turbo" else 10
    if sampling.steps < minimum:
        issues.append(
            f"{request.generation_mode} requires at least {minimum} sampling steps."
        )
    if (
        finishing.latent_upscale
        and not 1 <= sampling.latent_upscale_refine_steps < sampling.steps
    ):
        issues.append(
            "Refinement steps must be positive and smaller than base sampling steps."
        )
    if fmt != "Image" and not 2 <= output.duration <= 15:
        issues.append("Duration must be between 2 and 15 seconds.")
    if fmt == "Image" and not 1 <= output.image_frames <= 20:
        issues.append("Choose between 1 and 20 image frames.")
    if fmt == "Video" and not 1 <= output.batch_count <= 4:
        issues.append("Choose between 1 and 4 videos per batch.")
    effective = replace(
        request, sampling=sampling, output=output, finishing=finishing, cache_mode=cache
    )
    return ResolvedSettings(
        request, effective, tuple(adjustments), frozenset(inactive), tuple(issues)
    )


def transition_modes(
    saved: Mapping[str, Any] | None, values: Mapping[str, Any], action: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a new mode memory and controls. Only explicit actions apply presets."""
    memory = dict(saved or {})
    modes = {key: dict(value) for key, value in memory.get("modes", {}).items()}
    mode = values.get("generation_mode", "Turbo")
    previous = memory.get("active", mode)
    current = dict(values)
    # Input values on a mode event still contain the outgoing mode's settings.
    modes[previous] = {
        key: current[key]
        for key in (*PRESET_FIELDS, "preset", "cache_mode")
        if key in current
    }
    if action == "generation_mode" and mode != previous:
        current.update(
            modes.get(mode)
            or asdict(preset_settings(current.get("preset", "Fast"), mode))
        )
    elif action in {"preset", "restore"}:
        current.update(asdict(preset_settings(current.get("preset", "Fast"), mode)))
    elif action == "turbo_variant" and mode == "Turbo":
        current.update(
            steps=TURBO_STEPS.get(current["turbo_variant"], 4), scheduler="simple"
        )
    modes[mode] = {
        key: current[key]
        for key in (*PRESET_FIELDS, "preset", "cache_mode")
        if key in current
    }
    return {"active": mode, "modes": modes}, current


def valid_preference(
    value: Any, default: Any, *, choices=None, minimum=None, maximum=None
) -> Any:
    """Fail per field, never coerce strings into booleans or accept NaN."""
    if choices is not None and value not in choices:
        return default
    if isinstance(default, bool):
        return value if isinstance(value, bool) else default
    if isinstance(default, (int, float)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            return default
        if (
            minimum is not None
            and value < minimum
            or maximum is not None
            and value > maximum
        ):
            return default
        if isinstance(default, int) and int(value) != value:
            return default
    elif default is not None and not isinstance(value, type(default)):
        return default
    return value
