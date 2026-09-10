"""Extracted model types boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from h3_app.catalog import (
    LTX25_WORKFLOW_COMMON_MODEL_KEYS,
    LTX25_WORKFLOWS,
    TURBO_SETTINGS,
)
from h3_app.errors import H3Error
from h3_app.policy import normalize_turbo_variant
from h3_models import (
    DEFAULT_LTX25_MODEL,
    DEFAULT_SEEDVR2_MODEL,
    H3_TEXT_ENCODER_CHOICES,
    LTX25_ICLORA_MODEL_KEYS,
    LTX25_MODEL_CHOICES,
    LTX25_SHARED_MODEL_KEYS,
    MIN_VALID_MODEL_BYTES,
    MODEL_SPECS,
    MUSIC3_MODEL_CHOICES,
    MUSIC3_SHARED_MODEL_KEYS,
    SEEDVR2_MODEL_CHOICES,
)


@dataclass
class ModelProfile:
    label: str
    fl2va: str
    ref2va: str
    fl2va_source: str = "unknown"
    ref2va_source: str = "unknown"


@dataclass
class ModelConfig:
    profiles: dict[str, ModelProfile]
    default_profile: str
    text_encoder: str
    video_vae: str
    audio_vae: str
    text_encoders: dict[str, str] | None = None
    video_vae_int8: str | None = None
    video_vae_trt_encoder: str | None = None
    video_vae_trt_decoder: str | None = None
    video_vae_trt_source: str = "unknown"
    video_vae_int8_source: str = "unknown"
    image_vae_500k: str | None = None
    image_vae_500k_source: str = "unknown"
    turbo_lora: str | None = None
    turbo_source: str = "unknown"
    turbo_ref_lora: str | None = None
    turbo_ref_source: str = "unknown"
    turbo_8step_lora: str | None = None
    turbo_8step_source: str = "unknown"
    turbo_8step_ref_lora: str | None = None
    turbo_8step_ref_source: str = "unknown"
    larry_turbo_lora: str | None = None
    larry_turbo_source: str = "unknown"
    larry_turbo_ref_lora: str | None = None
    larry_turbo_ref_source: str = "unknown"
    seedvr2_dit: str | None = None
    seedvr2_dit_source: str = "unknown"
    seedvr2_models: dict[str, str] | None = None
    seedvr2_vae: str | None = None
    seedvr2_vae_source: str = "unknown"

    def profile_key(self, name: str) -> str:
        key = str(name).strip().lower()
        if key not in self.profiles:
            key = self.default_profile
        if key not in self.profiles:
            raise H3Error(f"Unknown model profile: {name}")
        return key

    def profile(self, name: str) -> ModelProfile:
        return self.profiles[self.profile_key(name)]

    def turbo_lora_for(self, mode: str, turbo_variant: str) -> str | None:
        reference = str(mode).strip().lower() == "reference media"
        spec = TURBO_SETTINGS[normalize_turbo_variant(turbo_variant)]
        return getattr(self, spec.ref_lora_attr if reference else spec.lora_attr)


def model_file_is_ready(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > MIN_VALID_MODEL_BYTES


def trt_vae_engine_name(onnx_name: str) -> str:
    return str(Path(onnx_name).with_suffix(".engine"))


def h3_text_encoder_settings(
    models: ModelConfig,
    model_choice: str,
) -> tuple[str, str, bool]:
    """Resolve an H3 text-encoder label to its inventory key and filename."""
    choice = str(model_choice)
    model_key = H3_TEXT_ENCODER_CHOICES.get(choice)
    if model_key is None:
        raise H3Error(f"Unknown H3 text encoder: {model_choice}")
    configured = dict(models.text_encoders or {})
    if not configured:
        # Legacy configs expose one unnamed encoder. Associate it only when
        # its filename identifies a known choice; changing the UI default must
        # not relabel an older INT8/NVFP4 file as BF16.
        for label, key in H3_TEXT_ENCODER_CHOICES.items():
            if models.text_encoder == MODEL_SPECS[key].local_name:
                configured[label] = models.text_encoder
                break
    filename = configured.get(choice, MODEL_SPECS[model_key].local_name)
    return model_key, filename, model_key == "text_encoder_bf16"


def ltx25_model_keys(model_choice: str = DEFAULT_LTX25_MODEL) -> dict[str, str]:
    selected_key = LTX25_MODEL_CHOICES.get(str(model_choice))
    if selected_key is None:
        raise H3Error(f"Unknown LTX-2.5 model: {model_choice}")
    return {
        "distilled": selected_key,
        **{key.removeprefix("ltx25_"): key for key in LTX25_SHARED_MODEL_KEYS},
    }


def ltx25_model_names(model_choice: str = DEFAULT_LTX25_MODEL) -> dict[str, str]:
    return {
        role: MODEL_SPECS[key].local_name
        for role, key in ltx25_model_keys(model_choice).items()
    }


def ltx25_workflow_entry(workflow_label: str) -> dict[str, Any]:
    entry = LTX25_WORKFLOWS.get(str(workflow_label))
    if entry is None:
        raise H3Error(f"Unknown official LTX-2.5 workflow: {workflow_label}")
    return entry


def ltx25_workflow_model_keys(workflow_label: str) -> tuple[str, ...]:
    entry = ltx25_workflow_entry(workflow_label)
    common = list(LTX25_WORKFLOW_COMMON_MODEL_KEYS)
    if entry.get("audio_only"):
        common.remove("ltx25_video_vae")
        common.remove("ltx25_video_vae_full")
    return tuple(dict.fromkeys((*common, *entry["extra_models"])))


def ltx25_official_inventory_keys() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *LTX25_WORKFLOW_COMMON_MODEL_KEYS,
                *LTX25_ICLORA_MODEL_KEYS,
                "ltx25_spatial_upscaler",
            )
        )
    )


def seedvr2_upscale_model_names(
    models: ModelConfig,
    model_choice: str,
) -> dict[str, str]:
    choice = str(model_choice)
    if choice not in SEEDVR2_MODEL_CHOICES:
        raise H3Error(f"Unknown SeedVR2 model: {model_choice}")
    configured_models = models.seedvr2_models or {}
    selected = configured_models.get(choice)
    if selected is None and choice == DEFAULT_SEEDVR2_MODEL:
        selected = models.seedvr2_dit
    configured = {"seedvr2_dit": selected, "seedvr2_vae": models.seedvr2_vae}
    missing_config = [key for key, value in configured.items() if not value]
    if missing_config:
        raise H3Error(
            "The model configuration predates SeedVR2 support. Re-run "
            "setup_h3.py before selecting SeedVR2. Missing keys: "
            + ", ".join(missing_config)
        )
    return {key: str(value) for key, value in configured.items()}


def music3_model_keys(model_choice: str) -> tuple[str, ...]:
    choice = str(model_choice)
    if choice not in MUSIC3_MODEL_CHOICES:
        raise H3Error(f"Unknown MiniMax Music 3 model: {model_choice}")
    return (MUSIC3_MODEL_CHOICES[choice], *MUSIC3_SHARED_MODEL_KEYS)
