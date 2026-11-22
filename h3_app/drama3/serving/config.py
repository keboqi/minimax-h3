"""Global Drama 3 serving configuration shared across Soundstage and Dubbing."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os

from h3_app.errors import H3Error
from h3_app.jobs import JOBS
from h3_gpu import default_drama3_checkpoint, is_a100
from h3_models import DEFAULT_DRAMA3_COMFY_MODEL, DRAMA3_COMFY_MODELS

BACKEND_SGLANG = "sglang"
BACKEND_COMFYUI = "comfyui"
BACKENDS = (BACKEND_SGLANG, BACKEND_COMFYUI)
BACKEND_CHOICES = (
    ("SGLang-Omni (parallel)", BACKEND_SGLANG),
    ("ComfyUI (sequential, fast start)", BACKEND_COMFYUI),
)

SGLANG_MODEL_CHOICES_ALL = (
    ("FP8 — fishaudio/drama-3 (~41.5 GB)", "fp8"),
    ("INT4 — fishaudio/drama-3-int4 (~24 GB)", "int4"),
)
COMFY_EXPERTS_CHOICES = (
    ("In VRAM (fastest, ~25–39 GB VRAM)", "in VRAM"),
    ("In RAM (offloaded experts)", "in RAM"),
)
COMFY_EXPERTS_MODES = ("in VRAM", "in RAM")
SETTINGS_KEY = "drama3.serving"


def sglang_model_choices() -> list[tuple[str, str]]:
    """Return SGLang-Omni Drama 3 checkpoint choices supported on this GPU profile."""
    if is_a100():
        return [choice for choice in SGLANG_MODEL_CHOICES_ALL if choice[1] != "fp8"]
    return list(SGLANG_MODEL_CHOICES_ALL)


def comfy_model_choices() -> list[tuple[str, str]]:
    """Return Drama-3-ComfyUI quantized checkpoint choices."""
    return [(label, key) for key, label in DRAMA3_COMFY_MODELS.items()]


def model_choices_for_backend(backend: str) -> list[tuple[str, str]]:
    """Return dropdown choices for the active Drama 3 serving backend."""
    if backend == BACKEND_COMFYUI:
        return comfy_model_choices()
    return sglang_model_choices()


@dataclass(frozen=True)
class Drama3GlobalSettings:
    """Global Drama 3 backend and model selection persisted in WorkspaceStore."""

    backend: str = BACKEND_SGLANG
    sglang_checkpoint: str = "fp8"
    comfy_model: str = DEFAULT_DRAMA3_COMFY_MODEL
    comfy_experts: str = "in VRAM"

    @property
    def backend_label(self) -> str:
        return "ComfyUI" if self.backend == BACKEND_COMFYUI else "SGLang-Omni"

    @property
    def active_model_key(self) -> str:
        return self.comfy_model if self.backend == BACKEND_COMFYUI else self.sglang_checkpoint

    @property
    def active_model_label(self) -> str:
        if self.backend == BACKEND_COMFYUI:
            return DRAMA3_COMFY_MODELS.get(self.comfy_model, self.comfy_model)
        return dict((value, label) for label, value in SGLANG_MODEL_CHOICES_ALL).get(
            self.sglang_checkpoint, self.sglang_checkpoint.upper()
        )


def _normalize_backend(raw: str | None) -> str:
    value = str(raw or "").strip().lower()
    return value if value in BACKENDS else BACKEND_SGLANG


def _normalize_sglang_checkpoint(raw: str | None) -> str:
    value = str(raw or "").strip().lower()
    if value in {"fp8", "int4"}:
        return value
    return default_drama3_checkpoint()


def _normalize_comfy_model(raw: str | None) -> str:
    value = str(raw or "").strip()
    return value if value in DRAMA3_COMFY_MODELS else DEFAULT_DRAMA3_COMFY_MODEL


def _normalize_comfy_experts(raw: str | None) -> str:
    value = str(raw or "").strip()
    for mode in COMFY_EXPERTS_MODES:
        if value.casefold() == mode.casefold():
            return mode
    return "in VRAM"


def default_drama3_settings() -> Drama3GlobalSettings:
    """Resolve Drama 3 global defaults from environment variables and GPU profile."""
    env = os.environ
    backend = _normalize_backend(env.get("H3_DRAMA3_BACKEND") or BACKEND_SGLANG)
    sglang_checkpoint = _normalize_sglang_checkpoint(
        env.get("H3_DRAMA3_SGLANG_CHECKPOINT") or default_drama3_checkpoint()
    )
    comfy_model = _normalize_comfy_model(env.get("H3_DRAMA3_COMFY_MODEL") or DEFAULT_DRAMA3_COMFY_MODEL)
    comfy_experts = _normalize_comfy_experts(env.get("H3_DRAMA3_COMFY_EXPERTS") or "in VRAM")
    return Drama3GlobalSettings(
        backend=backend,
        sglang_checkpoint=sglang_checkpoint,
        comfy_model=comfy_model,
        comfy_experts=comfy_experts,
    )


def _read_store_payload(store) -> dict:
    target = JOBS.store if store is None else store
    if target is None or not hasattr(target, "get_setting"):
        return {}
    raw = target.get_setting(SETTINGS_KEY)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


def has_persisted_sglang_checkpoint(store=None) -> str | None:
    """Return persisted SGLang checkpoint override if explicitly saved in the store."""
    payload = _read_store_payload(store)
    value = str(payload.get("sglang_checkpoint") or "").strip().lower()
    return value if value in {"fp8", "int4"} else None


def load_drama3_settings(store=None, *, coerce_a100: bool = True) -> Drama3GlobalSettings:
    """Load persisted Drama 3 global settings overlaid onto environment/GPU defaults."""
    defaults = default_drama3_settings()
    payload = _read_store_payload(store)
    backend = _normalize_backend(payload.get("backend") or defaults.backend)
    sglang_checkpoint = _normalize_sglang_checkpoint(
        payload.get("sglang_checkpoint") or defaults.sglang_checkpoint
    )
    if coerce_a100 and is_a100() and sglang_checkpoint == "fp8":
        sglang_checkpoint = "int4"
    comfy_model = _normalize_comfy_model(payload.get("comfy_model") or defaults.comfy_model)
    comfy_experts = _normalize_comfy_experts(payload.get("comfy_experts") or defaults.comfy_experts)
    return Drama3GlobalSettings(
        backend=backend,
        sglang_checkpoint=sglang_checkpoint,
        comfy_model=comfy_model,
        comfy_experts=comfy_experts,
    )


def validate_drama3_settings(settings: Drama3GlobalSettings) -> Drama3GlobalSettings:
    """Validate backend, model, and GPU compatibility constraints."""
    if settings.backend not in BACKENDS:
        raise H3Error(f"Unsupported Drama 3 backend: {settings.backend!r}.")
    if settings.sglang_checkpoint not in {"fp8", "int4"}:
        raise H3Error(f"Unsupported SGLang-Omni Drama 3 checkpoint: {settings.sglang_checkpoint!r}.")
    if settings.comfy_model not in DRAMA3_COMFY_MODELS:
        raise H3Error(f"Unsupported ComfyUI Drama 3 model: {settings.comfy_model!r}.")
    if settings.comfy_experts not in COMFY_EXPERTS_MODES:
        raise H3Error(f"Unsupported ComfyUI expert placement: {settings.comfy_experts!r}.")
    if settings.backend == BACKEND_SGLANG and is_a100() and settings.sglang_checkpoint == "fp8":
        raise H3Error(
            "SGLang-Omni's FP8 Drama 3 checkpoint requires an SM89+ GPU and cannot run on A100. "
            "Switch to SGLang-Omni INT4 or ComfyUI Drama 3 in System -> Drama 3."
        )
    return settings


def save_drama3_settings(settings: Drama3GlobalSettings, store=None) -> Drama3GlobalSettings:
    """Validate and persist global Drama 3 settings into WorkspaceStore."""
    validated = validate_drama3_settings(settings)
    target = JOBS.store if store is None else store
    if target is not None and hasattr(target, "set_setting"):
        target.set_setting(
            SETTINGS_KEY,
            json.dumps(
                {
                    "backend": validated.backend,
                    "sglang_checkpoint": validated.sglang_checkpoint,
                    "comfy_model": validated.comfy_model,
                    "comfy_experts": validated.comfy_experts,
                },
                sort_keys=True,
            ),
        )
    return validated


def describe_drama3_settings(settings: Drama3GlobalSettings) -> str:
    """Format a concise Markdown summary of active Drama 3 serving settings."""
    if settings.backend == BACKEND_COMFYUI:
        return (
            f"Active Drama 3 backend: **ComfyUI** · **{settings.active_model_label}** · "
            f"experts **{settings.comfy_experts}** (sequential chunk generation, fast start). "
            "Applies to Soundstage, Voice Design, and Dubbing."
        )
    return (
        f"Active Drama 3 backend: **SGLang-Omni** · **{settings.active_model_label}** "
        "(parallel batch generation). Applies to Soundstage, Voice Design, and Dubbing."
    )


def apply_drama3_settings_update(
    backend: str,
    model: str,
    comfy_experts: str,
    *,
    store=None,
) -> tuple[Drama3GlobalSettings, str]:
    """Apply a UI change to global Drama 3 settings, stopping SGLang when its config changes."""
    from . import server as drama3_server

    previous = load_drama3_settings(store)
    target_backend = backend if backend in BACKENDS else previous.backend
    sglang_checkpoint = previous.sglang_checkpoint
    comfy_model = previous.comfy_model
    if target_backend == BACKEND_SGLANG and model in {"fp8", "int4"}:
        sglang_checkpoint = model
    elif target_backend == BACKEND_COMFYUI and model in DRAMA3_COMFY_MODELS:
        comfy_model = model
    experts = comfy_experts if comfy_experts in COMFY_EXPERTS_MODES else previous.comfy_experts
    candidate = Drama3GlobalSettings(
        backend=target_backend,
        sglang_checkpoint=sglang_checkpoint,
        comfy_model=comfy_model,
        comfy_experts=experts,
    )
    error_note = ""
    try:
        saved = save_drama3_settings(candidate, store=store)
    except H3Error as exc:
        saved = previous
        error_note = f"\n\n**Could not apply change:** {exc}"
    else:
        if previous.backend == BACKEND_SGLANG and (
            saved.backend != BACKEND_SGLANG or saved.sglang_checkpoint != previous.sglang_checkpoint
        ):
            drama3_server.stop_all()
    return saved, describe_drama3_settings(saved) + error_note
