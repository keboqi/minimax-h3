"""Model controller with explicit runtime dependencies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import gradio as gr
from h3_models import DEFAULT_LTX25_MODEL
from h3_app.catalog import LTX25_UPSCALE
from h3_app.model_types import ModelConfig, ModelProfile
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelServices:
    COMFY_DIR: Any
    H3Error: Any
    MODELS_CONFIG: Any
    MODEL_SPECS: Any
    _runtime_config: Any
    ltx25_official_inventory_keys: Any
    ltx25_workflow_entry: Any
    ltx25_workflow_model_keys: Any
    model_file_is_ready: Any
    model_service: Any
    resolve_hf_token: Any
    stale_model_keys: Any
    sync_models: Any
    unload_comfy_models: Any
    _prepare_ltx25_model_set: Any
    ensure_trt_video_vae_engine: Any
    load_model_config: Any
    render_ltx25_official_model_inventory: Any


class ModelController:
    def __init__(self, services: ModelServices):
        self.services = services

    def load_model_config(self) -> ModelConfig:
        return self.services.model_service.load_model_config(
            runtime=self.services._runtime_config()
        )

    def trt_vae_decoder_paths(self, models: ModelConfig) -> tuple[Path, Path, Path]:
        return self.services.model_service.trt_vae_decoder_paths(
            models, runtime=self.services._runtime_config()
        )

    def ensure_h3_text_encoder(
        self, models: ModelConfig, model_choice: str
    ) -> tuple[str, bool]:
        return self.services.model_service.ensure_h3_text_encoder(
            models, model_choice, runtime=self.services._runtime_config()
        )

    def ensure_h3_semantic_bridge(self) -> None:
        return self.services.model_service.ensure_h3_semantic_bridge(
            runtime=self.services._runtime_config()
        )

    def ensure_h3_latent_upscaler_model(self, model_choice: str) -> bool:
        return self.services.model_service.ensure_h3_latent_upscaler_model(
            model_choice, runtime=self.services._runtime_config()
        )

    def ensure_profile_model(
        self, profile_key: str, profile: ModelProfile, mode: str
    ) -> bool:
        return self.services.model_service.ensure_profile_model(
            profile_key, profile, mode, runtime=self.services._runtime_config()
        )

    def ensure_turbo_lora(
        self, models: ModelConfig, turbo_variant: str, mode: str
    ) -> bool:
        return self.services.model_service.ensure_turbo_lora(
            models, turbo_variant, mode, runtime=self.services._runtime_config()
        )

    def ensure_base_video_vae(self, models: ModelConfig) -> bool:
        return self.services.model_service.ensure_base_video_vae(
            models, runtime=self.services._runtime_config()
        )

    def ensure_audio_vae(self, models: ModelConfig) -> bool:
        return self.services.model_service.ensure_audio_vae(
            models, runtime=self.services._runtime_config()
        )

    def ensure_int8_video_vae(
        self, models: ModelConfig, *, lynnreal: bool = False
    ) -> bool:
        return self.services.model_service.ensure_int8_video_vae(
            models, runtime=self.services._runtime_config(), lynnreal=lynnreal
        )

    def ensure_trt_video_vae_engine(
        self,
        models: ModelConfig,
        *,
        force: bool = False,
        progress=gr.Progress(track_tqdm=False),
    ) -> bool:
        return self.services.model_service.ensure_trt_video_vae_engine(
            models,
            force=force,
            progress=progress,
            release_backend=self.services.unload_comfy_models,
            runtime=self.services._runtime_config(),
        )

    def compile_trt_video_vae(self, progress=gr.Progress(track_tqdm=False)) -> str:
        """Build the local TensorRT decoder engine from the manual UI action."""
        try:
            self.services.ensure_trt_video_vae_engine(
                self.services.load_model_config(), force=True, progress=progress
            )
            return "TensorRT VAE decoder compiled and ready to use."
        except Exception as exc:
            return f"TensorRT VAE compilation failed: {exc}"

    def ensure_single_frame_image_vae(self, models: ModelConfig) -> bool:
        return self.services.model_service.ensure_single_frame_image_vae(
            models, runtime=self.services._runtime_config()
        )

    def missing_ltx25_model_names(
        self, model_choice: str = DEFAULT_LTX25_MODEL
    ) -> list[str]:
        return self.services.model_service.missing_ltx25_model_names(
            model_choice, runtime=self.services._runtime_config()
        )

    def ensure_ltx25_models(self, model_choice: str = DEFAULT_LTX25_MODEL) -> bool:
        return self.services.model_service.ensure_ltx25_models(
            model_choice, runtime=self.services._runtime_config()
        )

    def ensure_ltx25_ingredients_model(self) -> bool:
        return self.services.model_service.ensure_ltx25_ingredients_model(
            runtime=self.services._runtime_config()
        )

    def render_ltx25_official_model_inventory(self) -> str:
        keys = self.services.ltx25_official_inventory_keys()
        installed = 0
        rows = []
        for key in keys:
            spec = self.services.MODEL_SPECS[key]
            path = self.services.COMFY_DIR / "models" / spec.folder / spec.local_name
            ready = self.services.model_file_is_ready(path)
            installed += int(ready)
            status = "✅ Installed" if ready else "⬇️ Available"
            source = f"[{spec.repo_id}](https://huggingface.co/{spec.repo_id})"
            rows.append(
                f"| `{spec.local_name}` | `{spec.folder}` | {status} | {source} |"
            )
        return (
            f"**Installed: {installed}/{len(keys)}**\n\n| Model | ComfyUI folder | Status | Source / license |\n|---|---|---|---|\n"
            + "\n".join(rows)
        )

    def render_ltx25_workflow_details(self, workflow_label: str) -> str:
        entry = self.services.ltx25_workflow_entry(workflow_label)
        extra_names = [
            self.services.MODEL_SPECS[key].local_name for key in entry["extra_models"]
        ]
        extras = (
            ", ".join((f"`{name}`" for name in extra_names))
            if extra_names
            else "No use-case-specific checkpoint."
        )
        return f"### {workflow_label}\n\n{entry['description']}\n\n**Inputs:** {entry['inputs']}\n\n**Additional checkpoint(s):** {extras}\n\nModel preparation also installs the official BF16 transformer, text encoder, prompt enhancer, audio VAE, and (for video workflows) the full diffusion-decoder video VAE. These are large gated downloads.\n\n[Download official workflow JSON](/ltx25-workflows/{entry['id']}.json) · [Open ComfyUI](/comfyui/)\n\nThe template is also installed under **Workflows → Browse → LTX 2.5**. Download its models here first, then reload ComfyUI so its model dropdowns rescan the shared model folders."

    def _prepare_ltx25_model_set(self, required_keys: Iterable[str], label: str):
        """Lazily fetch a named set and refresh the visible model inventory."""
        try:
            required_keys = tuple(dict.fromkeys(required_keys))
            stale = self.services.stale_model_keys(
                root=self.services.COMFY_DIR / "models",
                manifest_path=self.services.MODELS_CONFIG.parent
                / "h3_model_manifest.json",
                model_keys=required_keys,
            )
            if not stale:
                yield (
                    f"Ready: all models for **{label}** are installed.",
                    self.services.render_ltx25_official_model_inventory(),
                )
                return
            names = ", ".join(
                (self.services.MODEL_SPECS[key].local_name for key in stale)
            )
            yield (
                f"Downloading models for **{label}**: {names}",
                self.services.render_ltx25_official_model_inventory(),
            )
            self.services.sync_models(
                root=self.services.COMFY_DIR / "models",
                manifest_path=self.services.MODELS_CONFIG.parent
                / "h3_model_manifest.json",
                token=self.services.resolve_hf_token(),
                log_prefix="[ltx25-workflow-on-demand]",
                model_keys=stale,
                download_workers=min(len(stale), 4),
            )
            missing = [
                self.services.MODEL_SPECS[key].local_name
                for key in required_keys
                if not self.services.model_file_is_ready(
                    self.services.COMFY_DIR
                    / "models"
                    / self.services.MODEL_SPECS[key].folder
                    / self.services.MODEL_SPECS[key].local_name
                )
            ]
            if missing:
                raise self.services.H3Error(
                    "Downloads did not produce: " + ", ".join(missing)
                )
            yield (
                f"Ready: installed all models for **{label}**. Open ComfyUI and refresh model definitions or reload the page.",
                self.services.render_ltx25_official_model_inventory(),
            )
        except Exception as exc:
            yield (
                f"Error downloading official workflow models. Open the Source / license links below, accept any gated terms, and authenticate with `hf auth login` or HF_TOKEN. Details: {exc}",
                self.services.render_ltx25_official_model_inventory(),
            )

    def prepare_ltx25_official_workflow(self, workflow_label: str):
        """Lazily fetch every checkpoint referenced by one official template."""
        yield from self.services._prepare_ltx25_model_set(
            self.services.ltx25_workflow_model_keys(workflow_label), workflow_label
        )

    def prepare_all_ltx25_official_models(self):
        """Download every missing model displayed in the official inventory."""
        yield from self.services._prepare_ltx25_model_set(
            self.services.ltx25_official_inventory_keys(),
            "all official workflow models",
        )

    def ensure_seedvr2_upscale_models(
        self, models: ModelConfig, model_choice: str
    ) -> bool:
        return self.services.model_service.ensure_seedvr2_upscale_models(
            models, model_choice, runtime=self.services._runtime_config()
        )

    def ensure_ltx25_upscale_models(
        self, model_choice: str = DEFAULT_LTX25_MODEL, *, option: str = LTX25_UPSCALE
    ) -> bool:
        return self.services.model_service.ensure_ltx25_upscale_models(
            model_choice, runtime=self.services._runtime_config(), option=option
        )

    def ensure_cq_image_enhance_models(self) -> bool:
        return self.services.model_service.ensure_cq_image_enhance_models(
            runtime=self.services._runtime_config()
        )

    def missing_music3_model_names(self, model_choice: str) -> list[str]:
        return self.services.model_service.missing_music3_model_names(
            model_choice, runtime=self.services._runtime_config()
        )

    def ensure_music3_models(self, model_choice: str) -> bool:
        return self.services.model_service.ensure_music3_models(
            model_choice, runtime=self.services._runtime_config()
        )

    def missing_qwen_image21_model_names(
        self, model_choice: str, text_encoder_choice: str, turbo_variant: str = "Off"
    ) -> list[str]:
        return self.services.model_service.missing_qwen_image21_model_names(
            model_choice,
            text_encoder_choice,
            turbo_variant,
            runtime=self.services._runtime_config(),
        )

    def ensure_qwen_image21_models(
        self, model_choice: str, text_encoder_choice: str, turbo_variant: str = "Off"
    ) -> bool:
        return self.services.model_service.ensure_qwen_image21_models(
            model_choice,
            text_encoder_choice,
            turbo_variant,
            runtime=self.services._runtime_config(),
        )

    def missing_yue2_model_names(self, model_choice: str) -> list[str]:
        return self.services.model_service.missing_yue2_model_names(
            model_choice, runtime=self.services._runtime_config()
        )

    def ensure_yue2_models(self, model_choice: str) -> bool:
        return self.services.model_service.ensure_yue2_models(
            model_choice, runtime=self.services._runtime_config()
        )
