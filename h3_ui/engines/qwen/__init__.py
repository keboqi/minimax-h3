"""Qwen Image 2.1 in Create: handlers and the feature record."""

from pathlib import Path
from typing import Any

import gradio as gr

from h3_app.prompt_writers import qwen as qwen_writer
from h3_app.catalog import (
    DEFAULT_GEMINI_PROMPT_MODEL, DEFAULT_OPENAI_PROMPT_MODEL, GEMINI_PROMPT_MODELS, QWEN_EDIT_SIZE_MANUAL,
    QWEN_EDIT_SIZE_MATCH, QWEN_EDIT_SIZE_MAX, QWEN_IMAGE21_DEFAULTS,
)
from h3_app.errors import H3Error
from h3_app.generation import qwen as qwen_generation
from h3_app.generation.requests import QwenImage21Request
from h3_models import QWEN_IMAGE21_MODEL_CHOICES, QWEN_IMAGE21_TEXT_ENCODER_CHOICES

from ...engine_readiness import bind_engine_readiness
from ...job_bindings import owned_interrupt
from ..controls import resolution_preset_values
from ..feature import EngineDeps, EngineFeature, _picker_destinations
from .bindings import bind_qwen_image21_view
from .view import QwenImage21View, build_qwen_image21_view


def qwen_edit_size_flags(edit_size: str) -> tuple[bool, bool]:
    """Map the mutually exclusive UI choice onto the request's (match input, max resolution) fields."""
    if edit_size not in {QWEN_EDIT_SIZE_MATCH, QWEN_EDIT_SIZE_MAX, QWEN_EDIT_SIZE_MANUAL}:
        raise H3Error("Choose a valid Qwen edit output size.")
    return edit_size == QWEN_EDIT_SIZE_MATCH, edit_size == QWEN_EDIT_SIZE_MAX


class QwenHandlers:
    """Generation and rewriting; the parameter names are the public API's."""

    def __init__(self, deps: EngineDeps):
        self.deps = deps

    def generate_qwen_image21(
        self,
        mode: str,
        model_choice: str,
        text_encoder_choice: str,
        prompt: str,
        negative_prompt: str,
        reference_images: Any,
        width: int,
        height: int,
        reference_resolution: int,
        edit_size: str,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        cache_device: str,
        cache_dtype: str,
        attention_backend: str = "pytorch attention",
        accelerator: str = "Off",
        turbo_variant: str = "Off",
        output_resolution_1k: str | None = None,
        output_resolution_2k: str | None = None,
        batch_count: int = 1,
        batch_edit_inputs: bool = False,
        prompt_enhancer: str = "Off",
        lightning_strength: float = 0.5,
        *,
        output_resolution: str | None = None,
        progress=gr.Progress(track_tqdm=False),
    ):
        selected_preset = output_resolution_2k or output_resolution_1k or output_resolution
        match_input_size, max_resolution = qwen_edit_size_flags(edit_size)
        if selected_preset:
            width, height = resolution_preset_values(selected_preset)
            match_input_size = False
            max_resolution = False
        uploaded = reference_images or []
        if isinstance(uploaded, (str, Path)):
            uploaded = [uploaded]
        staged = tuple(
            self.deps.stage_file(str(path), "qwen_image21_references", reuse=True) for path in uploaded
        )
        request = QwenImage21Request(
            mode=mode,
            model_choice=model_choice,
            text_encoder_choice=text_encoder_choice,
            prompt=prompt,
            negative_prompt=negative_prompt,
            reference_images=staged,
            width=width,
            height=height,
            reference_resolution=reference_resolution,
            match_input_size=match_input_size,
            seed=seed,
            batch_count=batch_count,
            batch_edit_inputs=bool(batch_edit_inputs),
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            cache_device=cache_device,
            cache_dtype=cache_dtype,
            attention_backend=attention_backend,
            accelerator=accelerator,
            turbo_variant=turbo_variant,
            max_resolution=bool(max_resolution),
            prompt_enhancer=prompt_enhancer,
            lightning_strength=float(lightning_strength),
        )
        yield from qwen_generation.generate_qwen_image21(
            request, self.deps.generation_services(), self.deps.runtime_config(), progress=progress
        )

    def enhance_qwen_image21_prompt(
        self,
        prompt: str,
        model: str,
        temporary_api_key: str,
        mode: str,
        reference_images: Any,
        width: int,
        height: int,
        backend: str = "Gemini",
        openai_api_key: str = "",
        edit_size: str = QWEN_EDIT_SIZE_MATCH,
        openai_model: str = DEFAULT_OPENAI_PROMPT_MODEL,
        openai_base_url: str | None = None,
    ) -> tuple[str, str]:
        """Describe edits at the size the edit will be generated at."""
        match_input_size, max_resolution = qwen_edit_size_flags(edit_size)
        if (match_input_size or max_resolution) and str(mode).strip().lower() == "image edit":
            uploaded = reference_images or []
            if isinstance(uploaded, (str, Path)):
                uploaded = [uploaded]
            if uploaded:
                source_width, source_height = qwen_generation.first_reference_dimensions(str(uploaded[0]))
                width, height = (
                    qwen_generation.max_qwen_edit_dimensions(source_width, source_height)
                    if max_resolution
                    else (source_width, source_height)
                )
        return qwen_writer.enhance_qwen_image21_prompt(
            prompt, model, temporary_api_key, mode, reference_images, width, height,
            backend=backend, openai_api_key=openai_api_key, openai_model=openai_model,
            openai_base_url=openai_base_url,
            runtime=self.deps.runtime_config(),
        )


def build(root, deps: EngineDeps) -> QwenImage21View:
    return build_qwen_image21_view(
        root,
        model_choices=QWEN_IMAGE21_MODEL_CHOICES,
        text_encoder_choices=QWEN_IMAGE21_TEXT_ENCODER_CHOICES,
        prompt_models=GEMINI_PROMPT_MODELS,
        default_prompt_model=DEFAULT_GEMINI_PROMPT_MODEL,
        defaults=QWEN_IMAGE21_DEFAULTS,
    )


def bind(view: QwenImage21View, deps: EngineDeps) -> QwenHandlers:
    handlers = QwenHandlers(deps)
    event = bind_qwen_image21_view(
        view, enhance_prompt=handlers.enhance_qwen_image21_prompt, generate=handlers.generate_qwen_image21,
    )
    bind_engine_readiness(view, "qwen")
    stopped = view.stop.click(owned_interrupt(deps.interrupt, "qwen_image21"), outputs=view.status,
                              queue=False, api_name=False)
    stopped.then(fn=None, cancels=[event], queue=False, api_name=False)
    return handlers


FEATURE = EngineFeature(
    "qwen", build, bind, settings_key="qwen_image21", settings=lambda view: dict(vars(view)),
    library_label="Qwen", library_inputs=lambda view: tuple(c for c in vars(view).values() if hasattr(c, "h3_library_picker")),
    library_destinations=lambda view: _picker_destinations("Qwen", vars(view).values()),
)
