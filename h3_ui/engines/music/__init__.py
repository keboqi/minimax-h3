"""MiniMax Music 3 in Create: handlers and the feature record."""

from typing import Any

import gradio as gr

from h3_app import prompt_service
from h3_app.catalog import (
    DEFAULT_GEMINI_PROMPT_MODEL, DEFAULT_LIGHTNING_PROMPT_MODEL, GEMINI_PROMPT_MODELS, MUSIC3_DEFAULTS,
)
from h3_app.generation import music as music_generation
from h3_app.generation.requests import MusicRequest
from h3_models import MUSIC3_MODEL_CHOICES

from ...engine_readiness import bind_engine_readiness
from ...job_bindings import owned_interrupt
from ...song_takes import take_facts
from ..feature import EngineDeps, EngineFeature
from .bindings import bind_music_view
from .view import MusicView, build_music_view


class MusicHandlers:
    """Generation, writing and take facts; the parameter names are the public API's."""

    def __init__(self, deps: EngineDeps):
        self.deps = deps

    def generate_music3(
        self,
        model_choice: str,
        caption: str,
        lyrics: str,
        max_duration: float,
        seed: int,
        steps: int,
        cfg: float,
        ar_cfg: float,
        top_k: int,
        tiled_decode: bool,
        progress=gr.Progress(track_tqdm=False),
    ):
        request = MusicRequest(
            model_choice=model_choice, caption=caption, lyrics=lyrics, max_duration=max_duration, seed=seed,
            steps=steps, cfg=cfg, ar_cfg=ar_cfg, top_k=top_k, tiled_decode=tiled_decode,
        )
        yield from music_generation.generate_music3(
            request, self.deps.generation_services(), self.deps.runtime_config(), progress=progress
        )

    def enhance_music3_prompt(
        self,
        prompt: str,
        model: str,
        temporary_api_key: str,
        lyrics: str,
        ref_image_1: Any,
        ref_image_2: Any,
        ref_image_3: Any,
        backend: str = "Gemini",
        lightning_api_key: str = "",
        lightning_model: str = DEFAULT_LIGHTNING_PROMPT_MODEL,
        duration: float = 120,
        vocals: str = "Auto",
        language: str = "Auto",
        use: str = "Auto",
        lyrics_policy: str = "Keep my lyrics",
        notes: str = "",
    ) -> tuple[str, str, str]:
        return prompt_service.enhance_music3_prompt(
            prompt, model, temporary_api_key, lyrics, ref_image_1, ref_image_2, ref_image_3,
            backend=backend, lightning_api_key=lightning_api_key, lightning_model=lightning_model,
            duration=duration, vocals=vocals, language=language, use=use, lyrics_policy=lyrics_policy,
            notes=notes, runtime=self.deps.runtime_config(),
        )

    def take_facts(self, value) -> dict[str, Any]:
        return take_facts(value, self.deps.resolve_result)


def build(root, deps: EngineDeps) -> MusicView:
    return build_music_view(
        root,
        prompt_models=GEMINI_PROMPT_MODELS,
        default_prompt_model=DEFAULT_GEMINI_PROMPT_MODEL,
        model_choices=MUSIC3_MODEL_CHOICES,
        defaults=MUSIC3_DEFAULTS,
    )


def bind(view: MusicView, deps: EngineDeps) -> MusicHandlers:
    handlers = MusicHandlers(deps)
    event = bind_music_view(
        view, enhance_prompt=handlers.enhance_music3_prompt, generate=handlers.generate_music3,
        take_facts=handlers.take_facts,
    )
    bind_engine_readiness(view, "music")
    stopped = view.stop.click(owned_interrupt(deps.interrupt, "music"), outputs=view.status,
                              queue=False, api_name=False)
    stopped.then(fn=None, cancels=[event], queue=False, api_name=False)
    return handlers


FEATURE = EngineFeature(
    "music", build, bind, settings_key="music3", settings=lambda view: dict(vars(view)),
    library_inputs=lambda view: tuple(view.reference_images),
    library_destinations=lambda view: {"Music · Reference images": view.reference_images},
)
