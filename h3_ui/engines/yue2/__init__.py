"""YuE2 in Create: handlers and the feature record."""

from typing import Any

import gradio as gr

from h3_app.prompt_writers import song as song_writer
from h3_app.catalog import (
    DEFAULT_GEMINI_PROMPT_MODEL, DEFAULT_OPENAI_PROMPT_MODEL, GEMINI_PROMPT_MODELS, YUE2_DEFAULTS,
)
from h3_app.generation import yue2 as yue2_generation
from h3_app.generation.requests import YuE2Request
from h3_models import YUE2_MODEL_CHOICES

from ...engine_readiness import bind_engine_readiness
from ...job_bindings import owned_interrupt
from ...song_takes import take_facts
from ..feature import EngineDeps, EngineFeature
from .bindings import bind_yue2_view
from .view import YuE2View, build_yue2_view


class YuE2Handlers:
    """Generation, writing and take facts; the parameter names are the public API's."""

    def __init__(self, deps: EngineDeps):
        self.deps = deps

    def generate_yue2(
        self,
        model_choice: str,
        style: str,
        lyrics: str,
        abc: str,
        mode: str,
        max_duration: float,
        seed: int,
        steps: int,
        cfg: float,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        max_abc_tokens: int,
        abc_temperature: float,
        abc_top_p: float,
        abc_top_k: int,
        abc_repetition_penalty: float,
        abc_penalty_window: int,
        tiled_decode: bool,
        progress=gr.Progress(track_tqdm=False),
    ):
        request = YuE2Request(
            model_choice=model_choice, style=style, lyrics=lyrics, abc=abc, mode=mode,
            max_duration=max_duration, seed=seed, steps=steps, cfg=cfg, temperature=temperature,
            top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty, max_abc_tokens=max_abc_tokens,
            abc_temperature=abc_temperature, abc_top_p=abc_top_p, abc_top_k=abc_top_k,
            abc_repetition_penalty=abc_repetition_penalty, abc_penalty_window=abc_penalty_window,
            tiled_decode=tiled_decode,
        )
        yield from yue2_generation.generate_yue2(
            request, self.deps.generation_services(), self.deps.runtime_config(), progress=progress
        )

    def enhance_yue2_prompt(
        self,
        style: str,
        model: str,
        temporary_api_key: str,
        lyrics: str,
        mode: str,
        duration: float,
        backend: str = "Gemini",
        openai_api_key: str = "",
        openai_model: str = DEFAULT_OPENAI_PROMPT_MODEL,
        vocals: str = "Auto",
        language: str = "Auto",
        use: str = "Auto",
        lyrics_policy: str = "Keep my lyrics",
        notes: str = "",
        openai_base_url: str | None = None,
    ) -> tuple[str, str, str]:
        return song_writer.enhance_yue2_prompt(
            style, model, temporary_api_key, lyrics, mode, duration,
            backend=backend, openai_api_key=openai_api_key, openai_model=openai_model, openai_base_url=openai_base_url,
            vocals=vocals, language=language, use=use, lyrics_policy=lyrics_policy, notes=notes,
            runtime=self.deps.runtime_config(),
        )

    def take_facts(self, value) -> dict[str, Any]:
        return take_facts(value, self.deps.resolve_result)


def build(root, deps: EngineDeps) -> YuE2View:
    return build_yue2_view(
        root,
        prompt_models=GEMINI_PROMPT_MODELS,
        default_prompt_model=DEFAULT_GEMINI_PROMPT_MODEL,
        model_choices=YUE2_MODEL_CHOICES,
        defaults=YUE2_DEFAULTS,
    )


def bind(view: YuE2View, deps: EngineDeps) -> YuE2Handlers:
    handlers = YuE2Handlers(deps)
    event = bind_yue2_view(
        view, enhance_prompt=handlers.enhance_yue2_prompt, generate=handlers.generate_yue2,
        take_facts=handlers.take_facts,
    )
    bind_engine_readiness(view, "yue2")
    stopped = view.stop.click(owned_interrupt(deps.interrupt, "yue2"), outputs=view.status,
                              queue=False, api_name=False)
    stopped.then(fn=None, cancels=[event], queue=False, api_name=False)
    return handlers


FEATURE = EngineFeature("yue2", build, bind, settings=lambda view: dict(vars(view)))
