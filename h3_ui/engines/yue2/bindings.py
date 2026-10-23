"""Reusable Gradio event-binding helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ...job_bindings import bind_gpu_action, bind_prompt_action, owned_generation
from ...song_controls import bind_song_check, bind_write_and_generate
from ...song_takes import bind_take_history
from .view import YuE2View


def bind_yue2_view(
    view: YuE2View,
    *,
    enhance_prompt: Callable[..., Any],
    generate: Callable[..., Any],
    take_facts: Callable[[Any], Any] = lambda _path: {},
) -> Any:
    writer_inputs = [
        view.style,
        view.prompt_model,
        view.api_key,
        view.lyrics,
        view.mode,
        view.duration,
        view.prompt_backend,
        view.openai_api_key,
        view.openai_model,
        view.vocals,
        view.language,
        view.use,
        view.lyrics_policy,
        view.notes,
        view.openai_base_url,
    ]
    bind_prompt_action(
        view.enhance.click,
        enhance_prompt,
        inputs=writer_inputs,
        outputs=[view.style, view.lyrics, view.enhance_status],
        show_progress="minimal",
        api_name="enhance_yue2_prompt",
    )
    bind_write_and_generate(
        view.write_generate, enhance_prompt, inputs=writer_inputs,
        drafts=[view.style, view.lyrics], status=view.enhance_status,
        review=view.enhance.h3_review, generate_id=view.run.elem_id,
    )
    bind_song_check(view, "YuE2", view.style, "Style")
    bind_take_history(
        view.takes, run=view.run, output=view.output, seed=view.seed, take_facts=take_facts,
        score=view.abc, mode=view.mode,
        last_score=view.last_score, reuse_last_score=view.reuse_score,
        drafts={
            "style": view.style, "lyrics": view.lyrics, "abc": view.abc, "mode": view.mode,
            "duration": view.duration, "vocals": view.vocals, "language": view.language,
            "use": view.use,
        },
    )
    return bind_gpu_action(
        view.run.click,
        owned_generation(generate, "yue2"),
        inputs=[
            view.model,
            view.style,
            view.lyrics,
            view.abc,
            view.mode,
            view.duration,
            view.seed,
            view.steps,
            view.cfg,
            view.temperature,
            view.top_p,
            view.top_k,
            view.repetition_penalty,
            view.max_abc_tokens,
            view.abc_temperature,
            view.abc_top_p,
            view.abc_top_k,
            view.abc_repetition_penalty,
            view.abc_penalty_window,
            view.tiled,
        ],
        outputs=[view.output, view.status],
        show_progress="minimal",
        api_name="generate_yue2",
    )
