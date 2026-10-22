"""Reusable Gradio event-binding helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .job_bindings import bind_gpu_action, bind_prompt_action, owned_generation
from .music_view import MusicView
from .song_controls import bind_song_check, bind_write_and_generate
from .song_takes import bind_take_history


def bind_music_view(
    view: MusicView,
    *,
    enhance_prompt: Callable[..., Any],
    generate: Callable[..., Any],
    take_facts: Callable[[Any], Any] = lambda _path: {},
) -> Any:
    writer_inputs = [
        view.caption,
        view.prompt_model,
        view.api_key,
        view.lyrics,
        *view.reference_images,
        view.prompt_backend,
        view.lightning_api_key,
        view.lightning_model,
        view.duration,
        view.vocals,
        view.language,
        view.use,
        view.lyrics_policy,
        view.notes,
    ]
    bind_prompt_action(
        view.enhance.click,
        enhance_prompt,
        inputs=writer_inputs,
        outputs=[view.caption, view.lyrics, view.enhance_status],
        show_progress="minimal",
        api_name="enhance_music3_prompt",
    )
    bind_write_and_generate(
        view.write_generate, enhance_prompt, inputs=writer_inputs,
        drafts=[view.caption, view.lyrics], status=view.enhance_status,
        review=view.enhance.h3_review, generate_id=view.run.elem_id,
    )
    bind_song_check(view, "MiniMax Music 3", view.caption)
    bind_take_history(
        view.takes, run=view.run, output=view.output, seed=view.seed, take_facts=take_facts,
        drafts={
            "caption": view.caption, "lyrics": view.lyrics, "duration": view.duration,
            "vocals": view.vocals, "language": view.language, "use": view.use,
        },
    )
    return bind_gpu_action(
        view.run.click,
        owned_generation(generate, "music"),
        inputs=[
            view.model,
            view.caption,
            view.lyrics,
            view.duration,
            view.seed,
            view.steps,
            view.cfg,
            view.ar_cfg,
            view.top_k,
            view.tiled,
        ],
        outputs=[view.output, view.status],
        show_progress="minimal",
        api_name="generate_music3",
    )
