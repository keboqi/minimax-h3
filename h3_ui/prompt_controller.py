"""Prompt controller with explicit runtime dependencies."""

from __future__ import annotations

from typing import Any
from h3_app.catalog import (
    DEFAULT_LIGHTNING_PROMPT_MODEL,
    DEFAULT_IMAGE_FRAMES,
    DEFAULT_RESULT_FORMAT,
)
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptServices:
    JOBS: Any
    _runtime_config: Any
    gr: Any
    nullcontext: Any
    h3_writer: Any


class PromptController:
    def __init__(self, services: PromptServices):
        self.services = services

    def enhance_h3_prompt(
        self,
        prompt: str,
        backend: str,
        local_base_model: str,
        local_max_new_tokens: int,
        local_temperature: float,
        local_top_p: float,
        local_greedy: bool,
        local_seed: int,
        gemini_model: str,
        gemini_api_key: str,
        lightning_api_key: str,
        mode: str,
        first_image: Any,
        last_image: Any,
        ref_image_1: Any,
        ref_image_2: Any,
        ref_image_3: Any,
        ref_image_4: Any,
        ref_image_5: Any,
        ref_image_6: Any,
        ref_image_7: Any,
        ref_image_8: Any,
        ref_image_9: Any,
        ref_video_1: Any,
        ref_video_2: Any,
        ref_video_3: Any,
        ref_audio_1: Any,
        ref_audio_2: Any,
        ref_audio_3: Any,
        duration: float,
        width: int,
        height: int,
        result_format: str = DEFAULT_RESULT_FORMAT,
        image_frames: int = DEFAULT_IMAGE_FRAMES,
        fl2va_audio_1: Any = None,
        fl2va_audio_2: Any = None,
        fl2va_audio_3: Any = None,
        lightning_model: str = DEFAULT_LIGHTNING_PROMPT_MODEL,
    ) -> tuple[str, str]:
        lease = (
            self.services.JOBS.maintenance("prompt-enhance")
            if backend == "Local MiniMax-H3 8B"
            else self.services.nullcontext()
        )
        with lease:
            return self.services.h3_writer.enhance_h3_prompt(
                prompt,
                backend,
                local_base_model,
                local_max_new_tokens,
                local_temperature,
                local_top_p,
                local_greedy,
                local_seed,
                gemini_model,
                gemini_api_key,
                lightning_api_key,
                mode,
                first_image,
                last_image,
                ref_image_1,
                ref_image_2,
                ref_image_3,
                ref_image_4,
                ref_image_5,
                ref_image_6,
                ref_image_7,
                ref_image_8,
                ref_image_9,
                ref_video_1,
                ref_video_2,
                ref_video_3,
                ref_audio_1,
                ref_audio_2,
                ref_audio_3,
                duration,
                width,
                height,
                result_format,
                image_frames,
                fl2va_audio_1,
                fl2va_audio_2,
                fl2va_audio_3,
                lightning_model=lightning_model,
                runtime=self.services._runtime_config(),
            )

    def prompt_writer_backend_visibility(self, backend: str) -> tuple[Any, Any, Any]:
        """Show only controls belonging to the selected prompt-writer backend."""
        local_selected = backend == "Local MiniMax-H3 8B"
        return (
            self.services.gr.update(visible=local_selected),
            self.services.gr.update(visible=backend == "Gemini"),
            self.services.gr.update(visible=backend == "Lightning AI"),
        )
