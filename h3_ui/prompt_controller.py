"""Prompt controller with explicit runtime dependencies."""

from __future__ import annotations

from typing import Any
from h3_app.catalog import (
    DEFAULT_IMAGE_FRAMES,
    DEFAULT_RESULT_FORMAT,
    QWEN_EDIT_SIZE_MATCH,
)
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptServices:
    H3Error: Any
    JOBS: Any
    Path: Any
    QWEN_EDIT_SIZE_MANUAL: Any
    QWEN_EDIT_SIZE_MATCH: Any
    QWEN_EDIT_SIZE_MAX: Any
    _runtime_config: Any
    gr: Any
    nullcontext: Any
    prompt_service: Any
    qwen_generation: Any
    qwen_edit_size_flags: Any


class PromptController:
    def __init__(self, services: PromptServices):
        self.services = services

    def enhance_music3_prompt(
        self,
        prompt: str,
        model: str,
        temporary_api_key: str,
        lyrics: str,
        ref_image_1: Any,
        ref_image_2: Any,
        ref_image_3: Any,
        backend: str = "Lightning AI",
        lightning_api_key: str = "",
    ) -> tuple[str, str, str]:
        return self.services.prompt_service.enhance_music3_prompt(
            prompt,
            model,
            temporary_api_key,
            lyrics,
            ref_image_1,
            ref_image_2,
            ref_image_3,
            backend=backend,
            lightning_api_key=lightning_api_key,
            runtime=self.services._runtime_config(),
        )

    def enhance_ltx25_prompt(
        self,
        prompt: str,
        model: str,
        temporary_api_key: str,
        mode: str,
        start_image: Any,
        middle_image: Any,
        end_image: Any,
        duration: float,
        width: int,
        height: int,
        backend: str = "Lightning AI",
        lightning_api_key: str = "",
    ) -> tuple[str, str]:
        return self.services.prompt_service.enhance_ltx25_prompt(
            prompt,
            model,
            temporary_api_key,
            mode,
            start_image,
            middle_image,
            end_image,
            duration,
            width,
            height,
            backend=backend,
            lightning_api_key=lightning_api_key,
            runtime=self.services._runtime_config(),
        )

    def qwen_edit_size_flags(self, edit_size: str) -> tuple[bool, bool]:
        """Map the mutually exclusive UI choice onto the existing request fields."""
        if edit_size not in {
            self.services.QWEN_EDIT_SIZE_MATCH,
            self.services.QWEN_EDIT_SIZE_MAX,
            self.services.QWEN_EDIT_SIZE_MANUAL,
        }:
            raise self.services.H3Error("Choose a valid Qwen edit output size.")
        return (
            edit_size == self.services.QWEN_EDIT_SIZE_MATCH,
            edit_size == self.services.QWEN_EDIT_SIZE_MAX,
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
        backend: str = "Lightning AI",
        lightning_api_key: str = "",
        edit_size: str = QWEN_EDIT_SIZE_MATCH,
    ) -> tuple[str, str]:
        match_input_size, max_resolution = self.services.qwen_edit_size_flags(edit_size)
        if (match_input_size or max_resolution) and str(
            mode
        ).strip().lower() == "image edit":
            uploaded = reference_images or []
            if isinstance(uploaded, (str, self.services.Path)):
                uploaded = [uploaded]
            if uploaded:
                source_width, source_height = (
                    self.services.qwen_generation.first_reference_dimensions(
                        str(uploaded[0])
                    )
                )
                width, height = (
                    self.services.qwen_generation.max_qwen_edit_dimensions(
                        source_width, source_height
                    )
                    if max_resolution
                    else (source_width, source_height)
                )
        return self.services.prompt_service.enhance_qwen_image21_prompt(
            prompt,
            model,
            temporary_api_key,
            mode,
            reference_images,
            width,
            height,
            backend=backend,
            lightning_api_key=lightning_api_key,
            runtime=self.services._runtime_config(),
        )

    def enhance_yue2_prompt(
        self,
        style: str,
        model: str,
        temporary_api_key: str,
        lyrics: str,
        mode: str,
        duration: float,
        backend: str = "Lightning AI",
        lightning_api_key: str = "",
    ) -> tuple[str, str, str]:
        return self.services.prompt_service.enhance_yue2_prompt(
            style,
            model,
            temporary_api_key,
            lyrics,
            mode,
            duration,
            backend=backend,
            lightning_api_key=lightning_api_key,
            runtime=self.services._runtime_config(),
        )

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
    ) -> tuple[str, str]:
        lease = (
            self.services.JOBS.maintenance("prompt-enhance")
            if backend == "Local MiniMax-H3 8B"
            else self.services.nullcontext()
        )
        with lease:
            return self.services.prompt_service.enhance_h3_prompt(
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
