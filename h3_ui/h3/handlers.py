"""MiniMax H3's Gradio callbacks: batched generation, the default-settings API, settings summaries and help."""

from __future__ import annotations

import html
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import gradio as gr

from h3_models import DEFAULT_LTX25_MODEL
from h3_app.catalog import (
    DEFAULT_RESULT_FORMAT,
    DEFAULT_IMAGE_FRAMES,
    DEFAULT_IMAGE_VAE,
    H3_LATENT_UPSCALE_STANDARD,
    MAX_REFERENCE_AUDIOS,
    MAX_REFERENCE_IMAGES,
    MAX_REFERENCE_VIDEOS,
    MAX_VIDEO_BATCH_COUNT,
    MIN_VIDEO_BATCH_COUNT,
    UI_DEFAULTS,
)
from h3_app.config import RuntimeConfig
from h3_app.contracts import GenerationArguments
from h3_app.errors import H3Error
from h3_app.jobs import CURRENT_JOB
from h3_app.policy import normalize_paths, normalize_result_format
from h3_app.provenance import read_snapshot, render_snapshot
from h3_app.settings import GenerationRequest, ResolutionContext, resolve_settings
from h3_ui.settings_presentation import render_settings

from .layout import image_frame_labels, start_frame_generation_resolution


@dataclass(frozen=True)
class H3Services:
    """The runtime, the full H3 generator and the public download link for a finished video."""

    runtime_config: Callable[[], RuntimeConfig]
    generate: Callable[..., Iterator[tuple[Any, str]]]
    download_url: Callable[[Any, gr.Request], str]


def video_batch_seeds(seed: int, batch_count: int) -> list[int]:
    """Resolve distinct seeds while preserving the single-run random-seed behavior."""
    count = int(batch_count)
    if not MIN_VIDEO_BATCH_COUNT <= count <= MAX_VIDEO_BATCH_COUNT:
        raise H3Error(
            f"Video batch count must be between {MIN_VIDEO_BATCH_COUNT} and {MAX_VIDEO_BATCH_COUNT}."
        )
    base_seed = int(seed)
    if count == 1:
        return [base_seed]
    return random.sample(range(0, 2**63 - 1), count)


def reference_prompt_help() -> str:
    return f"Use `<Picture 1>` through `<Picture {MAX_REFERENCE_IMAGES}>`, `<Video 1>` through `<Video {MAX_REFERENCE_VIDEOS}>`, and `<Audio 1>` through `<Audio {MAX_REFERENCE_AUDIOS}>` in the prompt."


def mode_help(mode: str) -> str:
    if mode == "Reference media":
        return (
            "Reference tags are ordered as images, then videos, then standalone audio. "
            + reference_prompt_help()
        )
    if mode == "First / last frame":
        return "Upload a first frame, a last frame, or both. This uses the FL2VA model."
    return "Prompt-only generation using FL2VA with native stereo audio."


def api_guide() -> str:
    defaults = UI_DEFAULTS
    return f"""## Generate through the API\n\nThe `/generate_video` endpoint accepts a prompt and preserves the existing **Video** defaults from the **MiniMax H3** tab, including default-on reuse of unchanged prompt/media conditioning:\n\n`{defaults["mode"]}` · `{defaults["model_profile"]}` · `{defaults["generation_mode"]} / {defaults["turbo_variant"]}` · `{defaults["duration"]}s` · `{defaults["width"]}×{defaults["height"]}` · `{defaults["steps"]} steps` · `{defaults["scheduler"]}` scheduler · random seed\n\nInstall the client and submit a job:\n\n```bash\npip install gradio_client\n```\n\n```python\nfrom gradio_client import Client\n\nclient = Client("http://127.0.0.1:7860")\ndownload_url, status = client.predict(\n    "A cinematic tracking shot through a rain-soaked neon city",\n    api_name="/generate_video",\n)\nprint(download_url)\nprint(status)\n```\n\n`download_url` is an HTTP URL served by this app, so it can be opened in a browser or downloaded with `curl -L -O` while the app is running.\n\nFor every control exposed by the MiniMax H3 tab, including **Image** and **Audio** result formats, use `/generate_video_advanced` and inspect the app's [OpenAPI schema](/gradio_api/openapi.json) for its current parameter list. Image selections can be persisted through `/save_h3_image_frames`. API requests share the same single-job queue as the UI.\n"""


class H3Handlers:
    """Callbacks bound to one application's runtime and generator."""

    def __init__(self, services: H3Services):
        self.services = services

    def generate_with_ui_defaults(
        self, prompt: str, request: gr.Request, progress=gr.Progress(track_tqdm=False)
    ):
        """Generate with UI defaults and return a reusable public download URL."""
        values = dict(UI_DEFAULTS)
        values.setdefault("sol_step_off", 0.0)
        values.setdefault("sol_sink_tokens", 0)
        values.setdefault("ltx25_model", DEFAULT_LTX25_MODEL)
        values.update(
            {f"ref_image_{index}": None for index in range(1, MAX_REFERENCE_IMAGES + 1)},
            **{f"ref_video_{index}": None for index in range(1, MAX_REFERENCE_VIDEOS + 1)},
            **{f"ref_audio_{index}": None for index in range(1, MAX_REFERENCE_AUDIOS + 1)},
            prompt=prompt,
            first_image=None,
            last_image=None,
        )
        for video, status in self.services.generate(**values, progress=progress):
            download_url = (
                self.services.download_url(video, request) if video is not None else None
            )
            yield (download_url, status)

    def generate_for_ui(self, batch_count: int, *args: Any):
        """Adapt shared generation to UI outputs, including 1-4 video variants."""
        arguments = GenerationArguments.from_positional(args)
        result_format = normalize_result_format(arguments.values["result_format"])
        count = int(batch_count) if result_format == "Video" else 1
        job = CURRENT_JOB.get()
        seeds = (
            list(job.replay_seeds)
            if job is not None and job.replay_seeds
            else (
                video_batch_seeds(int(arguments.values["seed"]), count)
                if result_format == "Video"
                else [int(arguments.values["seed"])]
            )
        )
        indices = list(range(count))
        if job is not None:
            if job.replay_seeds:
                seeds = list(job.replay_seeds)
                indices = list(job.replay_indices)
                count = len(seeds)
            else:
                seeds = [
                    random.randrange(0, 2**63 - 1) if seed < 0 else seed for seed in seeds
                ]
                job.variant_seeds.update(enumerate(seeds))
        first_update = True
        unchanged = (gr.update(),) * 7
        for batch_index, batch_seed in enumerate(seeds):
            if job is not None:
                job.variant = indices[batch_index]
                job.check()
            batch_arguments = arguments.with_seed(batch_seed)
            prefix = f"Video {batch_index + 1}/{count} · " if count > 1 else ""
            for result, status in self.services.generate(**batch_arguments.values):
                status = prefix + status
                video_updates = [gr.update() for _ in range(MAX_VIDEO_BATCH_COUNT)]
                if first_update:
                    first_update = False
                    video_updates = [
                        gr.update(
                            value=None, visible=result_format == "Video" and index < count
                        )
                        for index in range(MAX_VIDEO_BATCH_COUNT)
                    ]
                    yield (
                        *video_updates,
                        gr.update(visible=result_format == "Image"),
                        gr.update(value=[]),
                        gr.update(choices=[], value=[]),
                        [],
                        gr.update(value=None),
                        gr.update(value=None, visible=result_format == "Audio"),
                        gr.update(value=""),
                        status,
                    )
                    if result is None:
                        continue
                if result is None:
                    yield (*video_updates, *unchanged, status)
                    continue
                if result_format == "Image":
                    paths = normalize_paths(result)
                    labels = image_frame_labels(paths)
                    gallery = [(path, label) for path, label in zip(paths, labels)]
                    yield (
                        *video_updates,
                        gr.update(visible=True),
                        gr.update(value=gallery),
                        gr.update(choices=labels, value=[]),
                        paths,
                        gr.update(value=None),
                        gr.update(),
                        gr.update(),
                        status,
                    )
                elif result_format == "Audio":
                    yield (
                        *video_updates,
                        *unchanged[:5],
                        gr.update(value=result, visible=True),
                        gr.update(),
                        status,
                    )
                else:
                    video_updates[batch_index] = gr.update(value=result, visible=True)
                    yield (*video_updates, *unchanged, status)

    def compact_settings_summary(
        self,
        mode: str,
        model_profile: str,
        text_encoder: str,
        stage_model_offload: bool,
        reuse_unchanged_inputs: bool,
        use_int8_vae: bool,
        generation_mode: str,
        turbo_variant: str,
        duration: float,
        width: int,
        height: int,
        steps: int,
        scheduler: str,
        attention_mode: str,
        sla_preset: str,
        cache_mode: str,
        latent_upscale: bool,
        latent_upscaler_model: str,
        latent_upscale_refine_steps: int,
        postprocess: str,
        seedvr2_model: str,
        ltx25_model: str,
        force_offload: bool,
        split_upscale: bool,
        split_seconds: float,
        result_format: str = DEFAULT_RESULT_FORMAT,
        image_frames: int = DEFAULT_IMAGE_FRAMES,
        image_vae: str = DEFAULT_IMAGE_VAE,
        latent_upscale_method: str = H3_LATENT_UPSCALE_STANDARD,
        latent_split_tile_width: int = 512,
        latent_split_tile_height: int = 512,
        latent_split_overlap_ratio: float = 0.25,
        latent_split_fade_ratio: float = 0.5,
        latent_split_chunk_frames: int = 73,
        latent_split_temporal_overlap_frames: int = 22,
        latent_split_seam_denoise: float = 0.75,
        latent_split_seam_polish: str = "off",
        use_trt_vae: bool = False,
        use_lynnreal_vae: bool = False,
        video_encoder: str = "Original",
        lightvae_decoder: str = "none",
    ) -> str:
        values = dict(locals())
        del values["self"]
        return self.describe_settings(values)

    def result_settings_for_media(self, value):
        def media_path(item):
            if isinstance(item, dict):
                return media_path(item.get("path") or item.get("name") or item.get("image"))
            if isinstance(item, (tuple, list)):
                return media_path(item[0]) if item else None
            return item if isinstance(item, (str, Path)) else None

        paths = []
        items = value if isinstance(value, list) else [value]
        for item in items:
            path = media_path(item)
            if path:
                paths.append(self.resolve_result_path(path))
        return render_snapshot(paths)

    def resolve_result_path(self, path):
        """The managed output behind ``path``, which may be Gradio's cached copy of it.

        Gradio hands listeners a copy in its own cache, without the settings snapshot;
        a name with a 32-hex id identifies the original under the output folders.
        """
        if read_snapshot(path):
            return path
        name = Path(path).name
        if re.search("[0-9a-f]{32}", name):
            runtime = self.services.runtime_config()
            # Modal exposes the same output directory through both a symlink and
            # its volume path. Overlapping roots can also find the same file twice.
            candidates = {
                candidate.resolve()
                for root in (runtime.output_dir, runtime.outputs_dir)
                for candidate in root.rglob(name)
                if candidate.resolve().is_relative_to(root.resolve())
                and read_snapshot(candidate)
            }
            if len(candidates) == 1:
                return next(iter(candidates))
        return path

    def resolve_request_settings(self, values: dict):
        request = GenerationRequest.from_values(values)
        dimensions = None
        first = values.get("first_image", values.get("first"))
        if (
            request.output.result_format == "Image"
            and request.mode == "First / last frame"
            and first
        ):
            dimensions = start_frame_generation_resolution(
                first, alignment=64 if request.finishing.latent_upscale else 32
            )
        return resolve_settings(
            request, ResolutionContext(dimensions, self.services.runtime_config().memory_profile)
        )

    def describe_settings(self, values: dict) -> str:
        try:
            return render_settings(self.resolve_request_settings(values), values)
        except (ValueError, TypeError, H3Error) as exc:
            return (
                '<div role="alert">Check generation settings: '
                + html.escape(str(exc))
                + "</div>"
            )
