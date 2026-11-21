"""Render one prepared lip-sync window from a completed video dub."""

from __future__ import annotations

import math
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator
from uuid import uuid4

import numpy as np

from h3_app.drama3.serving.audio import encode_wav
from h3_app.errors import H3Error
from h3_app.generation.services import GenerationServices
from h3_app.jobs import check_cancelled
from h3_app.workflows.lip_sync import FPS, MAX_FRAMES, build_lip_sync_graph, validate_lip_sync_window

from .alignment import FITTED_AUDIO_RATE, open_alignment


def fitted_dialogue(video: Path, speaker: str, first_frame: int, frames: int) -> np.ndarray:
    """Rebuild only this speaker's fitted speech on a window of the dub timeline."""
    if first_frame < 0 or not 1 <= frames <= MAX_FRAMES:
        raise H3Error("Choose a valid lip-sync window on the 24 fps render timeline.")
    if video.suffix.lower() != ".mp4" or not video.is_file():
        raise H3Error("Choose a completed dubbed MP4.")
    first = round(first_frame * FITTED_AUDIO_RATE / FPS)
    stop = round((first_frame + frames) * FITTED_AUDIO_RATE / FPS)
    dialogue = np.zeros(stop - first, dtype=np.float32)
    try:
        with open_alignment(video) as (saved, clips):
            lines = saved["lines"]
            for key, (start, end) in saved["spoken"].items():
                index = int(key)
                if not 0 <= index < len(lines):
                    raise ValueError("Invalid fitted line index")
                if lines[index]["speaker"] != speaker:
                    continue
                if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                    raise ValueError("Invalid fitted line timing")
                clip_start, clip_stop = round(start * FITTED_AUDIO_RATE), round(end * FITTED_AUDIO_RATE)
                left, right = max(first, clip_start), min(stop, clip_stop)
                if left >= right:
                    continue
                clip = clips[key]
                if clip.ndim != 1 or len(clip) != clip_stop - clip_start or not np.isfinite(clip).all():
                    raise ValueError("Invalid fitted dialogue samples")
                dialogue[left - first:right - first] += clip[left - clip_start:right - clip_start]
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise H3Error("Saved fitted dialogue is missing or unreadable. Generate this dub again.") from exc
    if not dialogue.any():
        raise H3Error("This window contains no audible fitted dialogue for the selected speaker.")
    return dialogue


def render_lip_sync_preview(*, dub: Path, speaker: str, first_frame: int, crop: Path, mask: Path,
                            services: GenerationServices, refinement: dict[str, Any] | None = None,
                            strength: float = 0.4) -> Iterator[str | Path]:
    """Run the first local prototype on a supplied 24 fps face crop and grayscale mask.

    The crop and mask start at ``first_frame`` on the dub's 24 fps timeline, share
    its H3-grid frame count, and use the same crop coordinates. The output is a
    crop preview with fitted speech. Call within a GPU job when integrating the UI.
    """
    yield "Checking the lip-sync window"
    check_cancelled()
    picture = services.media.probe_video_metadata(crop)
    matte = services.media.probe_video_metadata(mask)
    width, height, frames = picture.width, picture.height, picture.frame_count
    validate_lip_sync_window(width, height, frames, strength)
    if (picture.fps != FPS or matte.fps != FPS
            or (matte.width, matte.height, matte.frame_count) != (width, height, frames)):
        raise H3Error("The face crop and mouth mask must have matching dimensions and frame counts at 24 fps.")
    dialogue = fitted_dialogue(dub, speaker, first_frame, frames)
    models = services.models.load_model_config()
    config = {"model_profile": models.default_profile, **(refinement or {}), "_source_duration": frames / FPS}
    client_id = uuid4().hex
    with TemporaryDirectory(prefix="h3-lip-sync-") as directory:
        wav = Path(directory) / "dialogue.wav"
        wav.write_bytes(encode_wav(dialogue, FITTED_AUDIO_RATE))
        staged_dialogue = services.media.stage_file(str(wav), "lip_sync")
    graph = build_lip_sync_graph(
        crop=services.media.stage_file(str(crop), "lip_sync"),
        mask=services.media.stage_file(str(mask), "lip_sync"), dialogue=staged_dialogue,
        width=width, height=height, frames=frames, models=models, refinement=config,
        output_prefix=f"dubbing/.lip-sync-preview/{client_id}", strength=strength,
    )
    yield "Rendering the H3 mouth patch"
    check_cancelled()
    queued_at = time.time()
    prompt_id = services.execution.submit_prompt(graph, client_id)
    for stage, _completed, _total, _step, _step_total in services.execution.poll_comfy_progress(prompt_id, graph):
        check_cancelled()
        yield stage
    output = services.media.resolve_output(services.execution.wait_for_history(prompt_id), queued_at)
    check_cancelled()
    if not output.name.startswith(f"{client_id}_"):
        raise H3Error("The render finished, but its lip-sync preview could not be located.")
    yield output
