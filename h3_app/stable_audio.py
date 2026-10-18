"""Stable Audio 3 Medium sounds from the shared resident audio tools worker.

Weights persist under `<comfy_dir>/models/stable_audio3/medium`; the model stays
loaded between requests (see :mod:`h3_app.audio_tools`).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import uuid

from h3_app import audio_tools
from h3_app.errors import H3Error

MODEL_ID = audio_tools.layout.STABLE_AUDIO_ID
DEFAULT_DURATION_SECONDS = 5
DEFAULT_LOOP_DURATION_SECONDS = 10
MAX_DURATION_SECONDS = audio_tools.layout.MAX_SOUND_SECONDS


def validate_sound_inputs(description: str, duration: int | float = DEFAULT_DURATION_SECONDS) -> tuple[str, int]:
    cleaned = str(description or "").strip()
    if not cleaned or len(cleaned) > 2000:
        raise H3Error("Describe the sound effect or music in 1–2000 characters.")
    try:
        seconds = int(round(float(duration or DEFAULT_DURATION_SECONDS)))
    except (TypeError, ValueError) as exc:
        raise H3Error("Sound duration must be a number of seconds between 1 and 47.") from exc
    if seconds < 1 or seconds > MAX_DURATION_SECONDS:
        raise H3Error(f"Sound duration must be between 1 and {MAX_DURATION_SECONDS} seconds.")
    return cleaned, seconds


def cached_sound_path(output_dir: Path, description: str, duration: int) -> Path:
    digest = hashlib.sha256(f"{description.strip().lower()}|{int(duration)}".encode("utf-8")).hexdigest()[:16]
    return output_dir / f"StableAudio3_cache_{digest}.wav"


def generate_sound_references(
    items: list[tuple[str, int | float] | dict],
    runtime,
    *,
    free_gpu,
    use_cache: bool = False,
    hf_token: str | None = None,
) -> list[Path]:
    """Generate one or more sound clips in a single Stable Audio 3 Medium worker run."""
    if not items:
        return []
    validated = [
        validate_sound_inputs(
            item.get("description") or item.get("prompt") or "",
            item.get("duration", 5),
        )
        if isinstance(item, dict)
        else validate_sound_inputs(item[0], item[1])
        for item in items
    ]
    output_dir = runtime.output_dir / "drama3_sounds"
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[Path | None] = [None] * len(validated)
    worker_items: list[dict] = []
    worker_indices: list[int] = []
    created_paths: list[Path] = []

    for idx, (desc, dur) in enumerate(validated):
        if use_cache:
            cached = cached_sound_path(output_dir, desc, dur)
            if cached.is_file() and cached.stat().st_size > 44:
                results[idx] = cached
                continue
            target = cached
        else:
            token = uuid.uuid4().hex
            target = output_dir / f"StableAudio3_{token}.wav"
        created_paths.append(target)
        worker_indices.append(idx)
        worker_items.append(
            {
                "prompt": desc,
                "duration": dur,
                "seed": -1,
                "steps": 8,
                "cfg_scale": 1.0,
                "output": str(target.resolve()),
            }
        )

    if not worker_items:
        return [path for path in results if path is not None]

    try:
        # The model and its bundled T5Gemma encoder are gated; weights download on first use.
        audio_tools.run("stable_audio", {"items": worker_items}, runtime, free_gpu=free_gpu, hf_token=hf_token)
        for idx, item in zip(worker_indices, worker_items):
            out_path = Path(item["output"])
            if not out_path.is_file() or out_path.stat().st_size <= 44:
                raise H3Error("Stable Audio 3 Medium returned no audio.")
            results[idx] = out_path
    except BaseException:
        for path in created_paths:
            path.unlink(missing_ok=True)
        raise
    return [path for path in results if path is not None]


def generate_sound_reference(
    description: str,
    duration: int | float,
    runtime,
    *,
    free_gpu,
    use_cache: bool = False,
    hf_token: str | None = None,
) -> Path:
    """Generate a single sound clip with Stable Audio 3 Medium."""
    paths = generate_sound_references(
        [(description, duration)],
        runtime,
        free_gpu=free_gpu,
        use_cache=use_cache,
        hf_token=hf_token,
    )
    return paths[0]
