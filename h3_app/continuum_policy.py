"""Sequence-mode limits, prompt compilation and upstream report parsing.

Sequence mode runs H3 Continuum from its first chunk. Its profile is separate
from Clip presets: acceleration patches, alternate VAEs and refinement are not
carried over until each combination is validated for saved-chunk reuse.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from h3_app.errors import H3Error

PROMPT_COMPILER_VERSION = 1

# Provisional launch limits; upstream accepts 1-16 chunks of 4-30 seconds, but
# memory for decode/assembly has only been budgeted for this envelope.
MAX_CHUNKS = 4
MAX_TOTAL_SECONDS = 30.0
MIN_CHUNK_SECONDS = 4.0
MAX_CHUNK_SECONDS = 10.0
DEFAULT_CHUNK_SECONDS = 5.0
MAX_REFERENCE_IMAGES = 9

SEQUENCE_MODES = ("Text to video", "First / last frame", "Reference media")

# Sequence owns a conservative recipe: Normal sampling, native attention and
# VAE, and sizes whose decode/assembly memory fits the launch limits.
SEQUENCE_SIZES = {
    "768×448 · 16:9": (768, 448),
    "1024×576 · 16:9": (1024, 576),
    "512×512 · square": (512, 512),
    "448×768 · 9:16": (448, 768),
}
DEFAULT_SEQUENCE_SIZE = "768×448 · 16:9"
DEFAULT_SEQUENCE_STEPS = 20
MIN_SEQUENCE_STEPS = 10
MAX_SEQUENCE_STEPS = 40
SEQUENCE_SCHEDULER = "simple"

# App label -> upstream continuity option.
CONTINUITY_CHOICES = {
    "Balanced": "Balanced — 22 frames",
    "Fast": "Fast — 5 frames",
    "Strong (experimental)": "Strong — 39 frames (Experimental)",
}
DEFAULT_CONTINUITY = "Balanced"

# Upstream review-control vocabulary (v3.9.8).
GENERATION_MODE_REVIEW = "Review Each Chunk"
REVIEW_CONTINUE = "Continue / Next"
REVIEW_REGENERATE_CURRENT = "Regenerate Current"
TAKE_AUTOMATIC = "Automatic"
TAKE_USE = "Use This Take"
TAKE_CONTINUE = "Continue From Here"
RUN_STORAGE_SAVE = "Save + Auto Resume"
REFERENCE_USE_ALL = "All chunks"
REFERENCE_USE_PER_CHUNK = "Per chunk"

CONTINUUM_SAMPLER_NODE = "H3ContinuumSamplerV39"
CONTINUUM_REFERENCES_NODE = "H3ContinuumReferenceImagesV39"
CONTINUUM_FINALIZE_NODE = "H3ContinuumAssembleSeamV35"
CONTINUUM_REQUIRED_NODES = frozenset(
    {
        CONTINUUM_SAMPLER_NODE,
        CONTINUUM_REFERENCES_NODE,
        CONTINUUM_FINALIZE_NODE,
        "PreviewAny",
    }
)


def normalize_text(value: str | None) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def compile_prompts(shared: str, actions: Sequence[str], chunk_count: int) -> tuple[str, ...]:
    """Return one effective prompt per chunk: shared scene, blank line, action.

    An empty action inherits the shared prompt. The result is serialized as an
    explicit JSON list, so delimiter-like text in prompts has no meaning.
    """
    if not 1 <= int(chunk_count) <= MAX_CHUNKS:
        raise H3Error(f"Sequence supports 1–{MAX_CHUNKS} chunks.")
    if len(actions) != int(chunk_count):
        raise H3Error(
            f"The sequence has {len(actions)} chunk prompt(s) for {chunk_count} chunk(s)."
        )
    base = normalize_text(shared)
    compiled = []
    for index, action in enumerate(actions, start=1):
        text = normalize_text(action)
        effective = "\n\n".join(part for part in (base, text) if part)
        if not effective:
            raise H3Error(
                f"Chunk {index} has no prompt. Write a shared prompt or a chunk action."
            )
        compiled.append(effective)
    return tuple(compiled)


def serialize_prompts(prompts: Sequence[str]) -> str:
    return json.dumps(list(prompts), ensure_ascii=False)


def chunk_selector(chunks: Iterable[int]) -> str:
    values = sorted({int(chunk) for chunk in chunks})
    if any(not 1 <= chunk <= 16 for chunk in values):
        raise H3Error("Reference chunk assignments must name chunks 1–16.")
    return ",".join(map(str, values)) or "off"


def sequence_issues(
    *,
    mode: str,
    chunk_count: int,
    chunk_seconds: float,
    width: int,
    height: int,
    has_first_image: bool,
    has_last_image: bool,
    reference_images: int,
    reference_videos: int,
    reference_audios: int,
) -> list[str]:
    """Explain each Clip setting Sequence cannot honour, before any GPU work."""
    issues = []
    if mode not in SEQUENCE_MODES:
        issues.append(f"Sequence does not support {mode} conditioning.")
    if not 1 <= int(chunk_count) <= MAX_CHUNKS:
        issues.append(f"Sequence supports 1–{MAX_CHUNKS} chunks.")
    seconds = float(chunk_seconds)
    if not MIN_CHUNK_SECONDS <= seconds <= MAX_CHUNK_SECONDS:
        issues.append(
            f"Chunk length must be {MIN_CHUNK_SECONDS:g}–{MAX_CHUNK_SECONDS:g} seconds."
        )
    if int(chunk_count) * seconds > MAX_TOTAL_SECONDS + 1e-6:
        issues.append(f"Sequence total is limited to {MAX_TOTAL_SECONDS:g} seconds.")
    if int(width) % 32 or int(height) % 32 or min(int(width), int(height)) < 256:
        issues.append("Sequence width and height must be multiples of 32 and at least 256.")
    if mode == "First / last frame":
        if not has_first_image:
            issues.append("Sequence first-frame mode needs a first image.")
        if has_last_image:
            issues.append(
                "A last frame is not yet validated for multi-chunk Sequence; clear it or use Clip."
            )
    if mode == "Reference media":
        if not reference_images:
            issues.append("Sequence reference mode needs at least one reference image.")
        if reference_videos or reference_audios:
            issues.append(
                "Sequence uses reference images only; reference video/audio stays in Clip."
            )
    return issues


def sequence_notes(
    *,
    generation_mode: str,
    attention_mode: str,
    cache_mode: str,
    latent_upscale: bool,
    postprocess: str,
    alternate_decoder: bool,
    semantic_bridge: bool,
) -> list[str]:
    notes = []
    if str(generation_mode).strip().lower() == "turbo":
        notes.append(
            "Turbo is not applied: it is not yet validated for saved-chunk reuse, "
            "so Sequence samples normally with its own step count."
        )
    if str(attention_mode).strip().lower() not in {"", "kitchen", "auto", "comfy kitchen"}:
        notes.append(f"{attention_mode} attention is not applied (native attention).")
    if str(cache_mode).strip().lower() != "off":
        notes.append(f"{cache_mode} cache is not applied.")
    if latent_upscale:
        notes.append("Latent upscale is not applied.")
    if str(postprocess) not in {"", "None"}:
        notes.append(f"{postprocess} finishing is not applied.")
    if alternate_decoder:
        notes.append("Alternate VAE decoders are not applied (native VAE).")
    if semantic_bridge:
        notes.append("Semantic Bridge is not applied.")
    return notes


@dataclass(frozen=True)
class RunStorageReport:
    reused: int
    generated: int
    total: int
    resume: str
    reuse_disabled: str | None


_STATUS = re.compile(
    r"Run Storage: [^\n;]*; (?P<reused>\d+) reused, (?P<generated>\d+) generated, "
    r"(?P<total>\d+) total; resume=(?P<resume>[^;]+);(?P<rest>[^\n]*)"
)
_DISABLED = re.compile(r"auto-resume disabled: (?P<reasons>.*?)\.(?:\s|$)")


def parse_run_storage(status: str) -> RunStorageReport | None:
    """Read the reuse counters upstream prints on the sampler status output."""
    matches = list(_STATUS.finditer(str(status or "")))
    if not matches:
        return None
    match = matches[-1]
    disabled = _DISABLED.search(match["rest"])
    return RunStorageReport(
        reused=int(match["reused"]),
        generated=int(match["generated"]),
        total=int(match["total"]),
        resume=match["resume"].strip(),
        reuse_disabled=disabled["reasons"] if disabled else None,
    )
