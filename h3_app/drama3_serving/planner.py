"""Split a Drama 3 script into speaker-based chunks, each sent with its speaker's voice."""

from __future__ import annotations

import re
from dataclasses import dataclass

from h3_app.workflows.drama3 import SPEAKER_TAG

# About one or two sentences of speech; Fish Audio's own chunker uses 300 bytes.
CHUNK_BYTES = 400
CONTROL = re.compile(r"<\|[^|>]*\|>|\[[^\[\]]*\]|</?[A-Za-z][\w-]*>")
WIDE_CHAR = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]")
# A sentence ends at terminal punctuation outside a cue; cues never contain line breaks.
SENTENCE_END = re.compile(r"(?<=[.!?;。！？；])(?![^\[]*\])\s+")


@dataclass(frozen=True)
class Chunk:
    index: int
    speaker: int
    text: str  # speech and cues for one speaker, without speaker tags


def parse_turns(script: str) -> list[tuple[int, str]]:
    """Ordered (speaker, text) turns; text before the first tag is speaker 0."""
    parts = SPEAKER_TAG.split(str(script or ""))
    turns = [(0, parts[0])] + [(int(speaker), body) for speaker, body in zip(parts[1::2], parts[2::2])]
    return [(speaker, " ".join(body.split())) for speaker, body in turns if body.strip()]


def _pack(sentences: list[str], limit: int) -> list[str]:
    chunks, current = [], ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate.encode("utf-8")) > limit:
            chunks.append(current)
            candidate = sentence
        current = candidate
    return [*chunks, current] if current else chunks


def plan_chunks(script: str, limit: int = CHUNK_BYTES) -> list[Chunk]:
    """Merge consecutive lines of one speaker, then cut at sentence ends near ``limit``."""
    merged: list[tuple[int, str]] = []
    for speaker, text in parse_turns(script):
        if merged and merged[-1][0] == speaker:
            merged[-1] = (speaker, f"{merged[-1][1]} {text}")
        else:
            merged.append((speaker, text))
    pieces = [(speaker, piece) for speaker, text in merged
              for piece in _pack(SENTENCE_END.split(text), limit)]
    return [Chunk(index, speaker, text) for index, (speaker, text) in enumerate(pieces)]


def spoken_text(text: str) -> str:
    """The words a listener hears: cues and tags removed."""
    return " ".join(CONTROL.sub(" ", str(text or "")).split())


def expected_seconds(text: str) -> float:
    """Rough spoken length: ~2.5 words/s, ~4.5 CJK characters/s, ~0.5 s per cue."""
    cues = len(re.findall(r"\[[^\[\]]*\]", str(text or "")))
    spoken = spoken_text(text)
    wide = len(WIDE_CHAR.findall(spoken))
    words = len(WIDE_CHAR.sub(" ", spoken).split())
    return words / 2.5 + wide / 4.5 + cues * 0.5
