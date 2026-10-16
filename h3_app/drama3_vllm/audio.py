"""WAV decoding, take checks and the joiner for chunked Drama 3 audio (numpy only)."""

from __future__ import annotations

import io
import struct
import wave
from dataclasses import dataclass

import numpy as np

SILENCE_DBFS = -45.0
FRAME_SECONDS = 0.02
EDGE_PAD_SECONDS = 0.08
MAX_INTERNAL_GAP = 1.0  # longer silences inside a take are shortened ...
KEPT_GAP = 0.5  # ... to this
TURN_GAP = 0.3  # between speakers
SAME_SPEAKER_GAP = 0.12  # between chunks of one speaker


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Mono float32 samples from PCM16/24/32 or float32 WAV, including WAVE_FORMAT_EXTENSIBLE."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("vLLM-Omni did not return a WAV file.")
    position, fmt, payload = 12, None, None
    while position + 8 <= len(data):
        tag, size = data[position:position + 4], struct.unpack("<I", data[position + 4:position + 8])[0]
        body = data[position + 8:position + 8 + size]
        if tag == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16]) + ((struct.unpack("<H", body[24:26])[0],) if len(body) >= 26 else (None,))
        elif tag == b"data":
            payload = body
        position += 8 + size + (size & 1)
    if fmt is None or payload is None:
        raise ValueError("The WAV file has no format or data chunk.")
    format_tag, channels, rate, _byte_rate, _align, bits, sub_format = fmt
    if format_tag == 0xFFFE:
        format_tag = sub_format
    if format_tag == 3 and bits == 32:
        samples = np.frombuffer(payload, dtype="<f4").astype(np.float32)
    elif format_tag == 1 and bits == 16:
        samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    elif format_tag == 1 and bits == 32:
        samples = np.frombuffer(payload, dtype="<i4").astype(np.float32) / 2147483648.0
    elif format_tag == 1 and bits == 24:
        raw = np.frombuffer(payload[: len(payload) // 3 * 3], dtype=np.uint8).reshape(-1, 3)
        ints = (raw[:, 0].astype(np.int32) | (raw[:, 1].astype(np.int32) << 8) | (raw[:, 2].astype(np.int32) << 16))
        samples = (np.where(ints >= 1 << 23, ints - (1 << 24), ints)).astype(np.float32) / 8388608.0
    else:
        raise ValueError(f"Unsupported WAV encoding (format {format_tag}, {bits} bits).")
    if channels > 1:
        samples = samples[: len(samples) // channels * channels].reshape(-1, channels).mean(axis=1)
    return samples, int(rate)


def encode_wav(samples: np.ndarray, rate: int) -> bytes:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(rate)
        target.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _silent_frames(samples: np.ndarray, rate: int) -> np.ndarray:
    size = max(1, int(rate * FRAME_SECONDS))
    frames = samples[: len(samples) // size * size].reshape(-1, size)
    if not len(frames):
        return np.zeros(0, dtype=bool)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    return rms < 10 ** (SILENCE_DBFS / 20)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) frame ranges where ``mask`` holds."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


@dataclass(frozen=True)
class TakeCheck:
    ok: bool
    reason: str
    seconds: float
    voiced_seconds: float
    score: float  # lower is better; ranks failed takes when every retry fails


def check_take(samples: np.ndarray, rate: int, expected_seconds: float) -> TakeCheck:
    """Reject takes that are mostly silent or far too short/long for their text."""
    silent = _silent_frames(samples, rate)
    seconds = len(samples) / rate
    voiced = float((~silent).sum()) * FRAME_SECONDS
    expected = max(0.8, expected_seconds)
    ratio = voiced / expected
    score = abs(np.log(max(ratio, 1e-3)))
    if voiced < 0.3:
        return TakeCheck(False, "no speech", seconds, voiced, 99.0)
    if voiced < 0.35 * expected - 1.0:
        return TakeCheck(False, f"too short ({voiced:.1f}s of speech for ~{expected:.1f}s of text)", seconds, voiced, score)
    if voiced > 2.5 * expected + 3.0:
        return TakeCheck(False, f"runs on ({voiced:.1f}s of speech for ~{expected:.1f}s of text)", seconds, voiced, score)
    if seconds > 1.5 and silent.mean() > 0.6:
        return TakeCheck(False, "mostly silent", seconds, voiced, score + 1.0)
    return TakeCheck(True, "ok", seconds, voiced, score)


def tidy_take(samples: np.ndarray, rate: int) -> np.ndarray:
    """Trim edge silence and shorten long silences inside the take."""
    silent = _silent_frames(samples, rate)
    size = max(1, int(rate * FRAME_SECONDS))
    voiced = np.flatnonzero(~silent)
    if not len(voiced):
        return samples[:0]
    pad = int(EDGE_PAD_SECONDS / FRAME_SECONDS)
    first, last = max(0, voiced[0] - pad), min(len(silent), voiced[-1] + 1 + pad)
    keep_frames = int(KEPT_GAP / FRAME_SECONDS)
    pieces, cursor = [], first
    for start, end in _runs(silent[first:last]):
        start, end = start + first, end + first
        if (end - start) * FRAME_SECONDS > MAX_INTERNAL_GAP:
            half = keep_frames // 2
            pieces.append(samples[cursor * size:(start + half) * size])
            cursor = end - (keep_frames - half)
    pieces.append(samples[cursor * size:last * size])
    return np.concatenate(pieces) if pieces else samples[:0]


def join_takes(takes: list[tuple[int, np.ndarray]], rate: int) -> np.ndarray:
    """Concatenate (speaker, samples) in order with short, consistent gaps."""
    pieces, previous = [], None
    for speaker, samples in takes:
        if previous is not None:
            gap = SAME_SPEAKER_GAP if speaker == previous else TURN_GAP
            pieces.append(np.zeros(int(rate * gap), dtype=np.float32))
        pieces.append(samples.astype(np.float32))
        previous = speaker
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
