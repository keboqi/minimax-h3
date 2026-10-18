"""WAV decoding, take checks and the joiner for chunked Drama 3 audio (numpy only)."""

from __future__ import annotations

import io
import struct
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from h3_app.errors import H3Error
from h3_app.processes import run_media_process

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
        raise ValueError("SGLang-Omni did not return a WAV file.")
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


def save_mp3(samples: np.ndarray, rate: int, target: Path) -> None:
    """Encode the joined PCM once, publishing only a complete MP3."""
    pending = target.with_suffix(".mp3.partial")
    try:
        with tempfile.TemporaryDirectory(prefix="drama3-audio-") as directory:
            source = Path(directory) / "joined.wav"
            source.write_bytes(encode_wav(samples, rate))
            proc = run_media_process(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
                 "-c:a", "libmp3lame", "-q:a", "0", "-f", "mp3", str(pending)],
                partial_outputs=(pending,),
            )
            if proc.returncode != 0 or not pending.is_file() or pending.stat().st_size == 0:
                raise H3Error(f"TTS MP3 encoding failed: {proc.stderr.strip()}")
            pending.replace(target)
    finally:
        pending.unlink(missing_ok=True)


def _silent_frames(samples: np.ndarray, rate: int) -> np.ndarray:
    size = max(1, int(rate * FRAME_SECONDS))
    frames = samples[: len(samples) // size * size].reshape(-1, size)
    if not len(frames):
        return np.zeros(0, dtype=bool)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    # Quiet/whispered takes need a lower threshold, including when trimming.
    # This is an activity detector, not a speech recognizer.
    peak = float(rms.max())
    threshold = max(10 ** (-75 / 20), min(10 ** (SILENCE_DBFS / 20), peak * 10 ** (-30 / 20)))
    return rms < threshold


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
    warnings: tuple[str, ...] = ()  # uncertain heuristics never trigger regeneration


def check_take(samples: np.ndarray, rate: int, expected_seconds: float) -> TakeCheck:
    """Retry near-silent takes; report uncertain duration estimates as advice only."""
    if rate <= 0 or not np.isfinite(samples).all():
        raise ValueError("TTS audio has an invalid sample rate or non-finite samples.")
    silent = _silent_frames(samples, rate)
    seconds = len(samples) / rate
    voiced = float((~silent).sum()) * FRAME_SECONDS
    expected = max(0.8, expected_seconds)
    ratio = voiced / expected
    score = abs(np.log(max(ratio, 1e-3)))
    if voiced < 0.08:
        return TakeCheck(False, "no speech", seconds, voiced, 99.0)
    warnings = []
    if voiced < 0.35 * expected - 1.0:
        warnings.append(f"possibly too short ({voiced:.1f}s of activity for ~{expected:.1f}s of text)")
    if voiced > 2.5 * expected + 3.0:
        warnings.append(f"possibly runs on ({voiced:.1f}s of activity for ~{expected:.1f}s of text)")
    # Padding is trimmed later; only assess gaps between the first and last activity.
    active = np.flatnonzero(~silent)
    interior = silent[active[0]:active[-1] + 1]
    if len(interior) * FRAME_SECONDS > 1.5 and interior.mean() > 0.6:
        warnings.append("long internal pauses")
    return TakeCheck(True, "; ".join(warnings) or "ok", seconds, voiced, score, tuple(warnings))


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


def resample_audio(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample 1D float32 audio from src_rate to dst_rate."""
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if src_rate <= 0 or dst_rate <= 0 or src_rate == dst_rate or audio.size == 0:
        return audio
    import math
    from scipy import signal

    divisor = math.gcd(int(src_rate), int(dst_rate))
    up = int(dst_rate) // divisor
    down = int(src_rate) // divisor
    return signal.resample_poly(audio, up, down).astype(np.float32, copy=False)


def load_audio_file(path: str | Path, target_rate: int) -> np.ndarray:
    """Load an audio file (WAV or ffmpeg-decoded format) as mono float32 at target_rate."""
    source = Path(path)
    if not source.is_file():
        raise H3Error(f"Sound audio file is missing: {source}")
    raw = source.read_bytes()
    if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        samples, rate = decode_wav(raw)
        return resample_audio(samples, rate, target_rate)
    with tempfile.TemporaryDirectory(prefix="drama3-sound-decode-") as directory:
        decoded_wav = Path(directory) / "decoded.wav"
        proc = run_media_process(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-ac",
                "1",
                "-ar",
                str(int(target_rate)),
                "-f",
                "wav",
                str(decoded_wav),
            ],
            partial_outputs=(decoded_wav,),
        )
        if proc.returncode != 0 or not decoded_wav.is_file():
            raise H3Error(f"Failed to decode sound clip {source.name}: {proc.stderr.strip()}")
        samples, rate = decode_wav(decoded_wav.read_bytes())
        return resample_audio(samples, rate, target_rate)


LOOP_CROSSFADE_SECONDS = 0.5


def loop_to_length(samples: np.ndarray, length: int, rate: int,
                   crossfade: float = LOOP_CROSSFADE_SECONDS) -> np.ndarray:
    """Repeat a clip to ``length`` samples, equal-power crossfading each seam instead of hard-cutting.

    Generated clips do not end where they begin, so a plain tile clicks at every repeat.
    """
    clip = np.asarray(samples, dtype=np.float32).reshape(-1)
    if clip.size == 0 or length <= 0:
        return np.zeros(max(0, length), dtype=np.float32)
    if clip.size >= length:
        return clip[:length].copy()
    overlap = min(int(crossfade * rate), clip.size // 4)
    if overlap <= 0:
        return np.tile(clip, -(-length // clip.size))[:length].astype(np.float32)
    ramp = np.linspace(0.0, np.pi / 2, overlap, dtype=np.float32)
    fade_in, fade_out = np.sin(ramp), np.cos(ramp)
    step = clip.size - overlap
    repeats = -(-(length - overlap) // step)
    out = np.zeros(repeats * step + overlap, dtype=np.float32)
    for index in range(repeats):
        piece = clip.copy()
        if index:
            piece[:overlap] *= fade_in
        if index < repeats - 1:
            piece[-overlap:] *= fade_out
        out[index * step:index * step + clip.size] += piece
    return out[:length]


DUCK_DEPTH_DB = 12.0
DUCK_ATTACK_SECONDS = 0.05
DUCK_RELEASE_SECONDS = 0.3
DUCK_FRAME_SECONDS = 0.01


def speech_duck_gain(speech: np.ndarray, rate: int, *, depth_db: float = DUCK_DEPTH_DB,
                     attack: float = DUCK_ATTACK_SECONDS, release: float = DUCK_RELEASE_SECONDS) -> np.ndarray:
    """Per-sample gain for sound tracks: ``depth_db`` lower while ``speech`` is audible.

    A sidechain envelope follower: 10 ms frames above the silence floor pull the
    gain down over ``attack`` and let it recover over ``release`` once speech stops.
    """
    samples = np.asarray(speech, dtype=np.float32).reshape(-1)
    if samples.size == 0 or rate <= 0:
        return np.ones(samples.size, dtype=np.float32)
    frame = max(1, int(rate * DUCK_FRAME_SECONDS))
    count = -(-samples.size // frame)
    padded = np.zeros(count * frame, dtype=np.float64)
    padded[: samples.size] = samples
    rms = np.sqrt(np.mean(padded.reshape(count, frame) ** 2, axis=1))
    speaking = 20.0 * np.log10(np.maximum(rms, 1e-10)) > SILENCE_DBFS
    floor = float(10.0 ** (-abs(depth_db) / 20.0))
    down = float(np.exp(-DUCK_FRAME_SECONDS / max(attack, 1e-4)))
    up = float(np.exp(-DUCK_FRAME_SECONDS / max(release, 1e-4)))
    gains = np.empty(count, dtype=np.float64)
    gain = 1.0
    for index, active in enumerate(speaking):
        target = floor if active else 1.0
        gain = target + (gain - target) * (down if target < gain else up)
        gains[index] = gain
    centers = (np.arange(count) + 0.5) * frame
    return np.interp(np.arange(samples.size), centers, gains).astype(np.float32)


def mix_parallel_tracks(tracks: list[dict | tuple], rate: int) -> np.ndarray:
    """Mix speech and sound tracks: loop, level sounds to speech RMS, duck looped beds under speech, fade, limit."""
    normalized: list[dict] = []
    for item in tracks:
        if isinstance(item, dict):
            normalized.append(item)
        elif isinstance(item, (tuple, list)) and len(item) >= 1:
            normalized.append({
                "audio": item[0],
                "is_sound": bool(item[1]) if len(item) > 1 else False,
                "loop": bool(item[2]) if len(item) > 2 else False,
                "volume": float(item[3]) if len(item) > 3 else 1.0,
            })
    valid = [t for t in normalized if t.get("audio") is not None and len(t["audio"]) > 0]
    if not valid:
        return np.zeros(0, dtype=np.float32)

    speech_lengths = [
        len(t["audio"])
        for t in valid
        if not t.get("is_sound", False)
    ]
    max_speech_length = max(speech_lengths) if speech_lengths else 0

    speech_rms_values = [
        float(np.sqrt(np.mean(np.asarray(t["audio"], dtype=np.float64) ** 2)))
        for t in valid
        if not t.get("is_sound", False) and len(t["audio"]) > 0
    ]
    avg_speech_rms = float(np.mean(speech_rms_values)) if speech_rms_values else 0.0

    processed: list[tuple[np.ndarray, bool, bool]] = []
    for track in valid:
        samples = np.asarray(track["audio"], dtype=np.float32).reshape(-1).copy()
        is_sound = bool(track.get("is_sound", False))
        loop = bool(track.get("loop", False))
        volume = float(track.get("volume", 1.0))

        if loop and max_speech_length > 0 and samples.size > 0:
            target_len = max_speech_length + rate
            samples = loop_to_length(samples, target_len, rate)
            fade_len = min(rate, samples.size)
            if fade_len > 0:
                samples[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)

        if is_sound and samples.size > 0:
            current_rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
            if avg_speech_rms > 0.0 and current_rms > 1e-7:
                target_sound_rms = avg_speech_rms * 0.7
                scaling_factor = min(target_sound_rms / current_rms, 2.0)
                samples = samples * float(scaling_factor)
            if volume != 1.0:
                samples = samples * volume

        processed.append((samples.astype(np.float32, copy=False), is_sound, is_sound and loop))

    max_len = max(len(arr) for arr, _is_sound, _bed in processed)
    speech = np.zeros(max_len, dtype=np.float32)
    for arr, is_sound, _bed in processed:
        if not is_sound:
            speech[: len(arr)] += arr
    # Only looped beds (ambience, music) dip under speech; a one-shot effect or stinger
    # is usually anchored to the sentence that narrates it and must stay in front.
    duck = speech_duck_gain(speech, rate) if speech_lengths else None
    mixed = speech.copy()
    for arr, is_sound, bed in processed:
        if is_sound:
            mixed[: len(arr)] += arr * duck[: len(arr)] if bed and duck is not None else arr

    if max_speech_length > 0 and mixed.size > max_speech_length + rate:
        target_len = max_speech_length + rate
        mixed = mixed[:target_len].copy()
        fade_len = min(rate, mixed.size)
        if fade_len > 0:
            mixed[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)

    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 1.0:
        mixed = mixed * (0.9 / peak)
    return mixed.astype(np.float32, copy=False)

