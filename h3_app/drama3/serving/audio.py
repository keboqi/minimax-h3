"""WAV decoding, take checks and the joiner for chunked Drama 3 audio (numpy only)."""

from __future__ import annotations

import io
import re
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
CUT_OFF_GAP = 0.04  # after a line cut off with a dash, the next speaker barges in
QUICK_REPLY_GAP = 0.22  # an answer to a question or exclamation comes quickly
WEIGHTED_GAP = 0.6  # before a line that opens slow, sad or hushed


def decode_wav(data: bytes, *, keep_channels: bool = False) -> tuple[np.ndarray, int]:
    """Float32 samples from PCM16/24/32 or float32 WAV, including WAVE_FORMAT_EXTENSIBLE.

    Mono unless ``keep_channels``, which returns a multichannel file as (frames, channels).
    """
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
        samples = samples[: len(samples) // channels * channels].reshape(-1, channels)
        if not keep_channels:
            samples = samples.mean(axis=1)
    return samples, int(rate)


def encode_wav(samples: np.ndarray, rate: int) -> bytes:
    """16-bit PCM WAV of mono samples, or of (frames, channels) samples."""
    samples = np.asarray(samples)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as target:
        target.setnchannels(1 if samples.ndim == 1 else samples.shape[1])
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


def check_take(samples: np.ndarray, rate: int, expected_seconds: float,
               max_seconds: float | None = None) -> TakeCheck:
    """Fail clearly broken takes; report uncertain duration estimates as advice only.

    A take fails when it is silent, holds a fifth or less of the activity a line of at
    least three seconds needs (the model skipped most of it), or ran until ``max_seconds``,
    the request's frame budget, stopped it.
    """
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
    if expected_seconds >= 3.0 and voiced < 0.2 * expected_seconds:
        return TakeCheck(False, f"far too short ({voiced:.1f}s of activity for ~{expected:.1f}s of text)",
                         seconds, voiced, score + 10.0)
    if max_seconds and seconds >= 0.97 * max_seconds:
        return TakeCheck(False, "ran until the length cap stopped it", seconds, voiced, score + 10.0)
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


MIN_LINE_PAUSE = 0.12  # a shorter silence inside a take is a breath, not the end of a line
PAUSE_BONUS = 2.0  # seconds of drift from the estimate a pause one second longer outweighs


def line_bounds(samples: np.ndarray, rate: int, weights: list[float]) -> list[tuple[int, int]]:
    """(start, end) samples of each script line inside one tidied take.

    ``weights`` estimate each line's share of the take. Lines are cut at pauses in the
    take, picking in order the pauses that are long and close to where the estimate puts
    each cut; a take with too few pauses is cut by the estimate alone.
    """
    length = len(samples)
    if len(weights) <= 1 or length == 0:
        return [(0, length)] * len(weights)
    total = sum(weights) or 1.0
    expected, running = [], 0.0
    for weight in weights[:-1]:
        running += weight
        expected.append(length * running / total)
    size = max(1, int(rate * FRAME_SECONDS))
    silent = _silent_frames(samples, rate)
    pauses = [(start * size, end * size) for start, end in _runs(silent)
              if start > 0 and end < len(silent) and (end - start) * FRAME_SECONDS >= MIN_LINE_PAUSE]
    cuts = len(expected)
    if len(pauses) < cuts:
        bounds = [(int(cut), int(cut)) for cut in expected]
    else:
        # cost[k][j]: the best cost of cuts 0..k with cut k at pause j, in seconds.
        def cost(k, j):
            start, end = pauses[j]
            return abs((start + end) / 2 - expected[k]) / rate - PAUSE_BONUS * (end - start) / rate
        best = [[cost(0, j) for j in range(len(pauses))]]
        back = [[-1] * len(pauses)]
        for k in range(1, cuts):
            row, links = [float("inf")] * len(pauses), [-1] * len(pauses)
            for j in range(k, len(pauses)):
                previous = min(range(k - 1, j), key=lambda i: best[-1][i])
                row[j], links[j] = best[-1][previous] + cost(k, j), previous
            best.append(row)
            back.append(links)
        j = min(range(cuts - 1, len(pauses)), key=lambda i: best[-1][i])
        chosen = []
        for k in range(cuts - 1, -1, -1):
            chosen.append(j)
            j = back[k][j]
        bounds = [pauses[j] for j in reversed(chosen)]
    starts = [0, *(end for _start, end in bounds)]
    ends = [*(start for start, _end in bounds), length]
    return list(zip(starts, ends))


_TRAILING_MARKUP = re.compile(r"(?:\s*(?:\[[^\[\]]*\]|<[^<>]*>))+\s*$")
_LEADING_MARKUP = re.compile(r"^(?:\s*<[^<>]*>)+\s*")
_CLOSING_QUOTES = "\"'”’」』）)"
_WEIGHTED_CUE = re.compile(r"\[[^\]]*\b(?:sad|sadly|soft|softly|slow|slowly|quiet|quietly|hush|hushed|hesitant|"
                           r"hesitantly|gentle|gently|tired|weary|grief|grieving|whisper|whispering|whispers|"
                           r"reluctant|reluctantly|heavy|somber|sombre|mournful|tearful)\b[^\]]*\]|<(?:slow|soft|whisper)>",
                           re.IGNORECASE)


def turn_gap(previous_text: str, next_text: str, same_speaker: bool) -> float:
    """Silence before ``next_text``, shaped by how ``previous_text`` ends and how the next line opens.

    A line cut off with a dash is barged in on, a question or exclamation is answered
    quickly, and a line opening slow, sad or hushed gets a longer breath before it.
    """
    if same_speaker:
        return SAME_SPEAKER_GAP
    ending = _TRAILING_MARKUP.sub("", str(previous_text or "")).rstrip().rstrip(_CLOSING_QUOTES).rstrip()
    if ending.endswith(("—", "–", "-", "~")):
        return CUT_OFF_GAP
    opening = _LEADING_MARKUP.sub("", str(next_text or ""))
    # A cue shapes the words after it, so only the line's first words set its entry.
    if _WEIGHTED_CUE.search(opening[:80]):
        return WEIGHTED_GAP
    if ending.endswith(("?", "!", "？", "！")):
        return QUICK_REPLY_GAP
    return TURN_GAP


def join_takes(takes: list[tuple], rate: int,
               spans: list[tuple[int, int]] | None = None) -> np.ndarray:
    """Concatenate (speaker, samples[, text]) in order with short gaps.

    With texts the gap follows the lines (see ``turn_gap``); without, it is the plain turn
    or same-speaker gap. ``spans``, when given, receives each take's (start, end) sample
    offsets in the result.
    """
    pieces, previous, cursor = [], None, 0
    for speaker, samples, *text in takes:
        text = text[0] if text else None
        if previous is not None:
            previous_speaker, previous_text = previous
            same = speaker == previous_speaker
            gap = (turn_gap(previous_text, text, same) if text is not None and previous_text is not None
                   else SAME_SPEAKER_GAP if same else TURN_GAP)
            pieces.append(np.zeros((int(rate * gap), *np.shape(samples)[1:]), dtype=np.float32))
            cursor += len(pieces[-1])
        pieces.append(samples.astype(np.float32))
        if spans is not None:
            spans.append((cursor, cursor + len(samples)))
        cursor += len(samples)
        previous = (speaker, text)
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)


def active_level_db(samples: np.ndarray, rate: int) -> float | None:
    """RMS level of a take's voiced frames in dBFS, or None for a silent take."""
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    silent = _silent_frames(samples, rate)
    if not silent.size or silent.all():
        return None
    size = max(1, int(rate * FRAME_SECONDS))
    frames = samples[: silent.size * size].reshape(-1, size)[~silent].astype(np.float64)
    return float(10.0 * np.log10(max(np.mean(frames ** 2), 1e-12)))


def resample_audio(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample mono or (frames, channels) float32 audio from src_rate to dst_rate."""
    audio = np.asarray(samples, dtype=np.float32)
    audio = audio if audio.ndim == 2 else audio.reshape(-1)
    if src_rate <= 0 or dst_rate <= 0 or src_rate == dst_rate or audio.size == 0:
        return audio
    import math
    from scipy import signal

    divisor = math.gcd(int(src_rate), int(dst_rate))
    up = int(dst_rate) // divisor
    down = int(src_rate) // divisor
    return signal.resample_poly(audio, up, down, axis=0).astype(np.float32, copy=False)


def load_audio_file(path: str | Path, target_rate: int, *, stereo: bool = False) -> np.ndarray:
    """Load an audio file (WAV or ffmpeg-decoded format) as float32 at target_rate.

    Mono, or with ``stereo`` (frames, 2), a mono file on both channels.
    """
    source = Path(path)
    if not source.is_file():
        raise H3Error(f"Sound audio file is missing: {source}")
    raw = source.read_bytes()
    if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        samples, rate = decode_wav(raw, keep_channels=stereo)
        return resample_audio(as_stereo(samples) if stereo else samples, rate, target_rate)
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
                "2" if stereo else "1",
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
        samples, rate = decode_wav(decoded_wav.read_bytes(), keep_channels=stereo)
        return resample_audio(as_stereo(samples) if stereo else samples, rate, target_rate)


def as_stereo(samples: np.ndarray) -> np.ndarray:
    """(frames, 2) float32: mono on both channels, the first two channels of a wider file."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 1:
        return np.repeat(samples[:, None], 2, axis=1)
    if samples.shape[1] == 1:
        return np.repeat(samples, 2, axis=1)
    return samples[:, :2]


def _along(gain: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """``gain`` per frame, shaped to scale mono or (frames, channels) ``samples``."""
    return gain if samples.ndim == 1 else gain[:, None]


def pan_gains(position) -> tuple:
    """Constant-power (left, right) gains for ``position`` from -1 (left) to 1 (right)."""
    angle = (np.clip(position, -1.0, 1.0) + 1.0) * np.pi / 4.0
    return np.cos(angle), np.sin(angle)


def pan_mono(samples: np.ndarray, position: float) -> np.ndarray:
    """A mono voice placed at ``position``; its loudness is the same anywhere."""
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    left, right = pan_gains(float(position))
    return np.stack([samples * left, samples * right], axis=1).astype(np.float32)


FAR_LOWPASS_HZ = 2500.0
FAR_REFLECTIONS = ((0.031, 0.35), (0.057, 0.25), (0.089, 0.18), (0.131, 0.12))  # (seconds, gain)
# Applied with a sound's volume, after it is leveled to the speech, which would otherwise undo it.
DISTANCE_GAINS = {"far": 0.4, "near": 1.25}  # about -8 dB and +2 dB


def distance_gain(distance: str) -> float:
    """The level change ``distance`` ("near" or "far") makes, applied with the sound's volume."""
    return DISTANCE_GAINS.get(distance, 1.0)


def place_sound(samples: np.ndarray, rate: int, pan: tuple[float, float] | None = None,
                distance: str = "") -> np.ndarray:
    """A stereo sound moved to ``pan`` (start, end positions) and given the tone of ``distance``.

    A far sound loses its highs and gains a few room reflections; its quieter level is
    ``distance_gain``, applied with the sound's volume. A panned stereo sound narrows
    toward mono as it moves to the side, so its image follows.
    """
    out = as_stereo(samples).copy()
    if distance == "far" and len(out):
        from scipy import signal

        sos = signal.butter(2, min(FAR_LOWPASS_HZ / (0.5 * rate), 0.99), btype="lowpass", output="sos")
        dry = signal.sosfilt(sos, out, axis=0).astype(np.float32)
        wet = dry.copy()
        for index, (seconds, gain) in enumerate(FAR_REFLECTIONS):
            offset = int(seconds * rate)
            if offset < len(dry):
                # Alternate sides so the reflections widen the sound instead of doubling it.
                wet[offset:] += dry[:-offset][:, ::-1 if index % 2 else 1] * gain
        out = wet
    if pan is not None and len(out):
        start, end = (float(np.clip(value, -1.0, 1.0)) for value in pan)
        positions = np.linspace(start, end, len(out), dtype=np.float32) if start != end else np.full(len(out), start, np.float32)
        mono = out.mean(axis=1)
        width = 1.0 - np.abs(positions)  # a sound at the far side is a point source
        left, right = pan_gains(positions)
        out = np.stack([out[:, 0] * width + mono * (1 - width) * left * np.sqrt(2),
                        out[:, 1] * width + mono * (1 - width) * right * np.sqrt(2)], axis=1)
    return out.astype(np.float32, copy=False)


LOOP_CROSSFADE_SECONDS = 0.5
LOOP_SEARCH_SECONDS = 2.0  # how far before a clip's end a better loop point is looked for


def loop_end(clip: np.ndarray, overlap: int, rate: int) -> int:
    """Where a looped clip should end so its crossfade into its own start is least audible.

    The last ``overlap`` samples before the chosen end are the ones most like the clip's
    first ``overlap`` samples, searched within ``LOOP_SEARCH_SECONDS`` of the end and
    without giving up more than a quarter of the clip.
    """
    from scipy import signal

    mono = clip.mean(axis=1) if clip.ndim == 2 else clip
    search = min(int(LOOP_SEARCH_SECONDS * rate), len(mono) // 4)
    if search <= 0 or len(mono) < 4 * overlap:
        return len(mono)
    head = mono[:overlap].astype(np.float64)
    region = mono[len(mono) - search - overlap:].astype(np.float64)
    score = signal.correlate(region, head, mode="valid", method="fft")
    energy = np.cumsum(np.concatenate(([0.0], region ** 2)))
    windows = energy[overlap:] - energy[:-overlap]
    score = score / np.sqrt(np.maximum(windows[: len(score)] * float(np.dot(head, head)), 1e-12))
    return len(mono) - search + int(np.argmax(score))


def loop_to_length(samples: np.ndarray, length: int, rate: int,
                   crossfade: float = LOOP_CROSSFADE_SECONDS) -> np.ndarray:
    """Repeat a clip to ``length`` samples, equal-power crossfading each seam instead of hard-cutting.

    Generated clips do not end where they begin, so a plain tile clicks at every repeat.
    """
    clip = np.asarray(samples, dtype=np.float32)
    clip = clip if clip.ndim == 2 else clip.reshape(-1)
    if len(clip) == 0 or length <= 0:
        return np.zeros((max(0, length), *clip.shape[1:]), dtype=np.float32)
    if len(clip) >= length:
        return clip[:length].copy()
    overlap = min(int(crossfade * rate), len(clip) // 4)
    if overlap > 0:
        clip = clip[:loop_end(clip, overlap, rate)]
    if overlap <= 0:
        return np.tile(clip, (-(-length // len(clip)), *([1] * (clip.ndim - 1))))[:length].astype(np.float32)
    ramp = np.linspace(0.0, np.pi / 2, overlap, dtype=np.float32)
    fade_in, fade_out = _along(np.sin(ramp), clip), _along(np.cos(ramp), clip)
    step = len(clip) - overlap
    repeats = -(-(length - overlap) // step)
    out = np.zeros((repeats * step + overlap, *clip.shape[1:]), dtype=np.float32)
    for index in range(repeats):
        piece = clip.copy()
        if index:
            piece[:overlap] *= fade_in
        if index < repeats - 1:
            piece[-overlap:] *= fade_out
        out[index * step:index * step + len(clip)] += piece
    return out[:length]


LOOP_FADE_IN_SECONDS = 1.0  # beds rise instead of starting at full level ...
LOOP_TAIL_SECONDS = 1.0  # ... and fade out over this after the block's speech, under what follows
SCENE_LEAD_IN_SECONDS = 0.8  # a block with a bed is heard for this long before its speech
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


@dataclass(frozen=True)
class Mix:
    audio: np.ndarray
    advance: int  # where the timeline moves on; sound tails beyond it overlap what follows
    speech_offset: int  # where the block's speech starts in ``audio``
    # Per track: how long a sound plays from the block's start (a bed loops to its tail); 0 for speech.
    sound_lengths: tuple[int, ...] = ()


def _fade(samples: np.ndarray, fade_in: int, fade_out: int) -> np.ndarray:
    fade_in, fade_out = min(fade_in, len(samples)), min(fade_out, len(samples))
    if fade_in > 0:
        samples[:fade_in] *= _along(np.sin(np.linspace(0.0, np.pi / 2, fade_in, dtype=np.float32)), samples)
    if fade_out > 0:
        samples[-fade_out:] *= _along(np.cos(np.linspace(0.0, np.pi / 2, fade_out, dtype=np.float32)), samples)
    return samples


def mix_parallel(tracks: list[dict | tuple], rate: int) -> Mix:
    """Mix a ``<parallel>`` block's speech and sound tracks.

    Sounds are leveled to the speech. A block with a looped bed opens with the bed alone
    for ``SCENE_LEAD_IN_SECONDS``; beds fade in, loop to the end of the speech, dip under
    it, and fade out over ``LOOP_TAIL_SECONDS``. One-shot effects and stingers play out in
    full. The block advances the timeline at the end of its speech (a block without speech:
    its longest sound), so bed and effect tails overlap what follows. Levels are not limited
    here; the finished mix is mastered once.
    """
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
        return Mix(np.zeros(0, dtype=np.float32), 0, 0, (0,) * len(normalized))
    # Mono tracks join a stereo block on both channels.
    stereo = any(np.ndim(t["audio"]) == 2 for t in valid)
    shape = lambda arr: as_stereo(arr) if stereo else np.asarray(arr, dtype=np.float32).reshape(-1)  # noqa: E731

    speech_tracks = [t for t in valid if not t.get("is_sound", False)]
    sound_tracks = [t for t in valid if t.get("is_sound", False)]
    has_bed = any(t.get("loop", False) for t in sound_tracks)
    offset = int(SCENE_LEAD_IN_SECONDS * rate) if speech_tracks and has_bed else 0
    speech_end = offset + max((int(t.get("advance", len(t["audio"]))) for t in speech_tracks), default=0)

    speech_rms_values = [float(np.sqrt(np.mean(np.asarray(t["audio"], dtype=np.float64) ** 2)))
                         for t in speech_tracks]
    avg_speech_rms = float(np.mean(speech_rms_values)) if speech_rms_values else 0.0

    sounds: list[tuple[np.ndarray, bool]] = []
    played: dict[int, int] = {}
    for track in sound_tracks:
        samples = shape(track["audio"]).copy()
        loop = bool(track.get("loop", False))
        if loop and speech_tracks:
            tail = int(LOOP_TAIL_SECONDS * rate)
            samples = _fade(loop_to_length(samples, speech_end + tail, rate),
                            int(LOOP_FADE_IN_SECONDS * rate), tail)
        current_rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
        if avg_speech_rms > 0.0 and current_rms > 1e-7:
            samples = samples * float(min(avg_speech_rms * 0.7 / current_rms, 2.0))
        samples = samples * float(track.get("volume", 1.0))
        sounds.append((samples.astype(np.float32, copy=False), loop))
        played[id(track)] = len(samples)

    length = max([offset + len(t["audio"]) for t in speech_tracks] + [len(arr) for arr, _bed in sounds])
    speech = np.zeros((length, 2) if stereo else length, dtype=np.float32)
    for track in speech_tracks:
        arr = shape(track["audio"])
        speech[offset:offset + len(arr)] += arr
    # Only looped beds (ambience, music) dip under speech; a one-shot effect or stinger
    # is usually anchored to the sentence that narrates it and must stay in front.
    duck = speech_duck_gain(speech.mean(axis=1) if stereo else speech, rate) if speech_tracks else None
    mixed = speech.copy()
    for arr, bed in sounds:
        mixed[: len(arr)] += arr * _along(duck[: len(arr)], arr) if bed and duck is not None else arr
    advance = speech_end if speech_tracks else max(len(arr) for arr, _bed in sounds)
    return Mix(mixed, advance, offset, tuple(played.get(id(track), 0) for track in normalized))


# Loudness of the finished mix (ITU-R BS.1770). A mono file plays on both speakers,
# about 3 LU louder than it measures, so -19 LUFS mono matches the usual -16 LUFS stereo.
TARGET_LUFS = -16.0
TARGET_LUFS_MONO = -19.0
MAX_LOUDNESS_GAIN_DB = 12.0
PEAK_CEILING_DBFS = -1.5  # sample peaks, leaving room for inter-sample and MP3 overshoot
LIMITER_BLOCK_SECONDS = 0.0025
LIMITER_LOOKAHEAD_SECONDS = 0.005
LIMITER_RELEASE_SECONDS = 0.08
LEVEL_CHUNK_STEPS = 600  # blocks per piece when metering or limiting a long mix


def _k_weighting(rate: int) -> np.ndarray:
    """BS.1770 K-weighting (high shelf, then high-pass) as second-order sections for ``rate``.

    The analog prototypes behind the standard's 48 kHz coefficients, re-discretized for ``rate``.
    """
    k = np.tan(np.pi * 1681.9744509555319 / rate)
    q = 0.7071752369554193
    vh = 10.0 ** (3.99984385397 / 20.0)
    vb = vh ** 0.499666774155
    norm = 1.0 + k / q + k * k
    shelf = [(vh + vb * k / q + k * k) / norm, 2.0 * (k * k - vh) / norm, (vh - vb * k / q + k * k) / norm,
             1.0, 2.0 * (k * k - 1.0) / norm, (1.0 - k / q + k * k) / norm]
    k = np.tan(np.pi * 38.13547087613982 / rate)
    q = 0.5003270373253953
    norm = 1.0 + k / q + k * k
    highpass = [1.0, -2.0, 1.0, 1.0, 2.0 * (k * k - 1.0) / norm, (1.0 - k / q + k * k) / norm]
    return np.array([shelf, highpass], dtype=np.float64)


def integrated_loudness(samples: np.ndarray, rate: int) -> float | None:
    """Gated integrated loudness in LUFS, or None when the audio is too short or silent.

    Channels of (frames, channels) audio are summed with equal weight, as for stereo.
    """
    from scipy import signal

    samples = np.asarray(samples, dtype=np.float32)
    samples = samples if samples.ndim == 2 else samples.reshape(-1, 1)
    step = int(0.1 * rate)  # 400 ms blocks overlap by 75%: four 100 ms steps each
    steps = len(samples) // step if rate > 0 else 0
    if steps < 4:
        return None
    sos = _k_weighting(rate)
    state = np.zeros((sos.shape[0], 2, samples.shape[1]))
    energy = np.empty(steps, dtype=np.float64)
    # Filtered in pieces so a long chapter never needs a float64 copy of itself.
    for first in range(0, steps, LEVEL_CHUNK_STEPS):
        last = min(first + LEVEL_CHUNK_STEPS, steps)
        weighted, state = signal.sosfilt(sos, samples[first * step:last * step].astype(np.float64), axis=0, zi=state)
        energy[first:last] = (weighted.reshape(-1, step, samples.shape[1]) ** 2).sum(axis=(1, 2))
    z = np.convolve(energy, np.ones(4), mode="valid") / (4 * step)
    z = z[-0.691 + 10.0 * np.log10(np.maximum(z, 1e-20)) > -70.0]  # absolute gate
    if not z.size:
        return None
    relative = -0.691 + 10.0 * np.log10(np.mean(z)) - 10.0
    z = z[-0.691 + 10.0 * np.log10(z) > relative]
    return float(-0.691 + 10.0 * np.log10(np.mean(z)))


def limit_peaks(samples: np.ndarray, rate: int, ceiling_db: float = PEAK_CEILING_DBFS) -> np.ndarray:
    """Look-ahead peak limiter: gain falls just before a peak and recovers smoothly after it."""
    from scipy.ndimage import minimum_filter1d

    samples = np.asarray(samples, dtype=np.float32)
    samples = samples if samples.ndim == 2 else samples.reshape(-1)
    ceiling = float(10.0 ** (ceiling_db / 20.0))
    if not samples.size or float(np.max(np.abs(samples))) <= ceiling:
        return samples
    size = max(1, int(rate * LIMITER_BLOCK_SECONDS))
    count = -(-len(samples) // size)
    padded = np.zeros(count * size, dtype=np.float32)
    # Linked channels: one gain for both keeps a voice where it was placed.
    padded[: len(samples)] = np.abs(samples).max(axis=1) if samples.ndim == 2 else np.abs(samples)
    peaks = padded.reshape(count, size).max(axis=1)
    needed = np.minimum(1.0, ceiling / np.maximum(peaks, 1e-9))
    # Each block also covers its neighbours, so gain interpolated between blocks never overshoots.
    reach = max(1, -(-int(rate * LIMITER_LOOKAHEAD_SECONDS) // size))
    needed = minimum_filter1d(needed, size=2 * reach + 1, mode="nearest")
    recover = 1.0 - float(np.exp(-LIMITER_BLOCK_SECONDS / LIMITER_RELEASE_SECONDS))
    gains = np.empty(count, dtype=np.float64)
    gain = 1.0
    for index, target in enumerate(needed):
        gain = target if target < gain else gain + (target - gain) * recover
        gains[index] = gain
    centers = (np.arange(count) + 0.5) * size
    out = np.empty_like(samples)
    piece = LEVEL_CHUNK_STEPS * size
    for start in range(0, len(samples), piece):
        stop = min(start + piece, len(samples))
        envelope = np.interp(np.arange(start, stop), centers, gains).astype(np.float32)
        out[start:stop] = np.clip(samples[start:stop] * _along(envelope, samples), -ceiling, ceiling)
    return out


def master(samples: np.ndarray, rate: int, target_lufs: float | None = None) -> np.ndarray:
    """Bring the finished mix to ``target_lufs`` and keep its peaks under the ceiling.

    The target defaults to ``TARGET_LUFS`` for stereo and ``TARGET_LUFS_MONO`` for mono.
    One gain for the whole piece keeps whispers, shouts and scene levels as mixed.
    """
    samples = np.asarray(samples, dtype=np.float32)
    samples = samples if samples.ndim == 2 else samples.reshape(-1)
    if target_lufs is None:
        target_lufs = TARGET_LUFS if samples.ndim == 2 else TARGET_LUFS_MONO
    loudness = integrated_loudness(samples, rate)
    if loudness is not None:
        gain_db = min(target_lufs - loudness, MAX_LOUDNESS_GAIN_DB)
        samples = samples * float(10.0 ** (gain_db / 20.0))
    return limit_peaks(samples, rate)
