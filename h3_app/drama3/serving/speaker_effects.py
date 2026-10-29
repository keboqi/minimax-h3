"""Speaker vocal post-processing effects for Drama 3 SGLang-Omni turns."""

from __future__ import annotations

import re
import numpy as np
from scipy import signal

SPEAKER_EFFECTS: tuple[str, ...] = (
    # Communication & Tech
    "phone_call",
    "radio_broadcast",
    "walkie_talkie",
    "bad_signal",
    "megaphone",
    "old_gramophone",
    "metallic_helmet",
    # Environment & Space
    "reverb_small_room",
    "reverb_large_hall",
    "stadium_announcement",
    "dream_sequence",
    "underwater",
    # Character & Creature
    "robotic_voice",
    "monster_voice",
    "intimate_whisper",
    "ghost_spirit",
    "giant_titan",
    "fairy_tiny",
    "alien_voice",
    "ancient_god",
)

# Where a voice stands, written in the effect slot; the mix places the voice, no effect runs.
DISTANCE_TAGS: tuple[str, ...] = ("far", "near")
SPEAKER_TAGS: tuple[str, ...] = (*SPEAKER_EFFECTS, *DISTANCE_TAGS)
# Place effects heard as a real room (see ``rooms``), by the room each names.
ROOM_EFFECT_ROOMS = {"reverb_small_room": "small_room", "reverb_large_hall": "large_hall"}

# Effects that carry a reverb of their own: a space, of which a voice is heard in only one.
SPACE_EFFECTS = frozenset({"metallic_helmet", "stadium_announcement", "dream_sequence", "underwater",
                           "monster_voice", "ghost_spirit", "giant_titan", "fairy_tiny", "ancient_god"})

SPEAKER_EFFECT_SET: frozenset[str] = frozenset(SPEAKER_TAGS)
_EFFECT_ALTERNATION = "|".join(re.escape(name) for name in SPEAKER_TAGS)
SPEAKER_EFFECT_TAG_RE = re.compile(rf"</?({_EFFECT_ALTERNATION})\s*>", re.IGNORECASE)
_OPENING_EFFECT_TAG_RE = re.compile(rf"<({_EFFECT_ALTERNATION})\s*>", re.IGNORECASE)


def extract_speaker_effects(text: str) -> tuple[str, tuple[str, ...]]:
    """Extract speaker vocal effect tags from turn text and return (cleaned_text, effects)."""
    effects: list[str] = []
    for match in _OPENING_EFFECT_TAG_RE.finditer(text or ""):
        effect_name = match.group(1).lower()
        if effect_name in SPEAKER_EFFECT_SET and effect_name not in effects:
            effects.append(effect_name)
    cleaned = SPEAKER_EFFECT_TAG_RE.sub("", text or "")
    cleaned = re.sub(r" {2,}", " ", cleaned).strip()
    return cleaned, tuple(effects)


def strip_speaker_effects(text: str) -> str:
    """Remove all speaker vocal effect tags from a script or line while preserving line breaks."""
    if not text:
        return ""
    lines: list[str] = []
    for line in text.splitlines():
        cleaned = SPEAKER_EFFECT_TAG_RE.sub("", line)
        cleaned = re.sub(r" {2,}", " ", cleaned).rstrip()
        lines.append(cleaned)
    return "\n".join(lines)


def has_speaker_effects(text: str) -> bool:
    """Return True when `text` contains any `<speaker_effect>` tag."""
    return bool(_OPENING_EFFECT_TAG_RE.search(text or ""))


def apply_speaker_effects(
    samples: np.ndarray,
    rate: int,
    effects: tuple[str, ...] | list[str],
    *,
    in_room: bool = False,
) -> np.ndarray:
    """Apply chained vocal DSP effects to 1D float32 audio samples.

    A real room's tag adds that room folded to mono and cut to the take's length; the mix
    renders rooms in stereo with their tails instead (see ``rooms``). Distance tags are
    placement, not effects, and are skipped. A voice ``in_room``, or tagged with one, leaves
    out its effects' own reverb: two spaces stacked wash the words out.
    """
    from .rooms import room_wet

    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    active = [name.lower() for name in effects if name and name.lower() in SPEAKER_EFFECTS]
    if not active or audio.size == 0 or rate <= 0:
        return audio

    in_room = in_room or any(name in ROOM_EFFECT_ROOMS for name in active)
    # Stacked effects keep one reverb between them, the last one's; a room leaves none.
    out, pending = audio, []
    # Effects run in their written order, so a room comes before the device it is heard through.
    for effect_name in [*active, None]:
        if effect_name is not None and effect_name not in ROOM_EFFECT_ROOMS:
            pending.append(effect_name)
            continue
        if pending:
            processed = _try_apply_pedalboard(out, rate, pending, in_room)
            if processed is None:
                processed = out.copy()
                for index, name in enumerate(pending):
                    spaced_later = any(later in SPACE_EFFECTS for later in pending[index + 1:])
                    processed = _apply_scipy_effect(processed, rate, name, in_room or spaced_later)
            out, pending = processed, []
        if effect_name is not None:
            out = out + room_wet(out, rate, ROOM_EFFECT_ROOMS[effect_name])[: len(out)].mean(axis=1)
    # Drive and gain stages can push either path past full scale.
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out = out * (0.95 / peak)
    return out.astype(np.float32, copy=False)


def _try_apply_pedalboard(
    audio: np.ndarray,
    rate: int,
    effects: list[str],
    in_room: bool = False,
) -> np.ndarray | None:
    try:
        from pedalboard import (  # type: ignore[import-not-found]
            Bitcrush,
            Chorus,
            Compressor,
            Delay,
            Distortion,
            Gain,
            HighpassFilter,
            LowpassFilter,
            PeakFilter,
            Pedalboard,
            Phaser,
            PitchShift,
            Reverb,
        )
    except Exception:
        return None

    # Each preset keeps the dry voice in front: echoes stay short and few, and a preset's
    # own reverb is its space, left out when the voice is heard in a real room instead.
    presets = {
        "phone_call": [
            HighpassFilter(cutoff_frequency_hz=400),
            LowpassFilter(cutoff_frequency_hz=3400),
            Distortion(drive_db=8),
            Compressor(threshold_db=-18, ratio=6),
            Gain(gain_db=-3),
        ],
        "radio_broadcast": [
            HighpassFilter(cutoff_frequency_hz=120),
            LowpassFilter(cutoff_frequency_hz=6000),
            Compressor(threshold_db=-20, ratio=8),
            Distortion(drive_db=6),
            Gain(gain_db=2),
        ],
        "walkie_talkie": [
            HighpassFilter(cutoff_frequency_hz=500),
            LowpassFilter(cutoff_frequency_hz=2800),
            Distortion(drive_db=14),
            Bitcrush(bit_depth=10),
            Compressor(threshold_db=-24, ratio=8),
            Gain(gain_db=-5),
        ],
        "bad_signal": [
            HighpassFilter(cutoff_frequency_hz=550),
            LowpassFilter(cutoff_frequency_hz=2600),
            Bitcrush(bit_depth=8),
            Phaser(rate_hz=4.0, depth=0.6, mix=0.45),
            Distortion(drive_db=10),
            Gain(gain_db=-3),
        ],
        "megaphone": [
            HighpassFilter(cutoff_frequency_hz=600),
            LowpassFilter(cutoff_frequency_hz=4000),
            PeakFilter(cutoff_frequency_hz=1800, gain_db=6, q=2.0),
            Distortion(drive_db=14),
            Delay(delay_seconds=0.08, feedback=0.1, mix=0.15),
            Compressor(threshold_db=-15, ratio=4),
            Gain(gain_db=-4),
        ],
        "old_gramophone": [
            HighpassFilter(cutoff_frequency_hz=500),
            LowpassFilter(cutoff_frequency_hz=3000),
            PeakFilter(cutoff_frequency_hz=1200, gain_db=6, q=2.0),
            Bitcrush(bit_depth=11),
            Chorus(rate_hz=0.8, depth=0.25, mix=0.2),
            Distortion(drive_db=8),
            Gain(gain_db=-3),
        ],
        "metallic_helmet": [
            HighpassFilter(cutoff_frequency_hz=300),
            LowpassFilter(cutoff_frequency_hz=4500),
            PeakFilter(cutoff_frequency_hz=900, gain_db=5, q=4.0),
            Delay(delay_seconds=0.012, feedback=0.35, mix=0.3),
            Reverb(room_size=0.08, damping=0.1, wet_level=0.2, dry_level=0.9, width=0.2),
            Compressor(threshold_db=-16, ratio=4),
        ],
        "stadium_announcement": [
            HighpassFilter(cutoff_frequency_hz=350),
            LowpassFilter(cutoff_frequency_hz=5000),
            Distortion(drive_db=6),
            Delay(delay_seconds=0.22, feedback=0.25, mix=0.2),
            Reverb(room_size=0.85, damping=0.4, wet_level=0.3, dry_level=0.85),
            Gain(gain_db=-2),
        ],
        "dream_sequence": [
            LowpassFilter(cutoff_frequency_hz=6000),
            Chorus(rate_hz=0.6, depth=0.3, mix=0.3),
            Phaser(rate_hz=0.3, depth=0.4, mix=0.2),
            Delay(delay_seconds=0.3, feedback=0.25, mix=0.15),
            Reverb(room_size=0.8, damping=0.3, wet_level=0.3, dry_level=0.85),
        ],
        "underwater": [
            LowpassFilter(cutoff_frequency_hz=1000),
            Phaser(rate_hz=1.2, depth=0.5, mix=0.35),
            Chorus(rate_hz=0.9, depth=0.35, mix=0.25),
            Reverb(room_size=0.5, damping=0.8, wet_level=0.2, dry_level=0.85),
            Gain(gain_db=4),
        ],
        "robotic_voice": [
            Chorus(rate_hz=2.5, depth=0.5, centre_delay_ms=12.0, feedback=0.3, mix=0.45),
            Phaser(rate_hz=1.5, depth=0.6, feedback=0.4, mix=0.3),
            Bitcrush(bit_depth=10),
            HighpassFilter(cutoff_frequency_hz=200),
        ],
        "monster_voice": [
            PitchShift(semitones=-5),
            Distortion(drive_db=10),
            LowpassFilter(cutoff_frequency_hz=3800),
            Reverb(room_size=0.5, damping=0.5, wet_level=0.15, dry_level=0.95),
            Gain(gain_db=-2),
        ],
        "intimate_whisper": [
            HighpassFilter(cutoff_frequency_hz=80),
            PeakFilter(cutoff_frequency_hz=160, gain_db=4, q=1.2),
            PeakFilter(cutoff_frequency_hz=6000, gain_db=5, q=1.5),
            Compressor(threshold_db=-26, ratio=6),
            Gain(gain_db=3),
        ],
        "ghost_spirit": [
            HighpassFilter(cutoff_frequency_hz=250),
            PitchShift(semitones=1),
            Chorus(rate_hz=1.2, depth=0.5, feedback=0.2, mix=0.35),
            Delay(delay_seconds=0.16, feedback=0.2, mix=0.15),
            Reverb(room_size=0.8, damping=0.3, wet_level=0.3, dry_level=0.85),
        ],
        "giant_titan": [
            PitchShift(semitones=-6),
            PeakFilter(cutoff_frequency_hz=110, gain_db=6, q=1.0),
            LowpassFilter(cutoff_frequency_hz=3500),
            Distortion(drive_db=6),
            Reverb(room_size=0.8, damping=0.4, wet_level=0.2, dry_level=0.9),
            Compressor(threshold_db=-16, ratio=4),
            Gain(gain_db=-2),
        ],
        "fairy_tiny": [
            PitchShift(semitones=6),
            HighpassFilter(cutoff_frequency_hz=350),
            Chorus(rate_hz=2.2, depth=0.25, mix=0.3),
            Reverb(room_size=0.4, damping=0.3, wet_level=0.2, dry_level=0.9),
        ],
        "alien_voice": [
            PitchShift(semitones=-2),
            Phaser(rate_hz=5.0, depth=0.7, feedback=0.5, mix=0.4),
            Delay(delay_seconds=0.02, feedback=0.3, mix=0.2),
            Chorus(rate_hz=3.0, depth=0.4, mix=0.3),
            HighpassFilter(cutoff_frequency_hz=250),
        ],
        "ancient_god": [
            PitchShift(semitones=-3),
            PeakFilter(cutoff_frequency_hz=140, gain_db=5, q=1.2),
            Chorus(rate_hz=0.8, depth=0.4, feedback=0.2, mix=0.3),
            Delay(delay_seconds=0.12, feedback=0.15, mix=0.12),
            Reverb(room_size=0.85, damping=0.3, wet_level=0.3, dry_level=0.85, width=1.0),
            Compressor(threshold_db=-18, ratio=4),
        ],
    }

    plugins = []
    for effect_name in effects:
        plugins.extend(presets.get(effect_name, ()))
    spaces = [index for index, plugin in enumerate(plugins) if isinstance(plugin, Reverb)]
    keep = set() if in_room else set(spaces[-1:])  # one space per voice, the last one written
    plugins = [plugin for index, plugin in enumerate(plugins) if index not in spaces or index in keep]
    if not plugins:
        return audio
    board = Pedalboard(plugins)
    processed = board(audio, rate)
    return np.asarray(processed, dtype=np.float32).reshape(-1)


def _highpass(audio: np.ndarray, rate: int, cutoff_hz: float) -> np.ndarray:
    nyq = 0.5 * rate
    norm = min(max(cutoff_hz / nyq, 0.001), 0.99)
    sos = signal.butter(2, norm, btype="highpass", output="sos")
    return signal.sosfilt(sos, audio).astype(np.float32, copy=False)


def _lowpass(audio: np.ndarray, rate: int, cutoff_hz: float) -> np.ndarray:
    nyq = 0.5 * rate
    norm = min(max(cutoff_hz / nyq, 0.001), 0.99)
    sos = signal.butter(2, norm, btype="lowpass", output="sos")
    return signal.sosfilt(sos, audio).astype(np.float32, copy=False)


def _bandpass(audio: np.ndarray, rate: int, low_hz: float, high_hz: float) -> np.ndarray:
    return _lowpass(_highpass(audio, rate, low_hz), rate, high_hz)


def _distort(audio: np.ndarray, drive_db: float) -> np.ndarray:
    drive = float(10.0 ** (drive_db / 20.0))
    return np.tanh(audio * drive).astype(np.float32, copy=False)


def _bitcrush(audio: np.ndarray, bits: int) -> np.ndarray:
    levels = float(2 ** max(bits, 2))
    return (np.round(audio * levels) / levels).astype(np.float32, copy=False)


def _delay(audio: np.ndarray, rate: int, delay_seconds: float, feedback: float, mix: float) -> np.ndarray:
    delay_samples = max(1, int(round(delay_seconds * rate)))
    if delay_samples >= audio.size:
        return audio
    wet = np.zeros_like(audio)
    cur = audio.copy()
    gain = 1.0
    for tap in range(1, 5):
        offset = tap * delay_samples
        if offset >= audio.size:
            break
        wet[offset:] += cur[:-offset] * gain
        gain *= feedback
    return ((1.0 - mix) * audio + mix * wet).astype(np.float32, copy=False)


def _reverb(audio: np.ndarray, rate: int, room_size: float, wet_level: float, dry_level: float) -> np.ndarray:
    taps_ms = (23.0, 41.0, 67.0, 97.0, 139.0)
    scale = max(0.2, min(room_size, 1.0))
    wet = np.zeros_like(audio)
    for idx, tap_ms in enumerate(taps_ms):
        offset = max(1, int(round((tap_ms * scale / 1000.0) * rate)))
        if offset < audio.size:
            decay = (0.65 * scale) ** (idx + 1)
            wet[offset:] += audio[:-offset] * decay
    return (dry_level * audio + wet_level * wet).astype(np.float32, copy=False)


def _ring_mod(audio: np.ndarray, rate: int, freq_hz: float, mix: float) -> np.ndarray:
    t = np.arange(audio.size, dtype=np.float32) / float(rate)
    mod = np.sin(2.0 * np.pi * freq_hz * t, dtype=np.float32)
    return ((1.0 - mix) * audio + mix * (audio * mod)).astype(np.float32, copy=False)


def _wsola_stretch(audio: np.ndarray, length: int, window: int = 1024) -> np.ndarray:
    """Time-stretch ``audio`` to ``length`` samples without changing pitch (WSOLA).

    Each 50%-overlapped Hann frame is taken near its nominal input position, at the
    offset that best continues the previous frame, so waveforms join in phase.
    """
    source = np.asarray(audio, dtype=np.float32).reshape(-1)
    if length <= 0:
        return np.zeros(0, dtype=np.float32)
    if source.size < 2 * window:  # too short to overlap frames: plain resampling
        positions = np.linspace(0, max(source.size - 1, 0), num=length)
        return np.interp(positions, np.arange(source.size), source).astype(np.float32)
    hop = window // 2
    tolerance = hop // 2
    padded = np.concatenate([source, np.zeros(window + 2 * tolerance, dtype=np.float32)])
    last_start = padded.size - window
    hann = np.hanning(window + 1)[:-1].astype(np.float32)  # periodic: 50% overlaps sum to one
    frames = -(-length // hop) + 1
    out = np.zeros(frames * hop + window, dtype=np.float32)
    norm = np.zeros_like(out)
    step = source.size / float(length)
    previous = 0
    for frame in range(frames):
        nominal = min(int(round(frame * hop * step)), source.size - 1)
        if frame == 0:
            start = nominal
        else:
            natural = min(previous + hop, last_start)
            low, high = max(0, nominal - tolerance), min(last_start, nominal + tolerance)
            reference = padded[natural:natural + window]
            score = np.correlate(padded[low:high + window], reference, mode="valid")
            start = low + int(np.argmax(score))
        out[frame * hop:frame * hop + window] += padded[start:start + window] * hann
        norm[frame * hop:frame * hop + window] += hann
        previous = start
    return (out[:length] / np.maximum(norm[:length], 1e-3)).astype(np.float32)


def _pitch_shift_preserve_len(audio: np.ndarray, semitones: float) -> np.ndarray:
    """Shift pitch by resampling, then time-stretch back to the original length."""
    if abs(semitones) < 1e-3 or audio.size < 16:
        return audio
    from fractions import Fraction

    ratio = Fraction(2.0 ** (-semitones / 12.0)).limit_denominator(200)  # output/input length
    resampled = signal.resample_poly(np.asarray(audio, dtype=np.float32), ratio.numerator, ratio.denominator)
    return _wsola_stretch(resampled, audio.size)


def _apply_scipy_effect(audio: np.ndarray, rate: int, effect_name: str, in_room: bool = False) -> np.ndarray:
    reverb = (lambda samples, *_args: samples) if in_room else _reverb  # the room is the space
    if effect_name == "phone_call":
        return _distort(_bandpass(audio, rate, 400, 3400), 6.0) * 0.75
    if effect_name == "radio_broadcast":
        return _distort(_bandpass(audio, rate, 120, 6000), 4.0) * 0.9
    if effect_name == "walkie_talkie":
        return _bitcrush(_distort(_bandpass(audio, rate, 500, 2800), 10.0), 10) * 0.65
    if effect_name == "bad_signal":
        return _ring_mod(_bitcrush(_bandpass(audio, rate, 550, 2600), 8), rate, 4.0, 0.3) * 0.75
    if effect_name == "megaphone":
        return _delay(_distort(_bandpass(audio, rate, 600, 4000), 10.0), rate, 0.08, 0.1, 0.15) * 0.7
    if effect_name == "old_gramophone":
        return _bitcrush(_distort(_bandpass(audio, rate, 500, 3000), 6.0), 11) * 0.75
    if effect_name == "metallic_helmet":
        return reverb(_delay(_bandpass(audio, rate, 300, 4500), rate, 0.012, 0.35, 0.3), rate, 0.15, 0.2, 0.9)
    if effect_name == "stadium_announcement":
        return reverb(_delay(_bandpass(audio, rate, 350, 5000), rate, 0.22, 0.25, 0.2), rate, 0.85, 0.3, 0.85)
    if effect_name == "dream_sequence":
        return reverb(_delay(_lowpass(audio, rate, 6000), rate, 0.3, 0.25, 0.15), rate, 0.8, 0.3, 0.85)
    if effect_name == "underwater":
        return reverb(_lowpass(audio, rate, 1000), rate, 0.5, 0.2, 0.85) * 1.3
    if effect_name == "robotic_voice":
        return _bitcrush(_ring_mod(_highpass(audio, rate, 200), rate, 55.0, 0.35), 10)
    if effect_name == "monster_voice":
        return reverb(_lowpass(_distort(_pitch_shift_preserve_len(audio, -5.0), 7.0), rate, 3800), rate, 0.5, 0.15, 0.95)
    if effect_name == "intimate_whisper":
        return (_highpass(audio, rate, 80) * 1.25).astype(np.float32, copy=False)
    if effect_name == "ghost_spirit":
        return reverb(_delay(_highpass(_pitch_shift_preserve_len(audio, 1.0), rate, 250), rate, 0.16, 0.2, 0.15), rate, 0.8, 0.3, 0.85)
    if effect_name == "giant_titan":
        return reverb(_lowpass(_distort(_pitch_shift_preserve_len(audio, -6.0), 5.0), rate, 3500), rate, 0.8, 0.2, 0.9)
    if effect_name == "fairy_tiny":
        return reverb(_highpass(_pitch_shift_preserve_len(audio, 6.0), rate, 350), rate, 0.4, 0.2, 0.9)
    if effect_name == "alien_voice":
        return _delay(_ring_mod(_highpass(_pitch_shift_preserve_len(audio, -2.0), rate, 250), rate, 35.0, 0.3), rate, 0.02, 0.3, 0.2)
    if effect_name == "ancient_god":
        return reverb(_delay(_pitch_shift_preserve_len(audio, -3.0), rate, 0.12, 0.15, 0.12), rate, 0.85, 0.3, 0.85)
    return audio
