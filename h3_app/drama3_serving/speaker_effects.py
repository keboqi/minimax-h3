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

SPEAKER_EFFECT_SET: frozenset[str] = frozenset(SPEAKER_EFFECTS)
_EFFECT_ALTERNATION = "|".join(re.escape(name) for name in SPEAKER_EFFECTS)
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
) -> np.ndarray:
    """Apply chained vocal DSP effects to 1D float32 audio samples."""
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    active = [name.lower() for name in effects if name and name.lower() in SPEAKER_EFFECT_SET]
    if not active or audio.size == 0 or rate <= 0:
        return audio

    out = _try_apply_pedalboard(audio, rate, active)
    if out is None:
        out = audio.copy()
        for effect_name in active:
            out = _apply_scipy_effect(out, rate, effect_name)
    # Drive and gain stages can push either path past full scale.
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out = out * (0.95 / peak)
    return out.astype(np.float32, copy=False)


def _try_apply_pedalboard(
    audio: np.ndarray,
    rate: int,
    effects: list[str],
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

    presets = {
        "phone_call": [
            HighpassFilter(cutoff_frequency_hz=400),
            LowpassFilter(cutoff_frequency_hz=3200),
            Distortion(drive_db=12),
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
            HighpassFilter(cutoff_frequency_hz=550),
            LowpassFilter(cutoff_frequency_hz=2600),
            Distortion(drive_db=22),
            Bitcrush(bit_depth=10),
            Compressor(threshold_db=-24, ratio=10),
            Gain(gain_db=-6),
        ],
        "bad_signal": [
            HighpassFilter(cutoff_frequency_hz=600),
            LowpassFilter(cutoff_frequency_hz=2200),
            Bitcrush(bit_depth=7),
            Phaser(rate_hz=4.0, depth=0.8, mix=0.7),
            Distortion(drive_db=15),
            Gain(gain_db=-4),
        ],
        "megaphone": [
            HighpassFilter(cutoff_frequency_hz=700),
            LowpassFilter(cutoff_frequency_hz=3800),
            PeakFilter(cutoff_frequency_hz=1800, gain_db=10, q=2.5),
            Distortion(drive_db=25),
            Delay(delay_seconds=0.08, feedback=0.2, mix=0.25),
            Compressor(threshold_db=-15, ratio=8),
            Gain(gain_db=-6),
        ],
        "old_gramophone": [
            HighpassFilter(cutoff_frequency_hz=650),
            LowpassFilter(cutoff_frequency_hz=2400),
            PeakFilter(cutoff_frequency_hz=1200, gain_db=8, q=3.0),
            Bitcrush(bit_depth=11),
            Chorus(rate_hz=0.8, depth=0.35, mix=0.3),
            Distortion(drive_db=14),
            Gain(gain_db=-4),
        ],
        "metallic_helmet": [
            HighpassFilter(cutoff_frequency_hz=300),
            LowpassFilter(cutoff_frequency_hz=4000),
            PeakFilter(cutoff_frequency_hz=900, gain_db=6, q=4.0),
            Delay(delay_seconds=0.015, feedback=0.45, mix=0.4),
            Reverb(room_size=0.08, damping=0.1, wet_level=0.35, dry_level=0.75, width=0.2),
            Compressor(threshold_db=-16, ratio=4),
        ],
        "reverb_small_room": [
            Reverb(room_size=0.2, damping=0.6, wet_level=0.25, dry_level=0.85, width=0.5),
        ],
        "reverb_large_hall": [
            Reverb(room_size=0.85, damping=0.3, wet_level=0.45, dry_level=0.7, width=1.0),
        ],
        "stadium_announcement": [
            HighpassFilter(cutoff_frequency_hz=450),
            LowpassFilter(cutoff_frequency_hz=4200),
            Distortion(drive_db=10),
            Delay(delay_seconds=0.22, feedback=0.45, mix=0.45),
            Reverb(room_size=0.92, damping=0.35, wet_level=0.5, dry_level=0.6),
            Gain(gain_db=-2),
        ],
        "dream_sequence": [
            LowpassFilter(cutoff_frequency_hz=5000),
            Chorus(rate_hz=0.6, depth=0.4, mix=0.4),
            Phaser(rate_hz=0.3, depth=0.5, mix=0.3),
            Delay(delay_seconds=0.3, feedback=0.4, mix=0.3),
            Reverb(room_size=0.9, damping=0.2, wet_level=0.55, dry_level=0.55),
        ],
        "underwater": [
            LowpassFilter(cutoff_frequency_hz=450),
            Phaser(rate_hz=1.8, depth=0.7, mix=0.6),
            Chorus(rate_hz=1.2, depth=0.5, mix=0.4),
            Reverb(room_size=0.6, damping=0.8, wet_level=0.4, dry_level=0.6),
            Gain(gain_db=4),
        ],
        "robotic_voice": [
            Chorus(rate_hz=2.5, depth=0.6, centre_delay_ms=12.0, feedback=0.5, mix=0.6),
            Phaser(rate_hz=1.5, depth=0.8, feedback=0.6, mix=0.5),
            Bitcrush(bit_depth=10),
            HighpassFilter(cutoff_frequency_hz=200),
        ],
        "monster_voice": [
            PitchShift(semitones=-6),
            Distortion(drive_db=15),
            LowpassFilter(cutoff_frequency_hz=2800),
            Reverb(room_size=0.6, damping=0.5, wet_level=0.3, dry_level=0.8),
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
            HighpassFilter(cutoff_frequency_hz=350),
            PitchShift(semitones=2),
            Chorus(rate_hz=1.5, depth=0.7, feedback=0.4, mix=0.6),
            Delay(delay_seconds=0.18, feedback=0.55, mix=0.45),
            Reverb(room_size=0.95, damping=0.15, wet_level=0.65, dry_level=0.45),
        ],
        "giant_titan": [
            PitchShift(semitones=-8),
            PeakFilter(cutoff_frequency_hz=110, gain_db=8, q=1.0),
            LowpassFilter(cutoff_frequency_hz=2200),
            Distortion(drive_db=10),
            Reverb(room_size=0.88, damping=0.4, wet_level=0.4, dry_level=0.75),
            Compressor(threshold_db=-16, ratio=5),
            Gain(gain_db=-2),
        ],
        "fairy_tiny": [
            PitchShift(semitones=6),
            HighpassFilter(cutoff_frequency_hz=350),
            Chorus(rate_hz=2.2, depth=0.25, mix=0.35),
            Reverb(room_size=0.4, damping=0.3, wet_level=0.25, dry_level=0.85),
        ],
        "alien_voice": [
            PitchShift(semitones=-2),
            Phaser(rate_hz=6.0, depth=0.9, feedback=0.75, mix=0.7),
            Delay(delay_seconds=0.025, feedback=0.65, mix=0.5),
            Chorus(rate_hz=3.5, depth=0.6, mix=0.5),
            HighpassFilter(cutoff_frequency_hz=250),
        ],
        "ancient_god": [
            PitchShift(semitones=-3),
            PeakFilter(cutoff_frequency_hz=140, gain_db=5, q=1.2),
            Chorus(rate_hz=0.8, depth=0.5, feedback=0.3, mix=0.5),
            Delay(delay_seconds=0.12, feedback=0.35, mix=0.35),
            Reverb(room_size=0.95, damping=0.2, wet_level=0.5, dry_level=0.65, width=1.0),
            Compressor(threshold_db=-18, ratio=4),
        ],
    }

    plugins = []
    for effect_name in effects:
        plugins.extend(presets.get(effect_name, ()))
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


def _apply_scipy_effect(audio: np.ndarray, rate: int, effect_name: str) -> np.ndarray:
    if effect_name == "phone_call":
        return _distort(_bandpass(audio, rate, 400, 3200), 8.0) * 0.75
    if effect_name == "radio_broadcast":
        return _distort(_bandpass(audio, rate, 120, 6000), 4.0) * 0.9
    if effect_name == "walkie_talkie":
        return _bitcrush(_distort(_bandpass(audio, rate, 550, 2600), 14.0), 10) * 0.65
    if effect_name == "bad_signal":
        return _ring_mod(_bitcrush(_bandpass(audio, rate, 600, 2200), 7), rate, 4.0, 0.35) * 0.75
    if effect_name == "megaphone":
        return _delay(_distort(_bandpass(audio, rate, 700, 3800), 16.0), rate, 0.08, 0.2, 0.25) * 0.65
    if effect_name == "old_gramophone":
        return _bitcrush(_distort(_bandpass(audio, rate, 650, 2400), 10.0), 11) * 0.75
    if effect_name == "metallic_helmet":
        return _reverb(_delay(_bandpass(audio, rate, 300, 4000), rate, 0.015, 0.45, 0.4), rate, 0.15, 0.35, 0.75)
    if effect_name == "reverb_small_room":
        return _reverb(audio, rate, 0.2, 0.25, 0.85)
    if effect_name == "reverb_large_hall":
        return _reverb(audio, rate, 0.85, 0.45, 0.7)
    if effect_name == "stadium_announcement":
        return _reverb(_delay(_bandpass(audio, rate, 450, 4200), rate, 0.22, 0.45, 0.45), rate, 0.92, 0.5, 0.6)
    if effect_name == "dream_sequence":
        return _reverb(_delay(_lowpass(audio, rate, 5000), rate, 0.3, 0.4, 0.3), rate, 0.9, 0.55, 0.55)
    if effect_name == "underwater":
        return _reverb(_lowpass(audio, rate, 450), rate, 0.6, 0.4, 0.6) * 1.3
    if effect_name == "robotic_voice":
        return _bitcrush(_ring_mod(_highpass(audio, rate, 200), rate, 55.0, 0.45), 10)
    if effect_name == "monster_voice":
        return _reverb(_lowpass(_distort(_pitch_shift_preserve_len(audio, -6.0), 10.0), rate, 2800), rate, 0.6, 0.3, 0.8)
    if effect_name == "intimate_whisper":
        return (_highpass(audio, rate, 80) * 1.25).astype(np.float32, copy=False)
    if effect_name == "ghost_spirit":
        return _reverb(_delay(_highpass(_pitch_shift_preserve_len(audio, 2.0), rate, 350), rate, 0.18, 0.55, 0.45), rate, 0.95, 0.65, 0.45)
    if effect_name == "giant_titan":
        return _reverb(_lowpass(_distort(_pitch_shift_preserve_len(audio, -8.0), 8.0), rate, 2200), rate, 0.88, 0.4, 0.75)
    if effect_name == "fairy_tiny":
        return _reverb(_highpass(_pitch_shift_preserve_len(audio, 6.0), rate, 350), rate, 0.4, 0.25, 0.85)
    if effect_name == "alien_voice":
        return _delay(_ring_mod(_highpass(_pitch_shift_preserve_len(audio, -2.0), rate, 250), rate, 35.0, 0.45), rate, 0.025, 0.65, 0.5)
    if effect_name == "ancient_god":
        return _reverb(_delay(_pitch_shift_preserve_len(audio, -3.0), rate, 0.12, 0.35, 0.35), rate, 0.95, 0.5, 0.65)
    return audio
