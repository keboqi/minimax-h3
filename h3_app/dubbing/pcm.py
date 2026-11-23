"""PCM audio shared by the dubbing timeline and clone references."""

from pathlib import Path
import wave

import numpy as np

FITTED_AUDIO_RATE = 44100


def read_slice(path: Path, start: float, end: float) -> np.ndarray:
    """Read and resample a mono slice of a 16-bit PCM WAV."""
    from h3_app.drama3.serving.audio import resample_audio

    with wave.open(str(path), "rb") as source:
        rate, channels, frames = source.getframerate(), source.getnchannels(), source.getnframes()
        first = max(0, min(frames, int(round(start * rate))))
        last = max(first, min(frames, int(round(end * rate))))
        source.setpos(first)
        pcm = np.frombuffer(source.readframes(last - first), dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm[:len(pcm) // channels * channels].reshape(-1, channels).mean(axis=1)
    return pcm if rate == FITTED_AUDIO_RATE else resample_audio(pcm, rate, FITTED_AUDIO_RATE)


def write_wav(samples: np.ndarray, rate: int, target: Path) -> Path:
    from h3_app.drama3.serving.audio import encode_wav

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(encode_wav(samples, rate))
    return target
