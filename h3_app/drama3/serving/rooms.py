"""Rooms a scene is heard in: generated stereo room responses, convolved beside the dry sound.

Each room's response is computed once in numpy: early reflections from a shoebox room
(image sources up to the third order, heard at two ears), then a diffuse tail of
independent noise per channel that decays faster in the highs, as real rooms do. The
direct sound is left out, so the response is only the room: it is mixed beside the dry
voice, whose level and position stay as they are, and its tail rings on past the take.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import zlib

import numpy as np

SPEED_OF_SOUND = 343.0
EAR_OFFSET = 0.09  # metres from the head's centre to each ear
MAX_ORDER = 3
LOW_SPLIT_HZ, HIGH_SPLIT_HZ = 300.0, 3000.0
LOW_DECAY, HIGH_DECAY = 1.1, 0.5  # each band's decay time as a share of the mid band's
EARLY_SHARE = 0.35  # the early reflections' share of a room's energy; the tail has the rest
OPEN_AIR_EARLY_SHARE = 0.8
MAX_SECONDS = 3.0


@dataclass(frozen=True)
class Room:
    size: tuple[float, float, float]  # width, depth and height in metres
    rt60: float  # seconds for the mid band to fall 60 dB
    send: float  # the room's level beside the dry sound
    source_distance: float = 1.5  # metres from the listener to the voice
    predelay: float = 0.0  # extra seconds before the first reflection
    open_air: bool = False  # outdoors: the ground reflects, distant walls barely
    low_cut: float = 180.0  # the room's own lows and highs are rolled off so it never muddies speech
    high_cut: float = 7000.0


ROOMS = {
    "small_room": Room((3.8, 3.2, 2.5), 0.42, 0.24, source_distance=1.2),
    "medium_room": Room((6.5, 5.0, 2.8), 0.7, 0.27, source_distance=1.8, predelay=0.004),
    "large_hall": Room((26.0, 18.0, 11.0), 1.9, 0.34, source_distance=4.0, predelay=0.015,
                       low_cut=140.0, high_cut=6000.0),
    "outdoors": Room((60.0, 60.0, 30.0), 0.35, 0.12, source_distance=2.0, open_air=True),
}
# Places that are a sound of their own rather than a real room; their vocal effects make them.
STYLIZED_ROOMS = ("underwater", "dream_sequence")
SCENE_ROOMS = (*ROOMS, *STYLIZED_ROOMS)


def scene_room_name(value: str) -> str:
    """The scene room ``value`` names (``reverb_`` prefixes and spaces allowed), or "" for none."""
    name = "_".join(str(value or "").strip().lower().replace("-", " ").split())
    name = name.removeprefix("reverb_")
    return name if name in SCENE_ROOMS else ""


def _image(index: int, source: float, length: float) -> float:
    """The image of ``source`` after ``index`` reflections along a wall pair ``length`` apart."""
    return index * length + (source if index % 2 == 0 else length - source)


def _early(room: Room, rate: int) -> tuple[np.ndarray, float]:
    """(early reflections at both ears, mixing time in seconds), times relative to the direct sound."""
    width, depth, height = room.size
    listener = np.array([width / 2, depth * 0.35, min(1.6, height - 0.3)])
    source = listener + np.array([0.0, min(room.source_distance, depth * 0.6), 0.0])
    volume = width * depth * height
    surface = 2 * (width * depth + width * height + depth * height)
    if room.open_air:
        order = 1
        walls, floor = 0.3, 0.75
    else:
        order = MAX_ORDER
        alpha = float(np.clip(0.161 * volume / (surface * room.rt60), 0.05, 0.95))  # Sabine
        walls = floor = float(np.sqrt(1 - alpha))
    direct = float(np.linalg.norm(source - listener))
    taps = []
    for nx in range(-order, order + 1):
        for ny in range(-order, order + 1):
            for nz in range(-order, order + 1):
                reflections = abs(nx) + abs(ny) + abs(nz)
                if not 0 < reflections <= order or (room.open_air and nz > 0):
                    continue
                image = np.array([_image(nx, source[0], width), _image(ny, source[1], depth),
                                  _image(nz, source[2], height)])
                gain = walls ** (abs(nx) + abs(ny)) * floor ** abs(nz)
                taps.append((image, gain))
    longest = max(np.linalg.norm(image - listener) for image, _gain in taps) + EAR_OFFSET
    early = np.zeros((int((longest - direct) / SPEED_OF_SOUND * rate) + 2, 2), dtype=np.float64)
    for image, gain in taps:
        toward = image - listener
        side = toward[0] / max(np.linalg.norm(toward), 1e-6)  # +1: from the right
        for channel, sign in ((0, -1.0), (1, 1.0)):
            ear = listener + np.array([sign * EAR_OFFSET, 0.0, 0.0])
            distance = float(np.linalg.norm(image - ear))
            # The head shadows a reflection arriving from the other side.
            shadow = 0.65 + 0.35 * (1 + sign * side) / 2
            delay = max(0, int(round((distance - direct) / SPEED_OF_SOUND * rate)))
            early[delay, channel] += gain * shadow / max(distance, 0.5)
    mixing = float(np.clip(3 * 4 * volume / surface / SPEED_OF_SOUND, 0.012, 0.09))
    return early, mixing


def _late(room: Room, rate: int, mixing: float, length: int, seed: int) -> np.ndarray:
    """A diffuse tail, independent per channel, building up until ``mixing`` seconds."""
    from scipy import signal

    rng = np.random.default_rng(seed)
    t = np.arange(length) / rate
    low, high = LOW_SPLIT_HZ / (0.5 * rate), min(HIGH_SPLIT_HZ / (0.5 * rate), 0.99)
    # Each band is its own noise, so the bands never cancel where they meet.
    bands = ((signal.butter(4, low, btype="lowpass", output="sos"), LOW_DECAY),
             (signal.butter(2, [low, high], btype="bandpass", output="sos"), 1.0),
             (signal.butter(4, high, btype="highpass", output="sos"), HIGH_DECAY))
    onset = np.clip(t / max(mixing, 1e-3), 0.0, 1.0)
    build = 0.5 - 0.5 * np.cos(np.pi * onset)
    tail = np.zeros((length, 2), dtype=np.float64)
    for channel in range(2):
        for sos, share in bands:
            band = signal.sosfilt(sos, rng.standard_normal(length))
            tail[:, channel] += band * np.exp(-6.9078 * t / (room.rt60 * share))
        tail[:, channel] *= build
    return tail


@lru_cache(maxsize=32)
def room_response(name: str, rate: int) -> np.ndarray:
    """``name``'s stereo response at ``rate``, (frames, 2) float32 with unit energy per channel."""
    from scipy import signal

    room = ROOMS[name]
    early, mixing = _early(room, rate)
    length = min(int((mixing + 1.2 * room.rt60 * LOW_DECAY) * rate), int(MAX_SECONDS * rate))
    length = max(length, len(early))
    late = _late(room, rate, mixing, length, zlib.crc32(name.encode()))
    early = np.pad(early, ((0, length - len(early)), (0, 0)))
    share = OPEN_AIR_EARLY_SHARE if room.open_air else EARLY_SHARE

    def unit(part):
        energy = float(np.sum(part ** 2)) / 2
        return part / np.sqrt(energy) if energy > 0 else part

    response = np.sqrt(share) * unit(early) + np.sqrt(1 - share) * unit(late)
    band = signal.butter(2, [room.low_cut / (0.5 * rate), min(room.high_cut / (0.5 * rate), 0.99)],
                         btype="bandpass", output="sos")
    response = unit(signal.sosfilt(band, response, axis=0))
    response = np.pad(response, ((int(room.predelay * rate), 0), (0, 0)))
    response = response.astype(np.float32)
    response.setflags(write=False)
    return response


def room_wet(samples: np.ndarray, rate: int, name: str, *, send: float = 1.0) -> np.ndarray:
    """The room ``name`` adds to a mono sound: stereo, as long as the sound plus the room's tail.

    ``send`` scales the room's own level, as ``near`` voices take less of it.
    """
    from scipy import signal

    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not mono.size or name not in ROOMS:
        return np.zeros((len(mono), 2), dtype=np.float32)
    response = room_response(name, rate)
    wet = np.stack([signal.oaconvolve(mono, response[:, channel]) for channel in range(2)], axis=1)
    return (wet * (ROOMS[name].send * send)).astype(np.float32, copy=False)
