"""Chunked, parallel Drama 3 generation through a managed SGLang-Omni server."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
import contextvars
from dataclasses import dataclass, field, replace
import hashlib
from pathlib import Path
import re
import shutil
import threading
import time
from typing import Any, Callable, Iterator
import uuid

import numpy as np
import requests

from h3_app import stable_audio
from h3_app.drama3 import sounds as drama3_sounds
from h3_app.errors import H3Error
from h3_app.generation.results import GenerationUpdate, fresh_audio
from h3_app.jobs import CURRENT_JOB
from h3_app.provenance import write_snapshot
from h3_app.workflows.drama3 import MAX_FRAMES, SPEAKER_TAG, request_cast, speaker_label

from . import audio, environment, rooms, takes as take_store
from .planner import Chunk, expected_seconds, plan_turns_chunks
from .server import ServerSettings, SpeechRequestError, ensure_model, persist_caches, shared_server
from .speaker_effects import DISTANCE_TAGS, ROOM_EFFECT_ROOMS, apply_speaker_effects
from .tree import (
    ParallelNode,
    PauseNode,
    SequentialNode,
    SoundNode,
    SpeechNode,
    TreeNode,
    is_sound_only_node,
    iter_sound_nodes,
    iter_speech_nodes,
    parse_drama3_tree,
    strip_drama3_sound_tags,
)

LABEL = "SGLang-Omni"
FRAMES_PER_SECOND = 21.5
# SGLang-Omni's Drama 3 request builder rejects max_new_tokens above this (~95 s of speech).
SERVER_MAX_NEW_TOKENS = 2048
HEARTBEAT_SECONDS = 5.0
# A take that clearly failed its check is generated again this many times even with retries off.
AUTOMATIC_RETRIES = 1
# Dropped connections and server errors are retried this many times, independent of take checks.
TRANSPORT_RETRIES = 2
TRANSPORT_BACKOFF_SECONDS = 2.0


@dataclass
class Take:
    chunk: Chunk
    samples: np.ndarray
    rate: int
    attempts: int  # 0 for a saved take
    check: audio.TakeCheck | None
    wav: bytes = b""  # the server's response, saved for reuse
    line_bounds: list[tuple[int, int]] | None = None  # tidied audio at take.rate
    line_words: list[list[tuple[str, float, float]]] | None = None  # each line's words, seconds in the tidied take


def frame_budget(text: str, limit: int) -> int:
    """Stop a runaway take early: generous for the text, never above the request's cap."""
    seconds = expected_seconds(text) * 2.5 + 5.0
    return int(max(256, min(limit, MAX_FRAMES, SERVER_MAX_NEW_TOKENS, seconds * FRAMES_PER_SECOND)))


@contextmanager
def _request_pool(workers: int, prefix: str):
    """A request pool whose exit never waits for in-flight requests.

    An interrupt or a failed chunk returns at once instead of after the slowest
    running request (up to the request timeout); ``stopped`` tells running
    workers not to send further attempts.
    """
    stopped = threading.Event()
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=prefix)
    try:
        yield pool, stopped
    finally:
        stopped.set()
        pool.shutdown(wait=False, cancel_futures=True)


def _request_speech(server, payload: dict, stopped: threading.Event) -> bytes:
    """One speech request, retried after a dropped connection or an HTTP 5xx."""
    for attempt in range(TRANSPORT_RETRIES + 1):
        if stopped.is_set():
            raise H3Error("Drama 3 generation was stopped.")
        try:
            return server.speech(payload)
        except (requests.ConnectionError, SpeechRequestError) as exc:
            if (isinstance(exc, SpeechRequestError) and exc.status < 500) or attempt == TRANSPORT_RETRIES:
                raise
        stopped.wait(TRANSPORT_BACKOFF_SECONDS * (attempt + 1))
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class Attempt:
    wav: bytes
    samples: np.ndarray
    rate: int
    check: audio.TakeCheck
    number: int  # 1-based attempt that produced this take


def best_take(server, *, text: str, reference: Path | None, transcript: str | None, max_new_tokens: int,
              sampling: dict, attempts: int, stopped: threading.Event) -> Attempt:
    """Generate up to ``attempts`` takes of one line; the first that passes its check, else the best."""
    best = None
    for number in range(1, attempts + 1):
        payload = server.payload(text=text, reference=reference, transcript=transcript,
                                 max_new_tokens=max_new_tokens, sampling=sampling)
        wav = _request_speech(server, payload, stopped)
        samples, rate = audio.decode_wav(wav)
        check = audio.check_take(samples, rate, expected_seconds(text),
                                 max_seconds=max_new_tokens / FRAMES_PER_SECOND)
        attempt = Attempt(wav, samples, rate, check, number)
        if check.ok:
            return attempt
        if best is None or check.score < best.check.score:
            best = attempt
    return best


# (resolved path, size, mtime_ns) -> content digest, so unchanged clips are not re-hashed every job.
_REFERENCE_DIGESTS: dict[tuple[str, int, int], str] = {}


def _reference_digest(source: Path) -> str:
    stat = source.stat()
    key = (str(source.resolve()), stat.st_size, stat.st_mtime_ns)
    digest = _REFERENCE_DIGESTS.get(key)
    if digest is None:
        digest = _REFERENCE_DIGESTS[key] = hashlib.sha256(source.read_bytes()).hexdigest()[:24]
    return digest


def _staged_reference(path: str, media_dir: Path) -> Path:
    """Copy a clip under the server's allowed media path, keyed by content."""
    source = Path(path)
    digest = _reference_digest(source)
    target = media_dir / "references" / f"{digest}{source.suffix.lower() or '.wav'}"
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return target


def _with_heartbeat(label: str, work: Callable[[Callable[[str], None]], object]) -> Iterator[GenerationUpdate]:
    """Run blocking work in a thread, reporting its latest progress message.

    The thread runs in this job's context, so the work's cancellation checks see an
    Interrupt. The generator returns the work's result (``result = yield from ...``).
    """
    context = contextvars.copy_context()
    result, error, latest = [], [], [label]

    def run():
        try:
            result.append(context.run(work, lambda message: latest.__setitem__(0, message)))
        except BaseException as exc:  # surfaced on the generator's thread
            error.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    started = time.monotonic()
    thread.start()
    while thread.is_alive():
        thread.join(HEARTBEAT_SECONDS)
        if thread.is_alive():
            yield GenerationUpdate(None, f"{latest[0]} ({time.monotonic() - started:.0f}s)")
    if error:
        raise error[0]
    return result[0]


def _ready_server(settings, server, label, free_gpu, hf_token, *, evict=True) -> Iterator[GenerationUpdate]:
    yield from _with_heartbeat(f"Preparing the {label} environment",
                               lambda _progress: environment.prepare(settings.venv))
    yield from _with_heartbeat("Downloading the Drama 3 checkpoint",
                               lambda _progress: ensure_model(settings, token=hf_token))
    yield from _with_heartbeat(f"Starting {label}",
                               lambda report: server.ensure_ready(free_gpu, report, evict=evict))


def preload_served(*, runtime, free_gpu: Callable[[], None], hf_token=None,
                   label: str = LABEL) -> Iterator[GenerationUpdate]:
    """Prepare, download and start (or wake) the server before its first request.

    The VoiceDesign/Stable Audio worker stays loaded; without room beside it this raises.
    """
    settings = ServerSettings.from_environment(runtime)
    yield from _ready_server(settings, shared_server(settings), label, free_gpu, hf_token, evict=False)


@dataclass(frozen=True)
class DesignedVoice:
    key: object
    error: Exception | None = None  # None: the clip was written to its target


def design_voices_served(takes: dict, *, runtime, free_gpu: Callable[[], None], hf_token=None,
                         sampling: dict, max_frames: int, parallel: int = 1,
                         label: str = LABEL) -> Iterator[GenerationUpdate | DesignedVoice]:
    """Generate ``{key: (text, target)}`` concurrently after one server preparation.

    Each take is one unchunked request without references, so a single invented
    voice speaks all of it. Yields status updates and one ``DesignedVoice`` per take;
    a failed take does not stop the others.
    """
    settings = ServerSettings.from_environment(runtime)
    server = shared_server(settings)
    yield from _ready_server(settings, server, label, free_gpu, hf_token)

    def synthesize(text: str, target: Path, stopped: threading.Event) -> None:
        payload = server.payload(text=text, reference=None, transcript=None,
                                 max_new_tokens=frame_budget(text, max_frames), sampling=sampling)
        samples, rate = audio.decode_wav(_request_speech(server, payload, stopped))
        samples = audio.tidy_take(samples, rate)
        if not samples.size:
            raise H3Error(f"Drama 3 ({label}) returned a silent voice take; generate again.")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(audio.encode_wav(samples, rate))

    total, done, started = len(takes), 0, time.monotonic()
    workers = max(1, min(int(parallel), total))
    yield GenerationUpdate(None, f"Generating {total} voice(s) on {label}, {workers} at a time")
    with _request_pool(workers, "drama3-voice") as (pool, stopped):
        pending = {pool.submit(synthesize, text, target, stopped): key for key, (text, target) in takes.items()}
        while pending:
            finished, _ = wait(pending, timeout=HEARTBEAT_SECONDS, return_when=FIRST_COMPLETED)
            job = CURRENT_JOB.get()
            if job is not None and job.cancelled.is_set():
                for future in pending:
                    future.cancel()
                job.check()
            for future in finished:
                done += 1
                yield DesignedVoice(pending.pop(future), future.exception())
            yield GenerationUpdate(None, f"Generated {done}/{total} voice(s) on {label} "
                                         f"({time.monotonic() - started:.0f}s)")
    persist_caches()


def design_voice_served(text: str, target: Path, **kwargs) -> Iterator[GenerationUpdate]:
    """One voice take; raises its failure."""
    for update in design_voices_served({0: (text, target)}, **kwargs):
        if isinstance(update, DesignedVoice):
            if update.error is not None:
                raise update.error
        else:
            yield update


def _resolve_sounds_for_tree(
    sound_nodes: list[SoundNode],
    request,
    runtime,
    free_gpu: Callable[[], None],
    hf_token=None,
) -> Iterator[GenerationUpdate | dict[str, Path]]:
    """Resolve or generate audio files for all `<sound>` nodes in the script tree."""
    clip_by_key: dict[str, Path] = {}
    override_meta: dict[str, tuple[str, int]] = {}

    for name, audio_path, description, duration in getattr(request, "sounds", ()):
        clean_name = str(name or "").strip().casefold()
        clean_desc = str(description or "").strip()
        dur = max(1, min(int(duration or 5), 47))
        if clean_name and clean_desc:
            override_meta[clean_name] = (clean_desc, dur)
        if audio_path:
            candidate = Path(str(audio_path))
            if candidate.is_file():
                if clean_name:
                    clip_by_key[clean_name] = candidate
                if clean_desc:
                    clip_by_key[clean_desc.casefold()] = candidate

    for node in sound_nodes:
        for lookup_name in (node.name,):
            if not lookup_name:
                continue
            key = lookup_name.strip().casefold()
            if key in clip_by_key:
                continue
            preset = drama3_sounds.find_sound(lookup_name, runtime)
            if preset is not None and preset.audio.is_file():
                clip_by_key[key] = preset.audio

    missing_keys: list[str] = []
    missing_specs: list[tuple[str, int]] = []
    for node in sound_nodes:
        name_key = (node.name or "").strip().casefold()
        desc_key = (node.description or "").strip().casefold()
        if (name_key and name_key in clip_by_key) or (desc_key and desc_key in clip_by_key):
            continue
        effective_desc, effective_dur = override_meta.get(name_key, (node.description, node.duration))
        target_key = name_key or desc_key
        if target_key and target_key not in missing_keys:
            missing_keys.append(target_key)
            missing_specs.append((effective_desc, effective_dur))

    if missing_specs:
        generated_result: list[Any] = []
        yield from _with_heartbeat(
            f"Generating {len(missing_specs)} sound(s) with Stable Audio 3 Medium",
            lambda _report: generated_result.append(
                stable_audio.generate_sound_references(
                    missing_specs,
                    runtime,
                    free_gpu=free_gpu,
                    use_cache=True,
                    hf_token=hf_token,
                )
            ),
        )
        batch = generated_result[0] if generated_result else []
        if isinstance(batch, dict):
            for key, path in batch.items():
                clip_by_key[str(key).casefold()] = Path(path)
        else:
            for key, path in zip(missing_keys, batch):
                clip_by_key[key] = Path(path)

    yield clip_by_key


def _sound_clip(node: SoundNode, sound_clips: dict[str, Path]) -> Path | None:
    name_key = (node.name or "").strip().casefold()
    desc_key = (node.description or "").strip().casefold()
    return sound_clips.get(name_key) or sound_clips.get(desc_key)


def _decoded_sound(clip_path: Path, rate: int, decoded: dict[Path, np.ndarray]) -> np.ndarray:
    """Decode each clip once per render; a cue repeated in the script reuses its samples."""
    if clip_path not in decoded:
        decoded[clip_path] = audio.load_audio_file(clip_path, rate, stereo=True)
    return decoded[clip_path]


def _placed_sound(node: SoundNode, ctx: "_Render") -> np.ndarray:
    """A sound cue's stereo clip at its volume, position and distance."""
    clip_path = _sound_clip(node, ctx.sound_clips)
    if clip_path is None:
        raise H3Error(f'Missing sound clip for "{node.name or node.description}".')
    samples = _decoded_sound(clip_path, ctx.rate, ctx.decoded)
    if not node.loop:  # a one-shot starts on its first sound, not on the clip's dead air
        samples = audio.trim_onset(samples, ctx.rate)
    if node.pan is not None or node.distance:
        samples = audio.place_sound(samples, ctx.rate, node.pan, node.distance)
    # A one-shot happens in the scene's room, as its voices do; beds are that room already.
    if ctx.room in rooms.ROOMS and not node.loop:
        wet = rooms.room_wet(samples.mean(axis=1), ctx.rate, ctx.room)
        placed = np.zeros((len(wet), samples.shape[1]), dtype=np.float32)
        placed[: len(samples)] = samples
        samples = placed + wet  # the room rings on past the clip
    elif ctx.room and not node.loop:
        samples = np.stack([apply_speaker_effects(samples[:, channel], ctx.rate, (ctx.room,))[: len(samples)]
                            for channel in range(samples.shape[1])], axis=1)
    return samples


# Place effects that describe the space a scene happens in, which its sounds share.
ROOM_EFFECTS = ("reverb_small_room", "reverb_large_hall", "underwater", "dream_sequence")
# A voice heard through a device or in a space of its own is not in the scene's room.
OWN_SPACE_EFFECTS = frozenset({"phone_call", "radio_broadcast", "walkie_talkie", "bad_signal", "old_gramophone",
                               "stadium_announcement", "underwater", "dream_sequence"})
NEAR_ROOM_SEND = 0.6  # a near voice is heard more directly, with less of the room


def _scene_room(node: TreeNode, ctx: "_Render") -> str | None:
    """The room of the scene ``node`` holds: its ``room``, else the one place its voices agree on."""
    if getattr(node, "room", ""):
        return node.room
    places = {ROOM_EFFECT_ROOMS.get(effect, effect) for speech in iter_speech_nodes(node)
              for chunk in ctx.node_chunks.get(id(speech), []) for effect in chunk.speaker_effects
              if effect in ROOM_EFFECTS}
    return places.pop() if len(places) == 1 else None


def _voice_room(chunk: Chunk, ctx: "_Render") -> str | None:
    """The room a line is heard in: its own place tag, else its scene's ``room`` for an in-world voice.

    The narrator, a voice through a device and a voice in a space of its own stay out of
    the scene's room.
    """
    own = [ROOM_EFFECT_ROOMS[effect] for effect in chunk.speaker_effects if effect in ROOM_EFFECT_ROOMS]
    if own:
        return own[0]
    if ctx.scene_room is None or is_narrator(chunk.speaker):
        return None
    if any(effect in OWN_SPACE_EFFECTS for effect in chunk.speaker_effects):
        return None
    return ctx.scene_room


# Speakers' takes are brought this close to the cast's typical level, never by more.
MAX_SPEAKER_GAIN_DB = 6.0


def speaker_gains(takes: dict[int, Take]) -> dict[str, float]:
    """A gain per speaker that evens out how loud each voice clip makes its lines.

    Each speaker's voiced level across all their takes is moved toward the median of the
    speakers, by at most ``MAX_SPEAKER_GAIN_DB``; whispers and shouts within one speaker
    keep their difference.
    """
    energy: dict[str, float] = {}
    length: dict[str, int] = {}
    for take in takes.values():
        level = audio.active_level_db(take.samples, take.rate)
        if level is not None:
            speaker = take.chunk.speaker
            energy[speaker] = energy.get(speaker, 0.0) + 10.0 ** (level / 10.0) * len(take.samples)
            length[speaker] = length.get(speaker, 0) + len(take.samples)
    levels = {speaker: 10.0 * np.log10(energy[speaker] / length[speaker]) for speaker in energy}
    if len(levels) < 2:
        return {}
    target = float(np.median(list(levels.values())))
    return {speaker: float(10.0 ** (np.clip(target - level, -MAX_SPEAKER_GAIN_DB, MAX_SPEAKER_GAIN_DB) / 20.0))
            for speaker, level in levels.items()}


# The narrator stands in the middle; characters take these places in order of first line.
NARRATOR_NAMES = frozenset({"narrator", "narration", "旁白", "storyteller", "story teller"})
CHARACTER_POSITIONS = (-0.3, 0.3, -0.15, 0.15, -0.45, 0.45)


def is_narrator(speaker: str) -> bool:
    return " ".join(str(speaker).split()).casefold() in NARRATOR_NAMES


def speaker_positions(chunks: list[Chunk]) -> dict[str, float]:
    """Where each speaker stands in the stereo image.

    The narrator is centred and close; with two or more characters they spread gently
    to both sides, alternating, so a conversation is heard across the scene. A lone
    voice or a single character stays centred.
    """
    characters = [speaker for speaker in dict.fromkeys(chunk.speaker for chunk in chunks)
                  if not is_narrator(speaker)]
    if len(characters) < 2:
        return {}
    return {speaker: CHARACTER_POSITIONS[index % len(CHARACTER_POSITIONS)]
            for index, speaker in enumerate(characters)}


@dataclass
class _Render:
    node_chunks: dict[int, list[Chunk]]
    takes: dict[int, Take]
    sound_clips: dict[str, Path]
    rate: int
    decoded: dict[Path, np.ndarray]
    gains: dict[str, float]
    positions: dict[str, float]
    room: str | None = None  # the enclosing scene's room or place effect, heard by its one-shot sounds
    scene_room: str | None = None  # the enclosing scene's ``room``, heard by its in-world voices too
    placements: dict[int, float] = field(default_factory=dict)  # each chunk's stereo position, as mixed
    words: dict[int, list[list[tuple[str, int, int]]]] = field(default_factory=dict)  # samples inside the take


def _edge_texts(node: TreeNode, ctx: _Render) -> tuple[str, str, str, str] | None:
    """(first speaker, first text, last speaker, last text) of the speech in ``node``."""
    chunks = [chunk for speech in iter_speech_nodes(node) for chunk in ctx.node_chunks.get(id(speech), [])]
    if not chunks:
        return None
    return chunks[0].speaker, chunks[0].text, chunks[-1].speaker, chunks[-1].text


# Where things play in a render, in samples: a chunk's (start, end, its lines' (start, end) inside
# the take) keyed by chunk index, and a sound cue's (start, end, {name, loop}) keyed ("sound", id).
Spans = dict[int | tuple[str, int], tuple[int, int, list[tuple[int, int]] | dict]]


def _sound_span(node: SoundNode, length: int) -> tuple[tuple[str, int], tuple[int, int, dict]]:
    """A ``spans`` entry for a sound cue playing ``length`` samples from where its block starts."""
    return ("sound", id(node)), (0, length, {"name": node.name or node.description, "loop": node.loop})


def _render(node: TreeNode, ctx: _Render, spans: Spans | None) -> tuple[np.ndarray, int]:
    """(audio, advance): ``node`` rendered, and where the timeline moves on after it.

    Audio past ``advance`` is a sound's tail, mixed under whatever follows.
    """
    rate = ctx.rate
    if isinstance(node, SpeechNode):
        chunks = ctx.node_chunks.get(id(node), [])
        rendered_turns: list[tuple[str, np.ndarray, str]] = []  # stereo, placed
        line_offsets: list[list[tuple[int, int]]] = []
        wets: list[np.ndarray | None] = []  # each take's room, which rings on past it
        for chunk in chunks:
            take = ctx.takes[chunk.index]
            tidied = audio.tidy_take(take.samples, take.rate)
            tidied = audio.resample_audio(tidied, take.rate, rate)
            gain = ctx.gains.get(chunk.speaker, 1.0)
            if gain != 1.0:
                tidied = (tidied * gain).astype(np.float32, copy=False)
            # Lines are found before effects, whose filtering can blur the pauses between them.
            lines = audio.line_bounds(tidied, rate, [max(expected_seconds(line), 0.3)
                                                     for line in chunk.lines or (chunk.text,)])
            if take.line_bounds is not None:
                lines = [(round(start * rate / take.rate), round(end * rate / take.rate))
                         for start, end in take.line_bounds]
            words = [[(text, round(start * rate), round(end * rate)) for text, start, end in line]
                     for line in take.line_words] if take.line_words is not None else None
            room = _voice_room(chunk, ctx)
            distance = next((effect for effect in chunk.speaker_effects if effect in DISTANCE_TAGS), "")
            # Real rooms and distance are placed below; the other effects shape the voice itself.
            effects = tuple(effect for effect in chunk.speaker_effects
                            if effect not in ROOM_EFFECT_ROOMS and effect not in DISTANCE_TAGS)
            if room in rooms.STYLIZED_ROOMS:  # a scene underwater or in a dream: its voices carry it
                effects, room = tuple(dict.fromkeys((*effects, room))), None
            if effects:
                dry = len(tidied)
                tidied = apply_speaker_effects(tidied, rate, effects)
                if dry and len(tidied) != dry:
                    lines = [(start * len(tidied) // dry, end * len(tidied) // dry) for start, end in lines]
                    if words is not None:
                        words = [[(text, start * len(tidied) // dry, end * len(tidied) // dry)
                                  for text, start, end in line] for line in words]
            if words is not None:
                ctx.words[chunk.index] = words
            # The room hears the voice as spoken; distance changes only what reaches the listener directly.
            wets.append(rooms.room_wet(tidied, rate, room, send=NEAR_ROOM_SEND if distance == "near" else 1.0)
                        if room else None)
            tidied = audio.voice_distance(tidied, rate, distance, in_room=room is not None)
            position = ctx.positions.get(chunk.speaker, 0.0)
            ctx.placements[chunk.index] = position
            rendered_turns.append((chunk.speaker, audio.pan_mono(tidied, position), chunk.text))
            line_offsets.append(lines)
        offsets: list[tuple[int, int]] = []
        joined = audio.join_takes(rendered_turns, rate, offsets)
        advance = len(joined)
        tails = [(start, wet) for (start, _end), wet in zip(offsets, wets) if wet is not None]
        if tails:
            out = np.zeros((max(advance, *(start + len(wet) for start, wet in tails)), 2), dtype=np.float32)
            out[:advance] = joined
            for start, wet in tails:
                out[start:start + len(wet)] += wet
            joined = out
        if spans is not None:
            spans.update((chunk.index, (*offset, lines))
                         for chunk, offset, lines in zip(chunks, offsets, line_offsets))
        return joined, advance

    if isinstance(node, PauseNode):  # a negative pause is an overlap, placed by its sequence
        silence = np.zeros((max(0, int(round(node.seconds * rate))), 2), dtype=np.float32)
        return silence, len(silence)

    if isinstance(node, SoundNode):
        samples = _placed_sound(node, ctx)
        volume = node.volume * audio.distance_gain(node.distance)
        if volume != 1.0:
            samples = (samples * float(volume)).astype(np.float32, copy=False)
        if spans is not None:
            spans.update([_sound_span(node, len(samples))])
        return samples, len(samples)

    if isinstance(node, SequentialNode):
        placed: list[tuple[int, np.ndarray]] = []
        cursor, previous = 0, None  # previous: the last speech before this child, with no pause or sound between
        floor = 0  # the earliest an overlap may start what follows
        for child in node.children:
            if isinstance(child, PauseNode) and child.seconds < 0:
                # A written overlap: what follows starts this much before the previous child ends,
                # in place of the usual gap, never past half of its last line.
                if placed:
                    cursor = max(cursor + int(round(child.seconds * rate)), floor)
                previous = None
                continue
            child_spans: Spans = {}
            seg, advance = _render(child, ctx, child_spans)
            if seg.size == 0:
                continue
            edges = None if isinstance(child, PauseNode) or is_sound_only_node(child) else _edge_texts(child, ctx)
            if edges is not None and previous is not None:
                # A barge-in's negative gap overlaps the previous child, never past half of its last line.
                gap = int(rate * audio.turn_gap(previous[1], edges[1], previous[0] == edges[0]))
                cursor = max(cursor + gap, floor)
            placed.append((cursor, seg))
            # A speech block holds several lines: half of the whole block could reach back before
            # its last line starts, putting what follows ahead of the line it answers.
            floor = cursor + advance // 2
            speech = [entry for key, entry in child_spans.items() if isinstance(key, int)]
            if speech:
                start, end, lines = max(speech, key=lambda entry: entry[0])
                first, last = (start + lines[-1][0], start + lines[-1][1]) if lines else (start, end)
                floor = cursor + first + (last - first) // 2
            if spans is not None:
                spans.update((index, (start + cursor, end + cursor, lines))
                             for index, (start, end, lines) in child_spans.items())
            cursor += advance
            previous = (edges[2], edges[3]) if edges is not None else None
        length = max([cursor] + [start + len(seg) for start, seg in placed])
        out = np.zeros((length, 2), dtype=np.float32)
        for start, seg in placed:
            out[start:start + len(seg)] += seg
        return out, cursor

    if isinstance(node, ParallelNode):
        # A scene's room is heard by the sounds inside it, and its voices when the script sets it;
        # its stage places the speakers it names for this scene.
        stage = {speaker: position for speaker, position in node.stage if not is_narrator(speaker)}
        ctx = replace(ctx, room=_scene_room(node, ctx) or ctx.room, scene_room=node.room or ctx.scene_room,
                      positions={**ctx.positions, **stage} if stage else ctx.positions)
        tracks: list[dict] = []
        track_spans: list[Spans | None] = []  # each child's own spans; None for a cue mixed here
        for child in node.children:
            if isinstance(child, SoundNode):
                # Volume and distance apply after the mix levels the sound to the speech.
                tracks.append({"audio": _placed_sound(child, ctx), "loop": child.loop, "is_sound": True,
                               "volume": child.volume * audio.distance_gain(child.distance)})
                track_spans.append(None)
            else:
                child_spans: Spans = {}
                child_audio, advance = _render(child, ctx, child_spans)
                tracks.append({"audio": child_audio, "advance": advance, "loop": False,
                               "is_sound": is_sound_only_node(child), "volume": 1.0})
                track_spans.append(child_spans)
        mix = audio.mix_parallel(tracks, rate)
        if spans is not None:
            for child, track, own, length in zip(node.children, tracks, track_spans, mix.sound_lengths):
                if own is None:
                    spans.update([_sound_span(child, length)])
                    continue
                # Parallel tracks all start together; speech waits for the lead-in, sounds do not.
                shift = 0 if track["is_sound"] else mix.speech_offset
                spans.update((key, (start + shift, end + shift, lines))
                             for key, (start, end, lines) in own.items())
        return mix.audio, mix.advance

    return np.zeros((0, 2), dtype=np.float32), 0


def _render_ast(
    node: TreeNode,
    node_chunks: dict[int, list[Chunk]],
    takes: dict[int, Take],
    sound_clips: dict[str, Path],
    rate: int,
    decoded: dict[Path, np.ndarray] | None = None,
    spans: Spans | None = None,
    gains: dict[str, float] | None = None,
    positions: dict[str, float] | None = None,
    placements: dict[int, float] | None = None,
    words: dict[int, list[list[tuple[str, int, int]]]] | None = None,
) -> np.ndarray:
    """Recursively render a Drama 3 AST node into (frames, 2) float32 stereo audio at ``rate``.

    ``spans``, when given, receives each chunk's (start, end) samples in the returned audio,
    keyed by chunk index, with the (start, end) of each of its script lines inside the take;
    and each sound cue's, keyed ``("sound", id)``, with its name and whether it loops.
    ``gains`` scales each speaker's takes (see ``speaker_gains``) and ``positions`` places
    them left to right (see ``speaker_positions``); unlisted speakers are centred. A scene's
    ``stage`` moves the speakers it names. ``placements``, when given, receives each chunk's
    position as mixed, keyed by chunk index; ``words``, each aligned take's words per script
    line as (text, start, end) samples inside the take.
    """
    ctx = _Render(node_chunks, takes, sound_clips, rate, {} if decoded is None else decoded, gains or {},
                  positions or {}, placements={} if placements is None else placements,
                  words={} if words is None else words)
    return _render(node, ctx, spans)[0]


def title_stem(title: str, limit: int = 60) -> str:
    """A file-name stem from a script title, keeping letters of any script; default when empty.

    Gradio keeps letters, digits, ``._-`` and spaces in cached copies, so the stem
    uses only letters, digits, ``-`` and ``_`` and still matches its original.
    """
    stem = re.sub(r"[^\w-]+", "_", str(title or ""), flags=re.UNICODE)
    stem = re.sub(r"_+", "_", stem).strip("_-")[:limit].rstrip("_-")
    return stem or "Drama3_sglang"


def generate_drama3_served(request, *, runtime, free_gpu: Callable[[], None], hf_token=None,
                           progress=lambda *args, **kwargs: None) -> Iterator[GenerationUpdate]:
    """Generate speaker chunks, optionally check/retry them, render sounds/effects, then join/mix.

    Saved takes are reused unless their key is in ``request.regenerate``; with
    ``request.remix_only`` nothing is generated and every take must be saved.
    """
    settings = ServerSettings.from_environment(runtime)
    server = shared_server(settings)
    started = time.monotonic()

    # Basic mode ignores staging, sound, pause and vocal-effect tags the text may still hold.
    script_text = (
        strip_drama3_sound_tags(request.text)
        if getattr(request, "script_mode", "Basic") == "Basic"
        else request.text
    )
    names, default_speaker, resolve = request_cast(request)
    tree = parse_drama3_tree(script_text, default_speaker=default_speaker, resolve=resolve)
    speech_nodes = iter_speech_nodes(tree)
    sound_nodes = iter_sound_nodes(tree)

    chunks: list[Chunk] = []
    node_chunks: dict[int, list[Chunk]] = {}
    for speech_node in speech_nodes:
        planned = plan_turns_chunks(speech_node.turns, start_index=len(chunks))
        node_chunks[id(speech_node)] = planned
        chunks.extend(planned)

    if not chunks:
        raise H3Error("Speech text is required.")
    # Each line goes out with its speaker's clip, found by the speaker's name.
    references = {speaker_label(speaker, names): (path, transcript) for speaker, path, transcript in request.voices}
    unvoiced = sorted({chunk.speaker for chunk in chunks} - set(references))
    if unvoiced:
        raise H3Error("Every speaker needs a reference voice; missing " + ", ".join(unvoiced) + ".")
    cast = ", ".join(f"{s}: {sum(c.speaker == s for c in chunks)}" for s in dict.fromkeys(c.speaker for c in chunks))
    sound_summary = f" · {len(sound_nodes)} sound cue(s)" if sound_nodes else ""
    yield GenerationUpdate(None, f"Planned {len(chunks)} chunk(s) · {cast}{sound_summary}")

    # Sounds come first: a missing or failed clip stops the job before any speech is spent.
    sound_clips: dict[str, Path] = {}
    if sound_nodes:
        for item in _resolve_sounds_for_tree(sound_nodes, request, runtime, free_gpu, hf_token):
            if isinstance(item, GenerationUpdate):
                yield item
            else:
                sound_clips = item
        missing = [node.name or node.description for node in sound_nodes
                   if _sound_clip(node, sound_clips) is None]
        if missing:
            raise H3Error("No clip for sound(s): " + ", ".join(f'"{name}"' for name in dict.fromkeys(missing)) + ".")

    label = LABEL
    # Each request is only the line's text and its speaker's clip with that clip's transcript.
    references = {
        speaker: (_staged_reference(path, settings.media_dir), SPEAKER_TAG.sub("", transcript).strip())
        for speaker, (path, transcript) in references.items()
    }
    sampling = {"temperature": float(request.temperature), "top_p": float(request.top_p),
                "top_k": int(request.top_k)}

    def budget(chunk: Chunk) -> int:
        return frame_budget(chunk.text, int(request.max_frames))

    keys: dict[int, str] = {}
    occurrences: dict[str, int] = {}
    for chunk in chunks:
        content_key = take_store.take_key(
            speaker=chunk.speaker, text=chunk.text, reference=references[chunk.speaker][0].stem,
            transcript=references[chunk.speaker][1], sampling=sampling, max_new_tokens=budget(chunk))
        occurrence = occurrences.get(content_key, 0)
        keys[chunk.index] = take_store.occurrence_key(content_key, occurrence)
        occurrences[content_key] = occurrence + 1
    forced = set(getattr(request, "regenerate", ()) or ())
    if forced and not forced & set(keys.values()):
        raise H3Error("That line changed since it was generated. Generate speech, then choose the line again.")
    store = take_store.takes_root(runtime)
    takes: dict[int, Take] = {}
    for chunk in chunks:
        if keys[chunk.index] in forced:
            continue
        saved = take_store.load_take(store, keys[chunk.index])
        if saved is not None:
            # Saved takes keep only their audio; checking it again keeps a failed take failed.
            check = audio.check_take(*saved, expected_seconds(chunk.text),
                                     max_seconds=budget(chunk) / FRAMES_PER_SECOND)
            takes[chunk.index] = Take(chunk, *saved, attempts=0, check=check)
    reused = len(takes)
    pending_chunks = [chunk for chunk in chunks if chunk.index not in takes]
    if getattr(request, "remix_only", False) and pending_chunks:
        raise H3Error(f"{len(pending_chunks)} line(s) have no saved take; generate speech first.")
    if reused:
        yield GenerationUpdate(None, f"Reusing {reused} saved take(s); generating {len(pending_chunks)}.")

    def synthesize(chunk: Chunk, stopped: threading.Event) -> Take:
        clip, transcript = references[chunk.speaker]
        attempt = best_take(server, text=chunk.text, reference=clip, transcript=transcript,
                            max_new_tokens=budget(chunk), sampling=sampling,
                            attempts=max(int(request.retries), AUTOMATIC_RETRIES) + 1, stopped=stopped)
        return Take(chunk, attempt.samples, attempt.rate, attempt.number, attempt.check, attempt.wav)

    if pending_chunks:
        yield from _ready_server(settings, server, label, free_gpu, hf_token)
        # The server stays loaded after the job; release_all() frees it for other engines.
        total = len(pending_chunks)
        done_count = 0
        workers = max(1, min(int(request.parallel), total))
        with _request_pool(workers, "drama3-sglang") as (pool, stopped):
            pending = {pool.submit(synthesize, chunk, stopped): chunk for chunk in pending_chunks}
            while pending:
                done, _ = wait(pending, timeout=HEARTBEAT_SECONDS, return_when=FIRST_COMPLETED)
                job = CURRENT_JOB.get()
                if job is not None and job.cancelled.is_set():
                    for future in pending:
                        future.cancel()
                    job.check()
                for future in done:
                    chunk = pending.pop(future)
                    take = takes[chunk.index] = future.result()
                    done_count += 1
                    try:
                        take_store.save_take(store, keys[chunk.index], take.wav)
                    except OSError as exc:  # a full disk must not fail the job
                        yield GenerationUpdate(None, f"Could not save take {chunk.index + 1}: {exc}")
                progress((done_count, total), desc="Generating speech")
                yield GenerationUpdate(None, f"Generating speech: {done_count}/{total} chunks "
                                             f"({time.monotonic() - started:.0f}s)")

    ordered = [takes[chunk.index] for chunk in chunks]
    alignment_notes = []
    # A take of several lines is aligned to cut them; word-by-word subtitles align every take,
    # saved ones too, so turning them on and remixing adds them without speaking again.
    # Only a multi-line take's failure leaves estimated line timing worth a note.
    word_timing = bool(getattr(request, "word_timing", False))
    aligned = ordered if word_timing else [take for take in ordered if len(take.chunk.lines) > 1]
    if aligned:
        from .alignment import align_take
        from h3_app.dubbing import tools

        yield GenerationUpdate(None, "Aligning subtitle words to generated speech…" if word_timing
                               else "Aligning subtitle lines to generated speech…")
        try:
            for take in aligned:
                job = CURRENT_JOB.get()
                if job is not None:
                    job.check()
                try:
                    take.line_bounds, words = align_take(take, store, runtime, free_gpu=free_gpu,
                                                         hf_token=hf_token)
                    take.line_words = words if word_timing else None
                except Exception as exc:
                    if job is not None:
                        job.check()
                    if len(take.chunk.lines) > 1:
                        alignment_notes.append(f"chunk {take.chunk.index + 1}: {exc}")
        finally:
            tools.stop()  # release alignment VRAM before the next speech job
    rate = ordered[0].rate
    spans: Spans = {}
    placements: dict[int, float] = {}
    word_spans: dict[int, list[list[tuple[str, int, int]]]] = {}
    joined = _render_ast(tree, node_chunks, takes, sound_clips, rate, spans=spans, gains=speaker_gains(takes),
                         positions=speaker_positions(chunks), placements=placements, words=word_spans)
    joined = audio.master(audio.add_room_tone(joined, rate), rate)

    # Where each take and script line plays in the mix, for synced subtitles;
    # a mix trimmed to fit cuts the last ones short.
    def seconds(sample):
        return round(min(sample, len(joined)) / rate, 3)

    timings = {index: (seconds(start), seconds(end), [(seconds(start + a), seconds(start + b)) for a, b in lines])
               for index, (start, end, lines) in spans.items() if isinstance(index, int)}
    # Each line's words where the mix plays them; a word the trimmed mix cuts off is dropped.
    line_words = {index: [[{"text": text, "start": seconds(spans[index][0] + a), "end": seconds(spans[index][0] + b)}
                           for text, a, b in line if spans[index][0] + a < len(joined)] for line in lines]
                  for index, lines in word_spans.items() if index in spans}
    # When each sound cue plays, in playback order.
    sound_timings = sorted(({**info, "start": seconds(start), "end": seconds(end)}
                            for key, (start, end, info) in spans.items() if not isinstance(key, int)),
                           key=lambda sound: (sound["start"], sound["end"]))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    title = str(getattr(request, "title", "") or "").strip()
    # A full 32-hex id lets the UI find this file (and its settings) from Gradio's cached copy.
    result = runtime.output_dir / "drama3" / f"{title_stem(title)}_{stamp}_{uuid.uuid4().hex}.mp3"
    result.parent.mkdir(parents=True, exist_ok=True)
    audio.save_mp3(joined, rate, result)
    failed = [take for take in ordered if take.check is not None and not take.check.ok]
    write_snapshot(result, {
        "family": "Drama 3",
        "settings": {
            "title": title,
            "backend": LABEL, "temperature": request.temperature,
            "top_p": request.top_p, "top_k": request.top_k, "parallel": request.parallel,
            "retries": request.retries, "reference_count": len(request.voices),
            "script_mode": getattr(request, "script_mode", "Basic"),
            "sound_count": len(sound_nodes),
            "word_timing": word_timing,
            "chunks": [{"speaker": t.chunk.speaker, "name": t.chunk.speaker,
                        "text": t.chunk.text, "take": keys[t.chunk.index], "reused": t.attempts == 0,
                        "attempts": t.attempts,
                        "speaker_effects": list(t.chunk.speaker_effects),
                        **({"pan": round(placements[t.chunk.index], 3)} if t.chunk.index in placements else {}),
                        "check": t.check.reason if t.check is not None else "skipped",
                        # False only for a take kept after every attempt failed its check
                        # (a saved take is checked again when reused).
                        "ok": t.check is None or t.check.ok,
                        "warnings": list(t.check.warnings) if t.check is not None else [],
                        "seconds": round(len(t.samples) / t.rate, 2),
                        "subtitle_timing": "aligned" if t.line_bounds is not None else "estimated",
                        **({"start": timings[t.chunk.index][0], "end": timings[t.chunk.index][1],
                            "lines": [{"text": text, "start": start, "end": end,
                                       **({"words": line_words[t.chunk.index][number]}
                                          if t.chunk.index in line_words
                                          and number < len(line_words[t.chunk.index]) else {})}
                                      for number, (text, (start, end)) in enumerate(
                                          zip(t.chunk.lines or (t.chunk.text,), timings[t.chunk.index][2]))]}
                           if t.chunk.index in timings else {})} for t in ordered],
            "sounds": sound_timings,
            "duration": round(len(joined) / rate, 3),
        },
    })
    if pending_chunks:
        persist_caches()  # kernels JIT-compiled for this job's shapes survive restarts too
    take_store.prune(store)
    progress(1, desc="Complete")
    note = (f"\n{len(failed)} chunk(s) kept their best take after every attempt failed a check: "
            + "; ".join(f"chunk {t.chunk.index + 1} ({t.chunk.speaker}, "
                        f"{f'{t.attempts} attempt(s)' if t.attempts else 'saved take'}) "
                        f"{t.check.reason}" for t in failed)) if failed else ""
    # Uncertain advisories stay in the result's details unless retries were asked for.
    advised = [take for take in ordered if request.retries and take.check is not None and take.check.warnings]
    if advised:
        note += "\nAudio advisories (takes kept): " + "; ".join(
            f"chunk {t.chunk.index + 1}: {t.check.reason}" for t in advised)
    if alignment_notes:
        note += "\nEstimated subtitle timing (alignment unavailable): " + "; ".join(alignment_notes)
    reuse_note = f" ({reused} reused)" if reused else ""
    yield GenerationUpdate(fresh_audio(result), f"Drama 3 ({label}) complete · {len(chunks)} chunks{reuse_note} · "
                                        f"{len(joined) / rate:.1f}s in {time.monotonic() - started:.0f}s{note}")
