"""Chunked, parallel Drama 3 generation through a managed SGLang-Omni server."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
import threading
import time
from typing import Any, Callable, Iterator
import uuid

import numpy as np
import requests

from h3_app import drama3_sounds, stable_audio
from h3_app.errors import H3Error
from h3_app.generation.results import GenerationUpdate, fresh_audio
from h3_app.jobs import CURRENT_JOB
from h3_app.provenance import write_snapshot
from h3_app.workflows.drama3 import MAX_FRAMES, SPEAKER_TAG

from . import audio, environment
from .planner import Chunk, expected_seconds, plan_turns_chunks
from .server import ServerSettings, SpeechRequestError, ensure_model, persist_caches, shared_server
from .speaker_effects import apply_speaker_effects
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

FRAMES_PER_SECOND = 21.5
# SGLang-Omni's Drama 3 request builder rejects max_new_tokens above this (~95 s of speech).
SERVER_MAX_NEW_TOKENS = 2048
HEARTBEAT_SECONDS = 5.0
# Dropped connections and server errors are retried this many times, independent of take checks.
TRANSPORT_RETRIES = 2
TRANSPORT_BACKOFF_SECONDS = 2.0


@dataclass
class Take:
    chunk: Chunk
    samples: np.ndarray
    rate: int
    attempts: int
    check: audio.TakeCheck | None


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


def _staged_reference(path: str, media_dir: Path) -> Path:
    """Copy a clip under the server's allowed media path, keyed by content."""
    source = Path(path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:24]
    target = media_dir / "references" / f"{digest}{source.suffix.lower() or '.wav'}"
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return target


def _with_heartbeat(label: str, work: Callable[[Callable[[str], None]], object]) -> Iterator[GenerationUpdate]:
    """Run blocking setup in a thread, reporting its latest progress message."""
    result, error, latest = [], [], [label]

    def run():
        try:
            result.append(work(lambda message: latest.__setitem__(0, message)))
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


def _ready_server(settings, server, label, free_gpu, hf_token) -> Iterator[GenerationUpdate]:
    yield from _with_heartbeat(f"Preparing the {label} environment",
                               lambda _progress: environment.prepare(settings.venv))
    yield from _with_heartbeat("Downloading the Drama 3 checkpoint",
                               lambda _progress: ensure_model(settings, token=hf_token))
    yield from _with_heartbeat(f"Freeing ComfyUI VRAM and starting {label}",
                               lambda report: server.ensure_ready(free_gpu, report))


@dataclass(frozen=True)
class DesignedVoice:
    key: object
    error: Exception | None = None  # None: the clip was written to its target


def design_voices_served(takes: dict, *, runtime, free_gpu: Callable[[], None], hf_token=None,
                         sampling: dict, max_frames: int, parallel: int = 1,
                         label: str = "SGLang-Omni") -> Iterator[GenerationUpdate | DesignedVoice]:
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


def _render_ast(
    node: TreeNode,
    node_chunks: dict[int, list[Chunk]],
    takes: dict[int, Take],
    sound_clips: dict[str, Path],
    rate: int,
) -> np.ndarray:
    """Recursively render a Drama 3 AST node into a 1D float32 audio array at ``rate``."""
    if isinstance(node, SpeechNode):
        chunks = node_chunks.get(id(node), [])
        rendered_turns: list[tuple[int, np.ndarray]] = []
        for chunk in chunks:
            take = takes[chunk.index]
            tidied = audio.tidy_take(take.samples, take.rate)
            tidied = audio.resample_audio(tidied, take.rate, rate)
            if chunk.speaker_effects:
                tidied = apply_speaker_effects(tidied, rate, chunk.speaker_effects)
            rendered_turns.append((chunk.speaker, tidied))
        return audio.join_takes(rendered_turns, rate)

    if isinstance(node, PauseNode):
        return np.zeros(max(0, int(round(node.seconds * rate))), dtype=np.float32)

    if isinstance(node, SoundNode):
        clip_path = _sound_clip(node, sound_clips)
        if clip_path is None:
            raise H3Error(f'Missing sound clip for "{node.name or node.description}".')
        samples = audio.load_audio_file(clip_path, rate)
        if node.volume != 1.0:
            samples = (samples * float(node.volume)).astype(np.float32, copy=False)
        return samples

    if isinstance(node, SequentialNode):
        if len(node.children) == 1:
            return _render_ast(node.children[0], node_chunks, takes, sound_clips, rate)
        pieces: list[np.ndarray] = []
        prev_was_speech = False
        for child in node.children:
            seg = _render_ast(child, node_chunks, takes, sound_clips, rate)
            if seg.size == 0:
                continue
            is_speech = isinstance(child, SpeechNode)
            if pieces and prev_was_speech and is_speech:
                pieces.append(np.zeros(int(rate * audio.TURN_GAP), dtype=np.float32))
            pieces.append(seg.astype(np.float32, copy=False))
            prev_was_speech = is_speech
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)

    if isinstance(node, ParallelNode):
        tracks: list[dict] = []
        for child in node.children:
            if isinstance(child, SoundNode):
                clip_path = _sound_clip(child, sound_clips)
                if clip_path is None:
                    raise H3Error(f'Missing sound clip for "{child.name or child.description}".')
                raw_samples = audio.load_audio_file(clip_path, rate)
                tracks.append(
                    {
                        "audio": raw_samples,
                        "loop": child.loop,
                        "is_sound": True,
                        "volume": child.volume,
                    }
                )
            else:
                child_audio = _render_ast(child, node_chunks, takes, sound_clips, rate)
                tracks.append(
                    {
                        "audio": child_audio,
                        "loop": False,
                        "is_sound": is_sound_only_node(child),
                        "volume": 1.0,
                    }
                )
        return audio.mix_parallel_tracks(tracks, rate)

    return np.zeros(0, dtype=np.float32)


def generate_drama3_served(request, *, runtime, free_gpu: Callable[[], None], hf_token=None,
                           progress=lambda *args, **kwargs: None) -> Iterator[GenerationUpdate]:
    """Generate speaker chunks, optionally check/retry them, render sounds/effects, then join/mix."""
    settings = ServerSettings.from_environment(runtime)
    server = shared_server(settings)
    started = time.monotonic()

    # Basic mode ignores staging, sound, pause and vocal-effect tags the text may still hold.
    script_text = (
        strip_drama3_sound_tags(request.text)
        if getattr(request, "script_mode", "Basic") == "Basic"
        else request.text
    )
    tree = parse_drama3_tree(script_text)
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
    names = {speaker: name or f"Speaker {speaker}" for speaker, name in request.voice_names}
    references = {speaker: (path, transcript) for speaker, path, transcript in request.voices}
    unvoiced = sorted({chunk.speaker for chunk in chunks} - set(references))
    if unvoiced:
        raise H3Error(f"Every speaker needs a reference voice on the {request.backend} backend; missing "
                      + ", ".join(f"<|speaker:{n}|>" for n in unvoiced) + ".")
    cast = ", ".join(f"{names.get(s, f'Speaker {s}')}: {sum(c.speaker == s for c in chunks)}"
                     for s in sorted({c.speaker for c in chunks}))
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

    label = request.backend
    yield from _ready_server(settings, server, label, free_gpu, hf_token)
    # The server stays loaded after the job; release_all() frees it for other engines.
    # Each request has one voice; the server tags its transcript as speaker 0 itself.
    references = {
        speaker: (_staged_reference(path, settings.media_dir), SPEAKER_TAG.sub("", transcript).strip())
        for speaker, (path, transcript) in references.items()
    }
    sampling = {"temperature": float(request.temperature), "top_p": float(request.top_p),
                "top_k": int(request.top_k)}

    def synthesize(chunk: Chunk, stopped: threading.Event) -> Take:
        best = None
        for attempt in range(int(request.retries) + 1):
            clip, transcript = references[chunk.speaker]
            payload = server.payload(text=chunk.text, reference=clip, transcript=transcript,
                                     max_new_tokens=frame_budget(chunk.text, int(request.max_frames)),
                                     sampling=sampling)
            samples, rate = audio.decode_wav(_request_speech(server, payload, stopped))
            if not request.retries:
                return Take(chunk, samples, rate, attempt + 1, None)
            check = audio.check_take(samples, rate, expected_seconds(chunk.text))
            take = Take(chunk, samples, rate, attempt + 1, check)
            if check.ok:
                return take
            if best is None or check.score < best.check.score:
                best = take
        return best

    takes: dict[int, Take] = {}
    total = len(chunks)
    workers = max(1, min(int(request.parallel), total))
    with _request_pool(workers, "drama3-sglang") as (pool, stopped):
        def run_all(batch: list[Chunk], stage: str) -> Iterator[GenerationUpdate]:
            pending = {pool.submit(synthesize, chunk, stopped): chunk for chunk in batch}
            while pending:
                done, _ = wait(pending, timeout=HEARTBEAT_SECONDS, return_when=FIRST_COMPLETED)
                job = CURRENT_JOB.get()
                if job is not None and job.cancelled.is_set():
                    for future in pending:
                        future.cancel()
                    job.check()
                for future in done:
                    chunk = pending.pop(future)
                    takes[chunk.index] = future.result()
                progress((len(takes), total), desc=stage)
                yield GenerationUpdate(None, f"{stage}: {len(takes)}/{total} chunks "
                                             f"({time.monotonic() - started:.0f}s)")

        yield from run_all(chunks, "Generating speech")

    ordered = [takes[chunk.index] for chunk in chunks]
    rate = ordered[0].rate
    joined = _render_ast(tree, node_chunks, takes, sound_clips, rate)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    result = runtime.output_dir / "drama3" / f"Drama3_sglang_{stamp}_{uuid.uuid4().hex[:6]}.mp3"
    result.parent.mkdir(parents=True, exist_ok=True)
    audio.save_mp3(joined, rate, result)
    failed = [take for take in ordered if take.check is not None and not take.check.ok]
    write_snapshot(result, {
        "family": "Drama 3",
        "settings": {
            "backend": request.backend, "seed": None, "temperature": request.temperature,
            "top_p": request.top_p, "top_k": request.top_k, "parallel": request.parallel,
            "retries": request.retries, "reference_count": len(request.voices),
            "script_mode": getattr(request, "script_mode", "Basic"),
            "sound_count": len(sound_nodes),
            "chunks": [{"speaker": t.chunk.speaker, "name": names.get(t.chunk.speaker, ""),
                        "seed": None, "attempts": t.attempts,
                        "speaker_effects": list(t.chunk.speaker_effects),
                        "check": t.check.reason if t.check is not None else "skipped",
                        "warnings": list(t.check.warnings) if t.check is not None else [],
                        "seconds": round(len(t.samples) / t.rate, 2)} for t in ordered],
        },
    })
    persist_caches()  # kernels JIT-compiled for this job's shapes survive restarts too
    progress(1, desc="Complete")
    note = (f"\n{len(failed)} chunk(s) kept their best take after {int(request.retries) + 1} attempt(s) failed a check: "
            + "; ".join(f"chunk {t.chunk.index + 1} ({names.get(t.chunk.speaker, t.chunk.speaker)}) "
                        f"{t.check.reason}" for t in failed)) if failed else ""
    advised = [take for take in ordered if take.check is not None and take.check.warnings]
    if advised:
        note += "\nAudio advisories (takes kept): " + "; ".join(
            f"chunk {t.chunk.index + 1}: {t.check.reason}" for t in advised)
    yield GenerationUpdate(fresh_audio(result), f"Drama 3 ({label}) complete · {len(chunks)} chunks · seed control unavailable · "
                                        f"{len(joined) / rate:.1f}s in {time.monotonic() - started:.0f}s{note}")
