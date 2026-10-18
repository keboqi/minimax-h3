"""Chunked, parallel Drama 3 generation through a managed SGLang-Omni server."""

from __future__ import annotations

import hashlib
import shutil
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from h3_app.errors import H3Error
from h3_app.generation.results import GenerationUpdate, fresh_audio
from h3_app.jobs import CURRENT_JOB
from h3_app.provenance import write_snapshot
from h3_app.workflows.drama3 import MAX_FRAMES, SPEAKER_TAG

from . import audio, environment
from .planner import Chunk, expected_seconds, plan_chunks
from .server import ServerSettings, ensure_model, persist_caches, shared_server

FRAMES_PER_SECOND = 21.5
HEARTBEAT_SECONDS = 5.0


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
    return int(max(256, min(limit, MAX_FRAMES, seconds * FRAMES_PER_SECOND)))


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


def design_voice_served(text: str, target: Path, *, runtime, free_gpu: Callable[[], None], hf_token=None,
                        sampling: dict, max_frames: int, label: str = "SGLang-Omni") -> Iterator[GenerationUpdate]:
    """One unchunked take without references, so a single invented voice speaks all of it."""
    settings = ServerSettings.from_environment(runtime)
    server = shared_server(settings)
    yield from _ready_server(settings, server, label, free_gpu, hf_token)
    yield GenerationUpdate(None, f"Generating the voice on {label}")
    payload = server.payload(text=text, reference=None, transcript=None,
                             max_new_tokens=frame_budget(text, max_frames), sampling=sampling)
    samples, rate = audio.decode_wav(server.speech(payload))
    samples = audio.tidy_take(samples, rate)
    if not samples.size:
        raise H3Error(f"Drama 3 ({label}) returned a silent voice take; generate again.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(audio.encode_wav(samples, rate))
    persist_caches()


def generate_drama3_served(request, *, runtime, free_gpu: Callable[[], None], hf_token=None,
                           progress=lambda *args, **kwargs: None) -> Iterator[GenerationUpdate]:
    """Generate speaker chunks, optionally check/retry them, then join."""
    settings = ServerSettings.from_environment(runtime)
    server = shared_server(settings)
    started = time.monotonic()
    chunks = plan_chunks(request.text)
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
    yield GenerationUpdate(None, f"Planned {len(chunks)} chunk(s) · {cast}")

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

    def synthesize(chunk: Chunk) -> Take:
        best = None
        for attempt in range(int(request.retries) + 1):
            clip, transcript = references[chunk.speaker]
            payload = server.payload(text=chunk.text, reference=clip, transcript=transcript,
                                     max_new_tokens=frame_budget(chunk.text, int(request.max_frames)),
                                     sampling=sampling)
            samples, rate = audio.decode_wav(server.speech(payload))
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
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="drama3-sglang") as pool:
        def run_all(batch: list[Chunk], stage: str) -> Iterator[GenerationUpdate]:
            pending = {pool.submit(synthesize, chunk): chunk for chunk in batch}
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
    joined = audio.join_takes([(take.chunk.speaker, audio.tidy_take(take.samples, take.rate)) for take in ordered],
                              rate)
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
            "chunks": [{"speaker": t.chunk.speaker, "name": names.get(t.chunk.speaker, ""),
                        "seed": None, "attempts": t.attempts,
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
