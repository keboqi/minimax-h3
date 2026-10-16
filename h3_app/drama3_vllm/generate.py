"""Chunked, parallel Drama 3 generation through the managed vLLM-Omni server."""

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
from .server import ServerSettings, ensure_model, shared_server

FRAMES_PER_SECOND = 21.5
HEARTBEAT_SECONDS = 5.0


@dataclass
class Take:
    chunk: Chunk
    samples: np.ndarray
    rate: int
    seed: int
    attempts: int
    check: audio.TakeCheck


def frame_budget(text: str, limit: int) -> int:
    """Stop a runaway take early: generous for the text, never above the request's cap."""
    seconds = expected_seconds(text) * 2.5 + 5.0
    return int(max(256, min(limit, MAX_FRAMES, seconds * FRAMES_PER_SECOND)))


def chunk_seed(base: int, index: int, attempt: int) -> int:
    return (int(base) + index * 7919 + attempt * 104729) % (2**63 - 1)


def _staged_reference(path: str, media_dir: Path) -> Path:
    """Copy a clip under the server's allowed media path, keyed by content."""
    source = Path(path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:24]
    target = media_dir / "references" / f"{digest}{source.suffix.lower() or '.wav'}"
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return target


def _with_heartbeat(label: str, work: Callable[[], object]) -> Iterator[GenerationUpdate]:
    """Run blocking setup in a thread while reporting that it is still going."""
    result, error = [], []

    def run():
        try:
            result.append(work())
        except BaseException as exc:  # surfaced on the generator's thread
            error.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    started = time.monotonic()
    thread.start()
    while thread.is_alive():
        thread.join(HEARTBEAT_SECONDS)
        if thread.is_alive():
            yield GenerationUpdate(None, f"{label} ({time.monotonic() - started:.0f}s)")
    if error:
        raise error[0]


def generate_drama3_vllm(request, *, runtime, free_gpu: Callable[[], None], hf_token=None,
                         seed: int, progress=lambda *args, **kwargs: None) -> Iterator[GenerationUpdate]:
    """Plan speaker chunks, send each with its speaker's reference, check, retry, join."""
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
        raise H3Error("Every speaker needs a reference voice on the vLLM-Omni backend; missing "
                      + ", ".join(f"<|speaker:{n}|>" for n in unvoiced) + ".")
    cast = ", ".join(f"{names.get(s, f'Speaker {s}')}: {sum(c.speaker == s for c in chunks)}"
                     for s in sorted({c.speaker for c in chunks}))
    yield GenerationUpdate(None, f"Planned {len(chunks)} chunk(s) · {cast}")

    yield from _with_heartbeat("Preparing the vLLM-Omni environment",
                               lambda: environment.prepare(settings.venv))
    yield from _with_heartbeat("Downloading the Drama 3 checkpoint for vLLM-Omni",
                               lambda: ensure_model(settings, token=hf_token))
    try:
        yield from _with_heartbeat("Freeing ComfyUI VRAM and starting vLLM-Omni",
                                   lambda: server.ensure_awake(free_gpu))
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
                take_seed = chunk_seed(seed, chunk.index, attempt)
                payload = {"input": chunk.text, "response_format": "wav", "seed": take_seed,
                           "max_new_tokens": frame_budget(chunk.text, int(request.max_frames)),
                           "extra_params": sampling}
                clip, transcript = references[chunk.speaker]
                payload.update(ref_audio=clip.resolve().as_uri(), ref_text=transcript)
                samples, rate = audio.decode_wav(server.speech(payload))
                check = audio.check_take(samples, rate, expected_seconds(chunk.text))
                take = Take(chunk, samples, rate, take_seed, attempt + 1, check)
                if check.ok:
                    return take
                if best is None or check.score < best.check.score:
                    best = take
            return best

        takes: dict[int, Take] = {}
        total = len(chunks)
        workers = max(1, min(int(request.parallel), total))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="drama3-vllm") as pool:
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
    finally:
        server.sleep()

    ordered = [takes[chunk.index] for chunk in chunks]
    rate = ordered[0].rate
    joined = audio.join_takes([(take.chunk.speaker, audio.tidy_take(take.samples, take.rate)) for take in ordered],
                              rate)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    result = runtime.output_dir / "drama3" / f"Drama3_vllm_{stamp}_{uuid.uuid4().hex[:6]}.wav"
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_bytes(audio.encode_wav(joined, rate))
    failed = [take for take in ordered if not take.check.ok]
    write_snapshot(result, {
        "family": "Drama 3",
        "settings": {
            "backend": "vLLM-Omni", "seed": seed, "temperature": request.temperature,
            "top_p": request.top_p, "top_k": request.top_k, "parallel": request.parallel,
            "retries": request.retries, "reference_count": len(request.voices),
            "chunks": [{"speaker": t.chunk.speaker, "name": names.get(t.chunk.speaker, ""),
                        "seed": t.seed, "attempts": t.attempts,
                        "check": t.check.reason, "seconds": round(t.check.seconds, 2)} for t in ordered],
        },
    })
    progress(1, desc="Complete")
    note = (f"\n{len(failed)} chunk(s) kept their best take after every retry failed a check: "
            + "; ".join(f"chunk {t.chunk.index + 1} ({names.get(t.chunk.speaker, t.chunk.speaker)}) "
                        f"{t.check.reason}" for t in failed)) if failed else ""
    yield GenerationUpdate(fresh_audio(result), f"Drama 3 (vLLM-Omni) complete · {len(chunks)} chunks · seed {seed} · "
                                        f"{len(joined) / rate:.1f}s in {time.monotonic() - started:.0f}s{note}")
