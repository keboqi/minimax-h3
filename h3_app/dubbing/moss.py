"""MOSS-Transcribe-Diarize on SGLang-Omni: Dub & Translate's local transcription that tells speakers apart.

It runs in Drama 3's SGLang-Omni environment (the pinned release serves the model)
as a second managed server on its own port. Like Drama 3's server it sleeps: once
an analysis has its lines, the CUDA state moves to host RAM and the next analysis
wakes it in seconds. Other engines' jobs sleep it too, and an unload stops it.
One request covers up to 90 minutes and returns ``[Sxx]``-labelled timed segments.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import threading
from typing import Callable

from h3_app.drama3.serving import environment
from h3_app.drama3.serving.server import FALLBACK_GPU_GIB, SglangOmniServer, gpu_memory
from h3_app.errors import H3Error

MODEL_REPO = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
MODEL_REVISION = "704aa4a9c304e8520be88901e0d1960158ef5b15"
DOWNLOAD_GB = 1.9
MAX_SECONDS = 90 * 60
# SGLang's runtime slack (CUDA graphs, activations) comes out of the rest of the free memory.
MAX_FRACTION = 0.9
# The job family that uses the server; every other one sleeps it.
FAMILY = "dubbing_analyze"
# Compact output, when a reply has no parsed segments: "[0.48][S01]Welcome everyone[1.66]".
SEGMENT_START = re.compile(r"\[(?P<start>\d+(?:\.\d+)?)\]\s*\[(?P<speaker>S\d+)\]", re.IGNORECASE)
TRAILING_END = re.compile(r"\[(?P<end>\d+(?:\.\d+)?)\]\s*$")
TIMESTAMP = re.compile(r"\[\d+(?:\.\d+)?\]")
SPEAKER_PREFIX = re.compile(r"^\s*\[(?P<speaker>S\d+)\]\s*", re.IGNORECASE)


@dataclass(frozen=True)
class MossSettings:
    venv: Path
    data_dir: Path
    model_root: Path
    port: int = 8097
    # What the server keeps: index-tts-vllm's 0.20 of an empty RTX PRO 6000, about 19 GiB.
    # That holds a 90-minute request's full 128K context (about 112 KiB of KV cache per
    # token, 14 GiB), the 2 GiB of weights and the encoder and decode graphs.
    memory_gib: float = 19.0
    # One analysis at a time: a single request gets the whole KV pool, and graphs for batch 1
    # alone keep startup's capture and compile short.
    max_seqs: int = 1
    start_timeout: float = 1800.0
    request_timeout: float = 3600.0
    sleep_mode: str = "auto"

    @classmethod
    def from_environment(cls, runtime) -> "MossSettings":
        env = os.environ
        drama3_data = Path(env.get("H3_DRAMA3_SGLANG_DATA") or runtime.comfy_dir.parent / "drama3-serving")
        return cls(
            venv=environment.default_venv(runtime.script_dir),
            data_dir=Path(env.get("H3_MOSS_SGLANG_DATA") or drama3_data / "moss"),
            model_root=Path(env.get("H3_MOSS_SGLANG_MODELS") or runtime.comfy_dir / "models" / "dubbing" / "moss"),
            port=int(env.get("H3_MOSS_SGLANG_PORT", "8097")),
            memory_gib=float(env.get("H3_MOSS_SGLANG_MEMORY_GIB", "19")),
            sleep_mode=env.get("H3_MOSS_SGLANG_SLEEP", env.get("H3_DRAMA3_SGLANG_SLEEP", "auto")).lower(),
        )

    def memory_fraction(self, free_mib: float | None) -> float:
        """``--mem-fraction-static`` for a launch with ``free_mib`` free (None: unknown).

        SGLang takes the fraction of the memory free when the server starts, not of the
        whole GPU, so a loaded Drama 3 beside it would otherwise shrink its KV cache.
        (Drama 3's stage-level gpu_memory_fraction is a whole-GPU share, but MOSS's stage
        factory does not take one, so ``--asr.gpu_memory_fraction`` would be ignored.)
        """
        free_gib = free_mib / 1024 if free_mib else FALLBACK_GPU_GIB
        return round(min(MAX_FRACTION, self.memory_gib / free_gib), 3)

    def footprint_mib(self, _total_mib: float = 0.0) -> float:
        """Free GPU memory a launch needs to keep ``memory_gib`` within the fraction cap."""
        return self.memory_gib * 1024 / MAX_FRACTION

    @property
    def model_dir(self) -> Path:
        return self.model_root / f"{MODEL_REPO.replace('/', '--')}--{MODEL_REVISION[:12]}"

    @property
    def media_dir(self) -> Path:
        return self.data_dir / "media"

    @property
    def runtime_dir(self) -> Path:
        return self.data_dir


class MossServer(SglangOmniServer):
    label = "MOSS (SGLang-Omni)"
    log_prefix = "moss-serving"
    log_name = "moss-sglang"
    cuda_workers = 1  # one stage: the Whisper encoder and the decoder share a process
    memory_settings = "H3_MOSS_SGLANG_MEMORY_GIB"

    def command(self) -> list[str]:
        settings = self.settings
        memory = gpu_memory()
        return [
            str(environment.executable(settings.venv, "sgl-omni")), "serve",
            "--model-path", str(settings.model_dir), "--host", "127.0.0.1", "--port", str(settings.port),
            "--asr.engine.mem_fraction_static", f"{settings.memory_fraction(memory and memory[0]):g}",
            "--asr.engine.max_running_requests", str(settings.max_seqs),
            "--asr.engine.cuda_graph_max_bs", str(settings.max_seqs),
        ]

    def transcribe(self, audio: Path) -> dict:
        """One recording as SGLang-Omni's verbose_json: ``{"text", "segments": [{start, end, text}]}``."""
        import requests

        with self._lock:
            if self.sleeping:
                raise H3Error("MOSS is asleep; wake it before transcribing.")
            self._active_requests += 1
        try:
            with Path(audio).open("rb") as stream:
                response = self.http().post(
                    f"{self.base_url}/v1/audio/transcriptions",
                    data={"response_format": "verbose_json", "temperature": "0"},
                    files={"file": (Path(audio).name, stream, "audio/wav")},
                    timeout=self.settings.request_timeout)
            if not response.ok:
                raise H3Error(f"{self.label} transcription failed (HTTP {response.status_code}): "
                              f"{response.text[:500]}")
            return response.json()
        except requests.RequestException:
            # A client timeout does not prove GPU work has stopped remotely.
            with self._lock:
                self._uncertain_request = True
            raise
        finally:
            with self._lock:
                self._active_requests -= 1


def parse_transcript(text: str, duration: float | None = None) -> list[dict]:
    """``[start][Sxx]words[end]`` runs as ``{"start", "end", "speaker", "text"}`` lines.

    A run without its end time ends where the next one starts (the last one: at
    ``duration``, else after a short estimate).
    """
    text = str(text or "")
    marks = list(SEGMENT_START.finditer(text))
    lines = []
    for index, mark in enumerate(marks):
        following = marks[index + 1] if index + 1 < len(marks) else None
        chunk = text[mark.end():following.start() if following else len(text)].strip()
        start = float(mark["start"])
        ended = TRAILING_END.search(chunk)
        words = " ".join(TIMESTAMP.sub(" ", chunk[:ended.start()] if ended else chunk).split())
        if not words:
            continue
        if ended:
            end = float(ended["end"])
        elif following:
            end = float(following["start"])
        else:
            end = duration if duration and duration > start else start + max(0.6, 0.35 * len(words.split()))
        if duration and duration > start:
            end = min(end, duration)
        lines.append({"start": round(start, 3), "end": round(max(end, start + 0.05), 3),
                      "speaker": mark["speaker"].upper(), "text": words})
    return lines


def reply_lines(reply: dict, duration: float | None = None) -> list[dict]:
    """Speaker lines from a verbose_json reply, whose segment texts start with ``[Sxx]``."""
    lines = []
    for segment in reply.get("segments") or []:
        found = SPEAKER_PREFIX.match(str(segment.get("text") or ""))
        words = " ".join(TIMESTAMP.sub(" ", str(segment.get("text") or "")[found.end() if found else 0:]).split())
        if words and segment.get("start") is not None and segment.get("end") is not None:
            lines.append({"start": float(segment["start"]), "end": float(segment["end"]),
                          "speaker": found["speaker"].upper() if found else "S01", "text": words})
    return lines or parse_transcript(reply.get("text", ""), duration)


def ensure_model(settings: MossSettings, *, token=None, log=print) -> Path:
    """Download the pinned checkpoint once."""
    target = settings.model_dir
    marker = target / ".h3-complete"
    if marker.is_file():
        return target
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    from huggingface_hub import snapshot_download

    log(f"[moss-serving] Downloading {MODEL_REPO}@{MODEL_REVISION[:12]} (~{DOWNLOAD_GB:.0f} GB, first use only).")
    snapshot_download(MODEL_REPO, revision=MODEL_REVISION, local_dir=target, token=token)
    marker.write_text(MODEL_REVISION, encoding="utf-8")
    return target


_SERVER: MossServer | None = None
_SERVER_LOCK = threading.Lock()


def shared_server(settings: MossSettings) -> MossServer:
    """The app's MOSS server while its configuration stays unchanged."""
    global _SERVER
    from h3_app.jobs import JOBS

    JOBS.add_gpu_claim_hook(release_for_job)
    with _SERVER_LOCK:
        if _SERVER is None or _SERVER.settings != settings:
            if _SERVER is not None:
                _SERVER.stop()
            _SERVER = MossServer(settings)
        return _SERVER


def release_for_job(family: str) -> None:
    """Analysis keeps MOSS; every other job sleeps it, and an unload stops it."""
    if family == "unload":
        stop_all()
    elif family != FAMILY:
        release_all()


def release_all() -> None:
    """Sleep an idle server (or stop it where sleep is unavailable)."""
    with _SERVER_LOCK:
        server = _SERVER
    if server is not None and server.running():
        server.release()


def stop_all() -> None:
    with _SERVER_LOCK:
        server = _SERVER
    if server is not None:
        server.stop()


def _make_room(settings: MossSettings) -> None:
    """Room for MOSS beside a loaded Drama 3: the idle dubbing tools go first, then Drama 3 sleeps."""
    from h3_app.drama3.serving import server as drama3_server

    from . import tools

    def short() -> bool:
        memory = gpu_memory()
        return memory is not None and memory[0] < settings.footprint_mib(memory[1])

    if short() and tools.stop() and not short():
        return
    if short():
        drama3_server.release_all()


def supported() -> bool:
    """SGLang-Omni, and so MOSS, runs only on Linux."""
    return os.name == "posix"


def transcribe(runtime, audio: Path, *, duration: float, free_gpu: Callable[[], None], hf_token=None,
               report: Callable[[str], None] = lambda _message: None) -> list[dict]:
    """Timed speaker lines of ``audio``; the server sleeps again once they are back."""
    if not supported():
        raise H3Error("MOSS runs on SGLang-Omni, which requires Linux. Use Qwen3-ASR or Gemini here.")
    if duration > MAX_SECONDS:
        raise H3Error(f"MOSS transcribes up to {MAX_SECONDS // 60} minutes at once; this source runs "
                      f"{duration / 60:.0f}. Use Qwen3-ASR or Gemini, or trim the source.")
    settings = MossSettings.from_environment(runtime)
    server = shared_server(settings)
    report("Preparing the SGLang-Omni environment (first use installs it)")
    environment.prepare(settings.venv)
    report("Downloading MOSS-Transcribe-Diarize (first use only)")
    ensure_model(settings, token=hf_token)
    _make_room(settings)
    server.ensure_ready(free_gpu, report)
    report("Finding speakers and transcribing with MOSS")
    try:
        return reply_lines(server.transcribe(audio), duration)
    finally:
        try:
            server.release()
        except H3Error as exc:  # still busy elsewhere; the next job's hook sleeps it
            server.log_line(f"MOSS stays awake: {exc}")
