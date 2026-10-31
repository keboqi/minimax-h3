"""Nemotron 3.5 Transcribe on vLLM: Dub & Translate's local transcription for overlapping speakers.

NVIDIA's NVFP4 checkpoint (Blackwell only, as on our RTX PRO 6000 servers) returns a
speaker-tagged transcript plus CTC word timings, each word with its speaker. It runs in
its own NeMo Speech + vLLM environment as a managed server that sleeps once the words are
back: vLLM's sleep mode moves the weights to host RAM and drops the KV cache, so
translation and Drama 3 get the GPU and the next analysis wakes it in seconds. Other
engines' jobs sleep it too, and an unload stops it. One request covers up to an hour.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Callable

from h3_requirements import (
    NEMOTRON_READY_MARKER as READY_MARKER,
    nemotron_fingerprint,
    nemotron_install_commands,
)

from h3_app.drama3.serving import checkpoint, environment as sglang_environment
from h3_app.drama3.serving.server import SglangOmniServer, gpu_memory, reclaim_audio_tools
from h3_app.errors import H3Error
from h3_app.workers.env import setup_lock, uv_command

from . import segments as segment_tools

MODEL_REPO = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4"
DOWNLOAD_GB = 20
# Sleep copies the weights (about the download) into pinned host RAM; leave this much beside them.
SLEEP_RAM_MARGIN_GIB = 8
MAX_SECONDS = 60 * 60
FAMILY = "dubbing_analyze"
# The model card's prompt; it tags up to four speakers.
PROMPT = ("Transcribe this audio with punctuation and capitalization. Include all spoken words and verbal "
          "fillers. Write numbers in their standard written form. Put speaker tags <spk:0> <spk:1> <spk:2> "
          "<spk:3> on the transcription.")
MAX_TOKENS = 16384
# Speech the request uploads: 16 kHz mono keeps an hour under vLLM's 180 MB clip limit.
UPLOAD_RATE = 16000
# Line splitting: a pause this long, or a sentence end once a line is this long, starts a new line,
# and no line runs longer than the last.
LINE_PAUSE = 1.0
SENTENCE_MIN = 2.0
LINE_MAX = 12.0
# Words of one speaker closer than this form one stretch of speaker activity.
ACTIVITY_GAP = 0.3
# Words may end this far past the audio (rounding at the last frame); further is a bad reply.
END_SLACK = 0.5
UNKNOWN_SPEAKER = "?"
SENTENCE_END = re.compile(r"[.?!。？！…]['\"”’)]*$")
SPEAKER_TAG = re.compile(r"(\d+)")
CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯豈-﫿]")


@dataclass(frozen=True)
class NemotronSettings:
    venv: Path
    data_dir: Path
    model_root: Path
    port: int = 8098
    # The checkpoint is about 20 GB; an hour's 65K-token context in FP8 KV cache (the model is
    # mostly Mamba, so its KV cache is small), the audio encoder and graphs fit in the rest.
    memory_gib: float = 40.0
    max_seqs: int = 1
    start_timeout: float = 2400.0  # FlashInfer compiles its NVFP4 kernels on the first start
    request_timeout: float = 3600.0
    sleep_mode: str = "auto"  # vLLM's level-1 sleep between analyses; "off" stops it instead
    # Recordings up to this long launch vLLM eager (no torch.compile, no CUDA graph capture):
    # a much shorter start for a slower decode that a short transcript hardly notices. 0: never.
    fast_start_seconds: float = 600.0

    def __post_init__(self):
        if self.sleep_mode not in {"auto", "off"}:
            raise ValueError("H3_NEMOTRON_SLEEP must be auto or off")

    @classmethod
    def from_environment(cls, runtime) -> "NemotronSettings":
        env = os.environ
        return cls(
            venv=default_venv(runtime.script_dir),
            data_dir=Path(env.get("H3_NEMOTRON_DATA") or runtime.comfy_dir.parent / "nemotron-serving"),
            model_root=Path(env.get("H3_NEMOTRON_MODELS") or runtime.comfy_dir / "models" / "dubbing" / "nemotron"),
            port=int(env.get("H3_NEMOTRON_PORT", "8098")),
            memory_gib=float(env.get("H3_NEMOTRON_MEMORY_GIB", "40")),
            sleep_mode=env.get("H3_NEMOTRON_SLEEP", "auto").lower(),
            fast_start_seconds=float(env.get("H3_NEMOTRON_FAST_START_SECONDS", "600")),
        )

    def utilization(self, total_mib: float | None) -> float:
        """vLLM's ``--gpu-memory-utilization``: a share of the whole GPU, not of what is free."""
        total_gib = total_mib / 1024 if total_mib else 96.0
        return round(min(0.9, self.memory_gib / total_gib), 3)

    def footprint_mib(self, _total_mib: float = 0.0) -> float:
        return self.memory_gib * 1024 + 1024  # plus the CUDA context

    @property
    def model_dir(self) -> Path:
        return self.model_root / MODEL_REPO.replace("/", "--")

    @property
    def media_dir(self) -> Path:
        return self.data_dir / "media"

    @property
    def runtime_dir(self) -> Path:
        return self.data_dir


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def default_venv(script_dir: Path) -> Path:
    configured = os.getenv("H3_NEMOTRON_VENV")
    return Path(configured) if configured else Path(script_dir) / "h3" / "nemotron-venv"


def is_ready(venv: Path) -> bool:
    try:
        return ((Path(venv) / "bin" / "python").is_file()
                and (Path(venv) / READY_MARKER).read_text(encoding="utf-8").strip() == nemotron_fingerprint())
    except OSError:
        return False


def prepare(venv: Path, *, run=None, log=print) -> None:
    """Create the environment once; readiness is published only after the smoke test."""
    venv = Path(venv)
    if not supported():
        raise H3Error("Nemotron runs on vLLM, which requires Linux.")
    run = run or (lambda command, **kwargs: subprocess.run(command, check=True, **kwargs))
    with setup_lock(venv.parent / f".{venv.name}.lock"):
        if is_ready(venv):
            return
        uv = uv_command()
        if not uv:
            raise H3Error("uv is required to set up Nemotron; run `pip install uv` and retry.")
        (venv / READY_MARKER).unlink(missing_ok=True)
        log("[nemotron-serving] Preparing the NeMo Speech + vLLM environment (first use takes several minutes).")
        env = sglang_environment.install_env(venv)
        for command in nemotron_install_commands(venv, uv):
            run(command, env=env)
        (venv / READY_MARKER).write_text(nemotron_fingerprint(), encoding="utf-8")
        log("[nemotron-serving] Nemotron environment ready.")


def ensure_model(settings: NemotronSettings, *, token=None, log=print) -> Path:
    """Download the checkpoint once."""
    target = settings.model_dir
    marker = target / ".h3-complete"
    if marker.is_file():
        return target
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    from huggingface_hub import snapshot_download

    log(f"[nemotron-serving] Downloading {MODEL_REPO} (~{DOWNLOAD_GB} GB, first use only).")
    snapshot_download(MODEL_REPO, local_dir=target, token=token)
    marker.write_text(MODEL_REPO, encoding="utf-8")
    return target


def supported() -> bool:
    return os.name == "posix"


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class NemotronServer(SglangOmniServer):
    label = "Nemotron (vLLM)"
    log_prefix = "nemotron-serving"
    log_name = "nemotron-vllm"
    cuda_workers = 1
    memory_settings = "H3_NEMOTRON_MEMORY_GIB"
    cuda_checkpoint = False  # vLLM sleeps itself (--enable-sleep-mode, /sleep and /wake_up)

    def __init__(self, settings: NemotronSettings):
        super().__init__(settings)
        self._asleep = False
        self._sleep_launch = False  # whether the running server was started with sleep mode
        self._eager_wanted = False  # the next launch is for a short recording
        self._eager_launch = False  # the running server was started eager

    @property
    def sleeping(self) -> bool:
        return self._asleep

    def prepare_for(self, duration: float) -> None:
        """Pick the launch mode for a recording; a long one replaces an eager server."""
        with self._lock:
            self._eager_wanted = 0 < duration <= self.settings.fast_start_seconds
            if self.running() and self._eager_launch and not self._eager_wanted:
                self.log_line(f"Restarting {self.label} with CUDA graphs for a {duration / 60:.0f}-minute recording")
                self.stop()

    def _engine(self, path: str, *, timeout: float = 300.0, **params) -> None:
        import requests

        response = requests.post(f"{self.base_url}{path}", params=params, timeout=timeout)
        if not response.ok:
            raise RuntimeError(f"{path} answered HTTP {response.status_code}: {response.text[:200]}")

    def _is_sleeping(self) -> bool:
        import requests

        response = requests.get(f"{self.base_url}/is_sleeping", timeout=10)
        response.raise_for_status()
        return bool(response.json()["is_sleeping"])

    def ensure_ready(self, free_gpu: Callable[[], None], progress: Callable[[str], None] = lambda _m: None,
                     *, evict: bool = True) -> None:
        """Wake a sleeping server when the GPU has room again, else start one."""
        import requests

        with self._lock:
            if self._asleep and self.running():
                # Asleep, the process keeps its CUDA context; waking maps the rest back in.
                needed = self.settings.footprint_mib() - 1024
                self._free_comfy_if_short(free_gpu, needed)
                memory = gpu_memory()
                if memory is not None and memory[0] < needed and evict:
                    memory = reclaim_audio_tools() or memory
                if memory is not None and memory[0] < needed:
                    raise H3Error(f"Not enough free GPU memory to wake {self.label}; unload other models and retry.")
                started = time.monotonic()
                progress(f"Waking {self.label} from host RAM")
                try:
                    self._engine("/wake_up")
                    if self._is_sleeping() or not self.healthy(timeout=10):
                        raise RuntimeError("server is not awake after /wake_up")
                    self._asleep = False
                    self.log_line(f"{self.label} awake after {time.monotonic() - started:.1f}s")
                    return
                except (OSError, RuntimeError, ValueError, KeyError, requests.RequestException) as exc:
                    self.log_line(f"Wake failed; restarting: {exc}")
                    self._sleep_disabled = True
                    self.stop()
            super().ensure_ready(free_gpu, progress, evict=evict)

    def release(self) -> None:
        """Sleep an idle server (weights to host RAM, KV cache dropped), or stop it."""
        import requests

        with self._lock:
            if self._active_requests:
                raise H3Error(f"{self.label} is still transcribing; GPU release was refused.")
            if self._asleep and self.running():
                return
            if (self.running() and self._sleep_launch and not self._uncertain_request
                    and not self._sleep_disabled):
                try:
                    needed = (DOWNLOAD_GB + SLEEP_RAM_MARGIN_GIB) * 2**30
                    if checkpoint.available_ram() < needed:
                        # Not a fault of the server: a later release may find the room.
                        raise ValueError(f"sleep needs {needed / 2**30:.0f} GiB of free host RAM")
                except (OSError, RuntimeError, ValueError, KeyError) as exc:
                    self.log_line(f"Sleep skipped; stopping instead: {exc}")
                else:
                    started = time.monotonic()
                    self.log_line(f"Putting {self.label} to sleep; weights move to host RAM")
                    try:
                        self._engine("/sleep", level=1)
                        if not self._is_sleeping():
                            raise RuntimeError("server does not report sleeping after /sleep")
                        self._asleep = True
                        self.log_line(f"{self.label} asleep after {time.monotonic() - started:.1f}s")
                        return
                    except (OSError, RuntimeError, ValueError, KeyError, requests.RequestException) as exc:
                        self.log_line(f"Sleep failed; stopping instead: {exc}")
                        self._sleep_disabled = True
            self.stop()

    def stop(self) -> None:
        with self._lock:
            super().stop()
            self._asleep = False

    def server_env(self) -> dict[str, str]:
        env = super().server_env()
        if self.settings.sleep_mode == "auto":
            env["VLLM_SERVER_DEV_MODE"] = "1"  # exposes /sleep, /wake_up and /is_sleeping
        env.update(
            VLLM_PLUGINS="nemo_speechlm",
            VLLM_MAX_AUDIO_DECODE_DURATION_S=str(MAX_SECONDS),
            VLLM_MAX_AUDIO_CLIP_FILESIZE_MB="180",
            VLLM_USE_V2_MODEL_RUNNER="0",  # the V2 runner drops CTC word timestamps
            VLLM_DISABLED_KERNELS="FlashInferFP8ScaledMMLinearKernel",
            VLLM_CACHE_ROOT=str(self.settings.runtime_dir / "cache" / "vllm"),
        )
        # FlashInfer compiles its quantized attention with CUDA 13's nvcc.
        if "/usr/local/cuda/bin" not in env.get("PATH", ""):
            env["PATH"] = env["PATH"] + os.pathsep + "/usr/local/cuda/bin"
        return env

    def command(self) -> list[str]:
        settings = self.settings
        memory = gpu_memory()
        # A sleep launch that fails disables sleep, so the retry starts without it.
        self._sleep_launch = settings.sleep_mode == "auto" and not self._sleep_disabled
        self._eager_launch = self._eager_wanted
        # NVIDIA's recipe, without speculative decoding, for one analysis at a time.
        return [
            str(Path(settings.venv) / "bin" / "vllm"), "serve", str(settings.model_dir),
            "--served-model-name", MODEL_REPO, "--host", "127.0.0.1", "--port", str(settings.port),
            "--trust-remote-code", "--dtype", "bfloat16", "--kv-cache-dtype", "fp8",
            "--gpu-memory-utilization", f"{settings.utilization(memory and memory[1]):g}",
            "--max-model-len", "65536", "--max-num-batched-tokens", "4096",
            "--max-num-seqs", str(settings.max_seqs),
            "--enable-chunked-prefill", "--no-enable-prefix-caching", "--use-replayssm",
            "--mamba-backend", "flashinfer", "--mamba-cache-mode", "none", "--mm-processor-cache-gb", "0",
            "--no-async-scheduling", "--limit-mm-per-prompt", '{"audio":{"count":1,"length":57600000}}',
            "--reasoning-parser", "nemotron_v3", "--trust-request-mm-kwargs",
            "--middleware", "nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware",
            *(["--enable-sleep-mode"] if self._sleep_launch else []),
            *(["--enforce-eager", "-O0"] if self._eager_launch else []),
        ]

    def transcribe(self, audio: Path) -> dict:
        """One 16 kHz WAV as the chat completion, with its top-level ``ctc_timestamps``."""
        import requests

        encoded = base64.b64encode(Path(audio).read_bytes()).decode("ascii")
        payload = {
            "model": MODEL_REPO, "temperature": 0, "top_p": 1.0, "max_tokens": MAX_TOKENS, "n": 1,
            "stream": False, "seed": 0, "skip_special_tokens": False, "spaces_between_special_tokens": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "mm_processor_kwargs": {"capture_ctc_timestamps": True},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{encoded}"}}]}],
        }
        del encoded
        with self._lock:
            if self._asleep:
                raise H3Error(f"{self.label} is asleep; wake it before transcribing.")
            self._active_requests += 1
        try:
            response = self.http().post(f"{self.base_url}/v1/chat/completions", json=payload,
                                        timeout=self.settings.request_timeout)
            if not response.ok:
                raise H3Error(f"{self.label} transcription failed (HTTP {response.status_code}): "
                              f"{response.text[:500]}")
            return response.json()
        except requests.RequestException:
            with self._lock:
                self._uncertain_request = True
            raise
        finally:
            with self._lock:
                self._active_requests -= 1


_SERVER: NemotronServer | None = None
_SERVER_LOCK = threading.Lock()


def shared_server(settings: NemotronSettings) -> NemotronServer:
    global _SERVER
    from h3_app.jobs import JOBS

    JOBS.add_gpu_claim_hook(release_for_job)
    with _SERVER_LOCK:
        if _SERVER is None or _SERVER.settings != settings:
            if _SERVER is not None:
                _SERVER.stop()
            _SERVER = NemotronServer(settings)
        return _SERVER


def release_for_job(family: str) -> None:
    """Analysis keeps Nemotron; every other job sleeps it, and an unload stops it."""
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


def _make_room(settings: NemotronSettings) -> None:
    """Room beside other servers: idle dubbing tools stop first, then Drama 3 and MOSS sleep."""
    from h3_app.drama3.serving import server as drama3_server

    from . import moss, tools

    def short() -> bool:
        memory = gpu_memory()
        return memory is not None and memory[0] < settings.footprint_mib(memory[1])

    if short() and tools.stop() and not short():
        return
    if short():
        drama3_server.release_all()
        moss.release_all()


# ---------------------------------------------------------------------------
# Reply
# ---------------------------------------------------------------------------


def speaker_of(tag) -> str | None:
    """``"<spk:0>"``, ``"spk:1"`` or ``2`` → ``speaker1``…; None when the word has no tag."""
    if tag is None or isinstance(tag, bool):
        return None
    if isinstance(tag, (int, float)):
        return segment_tools.speaker_id(int(tag) + 1)
    found = SPEAKER_TAG.search(str(tag))
    return segment_tools.speaker_id(int(found.group(1)) + 1) if found else None


def join_words(words: list[str]) -> str:
    """Words as text: spaced, except between CJK characters and before closing punctuation."""
    text = ""
    for word in words:
        if text and not (CJK.search(text[-1]) and CJK.search(word[0])) and word[0] not in ",.?!;:%)]}，。？！；：、":
            text += " "
        text += word
    return text


def _number(value) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("not finite")
    return number


def reply_words(reply: dict, duration: float) -> list[dict]:
    """Validated ``{"text", "start", "end", "speaker"}`` words in transcript order.

    A word without a tag belongs to the speaker before it (the first ones: to the first tag).
    Anything unusable raises: a failed or truncated reply is retried, never patched up.
    """
    choice = (reply.get("choices") or [{}])[0]
    if choice.get("finish_reason") == "length":
        raise H3Error("Nemotron's transcript was cut off before the end of the recording. Trim the source "
                      "or analyze it in parts.")
    stamps = reply.get("ctc_timestamps")
    if not isinstance(stamps, dict):
        raise H3Error("Nemotron returned no word timings; check that its server runs the CTC timestamp "
                      "middleware, then analyze again.")
    if stamps.get("error"):
        raise H3Error(f"Nemotron could not time the words: {str(stamps['error'])[:300]}. Analyze again.")
    words = []
    for raw in stamps.get("words") or []:
        try:
            text = str(raw.get("word") or "").strip()
            start, end = _number(raw["start"]), _number(raw["end"])
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise H3Error(f"Nemotron returned an unreadable word timing: {str(raw)[:120]}.") from exc
        if not text:
            continue
        if start < 0 or end < start or (duration and end > duration + END_SLACK):
            raise H3Error(f"Nemotron timed {text!r} at {start:.2f}–{end:.2f}s, outside the "
                          f"{duration:.1f}s recording. Analyze again.")
        words.append({"text": text, "start": round(start, 3), "end": round(min(end, duration) if duration else end, 3),
                      "speaker": speaker_of(raw.get("speaker"))})
    if not words:
        raise H3Error("No speech was found in this source.")
    first = next((word["speaker"] for word in words if word["speaker"]), segment_tools.DEFAULT_SPEAKER)
    for word in words:
        word["speaker"] = word["speaker"] or first
        first = word["speaker"]
    return words


def word_lines(words: list[dict]) -> list[dict]:
    """Readable lines, each one speaker's consecutive words; simultaneous speakers stay separate lines."""
    lines, current = [], []

    def close():
        if current:
            lines.append({"start": current[0]["start"], "end": current[-1]["end"], "speaker": current[0]["speaker"],
                          "source": join_words([word["text"] for word in current])})
            current.clear()

    for word in words:
        if current:
            last, length = current[-1], word["end"] - current[0]["start"]
            if (word["speaker"] != last["speaker"] or word["start"] - last["end"] > LINE_PAUSE or length > LINE_MAX
                    or (SENTENCE_END.search(last["text"]) and last["end"] - current[0]["start"] >= SENTENCE_MIN)):
                close()
        current.append(word)
    close()
    return segment_tools.normalize(lines)


def _intervals(spans: list[tuple[float, float]], gap: float = ACTIVITY_GAP) -> list[list[float]]:
    merged: list[list[float]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [[round(start, 3), round(end, 3)] for start, end in merged]


def reply_activity(reply: dict, words: list[dict]) -> dict[str, list[list[float]]]:
    """When each speaker talks: their words, plus the model's diarization where it names them.

    Diarization speech no transcript tag maps to is filed under UNKNOWN_SPEAKER, which counts
    as someone else for everyone, so it never makes a reference clip look clean.
    """
    spans: dict[str, list[tuple[float, float]]] = {}
    for word in words:
        spans.setdefault(word["speaker"], []).append((word["start"], word["end"]))
    stamps = reply.get("ctc_timestamps") or {}
    mapping = stamps.get("speaker_tag_to_diarization_speaker")
    by_label = {str(label): speaker_of(tag) for tag, label in mapping.items()} if isinstance(mapping, dict) else {}
    for entry in stamps.get("diarization") or []:
        try:
            if isinstance(entry, dict):
                start, end = _number(entry["start"]), _number(entry["end"])
                label = entry.get("speaker", entry.get("label"))
            else:
                start, end, label = _number(entry[0]), _number(entry[1]), entry[2]
        except (IndexError, KeyError, TypeError, ValueError):
            continue
        if end > start >= 0:
            spans.setdefault(by_label.get(str(label)) or UNKNOWN_SPEAKER, []).append((start, end))
    return {speaker: _intervals(found) for speaker, found in spans.items()}


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


def transcribe(runtime, audio: Path, *, duration: float, free_gpu: Callable[[], None], hf_token=None,
               report: Callable[[str], None] = lambda _message: None) -> dict:
    """``{"segments", "words", "activity"}`` for ``audio`` (16 kHz mono); the server sleeps once they are back."""
    if not supported():
        raise H3Error("Nemotron runs on vLLM, which requires Linux. Use Qwen3-ASR or Gemini here.")
    if duration > MAX_SECONDS:
        raise H3Error(f"Nemotron transcribes up to {MAX_SECONDS // 60} minutes at once; this source runs "
                      f"{duration / 60:.0f}. Use MOSS or Gemini, or trim the source.")
    settings = NemotronSettings.from_environment(runtime)
    server = shared_server(settings)
    report("Preparing the Nemotron environment (first use installs it)")
    prepare(settings.venv)
    report("Downloading Nemotron 3.5 Transcribe (first use only)")
    ensure_model(settings, token=hf_token)
    try:
        server.prepare_for(duration)
        _make_room(settings)
        server.ensure_ready(free_gpu, report)
        report("Finding speakers, words and overlaps with Nemotron")
        reply = server.transcribe(audio)
    finally:
        try:
            server.release()
        except H3Error as exc:  # still busy elsewhere; the next job's hook sleeps it
            server.log_line(f"Nemotron stays awake: {exc}")
    words = reply_words(reply, duration)
    return {"segments": word_lines(words), "words": words, "activity": reply_activity(reply, words)}
