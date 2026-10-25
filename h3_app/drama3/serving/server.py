"""Managed SGLang-Omni Drama 3 server, kept loaded between TTS jobs.

Other engines checkpoint idle CUDA workers to RAM when supported, otherwise
stop the server. The next TTS job restores the workers or restarts the server.
"""

from __future__ import annotations

import atexit
import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests
from requests.adapters import HTTPAdapter

from h3_requirements import DRAMA3_HF_REPO, DRAMA3_HF_REVISION

from h3_app.errors import H3Error

from . import checkpoint, environment

ANSI = re.compile(r"\x1b\[[0-9;]*m")
# Drivers that cannot checkpoint CUDA IPC need Omni's stages to exchange
# tensors through host SHM; this platform, a module for the Omni venv, does so.
SHM_PLATFORM_DIR = Path(__file__).with_name("omni_platform")
SHM_PLATFORM_SPEC = "h3_omni_shm:ShmCudaPlatform"
# Set while a Modal memory snapshot of the loaded server is pending. Modal captures
# the CUDA state itself, so a launch then skips the driver checkpoint job; servers
# started after the restore get sleep again. The captured one stops instead.
SNAPSHOT_CAPTURE = threading.Event()


class SpeechRequestError(H3Error):
    """A speech request the server answered with an HTTP error."""

    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class ServerSettings:
    venv: Path
    data_dir: Path
    model_root: Path
    port: int = 8096
    # The engine's share of total GPU memory: 37 GB of weights plus KV cache.
    engine_memory: float = 0.55
    codec_memory: float = 0.08
    max_seqs: int = 32
    start_timeout: float = 1800.0
    request_timeout: float = 900.0
    sleep_mode: str = "auto"

    def __post_init__(self):
        if self.sleep_mode not in {"auto", "off"}:
            raise ValueError("H3_DRAMA3_SGLANG_SLEEP must be auto or off")

    @classmethod
    def from_environment(cls, runtime) -> "ServerSettings":
        env = os.environ
        return cls(
            venv=environment.default_venv(runtime.script_dir),
            data_dir=Path(env.get("H3_DRAMA3_SGLANG_DATA") or runtime.comfy_dir.parent / "drama3-serving"),
            model_root=Path(env.get("H3_DRAMA3_SGLANG_MODELS") or runtime.comfy_dir / "models" / "drama3_serving"),
            port=int(env.get("H3_DRAMA3_SGLANG_PORT", "8096")),
            engine_memory=float(env.get("H3_DRAMA3_SGLANG_ENGINE_MEMORY", "0.55")),
            codec_memory=float(env.get("H3_DRAMA3_SGLANG_CODEC_MEMORY", "0.08")),
            max_seqs=int(env.get("H3_DRAMA3_SGLANG_MAX_SEQS", "32")),
            sleep_mode=env.get("H3_DRAMA3_SGLANG_SLEEP", "auto").lower(),
        )

    @property
    def model_dir(self) -> Path:
        target = self.model_root / f"{DRAMA3_HF_REPO.replace('/', '--')}--{DRAMA3_HF_REVISION[:12]}"
        if not target.exists():
            # Reuse the pinned checkpoint in an earlier sibling serving directory.
            # Removing a backend must not force another 44 GB download.
            candidates = sorted(self.model_root.parent.glob(f"*/{target.name}"))
            for candidate in candidates:
                marker = candidate / ".h3-complete"
                if marker.is_file() and marker.read_text(encoding="utf-8").strip() == DRAMA3_HF_REVISION:
                    return candidate
        return target

    @property
    def media_dir(self) -> Path:
        """Reference clips the server may read from local paths."""
        return self.data_dir / "media"

    @property
    def runtime_dir(self) -> Path:
        """Persistent SGLang logs and kernel caches."""
        if not (self.data_dir / "cache").is_dir():
            # Keep the existing SGLang kernel cache after renaming serving storage.
            candidates = sorted(path for path in self.data_dir.parent.glob("*/sglang")
                                if (path / "cache").is_dir())
            if len(candidates) == 1:
                return candidates[0]
        return self.data_dir


def ensure_model(settings: ServerSettings, *, token=None, log=print) -> Path:
    """Download the pinned official checkpoint once."""
    target = settings.model_dir
    marker = target / ".h3-complete"
    if marker.is_file():
        return target
    # hf_xet reads this when its first transfer starts; set it before then.
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    from huggingface_hub import snapshot_download

    log(f"[drama3-serving] Downloading {DRAMA3_HF_REPO}@{DRAMA3_HF_REVISION[:12]} (~44 GB, first use only).")
    snapshot_download(DRAMA3_HF_REPO, revision=DRAMA3_HF_REVISION, local_dir=target, token=token)
    marker.write_text(DRAMA3_HF_REVISION, encoding="utf-8")
    return target


def persist_caches() -> None:
    """Commit the Modal volume in the background so compile caches outlive the container.

    Writes to a Modal volume are only kept once committed; without this, every
    container restart or redeploy recompiled from scratch. No-op elsewhere.
    """
    name = os.getenv("H3_MODAL_COMMIT_VOLUME")
    if not name:
        return

    def commit():
        try:
            import modal

            modal.Volume.from_name(name).commit()
            print("[drama3-serving] Committed compile caches to the volume.", flush=True)
        except Exception as exc:  # never fail a job over cache persistence
            print(f"[drama3-serving] Could not commit compile caches: {exc}", flush=True)

    threading.Thread(target=commit, name="drama3-cache-commit", daemon=True).start()


def gpu_memory():
    """(free MiB, total MiB, holder summary) for GPU 0 from nvidia-smi, or None."""
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits", "-i", "0"],
            check=True, capture_output=True, text=True, timeout=15,
        ).stdout.strip().splitlines()[0]
        free, total = (float(part) for part in query.split(","))
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
        return free, total, "; ".join(line.strip() for line in apps.splitlines() if line.strip())
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def reclaim_audio_tools():
    """Stop the resident VoiceDesign/Stable Audio worker; fresh GPU memory if it was running."""
    from h3_app import audio_tools

    return gpu_memory() if audio_tools.stop() else None


class SglangOmniServer:
    label = "SGLang-Omni"
    health_path = "/health"

    def __init__(self, settings: ServerSettings):
        self.settings = settings
        self.process: subprocess.Popen | None = None
        self.log_path: Path | None = None
        self._log = None
        self._lock = threading.RLock()
        self._checkpoint_helper: Path | None = None
        self._checkpoint_baseline = 0
        self._checkpoint: checkpoint.CudaCheckpoint | None = None
        self._sleep_disabled = False
        self._active_requests = 0
        self._uncertain_request = False
        self._session: requests.Session | None = None

    @property
    def sleeping(self) -> bool:
        return self._checkpoint is not None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.settings.port}"

    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def healthy(self, timeout: float = 2.0) -> bool:
        try:
            return requests.get(f"{self.base_url}{self.health_path}", timeout=timeout).ok
        except requests.RequestException:
            return False

    def server_env(self) -> dict[str, str]:
        # Every compile and JIT cache lives with the persistent data, so a later
        # start reuses torch.compile output, Triton/Inductor kernels and
        # FlashInfer's JIT modules instead of rebuilding them (minutes each).
        cache = self.settings.runtime_dir / "cache"
        env = environment.install_env(self.settings.venv)
        env.update(
            HF_HUB_OFFLINE="1",  # the checkpoint is local; never re-resolve it at startup
            TORCHINDUCTOR_CACHE_DIR=str(cache / "torchinductor"),
            TRITON_CACHE_DIR=str(cache / "triton"),
            CUDA_CACHE_PATH=str(cache / "cuda"),
            FLASHINFER_WORKSPACE_BASE=str(cache / "flashinfer"),
            XDG_CACHE_HOME=str(cache / "xdg"),
        )
        return env

    def log_line(self, text: str, *, to_file: bool = True) -> None:
        print(f"[drama3-serving] {text}", flush=True)
        if to_file and self._log is not None:
            self._log.write((text + "\n").encode("utf-8", errors="replace"))

    def check_free_vram(self, progress: Callable[[str], None], *, evict: bool = True) -> None:
        """Fail fast, naming the holders, when the GPU lacks room for the configured share.

        ``evict`` first stops the VoiceDesign/Stable Audio worker to make room.
        """
        memory = gpu_memory()
        if memory is None:
            return
        free, total, holders = memory
        needed = (self.settings.engine_memory + self.settings.codec_memory) * total
        if free < needed and evict:
            memory = reclaim_audio_tools() or memory
            free, total, holders = memory
        message = f"GPU memory before {self.label}: {free / 1024:.1f} of {total / 1024:.1f} GiB free"
        progress(message)
        self.log_line(f"{message}; holders: {holders}")
        if free < needed:
            raise H3Error(
                f"{message}, but {self.label} needs about {needed / 1024:.1f} GiB "
                f"(H3_DRAMA3_SGLANG_ENGINE_MEMORY + CODEC_MEMORY). Still holding VRAM: {holders or 'unknown'}. "
                "Unload models from System and retry."
            )

    def _start(self) -> None:
        settings = self.settings
        settings.media_dir.mkdir(parents=True, exist_ok=True)
        logs = settings.runtime_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        self.log_path = logs / f"sglang-omni-{time.strftime('%Y%m%d-%H%M%S')}.log"
        self._log = self.log_path.open("ab", buffering=0)
        command = self.command()
        env = self.server_env()
        self._checkpoint_helper = None
        self._checkpoint_baseline = 0
        if settings.sleep_mode == "auto" and not self._sleep_disabled and not SNAPSHOT_CAPTURE.is_set():
            try:
                helper, ipc = checkpoint.prepare_helper(settings.runtime_dir / "cache")
                self._checkpoint_baseline = checkpoint.device_memory()[0]
                self._checkpoint_helper = helper
                if ipc:
                    # A fresh driver job per launch is required for inter-worker IPC.
                    command = [str(helper), "--launch-job", *command]
                else:
                    # Stage workers inherit both; install_env already dropped any PYTHONPATH.
                    env.update(PYTHONPATH=str(SHM_PLATFORM_DIR), SGLANG_OMNI_PLATFORM_SPEC=SHM_PLATFORM_SPEC)
                    self.log_line("Driver cannot checkpoint CUDA IPC; Omni stages exchange tensors through host SHM")
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError, requests.RequestException) as exc:
                self._sleep_disabled = True
                self.log_line(f"Sleep unavailable; using stop/restart: {exc}")
        self._uncertain_request = False
        self.log_line(f"Starting {self.label}: {' '.join(command)}")
        self.process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                        stdout=self._log, stderr=subprocess.STDOUT, start_new_session=True)
        atexit.register(self.stop)

    def last_log_line(self) -> str:
        """The latest meaningful server log line, for startup progress."""
        try:
            with self.log_path.open("rb") as stream:
                stream.seek(max(0, stream.seek(0, os.SEEK_END) - 8192))
                tail = stream.read().decode("utf-8", errors="replace")
        except (OSError, AttributeError):
            return ""
        lines = [ANSI.sub("", line).strip() for line in re.split(r"[\r\n]+", tail)]
        lines = [line for line in lines if line]
        return lines[-1][-180:] if lines else ""

    def _wait_healthy(self, progress: Callable[[str], None]) -> None:
        started, reported = time.monotonic(), 0.0
        while time.monotonic() - started < self.settings.start_timeout:
            if self.healthy():
                self.log_line(f"{self.label} ready after {time.monotonic() - started:.0f}s")
                persist_caches()
                return
            if self.process is not None and self.process.poll() is not None:
                raise H3Error(f"{self.label} exited during startup (code {self.process.returncode}). "
                              f"Last log lines:\n{self.log_tail()}")
            elapsed = time.monotonic() - started
            message = f"Starting {self.label} ({elapsed:.0f}s): {self.last_log_line()}"
            progress(message)
            if elapsed - reported >= 30:  # Modal logs show startup without the UI
                self.log_line(message, to_file=False)  # never echo back into the tailed log
                reported = elapsed
            time.sleep(3)
        raise H3Error(f"{self.label} did not become ready in {self.settings.start_timeout:.0f}s. "
                      f"Last log lines:\n{self.log_tail()}")

    def ensure_ready(self, free_gpu: Callable[[], None], progress: Callable[[str], None] = lambda _m: None,
                     *, evict: bool = True) -> None:
        """Free ComfyUI's VRAM, then restore, reuse, or start the server.

        Without ``evict`` the VoiceDesign/Stable Audio worker stays loaded; a GPU
        without room for the server then raises instead.
        """
        with self._lock:
            free_gpu()
            if self.sleeping and self.running():
                # Leave it asleep on insufficient VRAM so the user can unload
                # other models and retry without losing the checkpoint.
                memory = gpu_memory()
                if memory is not None and memory[0] * 2**20 < self._checkpoint.required_vram and evict:
                    memory = reclaim_audio_tools() or memory
                if memory is not None and memory[0] * 2**20 < self._checkpoint.required_vram:
                    raise H3Error("Not enough free GPU memory to wake SGLang; unload other models and retry.")
                started = time.monotonic()
                progress("Waking SGLang-Omni from host RAM")
                try:
                    self._checkpoint.wake()
                    if not self.healthy(timeout=10):
                        raise RuntimeError("Server is unhealthy after CUDA restore")
                    self._checkpoint = None
                    self.log_line(f"{self.label} awake after {time.monotonic() - started:.1f}s")
                    return
                except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
                    self.log_line(f"Wake failed; restarting: {exc}")
                    self._sleep_disabled = True
                    self.stop()
            if self.running() and self.healthy():
                return
            # Also clean up surviving workers when the parent has exited.
            if self.process is not None:
                self.stop()
            self.check_free_vram(progress, evict=evict)
            try:
                self._launch(progress)
            except (H3Error, OSError):
                # Only a sleep-enabled launch (job wrapper or SHM platform)
                # earns an ordinary retry.
                if self._sleep_disabled or self.settings.sleep_mode != "auto":
                    raise
                self._sleep_disabled = True
                self.log_line("Checkpoint-enabled launch failed; retrying ordinary startup once")
                self.check_free_vram(progress, evict=evict)
                self._launch(progress)

    def _launch(self, progress: Callable[[str], None]) -> None:
        try:
            self._start()
            self._wait_healthy(progress)
        except (H3Error, OSError):
            self.stop()
            raise

    def release(self) -> None:
        """Sleep an idle pipeline; a failed transition destroys the whole group."""
        with self._lock:
            if self._active_requests:
                raise H3Error("SGLang is still generating speech; GPU release was refused.")
            if self.sleeping and self.running():
                return
            if (self.running() and self._checkpoint_helper is not None
                    and not self._uncertain_request and not self._sleep_disabled):
                started = time.monotonic()
                self.log_line("Putting SGLang-Omni to sleep; preserving CUDA state in host RAM")
                self._checkpoint = checkpoint.CudaCheckpoint(
                    self._checkpoint_helper, self.process.pid,
                    baseline_used=self._checkpoint_baseline, log=self.log_line,
                )
                try:
                    self._checkpoint.sleep()
                    self.log_line(f"{self.label} asleep after {time.monotonic() - started:.1f}s; GPU state kept in host RAM")
                    return
                except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as exc:
                    self.log_line(f"Sleep unavailable; stopping instead: {exc}")
                    # Precondition misses (RAM headroom, topology) can pass on a
                    # later launch; a driver transition failure is not retried.
                    self._sleep_disabled = self._checkpoint.transition_started
            self.stop()

    def stop(self) -> None:
        with self._lock:
            process = self.process
            if process is not None:
                self.log_line(f"Stopping {self.label}")
                # A checkpoint failure can leave CUDA APIs locked. SIGTERM may
                # deadlock cleanup; destroy the group instead of trying to reuse it.
                force = self._checkpoint is not None and self._checkpoint.transition_started
                try:
                    os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
                    process.wait(timeout=30)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
                # The parent may exit before its workers, including after a crash.
                # Do not hand the GPU over until remaining group members are killed.
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    deadline = time.monotonic() + 10
                    while checkpoint.group_processes(process.pid):
                        if time.monotonic() >= deadline:
                            raise H3Error("SGLang workers did not exit; GPU handoff was refused.")
                        time.sleep(0.1)
                process.wait(timeout=30)
            self.process = None
            self._checkpoint = None
            self._checkpoint_helper = None
            self._uncertain_request = False
            if self._session is not None:
                self._session.close()
                self._session = None
            if self._log is not None:
                self._log.close()
                self._log = None

    def log_tail(self, lines: int = 25) -> str:
        try:
            text = ANSI.sub("", self.log_path.read_text(encoding="utf-8", errors="replace"))
            return "\n".join(text.splitlines()[-lines:])
        except (OSError, AttributeError):
            return "(no log)"

    def http(self) -> requests.Session:
        """One pooled session, so parallel chunk requests reuse their connections."""
        with self._lock:
            if self._session is None:
                session = requests.Session()
                size = max(1, int(getattr(self.settings, "max_seqs", 32)))
                session.mount("http://", HTTPAdapter(pool_connections=1, pool_maxsize=size))
                self._session = session
            return self._session

    def speech(self, payload: dict) -> bytes:
        with self._lock:
            if self.sleeping:
                raise H3Error("SGLang is asleep; wake it before generating speech.")
            self._active_requests += 1
        try:
            response = self.http().post(f"{self.base_url}/v1/audio/speech", json=payload,
                                        timeout=self.settings.request_timeout)
            if not response.ok:
                raise SpeechRequestError(f"{self.label} speech request failed (HTTP {response.status_code}): "
                                         f"{response.text[:500]}", response.status_code)
            return response.content
        except requests.RequestException:
            # A client timeout does not prove GPU work has stopped remotely.
            with self._lock:
                self._uncertain_request = True
            raise
        finally:
            with self._lock:
                self._active_requests -= 1

    def command(self) -> list[str]:
        settings = self.settings
        return [
            str(environment.executable(settings.venv, "sgl-omni")), "serve",
            "--model-path", str(settings.model_dir), "--host", "127.0.0.1", "--port", str(settings.port),
            "--tts_engine.gpu_memory_fraction", str(settings.engine_memory),
            "--vocoder.gpu_memory_fraction", str(settings.codec_memory),
            "--tts_engine.factory.max_running_requests", str(settings.max_seqs),
            # Upstream's defaults: prefill batches up to 2048 tokens and first frames
            # of up to 16 requests replay as CUDA graphs. Sleep keeps them across
            # engine switches, so their capture is paid once per cold start.
            "--tts_engine.engine.cuda_graph_backend_prefill", "breakable",
            "--tts_engine.factory.first_frame_cuda_graph_max_requests", "16",
            # This route requests complete WAVs, not streaming codec windows.
            "--vocoder.factory.window_cuda_graphs", "false",
            # SGLang-Omni defaults to fishaudio/s2-pro's codec.pth from the Hub; the
            # checkpoint ships the byte-identical file (sha256 74fc41c5...), so use it
            # and keep startup offline.
            "--vocoder.factory.codec_path", str(settings.model_dir / "codec.pth"),
        ]

    def payload(self, *, text, reference, transcript, max_new_tokens, sampling) -> dict:
        # FishOmni rejects per-request seeds on SGLang-Omni's speech endpoint.
        sampling = dict(sampling)
        # The Drama 3 request builder requires a positive top_k. SGLang uses
        # 1 << 30 (TOP_K_ALL) to represent the whole vocabulary internally.
        # It uncaps only the main token; codebook heads stay capped at 30.
        if sampling.get("top_k") == 0:
            sampling["top_k"] = 1 << 30
        # No reference lets Drama 3 invent the voice from the text's cues (voice design).
        references = [] if reference is None else [{"audio_path": str(Path(reference).resolve()), "text": transcript}]
        from h3_app.drama3.phonemes import to_phoneme_tags

        # Pronunciation notes (银行{yin2 hang2}) reach Drama 3 as its phoneme tags.
        return {"input": to_phoneme_tags(text), "response_format": "wav", "max_new_tokens": max_new_tokens,
                **sampling, "references": references}


_SERVER: SglangOmniServer | None = None
_SERVER_LOCK = threading.Lock()


def shared_server(settings: ServerSettings) -> SglangOmniServer:
    """Reuse the app's SGLang server while its configuration stays unchanged."""
    global _SERVER
    from h3_app.jobs import JOBS

    JOBS.add_gpu_claim_hook(release_for_job)
    with _SERVER_LOCK:
        if _SERVER is None or _SERVER.settings != settings:
            if _SERVER is not None:
                _SERVER.stop()
            _SERVER = SglangOmniServer(settings)
        return _SERVER


def release_for_job(family: str) -> None:
    """Other engines reclaim the GPU; ComfyUI TTS releases the server itself.

    Voice and sound design, Auto's preload and Dub & Translate share the GPU with a
    loaded server; dubbing's transcription tools release it only when VRAM is short.
    """
    if family == "unload":
        stop_all()
    elif family not in {"drama3", "drama3_voice", "drama3_sound", "drama3_preload",
                        "dubbing_analyze", "dubbing_lines", "dubbing"}:
        release_all()


def server_ready() -> bool:
    """Whether the managed SGLang server is running, awake and healthy."""
    with _SERVER_LOCK:
        return _SERVER is not None and _SERVER.running() and not _SERVER.sleeping and _SERVER.healthy()


def release_all() -> None:
    """Release the GPU held by the managed SGLang server."""
    with _SERVER_LOCK:
        if _SERVER is not None:
            _SERVER.release()


def stop_all() -> None:
    """Explicit unload/shutdown also discards sleeping CPU state."""
    with _SERVER_LOCK:
        if _SERVER is not None:
            _SERVER.stop()
