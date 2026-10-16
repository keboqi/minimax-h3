"""Lifecycle of the managed Drama 3 servers (vLLM-Omni and SGLang-Omni).

A server starts lazily inside a Drama 3 job, which already holds the app's GPU
lease, after ComfyUI frees its VRAM. It then stays loaded, so the next Drama 3
job starts generating at once. When any other engine takes the GPU lease the
server is released: stopped by default. vLLM-Omni's level-1 sleep is available
behind H3_DRAMA3_VLLM_RELEASE=sleep, but with Drama 3 (vllm-omni c4871a0) the
engine woke in a second and then never scheduled another request.
"""

from __future__ import annotations

import atexit
import json
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

from h3_requirements import DRAMA3_HF_REPO, DRAMA3_HF_REVISION

from h3_app.errors import H3Error

from . import environment

SERVED_MODEL = "drama-3"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
TRUTHY = {"1", "true", "yes"}
# Generated from the package's fish_omni.yaml with the venv's PyYAML, so the main
# environment needs no YAML dependency. JSON output is valid YAML for vLLM.
DEPLOY_CONFIG_SCRIPT = r"""
import json, pathlib, sys, yaml
import vllm_omni
settings = json.loads(sys.argv[2])
base = pathlib.Path(vllm_omni.__file__).parent / "deploy" / "fish_omni.yaml"
config = yaml.safe_load(base.read_text(encoding="utf-8"))
stages = {stage["stage_id"]: stage for stage in config["stages"]}
if set(stages) != {0, 1}:
    raise SystemExit("Unexpected fish_omni deploy schema; expected stages 0 and 1")
stages[0].update(gpu_memory_utilization=settings["engine_memory"], max_num_seqs=settings["max_seqs"])
stages[1].update(gpu_memory_utilization=settings["codec_memory"], max_num_seqs=settings["max_seqs"])
# The shipped config captures 39 graph sizes up to 4096 tokens: minutes at startup.
# Capture only batch sizes this app sends; larger prefill steps run eagerly.
sizes = [size for size in (1, 2, 4, 8, 16, 24, 32, 48, 64) if size <= settings["max_seqs"]]
compilation = stages[0].setdefault("compilation_config", {})
compilation.update(cudagraph_capture_sizes=sizes, max_cudagraph_capture_size=max(sizes))
config.setdefault("platforms", {}).setdefault("cuda", {})["cuda_mps"] = settings["cuda_mps"]
pathlib.Path(sys.argv[1]).write_text(json.dumps(config, indent=2), encoding="utf-8")
"""


@dataclass(frozen=True)
class ServerSettings:
    venv: Path
    data_dir: Path
    model_root: Path
    kind: str = "vllm"  # "vllm" or "sglang"
    port: int = 8095
    # The engine's share of total GPU memory: 37 GB of weights plus KV cache.
    engine_memory: float = 0.55
    codec_memory: float = 0.08
    max_seqs: int = 32
    cuda_mps: bool = False
    # DeepGEMM requantizes every FP8 expert through float32 at load time; under
    # sleep mode's dedicated weights pool that exhausted 95 GiB on an RTX PRO 6000.
    deep_gemm: bool = False
    release: str = "stop"  # vLLM-Omni only: "stop" or "sleep"
    start_timeout: float = 1800.0
    request_timeout: float = 900.0

    @classmethod
    def from_environment(cls, runtime, kind: str = "vllm") -> "ServerSettings":
        env = os.environ
        sglang = kind == "sglang"
        return cls(
            venv=environment.default_venv(runtime.script_dir, kind),
            data_dir=Path(env.get("H3_DRAMA3_VLLM_DATA") or runtime.comfy_dir.parent / "drama3-vllm"),
            model_root=Path(env.get("H3_DRAMA3_VLLM_MODELS") or runtime.comfy_dir / "models" / "drama3_vllm"),
            kind=kind,
            port=int(env.get("H3_DRAMA3_SGLANG_PORT", "8096") if sglang else env.get("H3_DRAMA3_VLLM_PORT", "8095")),
            engine_memory=float(env.get("H3_DRAMA3_VLLM_ENGINE_MEMORY", "0.55")),
            codec_memory=float(env.get("H3_DRAMA3_VLLM_CODEC_MEMORY", "0.08")),
            max_seqs=int(env.get("H3_DRAMA3_VLLM_MAX_SEQS", "32")),
            cuda_mps=env.get("H3_DRAMA3_VLLM_CUDA_MPS", "0").lower() in TRUTHY,
            deep_gemm=env.get("H3_DRAMA3_VLLM_DEEP_GEMM", "0").lower() in TRUTHY,
            release="stop" if sglang else env.get("H3_DRAMA3_VLLM_RELEASE", "stop").lower(),
        )

    @property
    def model_dir(self) -> Path:
        return self.model_root / f"{DRAMA3_HF_REPO.replace('/', '--')}--{DRAMA3_HF_REVISION[:12]}"

    @property
    def media_dir(self) -> Path:
        """Reference clips the server may read from local paths."""
        return self.data_dir / "media"

    @property
    def runtime_dir(self) -> Path:
        """Logs, caches and generated config for this backend."""
        return self.data_dir if self.kind == "vllm" else self.data_dir / "sglang"


def ensure_model(settings: ServerSettings, *, token=None, log=print) -> Path:
    """Download the pinned official checkpoint once (shared by both backends)."""
    target = settings.model_dir
    marker = target / ".h3-complete"
    if marker.is_file():
        return target
    from huggingface_hub import snapshot_download

    log(f"[drama3-serving] Downloading {DRAMA3_HF_REPO}@{DRAMA3_HF_REVISION[:12]} (~44 GB, first use only).")
    snapshot_download(DRAMA3_HF_REPO, revision=DRAMA3_HF_REVISION, local_dir=target, token=token)
    marker.write_text(DRAMA3_HF_REVISION, encoding="utf-8")
    return target


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


class ManagedServer:
    label = "Drama 3 server"
    health_path = "/health"

    def __init__(self, settings: ServerSettings):
        self.settings = settings
        self.process: subprocess.Popen | None = None
        self.sleeping = False
        self.log_path: Path | None = None
        self._log = None
        self._lock = threading.RLock()

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

    def command(self) -> list[str]:
        raise NotImplementedError

    def server_env(self) -> dict[str, str]:
        cache = self.settings.runtime_dir / "cache"
        env = environment.install_env(self.settings.venv)
        env.update(
            HF_HUB_OFFLINE="1",  # the checkpoint is local; never re-resolve it at startup
            TORCHINDUCTOR_CACHE_DIR=str(cache / "torchinductor"),
            TRITON_CACHE_DIR=str(cache / "triton"),
            CUDA_CACHE_PATH=str(cache / "cuda"),
        )
        return env

    def log_line(self, text: str) -> None:
        print(f"[drama3-serving] {text}", flush=True)
        if self._log is not None:
            self._log.write((text + "\n").encode("utf-8", errors="replace"))

    def check_free_vram(self, progress: Callable[[str], None]) -> None:
        """Fail fast, naming the holders, when the GPU lacks room for the configured share."""
        memory = gpu_memory()
        if memory is None:
            return
        free, total, holders = memory
        needed = (self.settings.engine_memory + self.settings.codec_memory) * total
        message = f"GPU memory before {self.label}: {free / 1024:.1f} of {total / 1024:.1f} GiB free"
        progress(message)
        self.log_line(f"{message}; holders: {holders}")
        if free < needed:
            raise H3Error(
                f"{message}, but {self.label} needs about {needed / 1024:.1f} GiB "
                f"(H3_DRAMA3_VLLM_ENGINE_MEMORY + CODEC_MEMORY). Still holding VRAM: {holders or 'unknown'}. "
                "Unload models from System and retry."
            )

    def _start(self) -> None:
        settings = self.settings
        settings.media_dir.mkdir(parents=True, exist_ok=True)
        logs = settings.runtime_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        self.log_path = logs / f"{settings.kind}-omni-{time.strftime('%Y%m%d-%H%M%S')}.log"
        self._log = self.log_path.open("ab", buffering=0)
        command = self.command()
        self.log_line(f"Starting {self.label}: {' '.join(command)}")
        self.process = subprocess.Popen(command, env=self.server_env(), stdin=subprocess.DEVNULL,
                                        stdout=self._log, stderr=subprocess.STDOUT, start_new_session=True)
        self.sleeping = False
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
                return
            if self.process is not None and self.process.poll() is not None:
                raise H3Error(f"{self.label} exited during startup (code {self.process.returncode}). "
                              f"Last log lines:\n{self.log_tail()}")
            elapsed = time.monotonic() - started
            message = f"Starting {self.label} ({elapsed:.0f}s): {self.last_log_line()}"
            progress(message)
            if elapsed - reported >= 30:  # Modal logs show startup without the UI
                self.log_line(message)
                reported = elapsed
            time.sleep(3)
        raise H3Error(f"{self.label} did not become ready in {self.settings.start_timeout:.0f}s. "
                      f"Last log lines:\n{self.log_tail()}")

    def ensure_ready(self, free_gpu: Callable[[], None], progress: Callable[[str], None] = lambda _m: None) -> None:
        """Reuse a loaded server; otherwise free ComfyUI's VRAM and start it."""
        with self._lock:
            if self.running() and not self.sleeping and self.healthy():
                return
            free_gpu()
            self.check_free_vram(progress)
            if self.running():
                self.stop()  # alive but unhealthy: restart cleanly
            self._start()
            self._wait_healthy(progress)

    def release(self) -> None:
        """Give the GPU back to other engines."""
        self.stop()

    def stop(self) -> None:
        with self._lock:
            process, self.process = self.process, None
            self.sleeping = False
            if process is not None and process.poll() is None:
                self.log_line(f"Stopping {self.label}")
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=30)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
            if self._log is not None:
                self._log.close()
                self._log = None

    def log_tail(self, lines: int = 25) -> str:
        try:
            text = ANSI.sub("", self.log_path.read_text(encoding="utf-8", errors="replace"))
            return "\n".join(text.splitlines()[-lines:])
        except (OSError, AttributeError):
            return "(no log)"

    def payload(self, *, text, reference, transcript, seed, max_new_tokens, sampling) -> dict:
        raise NotImplementedError

    def speech(self, payload: dict) -> bytes:
        response = requests.post(f"{self.base_url}/v1/audio/speech", json=payload,
                                 timeout=self.settings.request_timeout)
        if not response.ok:
            raise H3Error(f"{self.label} speech request failed (HTTP {response.status_code}): {response.text[:500]}")
        return response.content


class VllmOmniServer(ManagedServer):
    label = "vLLM-Omni"
    health_path = "/v1/models"

    def healthy(self, timeout: float = 2.0) -> bool:
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=timeout)
            models = {str(item.get("id")) for item in response.json().get("data", [])}
            return response.ok and SERVED_MODEL in models
        except (requests.RequestException, ValueError, AttributeError):
            return False

    def write_deploy_config(self) -> Path:
        settings = self.settings
        target = settings.runtime_dir / "deploy" / "fish_omni.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        options = {"engine_memory": settings.engine_memory, "codec_memory": settings.codec_memory,
                   "max_seqs": settings.max_seqs, "cuda_mps": settings.cuda_mps}
        subprocess.run([str(environment.python_path(settings.venv)), "-c", DEPLOY_CONFIG_SCRIPT,
                        str(target), json.dumps(options)],
                       check=True, env=environment.install_env(settings.venv), capture_output=True, text=True)
        return target

    def command(self, deploy_config: Path | None = None) -> list[str]:
        settings = self.settings
        deploy_config = deploy_config or self.write_deploy_config()
        command = [
            str(environment.executable(settings.venv, "vllm")), "serve", str(settings.model_dir), "--omni",
            "--host", "127.0.0.1", "--port", str(settings.port),
            "--served-model-name", SERVED_MODEL,
            "--deploy-config", str(deploy_config),
            "--allowed-local-media-path", str(settings.media_dir),
        ]
        if settings.release == "sleep":
            command.append("--enable-sleep-mode")
        return command

    def server_env(self) -> dict[str, str]:
        env = super().server_env()
        env["VLLM_CACHE_ROOT"] = str(self.settings.runtime_dir / "cache" / "vllm")
        if not self.settings.deep_gemm:
            env.update(VLLM_USE_DEEP_GEMM="0", VLLM_MOE_USE_DEEP_GEMM="0")
        return env

    def _post(self, path: str, payload: dict, timeout: float = 120.0):
        response = requests.post(f"{self.base_url}{path}", json=payload, timeout=timeout)
        if not response.ok:
            raise H3Error(f"vLLM-Omni {path} failed (HTTP {response.status_code}): {response.text[:500]}")
        return response.json() if response.content else {}

    @staticmethod
    def _succeeded(result) -> bool:
        acks = result if isinstance(result, list) else result.get("acks", [result])
        return all(str(ack.get("status", "SUCCESS")).upper() in {"SUCCESS", "SKIPPED"} for ack in acks)

    def _responds(self, timeout: float = 120.0) -> bool:
        """A woken engine must finish a tiny request; Drama 3 hung after wake-up in testing."""
        try:
            response = requests.post(f"{self.base_url}/v1/audio/speech", timeout=timeout,
                                     json={"model": SERVED_MODEL, "input": "Hello.", "response_format": "wav",
                                           "max_new_tokens": 32})
            return response.ok and len(response.content) > 44
        except requests.RequestException:
            return False

    def ensure_ready(self, free_gpu: Callable[[], None], progress: Callable[[str], None] = lambda _m: None) -> None:
        with self._lock:
            if self.running() and self.sleeping:
                free_gpu()
                self.check_free_vram(progress)
                progress("Waking vLLM-Omni")
                result = self._post("/v1/omni/wakeup", {"stage_ids": [0, 1]}, timeout=600)
                self.sleeping = False
                if self._succeeded(result) and self._responds():
                    return
                self.log_line("vLLM-Omni did not respond after waking; restarting it.")
                self.stop()
            super().ensure_ready(free_gpu, progress)

    def release(self) -> None:
        with self._lock:
            if self.settings.release != "sleep":
                self.stop()
                return
            if not self.running() or self.sleeping:
                return
            try:
                result = self._post("/v1/omni/sleep", {"stage_ids": [0, 1], "level": 1}, timeout=300)
                if not self._succeeded(result):
                    raise H3Error(f"vLLM-Omni sleep failed: {result}")
                self.sleeping = True
                self.log_line("vLLM-Omni is sleeping (weights in host RAM).")
            except (H3Error, requests.RequestException) as exc:
                self.log_line(f"Sleep failed ({exc}); stopping the server to free the GPU.")
                self.stop()

    def payload(self, *, text, reference, transcript, seed, max_new_tokens, sampling) -> dict:
        return {"model": SERVED_MODEL, "input": text, "response_format": "wav", "seed": seed,
                "max_new_tokens": max_new_tokens, "extra_params": sampling,
                "ref_audio": Path(reference).resolve().as_uri(), "ref_text": transcript}


class SglangOmniServer(ManagedServer):
    label = "SGLang-Omni"

    def command(self) -> list[str]:
        settings = self.settings
        return [
            str(environment.executable(settings.venv, "sgl-omni")), "serve",
            "--model-path", str(settings.model_dir), "--host", "127.0.0.1", "--port", str(settings.port),
            "--tts_engine.gpu_memory_fraction", str(settings.engine_memory),
            "--vocoder.gpu_memory_fraction", str(settings.codec_memory),
            "--tts_engine.factory.max_running_requests", str(settings.max_seqs),
        ]

    def payload(self, *, text, reference, transcript, seed, max_new_tokens, sampling) -> dict:
        return {"input": text, "response_format": "wav", "seed": seed, "max_new_tokens": max_new_tokens,
                **sampling, "references": [{"audio_path": str(Path(reference).resolve()), "text": transcript}]}


SERVER_CLASSES = {"vllm": VllmOmniServer, "sglang": SglangOmniServer}
_SERVERS: dict[str, ManagedServer] = {}
_SERVERS_LOCK = threading.Lock()


def shared_server(settings: ServerSettings) -> ManagedServer:
    """One managed server per backend per app process."""
    from h3_app.jobs import JOBS

    JOBS.add_gpu_claim_hook(release_for_job)
    with _SERVERS_LOCK:
        current = _SERVERS.get(settings.kind)
        if current is None or current.settings != settings:
            if current is not None:
                current.stop()
            current = _SERVERS[settings.kind] = SERVER_CLASSES[settings.kind](settings)
        return current


def release_for_job(family: str) -> None:
    """GPU-claim hook: every job but Drama 3 releases the servers.

    Drama 3 jobs decide for themselves: the served route keeps its own backend
    loaded, and the ComfyUI route releases both before running.
    """
    if family != "drama3":
        release_all()


def release_all(except_kind: str | None = None) -> None:
    """Free the GPU from every Drama 3 server, optionally keeping one backend."""
    with _SERVERS_LOCK:
        servers = [server for kind, server in _SERVERS.items() if kind != except_kind]
    for server in servers:
        server.release()
