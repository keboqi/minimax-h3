"""Lifecycle of the managed vLLM-Omni Drama 3 server.

The server starts lazily inside a Drama 3 job, which already holds the app's GPU
lease. Before it starts or wakes, ComfyUI frees its VRAM; when the job ends the
server sleeps at level 1 (weights parked in host RAM), so ComfyUI jobs never run
next to it and the next Drama 3 job wakes in seconds instead of reloading.
"""

from __future__ import annotations

import atexit
import json
import os
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
config.setdefault("platforms", {}).setdefault("cuda", {})["cuda_mps"] = settings["cuda_mps"]
pathlib.Path(sys.argv[1]).write_text(json.dumps(config, indent=2), encoding="utf-8")
"""


@dataclass(frozen=True)
class ServerSettings:
    venv: Path
    data_dir: Path
    model_root: Path
    port: int = 8095
    # The engine's share of total GPU memory: 37 GB of weights plus KV cache.
    engine_memory: float = 0.55
    codec_memory: float = 0.08
    max_seqs: int = 32
    cuda_mps: bool = False
    start_timeout: float = 1800.0
    request_timeout: float = 900.0

    @classmethod
    def from_environment(cls, runtime) -> "ServerSettings":
        env = os.environ
        data_dir = Path(env.get("H3_DRAMA3_VLLM_DATA") or runtime.comfy_dir.parent / "drama3-vllm")
        return cls(
            venv=environment.default_venv(runtime.script_dir),
            data_dir=data_dir,
            model_root=Path(env.get("H3_DRAMA3_VLLM_MODELS") or runtime.comfy_dir / "models" / "drama3_vllm"),
            port=int(env.get("H3_DRAMA3_VLLM_PORT", "8095")),
            engine_memory=float(env.get("H3_DRAMA3_VLLM_ENGINE_MEMORY", "0.55")),
            codec_memory=float(env.get("H3_DRAMA3_VLLM_CODEC_MEMORY", "0.08")),
            max_seqs=int(env.get("H3_DRAMA3_VLLM_MAX_SEQS", "32")),
            cuda_mps=env.get("H3_DRAMA3_VLLM_CUDA_MPS", "0").lower() in {"1", "true", "yes"},
        )

    @property
    def model_dir(self) -> Path:
        return self.model_root / f"{DRAMA3_HF_REPO.replace('/', '--')}--{DRAMA3_HF_REVISION[:12]}"

    @property
    def media_dir(self) -> Path:
        """Reference clips the server may read through file:// URIs."""
        return self.data_dir / "media"


def ensure_model(settings: ServerSettings, *, token=None, log=print) -> Path:
    """Download the pinned official checkpoint once; a marker records completion."""
    target = settings.model_dir
    marker = target / ".h3-complete"
    if marker.is_file():
        return target
    from huggingface_hub import snapshot_download

    log(f"[drama3-vllm] Downloading {DRAMA3_HF_REPO}@{DRAMA3_HF_REVISION[:12]} (~44 GB, first use only).")
    snapshot_download(DRAMA3_HF_REPO, revision=DRAMA3_HF_REVISION, local_dir=target, token=token)
    marker.write_text(DRAMA3_HF_REVISION, encoding="utf-8")
    return target


class Drama3VllmServer:
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
            response = requests.get(f"{self.base_url}/v1/models", timeout=timeout)
            models = {str(item.get("id")) for item in response.json().get("data", [])}
            return response.ok and SERVED_MODEL in models
        except (requests.RequestException, ValueError, AttributeError):
            return False

    def write_deploy_config(self) -> Path:
        settings = self.settings
        target = settings.data_dir / "deploy" / "fish_omni.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        options = {"engine_memory": settings.engine_memory, "codec_memory": settings.codec_memory,
                   "max_seqs": settings.max_seqs, "cuda_mps": settings.cuda_mps}
        subprocess.run([str(environment.python_path(settings.venv)), "-c", DEPLOY_CONFIG_SCRIPT,
                        str(target), json.dumps(options)],
                       check=True, env=environment.install_env(settings.venv), capture_output=True, text=True)
        return target

    def command(self, deploy_config: Path) -> list[str]:
        settings = self.settings
        return [
            str(environment.vllm_executable(settings.venv)), "serve", str(settings.model_dir), "--omni",
            "--host", "127.0.0.1", "--port", str(settings.port),
            "--served-model-name", SERVED_MODEL,
            "--enable-sleep-mode",
            "--deploy-config", str(deploy_config),
            "--allowed-local-media-path", str(settings.media_dir),
        ]

    def server_env(self) -> dict[str, str]:
        cache = self.settings.data_dir / "cache"
        env = environment.install_env(self.settings.venv)
        env.update(
            HF_HUB_OFFLINE="1",  # the checkpoint is local; never re-resolve it at startup
            VLLM_CACHE_ROOT=str(cache / "vllm"),
            TORCHINDUCTOR_CACHE_DIR=str(cache / "torchinductor"),
            TRITON_CACHE_DIR=str(cache / "triton"),
            CUDA_CACHE_PATH=str(cache / "cuda"),
        )
        return env

    def _start(self) -> None:
        settings = self.settings
        settings.media_dir.mkdir(parents=True, exist_ok=True)
        logs = settings.data_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        self.log_path = logs / f"vllm-omni-{time.strftime('%Y%m%d-%H%M%S')}.log"
        self._log = self.log_path.open("ab", buffering=0)
        command = self.command(self.write_deploy_config())
        self._log.write(("$ " + " ".join(command) + "\n").encode("utf-8"))
        self.process = subprocess.Popen(command, env=self.server_env(), stdin=subprocess.DEVNULL,
                                        stdout=self._log, stderr=subprocess.STDOUT, start_new_session=True)
        self.sleeping = False
        atexit.register(self.stop)

    def _wait_healthy(self, progress: Callable[[str], None]) -> None:
        started = time.monotonic()
        while time.monotonic() - started < self.settings.start_timeout:
            if self.healthy():
                return
            if self.process is not None and self.process.poll() is not None:
                raise H3Error(f"vLLM-Omni exited during startup (code {self.process.returncode}). "
                              f"Last log lines:\n{self.log_tail()}")
            progress(f"Starting vLLM-Omni ({time.monotonic() - started:.0f}s; compiles on first start)")
            time.sleep(3)
        raise H3Error(f"vLLM-Omni did not become ready in {self.settings.start_timeout:.0f}s. "
                      f"Last log lines:\n{self.log_tail()}")

    def _post(self, path: str, payload: dict, timeout: float = 120.0):
        response = requests.post(f"{self.base_url}{path}", json=payload, timeout=timeout)
        if not response.ok:
            raise H3Error(f"vLLM-Omni {path} failed (HTTP {response.status_code}): {response.text[:500]}")
        return response.json() if response.content else {}

    @staticmethod
    def _succeeded(result) -> bool:
        acks = result if isinstance(result, list) else result.get("acks", [result])
        return all(str(ack.get("status", "SUCCESS")).upper() in {"SUCCESS", "SKIPPED"} for ack in acks)

    def ensure_awake(self, free_gpu: Callable[[], None], progress: Callable[[str], None] = lambda _m: None) -> None:
        """Start or wake the server after ComfyUI has released its VRAM."""
        with self._lock:
            if self.running() and not self.sleeping and self.healthy():
                return
            free_gpu()
            if self.running() and self.sleeping:
                progress("Waking vLLM-Omni")
                result = self._post("/v1/omni/wakeup", {"stage_ids": [0, 1]}, timeout=600)
                if not self._succeeded(result):
                    raise H3Error(f"vLLM-Omni wake-up failed: {result}")
                self.sleeping = False
                return
            if self.running():
                self.stop()  # alive but unhealthy: restart cleanly
            self._start()
            self._wait_healthy(progress)

    def sleep(self) -> None:
        """Park weights in host RAM so ComfyUI regains the GPU; restart if sleep fails."""
        with self._lock:
            if not self.running() or self.sleeping:
                return
            try:
                result = self._post("/v1/omni/sleep", {"stage_ids": [0, 1], "level": 1}, timeout=300)
                if not self._succeeded(result):
                    raise H3Error(f"vLLM-Omni sleep failed: {result}")
                self.sleeping = True
            except (H3Error, requests.RequestException) as exc:
                print(f"[drama3-vllm] Sleep failed ({exc}); stopping the server to free the GPU.", flush=True)
                self.stop()

    def stop(self) -> None:
        with self._lock:
            process, self.process = self.process, None
            self.sleeping = False
            if process is not None and process.poll() is None:
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
            return "\n".join(self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
        except (OSError, AttributeError):
            return "(no log)"

    def speech(self, payload: dict) -> bytes:
        response = requests.post(f"{self.base_url}/v1/audio/speech", json={"model": SERVED_MODEL, **payload},
                                 timeout=self.settings.request_timeout)
        if not response.ok:
            raise H3Error(f"vLLM-Omni speech request failed (HTTP {response.status_code}): {response.text[:500]}")
        return response.content


_SERVER: Drama3VllmServer | None = None
_SERVER_LOCK = threading.Lock()


def shared_server(settings: ServerSettings) -> Drama3VllmServer:
    """One managed server per app process."""
    global _SERVER
    with _SERVER_LOCK:
        if _SERVER is None or _SERVER.settings != settings:
            if _SERVER is not None:
                _SERVER.stop()
            _SERVER = Drama3VllmServer(settings)
        return _SERVER
