"""Linux CUDA sleep/wake for the whole managed Omni process group.

Uses NVIDIA's driver checkpoint support, not SIGSTOP (which retains VRAM).
Driver 610 is required because Omni uses CUDA IPC between its three stages.
No CRIU or process-image persistence is involved: CPU processes remain alive.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import requests

NVIDIA_REF = "00d5cce84c628088d6caa203fc4af40c1538b6f7"
HELPER_SHA256 = "707fa7f54136824d6c1d6dd724b9b1717610f831033c00d06da474de363a06db"
HELPER_URL = (f"https://raw.githubusercontent.com/NVIDIA/cuda-checkpoint/{NVIDIA_REF}"
              "/bin/x86_64_Linux/cuda-checkpoint")
GIB = 2**30


def run(command: list[str], timeout: float = 30) -> str:
    result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                            text=True, timeout=timeout)
    if result.returncode:
        operation = " ".join(command[1:])
        raise RuntimeError(f"{Path(command[0]).name} {operation} failed: {(result.stderr or result.stdout)[-500:]}")
    return result.stdout.strip()


def prepare_helper(cache: Path) -> Path:
    """Fetch a checksum-pinned helper only on a compatible Linux GPU host."""
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise RuntimeError("CUDA sleep requires Linux x86_64")
    versions = run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if not versions or any(int(v.split(".")[0]) < 610 for v in versions.splitlines()):
        found = ", ".join(versions.splitlines()) or "unknown"
        raise RuntimeError(f"CUDA sleep requires NVIDIA driver 610+ for Omni's CUDA IPC (found {found})")
    target = cache / "cuda-checkpoint" / NVIDIA_REF / "cuda-checkpoint"
    if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != HELPER_SHA256:
        response = requests.get(HELPER_URL, timeout=30)
        response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest() != HELPER_SHA256:
            raise RuntimeError("NVIDIA checkpoint helper checksum mismatch")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".partial")
        temporary.write_bytes(response.content)
        temporary.chmod(0o700)
        temporary.replace(target)
    target.chmod(0o700)
    help_text = run([str(target), "--help"])
    if "--launch-job" not in help_text or "--action" not in help_text:
        raise RuntimeError("NVIDIA driver does not expose checkpoint job support")
    return target


@dataclass(frozen=True)
class Process:
    pid: int
    group: int
    session: int
    started: int


def process_info(pid: int, proc: Path = Path("/proc")) -> Process:
    # comm may contain spaces and parentheses. Field 3 follows the last ')'.
    fields = (proc / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    if fields[0] == "Z":
        raise ProcessLookupError(f"Process {pid} is a zombie")
    return Process(pid, int(fields[2]), int(fields[3]), int(fields[19]))


def group_processes(group: int, proc: Path = Path("/proc")) -> dict[int, Process]:
    found = {}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            info = process_info(int(entry.name), proc)
        except (FileNotFoundError, ProcessLookupError):
            continue
        if info.group == group and info.session == group:
            found[info.pid] = info
    return found


def cuda_workers(members: dict[int, Process], proc: Path = Path("/proc")) -> list[Process]:
    """Find local CUDA users without NVML's host/virtualized process IDs.

    Modal's gVisor runtime attributes all GPU allocations to PID 1 in NVML.
    Device handles belong to the actual local workers and remain scoped to our
    process group. A control-only handle is insufficient to identify a CUDA user.
    """
    workers = []
    for pid, process in sorted(members.items()):
        for descriptor in (proc / str(pid) / "fd").iterdir():
            try:
                device = os.readlink(descriptor)
            except FileNotFoundError:  # a temporary descriptor was closed
                continue
            if re.fullmatch(r"/dev/nvidia\d+", device):
                workers.append(process)
                break
    return workers


def device_memory() -> tuple[int, int]:
    """Device-wide used/total bytes, independent of NVML's PID attribution."""
    output = run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                  "--format=csv,noheader,nounits", "-i", "0"])
    used, total = (int(part.strip()) * 2**20 for part in output.split(","))
    if used < 0 or total <= 0 or used > total:
        raise RuntimeError(f"Invalid GPU memory counters: {output}")
    return used, total


def available_ram(proc: Path = Path("/proc"), cgroup: Path = Path("/sys/fs/cgroup")) -> int:
    """Host headroom bounded by visible cgroup limits (including parent limits)."""
    info = dict(line.split(":", 1) for line in (proc / "meminfo").read_text().splitlines())
    available = int(info["MemAvailable"].split()[0]) * 1024
    for line in (proc / "self/cgroup").read_text().splitlines():
        _, controllers, relative = line.split(":", 2)
        if controllers == "":
            root, limit_file, used_file = cgroup, "memory.max", "memory.current"
        elif "memory" in controllers.split(","):
            root, limit_file, used_file = cgroup / "memory", "memory.limit_in_bytes", "memory.usage_in_bytes"
        else:
            continue
        # Container mounts often expose the current cgroup at their root.
        paths = [root]
        if ".." not in Path(relative).parts:
            nested = root / relative.lstrip("/")
            while nested != root:
                paths.append(nested)
                nested = nested.parent
        limits_seen = False
        for path in paths:
            if not (path / limit_file).is_file():
                continue
            limits_seen = True
            limit = (path / limit_file).read_text().strip()
            if limit != "max":
                used = int((path / used_file).read_text())
                available = min(available, max(0, int(limit) - used))
        if not limits_seen:
            raise RuntimeError("Cannot determine container RAM headroom for CUDA sleep")
    return available


class CudaCheckpoint:
    def __init__(self, helper: Path, group: int, *, baseline_used: int = 0, log=print):
        self.helper = helper
        self.group = group
        self.members: dict[int, Process] = {}
        self.workers: list[Process] = []
        self.required_vram = 0
        self.transition_started = False
        self.baseline_used = baseline_used
        self.log = log

    def expect_state(self, process: Process, expected: str) -> None:
        if process_info(process.pid) != process:
            raise RuntimeError(f"CUDA worker {process.pid} exited or changed identity")
        state = run([str(self.helper), "--get-state", "--pid", str(process.pid)], timeout=15)
        if state != expected:
            raise RuntimeError(f"CUDA worker {process.pid}: expected {expected}, got {state!r}")

    def action(self, process: Process, action: str) -> None:
        if process_info(process.pid) != process:
            raise RuntimeError(f"CUDA worker {process.pid} exited or changed identity")
        command = [str(self.helper), "--action", action, "--pid", str(process.pid)]
        if action == "lock":
            command += ["--timeout", "10000"]
        run(command, timeout=60)

    def sleep(self) -> None:
        self.members = group_processes(self.group)
        if self.group not in self.members:
            raise RuntimeError("Managed Omni process group is no longer alive")
        self.workers = cuda_workers(self.members)
        # Preprocessing, engine and vocoder each hold a context (the serving
        # parent may too). Fewer means a worker is missing or unreadable.
        if len(self.workers) < 3:
            raise RuntimeError(f"Cannot identify the Omni GPU workers from local device handles; "
                               f"group={list(self.members)}, CUDA workers={[p.pid for p in self.workers]}")
        before, total = device_memory()
        if not 0 <= self.baseline_used < before:
            raise RuntimeError("GPU usage does not exceed the pre-launch baseline; refusing an unverifiable sleep")
        self.required_vram = before - self.baseline_used
        needed = self.required_vram + max(2 * GIB, self.required_vram // 10)
        if available_ram() < needed:
            raise RuntimeError(f"CUDA sleep needs at least {needed / GIB:.1f} GiB of free host/container RAM")
        self.log(f"CUDA sleep workers (local PIDs): {[p.pid for p in self.workers]}; "
                 f"GPU used {before / GIB:.2f} GiB, pre-launch baseline {self.baseline_used / GIB:.2f} GiB")
        for process in self.workers:
            self.expect_state(process, "running")
        if group_processes(self.group) != self.members:
            raise RuntimeError("Omni process topology changed before sleep")
        self.transition_started = True
        # NVIDIA's IPC protocol: lock everyone before checkpointing anyone.
        for process in self.workers:
            self.action(process, "lock")
        for process in self.workers:
            self.action(process, "checkpoint")
        if group_processes(self.group) != self.members:
            raise RuntimeError("Omni process topology changed during sleep")
        for process in self.workers:
            self.expect_state(process, "checkpointed")
        # NVML may lag the driver transition. Allow a small context/accounting
        # margin above the GPU usage recorded before this server was launched.
        deadline = time.monotonic() + 5
        residual_margin = min(GIB, self.required_vram // 20)
        while True:
            after, after_total = device_memory()
            if after_total != total:
                raise RuntimeError("GPU capacity changed during sleep")
            if after <= self.baseline_used + residual_margin:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f"CUDA checkpoint did not release worker GPU memory: "
                                   f"{before / GIB:.2f} -> {after / GIB:.2f} GiB; "
                                   f"baseline {self.baseline_used / GIB:.2f} GiB")
            time.sleep(0.1)
        self.log(f"CUDA sleep verified: GPU used {before / GIB:.2f} -> {after / GIB:.2f} GiB; "
                 f"freed {(before - after) / GIB:.2f} GiB")

    def wake(self) -> None:
        if not self.workers or group_processes(self.group) != self.members:
            raise RuntimeError("Sleeping Omni process topology changed; restart required")
        for process in self.workers:
            self.expect_state(process, "checkpointed")
        # Restore/unlock in the same order used to checkpoint, per NVIDIA's
        # multi-process IPC example. No speech is admitted until all return.
        for process in self.workers:
            self.action(process, "restore")
            self.action(process, "unlock")
            self.expect_state(process, "running")
        self.transition_started = False
        self.log(f"CUDA wake verified for local worker PIDs {[p.pid for p in self.workers]}")
