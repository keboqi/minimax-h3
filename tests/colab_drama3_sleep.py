"""Manual Colab probe: real Drama 3 SGLang sleep/wake cycles with speech after each.

Stop the H3 app first (it shares the GPU and port 8096), then run from the
workspace with the app's Python: `python3 -m tests.colab_drama3_sleep`.
On first use it prepares the SGLang venv and downloads the ~44 GB checkpoint.
On drivers 580-609 it also proves the stage workers run the host SHM platform.
`--compare-ipc` finally restarts without sleep (stock CUDA IPC) to time the
same speech, so the SHM transport's cost is visible. Prints a JSON report and
exits non-zero when any cycle failed.
"""

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from h3_app.drama3.serving import checkpoint, environment, generate
from h3_app.drama3.serving.server import SHM_PLATFORM_SPEC, ServerSettings, SglangOmniServer, ensure_model
from h3_app.workflows.drama3 import MAX_FRAMES

ROOT = Path(__file__).resolve().parents[1]
TEXT = "The lighthouse keeper counted the ships twice, then wrote the number down before the storm arrived."
SAMPLING = {"temperature": 2.0, "top_p": 1.0, "top_k": 30}  # the Drama 3 tab's defaults


def speak(server, count: int) -> dict:
    started = time.monotonic()
    checks = []
    for _ in range(count):
        attempt = generate.best_take(server, text=TEXT, reference=None, transcript=None,
                                     max_new_tokens=generate.frame_budget(TEXT, MAX_FRAMES),
                                     sampling=SAMPLING, attempts=1, stopped=threading.Event())
        checks.append({"ok": attempt.check.ok, "reason": attempt.check.reason,
                       "seconds": round(attempt.check.seconds, 2)})
    return {"request_seconds": round((time.monotonic() - started) / count, 2), "takes": checks}


def worker_platforms(server) -> dict[int, str]:
    """The Omni platform each GPU worker was launched with, read from its environment."""
    found = {}
    for process in checkpoint.cuda_workers(checkpoint.group_processes(server.process.pid)):
        entries = Path(f"/proc/{process.pid}/environ").read_bytes().split(b"\0")
        values = dict(entry.decode(errors="replace").split("=", 1) for entry in entries if b"=" in entry)
        found[process.pid] = values.get("SGLANG_OMNI_PLATFORM_SPEC", "stock")
    return found


def gib(value: int) -> float:
    return round(value / checkpoint.GIB, 2)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comfy-dir", type=Path,
                        default=Path(os.environ.get("COMFY_DIR") or ROOT / "h3" / "ComfyUI"))
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--requests", type=int, default=2, help="speech requests timed per phase")
    parser.add_argument("--compare-ipc", action="store_true")
    args = parser.parse_args(argv)

    os.environ.pop("H3_DRAMA3_SGLANG_SLEEP", None)  # this probe is about sleep=auto
    settings = ServerSettings.from_environment(SimpleNamespace(script_dir=ROOT, comfy_dir=args.comfy_dir))
    report: dict = {"driver": checkpoint.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
                    "available_ram_gib": gib(checkpoint.available_ram()), "cycles": []}
    try:
        report["driver_checkpoints_ipc"] = checkpoint.prepare_helper(settings.runtime_dir / "cache")[1]
    except RuntimeError as exc:
        report["error"] = f"Sleep is unavailable on this host: {exc}"
        print(json.dumps(report, indent=2))
        return 1

    environment.prepare(settings.venv)
    ensure_model(settings, token=os.environ.get("HF_TOKEN"))
    server = SglangOmniServer(settings)
    try:
        started = time.monotonic()
        server.ensure_ready(lambda: None, lambda message: print(message, flush=True))
        report["cold_start_seconds"] = round(time.monotonic() - started, 1)
        if server._checkpoint_helper is None:
            raise RuntimeError("The server started without sleep; see 'Sleep unavailable' in the log")
        platforms = worker_platforms(server)
        report["worker_platforms"] = platforms
        shm = not report["driver_checkpoints_ipc"]
        # Workers re-export the spec as the platform class's module path.
        expected = SHM_PLATFORM_SPEC.split(":")[0]
        if shm and not all(spec.startswith(expected) for spec in platforms.values()):
            raise RuntimeError(f"Not every GPU worker runs the SHM platform: {platforms}")
        report["awake_speech"] = speak(server, args.requests)
        for cycle in range(1, args.cycles + 1):
            used_awake = checkpoint.device_memory()[0]
            started = time.monotonic()
            server.release()
            sleep_seconds = time.monotonic() - started
            if not server.sleeping:
                raise RuntimeError(f"Cycle {cycle}: the server stopped instead of sleeping")
            used_asleep = checkpoint.device_memory()[0]
            started = time.monotonic()
            server.ensure_ready(lambda: None)
            wake_seconds = time.monotonic() - started
            if server.sleeping or server._sleep_disabled:
                raise RuntimeError(f"Cycle {cycle}: wake failed and the server restarted")
            report["cycles"].append({"cycle": cycle, "gpu_used_awake_gib": gib(used_awake),
                                     "gpu_used_asleep_gib": gib(used_asleep),
                                     "sleep_seconds": round(sleep_seconds, 1),
                                     "wake_seconds": round(wake_seconds, 1),
                                     "speech": speak(server, args.requests)})
        if args.compare_ipc:
            server.stop()
            server = SglangOmniServer(replace(settings, sleep_mode="off"))
            server.ensure_ready(lambda: None, lambda message: print(message, flush=True))
            report["stock_ipc_speech"] = speak(server, args.requests)
    except Exception as exc:  # report every failure with the log tail
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["log_tail"] = server.log_tail(40)
    finally:
        report["log"] = str(server.log_path)
        server.stop()
    takes = [take for phase in [report.get("awake_speech"), *(c["speech"] for c in report["cycles"])]
             if phase for take in phase["takes"]]
    report["all_takes_ok"] = bool(takes) and all(take["ok"] for take in takes)
    print(json.dumps(report, indent=2))
    return 1 if report.get("error") or len(report["cycles"]) != args.cycles or not report["all_takes_ok"] else 0


if __name__ == "__main__":
    sys.exit(main())
