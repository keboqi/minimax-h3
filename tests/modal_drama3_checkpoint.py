"""Manual Modal sleep/wake probe: three CUDA workers, graphs and IPC buffers.

Run explicitly with `modal run tests/modal_drama3_checkpoint.py`. Allocates a
temporary RTX PRO 6000; does not load models, mount volumes or deploy the app.
"""
from pathlib import Path
import modal

app = modal.App("h3-cuda-sleep-probe")
image = (modal.Image.debian_slim(python_version="3.12").pip_install("requests")
         .add_local_file(str(Path(__file__).resolve().parents[1] / "h3_app/drama3/serving/checkpoint.py"),
                         "/root/h3_checkpoint.py"))


@app.function(image=image, gpu="RTX-PRO-6000", cpu=1, memory=1024, timeout=180)
def probe():
    import json
    import os
    import signal
    import subprocess
    import sys
    import time
    import h3_checkpoint as cp

    child = '''
import ctypes as c, json, os, sys
lib = c.CDLL("libcuda.so.1")
def check(code):
    if code: raise RuntimeError("CUDA error " + str(code))
check(lib.cuInit(0))
device = c.c_int()
check(lib.cuDeviceGet(c.byref(device), 0))
ctx = c.c_void_p()
check(lib.cuCtxCreate_v2(c.byref(ctx), 0, device))
ptr = c.c_uint64()
size = c.c_size_t(16 * 1024 * 1024)
check(lib.cuMemAlloc_v2(c.byref(ptr), size))
check(lib.cuMemsetD8_v2(ptr, 37, size))
check(lib.cuCtxSynchronize())
stream, graph, executable = c.c_void_p(), c.c_void_p(), c.c_void_p()
check(lib.cuStreamCreate(c.byref(stream), 1))
check(lib.cuStreamBeginCapture(stream, 0))
check(lib.cuMemsetD8Async(ptr, 41, size, stream))
check(lib.cuStreamEndCapture(stream, c.byref(graph)))
check(lib.cuGraphInstantiateWithFlags(c.byref(executable), graph, c.c_uint64(0)))
class Handle(c.Structure):
    _fields_ = [("bytes", c.c_ubyte * 64)]
handle = Handle()
check(lib.cuIpcGetMemHandle(c.byref(handle), ptr))
print(json.dumps({"pid": os.getpid(), "handle": bytes(handle).hex()}), flush=True)
peers = []
for line in sys.stdin:
    command = json.loads(line)
    if "handles" in command:
        for item in command["handles"]:
            if item["pid"] == os.getpid(): continue
            peer = c.c_uint64()
            handle = Handle.from_buffer_copy(bytes.fromhex(item["handle"]))
            check(lib.cuIpcOpenMemHandle_v2(c.byref(peer), handle, 1))
            peers.append(peer)
    if command.get("graph"):
        check(lib.cuGraphLaunch(executable, stream))
        check(lib.cuStreamSynchronize(stream))
    value = c.c_ubyte()
    check(lib.cuMemcpyDtoH_v2(c.byref(value), ptr, c.c_size_t(1)))
    values = []
    for peer in peers:
        peer_value = c.c_ubyte()
        check(lib.cuMemcpyDtoH_v2(c.byref(peer_value), peer, c.c_size_t(1)))
        values.append(peer_value.value)
    print(json.dumps({"pid": os.getpid(), "value": value.value, "peers": values}), flush=True)
'''
    root = f'''
import json, subprocess, sys
children = [subprocess.Popen([sys.executable, "-u", "-c", {child!r}], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True) for _ in range(3)]
workers = [json.loads(p.stdout.readline()) for p in children]
for p in children:
    p.stdin.write(json.dumps({{"handles": workers}}) + "\\n"); p.stdin.flush()
print(json.dumps([json.loads(p.stdout.readline()) for p in children]), flush=True)
for line in sys.stdin:
    for p in children:
        p.stdin.write(line); p.stdin.flush()
    print(json.dumps([json.loads(p.stdout.readline()) for p in children]), flush=True)
'''
    helper = cp.prepare_helper(Path("/tmp/h3-sleep-probe"))
    baseline = cp.device_memory()[0]
    process = subprocess.Popen([str(helper), "--launch-job", sys.executable, "-u", "-c", root],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                               start_new_session=True)
    report = {}
    try:
        workers = json.loads(process.stdout.readline())
        report["workers"] = workers
        report["nvml_processes"] = cp.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader"])
        report["group"] = list(cp.group_processes(process.pid))
        report["device_handles"] = {}
        for item in workers:
            pid = item["pid"]
            targets = []
            for fd in Path(f"/proc/{pid}/fd").iterdir():
                try:
                    target = os.readlink(fd)
                except FileNotFoundError:
                    continue
                if "nvidia" in target:
                    targets.append(target)
            report["device_handles"][pid] = sorted(set(targets))
        report["available_ram_gib"] = cp.available_ram() / cp.GIB
        report["cycles"] = []
        for cycle in range(3):
            controller = cp.CudaCheckpoint(helper, process.pid, baseline_used=baseline)
            started = time.monotonic()
            controller.sleep()
            asleep = cp.device_memory()[0]
            sleep_seconds = time.monotonic() - started
            started = time.monotonic()
            controller.wake()
            wake_seconds = time.monotonic() - started
            # Replay every captured graph, then verify every imported IPC buffer.
            process.stdin.write(json.dumps({"graph": True}) + "\n"); process.stdin.flush()
            json.loads(process.stdout.readline())
            process.stdin.write("{}\n"); process.stdin.flush()
            values = json.loads(process.stdout.readline())
            assert all(item["value"] == 41 and item["peers"] == [41, 41] for item in values), values
            report["cycles"].append({"cycle": cycle + 1, "asleep_used_gib": asleep / cp.GIB,
                                     "sleep_seconds": sleep_seconds, "wake_seconds": wake_seconds,
                                     "restored_values_and_ipc": values})
    except Exception as exc:
        report["error"] = str(exc)
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)
    return report


@app.local_entrypoint()
def main():
    import json
    report = probe.remote()
    print(json.dumps(report, indent=2))
    if report.get("error") or len(report.get("cycles", [])) != 3:
        raise RuntimeError("CUDA sleep/wake probe failed; inspect the report above")
