"""RGBA pipe adapter for RunningHub's pinned DNR2 Linux/Windows host.

Protocol and motion handling follow ComfyUI-RH-DLSS5 (MIT, see docs/licenses).
Only the processing host launches this worker; importing the UI does not load it.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from h3_nvidia_runtime import (check_vulkan, display_reachable, ensure_display, pin_gpu, runtime_env,
                               r580_host_dir, runtime_root, wine_env, wine_run, xvfb_log_path)

HEADER = struct.Struct("<4s12I5f")
FRAME = struct.Struct("<4sII")
REPLY = struct.Struct("<4sIII")
SCALES = {1.0: 5, 1.5: 2, 1.724: 1, 2.0: 0, 3.0: 3}


def read_exact(stream, length, *, eof=False):
    result = bytearray()
    while len(result) < length:
        piece = stream.read(length - len(result))
        if not piece:
            if eof and not result:
                return None
            raise RuntimeError(f"DLSS stream ended after {len(result)} of {length} expected bytes.")
        result.extend(piece)
    return bytes(result)


def decode_reply(stream, index, count):
    magic, returned_index, ok, returned_count = REPLY.unpack(read_exact(stream, REPLY.size))
    if magic != b"OUT1" or returned_index != index:
        raise RuntimeError(f"Invalid DLSS frame reply: signature={magic!r}, index={returned_index}, expected={index}.")
    if not ok:
        length = struct.unpack("<I", read_exact(stream, 4))[0]
        if length > 65535:
            raise RuntimeError("DLSS host returned an oversized error reply.")
        raise RuntimeError(read_exact(stream, length).decode("utf-8", "replace"))
    if returned_count != count:
        raise RuntimeError(f"DLSS returned {returned_count} floats; expected {count}.")
    return read_exact(stream, count * 4)


def transform(args):
    import numpy as np
    from h3_app.dlss_motion import TemporalGuideGenerator

    config = runtime_env()
    runtime = Path(config["H3_DLSSNR_DLL_DIR"])
    exe = Path(config["H3_DLSSNR_EXE"])
    if args.ngx_core == "r580":
        # Same host files, prepared by ensure_r580_core with a different _nvngx.dll.
        runtime = r580_host_dir()
        exe = runtime / exe.name
        if not (runtime / "_nvngx.dll").is_file() or not exe.is_file():
            raise RuntimeError("The experimental R580 NGX core is not prepared.")
    settings = json.loads(args.settings)
    ratio = (args.out_width / args.width + args.out_height / args.height) / 2
    factor = min(SCALES, key=lambda value: abs(value - ratio))
    if abs(ratio - factor) > 0.03:
        raise RuntimeError("DLSS supports only native, 1.5×, 1.724×, 2× and 3× output sizes.")
    command = [str(exe), str(runtime)]
    diagnostics = None
    env = dict(os.environ)
    env.update(DLSS5NR_DISABLE_OTHER_SINKS="1", DLSS5NR_HDR="0", DLSS5NR_SNR_FILENAME="nvngx_dlssnr.dll")
    if os.name != "nt":
        wine = config["H3_DLSSNR_WINE"]
        env = ensure_display(wine_env(config["H3_DLSSNR_WINEPREFIX"], wine))
        pin_gpu(env)
        check_vulkan(env)
        command = [wine, str(exe), "Z:" + str(runtime).replace("/", "\\")]
        # NGX resolves its directory via the adapter's NGXCore\NGXPath query,
        # which Wine cannot answer (0xD0000002), then via this key. Proton's
        # wine.inf sets it; upstream Wine does not, and Init fails with 0xBAD00002.
        wine_run(wine, ["reg", "add", r"HKLM\Software\NVIDIA Corporation\Global\NGXCore",
                        "/v", "FullPath", "/t", "REG_SZ", "/d", "Z:" + str(runtime).replace("/", "\\"), "/f"],
                 env)
        # Graphics-layer logs go to files (never the binary stdout) and are
        # attached to failures; otherwise device errors arrive without a cause.
        diagnostics = tempfile.TemporaryDirectory(prefix="h3-dlss-log-")
        env.update(VKD3D_DEBUG="warn", VKD3D_LOG_FILE=os.path.join(diagnostics.name, "vkd3d.log"),
                   DXVK_LOG_LEVEL="warn", DXVK_LOG_PATH=diagnostics.name,
                   DXVK_NVAPI_LOG_LEVEL="info", DXVK_NVAPI_LOG_PATH=diagnostics.name,
                   WINEDEBUG="err+all,fixme-all")
    logs = deque(maxlen=5000)  # Full NGX verbose trail for the saved report.
    # Keep the child in our process group so pipeline cancellation kills it too.
    process = subprocess.Popen(command, cwd=exe.parent, env=env, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def drain():
        pending = b""
        while True:
            chunk = process.stderr.read1(1024)
            if not chunk:
                break
            pending += chunk
            while b"\n" in pending or len(pending) >= 4096:
                if b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                else:
                    line, pending = pending[:4096], pending[4096:]
                text = line.decode("utf-8", "replace")
                logs.append(text)
                print(text, file=sys.stderr, flush=True)
        if pending:
            text = pending.decode("utf-8", "replace")
            logs.append(text)
            print(text, file=sys.stderr, flush=True)

    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    guide = TemporalGuideGenerator(args.width, args.height)
    count = args.out_width * args.out_height * 3
    try:
        process.stdin.write(HEADER.pack(
            b"DNR2", args.width, args.height, args.out_width, args.out_height,
            0, args.frames, SCALES[factor], 0, 0,
            ("Default", "Natural", "Cinematic").index(settings["style"]), 1, 0,
            settings["intensity"], 1.0, 1.0, 1.0, 1.0,
        ))
        process.stdin.flush()
        for index in range(args.frames):
            raw = read_exact(sys.stdin.buffer, args.width * args.height * 4)
            rgb = np.frombuffer(raw, dtype=np.uint8).reshape(args.height, args.width, 4)[..., :3]
            if args.motion:
                motion, reset = guide.process(rgb)
            else:
                motion, reset = np.zeros((args.height, args.width, 2), dtype=np.float16), True
            process.stdin.write(FRAME.pack(b"FRM2", index, int(reset)))
            process.stdin.write((np.ascontiguousarray(rgb, dtype="<f4") / np.float32(255.0)).tobytes())
            process.stdin.write(np.ascontiguousarray(motion, dtype="<f2").tobytes())
            process.stdin.flush()
            result = np.frombuffer(decode_reply(process.stdout, index, count), dtype="<f4")
            if not np.isfinite(result).all():
                raise RuntimeError("DLSS returned non-finite pixels.")
            if index == 0:
                print("feature ready; streaming (RunningHub DNR2 frame evaluated)", file=sys.stderr, flush=True)
                print("optical flow: " + ("OpenCV DIS" if args.motion and guide.available else "disabled; per-frame reset"),
                      file=sys.stderr, flush=True)
            rgba = np.empty((args.out_height, args.out_width, 4), dtype=np.uint8)
            rgba[..., :3] = np.clip(result.reshape(args.out_height, args.out_width, 3) * 255 + 0.5, 0, 255).astype(np.uint8)
            rgba[..., 3] = 255
            sys.stdout.buffer.write(rgba.tobytes())
            sys.stdout.buffer.flush()
        if read_exact(sys.stdin.buffer, 1, eof=True) is not None:
            raise RuntimeError("Source contains more frames than its declared frame count.")
        process.stdin.close()
        if read_exact(process.stdout, 4) != b"END1":
            raise RuntimeError("DLSS host did not finish its frame stream.")
        # Upstream documents a model teardown hang after successful END1.
        try:
            if process.wait(timeout=3):
                raise RuntimeError("DLSS host exited unsuccessfully after END1.")
        except subprocess.TimeoutExpired:
            pass
    except Exception as exc:
        # Stop the host first so its stderr and log files are complete.
        stop_host(process)
        thread.join(timeout=1)
        raise RuntimeError(str(exc) + "\nDLSS engine log:\n" + "\n".join(list(logs)[-40:])
                           + graphics_logs(diagnostics, env, logs)) from exc
    finally:
        stop_host(process)
        thread.join(timeout=1)
        if diagnostics is not None:
            diagnostics.cleanup()


def stop_host(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


def graphics_logs(diagnostics, env, engine_log=()):
    if diagnostics is None:
        return ""
    display = env.get("DISPLAY", "")
    header = f"\nDISPLAY={display or '<unset>'} reachable={bool(display) and display_reachable(display)}"
    full, brief = [header], [header]
    for path in [*sorted(Path(diagnostics.name).iterdir()), xvfb_log_path()]:
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if text:
            full.append(f"--- {path.name} ---\n" + text[-20000:])
            brief.append(f"--- {path.name} (tail) ---\n" + text[-1200:])
    # The UI keeps only the error's tail; keep the whole report on the host.
    saved = runtime_root() / "logs" / "dlss-last-failure.log"
    try:
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_text("DLSS engine log:\n" + "\n".join(engine_log) + "\n".join(full) + "\n", encoding="utf-8")
        brief.append(f"Full report: {saved}")
    except OSError:
        pass
    return "\n".join(brief)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("width", "height", "out-width", "out-height", "frames"):
        parser.add_argument("--" + name, type=int, required=True)
    parser.add_argument("--motion", type=int, choices=(0, 1), default=1)
    parser.add_argument("--ngx-core", choices=("default", "r580"), default="default")
    parser.add_argument("--settings", required=True)
    args = parser.parse_args()
    if not 1 <= args.frames <= 1_000_000:
        raise RuntimeError("DLSS requires a known frame count from 1 to 1,000,000.")
    transform(args)


if __name__ == "__main__":
    main()
