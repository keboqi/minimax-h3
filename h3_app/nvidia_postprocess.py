"""Optional NVIDIA video enhancement services, independent of Comfy graphs."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Iterator

from h3_nvidia_runtime import (
    enabled, ensure_r580_core, reset_ngx_state, runtime_env,
    stop_wine_server, wine_env,
)

from .catalog import DLSS5_NR, NVIDIA_POSTPROCESS_OPTIONS, UPSCALE_RESOLUTION_PRESETS
from .config import RuntimeConfig
from .errors import H3Error
from .jobs import check_cancelled
from .media_tools import has_encoder
from .nvidia_pipeline import stream_pipeline
from .processes import run_media_process
from .provenance import read_snapshot, snapshot_path, write_snapshot


@dataclass(frozen=True)
class NvidiaSettings:
    original_size: bool = False
    style: str = "Natural"
    intensity: float = 1.0
    # "resize": Lanczos to the target, then DLSS 5 at 1× (works under Wine).
    # "dlss_sr": DLSS SR carrier; on Linux it uses the experimental R580 NGX core.
    upscaler: str = "resize"

    @classmethod
    def parse(cls, value=None):
        if value is not None and not isinstance(value, dict):
            raise H3Error("Invalid NVIDIA enhancement settings.")
        try:
            settings = cls(**(value or {}))
        except TypeError as exc:
            raise H3Error("Unknown NVIDIA enhancement setting.") from exc
        if settings.style not in ("Default", "Natural", "Cinematic"):
            raise H3Error("Invalid DLSS style.")
        val = settings.intensity
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or not 0 <= val <= 1:
            raise H3Error("DLSS intensity must be between 0 and 1.")
        if not isinstance(settings.original_size, bool):
            raise H3Error("Original size must be a boolean.")
        if settings.upscaler not in ("resize", "dlss_sr"):
            raise H3Error("Invalid DLSS upscaler.")
        return settings


@dataclass(frozen=True)
class NvidiaUpdate:
    message: str
    frames: int = 0
    total: int = 0
    result: Path | None = None


def file_hash(path):
    path = Path(path)
    stat = path.stat()
    return _file_hash(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=32)
def _file_hash(path, size, modified):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dlss_config(*, verify=False):
    config = runtime_env()
    if not enabled("H3_ENABLE_DLSSNR"):
        raise H3Error("DLSS enhancement is disabled on this processing host.")
    if config.get("H3_DLSSNR_SETUP_ERROR"):
        raise H3Error("Automatic DLSS setup failed: " + config["H3_DLSSNR_SETUP_ERROR"])
    configured = config.get("H3_DLSSNR_EXE", "")
    if not configured or not Path(configured).is_file():
        raise H3Error("DLSS runtime is missing. Restart run_h3.sh or redeploy Modal to install it automatically.")
    exe = Path(configured).expanduser().resolve()
    directory = Path(config.get("H3_DLSSNR_DLL_DIR", str(exe.parent))).expanduser().resolve()
    model = directory / "nvngx_dlssnr.dll"
    if not model.is_file():
        raise H3Error("Missing nvngx_dlssnr.dll in H3_DLSSNR_DLL_DIR (or beside the helper).")
    for required in (directory / "_nvngx.dll", directory / "nvngx_dlss.dll",
                     directory / "caller/nvngx.dll_comfy.dll", exe.parent / "dlss5nr_bridge.dll"):
        if not required.is_file():
            raise H3Error("Missing DLSS host component: " + str(required))
    if exe.name != "dlss5nr_host.exe":
        raise H3Error("The previous video2dlssnr helper is unsupported. Restart the launcher to install the RunningHub host.")
    expected = config.get("H3_DLSSNR_MODEL_SHA256", "").lower()
    if verify and expected and file_hash(model) != expected:
        raise H3Error("DLSS neural model SHA-256 does not match H3_DLSSNR_MODEL_SHA256.")
    wine = None
    if os.name != "nt":
        wine = shutil.which(config.get("H3_DLSSNR_WINE", "wine"))
        if not wine:
            raise H3Error("Install a compatible Wine runner and configure H3_DLSSNR_WINE.")
        if not (Path(wine).parent / "wineserver").is_file():
            raise H3Error("H3_DLSSNR_WINE needs a sibling wineserver for dedicated-prefix cleanup.")
        prefix = Path(config.get("H3_DLSSNR_WINEPREFIX", "")).expanduser()
        if not prefix.is_absolute() or not (prefix / "system.reg").is_file():
            raise H3Error("H3_DLSSNR_WINEPREFIX must be an initialized, dedicated absolute Wine prefix.")
        for name in ("dxgi.dll", "d3d11.dll", "d3d12.dll", "d3d12core.dll", "nvapi64.dll", "nvofapi64.dll"):
            if not (prefix / "drive_c/windows/system32" / name).is_file():
                raise H3Error("DLSS Wine graphics setup is incomplete: " + name + ". Restart the launcher to repair it.")
    return exe, directory, model, wine


def availability(method):
    """Cheap prerequisite check only; never initialize a GPU while browsing."""
    try:
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise H3Error("FFmpeg and ffprobe must be installed on the processing host.")
        if method == DLSS5_NR:
            dlss_config()
        else:
            return True, ""
        return True, "Runtime files found. Check runtime or enhance to verify GPU execution. SDR images and videos only."
    except (H3Error, OSError, ValueError, ImportError) as exc:
        return False, str(exc)


def target_dimensions(method, width, height, resolution, settings, *, image=False):
    if settings.original_size:
        w, h = width, height
    else:
        if resolution not in UPSCALE_RESOLUTION_PRESETS:
            raise H3Error("Choose a valid target resolution.")
        box_w, box_h = UPSCALE_RESOLUTION_PRESETS[resolution]
        scale = min(box_w / width, box_h / height)
        factors = [factor for factor in (1.0, 1.5, 1.724, 2.0, 3.0) if factor <= scale + 0.001]
        if not factors:
            raise H3Error("DLSS cannot downscale. Choose original size or a larger target.")
        factor = max(factors)
        w, h = (width, height) if factor == 1 else (round(width * factor / 2) * 2, round(height * factor / 2) * 2)
    if max(w, h) > 7680 or min(w, h) > 4320 or min(w, h) < 2:
        raise H3Error("DLSS output must fit the 7680 × 4320 envelope in either orientation.")
    if not image and (w % 2 or h % 2):
        raise H3Error("Video output dimensions must be even.")
    return w, h


def probe_source(source, runtime, *, count=False):
    cmd = ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json"]
    if count:
        cmd += ["-count_frames"]
    proc = run_media_process(cmd + [str(source)], timeout=runtime.generation_timeout)
    try:
        payload = json.loads(proc.stdout)
        video = next(s for s in payload["streams"] if s["codec_type"] == "video")
        rate = Fraction(video["avg_frame_rate"])
        if proc.returncode or rate <= 0 or int(video["width"]) < 2 or int(video["height"]) < 2:
            raise ValueError()
        return payload, video, rate
    except (KeyError, ValueError, StopIteration, ZeroDivisionError) as exc:
        raise H3Error("Could not inspect a usable video stream: " + proc.stderr[-2000:]) from exc


def validate_source(video):
    if abs(float(video.get("start_time", 0))) > 0.001:
        raise H3Error("Normalize the video's start timestamp to zero before NVIDIA processing.")
    if video.get("color_transfer") in {"smpte2084", "arib-std-b67", "linear"} or video.get("color_primaries") == "bt2020":
        raise H3Error("NVIDIA post-processing currently supports SDR only. HDR/linear sources are unsupported.")
    if video.get("tags", {}).get("rotate") or any(
        s.get("rotation", 0) != 0 for s in video.get("side_data_list", [])
    ):
        raise H3Error("Normalize the video's rotation before NVIDIA processing.")
    sar = video.get("sample_aspect_ratio", "1:1")
    if sar not in {"1:1", "0:1", "N/A"}:
        raise H3Error("Normalize non-square pixels before NVIDIA processing.")


def validate_cfr(source, runtime):
    # vfrdet scans timestamps in bounded memory. Reject timing irregularities
    # conservatively, including timestamp jitter, rather than flattening VFR.
    proc = run_media_process([
        "ffmpeg", "-hide_banner", "-nostdin", "-i", str(source), "-map", "0:v:0",
        "-vf", "vfrdet", "-an", "-f", "null", "-",
    ], timeout=runtime.generation_timeout)
    matches = re.findall(r"VFR:([\d.]+)", proc.stderr)
    if proc.returncode or not matches:
        raise H3Error("Could not verify source frame timing: " + proc.stderr[-2000:])
    if float(matches[-1]) != 0:
        raise H3Error("Variable or irregular frame timing is unsupported. Convert the video to constant frame rate first.")


@contextmanager
def worker_context(method, runtime):
    """Reserve a dedicated Wine prefix across app processes; no global kill."""
    if os.name == "nt":
        yield None
        return
    import fcntl
    config = runtime_env()
    prefix = Path(config["H3_DLSSNR_WINEPREFIX"]).expanduser().resolve()
    with (prefix / ".h3-dlssnr.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise H3Error("The dedicated DLSS Wine prefix is busy in another process.") from exc
        wine = shutil.which(config.get("H3_DLSSNR_WINE", "wine"))
        env = wine_env(prefix, wine)
        try:
            reset_ngx_state(prefix)
        except OSError as exc:
            raise H3Error("Could not reset NGX state in the dedicated DLSS Wine prefix.") from exc
        failed = False
        try:
            yield env
        except BaseException:
            failed = True
            raise
        finally:
            # wineserver can detach from the launched process group. Stop only
            # this operator-dedicated prefix, while still holding its lock.
            server = Path(wine).parent / "wineserver" if wine else None
            if server and server.is_file():
                try:
                    stop_wine_server(wine, env, timeout=10)
                except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
                    if not failed:
                        raise H3Error("Could not stop the dedicated DLSS Wine server.") from exc


def worker_command(method, width, height, out_width, out_height, settings, *, frames=1, motion=True,
                   ngx_core="default"):
    dlss_config(verify=True)
    # Preserve the virtualenv executable path; resolving its symlink loses its packages.
    command = [os.path.abspath(sys.executable), str(Path(__file__).with_name("dlss_worker.py")),
               "--width", str(width), "--height", str(height), "--out-width", str(out_width),
               "--out-height", str(out_height), "--frames", str(frames), "--motion", str(int(motion)),
               "--ngx-core", ngx_core, "--settings", json.dumps(asdict(settings))]
    return command, None


@dataclass(frozen=True)
class UpscalePlan:
    """What the DLSS worker sees: its input size and which NGX core it uses."""
    width: int
    height: int
    resize: bool = False
    ngx_core: str = "default"
    core_info: dict | None = None


def plan_upscale(settings, width, height, out_width, out_height):
    if (width, height) == (out_width, out_height):
        return UpscalePlan(width, height)
    if settings.upscaler == "resize":
        # Under Wine the DLSS SR carrier's signed load fails; enlarge first and
        # run the neural pass natively at the target size.
        return UpscalePlan(out_width, out_height, resize=True)
    if os.name == "nt":
        return UpscalePlan(width, height)
    try:
        info = ensure_r580_core()
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        raise H3Error("Experimental R580 NGX core setup failed: " + str(exc)) from exc
    return UpscalePlan(width, height, ngx_core="r580", core_info=info)


def probe_runtime(method, runtime):
    ready, message = availability(method)
    if not ready:
        raise H3Error(message)
    # A deterministic textured input. Native resolution avoids SR being mistaken
    # for neural rendering. Two frames test retained temporal state too.
    width, height = 640, 360
    channels = 4
    raw = bytearray(width * height * channels)
    for i in range(width * height):
        x, y = i % width, i // width
        raw[i * channels:i * channels + 3] = bytes((x * 7 % 256, y * 11 % 256, (x ^ y) % 256))
        raw[i * channels + 3] = 255
    settings = NvidiaSettings(original_size=True)
    command, cwd = worker_command(method, width, height, width, height, settings, frames=2)
    result = bytearray()
    last = None
    with worker_context(method, runtime) as env:
        for update in stream_pipeline([command], input_size=len(raw), output_size=len(raw),
                                 timeout=min(runtime.generation_timeout, 180), worker_env=env,
                                 worker_cwd=cwd, source_bytes=bytes(raw) * 2, capture=result):
            last = update
    changed = any(result[channel::4] != (raw * 2)[channel::4] for channel in range(3))
    if not changed:
        raise H3Error("Runtime returned unchanged frames; neural execution could not be verified.")
    # Ignore alpha: a failed GPU backend can return opaque, all-black frames.
    if not any(any(result[channel::channels]) for channel in range(3)):
        raise H3Error("Runtime returned black frames for textured input; neural execution failed.")
    if method == DLSS5_NR and not any("feature ready; streaming" in line for line in last.diagnostics):
        raise H3Error("DLSS helper did not confirm neural feature initialization. Restart the launcher to repair the RunningHub host.")
    return "Runtime check passed: two frames evaluated and enhanced on the processing host."


def _encoder():
    if has_encoder("hevc_nvenc"):
        return "hevc_nvenc", ["-c:v", "hevc_nvenc", "-preset", "p5", "-cq", "19", "-b:v", "0"]
    if has_encoder("libx265"):
        return "libx265", ["-c:v", "libx265", "-preset", "medium", "-crf", "19"]
    raise H3Error("No usable HEVC encoder. Install FFmpeg with hevc_nvenc or libx265.")


def runtime_details(method, diagnostics, *, upscaled, motion=True, plan=None):
    details = {}
    if method == DLSS5_NR:
        if not any("feature ready; streaming" in line for line in diagnostics):
            raise H3Error("DLSS helper did not confirm neural feature initialization for this media.")
        exe, _, model, wine = dlss_config()
        details.update(helper_sha256=file_hash(exe), model_sha256=file_hash(model),
                       backend="RunningHub DNR2", runner=Path(wine).name if wine else "Windows", motion="auto" if motion else "disabled",
                       stage_order="Lanczos → NR" if plan and plan.resize else "DLSS SR → NR" if upscaled else "NR")
        if plan and plan.core_info:
            details["ngx_core"] = {"experimental": "R580", **plan.core_info}
        for line in diagnostics:
            if "optical flow:" in line:
                details["motion_backend"] = line.split("optical flow:", 1)[1].strip()
    return details


def postprocess(source: Path, method: str, resolution: str, settings_value, *, runtime: RuntimeConfig) -> Iterator[NvidiaUpdate]:
    if method not in NVIDIA_POSTPROCESS_OPTIONS:
        raise H3Error("Unsupported NVIDIA enhancement method.")
    settings = NvidiaSettings.parse(settings_value)
    ready, message = availability(method)
    if not ready:
        raise H3Error(message)
    payload, video, rate = probe_source(source, runtime)
    validate_source(video)
    width, height = int(video["width"]), int(video["height"])
    out_width, out_height = target_dimensions(method, width, height, resolution, settings)
    total = int(video.get("nb_frames", "0")) if str(video.get("nb_frames", "0")).isdigit() else 0
    if not total:
        yield NvidiaUpdate("Counting source frames")
        _, counted, _ = probe_source(source, runtime, count=True)
        value = str(counted.get("nb_read_frames", "0"))
        total = int(value) if value.isdigit() else 0
    if not 1 <= total <= 1_000_000:
        raise H3Error("DLSS requires a known frame count from 1 to 1,000,000.")
    yield NvidiaUpdate("Checking source frame timing", total=total)
    validate_cfr(source, runtime)
    if method == DLSS5_NR:
        yield NvidiaUpdate("Checking DLSS 5 neural execution", total=total)
        probe_runtime(method, runtime)
    if settings.upscaler == "dlss_sr" and os.name != "nt" and (out_width, out_height) != (width, height):
        yield NvidiaUpdate("Preparing experimental R580 NGX core (one-time driver download)", total=total)
    plan = plan_upscale(settings, width, height, out_width, out_height)
    encoder, encoder_args = _encoder()
    category = "dlss5"
    root = runtime.outputs_dir / category
    root.mkdir(parents=True, exist_ok=True)
    channels = 4
    pixel = "rgba"
    audio = [s for s in payload["streams"] if s["codec_type"] == "audio"]
    copy_audio = all(s.get("codec_name") in {"aac", "mp3", "ac3", "eac3", "alac"} for s in audio)
    matrix = video.get("color_space", "bt709")
    if matrix not in {"bt709", "bt470bg", "smpte170m", "smpte240m", "fcc"}:
        matrix = "bt709"
    primaries = video.get("color_primaries", "bt709")
    transfer = video.get("color_transfer", "bt709")
    if primaries in {"unknown", "unspecified"}:
        primaries = "bt709"
    if transfer in {"unknown", "unspecified"}:
        transfer = "bt709"
    with tempfile.TemporaryDirectory(prefix=".processing-", dir=root) as directory:
        pending = Path(directory) / "output.mp4"
        output = root / (source.stem + "_" + Path(directory).name.removeprefix(".processing-") + ".mp4")
        resize = ["-vf", f"scale={out_width}:{out_height}:flags=lanczos"] if plan.resize else []
        decode = ["ffmpeg", "-v", "error", "-nostdin", "-noautorotate", "-i", str(source),
                  "-map", "0:v:0", "-an", "-fps_mode", "passthrough", *resize,
                  "-pix_fmt", pixel, "-f", "rawvideo", "pipe:1"]
        command, cwd = worker_command(method, plan.width, plan.height, out_width, out_height, settings,
                                      frames=total, ngx_core=plan.ngx_core)
        encode = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", pixel,
            "-video_size", f"{out_width}x{out_height}", "-framerate", str(rate), "-i", "pipe:0",
            "-i", str(source), "-map", "0:v:0", "-map", "1:a?", "-map_metadata", "1",
            *encoder_args, "-vf", f"scale=out_color_matrix={matrix}:out_range=tv", "-pix_fmt", "yuv420p",
            "-colorspace", matrix, "-color_primaries", primaries, "-color_trc", transfer,
            "-color_range", "tv", "-c:a", "copy" if copy_audio else "aac"]
        if not copy_audio:
            encode += ["-b:a", "192k"]
        encode += ["-movflags", "+faststart", str(pending)]
        last = None
        with worker_context(method, runtime) as env:
            for update in stream_pipeline([decode, command, encode],
                input_size=plan.width * plan.height * channels, output_size=out_width * out_height * channels,
                timeout=runtime.generation_timeout, worker_env=env, worker_cwd=cwd):
                last = update
                yield NvidiaUpdate(f"Processing {update.frames}" + (f"/{total}" if total else "") + f" frames · {encoder}", update.frames, total)
        yield NvidiaUpdate("Validating enhanced video", last.frames, total)
        result_payload, result_meta, result_rate = probe_source(pending, runtime, count=True)
        if (int(result_meta["width"]), int(result_meta["height"])) != (out_width, out_height) or result_rate != rate:
            raise H3Error("Enhanced video dimensions or frame rate do not match the request.")
        if int(result_meta.get("nb_read_frames", 0)) != last.frames or total and last.frames != total:
            raise H3Error("Enhanced video frame count does not match its source.")
        if len([s for s in result_payload["streams"] if s["codec_type"] == "audio"]) != len(audio):
            raise H3Error("Enhanced video did not preserve the source audio streams.")
        check_cancelled()
        gpu_details = runtime_details(method, last.diagnostics,
                                      upscaled=(out_width, out_height) != (width, height), plan=plan)
        details = {"method": method, **asdict(settings), "source_resolution": f"{width}x{height}",
            "target_resolution": f"{out_width}x{out_height}", "frame_rate": str(rate), "frames": last.frames,
            "encoder": encoder, "encoder_fallback": encoder != "hevc_nvenc",
            "audio": "copy" if copy_audio else "aac", **gpu_details}
        source_snapshot = read_snapshot(source) or {}
        published = False
        try:
            pending.replace(output)
            published = True
            write_snapshot(output, {"family": category, "settings": details,
                "source_asset_ids": [source_snapshot["asset_id"]] if source_snapshot.get("asset_id") else []})
        except BaseException:
            if published:
                output.unlink(missing_ok=True)
                snapshot_path(output).unlink(missing_ok=True)
            raise
        yield NvidiaUpdate("Enhancement complete", last.frames, total, output)
