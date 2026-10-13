"""Shared, GPU-free provisioning for standalone, Colab and Modal enhancement.

Only startup prepares Wine and the NGX runtime. UI discovery reads the manifest;
it never downloads software, starts Wine, or initializes a GPU.
"""
from __future__ import annotations

import argparse
import atexit
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import threading
import tarfile
import tempfile
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import zipfile

from h3_sources import (
    DLSSNR_SHA256, DLSSNR_URL, DLSSNR_VERSION,
    RH_DLSS_REF, RH_DLSS_FILES, DLSS_GRAPHICS, NGX_WINDOWS_URL, NGX_WINDOWS_SHA256, NVAPI_HEADLESS_REF,
    WINE_COMPAT_SHA256, WINE_COMPAT_URL, WINE_COMPAT_VERSION,
)

LINUX_PACKAGES = (
    "libvulkan1", "libgl1", "libegl1", "libx11-6", "libxext6", "libxrender1",
    "libxrandr2", "libxi6", "libxfixes3", "libxcursor1", "libxcomposite1",
    "libxinerama1", "libxxf86vm1", "libfreetype6",
    "libfontconfig1", "libdbus-1-3", "libsm6", "libice6", "libusb-1.0-0",
    "libudev1", "libunwind8", "xvfb", "xauth", "dpkg", "vulkan-tools", "zstd", "p7zip-full",
    "git", "meson", "ninja-build", "g++-mingw-w64-x86-64-posix",
)
HELPER_FILES = ("video2dlssnr.exe", "nvngx.dll_dlssnr.dll", "nvngx_dlss.dll", "nvngx_dlssnr.dll")


def runtime_root(root=None):
    return Path(root or os.getenv("H3_NVIDIA_RUNTIME_DIR") or Path(__file__).resolve().parent / "h3/nvidia").expanduser().resolve()


def enabled(name):
    return os.getenv(name, "1") != "0"


def read_manifest(root):
    try:
        return json.loads((Path(root) / "runtime.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_manifest(root, data):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, delete=False) as handle:
        json.dump(data, handle, indent=2)
        temporary = Path(handle.name)
    os.replace(temporary, root / "runtime.json")


def runtime_env(root=None):
    """Return managed defaults, preserving explicit operator overrides."""
    root = runtime_root(root)
    manifest = read_manifest(root)
    keys = ("H3_DLSSNR_EXE", "H3_DLSSNR_DLL_DIR", "H3_DLSSNR_MODEL_SHA256",
            "H3_DLSSNR_WINE", "H3_DLSSNR_WINEPREFIX")
    result = {key: str(manifest["env"][key]) for key in keys if key in manifest.get("env", {})}
    # A custom helper is a complete independent configuration. Don't combine it
    # with model/runner/prefix paths belonging to our managed helper.
    if os.getenv("H3_DLSSNR_EXE"):
        result.clear()
    result.update({key: os.environ[key] for key in keys if key in os.environ})
    if manifest.get("error") and not os.getenv("H3_DLSSNR_EXE"):
        result["H3_DLSSNR_SETUP_ERROR"] = str(manifest["error"])
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, destination, expected=None):
    """Verify a complete download before publishing or extracting it."""
    destination = Path(destination)
    if destination.is_file() and expected and sha256(destination) == expected:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        partial = destination.with_suffix(destination.suffix + ".partial")
        try:
            request = Request(url, headers={"User-Agent": "MiniMax-H3-runtime-setup"})
            with urlopen(request, timeout=60) as response, partial.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
            if expected and sha256(partial) != expected:
                raise ValueError(f"Runtime archive checksum mismatch: {url}")
            os.replace(partial, destination)
            return destination
        except (OSError, ValueError) as exc:
            partial.unlink(missing_ok=True)
            if attempt == 2 or isinstance(exc, HTTPError) and exc.code == 404:
                if isinstance(exc, HTTPError):
                    raise RuntimeError(f"Download failed for {url}: {exc}") from exc
                raise
            time.sleep(1 + attempt)


def install_archive(root, name, url, checksum, required, *, helper=False):
    target = root / name
    marker = target / ".h3-install.json"
    try:
        installed = json.loads(marker.read_text(encoding="utf-8"))
        if installed["sha256"] == checksum and all(
            (target / path).is_file() and (target / path).stat().st_size == size
            for path, size in installed["files"].items()
        ) and all((target / path).is_file() for path in required):
            return target
    except (OSError, ValueError, KeyError):
        pass
    print(f"[h3-nvidia] Installing {name}", flush=True)
    archive = download(url, root / "downloads" / url.rsplit("/", 1)[-1], checksum)
    with tempfile.TemporaryDirectory(prefix=".install-", dir=root) as directory:
        staging = Path(directory)
        if helper:
            with zipfile.ZipFile(archive) as bundle:
                # Upstream builds this ZIP on Windows and stores backslashes.
                # ZipInfo normalizes them only on Windows, so use the entry
                # objects after normalizing names explicitly on every platform.
                entries = {}
                for entry in bundle.infolist():
                    normalized = entry.filename.replace("\\", "/")
                    if normalized in entries:
                        raise RuntimeError(f"Runtime archive contains duplicate entry {normalized}")
                    entries[normalized] = entry
                selected = ("README.md", *("out/" + filename for filename in HELPER_FILES))
                for filename in selected:
                    if filename not in entries:
                        raise RuntimeError(f"Runtime archive is missing {filename}")
                for filename in HELPER_FILES:
                    with bundle.open(entries["out/" + filename]) as source, (staging / filename).open("wb") as dest:
                        shutil.copyfileobj(source, dest)
                (staging / "README.md").write_bytes(bundle.read(entries["README.md"]))
            content = staging
        else:
            with tarfile.open(archive) as bundle:
                # Python 3.12's data filter rejects paths and links escaping staging.
                bundle.extractall(staging, filter="data")
            content = staging / name
        for path in required:
            if not (content / path).is_file():
                raise RuntimeError(f"Runtime archive is missing {path}")
        data = {"sha256": checksum, "url": url,
                "files": {path: (content / path).stat().st_size for path in required}}
        (content / ".h3-install.json").write_text(json.dumps(data), encoding="utf-8")
        if target.exists():
            # This directory is a versioned installer-owned child of root.
            if target.is_symlink() or target.resolve().parent != root.resolve():
                raise RuntimeError("Refusing to replace a runtime directory outside the installation root.")
            shutil.rmtree(target)
        if content == staging:
            target.mkdir()
            for path in staging.iterdir():
                shutil.move(str(path), target / path.name)
        else:
            shutil.move(str(content), target)
    archive.unlink(missing_ok=True)
    return target


def install_linux_packages(root):
    if (root / ".linux-packages-v3").is_file() and shutil.which("Xvfb") and shutil.which("vulkaninfo") and shutil.which("7z"):
        return
    command = ["apt-get"]
    if os.geteuid() != 0:
        command.insert(0, "sudo")
    if not shutil.which(command[0]):
        raise RuntimeError("Automatic DLSS Linux setup requires apt-get and root or sudo access.")
    env = dict(os.environ, DEBIAN_FRONTEND="noninteractive")
    subprocess.run(command + ["update"], check=True, env=env, timeout=600)
    subprocess.run(command + ["install", "-y", "--no-install-recommends", *LINUX_PACKAGES],
                   check=True, env=env, timeout=600)
    (root / ".linux-packages-v3").touch()


def stop_wine_server(wine, env, *, timeout=15):
    result = subprocess.run([str(Path(wine).parent / "wineserver"), "-k"], env=env,
                            check=False, timeout=timeout, capture_output=True, text=True)
    detail = (result.stderr or result.stdout or "").strip()
    # Wine can return 1 when the prefix's server has already exited. No shared
    # server is started or killed to work around that normal cleanup condition.
    absent = not detail or "not running" in detail.lower() or "no server" in detail.lower()
    if result.returncode not in (0, 1) or (result.returncode == 1 and not absent):
        raise RuntimeError(f"DLSS Wine cleanup failed (exit {result.returncode}): {detail[-2000:]}")


def copy_dll(source, destination):
    # Never write through a prefix symlink into a runner's built-in libraries.
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    shutil.copy2(source, destination)


@contextmanager
def installation_lock(root):
    with (root / ".install.lock").open("a+b") as lock:
        if os.name != "nt":
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield


_display_lock = threading.Lock()
_display_process = None
_display_name = None


def close_display():
    global _display_process
    if _display_process is not None and _display_process.poll() is None:
        _display_process.terminate()
        try:
            _display_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _display_process.kill()
    _display_process = None


atexit.register(close_display)


def display_reachable(name):
    """Connect to a local X display's socket (abstract or filesystem)."""
    import socket
    match = re.fullmatch(r"(?:unix)?:(\d+)(?:\.\d+)?", name or "")
    if not match:
        return bool(name)  # Remote/TCP displays are the operator's responsibility.
    path = "/tmp/.X11-unix/X" + match.group(1)
    for address in (chr(0) + path, path):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            probe.settimeout(2)
            try:
                probe.connect(address)
                return True
            except OSError:
                continue
    return False


def xvfb_log_path():
    return runtime_root() / "logs" / "xvfb.log"


def read_display_number(reader, process, timeout=10):
    """Read Xvfb's -displayfd reply through its newline.

    Xvfb writes the number and the newline separately and treats a failed
    write as fatal, so closing the pipe after the digits kills the server.
    """
    import select
    data = b""
    deadline = time.monotonic() + timeout
    while b"\n" not in data:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([reader], [], [], remaining)[0]:
            process.terminate()
            process.wait(timeout=5)
            raise RuntimeError("Xvfb did not become ready for DLSS.")
        chunk = os.read(reader, 32)
        if not chunk:
            break
        data += chunk
    return data.decode(errors="replace").strip()


def ensure_display(env):
    """Keep Xvfb separate from the host's binary stdout; never use xvfb-run."""
    global _display_process, _display_name
    if os.name == "nt" or display_reachable(env.get("DISPLAY")):
        return env
    with _display_lock:
        if _display_process is None or _display_process.poll() is not None or not display_reachable(_display_name):
            close_display()
            if not shutil.which("Xvfb"):
                raise RuntimeError("Headless DLSS requires Xvfb. Restart the launcher to provision it.")
            reader, writer = os.pipe()
            log = xvfb_log_path()
            try:
                log.parent.mkdir(parents=True, exist_ok=True)
                # -noreset keeps server state stable when the reachability probe
                # disconnects before Wine connects.
                with log.open("wb") as output:
                    process = subprocess.Popen(
                        ["Xvfb", "-displayfd", str(writer), "-screen", "0", "1280x720x24",
                         "-nolisten", "tcp", "-noreset"],
                        pass_fds=(writer,), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                        stderr=output,
                    )
                os.close(writer)
                writer = None
                number = read_display_number(reader, process)
                if not number.isdigit() or process.poll() is not None or not display_reachable(":" + number):
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=5)
                    raise RuntimeError("Xvfb could not provide a reachable headless display.")
                _display_process, _display_name = process, ":" + number
            finally:
                os.close(reader)
                if writer is not None:
                    os.close(writer)
        env["DISPLAY"] = _display_name
    return env


VULKAN_ICD_DIRS = ("/etc/vulkan/icd.d", "/usr/share/vulkan/icd.d", "/usr/local/share/vulkan/icd.d")
DRIVER_LIBRARY_DIRS = ("/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib", "/usr/local/nvidia/lib64",
                       "/usr/local/nvidia/lib", "/usr/lib/wsl/lib")


def nvidia_icd_manifests():
    found = []
    for directory in VULKAN_ICD_DIRS:
        for path in sorted(Path(directory).glob("*.json")) if Path(directory).is_dir() else ():
            try:
                library = json.loads(path.read_text(encoding="utf-8"))["ICD"]["library_path"]
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if "nvidia" in str(library).lower():
                found.append(path)
    return found


def ldconfig_listing():
    ldconfig = shutil.which("ldconfig") or ("/sbin/ldconfig" if Path("/sbin/ldconfig").is_file() else None)
    if not ldconfig:
        return ""
    try:
        return subprocess.run([ldconfig, "-p"], capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def find_driver_library(name, listing=None):
    """Return (path, in_loader_cache) for a driver library, or (None, False)."""
    for line in (ldconfig_listing() if listing is None else listing).splitlines():
        if line.strip().startswith(name) and "x86-64" in line and "=>" in line:
            return Path(line.split("=>", 1)[1].strip()), True
    directories = [*os.environ.get("LD_LIBRARY_PATH", "").split(":"), *DRIVER_LIBRARY_DIRS]
    for directory in filter(None, directories):
        matches = sorted(Path(directory).glob(name + "*")) if Path(directory).is_dir() else []
        if matches:
            return matches[0], False
    return None, False


def vulkan_driver_env(env, root=None):
    """Register the NVIDIA Vulkan ICD when the container mounted its library but not its manifest.

    Several providers inject libGLX_nvidia without /etc/vulkan/icd.d/nvidia_icd.json,
    which the loader reports as "Found no drivers". The driver library itself is never
    supplied by us: if it is absent, the container lacks graphics capabilities.
    """
    if platform.system() == "Windows" or env.get("VK_ICD_FILENAMES") or env.get("VK_DRIVER_FILES") or nvidia_icd_manifests():
        return env
    library, cached = find_driver_library("libGLX_nvidia.so.0")
    if library is None:
        return env
    manifest = runtime_root(root) / "vulkan" / "nvidia_icd.json"
    content = json.dumps({"file_format_version": "1.0.1",
                          "ICD": {"library_path": str(library), "api_version": "1.3.0"}}, indent=2)
    try:
        if not manifest.is_file() or manifest.read_text(encoding="utf-8") != content:
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(content, encoding="utf-8")
    except OSError:
        return env
    env["VK_DRIVER_FILES"] = env["VK_ICD_FILENAMES"] = str(manifest)
    # Driver dependencies (libnvidia-glcore, libnvidia-rtcore...) sit beside the ICD.
    # Expose only that directory, never the application's CUDA library paths.
    if not cached:
        env["LD_LIBRARY_PATH"] = str(library.parent)
    return env


def vulkan_diagnostics(env):
    """Summarize what the container exposes so operators can fix provider configuration."""
    if platform.system() == "Windows":
        return ""
    manifests = nvidia_icd_manifests()
    details = [
        "NVIDIA_DRIVER_CAPABILITIES=" + (env.get("NVIDIA_DRIVER_CAPABILITIES") or "<unset>"),
        "NVIDIA ICD manifests: " + (", ".join(map(str, manifests)) or "none"),
        "generated ICD: " + (env.get("VK_DRIVER_FILES") or "none"),
    ]
    listing = ldconfig_listing()
    for name in ("libGLX_nvidia.so.0", "libnvidia-glcore.so", "libnvidia-rtcore.so", "libnvidia-glvkspirv.so"):
        details.append(f"{name}: {find_driver_library(name, listing)[0] or 'missing'}")
    return "Host diagnostics: " + "; ".join(details) + ". "


def wine_env(prefix, wine):
    env = dict(os.environ)
    # CUDA libraries injected by the application can break Wine NVIDIA shims.
    env.pop("LD_LIBRARY_PATH", None)
    vulkan_driver_env(env)
    env.update(WINEPREFIX=str(prefix), WINEDEBUG="-all", WINEARCH="win64",
               WINEDLLOVERRIDES="d3d12,d3d12core,dxgi,d3d11,nvapi64,nvofapi64=n,b;winemenubuilder.exe=d",
               DXVK_ENABLE_NVAPI="1", DXVK_NVAPI_DRS_NGX_DLSS_NR_OVERRIDE="on",
               DXVK_LOG_LEVEL="none", VKD3D_DEBUG="none", DLSS5NR_DISABLE_OTHER_SINKS="1",
               DLSS5NR_HDR="0", DLSS5NR_SNR_FILENAME="nvngx_dlssnr.dll")
    return env


def pin_gpu(env):
    """DXGI ignores CUDA_VISIBLE_DEVICES; select the same physical GPU by UUID."""
    visible = env.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    explicit = env.get("H3_DLSSNR_GPU_UUID", "").strip()
    if not visible and not explicit:
        env.setdefault("DLSS5NR_GPU_INDEX", "0")
        return
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                            capture_output=True, text=True, check=True, timeout=15)
    table = dict(line.strip().split(",", 1) for line in result.stdout.splitlines() if "," in line)
    selected = explicit or (visible if visible.startswith("GPU-") else table.get(visible, "").strip())
    uid = selected.removeprefix("GPU-").replace("-", "").lower()
    if not re.fullmatch(r"[a-f0-9]{32}", uid):
        raise RuntimeError("Cannot map the assigned CUDA GPU to Vulkan. Set H3_DLSSNR_GPU_UUID to its GPU UUID.")
    env.update(DXVK_FILTER_DEVICE_UUID=uid, DLSS5NR_GPU_INDEX="0")


def check_vulkan(env):
    """Check the real driver, not merely that libvulkan1 or CUDA is installed."""
    # Surface support is irrelevant to offscreen DLSS; an X hiccup must not mask the GPU.
    env = {key: value for key, value in env.items() if key not in ("DISPLAY", "WAYLAND_DISPLAY")}
    result = subprocess.run(["vulkaninfo", "--summary"], env=env, capture_output=True,
                            text=True, timeout=30)
    if result.returncode or not re.search(r"deviceName\s*=.*NVIDIA", result.stdout, re.I):
        raise RuntimeError("DLSS requires a working NVIDIA Vulkan device inside this processing host. "
                           "Expose graphics driver capabilities when creating the container, including "
                           "the NVIDIA ICD and matching driver libraries (including libnvidia-rtcore), "
                           "e.g. NVIDIA_DRIVER_CAPABILITIES=all or --gpus 'all,\"capabilities=compute,utility,graphics\"'. "
                           + vulkan_diagnostics(env) + (result.stderr or result.stdout)[-2000:])
    return result.stdout[-4000:]


def wine_run(wine, args, env):
    env = dict(ensure_display(env))
    # Mono/Gecko install prompts are invisible dialogs under Xvfb and block wineboot.
    env["WINEDLLOVERRIDES"] = env.get("WINEDLLOVERRIDES", "") + ";mscoree,mshtml="
    # wineserver and Wine's services outlive wineboot and inherit its handles, so
    # pipes would never reach EOF. Use a file, which needs no EOF to be read.
    with tempfile.TemporaryFile() as log:
        try:
            result = subprocess.run([str(wine), *args], env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=300)
        except subprocess.TimeoutExpired as exc:
            log.seek(0)
            raise RuntimeError(f"DLSS Wine setup timed out running {args[0]}: "
                               + log.read().decode("utf-8", "replace")[-2000:]) from exc
        log.seek(0)
        output = log.read().decode("utf-8", "replace")
    if result.returncode:
        raise RuntimeError(f"DLSS Wine setup failed running {args[0]}: " + output[-2000:])


def install_rh_host(root):
    target = root / ("rh-dlss-" + RH_DLSS_REF)
    for relative, expected in RH_DLSS_FILES.items():
        name = "caller/" + Path(relative).name if "/caller/" in relative else Path(relative).name
        path = target / name
        def matches():
            if not path.is_file():
                return False
            data = path.read_bytes()
            return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == expected
        if not matches():
            url = f"https://raw.githubusercontent.com/RH-RunningHub/ComfyUI-RH-DLSS5/{RH_DLSS_REF}/{relative}"
            download(url, path)
            if not matches():
                path.unlink(missing_ok=True)
                raise RuntimeError("Upstream DLSS host blob verification failed: " + relative)
    return target


def install_graphics(root):
    """Pin the independently versioned graphics stack used by the Linux reference."""
    dlls = {}
    selections = (("dxgi.dll", "d3d11.dll"), ("d3d12.dll", "d3d12core.dll"), ("nvapi64.dll", "nvofapi64.dll"))
    for (name, url, checksum, top), files in zip(DLSS_GRAPHICS, selections):
        target = root / name
        paths = {file: target / top / "x64" / file for file in files}
        marker = target / ".h3-graphics.json"
        try:
            cached = json.loads(marker.read_text())
            valid = cached["archive"] == checksum and all(sha256(p) == cached["files"][n] for n, p in paths.items())
        except (OSError, ValueError, KeyError):
            valid = False
        if not valid:
            archive = download(url, root / "downloads" / url.rsplit("/", 1)[-1], checksum)
            target.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=root) as directory:
                source = archive
                if url.endswith(".zst"):
                    source = Path(directory) / "graphics.tar"
                    with source.open("wb") as output:
                        subprocess.run(["zstd", "-dc", str(archive)], stdout=output, check=True, timeout=180)
                with tarfile.open(source) as bundle:
                    bundle.extractall(target, filter="data")
            if not all(p.is_file() for p in paths.values()):
                raise RuntimeError("Graphics archive is missing required x64 DLLs: " + name)
            marker.write_text(json.dumps({"archive": checksum, "files": {n: sha256(p) for n, p in paths.items()}}))
            archive.unlink(missing_ok=True)
        dlls.update(paths)
    return dlls


def install_windows_core(root, target):
    """Extract the Windows NGX core as data; never install or execute the driver."""
    core = target / "_nvngx.dll"
    marker = target / ".h3-core.json"
    try:
        cached = json.loads(marker.read_text())
        if cached["archive"] == NGX_WINDOWS_SHA256 and sha256(core) == cached["file"]:
            return
    except (OSError, ValueError, KeyError):
        pass
    archive = download(NGX_WINDOWS_URL, root / "downloads" / "windows-ngx-driver.exe", NGX_WINDOWS_SHA256)
    extract_ngx_core(root, archive, core)
    marker.write_text(json.dumps({"archive": NGX_WINDOWS_SHA256, "file": sha256(core)}))
    archive.unlink(missing_ok=True)


def extract_ngx_core(root, archive, core):
    with tempfile.TemporaryDirectory(dir=root) as directory:
        staging = Path(directory)
        subprocess.run(["7z", "e", "-y", str(archive), "-r", "_nvngx.dll", "_nvngx.dl_", "-o" + directory],
                       check=True, capture_output=True, timeout=180)
        if not (staging / "_nvngx.dll").is_file() and (staging / "_nvngx.dl_").is_file():
            subprocess.run(["7z", "e", "-y", str(staging / "_nvngx.dl_"), "-o" + directory],
                           check=True, capture_output=True, timeout=60)
        extracted = staging / "_nvngx.dll"
        if not extracted.is_file() or extracted.read_bytes()[:2] != b"MZ":
            raise RuntimeError(f"Windows driver package {archive.name} does not contain a valid _nvngx.dll.")
        copy_dll(extracted, core)


# Experimental: NGX core from the host driver's R580 branch for the DLSS SR
# carrier, whose signed-module load fails under Wine with the 616 core. These
# packages are not pinned by checksum; the downloaded hash is recorded instead.
R580_DRIVER_CANDIDATES = ("581.57", "581.42", "581.29", "580.97", "580.88")


def r580_host_dir(root=None):
    return runtime_root(root) / ("rh-dlss-" + RH_DLSS_REF + "-r580")


def mirror_host(source, target):
    """Share the managed host's files; only _nvngx.dll differs."""
    for path in source.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in (".dll", ".exe") or path.name == "_nvngx.dll":
            continue
        destination = target / path.relative_to(source)
        if destination.is_file() and destination.stat().st_size == path.stat().st_size \
                and sha256(destination) == sha256(path):
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.unlink(missing_ok=True)
        try:
            os.link(path, destination)
        except OSError:
            shutil.copy2(path, destination)


def ensure_r580_core(root=None):
    """Prepare a host directory using a Windows R580 NGX core. Returns its provenance."""
    root = runtime_root(root)
    target = r580_host_dir(root)
    marker = target / ".h3-r580.json"
    with installation_lock(root):
        host = Path(runtime_env(root).get("H3_DLSSNR_DLL_DIR", ""))
        if not (host / "dlss5nr_host.exe").is_file():
            raise RuntimeError("The managed DLSS host is not installed. Restart the launcher first.")
        mirror_host(host, target)
        try:
            cached = json.loads(marker.read_text(encoding="utf-8"))
            if sha256(target / "_nvngx.dll") == cached["core_sha256"]:
                return cached
        except (OSError, ValueError, KeyError):
            pass
        override = os.getenv("H3_DLSSNR_R580_DRIVER", "").strip()
        failures = []
        for version in (override,) if override else R580_DRIVER_CANDIDATES:
            url = (f"https://us.download.nvidia.com/Windows/{version}/"
                   f"{version}-desktop-win10-win11-64bit-international-dch-whql.exe")
            try:
                print(f"[h3-nvidia] Downloading Windows R580 driver {version} for its NGX core", flush=True)
                archive = download(url, root / "downloads" / f"windows-ngx-{version}.exe")
            except (OSError, RuntimeError) as exc:
                failures.append(f"{version}: {exc}")
                continue
            try:
                info = {"driver": version, "url": url, "archive_sha256": sha256(archive)}
                extract_ngx_core(root, archive, target / "_nvngx.dll")
            finally:
                archive.unlink(missing_ok=True)
            info["core_sha256"] = sha256(target / "_nvngx.dll")
            marker.write_text(json.dumps(info), encoding="utf-8")
            return info
        raise RuntimeError("Could not obtain a Windows R580 driver package. Set H3_DLSSNR_R580_DRIVER to an "
                           "available R580 version (for example 581.57). " + "; ".join(failures)[-1500:])


def install_headless_nvapi(root):
    """Build only on the processing host/image; fix the documented server LUID gap.

    Use the actual DXGI adapter's LUID when Vulkan cannot supply one. This keeps
    device identity consistent without inventing an ID or selecting another GPU.
    """
    target = root / ("nvapi-headless-v1-" + NVAPI_HEADLESS_REF)
    dll = target / "nvapi64.dll"
    marker = target / "source.json"
    try:
        cached = json.loads(marker.read_text())
        if cached["ref"] == NVAPI_HEADLESS_REF and sha256(dll) == cached["sha256"]:
            return dll
    except (OSError, ValueError, KeyError):
        pass
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".nvapi-build-", dir=root) as directory:
        source = Path(directory) / "source"
        def run(command, **kwargs):
            result = subprocess.run(command, capture_output=True, text=True, timeout=1200, **kwargs)
            if result.returncode:
                raise RuntimeError("Headless NVAPI build failed: " + (result.stderr + result.stdout)[-4000:])
        run(["git", "clone", "--no-checkout", "https://github.com/jp7677/dxvk-nvapi.git", str(source)])
        run(["git", "checkout", "--detach", NVAPI_HEADLESS_REF], cwd=source)
        run(["git", "submodule", "update", "--init", "--recursive"], cwd=source)
        adapter = source / "src/nvapi/nvapi_adapter.cpp"
        text = adapter.read_text(encoding="utf-8")
        old = """        if (!m_vkIdProperties.deviceLUIDValid)
            return {};"""
        new = """        if (!m_vkIdProperties.deviceLUIDValid) {
            DXGI_ADAPTER_DESC1 desc{};
            if (FAILED(m_dxgiAdapter->GetDesc1(&desc)))
                return {};
            return std::make_optional(desc.AdapterLuid);
        }"""
        if text.count(old) != 1:
            raise RuntimeError("Pinned NVAPI source does not match the reviewed headless patch.")
        adapter.write_text(text.replace(old, new), encoding="utf-8")
        methods = ["nvapi.cpp", "nvapi_sys.cpp", "nvapi_disp.cpp", "nvapi_mosaic.cpp", "nvapi_ngx.cpp",
                   "nvapi_drs.cpp", "nvapi_gpu.cpp", "nvapi_d3d.cpp", "nvapi_d3d11.cpp", "nvapi_d3d12.cpp",
                   "nvapi_vulkan.cpp", "nvapi_interface.cpp"]
        run(["python3", "validate-methods.py", *("src/" + name for name in methods),
             "external/nvapi/nvapi_interface.h"], cwd=source)
        # POSIX MinGW supplies std::thread; the distribution's default may use win32 threads.
        cross = source / "build-win64.txt"
        cross.write_text(cross.read_text().replace("x86_64-w64-mingw32-g++'", "x86_64-w64-mingw32-g++-posix'"))
        build = Path(directory) / "build"
        run(["meson", "setup", str(build), str(source), "--cross-file", str(cross),
             "--buildtype", "release", "-Denable_tests=false"])
        run(["meson", "compile", "-C", str(build), "-j", "2"])
        copy_dll(build / "src/nvapi64.dll", dll)
    marker.write_text(json.dumps({"ref": NVAPI_HEADLESS_REF, "patch": "DXGI LUID fallback v1", "sha256": sha256(dll)}))
    return dll


def initialize_prefix(root, graphics, env):
    prefix, wine = Path(env["H3_DLSSNR_WINEPREFIX"]), Path(env["H3_DLSSNR_WINE"])
    prefix.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (prefix / ".h3-dlssnr.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        process_env = ensure_display(wine_env(prefix, wine))
        try:
            if not (prefix / "system.reg").is_file():
                wine_run(wine, [r"C:\windows\system32\wineboot.exe", "-u"], process_env)
            if not (prefix / "system.reg").is_file():
                raise RuntimeError("Wine did not initialize the dedicated DLSS prefix.")
            for name, source in graphics.items():
                copy_dll(source, prefix / "drive_c/windows/system32" / name)
            check_vulkan(process_env)
        except BaseException:
            try:
                stop_wine_server(wine, process_env)
            except Exception as cleanup_error:
                print(f"[h3-nvidia] Wine cleanup also failed: {cleanup_error}", flush=True)
            raise
        else:
            stop_wine_server(wine, process_env)


def ensure_runtime(root=None, *, initialize=False, system_packages=True):
    if not enabled("H3_ENABLE_DLSSNR") or os.getenv("H3_DLSSNR_EXE"):
        return
    root = runtime_root(root)
    root.mkdir(parents=True, exist_ok=True)
    with installation_lock(root):
        manifest = {"backend": "runninghub-dnr2", "upstream_ref": RH_DLSS_REF, "initialized": False}
        try:
            if platform.system() not in {"Linux", "Windows"} or platform.machine().lower() not in {"x86_64", "amd64"}:
                raise RuntimeError("DLSS setup supports Linux x86_64 and Windows x64.")
            if os.name != "nt" and system_packages:
                install_linux_packages(root)
            if os.name != "nt" and initialize:
                # Diagnose provider graphics access before downloading/building runtimes.
                graphics_env = dict(os.environ)
                graphics_env.pop("LD_LIBRARY_PATH", None)
                check_vulkan(ensure_display(vulkan_driver_env(graphics_env, root)))
            host = install_rh_host(root)
            # Reuse the existing pinned model archive; its old executable is not launched.
            models = install_archive(root, "video2dlssnr-" + DLSSNR_VERSION,
                                     DLSSNR_URL, DLSSNR_SHA256, HELPER_FILES, helper=True)
            for name in ("nvngx_dlss.dll", "nvngx_dlssnr.dll"):
                if not (host / name).is_file() or sha256(host / name) != sha256(models / name):
                    copy_dll(models / name, host / name)
            install_windows_core(root, host)
            env = {"H3_DLSSNR_EXE": str(host / "dlss5nr_host.exe"), "H3_DLSSNR_DLL_DIR": str(host),
                   "H3_DLSSNR_MODEL_SHA256": sha256(host / "nvngx_dlssnr.dll")}
            manifest["env"] = env
            if os.name != "nt":
                graphics = install_graphics(root)
                graphics["nvapi64.dll"] = install_headless_nvapi(root)
                runner = install_archive(root, WINE_COMPAT_VERSION, WINE_COMPAT_URL, WINE_COMPAT_SHA256,
                                         ("bin/wine", "bin/wineserver", "lib/wine/x86_64-unix/ntdll.so",
                                          "lib/wine/x86_64-windows/wineboot.exe"))
                env.update(H3_DLSSNR_WINE=str(runner / "bin/wine"),
                           H3_DLSSNR_WINEPREFIX=str(root / "prefix-rh-dnr2-wine11"))
                if initialize:
                    initialize_prefix(root, graphics, env)
            manifest["initialized"] = initialize or os.name == "nt"
            write_manifest(root, manifest)
            print("[h3-nvidia] DLSS Linux host files ready; GPU evaluation is checked on demand", flush=True)
        except Exception as exc:
            manifest.update(error=str(exc), initialized=False)
            write_manifest(root, manifest)
            raise


def prepare_runtime(root=None, *, system_packages=True):
    try:
        ensure_runtime(root, initialize=True, system_packages=system_packages)
    except Exception as exc:
        print(f"[h3-nvidia] DLSS unavailable on this host: {exc}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root")
    parser.add_argument("--initialize", action="store_true")
    args = parser.parse_args()
    if args.initialize:
        prepare_runtime(args.root)
    else:
        ensure_runtime(args.root)
