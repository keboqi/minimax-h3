"""Shared, GPU-free provisioning for standalone, Colab and Modal enhancement.

Only startup prepares Wine/driver bridges. UI discovery reads the manifest;
it never downloads software, starts Wine, or initializes a GPU.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from urllib.request import Request, urlopen
import zipfile

from h3_sources import (
    DLSSNR_SHA256, DLSSNR_URL, DLSSNR_VERSION,
    PROTON_SHA256, PROTON_URL, PROTON_VERSION,
    WINE_COMPAT_SHA256, WINE_COMPAT_URL, WINE_COMPAT_VERSION,
)

LINUX_PACKAGES = (
    "libvulkan1", "libgl1", "libegl1", "libx11-6", "libxext6", "libxrender1",
    "libxrandr2", "libxi6", "libxfixes3", "libxcursor1", "libxcomposite1",
    "libxinerama1", "libxxf86vm1", "libfreetype6",
    "libfontconfig1", "libdbus-1-3", "libsm6", "libice6", "libusb-1.0-0",
    "libudev1", "libunwind8", "xvfb", "xauth",
)
HELPER_FILES = ("video2dlssnr.exe", "nvngx.dll_dlssnr.dll", "nvngx_dlss.dll", "nvngx_dlssnr.dll")
TRANSLATION_DLLS = {
    "dxgi.dll": "dxvk", "d3d11.dll": "dxvk",
    "d3d12.dll": "vkd3d-proton", "d3d12core.dll": "vkd3d-proton",
    "nvapi64.dll": "nvapi",
}
BRIDGE_DIRS = (
    "/usr/lib/nvidia/wine", "/usr/lib64/nvidia/wine", "/lib/nvidia/wine",
    "/lib64/nvidia/wine", "/usr/lib/x86_64-linux-gnu/nvidia/wine",
    "/usr/lib/x86_64-linux-gnu/nvidia/current/wine",
    "/usr/lib/x86_64-linux-gnu/nvidia/current/nvidia/wine",
)
NATIVE_OVERRIDES = "d3d12,d3d12core,dxgi,d3d11,nvapi64,nvngx,_nvngx=n;winemenubuilder.exe=d"


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
        except (OSError, ValueError):
            partial.unlink(missing_ok=True)
            if attempt == 2:
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
                for filename in HELPER_FILES:
                    with bundle.open("out/" + filename) as source, (staging / filename).open("wb") as dest:
                        shutil.copyfileobj(source, dest)
                (staging / "README.md").write_bytes(bundle.read("README.md"))
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
    if (root / ".linux-packages-v2").is_file() and shutil.which("xvfb-run"):
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
    (root / ".linux-packages-v2").touch()


def wine_env(prefix, wine):
    lib = str(Path(wine).parent.parent / "lib")
    return dict(os.environ, WINEPREFIX=str(prefix), WINEDEBUG="-all", WINEARCH="win64",
                WINEDLLOVERRIDES=NATIVE_OVERRIDES, DXVK_ENABLE_NVAPI="1", PROTON_ENABLE_NVAPI="1",
                LD_LIBRARY_PATH=lib + (":" + os.environ["LD_LIBRARY_PATH"] if os.getenv("LD_LIBRARY_PATH") else ""))


def headless_command(command):
    if os.name != "nt" and not os.getenv("DISPLAY"):
        xvfb = shutil.which("xvfb-run")
        if not xvfb:
            raise RuntimeError("The headless DLSS display is missing. Restart run_h3.sh to repair NVIDIA setup.")
        return [xvfb, "-a", "-s", "-screen 0 1280x720x24", *command]
    return command


def wine_run(wine, args, env):
    result = subprocess.run(headless_command([str(wine), *args]), env=env, text=True,
                            capture_output=True, timeout=180)
    if result.returncode:
        raise RuntimeError("DLSS Wine setup failed: " + (result.stderr or result.stdout)[-2000:])


def copy_dll(source, destination):
    # Proton's template may contain symlinks into its bundle. Never write through
    # those links and corrupt the runner's built-in libraries.
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    shutil.copy2(source, destination)


def driver_bridges(root):
    """Extract matching user-space bridges only; never install/replace a driver."""
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                check=True, capture_output=True, text=True, timeout=15)
        version = result.stdout.splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError) as exc:
        raise RuntimeError("An NVIDIA GPU and a working host driver are required for DLSS.") from exc
    if not re.fullmatch(r"\d+\.\d+(?:\.\d+)?", version):
        raise RuntimeError("Could not identify the NVIDIA host driver for DLSS setup.")
    for directory in [*(Path(path) for path in BRIDGE_DIRS), root / "drivers" / version]:
        if all((directory / name).is_file() for name in ("nvngx.dll", "_nvngx.dll")):
            return directory, version
    print(f"[h3-nvidia] Obtaining NGX bridges for host driver {version}", flush=True)
    url = f"https://us.download.nvidia.com/XFree86/Linux-x86_64/{version}/NVIDIA-Linux-x86_64-{version}.run"
    archive = download(url, root / "downloads" / f"nvidia-{version}.run")
    with tempfile.TemporaryDirectory(prefix=".driver-", dir=root) as directory:
        extracted = Path(directory) / "driver"
        subprocess.run(["sh", str(archive), "--extract-only", "--target", str(extracted)],
                       check=True, capture_output=True, timeout=180)
        destination = root / "drivers" / version
        destination.mkdir(parents=True, exist_ok=True)
        sources = {}
        for name in ("nvngx.dll", "_nvngx.dll"):
            matches = list(extracted.rglob(name))
            if not matches:
                raise RuntimeError(f"NVIDIA driver {version} does not include {name}; DLSS is unavailable on this host.")
            sources[name] = matches[0]
        for name, source in sources.items():
            shutil.copy2(source, destination / name)
        (destination / "source.json").write_text(json.dumps({"url": url, "sha256": sha256(archive)}), encoding="utf-8")
    archive.unlink(missing_ok=True)
    return destination, version


def initialize_prefix(root, proton, env):
    prefix = Path(env["H3_DLSSNR_WINEPREFIX"])
    prefix.mkdir(parents=True, exist_ok=True)
    with (prefix / ".h3-dlssnr.lock").open("a+b") as lock:
        if os.name != "nt":
            import fcntl
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("The dedicated DLSS prefix is busy; retry startup after enhancement finishes.") from exc
        return _initialize_prefix(root, proton, env)


def _initialize_prefix(root, proton, env):
    prefix = Path(env["H3_DLSSNR_WINEPREFIX"])
    wine = Path(env["H3_DLSSNR_WINE"])
    process_env = wine_env(prefix, wine)
    try:
        if not (prefix / "system.reg").is_file():
            wine_run(wine, ["wineboot", "-u"], dict(process_env, WINEDLLOVERRIDES="winemenubuilder.exe=d"))
        if not (prefix / "system.reg").is_file():
            raise RuntimeError("Wine did not initialize the managed DLSS prefix.")
        system32 = prefix / "drive_c/windows/system32"
        for name, package in TRANSLATION_DLLS.items():
            copy_dll(proton / "files/lib/wine" / package / "x86_64-windows" / name, system32 / name)
        bridges, version = driver_bridges(root)
        for name in ("nvngx.dll", "_nvngx.dll"):
            copy_dll(bridges / name, system32 / name)
            copy_dll(bridges / name, prefix / "drive_c/ngx_bridge" / name)
        keys = (
            (r"HKLM\SOFTWARE\NVIDIA Corporation\Global\NGXCore", "FullPath"),
            (r"HKLM\SOFTWARE\NVIDIA Corporation\Global\NGXCore", "NGXPath"),
            (r"HKLM\System\CurrentControlSet\Services\nvlddmkm\NGXCore", "NGXPath"),
            (r"HKLM\System\CurrentControlSet\Services\nvlddmkm\Parameters\NGXCore", "NGXPath"),
        )
        for key, value in keys:
            wine_run(wine, ["reg", "add", key, "/v", value, "/t", "REG_SZ", "/d", r"C:\ngx_bridge", "/f"], process_env)
        return version
    finally:
        subprocess.run([str(wine.parent / "wineserver"), "-k"], env=process_env,
                       check=True, timeout=15, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@contextmanager
def installation_lock(root):
    with (root / ".install.lock").open("a+b") as lock:
        if os.name != "nt":
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield


def choose_wine_runner(root, proton):
    # GE's ntdll requires glibc 2.38. Colab/Ubuntu 22.04 can still use its PE
    # graphics DLLs with this separately pinned glibc-2.27-compatible Wine build.
    _, version = platform.libc_ver()
    parts = tuple(int(part) for part in version.split(".")[:2]) if version else (0, 0)
    if parts >= (2, 38):
        return proton / "files/bin/wine", PROTON_VERSION
    compatible = install_archive(root, WINE_COMPAT_VERSION, WINE_COMPAT_URL,
                                 WINE_COMPAT_SHA256, ("bin/wine", "bin/wineserver"))
    return compatible / "bin/wine", WINE_COMPAT_VERSION


def ensure_runtime(root=None, *, initialize=False, system_packages=True):
    """Install immutable files at build time, initialize against the live host later."""
    if not enabled("H3_ENABLE_DLSSNR") or os.getenv("H3_DLSSNR_EXE"):
        return
    root = runtime_root(root)
    root.mkdir(parents=True, exist_ok=True)
    with installation_lock(root):
        manifest = read_manifest(root)
        try:
            if platform.system() not in {"Linux", "Windows"} or platform.machine().lower() not in {"x86_64", "amd64"}:
                raise RuntimeError("Automatic DLSS setup supports Linux x86_64 and Windows x64.")
            helper = install_archive(root, "video2dlssnr-" + DLSSNR_VERSION,
                                     DLSSNR_URL, DLSSNR_SHA256, HELPER_FILES, helper=True)
            env = {"H3_DLSSNR_EXE": str(helper / "video2dlssnr.exe"), "H3_DLSSNR_DLL_DIR": str(helper),
                   "H3_DLSSNR_MODEL_SHA256": sha256(helper / "nvngx_dlssnr.dll")}
            manifest = {"env": env, "helper_version": DLSSNR_VERSION, "initialized": os.name == "nt"}
            if os.name != "nt":
                if system_packages:
                    install_linux_packages(root)
                required = ("files/bin/wine", "files/bin/wineserver", *(
                    f"files/lib/wine/{package}/x86_64-windows/{name}" for name, package in TRANSLATION_DLLS.items()))
                proton = install_archive(root, PROTON_VERSION, PROTON_URL, PROTON_SHA256, required)
                wine, runner_version = choose_wine_runner(root, proton)
                env.update(H3_DLSSNR_WINE=str(wine),
                           H3_DLSSNR_WINEPREFIX=str(root / "prefix"))
                manifest["runner_version"] = runner_version
                if initialize:
                    manifest["driver_version"] = initialize_prefix(root, proton, env)
                    manifest["initialized"] = True
            write_manifest(root, manifest)
            print("[h3-nvidia] Runtime files ready" + ("; host prefix initialized" if initialize else ""), flush=True)
        except Exception as exc:
            manifest["error"] = str(exc)
            manifest["initialized"] = False
            write_manifest(root, manifest)
            raise


def prepare_runtime(root=None, *, system_packages=True):
    """Keep generation usable when this experimental optional backend cannot start."""
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
