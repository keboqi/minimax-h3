"""Shared, GPU-free provisioning for standalone, Colab and Modal enhancement.

Only startup prepares Wine/driver bridges. UI discovery reads the manifest;
it never downloads software, starts Wine, or initializes a GPU.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gzip
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
from urllib.error import HTTPError
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
    "libudev1", "libunwind8", "xvfb", "xauth", "dpkg",
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
NVIDIA_REPO = "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/"
NGX_FILES = ("libnvidia-ngx.so.1", "nvngx.dll", "_nvngx.dll")
NGX_DIRS = ("/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib/nvidia",
            "/usr/local/nvidia/lib64", "/usr/lib/x86_64-linux-gnu/nvidia/current")


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
    if (root / ".linux-packages-v2").is_file() and shutil.which("xvfb-run") and shutil.which("dpkg-deb"):
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


def native_env(root=None, base=None):
    """Set the loader path before launching a worker, including Wine's NGX core."""
    env = dict(os.environ if base is None else base)
    library = read_manifest(runtime_root(root)).get("vsr", {}).get("library")
    if library:
        directory = str(Path(library).parent)
        paths = env.get("LD_LIBRARY_PATH", "").split(":")
        env["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys([directory, *(path for path in paths if path)]))
        env["H3_VSR_NGX_LIBRARY"] = str(library)
    return env


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
    # Proton's template may contain symlinks into its bundle. Never write through
    # those links and corrupt the runner's built-in libraries.
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    shutil.copy2(source, destination)


def seed_proton_prefix(proton, prefix):
    """Proton requires its bundled prefix; bare wineboot cannot create one."""
    template = proton / "files/share/default_pfx"
    if not (template / "system.reg").is_file():
        raise RuntimeError("The bundled Proton prefix template is missing; restart setup to repair it.")
    libraries = (proton / "files/lib/wine").resolve()
    for directory, dirs, names in os.walk(template):
        source_dir = Path(directory)
        destination_dir = prefix / source_dir.relative_to(template)
        destination_dir.mkdir(parents=True, exist_ok=True)
        for name in dirs + names:
            source, destination = source_dir / name, destination_dir / name
            if destination.exists():
                continue
            if source.is_symlink():
                # Relative builtin links point outside the template. Relocating
                # them unchanged would leave kernel32 and the other DLLs missing.
                target = source.resolve()
                link = str(target) if target.is_relative_to(libraries) else os.readlink(source)
                destination.unlink(missing_ok=True)
                destination.symlink_to(link)
            elif source.is_file():
                destination.unlink(missing_ok=True)
                shutil.copy2(source, destination)
    devices = prefix / "dosdevices"
    devices.mkdir(exist_ok=True)
    for name, target in (("c:", "../drive_c"), ("z:", "/")):
        link = devices / name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(target)
    # Match Proton's launcher: its prebuilt registry is already initialized.
    timestamp = int((proton / "files/share/wine/wine.inf").stat().st_mtime)
    (prefix / ".update-timestamp").write_text(str(timestamp) + "\n", encoding="ascii")


def host_driver_version():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                check=True, capture_output=True, text=True, timeout=15)
        version = result.stdout.splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError) as exc:
        raise RuntimeError("NVIDIA enhancement requires a GPU and a working host driver (nvidia-smi).") from exc
    if not re.fullmatch(r"\d+\.\d+(?:\.\d+)?", version):
        raise RuntimeError("Could not identify the NVIDIA host driver for NGX setup.")
    return version


def driver_package(version, index):
    """Resolve only the exact host version and architecture in NVIDIA's index."""
    for stanza in index.replace("\r\n", "\n").split("\n\n"):
        fields = dict(line.split(": ", 1) for line in stanza.splitlines() if ": " in line and not line.startswith(" "))
        if (fields.get("Package") != "libnvidia-gl-" + version.split(".")[0]
                or fields.get("Architecture") != "amd64"
                or not fields.get("Version", "").startswith(version + "-")):
            continue
        filename = fields.get("Filename", "").removeprefix("./")
        checksum = fields.get("SHA256", "")
        if not re.fullmatch(r"[\w.+~-]+\.deb", filename) or not re.fullmatch(r"[a-f0-9]{64}", checksum):
            raise RuntimeError("NVIDIA package index contains an invalid filename or checksum.")
        return NVIDIA_REPO + filename, checksum
    return None


def driver_components(root, version):
    """Extract NGX files into the app only. Never execute a driver installer."""
    destination = root / "drivers" / version
    marker = destination / "source.json"
    try:
        installed = json.loads(marker.read_text(encoding="utf-8"))
        if installed["driver_version"] == version and all(
            (destination / name).is_file() and sha256(destination / name) == installed["files"][name]
            for name in NGX_FILES
        ):
            return destination
    except (OSError, ValueError, KeyError):
        pass
    print(f"[h3-nvidia] Obtaining native NGX and Wine bridges for host driver {version}", flush=True)
    index = download(NVIDIA_REPO + "Packages.gz", root / "downloads/Packages.gz")
    package = driver_package(version, gzip.decompress(index.read_bytes()).decode("utf-8"))
    if package is None:
        raise RuntimeError(f"NVIDIA's package repository has no exact amd64 NGX package for host driver {version}. "
                           "Ask the host provider to expose libnvidia-ngx.so.1 and NVIDIA's Wine bridges; "
                           "a different driver's libraries cannot be substituted.")
    url, checksum = package
    archive = download(url, root / "downloads" / url.rsplit("/", 1)[-1], checksum)
    if not shutil.which("dpkg-deb"):
        raise RuntimeError("NGX extraction requires dpkg-deb; restart run_h3.sh or redeploy Modal to install prerequisites.")
    with tempfile.TemporaryDirectory(prefix=".driver-", dir=root) as directory:
        extracted = Path(directory)
        # -x extracts data only, without installing the package or running scripts.
        subprocess.run(["dpkg-deb", "-x", str(archive), str(extracted)],
                       check=True, capture_output=True, timeout=180)
        sources = {"libnvidia-ngx.so.1": extracted / "usr/lib/x86_64-linux-gnu" / f"libnvidia-ngx.so.{version}"}
        sources.update({name: extracted / "usr/lib/x86_64-linux-gnu/nvidia/wine" / name
                        for name in ("nvngx.dll", "_nvngx.dll")})
        for name, source in sources.items():
            if not source.is_file() or source.is_symlink() or not source.stat().st_size:
                raise RuntimeError(f"NVIDIA package for {version} is missing {name}.")
        destination.mkdir(parents=True, exist_ok=True)
        for name, source in sources.items():
            copy_dll(source, destination / name)
        marker.write_text(json.dumps({"url": url, "sha256": checksum, "driver_version": version,
                                     "files": {name: sha256(destination / name) for name in NGX_FILES}}), encoding="utf-8")
    archive.unlink(missing_ok=True)
    return destination


def driver_bridges(root):
    version = host_driver_version()
    for directory in (Path(path) for path in BRIDGE_DIRS):
        if all((directory / name).is_file() for name in ("nvngx.dll", "_nvngx.dll")):
            return directory, version
    return driver_components(root, version), version


def prepare_native_runtime(root):
    """VSR needs the driver NGX core even when DLSS is disabled or fails."""
    if platform.system() != "Linux" or not (enabled("H3_ENABLE_NVIDIA_VSR") or enabled("H3_ENABLE_DLSSNR")):
        return
    with installation_lock(root):
        manifest = read_manifest(root)
        try:
            if platform.machine().lower() not in {"x86_64", "amd64"}:
                raise RuntimeError("Automatic native NGX setup supports Linux x86_64.")
            version = host_driver_version()
            paths = [*(Path(path) for path in os.getenv("LD_LIBRARY_PATH", "").split(":") if path),
                     *(Path(path) for path in NGX_DIRS)]
            library = next((directory / name for directory in paths
                            for name in (f"libnvidia-ngx.so.{version}", "libnvidia-ngx.so.1")
                            if (directory / name).is_file()), None)
            if library is None:
                library = driver_components(root, version) / "libnvidia-ngx.so.1"
            manifest["vsr"] = {"library": str(library.resolve()), "driver_version": version}
            print(f"[h3-nvidia] Native NGX for driver {version}: {library}", flush=True)
        except Exception as exc:
            manifest["vsr"] = {"error": str(exc)}
            print(f"[h3-nvidia] VSR NGX setup failed: {exc}", flush=True)
        write_manifest(root, manifest)


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
    failed = False
    try:
        if not (prefix / "system.reg").is_file() or not (prefix / "drive_c/windows/system32/kernel32.dll").is_file():
            if wine == proton / "files/bin/wine":
                seed_proton_prefix(proton, prefix)
            else:
                wine_run(wine, [r"C:\windows\system32\wineboot.exe", "-u"], dict(process_env, WINEDLLOVERRIDES="winemenubuilder.exe=d"))
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
            wine_run(wine, [r"C:\windows\system32\reg.exe", "add", key, "/v", value, "/t", "REG_SZ", "/d", r"C:\ngx_bridge", "/f"], process_env)
        return version
    except BaseException:
        failed = True
        raise
    finally:
        try:
            stop_wine_server(wine, process_env)
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            if not failed:
                raise
            print(f"[h3-nvidia] Wine cleanup also failed: {exc}", flush=True)


@contextmanager
def installation_lock(root):
    with (root / ".install.lock").open("a+b") as lock:
        if os.name != "nt":
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield


def choose_wine_runner(root, proton):
    # GE's ntdll requires glibc 2.38. Colab/Ubuntu 22.04 can still use its PE
    # graphics DLLs with a glibc-2.27-compatible WoW64 build, without 32-bit libs.
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
            manifest = {**manifest, "env": env, "helper_version": DLSSNR_VERSION, "initialized": os.name == "nt"}
            manifest.pop("error", None)
            if os.name != "nt":
                if system_packages:
                    install_linux_packages(root)
                required = ("files/bin/wine", "files/bin/wineserver", "files/share/default_pfx/system.reg",
                            "files/share/default_pfx/drive_c/windows/system32/kernel32.dll", *(
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
    root = runtime_root(root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        ensure_runtime(root, initialize=True, system_packages=system_packages)
    except Exception as exc:
        print(f"[h3-nvidia] DLSS unavailable on this host: {exc}", flush=True)
    prepare_native_runtime(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root")
    parser.add_argument("--initialize", action="store_true")
    args = parser.parse_args()
    if args.initialize:
        prepare_runtime(args.root)
    else:
        ensure_runtime(args.root)
