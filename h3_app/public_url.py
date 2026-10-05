"""Cloudflare public URLs and bounded tunnel process cleanup."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import threading
from pathlib import Path
from urllib.request import urlopen


def public_url_settings() -> tuple[bool, bool]:
    """Use Cloudflare by default; an explicit Gradio Share opt-in replaces it."""
    truthy = {"1", "true", "yes", "on"}
    share = os.getenv("GRADIO_SHARE", "false").strip().lower() in truthy
    cloudflare = os.getenv(
        "ENABLE_CLOUDFLARE_TUNNEL", "false" if share else "true"
    ).strip().lower() in truthy
    return cloudflare, share


def cloudflared_binary(cache_dir: Path) -> str:
    installed = shutil.which("cloudflared")
    if installed:
        return installed
    system = platform.system().lower()
    machine = platform.machine().lower()
    architecture = {
        "x86_64": "amd64",
        "amd64": "amd64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }.get(machine)
    if (
        system not in {"linux", "windows"}
        or architecture is None
        or (system == "windows" and architecture != "amd64")
    ):
        raise RuntimeError("Install cloudflared on PATH for this platform")
    suffix = ".exe" if system == "windows" else ""
    binary = cache_dir / f"cloudflared-{system}-{architecture}{suffix}"
    if not binary.is_file():
        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = binary.with_suffix(".partial")
        url = f"https://github.com/cloudflare/cloudflared/releases/latest/download/{binary.name}"
        print("[h3-ui] Downloading cloudflared...", flush=True)
        try:
            with urlopen(url, timeout=120) as response, temporary.open("wb") as target:
                shutil.copyfileobj(response, target)
            temporary.chmod(0o755)
            os.replace(temporary, binary)
        finally:
            temporary.unlink(missing_ok=True)
    return str(binary)


def stop_cloudflare(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            proc.wait(timeout=5)
    else:
        proc.wait()


def drain_cloudflare_logs(proc: subprocess.Popen) -> None:
    """Keep draining both streams after announcing the public URL once."""
    announced = False
    last_line = "No startup output received"
    with proc.stdout:
        for line in proc.stdout:
            last_line = line.strip() or last_line
            match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
            if match and not announced:
                announced = True
                url = match.group(0)
                print(f"[h3-ui] Public Cloudflare URL: {url}", flush=True)
                print(f"[h3-ui] ComfyUI Editor Proxy: {url}/comfyui/", flush=True)
    if not announced:
        print(f"[h3-ui] Cloudflare exited without a public URL: {last_line}", flush=True)


def launch_cloudflare(port: int, *, cache_dir: Path) -> subprocess.Popen | None:
    proc = None
    try:
        binary = cloudflared_binary(cache_dir)
        proc = subprocess.Popen(
            [binary, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}),
        )
        threading.Thread(target=drain_cloudflare_logs, args=(proc,), daemon=True).start()
        return proc
    except (KeyboardInterrupt, SystemExit):
        stop_cloudflare(proc)
        raise
    except Exception as exc:
        stop_cloudflare(proc)
        print(f"[h3-ui] Could not create Cloudflare URL: {exc}", flush=True)
        return None
