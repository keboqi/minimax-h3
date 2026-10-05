"""Verify live Gradio progress when public HTTP SSE requests are unavailable."""

import argparse
import contextlib
import io
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from unittest.mock import patch

from playwright.sync_api import sync_playwright, expect
import requests


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / ".cache" / "queue-transport"


def run(*, cloudflare=False):
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    log_path = ARTIFACTS / "server.log"
    with TemporaryDirectory(prefix="queue-transport-") as directory:
        root = Path(directory)
        with socket.socket() as socket_port:
            socket_port.bind(("127.0.0.1", 0))
            port = socket_port.getsockname()[1]
        env = dict(os.environ, HF_HUB_OFFLINE="1", H3_WORKSPACE_DIR=str(root / "workspace"))
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", "-m", "tests.queue_transport_fixture", str(port), directory],
                cwd=ROOT, env=env, stdout=log, stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            tunnel = None
            try:
                url = f"http://127.0.0.1:{port}"
                for _ in range(100):
                    if process.poll() is not None:
                        raise RuntimeError(log_path.read_text())
                    try:
                        if requests.get(url + "/test/ready", timeout=1).ok:
                            break
                    except requests.RequestException:
                        pass
                    time.sleep(0.2)
                else:
                    raise TimeoutError("Queue fixture server unavailable")
                browser_url = url
                if cloudflare:
                    from h3_app import public_url

                    tunnel_output = io.StringIO()

                    def record_tunnel_logs(proc):
                        with proc.stdout:
                            for line in proc.stdout:
                                tunnel_output.write(line)

                    with contextlib.redirect_stdout(tunnel_output), patch.object(public_url, "drain_cloudflare_logs", record_tunnel_logs):
                        tunnel = public_url.launch_cloudflare(port, cache_dir=ROOT / ".cache" / "cloudflared")
                        if tunnel is None:
                            raise RuntimeError(tunnel_output.getvalue())
                        deadline = time.monotonic() + 60
                        while time.monotonic() < deadline:
                            match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", tunnel_output.getvalue())
                            if match:
                                browser_url = match.group(0)
                                break
                            if tunnel.poll() is not None:
                                raise RuntimeError(tunnel_output.getvalue())
                            time.sleep(0.2)
                        else:
                            raise TimeoutError("Cloudflare did not announce a public URL")
                    deadline = time.monotonic() + 30
                    last_error = None
                    while time.monotonic() < deadline:
                        try:
                            response = requests.get(browser_url + "/test/ready", timeout=3)
                            if response.ok and response.json().get("ready"):
                                break
                        except (requests.RequestException, ValueError) as error:
                            last_error = error
                        time.sleep(0.5)
                    else:
                        raise TimeoutError(f"Cloudflare endpoint unavailable: {last_error}\n{tunnel_output.getvalue()}")
                with sync_playwright() as playwright:
                    chrome = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
                    executable = os.getenv("H3_BROWSER_EXECUTABLE")
                    browser = playwright.chromium.launch(
                        headless=True,
                        **({"executable_path": executable or str(chrome)} if executable or chrome.exists() else {}),
                    )
                    page = browser.new_page()
                    errors = []
                    streams = []
                    blocked = []
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.on("websocket", lambda stream: streams.append(stream.url))

                    def block_sse(route):
                        blocked.append(route.request.url)
                        route.abort()

                    page.route("**/gradio_api/queue/data?**", block_sse)
                    page.route("**/gradio_api/heartbeat/**", block_sse)
                    page.goto(browser_url, wait_until="domcontentloaded")
                    page.get_by_role("button", name="Generate fixture", exact=True).wait_for()
                    output = page.get_by_label("Generation progress", exact=True)
                    expect(output).to_have_value("Ready", timeout=15000)
                    page.get_by_role("button", name="Generate fixture", exact=True).click()
                    expect(output).to_have_value("Sampling step 1 / 3", timeout=15000)
                    page.screenshot(path=str(ARTIFACTS / "progress.png"))
                    # Completion is impossible until the browser has seen step 1.
                    assert any("/gradio_api/queue/data?" in stream for stream in streams), streams
                    requests.post(url + "/test/continue", timeout=5).raise_for_status()
                    expect(output).to_have_value("Generation complete", timeout=10000)
                    assert not blocked, blocked
                    assert not errors, errors
                    browser.close()
                    route = "a live Cloudflare Quick Tunnel" if cloudflare else "localhost"
                    print(f"PASS: intermediate generation progress and completion arrived over WebSockets through {route} with HTTP SSE blocked")
            except BaseException:
                print(log_path.read_text(encoding="utf-8", errors="replace"))
                raise
            finally:
                try:
                    if tunnel is not None:
                        from h3_app.public_url import stop_cloudflare

                        stop_cloudflare(tunnel)
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloudflare", action="store_true", help="Also verify through a temporary public Cloudflare Quick Tunnel")
    run(cloudflare=parser.parse_args().cloudflare)
