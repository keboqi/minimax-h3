"""Count browser round trips per user action: python -m tests.browser_ui_requests.

Each queued event costs a queue join plus a stream (a new WebSocket behind the
public proxy); an unqueued event costs one POST. On slow links the number of
round trips, not server CPU, sets the perceived latency. Pass --report to print
counts without enforcing the budgets.
"""

from pathlib import Path
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import requests
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
# Requests per action after the page has settled. Every edit should resolve
# its settings, presentation and saved preferences in one response.
# Gradio's dropdown dispatches input twice per selection; a slider's number
# box dispatches release (with the previous value, answered without updates)
# and input; typing in Width also previews its alignment before the committed
# edit; a mode or task switch also refreshes the conditioned-image canvas
# preview.
BUDGETS = {
    "load": 1,
    "size preset": 2,
    "width blur": 2,
    "steps edit": 2,
    "mode switch": 2,
    "task switch": 2,
    "idle 10s": 0,
    "saved reload": 4,
    "jobs tab 5s": 4,
}


class Recorder:
    def __init__(self, page):
        self.events = []
        self.pending = 0
        page.on("request", self._request)
        page.on("requestfinished", self._finished)
        page.on("requestfailed", self._finished)

    @staticmethod
    def _tracked(request):
        path = request.url.split("?", 1)[0]
        return request.method == "POST" and (
            path.endswith("/gradio_api/queue/join")
            or "/gradio_api/run/" in path
        )

    def _request(self, request):
        if self._tracked(request):
            self.pending += 1
            self.events.append((time.monotonic(), request.url, request.post_data))

    def _finished(self, request):
        if self._tracked(request):
            self.pending -= 1

    def settle(self, page, quiet=1.5, limit=20):
        deadline = time.monotonic() + limit
        while time.monotonic() < deadline:
            page.wait_for_timeout(100)
            last = self.events[-1][0] if self.events else 0
            if self.pending == 0 and time.monotonic() - last >= quiet:
                return
        raise TimeoutError("UI requests did not settle")

    def measure(self, page, action, *, quiet=1.5):
        start = len(self.events)
        action()
        self.settle(page, quiet=quiet)
        events = self.events[start:]
        if os.getenv("H3_UI_REQUESTS_VERBOSE"):
            for _, url, body in events:
                payload = json.loads(body or "{}")
                print("   ", url.rsplit("/", 2)[-2:], "fn", payload.get("fn_index"),
                      "trigger", payload.get("trigger_id"), flush=True)
        return events


def describe(events):
    queued = sum("/queue/join" in url for _, url, _ in events)
    return f"{len(events)} requests ({queued} queued, {len(events) - queued} direct)"


def run(report=False):
    with socket.socket() as socket_:
        socket_.bind(("127.0.0.1", 0))
        port = socket_.getsockname()[1]
    source = (
        "from h3_ui import application as app;from h3_ui.styles import H3_SETUP_CSS,H3_WORKSPACE_CSS;"
        "app.backend_status=lambda:'Connected browser test fixture';"
        "app.build_ui().queue(default_concurrency_limit=1,max_size=8).launch("
        f"server_name='127.0.0.1',server_port={port},inbrowser=False,ssr_mode=False,css=H3_SETUP_CSS+H3_WORKSPACE_CSS)"
    )
    with tempfile.TemporaryDirectory(prefix="h3-ui-requests-") as state, tempfile.TemporaryFile(mode="w+b") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", source],
            cwd=ROOT,
            env={**os.environ, "H3_WORKSPACE_DIR": state},
            stdout=log,
            stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            url = f"http://127.0.0.1:{port}"
            for _ in range(150):
                if process.poll() is not None:
                    raise RuntimeError("UI fixture exited")
                try:
                    if requests.get(url + "/config", timeout=1).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.2)
            else:
                raise TimeoutError("UI fixture did not start")
            with sync_playwright() as p:
                executable = os.getenv("H3_BROWSER_EXECUTABLE")
                chrome = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
                if not executable and chrome.exists():
                    executable = str(chrome)
                browser = p.chromium.launch(
                    headless=True,
                    **({"executable_path": executable} if executable else {}),
                )
                page = browser.new_context(viewport={"width": 1440, "height": 1000}).new_page()
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                recorder = Recorder(page)
                card = page.locator(".h3-setup-card")
                counts = {}

                def load():
                    page.goto(url, wait_until="domcontentloaded")
                    page.locator('.h3-setup-card[data-settings-ready="true"]').wait_for()

                counts["load"] = recorder.measure(page, load, quiet=3)
                page.get_by_text("Advanced settings", exact=True).click()
                page.get_by_text("Exact dimensions, seed & image frames", exact=True).click()
                recorder.settle(page)

                def size_preset():
                    page.get_by_label("Size", exact=True).click()
                    page.get_by_role("option", name="1080p", exact=True).click()
                    expect(card).to_contain_text("1920")

                counts["size preset"] = recorder.measure(page, size_preset)
                width = page.get_by_label("Width", exact=True)

                def width_blur():
                    width.fill("1280")
                    width.press("Tab")
                    expect(card).to_contain_text("1280")

                counts["width blur"] = recorder.measure(page, width_blur)
                page.get_by_text("Sampling steps (advanced)", exact=True).click()
                recorder.settle(page)
                steps = (
                    page.get_by_text("Steps", exact=True)
                    .locator('xpath=ancestor::div[contains(@class,"block")][1]')
                    .locator('input[type="number"]')
                )

                def steps_edit():
                    steps.fill("6")
                    steps.press("Tab")
                    expect(card).to_contain_text("6 steps")

                counts["steps edit"] = recorder.measure(page, steps_edit)

                def mode_switch():
                    page.get_by_label("First / last frame", exact=True).check()
                    expect(page.get_by_text("Last frame", exact=True).first).to_be_visible()

                counts["mode switch"] = recorder.measure(page, mode_switch)

                def task_switch():
                    page.locator(".h3-task-picker").get_by_label("Image", exact=True).check()
                    expect(card).to_contain_text("Image")

                counts["task switch"] = recorder.measure(page, task_switch)
                counts["idle 10s"] = recorder.measure(
                    page, lambda: page.wait_for_timeout(10000), quiet=0.5
                )
                # Edits must save themselves: reload restores them in one pass.
                page.wait_for_function(
                    "Object.keys(localStorage).some(k=>k.includes('minimax-h3:settings:v3'))"
                )

                def saved_reload():
                    load()
                    expect(card).to_contain_text("Image")
                    expect(card).to_contain_text("6 steps")

                counts["saved reload"] = recorder.measure(page, saved_reload, quiet=3)
                # The job list polls only while its tab is visible.
                counts["jobs tab 5s"] = recorder.measure(
                    page,
                    lambda: (
                        page.get_by_role("tab", name="Jobs", exact=True).click(),
                        page.wait_for_timeout(5000),
                    ),
                    quiet=0.5,
                )
                failures = []
                for action, events in counts.items():
                    budget = BUDGETS[action]
                    print(f"{action:12} {describe(events)} · budget {budget}")
                    if len(events) > budget:
                        failures.append(action)
                assert not errors, errors
                browser.close()
                if failures and not report:
                    raise AssertionError("Request budget exceeded: " + ", ".join(failures))
                print("UI request budgets passed." if not failures else "Report only.")
        except Exception:
            log.seek(0)
            print(log.read().decode("utf-8", errors="replace")[-10000:])
            raise
        finally:
            process.terminate()
            process.wait(timeout=15)


if __name__ == "__main__":
    run(report="--report" in sys.argv)
