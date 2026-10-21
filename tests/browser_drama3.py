"""Drama 3 multi-speaker UI budgets: python -m tests.browser_drama3.

Gradio's frontend refreshes the loading status of every component an earlier
server event used, on each server event, so slot titles, clip flags, panel
counts and Add/Remove must stay in the browser. This fills all 16 speakers,
generates every voice with the fixture backend and checks the request counts
and the titles those browser-side handlers keep up to date.
"""

import json
import re
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import requests
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
# Server requests per action once the page has settled.
BUDGETS = {
    "apply 16-speaker reply": 2,  # the writer, then its sound-slot sync
    "type a speaker name": 0,
    "match speaker on blur": 1,
    "generate all voices": 2,  # admission, then the queued job
    "remove a speaker": 0,
    "add a speaker": 0,
    "speech preset": 0,
}


def writer_reply():
    return json.dumps({
        "script": "\n".join(f"<|speaker:{i}|> Line {i}." for i in range(16)),
        "speakers": [{"speaker_id": i, "name": f"Character {i}",
                      "description": "A warm midrange voice with crisp articulation and a natural pace.",
                      "preview_text": "The morning is quiet, and I am ready to begin our journey."}
                     for i in range(16)],
    })


def run():
    with socket.socket() as socket_:
        socket_.bind(("127.0.0.1", 0))
        port = socket_.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="h3-drama3-ui-") as state, \
            tempfile.TemporaryFile(mode="w+b") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", "-m", "tests.workspace_fixture", str(port)],
            cwd=ROOT, env={**os.environ, "H3_WORKSPACE_DIR": state}, stdout=log, stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            url = f"http://127.0.0.1:{port}"
            for _ in range(300):
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
                browser = p.chromium.launch(headless=True, **({"executable_path": executable} if executable else {}))
                page = browser.new_page(viewport={"width": 1440, "height": 1000})
                errors, posts = [], []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on("request", lambda request: posts.append(time.monotonic())
                        if request.method == "POST" and "/gradio_api/" in request.url else None)
                expect.set_options(timeout=20000)
                page.goto(url, wait_until="domcontentloaded")
                page.locator('.h3-setup-card[data-settings-ready="true"]').wait_for(timeout=60000)
                page.locator(".h3-task-picker").get_by_label("Soundstage", exact=True).check()
                form = page.locator('#h3-engine-tabs > [role="tabpanel"]:visible')
                counts = {}

                def measure(label, action, quiet=2.0):
                    posts.clear()
                    action()
                    last, seen = time.monotonic(), 0
                    while time.monotonic() - last < quiet:
                        page.wait_for_timeout(100)
                        if len(posts) != seen:
                            seen, last = len(posts), time.monotonic()
                    counts[label] = len(posts)

                panel = form.locator("button", has_text="Speaker voices").first
                form.get_by_text("Rewrite as a Drama 3 script", exact=True).click()
                form.get_by_text("Use another chat model", exact=True).click()
                form.get_by_label("Pasted writer reply").fill(writer_reply())
                measure("apply 16-speaker reply",
                        lambda: form.get_by_role("button", name="Apply pasted reply").click())
                expect(panel).to_contain_text("16 speakers · 0 with a voice")
                expect(panel).to_have_class(re.compile(r"\bopen\b"))  # the Cast column starts open
                expect(form.locator("button", has_text="Speaker 15 · Character 15")).to_have_count(1)
                # Voice slots start collapsed.
                form.locator("button", has_text="Speaker 0 · Character 0").click()
                name = form.get_by_label("Speaker 0 name")
                name.click()
                name.press("End")
                measure("type a speaker name", lambda: name.press_sequentially(" the Bold", delay=40))
                expect(form.locator("button", has_text="Speaker 0 · Character 0 the Bold")).to_have_count(1)
                measure("match speaker on blur", lambda: name.press("Tab"))
                measure("generate all voices",
                        lambda: form.get_by_role("button", name="Generate all speaker voices").click())
                expect(form.get_by_text("Generated 16 of 16 speaker voices.")).to_be_visible()
                expect(panel).to_contain_text("16 speakers · 16 with a voice")
                expect(form.locator("button", has_text="Speaker 15 · Character 15 · voice ready")).to_have_count(1)
                measure("remove a speaker", lambda: form.get_by_role("button", name="Remove last speaker").click())
                expect(panel).to_contain_text("15 speakers · 15 with a voice")
                measure("add a speaker", lambda: form.get_by_role("button", name="Add speaker").click())
                # The removed slot came back cleared.
                expect(panel).to_contain_text("16 speakers · 15 with a voice")
                expect(form.locator("button", has_text="Speaker 15")).not_to_contain_text("Character")
                measure("speech preset", lambda: form.get_by_label("Expressive", exact=True).check())
                assert not errors, errors
                browser.close()
            print("Drama 3 requests per action:", counts)
            over = {label: (counts[label], limit) for label, limit in BUDGETS.items() if counts[label] > limit}
            assert not over, f"Request budgets exceeded (count, budget): {over}"
            print("Drama 3 browser acceptance passed: 16-speaker writer fill, titles, batch voices, Add/Remove, presets.")
        except Exception:
            log.seek(0)
            print(log.read().decode("utf-8", errors="replace")[-10000:])
            raise
        finally:
            process.terminate()
            process.wait(timeout=15)


if __name__ == "__main__":
    run()
