"""Run the UI-only browser acceptance checks: python tests/browser_settings.py.

Uses installed Chrome on Windows, or Playwright Chromium elsewhere. No GPU or
model downloads. H3_BROWSER_EXECUTABLE can select another Chromium executable.
"""

from pathlib import Path
import os
import socket
import subprocess
import sys
import tempfile
import time
import requests
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]


def run():
    with socket.socket() as socket_:
        socket_.bind(("127.0.0.1", 0))
        port = socket_.getsockname()[1]
    source = (
        "import gradio_app as app;"
        "app.backend_status=lambda:'Connected browser test fixture';"
        "app.build_ui().queue(default_concurrency_limit=1,max_size=8).launch("
        f"server_name='127.0.0.1',server_port={port},inbrowser=False,ssr_mode=False,css=app.H3_UI_CSS)"
    )
    with tempfile.TemporaryFile(mode="w+b") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", source],
            cwd=ROOT,
            stdout=log,
            stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            url = f"http://127.0.0.1:{port}"
            for _ in range(100):
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
                context = browser.new_context(viewport={"width": 1440, "height": 1000})
                page = context.new_page()
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(url, wait_until="domcontentloaded")
                card = page.locator(".h3-setup-card")
                page.locator('.h3-setup-card[data-settings-ready="true"]').wait_for()
                preset = page.locator(".h3-run-panel")
                preset.get_by_label("Quality", exact=True).first.check()
                expect(card).to_contain_text("Turbo · 8 steps")
                expect(card).not_to_contain_text("Modified")
                expect(card).to_contain_text("Base model: Speed")
                page.get_by_label("Prompt", exact=True).press_sequentially(
                    "A quiet lake at sunrise", delay=15
                )
                page.get_by_label("Prompt", exact=True).press("Tab")
                expect(
                    page.get_by_role("button", name="Generate video", exact=True)
                ).to_be_enabled(timeout=15000)
                page.get_by_text("Output essentials", exact=True).click()
                steps = (
                    page.get_by_text("Steps", exact=True)
                    .locator('xpath=ancestor::div[contains(@class,"block")][1]')
                    .locator('input[type="number"]')
                )
                steps.fill("10")
                steps.press("Tab")
                expect(card).to_contain_text("Turbo · 10 steps")
                expect(card).to_contain_text("Modified")
                page.get_by_label("Normal", exact=True).check()
                expect(card).to_contain_text("Normal · 20 steps")
                page.get_by_label("Turbo", exact=True).check()
                expect(card).to_contain_text("Turbo · 10 steps")
                page.get_by_role("button", name="Restore preset settings").click()
                expect(card).to_contain_text("Turbo · 8 steps")
                expect(card).not_to_contain_text("Modified")
                # Persist a genuine override and verify another session starts clean.
                steps.fill("11")
                steps.press("Tab")
                expect(card).to_contain_text("Turbo · 11 steps")
                page.wait_for_function(
                    "Object.keys(localStorage).some(k=>k.includes('minimax-h3:settings:v3'))"
                )
                # Wait for the save triggered after the transition to complete.
                page.wait_for_timeout(500)
                page.reload(wait_until="domcontentloaded")
                page.locator('.h3-setup-card[data-settings-ready="true"]').wait_for()
                expect(card).to_contain_text("Turbo · 11 steps")
                expect(card).to_contain_text("Quality")
                second = browser.new_context()
                second_page = second.new_page()
                second_page.goto(url, wait_until="domcontentloaded")
                second_page.locator(
                    '.h3-setup-card[data-settings-ready="true"]'
                ).wait_for()
                expect(second_page.locator(".h3-setup-card")).to_contain_text(
                    "Turbo · 4 steps"
                )
                second.close()
                # Audio retains the native-refinement preference for the next video.
                page.get_by_label("Audio", exact=True).check()
                expect(card).to_contain_text("Audio · 5 seconds")
                expect(card).not_to_contain_text("H3 output")
                page.get_by_label("Video", exact=True).check()
                expect(card).to_contain_text("Native 2× refinement")
                page.get_by_text(
                    "Performance & sampling (advanced)", exact=True
                ).click()
                page.get_by_label("Sol-Attn", exact=True).check()
                expect(page.get_by_text("Sol-Attn tau", exact=True)).to_be_visible()
                page.get_by_label("SLA", exact=True).check()
                expect(page.get_by_text("Sol-Attn tau", exact=True)).not_to_be_visible()
                # Layout must keep result metadata inside the two-column H3 row.
                artifact = ROOT / ".cache" / "ui-review"
                artifact.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(artifact / "desktop.png"), full_page=True)
                page.set_viewport_size({"width": 390, "height": 844})
                page.screenshot(path=str(artifact / "mobile.png"), full_page=True)
                assert not page.evaluate(
                    "document.documentElement.scrollWidth > innerWidth + 2"
                ), "Horizontal overflow"
                assert not errors, errors
                browser.close()
                print(
                    "Browser acceptance passed: preset, override, reset, mode memory, reload, session isolation, conditional controls and narrow layout."
                )
        except Exception:
            log.seek(0)
            print(log.read().decode("utf-8", errors="replace")[-10000:])
            raise
        finally:
            process.terminate()
            process.wait(timeout=15)


if __name__ == "__main__":
    run()
