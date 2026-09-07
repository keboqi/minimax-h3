"""Run FL2VA voice browser acceptance: python tests/browser_voice_refs.py.

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
        f"server_name='127.0.0.1',server_port={port},inbrowser=False,ssr_mode=False,css=app.H3_SETUP_CSS)"
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
                expect.set_options(timeout=15000)
                prompt = page.get_by_label("Prompt", exact=True)
                prompt.fill('The woman uses the voice timbre of <Audio 1> and says, "Hello."')
                generate = page.get_by_role("button", name="Generate video", exact=True)
                page.get_by_text("Model and memory (advanced)", exact=True).click()
                bridge = page.get_by_label("Semantic Bridge (experimental)", exact=True)
                bridge.check()
                expect(card).to_contain_text("Experimental v1")
                section = page.get_by_text("Optional voice references (experimental)", exact=True)
                expect(section).not_to_be_visible()
                page.get_by_label("First / last frame", exact=True).check()
                expect(section).to_be_visible()
                section.click()
                import io
                import wave
                audio_bytes = io.BytesIO()
                with wave.open(audio_bytes, "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(32000)
                    wav.writeframes(b"\x00\x00" * 64000)
                page.locator('#fl2va-voice-1 input[type="file"]').set_input_files(
                    {"name": "voice.wav", "mimeType": "audio/wav", "buffer": audio_bytes.getvalue()}
                )
                expect(card).to_contain_text("1 FL2VA voice reference(s)")
                expect(bridge).to_be_disabled()
                # A voice sample alone never substitutes for the first/last frame.
                expect(generate).to_be_disabled()
                page.get_by_label("Reference media", exact=True).check()
                expect(section).not_to_be_visible()
                expect(card).not_to_contain_text("FL2VA voice reference(s)")
                expect(generate).to_be_disabled()
                # Ref2VA's own upload controls remain empty.
                page.get_by_text("Reference audio · up to 3", exact=True).click()
                expect(page.locator('audio')).to_have_count(0)
                page.get_by_label("First / last frame", exact=True).check()
                expect(section).to_be_visible()
                expect(card).to_contain_text("1 FL2VA voice reference(s)")
                expect(page.locator('#fl2va-voice-1 audio').first).to_be_attached()
                page.get_by_label("Text to video", exact=True).check()
                expect(section).not_to_be_visible()
                expect(bridge).to_be_enabled()
                expect(bridge).to_be_checked()
                expect(generate).to_be_enabled()
                # Desktop and narrow layouts with the new voice section open.
                page.get_by_label("First / last frame", exact=True).check()
                expect(section).to_be_visible()
                artifact = ROOT / ".cache" / "ui-review"
                artifact.mkdir(parents=True, exist_ok=True)
                section.scroll_into_view_if_needed()
                page.screenshot(path=str(artifact / "fl2va-voices-desktop.png"))
                page.set_viewport_size({"width": 390, "height": 844})
                section.scroll_into_view_if_needed()
                page.screenshot(path=str(artifact / "fl2va-voices-mobile.png"))
                assert not page.evaluate("document.documentElement.scrollWidth > innerWidth + 2"), "Horizontal overflow"
                assert not errors, errors
                browser.close()
                print("FL2VA voice browser acceptance passed: upload, mode isolation, retained samples, keyframe requirement, bridge preference, and narrow layout.")
        except Exception:
            log.seek(0)
            print(log.read().decode("utf-8", errors="replace")[-10000:])
            raise
        finally:
            process.terminate()
            process.wait(timeout=15)


if __name__ == "__main__":
    run()
