"""Workspace browser acceptance without a GPU or model downloads."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import requests
from PIL import Image
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARTIFACTS = ROOT / ".cache/ui-redesign/workspace"


def run():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {**os.environ, "H3_UI_LAYOUT": "workspace", "HF_HUB_OFFLINE": "1"}
    with (ARTIFACTS / "server.log").open("wb") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", "-m", "tests.workspace_fixture", str(port)],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            url = f"http://127.0.0.1:{port}"
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError((ARTIFACTS / "server.log").read_text())
                try:
                    if requests.get(url + "/test/jobs", timeout=1).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.2)
            else:
                raise TimeoutError("Workspace server unavailable")
            with sync_playwright() as p:
                executable = os.getenv("H3_BROWSER_EXECUTABLE")
                chrome = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
                browser = p.chromium.launch(
                    headless=True,
                    **(
                        {"executable_path": executable or str(chrome)}
                        if executable or chrome.exists()
                        else {}
                    ),
                )
                context = browser.new_context(viewport={"width": 1440, "height": 1000})
                page = context.new_page()
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                loaded_at = time.perf_counter()
                page.goto(url, wait_until="domcontentloaded")
                page.locator('.h3-setup-card[data-settings-ready="true"]').wait_for()
                time_to_ready = time.perf_counter() - loaded_at
                # Task-filtered engine restoration includes choices unavailable
                # in the default Video task. Passwords and prompts stay blank.
                page.locator(".h3-task-picker").get_by_label(
                    "Image", exact=True
                ).check()
                page.get_by_label("Engine", exact=True).click()
                page.get_by_role("option", name="Qwen Image 2.1", exact=True).click()
                qwen = page.locator('#h3-engine-tabs > [role="tabpanel"]:visible')
                qwen.get_by_label("Seed (-1 random)", exact=True).fill("123")
                qwen.get_by_label("Seed (-1 random)", exact=True).press("Tab")
                qwen.get_by_text("Prompt writer / enhancer", exact=True).click()
                qwen.get_by_label("Temporary Lightning API key", exact=True).fill(
                    "fixture-secret"
                )
                page.wait_for_timeout(600)
                page.reload(wait_until="domcontentloaded")
                qwen = page.locator('#h3-engine-tabs > [role="tabpanel"]:visible')
                expect(qwen.get_by_label("Seed (-1 random)", exact=True)).to_have_value(
                    "123", timeout=15000
                )
                qwen.get_by_text("Prompt writer / enhancer", exact=True).click()
                expect(
                    qwen.get_by_label("Temporary Lightning API key", exact=True)
                ).to_have_value("")
                page.locator(".h3-task-picker").get_by_label(
                    "Video", exact=True
                ).check()
                # Hide engine-tab navigation using supported semantic roles.
                h3 = page.locator("#h3-composer")
                expect(h3.get_by_label("Prompt", exact=True)).to_be_visible()
                # Input-derived Image output must remain authoritative with the
                # workspace canvas controls present and closed expert controls.
                first = ARTIFACTS / "first-fixture.png"
                replacement = ARTIFACTS / "replacement-fixture.png"
                Image.new("RGB", (768, 1152), "navy").save(first)
                Image.new("RGB", (640, 960), "teal").save(replacement)
                page.locator(".h3-task-picker").get_by_label(
                    "Image", exact=True
                ).check()
                page.get_by_label("Engine", exact=True).click()
                page.get_by_role("option", name="MiniMax H3", exact=True).click()
                h3.get_by_label("First / last frame", exact=True).check()
                page.locator('#first-frame-image input[type="file"]').set_input_files(
                    str(first)
                )
                expect(page.locator(".h3-setup-card")).to_contain_text("768×1152")
                expect(h3.get_by_label("Aspect ratio", exact=True)).to_be_disabled()
                expect(h3.get_by_label("Size", exact=True)).to_be_disabled()
                page.locator(".h3-task-picker").get_by_label(
                    "Video", exact=True
                ).check()
                h3.get_by_label("Reference media", exact=True).check()
                h3.get_by_role("button", name="Add image reference", exact=True).click()
                picture = h3.get_by_text("Picture 2", exact=True).locator(
                    'xpath=ancestor::div[contains(@class,"block")][1]'
                )
                picture.locator('input[type="file"]').set_input_files(str(first))
                expect(h3).to_contain_text("<Picture 2> → engine <Picture 1>")
                picture.locator('input[type="file"]').set_input_files(str(replacement))
                expect(h3).to_contain_text("replaced: confirm before using this tag")
                h3.get_by_role(
                    "button", name="Confirm replaced references", exact=True
                ).click()
                expect(h3).not_to_contain_text(
                    "replaced: confirm before using this tag"
                )
                h3.get_by_label("Text to video", exact=True).check()
                expect(page.locator('#h3-engine-tabs [role="tablist"]')).to_be_hidden()
                (ARTIFACTS / "engine-dom.html").write_text(
                    page.locator("#h3-engine-tabs").evaluate("(el)=>el.outerHTML"),
                    encoding="utf-8",
                )
                for width in (390, 768, 1280, 1440):
                    page.set_viewport_size({"width": width, "height": 1000})
                    page.screenshot(
                        path=str(ARTIFACTS / f"h3-{width}.png"), full_page=True
                    )
                    assert page.evaluate(
                        "document.documentElement.scrollWidth <= innerWidth + 2"
                    ), f"Horizontal overflow at {width}"
                prompt = h3.get_by_label("Prompt", exact=True)
                generate = h3.get_by_role("button", name="Generate video", exact=True)
                prompt.focus()
                page.keyboard.press("Tab")
                focused = page.locator(":focus")
                expect(focused).to_be_visible()
                assert (
                    focused.evaluate(
                        "el => parseFloat(getComputedStyle(el).outlineWidth)"
                    )
                    >= 2
                )
                h3.get_by_text("Prompt writer / enhancer", exact=True).click()
                prompt.fill("Preview original")
                h3.get_by_role(
                    "button", name="Generate / enhance prompt", exact=True
                ).click()
                expect(h3.get_by_label("Suggested prompt", exact=True)).to_have_value(
                    "Preview original Enhanced fixture."
                )
                expect(prompt).to_have_value("Preview original")
                prompt.fill("Edited while previewing")
                h3.get_by_role(
                    "button", name="Accept suggested prompt", exact=True
                ).click()
                expect(prompt).to_have_value("Edited while previewing")
                expect(
                    h3.get_by_label("Prompt enhancer status", exact=True)
                ).to_have_value(
                    "The prompt or input media changed. Enhance the current draft again."
                )
                h3.get_by_role(
                    "button", name="Generate / enhance prompt", exact=True
                ).click()
                expect(h3.get_by_label("Suggested prompt", exact=True)).to_have_value(
                    "Edited while previewing Enhanced fixture."
                )
                h3.get_by_role(
                    "button", name="Accept suggested prompt", exact=True
                ).click()
                expect(prompt).to_have_value(
                    "Edited while previewing Enhanced fixture."
                )
                h3.get_by_role(
                    "button", name="Undo accepted prompt", exact=True
                ).click()
                expect(prompt).to_have_value("Edited while previewing")
                h3.get_by_text("Prompt writer / enhancer", exact=True).click()
                response_times = []
                for index in range(5):
                    changed_at = time.perf_counter()
                    prompt.fill("")
                    expect(generate).to_be_disabled()
                    response_times.append(time.perf_counter() - changed_at)
                    changed_at = time.perf_counter()
                    prompt.fill(f"Latency sample {index}")
                    expect(generate).to_be_enabled()
                    response_times.append(time.perf_counter() - changed_at)
                prompt.fill("Request A")
                expect(generate).to_be_enabled()
                generate.click()
                prompt.fill("Request B fail")
                generate.click()
                page.get_by_role("tab", name="Jobs", exact=True).click()
                expect(page.locator(".h3-job-table")).to_contain_text(
                    "queued", timeout=10000
                )
                captured = requests.get(url + "/test/jobs", timeout=2).json()
                assert len(captured) == 2, captured
                from h3_app.contracts import GENERATION_FIELDS

                prompt_index = GENERATION_FIELDS.index("prompt") + 1
                assert [job["values"][prompt_index] for job in captured] == [
                    "Request A",
                    "Request B fail",
                ], captured
                # The second request remains its own snapshot after the form edit.
                expect(page.locator(".h3-job-table")).to_contain_text(
                    "failed", timeout=20000
                )
                captured = requests.get(url + "/test/jobs", timeout=2).json()
                failed = captured[1]
                assert (
                    captured[0]["state"] == "completed" and failed["state"] == "failed"
                ), captured
                # Native dropdown chooses the failed job by its visible stable ID.
                page.get_by_label("Job", exact=True).click()
                page.get_by_role(
                    "option", name=f"{failed['id'][:8]} · h3 · failed", exact=True
                ).click()
                page.get_by_role(
                    "button", name="Retry failed variants", exact=True
                ).click()
                expect(page.locator(".h3-job-table")).to_contain_text(
                    "running", timeout=10000
                )
                captured = requests.get(url + "/test/jobs", timeout=2).json()
                assert captured[-1]["retry_of"] == failed["id"]
                assert captured[-1]["seeds"] == failed["seeds"]
                # A rejected queue handoff must become a terminal failed job.
                page.get_by_role("tab", name="Create", exact=True).click()
                prompt.fill("Rejected handoff")
                expect(generate).to_be_enabled()
                requests.post(url + "/test/queue/0", timeout=2).raise_for_status()
                generate.click()
                page.get_by_role("tab", name="Jobs", exact=True).click()
                expect(page.locator(".h3-job-table")).to_contain_text(
                    "failed", timeout=10000
                )
                for _ in range(30):
                    after_rejection = requests.get(url + "/test/jobs", timeout=2).json()
                    if after_rejection[-1]["state"] == "failed":
                        break
                    page.wait_for_timeout(100)
                assert after_rejection[-1]["state"] == "failed", after_rejection
                requests.post(url + "/test/queue/8", timeout=2).raise_for_status()
                # A separate browser session cannot see the first session's jobs.
                other = browser.new_context(viewport={"width": 390, "height": 1000})
                outsider = other.new_page()
                outsider.goto(url, wait_until="domcontentloaded")
                outsider.get_by_role("tab", name="Jobs", exact=True).click()
                expect(
                    outsider.get_by_text("No jobs in this session yet.", exact=True)
                ).to_be_visible()
                expect(outsider.locator(".h3-job-table")).to_have_count(0)
                other.close()
                page.screenshot(path=str(ARTIFACTS / "jobs.png"), full_page=True)
                page.emulate_media(color_scheme="dark", reduced_motion="reduce")
                page.goto(url + "/?__theme=dark", wait_until="domcontentloaded")
                page.locator('.h3-setup-card[data-settings-ready="true"]').wait_for()
                for width in (390, 1440):
                    page.set_viewport_size({"width": width, "height": 1000})
                    page.screenshot(
                        path=str(ARTIFACTS / f"h3-dark-{width}.png"), full_page=True
                    )
                    assert page.evaluate(
                        "document.documentElement.scrollWidth <= innerWidth + 2"
                    )
                assert not errors, errors
                browser.close()
                (ARTIFACTS / "results.json").write_text(
                    json.dumps(
                        {
                            "widths": [390, 768, 1280, 1440],
                            "jobs": captured,
                            "time_to_ready_seconds": time_to_ready,
                            "individual_edit_response_seconds": response_times,
                        },
                        indent=2,
                    )
                )
        finally:
            process.terminate()
            process.wait(timeout=15)
    print("Workspace browser checks passed; artifacts:", ARTIFACTS)


if __name__ == "__main__":
    run()
