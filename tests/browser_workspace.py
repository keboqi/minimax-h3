"""Workspace browser acceptance without a GPU or model downloads."""

import json
import re
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
    env = {key: value for key, value in os.environ.items() if key != "H3_UI_LAYOUT"}
    env["HF_HUB_OFFLINE"] = "1"
    import tempfile

    state = tempfile.TemporaryDirectory()
    env["H3_WORKSPACE_DIR"] = state.name
    env["GRADIO_OUTPUT_DIR"] = str(Path(state.name) / "media")
    env["COMFY_DIR"] = str(Path(state.name) / "comfy")
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
                phase = ["workspace startup"]
                page.on(
                    "pageerror",
                    lambda error: errors.append(
                        f"{phase[0]}: {error.message}\n{error.stack}"
                    ),
                )
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
                output_settings = page.locator("#h3-output-settings")
                expect(
                    output_settings.get_by_label("Aspect ratio", exact=True)
                ).to_be_disabled()
                expect(
                    output_settings.get_by_label("Size", exact=True)
                ).to_be_disabled()
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
                expect(
                    h3.get_by_text("Model & generation (advanced)", exact=True)
                ).to_have_count(0)
                expect(
                    h3.get_by_role(
                        "slider", name="range slider for Seconds", exact=True
                    )
                ).to_be_visible()
                expect(h3.get_by_text("Base model", exact=True)).to_have_count(0)
                page.locator("#h3-advanced-settings").get_by_text(
                    "Advanced settings", exact=True
                ).click()
                expect(
                    page.locator("#h3-advanced-settings").get_by_text(
                        "Generation", exact=True
                    )
                ).to_be_visible()
                expect(
                    page.locator("#h3-advanced-settings").get_by_text(
                        "Base model", exact=True
                    )
                ).to_be_visible()
                page.locator("#h3-advanced-settings").get_by_role(
                    "tab", name="Sampling & performance", exact=True
                ).click()
                expect(
                    page.locator("#h3-advanced-settings").get_by_text(
                        "Attention", exact=True
                    )
                ).to_be_visible()
                page.locator("#h3-advanced-settings").get_by_text(
                    "Advanced settings", exact=True
                ).click()
                expect(
                    page.locator('#h3-engine-tabs > div > [role="tablist"]')
                ).to_be_hidden()
                (ARTIFACTS / "engine-dom.html").write_text(
                    page.locator("#h3-engine-tabs").evaluate("(el)=>el.outerHTML"),
                    encoding="utf-8",
                )
                for width in (390, 768, 1280, 1440):
                    page.set_viewport_size({"width": width, "height": 1000})
                    page.evaluate("window.scrollTo(0, 0)")
                    page.screenshot(
                        path=str(ARTIFACTS / f"h3-{width}.png"), full_page=True
                    )
                    assert page.evaluate(
                        "document.documentElement.scrollWidth <= innerWidth + 2"
                    ), f"Horizontal overflow at {width}"
                    if width >= 1280:
                        assert (
                            h3.get_by_label("Prompt", exact=True).bounding_box()["y"]
                            < 500
                        )
                        assert (
                            abs(
                                h3.bounding_box()["y"]
                                - page.locator("#h3-preview").bounding_box()["y"]
                            )
                            < 8
                        )
                prompt = h3.get_by_label("Prompt", exact=True)
                generate = page.get_by_role("button", name="Generate video", exact=True)
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
                    page.wait_for_function(
                        "document.querySelector('.h3-primary-action').disabled"
                    )
                    response_times.append(time.perf_counter() - changed_at)
                    changed_at = time.perf_counter()
                    prompt.fill(f"Latency sample {index}")
                    page.wait_for_function(
                        "!document.querySelector('.h3-primary-action').disabled"
                    )
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
                first_row = page.locator(f'[data-job-id="{captured[0]["id"]}"]')
                first_row.click()
                expect(page.get_by_label("Job", exact=True)).to_have_value(
                    re.compile(captured[0]["id"][:8])
                )
                expect(first_row).to_have_attribute("aria-selected", "true")
                expect(
                    page.get_by_role("button", name="Cancel selected job", exact=True)
                ).to_be_enabled()
                from h3_app.contracts import GENERATION_FIELDS

                prompt_index = GENERATION_FIELDS.index("prompt") + 1
                assert [job["values"][prompt_index] for job in captured] == [
                    "Request A",
                    "Request B fail",
                ], captured
                requests.post(url + "/test/release-a", timeout=2).raise_for_status()
                # The second request remains its own snapshot after the form edit.
                expect(page.locator(".h3-job-table")).to_contain_text(
                    "failed", timeout=20000
                )
                captured = requests.get(url + "/test/jobs", timeout=2).json()
                failed = captured[1]
                assert (
                    captured[0]["state"] == "completed" and failed["state"] == "failed"
                ), captured
                # A selected running job updates its actions without reselecting.
                expect(
                    page.get_by_role("button", name="Cancel selected job", exact=True)
                ).to_be_disabled()
                # Native dropdown chooses the failed job by its visible stable ID.
                failed_row = page.locator(f'[data-job-id="{failed["id"]}"]')
                failed_row.focus()
                failed_row.press("Enter")
                expect(page.get_by_label("Job", exact=True)).to_have_value(
                    re.compile(failed["id"][:8])
                )
                expect(page.locator(".h3-job-details").first).to_contain_text(
                    "Fixture variant failed"
                )
                page.get_by_text("Retry & recovery", exact=True).click()
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
                expect(outsider.get_by_text("No jobs yet", exact=True)).to_be_visible()
                expect(outsider.locator(".h3-job-table")).to_have_count(0)
                other.close()
                # Explicit project saving retains content, while recovery keys
                # reconnect a new browser owner without sharing other owners.
                page.get_by_text("Saved projects", exact=True).click()
                page.get_by_label("Project name", exact=True).fill("Browser project")
                page.get_by_role(
                    "button", name="Save prompt, settings & sources", exact=True
                ).click()
                expect(page.get_by_label("Saved project", exact=True)).to_have_value(
                    "Browser project"
                )
                page.get_by_text("Browser ownership and recovery", exact=True).click()
                page.get_by_role(
                    "button", name="Show my recovery key", exact=True
                ).click()
                key_field = page.get_by_label("Browser recovery key", exact=True)
                expect(key_field).not_to_be_empty()
                recovery_key = key_field.input_value()
                restored_context = browser.new_context()
                restore_response = restored_context.request.post(
                    url + "/workspace/owner/recover", data={"key": recovery_key}
                )
                assert restore_response.ok, restore_response.text()
                recovered_page = restored_context.new_page()
                recovered_page.goto(url, wait_until="domcontentloaded")
                recovered_page.get_by_role("tab", name="Jobs", exact=True).click()
                expect(recovered_page.locator(".h3-job-table")).to_contain_text(
                    failed["id"][:8]
                )
                recovered_page.get_by_text("Saved projects", exact=True).click()
                recovered_page.get_by_role(
                    "button", name="Refresh projects", exact=True
                ).click()
                recovered_page.get_by_label("Saved project", exact=True).click()
                expect(
                    recovered_page.get_by_role(
                        "option", name="Browser project", exact=True
                    )
                ).to_be_visible()
                restored_context.close()
                # Every engine accepts an owned immutable request without a GPU.
                for task, label, field, action in (
                    ("Video", "LTX 2.5", "Positive prompt", "Generate with LTX-2.5"),
                    (
                        "Image",
                        "Qwen Image 2.1",
                        "Prompt / edit instruction",
                        "Generate with Qwen Image 2.1",
                    ),
                    (
                        "Audio / Music",
                        "MiniMax Music 3",
                        "Music caption",
                        "Generate with Music 3",
                    ),
                    ("Audio / Music", "YuE2", "Style prompt", "Generate with YuE2"),
                ):
                    phase[0] = f"engine switch: {label}"
                    page.get_by_role("tab", name="Create", exact=True).click()
                    page.locator(".h3-task-picker").get_by_label(
                        task, exact=True
                    ).check()
                    page.get_by_label("Engine", exact=True).click()
                    page.get_by_role("option", name=label, exact=True).click()
                    form = page.locator('#h3-engine-tabs > [role="tabpanel"]:visible')
                    form.get_by_label(field, exact=True).fill("Fixture request")
                    form.get_by_role("button", name=action, exact=True).click()
                    expect(
                        form.get_by_label("Generation progress", exact=True)
                    ).to_have_value(re.compile("Fixture .* completed"), timeout=15000)
                assert {
                    job["family"]
                    for job in requests.get(url + "/test/jobs", timeout=2).json()
                } == {"h3", "ltx", "qwen_image21", "music", "yue2"}
                # One filtered gallery drives preview, shared annotations and comparison.
                phase[0] = "Media image selection"
                page.get_by_role("tab", name="Media", exact=True).click()
                page.locator("#h3-library-kind").get_by_label(
                    "Image", exact=True
                ).check()
                expect(
                    page.locator("#generated-video-gallery .thumbnail-item")
                ).to_have_count(2)
                page.get_by_role("button", name="Search library", exact=True).click()
                page.get_by_text("Tags & lineage", exact=True).click()
                page.locator("#generated-video-gallery .thumbnail-item").filter(
                    has_text="alpha.png"
                ).click()
                expect(
                    page.get_by_label("Tags (comma separated)", exact=True)
                ).to_have_value("")
                page.get_by_role("button", name="Add to compare A", exact=True).click()
                expect(page.locator("#h3-compare-a")).to_contain_text("alpha.png")
                expect(
                    page.get_by_role(
                        "button", name="Compare selected outputs", exact=True
                    )
                ).to_be_disabled()
                page.get_by_role("button", name="Add to compare B", exact=True).click()
                expect(page.locator("body")).to_contain_text(
                    "Choose a different thumbnail."
                )
                expect(page.locator("#h3-compare-b img")).to_have_count(0)
                page.get_by_label("Tags (comma separated)", exact=True).fill(
                    "browser-test"
                )
                page.get_by_label("Favorite", exact=True).check()
                page.get_by_role(
                    "button", name="Save asset annotations", exact=True
                ).click()
                expect(page.locator("body")).to_contain_text("Asset annotations saved.")
                page.get_by_label("Search media", exact=True).fill("browser-test")
                page.get_by_label("Favorites only", exact=True).check()
                page.get_by_role("button", name="Search library", exact=True).click()
                expect(page.locator("body")).to_contain_text("1 matching assets")
                expect(
                    page.locator("#generated-video-gallery .thumbnail-item")
                ).to_have_count(1)
                expect(page.locator("#generated-video-gallery")).to_contain_text(
                    "alpha.png"
                )
                expect(page.locator("#h3-compare-a")).to_contain_text("alpha.png")
                page.get_by_label("Search media", exact=True).fill("missing-test-asset")
                phase[0] = "Media empty search"
                page.get_by_role("button", name="Search library", exact=True).click()
                expect(
                    page.locator("#generated-video-gallery .thumbnail-item")
                ).to_have_count(0)
                expect(page.locator("#h3-compare-a")).to_contain_text("alpha.png")
                expect(
                    page.get_by_role("button", name="Add to compare B", exact=True)
                ).to_be_disabled()
                expect(
                    page.get_by_label("Tags (comma separated)", exact=True)
                ).to_have_value("")
                page.get_by_label("Search media", exact=True).fill("")
                phase[0] = "Media restore search and image comparison"
                page.get_by_label("Favorites only", exact=True).uncheck()
                page.get_by_role("button", name="Search library", exact=True).click()
                for slot, name in (("A", "alpha.png"), ("B", "beta.png")):
                    page.locator("#generated-video-gallery .thumbnail-item").filter(
                        has_text=name
                    ).click()
                    expect(
                        page.get_by_role(
                            "button", name=f"Add to compare {slot}", exact=True
                        )
                    ).to_be_enabled()
                    page.get_by_role(
                        "button", name=f"Add to compare {slot}", exact=True
                    ).click()
                    expect(page.locator(f"#h3-compare-{slot.lower()}")).to_contain_text(
                        name
                    )
                page.get_by_role(
                    "button", name="Compare selected outputs", exact=True
                ).click()
                expect(
                    page.get_by_text("Image comparison A / B", exact=True)
                ).to_be_visible()
                page.locator("#h3-library-kind").get_by_label(
                    "Video", exact=True
                ).check()
                phase[0] = "Media video comparison"
                expect(
                    page.locator("#generated-video-gallery .thumbnail-item")
                ).to_have_count(2)
                for slot, name in (("A", "alpha.mp4"), ("B", "beta.mp4")):
                    page.locator("#generated-video-gallery .thumbnail-item").filter(
                        has_text=name
                    ).click()
                    expect(
                        page.get_by_role(
                            "button", name=f"Add to compare {slot}", exact=True
                        )
                    ).to_be_enabled()
                    page.get_by_role(
                        "button", name=f"Add to compare {slot}", exact=True
                    ).click()
                    expect(page.locator(f"#h3-compare-{slot.lower()}")).to_contain_text(
                        name
                    )
                    if slot == "A":
                        expect(page.locator("#h3-compare-b img")).to_have_count(0)
                        expect(
                            page.locator(".h3-compare-status").first
                        ).to_contain_text("Started a new video pair")
                        expect(
                            page.get_by_role(
                                "button", name="Compare selected outputs", exact=True
                            )
                        ).to_be_disabled()
                page.get_by_role(
                    "button", name="Compare selected outputs", exact=True
                ).click()
                expect(page.locator("[data-state]")).to_contain_text(
                    "Shared timeline: 1.00 seconds"
                )
                assert page.locator("[data-video-a]").evaluate("v => v.muted")
                assert page.locator("[data-video-b]").evaluate("v => v.muted")
                page.get_by_role("button", name="Play both", exact=True).click()
                expect(
                    page.get_by_role("button", name="Pause both", exact=True)
                ).to_be_visible()
                page.get_by_role("button", name="Pause both", exact=True).click()
                page.get_by_role("button", name="Clear B", exact=True).click()
                expect(page.locator("#h3-compare-b img")).to_have_count(0)
                expect(
                    page.get_by_role(
                        "button", name="Compare selected outputs", exact=True
                    )
                ).to_be_disabled()
                expect(page.locator("[data-video-a]")).to_have_count(0)
                page.get_by_role("button", name="Add to compare B", exact=True).click()
                page.get_by_role(
                    "button", name="Compare selected outputs", exact=True
                ).click()
                expect(page.locator("[data-state]")).to_contain_text(
                    "Shared timeline: 1.00 seconds"
                )
                page.screenshot(path=str(ARTIFACTS / "media.png"), full_page=True)
                for tab_name in ("Media", "Jobs"):
                    page.get_by_role("tab", name=tab_name, exact=True).click()
                    for width in (390, 1440):
                        page.set_viewport_size({"width": width, "height": 1000})
                        expect(
                            page.locator(
                                ".h3-gallery-shell"
                                if tab_name == "Media"
                                else ".h3-jobs-shell"
                            ).first
                        ).to_be_visible()
                        page.screenshot(
                            path=str(ARTIFACTS / f"{tab_name.lower()}-{width}.png"),
                            full_page=True,
                        )
                        assert page.evaluate(
                            "document.documentElement.scrollWidth <= innerWidth + 2"
                        ), f"{tab_name} overflows at {width}px"
                page.emulate_media(color_scheme="dark", reduced_motion="reduce")
                page.goto(url + "/?__theme=dark", wait_until="domcontentloaded")
                page.locator('.h3-setup-card[data-settings-ready="true"]').wait_for(
                    state="attached"
                )
                page.locator(".h3-task-picker").get_by_label(
                    "Video", exact=True
                ).check()
                page.get_by_label("Engine", exact=True).click()
                page.get_by_role("option", name="MiniMax H3", exact=True).click()
                expect(page.locator("#h3-composer")).to_be_visible()
                for width in (390, 1440):
                    page.set_viewport_size({"width": width, "height": 1000})
                    page.screenshot(
                        path=str(ARTIFACTS / f"h3-dark-{width}.png"), full_page=True
                    )
                    assert page.evaluate(
                        "document.documentElement.scrollWidth <= innerWidth + 2"
                    )
                for tab_name in ("Media", "Jobs"):
                    page.get_by_role("tab", name=tab_name, exact=True).click()
                    for width in (390, 1440):
                        page.set_viewport_size({"width": width, "height": 1000})
                        page.screenshot(
                            path=str(
                                ARTIFACTS / f"{tab_name.lower()}-dark-{width}.png"
                            ),
                            full_page=True,
                        )
                        assert page.evaluate(
                            "document.documentElement.scrollWidth <= innerWidth + 2"
                        ), f"{tab_name} dark overflows at {width}px"
                page.get_by_role("tab", name="Create", exact=True).click()
                page.set_viewport_size({"width": 1280, "height": 1000})
                page.evaluate("document.documentElement.style.zoom = '2'")
                assert page.evaluate(
                    "document.documentElement.scrollWidth <= innerWidth + 2"
                ), "Overflow at 200% zoom"
                page.screenshot(
                    path=str(ARTIFACTS / "h3-200-percent.png"), full_page=True
                )
                page.evaluate("document.documentElement.style.zoom = '1'")
                page.get_by_role("tab", name="Create", exact=True).focus()
                page.keyboard.press("Tab")
                assert page.evaluate("document.activeElement !== document.body")
                assert page.locator('[role="status"]').count() > 0
                assert sorted(response_times)[-1] < 0.25, response_times
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
