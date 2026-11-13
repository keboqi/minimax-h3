"""Backend-free regressions for draft saves and generated images in the Image page."""

import io
from pathlib import Path
import time

from PIL import Image
from playwright.sync_api import expect, sync_playwright


def run():
    static = Path(__file__).resolve().parents[1] / "h3_ui/image/page"
    images = [{"path": f"{index}.png", "name": f"{index}.png", "url": f"/fixture/{index}.png"}
              for index in range(4)]
    state = dict(engine="qwen", mode="Text to image", editing=False, generate_only=False,
                 prompt="Bird", ui={}, options={}, references=[], fields=[], changed={},
                 preset=None, batch=None, edit_size=None, separate=None, layout=None,
                 width=1024, height=1024, seed=-1, guidance="", readiness={"ready": True, "message": "Ready"},
                 can_enhance=False, writer="", results={"images": images, "settings": ""})
    boot = dict(state=state, runs={}, gpu=True, modes=["Text to image", "Image edit"],
                engines=[{"id": "qwen", "label": "Qwen", "description": "Fixture"},
                         {"id": "flux3", "label": "FLUX", "description": "Fixture"}], sizes={}, groups=[])
    colors = ["red", "green", "blue", "yellow"]
    writes = []
    held = []
    delay_prompt = False

    def save_prompt(request):
        writes.append("prompt")
        body = request.request.post_data_json
        state["prompt"] = body["prompt"]
        if "layout" in body:
            assert body["layout"]["name"] == state["layout"]["name"]
            state["layout"]["value"] = body["layout"]["value"]
        request.fulfill(json={"state": state, "runs": {}})

    def route(request):
        path = request.request.url.split("test.local", 1)[1]
        if path == "/image/api/boot":
            request.fulfill(json=boot)
        elif path == "/image/api/prompt":
            if delay_prompt:
                held.append(request)
            else:
                save_prompt(request)
        elif path == "/image/api/settings":
            writes.append("settings")
            if state["engine"] == "flux3":
                assert state["layout"]["value"] == "A box around the bird"
            state["engine"] = request.request.post_data_json["changes"]["engine"]
            state["layout"] = (dict(name="flux3__layout", value="", hint="", placeholder="")
                               if state["engine"] == "flux3" else None)
            request.fulfill(json={"state": state, "runs": {}})
        elif path == "/image/api/empty":
            writes.append("empty")
            state.update(prompt="", references=[], mode="Text to image")
            request.fulfill(json={"state": state, "runs": {}})
        elif path.startswith("/fixture/"):
            index = int(Path(path).stem)
            data = io.BytesIO()
            Image.new("RGB", (40, 40), colors[index]).save(data, format="PNG")
            request.fulfill(body=data.getvalue(), content_type="image/png")
        elif path in ("/image/", "/image/app.js", "/image/app.css"):
            file = static / (Path(path).name if not path.endswith("/") else "index.html")
            content_type = {".html": "text/html", ".js": "application/javascript", ".css": "text/css"}[file.suffix]
            request.fulfill(body=file.read_bytes(), content_type=content_type)
        else:
            request.abort()

    with sync_playwright() as playwright:
        chrome = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
        browser = playwright.chromium.launch(executable_path=str(chrome) if chrome.exists() else None)
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.route("**/*", route)
        page.goto("http://test.local/image/")

        def switch_with_delayed_save(label, text, layout=None):
            nonlocal delay_prompt
            delay_prompt = True
            page.locator("#prompt").fill(text)
            if layout is not None:
                page.locator("#layout-box").evaluate("node => { node.open = true; }")
                page.locator("#layout").fill(layout)
            before = writes.count("settings")
            page.get_by_role("button", name=label, exact=True).click()
            # Hold blur saves while the settings click is processed.
            page.wait_for_timeout(100)
            assert held, "Expected a pending draft save"
            assert writes.count("settings") == before, "Settings overtook the draft save"
            expect(page.locator("#prompt")).to_have_value(text)
            if layout is not None:
                expect(page.locator("#layout")).to_have_value(layout)
            deadline = time.monotonic() + 5
            while writes.count("settings") == before and time.monotonic() < deadline:
                if held:
                    save_prompt(held.pop(0))
                page.wait_for_timeout(10)
            assert writes.count("settings") == before + 1
            delay_prompt = False
            expect(page.locator("#prompt")).to_have_value(text)
            assert state["prompt"] == text

        switch_with_delayed_save("FLUX", "A new prompt")
        switch_with_delayed_save("Qwen", "An edited prompt", "A box around the bird")
        page.reload()
        expect(page.locator("#prompt")).to_have_value("An edited prompt")
        for index in (0, 1, 2, 3, 0, 3, 1):
            page.get_by_role("button", name=f"Image {index + 1}", exact=True).click()
            picture = page.locator("#result-image")
            expect(picture).to_have_attribute("src", images[index]["url"])
            page.wait_for_function("() => { const img = document.getElementById('result-image'); return img.complete && img.naturalWidth > 0; }")
            pixel = picture.evaluate("""img => {
                const canvas = document.createElement('canvas');
                const context = canvas.getContext('2d');
                context.drawImage(img, 0, 0);
                return Array.from(context.getImageData(0, 0, 1, 1).data).slice(0, 3);
            }""")
            assert pixel == list(Image.new("RGB", (1, 1), colors[index]).getpixel((0, 0))), pixel
            expect(page.locator("#result-link")).to_have_attribute("href", images[index]["url"])
        page.locator("#prompt").fill("A draft to discard")
        page.get_by_role("button", name="Empty", exact=True).click()
        expect(page.locator("#prompt")).to_have_value("")
        assert writes[-1] == "empty", writes
        page.reload()
        expect(page.locator("#prompt")).to_have_value("")
        expect(page.locator("#result-image")).to_be_visible()
        assert not errors, errors
        browser.close()
    print("Image page: delayed draft saves, model switching, preview pixels and Empty passed.")


if __name__ == "__main__":
    run()
