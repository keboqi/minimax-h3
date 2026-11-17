"""Focused browser smoke for clock-driven Illustrate effects, using local fake images."""
import base64
import json
from io import BytesIO
from pathlib import Path
import socket
from threading import Thread
import time
import wave

from playwright.sync_api import expect, sync_playwright
import uvicorn

from tests.test_drama3_page_video import API, NAME, PLAN, Drama3PageVideoTests


def run():
    fixture = Drama3PageVideoTests()
    fixture.setUp()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(fixture.client.app, log_level="error"))
    thread = Thread(target=lambda: server.run(sockets=[listener]), daemon=True)
    try:
        fixture.ok(fixture.client.post(f"{API}/project/video/reply", params=NAME, json={"reply": json.dumps(PLAN)}))
        list(fixture.fake_export(fixture.mix.name, "qwen", "ink wash", 2, True, "landscape", '"all"'))
        audio = BytesIO()
        with wave.open(audio, "wb") as wav:
            wav.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
            wav.writeframes(b"\0\0" * 8000 * 8)
        thread.start()
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        with sync_playwright() as p:
            chrome = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
            browser = p.chromium.launch(headless=True, **({"executable_path": str(chrome)} if chrome.exists() else {}))
            page = browser.new_page(viewport={"width": 1440, "height": 1000}, reduced_motion="no-preference")
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.route("**/soundstage/api/audio?*", lambda route: route.fulfill(body=audio.getvalue(), content_type="audio/wav"))
            page.goto(f"http://127.0.0.1:{port}/soundstage/", wait_until="networkidle")
            page.locator("#step-illustrate").click()
            expect(page.locator("#still-back")).to_be_visible()
            assert page.locator("#shot-inspector select").count() == 2, (errors, page.locator("#screen-illustrate").inner_text())
            with page.expect_response(lambda response: "/project/video/shot" in response.url and response.request.method == "PATCH"):
                page.get_by_label("Camera movement", exact=True).select_option("push_in")
            expect(page.get_by_label("Camera movement", exact=True)).to_have_value("push_in")
            page.wait_for_load_state("networkidle")
            assert fixture.project()["video"]["plan"]["rows"][0]["effects"] == ["push_in"], "camera did not save"
            with page.expect_response(lambda response: "/project/video/shot" in response.url and response.request.method == "PATCH"):
                page.get_by_label("Atmosphere", exact=True).select_option("rain")
            expect(page.get_by_label("Atmosphere", exact=True)).to_have_value("rain")
            page.wait_for_load_state("networkidle")
            actual = fixture.project()["video"]["plan"]["rows"][0]["effects"]
            assert actual == ["push_in", "rain"], (actual, page.locator("body").inner_text(), errors)
            page.evaluate("""async source => {
              const audio = document.getElementById('mix-audio');
              await new Promise(resolve => { audio.addEventListener('canplaythrough', resolve, {once: true}); audio.src = source; audio.load(); });
            }""", "data:audio/wav;base64," + base64.b64encode(audio.getvalue()).decode())
            # Seek the actual player while paused; every frame must reconstruct deterministically.
            frame = page.evaluate("""async () => {
              const audio = document.getElementById('mix-audio');
              const canvas = document.getElementById('still-effects');
              const draw = async t => {
                await new Promise(resolve => { audio.addEventListener('seeked', resolve, {once: true}); audio.currentTime = t; });
                return {image: canvas.toDataURL(), transform: document.getElementById('still-back').style.transform,
                  time: audio.currentTime, duration: audio.duration, clock: document.getElementById('clock').textContent};
              };
              const a = await draw(1), b = await draw(2), c = await draw(1);
              return {moves: a.image !== b.image && a.transform !== b.transform,
                seeks: a.image === c.image && a.transform === c.transform,
                transforms: [a.transform, b.transform, c.transform], times: [a.time, b.time], duration: a.duration,
                clock: a.clock, imageChanged: a.image !== b.image};
            }""")
            assert frame["moves"] and frame["seeks"], frame
            with page.expect_response(lambda response: response.url.split("?")[0].endswith("/project/video") and response.request.method == "PATCH"):
                page.get_by_label("Visual effects", exact=True).uncheck()
            expect(page.locator("#still-back")).to_have_css("transform", "none")
            page.wait_for_load_state("networkidle")
            assert fixture.project()["video"]["effects_enabled"] is False
            with page.expect_response(lambda response: response.url.split("?")[0].endswith("/project/video") and response.request.method == "PATCH"):
                page.get_by_label("Visual effects", exact=True).check()
            page.wait_for_load_state("networkidle")
            with page.expect_response(lambda response: "/project/video/shot" in response.url and response.request.method == "PATCH"):
                page.get_by_label("Atmosphere", exact=True).select_option("lightning")
            expect(page.get_by_label("Atmosphere", exact=True)).to_have_value("lightning")
            page.wait_for_load_state("networkidle")
            page.emulate_media(reduced_motion="reduce")
            expect(page.locator("#still-back")).to_have_css("transform", "none")
            expect(page.get_by_label("Visual effects", exact=True)).not_to_be_checked()
            expect(page.locator("#effects-motion-hint")).to_be_visible()
            with page.expect_response(lambda response: response.url.split("?")[0].endswith("/project/video") and response.request.method == "PATCH"):
                page.get_by_label("Visual effects", exact=True).check()
            expect(page.locator("#still-back")).not_to_have_css("transform", "none")
            assert fixture.project()["video"]["effects_override_reduced_motion"] is True
            # The explicit choice survives a reload with reduced motion still enabled.
            page.reload(wait_until="networkidle")
            expect(page.get_by_label("Visual effects", exact=True)).to_be_checked()
            expect(page.locator("#still-back")).not_to_have_css("transform", "none")
            with page.expect_response(lambda response: response.url.split("?")[0].endswith("/project/video") and response.request.method == "PATCH"):
                page.get_by_label("Visual effects", exact=True).uncheck()
            assert fixture.project()["video"]["effects_override_reduced_motion"] is False
            expect(page.locator("#still-back")).to_have_css("transform", "none")
            page.emulate_media(reduced_motion="no-preference")
            expect(page.locator("#still-back")).to_have_css("transform", "none")
            assert not errors, errors
            browser.close()
        print("PASS: effect controls, persistence, canvas animation, seeking, reduced motion, module loading")
    finally:
        server.should_exit = True
        if thread.is_alive():
            thread.join(timeout=10)
        listener.close()
        fixture.doCleanups()


if __name__ == "__main__":
    run()
