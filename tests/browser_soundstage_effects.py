"""Focused browser smoke for clock-driven Illustrate effects, using local fake images."""
import base64
import json
from io import BytesIO
from pathlib import Path
import socket
import sys
from threading import Thread
import time
import wave

from PIL import Image
from playwright.sync_api import expect, sync_playwright
import uvicorn

from tests.test_drama3_page_video import API, NAME, PLAN, Drama3PageVideoTests


def check_webgpu_composite(page):
    sizes = page.evaluate("""async () => {
      const {renderWebGPU, ATMO_IDS, LIGHT_IDS} = await import('/soundstage/effects_webgpu.js');
      const host = document.createElement('canvas');
      const probe = document.createElement('div');
      probe.id = 'composite-probe';
      probe.style = 'position: fixed; top: 0; left: 0; z-index: 9999; width: 160px; height: 90px';
      probe.append(host);
      document.body.append(probe);
      async function still(color) {
        const source = document.createElement('canvas');
        source.width = 160;
        source.height = 90;
        const context = source.getContext('2d');
        context.fillStyle = color;
        context.fillRect(0, 0, 160, 90);
        const image = new Image();
        image.src = source.toDataURL();
        await image.decode();
        return image;
      }
      const back = await still('#f00'), front = await still('#0f0');
      renderWebGPU(host, ATMO_IDS.snow, 0, 1, 160, 90, {stillBack: back});
      const canvas = probe.querySelector('.still-effects-gpu');
      const overlay = canvas.width;
      renderWebGPU(host, 0, LIGHT_IDS.flash, 1, 160, 90, {
        stillBack: back, stillFront: front, fading: true, fadeMix: 0.5,
        backLightId: LIGHT_IDS.vignette, cueElapsed: 1,
      });
      return [overlay, canvas.width];
    }""")
    assert sizes == [160, 320], sizes
    frame = Image.open(BytesIO(page.locator("#composite-probe").screenshot())).convert("RGB")
    red, green, blue = frame.getpixel((4, 4))
    assert red < 100 and 125 <= green <= 130 and blue == 0, (red, green, blue)
    page.locator("#composite-probe").evaluate("node => node.remove()")


def run(*, webgpu=False):
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
            flags = ["--enable-unsafe-webgpu", "--use-webgpu-adapter=swiftshader", "--enable-unsafe-swiftshader"] if webgpu else []
            browser = p.chromium.launch(headless=True, args=flags,
                                        **({"executable_path": str(chrome)} if chrome.exists() else {}))
            page = browser.new_page(viewport={"width": 1440, "height": 1000}, reduced_motion="no-preference",
                                    device_scale_factor=2 if webgpu else 1)
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            def record_console_error(msg):
                if msg.type == "warning" and msg.text == "No available adapters.":
                    return
                if msg.type in ("error", "warning"):
                    errors.append(f"[{msg.type}] {msg.text}")

            page.on("console", record_console_error)
            page.route("**/soundstage/api/audio?*", lambda route: route.fulfill(body=audio.getvalue(), content_type="audio/wav"))
            page.goto(f"http://127.0.0.1:{port}/soundstage/", wait_until="networkidle")
            fullscreen = page.locator("#player-fullscreen")
            expect(fullscreen).to_have_count(0)
            page.locator("#step-illustrate").click()
            expect(page.locator("#still-back")).to_be_visible()
            inline_width = page.locator("#video-stage").evaluate("stage => stage.clientWidth")
            fullscreen.click()
            expect(page.locator("#player:fullscreen #still-back")).to_be_visible()
            expect(page.locator("#player:fullscreen #play")).to_be_visible()
            expect(page.locator("#player:fullscreen #video-caption")).to_contain_text("Plums!")
            assert page.locator("#video-stage").evaluate("stage => stage.clientWidth") > inline_width
            expect(fullscreen).to_have_text("Exit fullscreen")
            fullscreen.click()
            expect(fullscreen).to_have_text("Fullscreen")
            expect(page.locator("#player:fullscreen")).to_have_count(0)
            engine = page.locator("#effects-engine")
            expect(engine).to_have_text("Effects engine · Idle")
            expect(page.locator("#lanes .effect-lane")).to_have_count(0)
            assert page.locator("#shot-inspector select").count() == 3, (errors, page.locator("#screen-illustrate").inner_text())
            with page.expect_response(lambda response: "/project/video/shot" in response.url and response.request.method == "PATCH"):
                page.get_by_label("Camera movement", exact=True).select_option("push_in")
            expect(page.get_by_label("Camera movement", exact=True)).to_have_value("push_in")
            page.wait_for_load_state("networkidle")
            assert fixture.project()["video"]["plan"]["rows"][0]["effects"] == ["push_in"], "camera did not save"
            expect(engine).to_have_text("Effects engine · Camera only")
            with page.expect_response(lambda response: "/project/video/shot" in response.url and response.request.method == "PATCH"):
                page.get_by_label("Atmosphere", exact=True).select_option("rain")
            expect(page.get_by_label("Atmosphere", exact=True)).to_have_value("rain")
            page.wait_for_load_state("networkidle")
            actual = fixture.project()["video"]["plan"]["rows"][0]["effects"]
            assert actual == ["push_in", "rain"], (actual, page.locator("body").inner_text(), errors)
            effect = page.locator("#lanes .effect-clip")
            expect(effect).to_have_count(1)
            expect(effect).to_have_text("Slow push in · Rain")
            assert effect.evaluate("clip => clip.style.width") == page.locator(
                '#lanes .shot-clip[data-shot="0"]').evaluate("clip => clip.style.width")
            page.evaluate("""async source => {
              const audio = document.getElementById('mix-audio');
              await new Promise(resolve => { audio.addEventListener('canplaythrough', resolve, {once: true}); audio.src = source; audio.load(); });
            }""", "data:audio/wav;base64," + base64.b64encode(audio.getvalue()).decode())
            # Picking a shot while the mix plays must seek without pausing it.
            page.locator("#play").click()
            page.wait_for_function("!document.getElementById('mix-audio').paused")
            page.locator('#shot-list .shot-row[data-shot="1"]').click()
            page.wait_for_function("document.getElementById('mix-audio').currentTime >= 4")
            assert page.locator("#mix-audio").evaluate("audio => !audio.paused")
            page.locator('#lanes .shot-clip[data-shot="0"]').click()
            page.wait_for_function("document.getElementById('mix-audio').currentTime < 3")
            assert page.locator("#mix-audio").evaluate("audio => !audio.paused")
            page.locator("#play").click()
            # Let the renderer settle before choosing which canvas to capture.
            status = page.evaluate("""async () => {
              const {gpuStatus} = await import('/soundstage/effects_webgpu.js');
              const deadline = performance.now() + 10000;
              while (!['ready', 'unavailable'].includes(gpuStatus()) && performance.now() < deadline) {
                await new Promise(resolve => setTimeout(resolve, 50));
              }
              return gpuStatus();
            }""")
            # Seek the actual player while paused; every frame must reconstruct deterministically.
            def seek_frame(t):
                info = page.evaluate("""async t => {
                  const audio = document.getElementById('mix-audio');
                  await new Promise(resolve => { audio.addEventListener('seeked', resolve, {once: true}); audio.currentTime = t; });
                  const gpu = document.querySelector('.still-effects-gpu');
                  const active = gpu && gpu.style.visibility === 'visible' ? gpu : document.getElementById('still-effects');
                  return {className: active.className, transform: document.getElementById('still-back').style.transform,
                    time: audio.currentTime, duration: audio.duration, clock: document.getElementById('clock').textContent};
                }""", t)
                selector = ".still-effects-gpu" if "still-effects-gpu" in info["className"] else "#still-effects"
                info["image"] = page.locator(selector).screenshot()
                return info

            a, b, c = seek_frame(1), seek_frame(2), seek_frame(1)
            frame = {
                "moves": a["image"] != b["image"] and a["transform"] != b["transform"],
                "seeks": a["image"] == c["image"] and a["transform"] == c["transform"],
                "className": a["className"],
                "transforms": [a["transform"], b["transform"], c["transform"]],
                "times": [a["time"], b["time"]], "duration": a["duration"],
                "clock": a["clock"], "imageChanged": a["image"] != b["image"],
            }
            assert frame["moves"] and frame["seeks"], (frame, errors)
            backend = "WebGPU" if "still-effects-gpu" in c["className"] else "Canvas 2D"
            if webgpu:
                assert backend == "WebGPU", (status, frame, errors)
            expect(engine).to_have_text(f"Effects engine · {backend}")
            with page.expect_response(lambda response: response.url.split("?")[0].endswith("/project/video") and response.request.method == "PATCH"):
                page.get_by_label("Visual effects", exact=True).uncheck()
            expect(page.locator("#still-back")).to_have_css("transform", "none")
            expect(engine).to_have_text("Effects engine · Off")
            page.wait_for_load_state("networkidle")
            assert fixture.project()["video"]["effects_enabled"] is False
            with page.expect_response(lambda response: response.url.split("?")[0].endswith("/project/video") and response.request.method == "PATCH"):
                page.get_by_label("Visual effects", exact=True).check()
            page.wait_for_load_state("networkidle")
            with page.expect_response(lambda response: "/project/video/shot" in response.url and response.request.method == "PATCH"):
                page.get_by_label("Lighting & framing", exact=True).select_option("lightning")
            expect(page.get_by_label("Lighting & framing", exact=True)).to_have_value("lightning")
            page.wait_for_load_state("networkidle")
            assert fixture.project()["video"]["plan"]["rows"][0]["effects"] == ["push_in", "rain", "lightning"]
            page.wait_for_load_state("networkidle")
            page.emulate_media(reduced_motion="reduce")
            expect(page.locator("#still-back")).to_have_css("transform", "none")
            expect(page.get_by_label("Visual effects", exact=True)).not_to_be_checked()
            expect(page.locator("#effects-motion-hint")).to_be_visible()
            expect(engine).to_have_text("Effects engine · Paused")
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
            # A planned effect stays on the timeline when its playback is off.
            expect(effect).to_have_text("Slow push in · Rain · Lightning")
            page.locator('#lanes .shot-clip[data-shot="1"]').click()
            effect.click()
            expect(effect).to_have_attribute("aria-pressed", "true")
            page.wait_for_function("document.getElementById('mix-audio').currentTime < 1")
            page.locator("#step-script").click()
            page.locator(".estimate-toggle").click()
            expect(page.locator("#estimate .effect-clip")).to_have_text("Slow push in · Rain · Lightning")
            page.locator("#step-mix").click()
            expect(effect).to_have_text("Slow push in · Rain · Lightning")
            if webgpu:
                check_webgpu_composite(page)
            assert not errors, errors
            browser.close()
        print(f"PASS ({backend}): effect controls, persistence, animation, seeking, reduced motion, module loading")
    finally:
        server.should_exit = True
        if thread.is_alive():
            thread.join(timeout=10)
        listener.close()
        fixture.doCleanups()


if __name__ == "__main__":
    run(webgpu="--webgpu" in sys.argv)
