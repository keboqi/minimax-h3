"""Focused browser smoke for clock-driven Illustrate effects, using local fake images."""
import base64
import json
from io import BytesIO
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread
import time
from unittest.mock import patch
import wave

from PIL import Image
from playwright.sync_api import expect, sync_playwright
import uvicorn

from h3_app.drama3.visuals import FPS, Shot, render_video
from h3_ui.drama3 import effects_export
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
            if not webgpu:
                page.add_init_script("Object.defineProperty(navigator, 'gpu', {value: undefined})")
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            def record_console_error(msg):
                if msg.type == "warning" and msg.text == "No available adapters.":
                    return
                if msg.type in ("error", "warning"):
                    errors.append(f"[{msg.type}] {msg.text}")

            page.on("console", record_console_error)
            page.route("**/soundstage/api/audio?*", lambda route: route.fulfill(body=audio.getvalue(), content_type="audio/wav"))
            url = f"http://127.0.0.1:{port}/soundstage/"
            page.goto(url, wait_until="networkidle")
            fullscreen = page.locator("#player-fullscreen")
            expect(fullscreen).to_have_count(0)
            page.locator("#step-illustrate").click()
            summary = page.locator("#shot-effects")
            dialog = page.locator("#effects-dialog")
            source = dialog.get_by_label("Preview source")
            preview_image = dialog.locator(".effect-preview img")
            summary.click()
            expect(source).to_have_value("camera")
            assert source.locator('option[value="shot"]').evaluate("option => option.disabled")
            expect(preview_image).to_have_attribute("src", f"{url}effect-demo-camera.webp")
            preview_image.evaluate("image => image.decode()")
            dialog.get_by_role("button", name="Cancel", exact=True).click()
            list(fixture.fake_export(fixture.mix.name, "qwen", "ink wash", 2, True, "landscape", '"all"'))
            page.reload(wait_until="networkidle")
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
            expect(page.locator("#shot-inspector select")).to_have_count(0)
            apply = dialog.get_by_role("button", name="Apply effects")
            first_shot, second_shot = fixture.project()["video"]["plan"]["rows"]
            page.locator('#shot-list .shot-row[data-shot="1"]').click()
            summary.click()
            expect(source).to_have_value("shot")
            expect(preview_image).to_have_attribute("src", second_shot["image"])
            dialog.get_by_role("button", name="Cancel", exact=True).click()
            page.evaluate("""() => {
              const decode = HTMLImageElement.prototype.decode;
              HTMLImageElement.prototype.decode = function () {
                if (this.closest('.effect-preview') && this.src.includes('/api/video/still?')) {
                  HTMLImageElement.prototype.decode = decode;
                  return Promise.reject(new Error('Cannot decode the shot'));
                }
                return decode.call(this);
              };
            }""")
            summary.click()
            expect(source).to_have_value("camera")
            expect(dialog.get_by_role("status")).to_contain_text("The shot image couldn’t load.")
            expect(preview_image).to_have_attribute("src", f"{url}effect-demo-camera.webp")
            dialog.get_by_role("button", name="Cancel", exact=True).click()
            page.locator('#shot-list .shot-row[data-shot="0"]').click()
            expect(summary).to_contain_text("No effects selected")
            summary.click()
            expect(dialog).to_be_visible()
            expect(apply).to_be_disabled()
            expect(source).to_have_value("shot")
            expect(preview_image).to_have_attribute("src", first_shot["image"])
            preview_image.evaluate("image => image.decode()")
            push = dialog.get_by_role("button", name="Slow push in", exact=True)
            close_picker = dialog.get_by_role("button", name="Close", exact=True)
            close_picker.hover()
            push_image = push.locator("img")
            push_poster = push_image.get_attribute("style")
            push.hover()
            page.wait_for_function("""poster => document.querySelector(
                '#effects-dialog [data-effect="push_in"] img').getAttribute('style') !== poster""", arg=push_poster)
            expect(push).to_have_attribute("aria-pressed", "false")
            expect(apply).to_be_disabled()
            page.emulate_media(reduced_motion="reduce")
            expect(push_image).to_have_attribute("style", push_poster)
            close_picker.hover()
            push.hover()
            assert push_image.evaluate("""async image => {
                const poster = image.getAttribute('style');
                for (let i = 0; i < 8; i++) {
                  await new Promise(requestAnimationFrame);
                  if (image.getAttribute('style') !== poster) return false;
                }
                return true;
            }""")
            page.emulate_media(reduced_motion="no-preference")
            close_picker.hover()
            push.click()
            expect(push).to_have_attribute("aria-pressed", "true")
            for demo in ("camera", "atmosphere", "lighting"):
                source.select_option(demo)
                expect(preview_image).to_have_attribute("src", f"{url}effect-demo-{demo}.webp")
                preview_image.evaluate("image => image.decode()")
            atmosphere = dialog.get_by_role("button", name="Atmosphere", exact=True)
            atmosphere.click()
            rain = dialog.get_by_role("button", name="Rain", exact=True)
            rain_canvas = rain.locator("canvas")
            rain_poster = rain_canvas.evaluate("canvas => canvas.toDataURL()")
            rain.hover()
            page.wait_for_function("""poster => document.querySelector(
                '#effects-dialog [data-effect="rain"] canvas').toDataURL() !== poster""", arg=rain_poster)
            expect(rain).to_have_attribute("aria-pressed", "false")
            atmosphere.hover()
            assert rain_canvas.evaluate("canvas => canvas.toDataURL()") == rain_poster
            expect(source).to_have_value("lighting")
            expect(dialog.locator(".effect-summary-chips")).to_have_text("Slow push in")
            source.select_option("shot")
            expect(preview_image).to_have_attribute("src", first_shot["image"])
            play_preview = dialog.get_by_role("button", name="Play preview")
            play_preview.click()
            page.wait_for_function("document.querySelector('.effect-preview img').style.transform !== 'scale(1)'")
            dialog.get_by_role("button", name="Stop preview").click()
            assert fixture.project()["video"]["plan"]["rows"][0]["effects"] == []
            with page.expect_response(lambda response: "/project/video/shot" in response.url and response.request.method == "PATCH"):
                apply.click()
            expect(dialog).to_be_hidden()
            expect(summary).to_be_focused()
            expect(summary).to_contain_text("Slow push in")
            page.wait_for_load_state("networkidle")
            assert fixture.project()["video"]["plan"]["rows"][0]["effects"] == ["push_in"], "camera did not save"
            expect(engine).to_have_text("Effects engine · Camera only")
            summary.click()
            expect(source).to_have_value("shot")
            expect(push).to_have_attribute("aria-pressed", "true")
            dialog.get_by_role("button", name="None", exact=True).click()
            dialog.get_by_role("button", name="Cancel", exact=True).click()
            expect(summary).to_contain_text("Slow push in")
            summary.click()
            dialog.get_by_role("button", name="Atmosphere", exact=True).click()
            expect(preview_image).to_have_attribute("src", first_shot["image"])
            dialog.get_by_role("button", name="Rain", exact=True).click()
            expect(dialog.locator(".effect-summary-chips")).to_contain_text("Rain")
            with page.expect_response(lambda response: "/project/video/shot" in response.url and response.request.method == "PATCH"):
                apply.click()
            expect(summary).to_contain_text("Rain")
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
            assert backend == ("WebGPU" if webgpu else "Canvas 2D"), (status, frame, errors)
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
            page.evaluate("""() => {
              const originalFetch = window.fetch;
              window.effectWrites = [];
              window.fetch = async (url, options) => {
                if (url.includes('/project/video/shot') && options?.method === 'PATCH') {
                  window.effectWrites.push(JSON.parse(options.body));
                  if (window.effectWrites.length === 1) {
                    await new Promise(resolve => { window.releaseEffectWrite = resolve; });
                  }
                }
                return originalFetch(url, options);
              };
              window.restoreEffectFetch = () => { window.fetch = originalFetch; };
            }""")
            summary.click()
            dialog.get_by_role("button", name="Atmosphere", exact=True).click()
            dialog.get_by_role("button", name="Snow", exact=True).click()
            dialog.get_by_role("button", name="Lighting & framing", exact=True).click()
            expect(preview_image).to_have_attribute("src", first_shot["image"])
            dialog.get_by_role("button", name="Lightning", exact=True).click()
            expect(dialog.get_by_text("Playing this preview includes brief flashes.")).to_be_visible()
            assert page.evaluate("window.effectWrites.length") == 0
            apply.click()
            page.wait_for_function("window.releaseEffectWrite !== undefined")
            expect(dialog.get_by_role("button", name="Saving…")).to_be_disabled()
            page.keyboard.press("Escape")
            expect(dialog).to_be_visible()
            page.evaluate("window.releaseEffectWrite()")
            expect(effect).to_have_text("Slow push in · Snow · Lightning")
            assert page.evaluate("window.effectWrites") == [
                {"index": 0, "effects": ["push_in", "snow", "lightning"]},
            ]
            page.evaluate("window.restoreEffectFetch()")
            summary.click()
            dialog.get_by_role("button", name="Atmosphere", exact=True).click()
            dialog.get_by_role("button", name="Rain", exact=True).click()
            with page.expect_response(lambda response: "/project/video/shot" in response.url and response.request.method == "PATCH"):
                apply.click()
            expect(effect).to_have_text("Slow push in · Rain · Lightning")
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
            if backend == "Canvas 2D":
                alpha = page.evaluate("""async () => {
                  const {paintEffects} = await import('/soundstage/effects.js');
                  const canvas = document.createElement('canvas');
                  canvas.style.cssText = 'width: 160px; height: 90px';
                  document.body.append(canvas);
                  const context = canvas.getContext('2d', {willReadFrequently: true});
                  const alpha = [0, 0.5, 1].map(fadeMix => {
                    paintEffects(canvas, ['vignette'], 0, null, {
                      fading: true, fadeMix, backEffects: ['vignette'],
                    });
                    return context.getImageData(0, 0, 1, 1).data[3];
                  });
                  canvas.remove();
                  return alpha;
                }""")
                assert min(alpha) > 0 and max(alpha) - min(alpha) <= 1, alpha
            assert not errors, errors
            browser.close()
        print(f"PASS ({backend}): effect controls, persistence, animation, seeking, reduced motion, module loading")
    finally:
        server.should_exit = True
        if thread.is_alive():
            thread.join(timeout=10)
        listener.close()
        fixture.doCleanups()


def run_export():
    with TemporaryDirectory() as temp:
        folder = Path(temp)
        size = (160, 90)
        images = [folder / "before.png", folder / "after.png"]
        Image.new("RGB", size, (48, 48, 48)).save(images[0])
        Image.new("RGB", size, (192, 48, 32)).save(images[1])
        audio = folder / "mix.wav"
        with wave.open(str(audio), "wb") as wav:
            wav.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
            wav.writeframes(bytes(8000 * 2 * 2))
        shots = [Shot(0, FPS, False, ""), Shot(FPS, FPS, True, "")]
        timeline = [{"index": 0, "start": 0, "end": 1, "fade": 0, "effects": ["push_in", "flash"]},
                    {"index": 1, "start": 1, "end": 2, "fade": 0.5, "effects": []}]
        sounds = [{"name": "Impact", "start": 0.75, "end": 0.8, "loop": False}]

        def capture(workspace):
            return effects_export.capture_frames(workspace, timeline, sounds, images, size, FPS * 2)

        with patch.object(effects_export, "BATCH_FRAMES", 12):
            video = render_video(audio, shots, images, folder, size=size, frame_batches=capture)
        streams = json.loads(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(video)]))["streams"]
        visual, sound = streams
        assert (visual["width"], visual["height"], int(visual["nb_frames"])) == (*size, FPS * 2)
        assert sound["codec_name"] == "aac" and abs(float(sound["duration"]) - 2) < 1 / FPS
        pixels = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(video), "-an",
                                          "-vf", "crop=2:2:80:44,scale=1:1", "-pix_fmt", "rgb24",
                                          "-f", "rawvideo", "-"])
        def red(frame):
            return pixels[frame * 3]

        assert red(19) > red(17) + 30, (red(17), red(19))
        assert red(24) < red(30) < red(36) - 25, (red(24), red(30), red(36))
        assert red(42) > 170, red(42)
    print("PASS: MP4 frame count, soundtrack, sound-cued flash, and dissolve")


if __name__ == "__main__":
    run(webgpu="--webgpu" in sys.argv)
    run_export()
