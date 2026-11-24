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
from PIL import ImageDraw
from playwright.sync_api import expect, sync_playwright
import uvicorn

from h3_app.drama3.visuals import FPS, Shot, render_video
from h3_ui.drama3 import effects_export
from tests.test_drama3_page_video import API, NAME, PLAN, Drama3PageVideoTests


def check_webgpu_composite(page):
    sizes = page.evaluate("""async () => {
      const {probeWebGPU, renderWebGPU, ATMO_IDS, LIGHT_IDS} = await import('/soundstage/effects_webgpu.js');
      await probeWebGPU();
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


def _encode_split_depth_png(size=(160, 90), z_left=1.5, z_right=25.0):
    img = Image.new("RGB", size)
    inv_min, inv_max = 0.0, 1.0 / 1.2
    half_w = size[0] // 2

    def pack(z):
        u = max(0.0, min(1.0, ((1.0 / z) - inv_min) / (inv_max - inv_min)))
        q = int(round(u * 65535.0))
        return (q >> 8, q & 0xFF, 0)

    px_left, px_right = pack(z_left), pack(z_right)
    pixels = [px_left if x < half_w else px_right for _y in range(size[1]) for x in range(size[0])]
    img.putdata(pixels)
    return img, {
        "inv_min": inv_min, "inv_max": inv_max, "width": size[0], "height": size[1],
        "f_px": float(max(size) * 1.1), "depth_strength": 1.0, "parallax_scale": 1.0,
        "z_p10": z_left, "z_p25": z_left, "z_p50": 4.0, "z_p90": z_right,
        "subject_z": z_left, "subject_uv": [0.25, 0.5], "rack_start_z": z_right,
        "warm_anchor": {"u": 0.5, "v": 0.5, "z": 3.0, "strength": 0.5},
        "sky_dir": {"u": 0.5, "v": 0.1},
    }


def _to_data_url(image):
    buf = BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _row_contrast(rgb_img, x0, x1, y):
    vals = [sum(rgb_img.getpixel((x, y))) / 3.0 for x in range(x0, x1)]
    return sum(abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1)) / max(1, len(vals) - 1)


def check_webgpu_depth_and_layers(page):
    depth_img, depth_meta = _encode_split_depth_png((160, 90), z_left=1.5, z_right=25.0)
    dark_img = Image.new("RGB", (160, 90), (24, 24, 24))
    stripe_img = Image.new("RGB", (160, 90))
    stripe_img.putdata([
        ((235, 235, 235) if (x // 3) % 2 == 0 else (20, 20, 20))
        for _y in range(90) for x in range(160)
    ])
    left_mask = Image.new("RGBA", (160, 90), (0, 0, 0, 0))
    ImageDraw.Draw(left_mask).rectangle((0, 0, 79, 89), fill=(255, 255, 255, 255))

    urls = {
        "depth": _to_data_url(depth_img),
        "dark": _to_data_url(dark_img),
        "stripe": _to_data_url(stripe_img),
        "mask": _to_data_url(left_mask),
    }
    fog_meta = {**depth_meta, "subject_z": 3.2}
    page.evaluate("""async ({urls}) => {
      const {probeWebGPU} = await import('/soundstage/effects_webgpu.js');
      await probeWebGPU(undefined, {layers: true});
      const probe = document.createElement('div');
      probe.id = 'depth-probe';
      probe.style = 'position: fixed; top: 0; left: 0; z-index: 9999; width: 160px; height: 90px';
      const host = document.createElement('canvas');
      probe.append(host);
      document.body.append(probe);
      async function load(src) {
        const im = new Image();
        im.src = src;
        await im.decode();
        return im;
      }
      window.__depthProbe = {
        host,
        depth: await load(urls.depth),
        dark: await load(urls.dark),
        stripe: await load(urls.stripe),
        mask: await load(urls.mask),
      };
    }""", {"urls": urls})

    # 1) Volumetric Beer-Lambert fog: far right (25m) is noticeably brighter/foggier than near left (1.5m).
    assert page.evaluate("""({meta}) => import('/soundstage/effects_webgpu.js').then(({renderWebGPU, ATMO_IDS}) => {
      const p = window.__depthProbe;
      return renderWebGPU(p.host, ATMO_IDS.fog, 0, 1.5, 160, 90, {
        stillBack: p.dark, depthImage: p.depth, depthMeta: meta,
      });
    })""", {"meta": fog_meta})
    fog_frame = Image.open(BytesIO(page.locator("#depth-probe").screenshot())).convert("RGB")
    w, h = fog_frame.size
    left_luma = sum(sum(fog_frame.getpixel((x, y))) / 3.0 for x in range(w // 8, w * 3 // 8, 4) for y in range(h // 2, h * 7 // 8, 4))
    right_luma = sum(sum(fog_frame.getpixel((x, y))) / 3.0 for x in range(w * 5 // 8, w * 7 // 8, 4) for y in range(h // 2, h * 7 // 8, 4))
    assert right_luma > left_luma + 50, (left_luma, right_luma)

    # 2) Optical thin-lens rack_focus: focus near left (blur=0) keeps left stripes sharper; blur=1 flips focus to far right.
    assert page.evaluate("""({meta}) => import('/soundstage/effects_webgpu.js').then(({renderWebGPU}) => {
      const p = window.__depthProbe;
      return renderWebGPU(p.host, 0, 0, 1.0, 160, 90, {
        stillBack: p.stripe, depthImage: p.depth, depthMeta: meta, rackActive: true,
        cam: {tx: 0, ty: 0, scale: 1, shake: 0, angle: 0, blur: 0},
      });
    })""", {"meta": depth_meta})
    near_frame = Image.open(BytesIO(page.locator("#depth-probe").screenshot())).convert("RGB")
    near_left_c = _row_contrast(near_frame, w // 6, w * 2 // 5, h // 2)
    near_right_c = _row_contrast(near_frame, w * 3 // 5, w * 5 // 6, h // 2)
    assert near_left_c > near_right_c + 15, (near_left_c, near_right_c)

    assert page.evaluate("""({meta}) => import('/soundstage/effects_webgpu.js').then(({renderWebGPU}) => {
      const p = window.__depthProbe;
      return renderWebGPU(p.host, 0, 0, 0.05, 160, 90, {
        stillBack: p.stripe, depthImage: p.depth, depthMeta: meta, rackActive: true,
        cam: {tx: 0, ty: 0, scale: 1, shake: 0, angle: 0, blur: 1.0},
      });
    })""", {"meta": depth_meta})
    far_frame = Image.open(BytesIO(page.locator("#depth-probe").screenshot())).convert("RGB")
    far_left_c = _row_contrast(far_frame, w // 6, w * 2 // 5, h // 2)
    far_right_c = _row_contrast(far_frame, w * 3 // 5, w * 5 // 6, h // 2)
    assert far_right_c > far_left_c + 15, (far_left_c, far_right_c)

    # A speaker cue alone must focus the far stripes without moving the camera pivot.
    assert page.evaluate("""({meta}) => import('/soundstage/effects_webgpu.js').then(({renderWebGPU}) => {
      const p = window.__depthProbe;
      return renderWebGPU(p.host, 0, 0, 1.0, 160, 90, {
        stillBack: p.stripe, depthImage: p.depth, depthMeta: {...meta, focus_z: 25},
      });
    })""", {"meta": depth_meta})
    cue_frame = Image.open(BytesIO(page.locator("#depth-probe").screenshot())).convert("RGB")
    assert (_row_contrast(cue_frame, w * 3 // 5, w * 5 // 6, h // 2)
            > _row_contrast(cue_frame, w // 6, w * 2 // 5, h // 2) + 15)

    # 3) Unified SAM renderWebGPULayers: inside cold_tint on left + behind fog (region_depth_m=3.0).
    assert page.evaluate("""({meta}) => import('/soundstage/effects_webgpu.js').then(({renderWebGPULayers}) => {
      const p = window.__depthProbe;
      const layers = [
        {preset: 'cold_tint', mode: 'inside', intensity: 1, feather: 0, maskImages: [p.mask], maskUrls: ['left']},
        {preset: 'fog', mode: 'behind', intensity: 1, feather: 0, region_depth_m: 3.0, maskImages: [p.mask], maskUrls: ['left']},
        {preset: 'fog', mode: 'behind', intensity: 1, feather: 0, region_depth_m: 3.0, maskImages: [p.mask], maskUrls: ['left']},
      ];
      return renderWebGPULayers(p.host, layers, 1.5, 160, 90, {
        stillBack: p.dark, depthImage: p.depth, depthMeta: meta, layers,
      });
    })""", {"meta": fog_meta})
    layer_frame = Image.open(BytesIO(page.locator("#depth-probe").screenshot())).convert("RGB")
    lr, _lg, lb = layer_frame.getpixel((w // 4, h // 2))
    rr, rg, rb = layer_frame.getpixel((w * 3 // 4, h * 3 // 4))
    assert lb > lr + 4, (lr, lb)
    assert (rr + rg + rb) / 3.0 > 28, (rr, rg, rb)

    # 4) Emissive screen presets (warm_glow + sunbeams) never darken bright highlights, and thunder-synced lightning respects cue time.
    bright_url = _to_data_url(Image.new("RGB", (160, 90), (235, 235, 235)))
    assert page.evaluate("""async ({brightUrl}) => {
      const {renderWebGPU, ATMO_IDS, LIGHT_IDS} = await import('/soundstage/effects_webgpu.js');
      const p = window.__depthProbe;
      const bright = new Image();
      bright.src = brightUrl;
      await bright.decode();
      p.bright = bright;
      return renderWebGPU(p.host, ATMO_IDS.sunbeams, LIGHT_IDS.warm_glow, 1.0, 160, 90, {stillBack: bright});
    }""", {"brightUrl": bright_url})
    bright_frame = Image.open(BytesIO(page.locator("#depth-probe").screenshot())).convert("RGB")
    br, bg, bb = bright_frame.getpixel((w // 2, h // 2))
    assert br >= 234 and bg >= 234 and bb >= 234, (br, bg, bb)

    assert page.evaluate("""() => import('/soundstage/effects_webgpu.js').then(({renderWebGPU, LIGHT_IDS}) => {
      const p = window.__depthProbe;
      return renderWebGPU(p.host, 0, LIGHT_IDS.lightning, 0.12, 160, 90, {
        stillBack: p.dark, lightningCueElapsed: 99,
      });
    })""")
    pre_thunder = Image.open(BytesIO(page.locator("#depth-probe").screenshot())).convert("RGB")
    assert page.evaluate("""() => import('/soundstage/effects_webgpu.js').then(({renderWebGPU, LIGHT_IDS}) => {
      const p = window.__depthProbe;
      return renderWebGPU(p.host, 0, LIGHT_IDS.lightning, 1.8, 160, 90, {
        stillBack: p.dark, lightningCueElapsed: 0.05,
      });
    })""")
    on_thunder = Image.open(BytesIO(page.locator("#depth-probe").screenshot())).convert("RGB")
    assert sum(pre_thunder.getpixel((w // 2, h // 4))) < 90, pre_thunder.getpixel((w // 2, h // 4))
    assert sum(on_thunder.getpixel((w // 2, h // 4))) > 240, on_thunder.getpixel((w // 2, h // 4))

    page.evaluate("""() => {
      delete window.__depthProbe;
      document.getElementById('depth-probe')?.remove();
    }""")


def check_enhancement(page, fixture):
    from h3_ui.drama3 import effect_regions, visuals

    fixture.runtime.script_dir = Path.cwd()
    video = fixture.project()["video"]
    folder = visuals.stills_dir(fixture.mix, fixture.runtime.output_dir)
    labels = {subject["key"]: subject for subject in video["plan"]["subjects"]}
    for row in video["plan"]["rows"]:
        if not row["image_file"]:
            continue
        image = row["image_file"]
        path = effect_regions.manifest_path(folder, image, effect_regions.concepts(row, labels))
        path.parent.mkdir(exist_ok=True)
        mask1 = f"{image}.{'a' * 32}.region_1.png"
        mask2 = f"{image}.{'b' * 32}.region_2.png"
        with Image.open(folder / image) as still:
            Image.new("RGBA", still.size, (255, 255, 255, 180)).save(path.parent / mask1)
            Image.new("RGBA", still.size, (255, 255, 255, 180)).save(path.parent / mask2)
        path.write_text(json.dumps([
            {"id": "region_1", "label": "window", "mask": mask1, "box": [0, 0, 0.4, 1]},
            {"id": "region_2", "label": "lamp", "mask": mask2, "box": [0.6, 0, 1, 1]},
        ]), encoding="utf-8")
    api = f"{API}/project/video/effects"
    prompt = fixture.ok(fixture.client.post(f"{api}/prompt", params=NAME))["prompt"]
    request = json.loads(prompt.split("\n\n")[-1])
    reply = {"revision": request["revision"], "shots": [
        {"shot_id": shot["shot_id"], "action": "replace" if index == 0 else "keep",
         "effects": [{"preset": "rain", "mode": "inside", "region_id": "region_1"},
                     {"preset": "cold_tint", "mode": "inside", "region_id": "region_1"}]}
        for index, shot in enumerate(request["shots"])
    ]}
    page.reload(wait_until="networkidle")
    page.locator("#step-illustrate").click()
    page.get_by_role("button", name="Enhance visual effects").click()
    dialog = page.locator("#effect-pass-dialog")
    dialog.get_by_label("Pasted effects reply").fill(json.dumps(reply))
    dialog.get_by_role("button", name="Preview pasted result").click()
    expect(dialog.get_by_text("1 proposed · 1 kept · 0 skipped")).to_be_visible()
    expect(dialog.get_by_role("slider", name="Preview time")).to_be_visible()
    dialog.get_by_role("button", name="Apply all").click()
    expect(dialog.get_by_role("button", name="Apply all")).to_have_count(0)
    dialog.get_by_role("button", name="Close").click()
    page.locator('.shot-row[data-shot="0"]').click()
    page.locator("#shot-effects").click()

    # Verify mask refine overlay/click alignment and right-click exclude.
    dialog.locator(".pass-seg-chip").first.click()
    mask_box_el = dialog.locator(".pass-mask")
    mask_img_el = mask_box_el.locator("img")
    mask_img_el.evaluate("img => img.decode()")
    mask_box = mask_box_el.bounding_box()
    img_box = mask_img_el.bounding_box()
    assert mask_box and img_box
    assert abs(mask_box["width"] - img_box["width"]) <= 1, (mask_box, img_box)
    assert abs(mask_box["height"] - img_box["height"]) <= 1, (mask_box, img_box)
    assert abs(mask_box["x"] - img_box["x"]) <= 1, (mask_box, img_box)
    assert abs(mask_box["y"] - img_box["y"]) <= 1, (mask_box, img_box)
    mask_box_el.click(position={"x": 40, "y": 40})
    mask_box_el.click(button="right", position={"x": 80, "y": 40})
    expect(dialog.locator(".pass-point.include")).to_have_count(1)
    expect(dialog.locator(".pass-point.exclude")).to_have_count(1)
    dialog.get_by_role("button", name="Done").click()

    # Verify adding a new segment with label and include/exclude points.
    dialog.get_by_role("button", name="+ Add segment").click()
    expect(dialog.locator(".pass-mask-overlay")).to_have_count(0)
    dialog.get_by_label("Segment label").fill("lantern")
    add_mask_el = dialog.locator(".pass-mask")
    add_mask_el.locator("img").evaluate("img => img.decode()")
    add_mask_el.click(position={"x": 45, "y": 45})
    add_mask_el.click(button="right", position={"x": 90, "y": 45})
    expect(dialog.locator(".pass-point.include")).to_have_count(1)
    expect(dialog.locator(".pass-point.exclude")).to_have_count(1)
    dialog.get_by_role("button", name="Done").click()

    # Grouped multi-effect + multi-object set.
    expect(dialog.get_by_text("1 set · 2 / 16 effects")).to_be_visible()
    set0 = dialog.locator('.pass-set-card[data-set-index="0"]')
    set0_effects = set0.get_by_role("combobox", name="Effect")
    expect(set0_effects).to_have_count(2)
    assert set0_effects.nth(0).locator("optgroup").count() == 3
    set0_effects.nth(0).select_option("snow")
    expect(set0_effects.nth(0)).to_have_value("snow")

    preset_dialog = page.locator("#effect-preset-dialog")
    set0.locator('.pass-effect-row[data-effect-index="0"]').get_by_role("button", name="Previews").click()
    expect(preset_dialog).to_be_visible()
    expect(dialog).to_be_visible()
    expect(preset_dialog.get_by_role("button", name="Atmosphere", exact=True)).to_have_attribute("aria-pressed", "true")
    expect(preset_dialog.get_by_role("button", name="Snow", exact=True)).to_have_attribute("aria-pressed", "true")
    rain_card = preset_dialog.get_by_role("button", name="Rain", exact=True)
    rain_canvas = rain_card.locator("canvas")
    rain_poster = rain_canvas.evaluate("canvas => canvas.toDataURL()")
    rain_card.hover()
    expect(rain_card).to_have_attribute("data-playing", "")
    page.wait_for_function(
        """poster => document.querySelector('#effect-preset-dialog [data-effect="rain"] canvas').toDataURL() !== poster""",
        arg=rain_poster,
    )
    shot_image = video["plan"]["rows"][0]["image"]
    preset_dialog.get_by_role("button", name="This shot", exact=True).click()
    expect(rain_card.locator("img")).to_have_attribute("src", shot_image)
    preset_dialog.get_by_role("button", name="Demo scenes", exact=True).click()
    assert (rain_card.locator("img").get_attribute("src") or "").endswith("effect-demo-atmosphere.webp")
    rain_card.click()
    expect(preset_dialog).to_be_hidden()
    expect(set0_effects.nth(0)).to_have_value("rain")

    set0.get_by_role("combobox", name="Add object").select_option("region_2")
    expect(set0.locator(".pass-region-chip")).to_have_count(2)

    set0.get_by_role("button", name="+ Add effect").click()
    expect(preset_dialog).to_be_visible()
    preset_dialog.get_by_role("button", name="Camera movement", exact=True).click()
    expect(preset_dialog.get_by_role("button", name="Slow push in", exact=True)).to_be_disabled()
    expect(preset_dialog.get_by_text("Camera moves apply to Full frame only.")).to_be_visible()
    page.keyboard.press("Escape")
    expect(preset_dialog).to_be_hidden()

    # Full frame set with Camera + Lightning + Rain sharing intensity.
    dialog.get_by_role("button", name="+ Add effect set").click()
    expect(preset_dialog).to_be_visible()
    expect(preset_dialog.get_by_role("heading", name="Add effect set")).to_be_visible()
    push_card = preset_dialog.get_by_role("button", name="Slow push in", exact=True)
    push_image = push_card.locator("img")
    push_poster = push_image.get_attribute("style")
    push_card.hover()
    expect(push_card).to_have_attribute("data-playing", "")
    page.wait_for_function(
        """poster => document.querySelector('#effect-preset-dialog [data-effect="push_in"] img').getAttribute('style') !== poster""",
        arg=push_poster,
    )
    push_card.click()
    expect(preset_dialog).to_be_hidden()
    expect(dialog.get_by_text("2 sets · 3 / 16 effects")).to_be_visible()

    set1 = dialog.locator('.pass-set-card[data-set-index="1"]')
    set1.get_by_role("button", name="+ Add effect").click()
    expect(preset_dialog).to_be_visible()
    preset_dialog.get_by_role("button", name="Lighting & framing", exact=True).click()
    preset_dialog.get_by_role("button", name="Lightning", exact=True).click()
    expect(preset_dialog).to_be_hidden()

    set1.get_by_role("button", name="+ Add effect").click()
    expect(preset_dialog).to_be_visible()
    preset_dialog.get_by_role("button", name="Atmosphere", exact=True).click()
    preset_dialog.get_by_role("button", name="Rain", exact=True).click()
    expect(preset_dialog).to_be_hidden()
    expect(dialog.get_by_text("2 sets · 5 / 16 effects")).to_be_visible()

    set1.get_by_role("slider", name="intensity").fill("0.7")
    expect(set1.get_by_text("70%")).to_be_visible()
    set1_apply = set1.get_by_role("combobox", name="Apply")
    for mode in ("inside", "outside", "behind"):
        assert set1_apply.locator(f'option[value="{mode}"]').is_disabled()
    expect(set1.get_by_text("Camera moves apply to Full frame only.")).to_be_visible()
    assert set0_effects.nth(0).locator('optgroup[label="Camera movement"]').is_disabled()
    assert not set1.get_by_role("combobox", name="Effect").nth(0).locator('optgroup[label="Camera movement"]').is_disabled()

    # Separate set with different intensity.
    dialog.get_by_role("button", name="+ Add effect set").click()
    expect(preset_dialog).to_be_visible()
    preset_dialog.get_by_role("button", name="Atmosphere", exact=True).click()
    preset_dialog.get_by_role("button", name="Fog", exact=True).click()
    expect(preset_dialog).to_be_hidden()
    set2 = dialog.locator('.pass-set-card[data-set-index="2"]')
    set2.get_by_role("slider", name="intensity").fill("0.4")
    expect(set2.get_by_text("40%")).to_be_visible()

    with page.expect_response(lambda response: "/project/video/effects/shot" in response.url and response.request.method == "POST"):
        dialog.get_by_role("button", name="Save shot effects").click()
    expect(dialog).to_be_hidden()
    page.wait_for_load_state("networkidle")
    saved_effects = fixture.project()["video"]["effect_enhancement"]["shots"][0]["effects"]
    assert saved_effects == [
        {"preset": "rain", "mode": "inside", "region_ids": ["region_1", "region_2"]},
        {"preset": "cold_tint", "mode": "inside", "region_ids": ["region_1", "region_2"]},
        {"preset": "push_in", "mode": "full_frame"},
        {"preset": "lightning", "mode": "full_frame", "intensity": 0.7},
        {"preset": "rain", "mode": "full_frame", "intensity": 0.7},
        {"preset": "fog", "mode": "full_frame", "intensity": 0.4},
    ], saved_effects

    page.locator("#shot-effects").click()
    expect(dialog.get_by_text("3 sets · 6 / 16 effects")).to_be_visible()
    dialog.get_by_role("button", name="Close").click()
    expect(dialog).to_be_hidden()


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
            assert not errors, errors
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
            hover_previews = dialog.get_by_label("Hover previews")
            expect(hover_previews).to_be_checked()
            close_picker.hover()
            push_image = push.locator("img")
            push_poster = push_image.get_attribute("style")
            push.hover()
            expect(push).to_have_attribute("data-playing", "")
            page.wait_for_function("""poster => document.querySelector(
                '#effects-dialog [data-effect="push_in"] img').getAttribute('style') !== poster""", arg=push_poster)
            expect(push).to_have_attribute("aria-pressed", "false")
            expect(apply).to_be_disabled()
            page.emulate_media(reduced_motion="reduce")
            expect(hover_previews).to_be_checked()
            hover_previews.uncheck()
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
            close_picker.click()
            summary.click()
            expect(hover_previews).to_be_checked()
            push.hover()
            expect(push).to_have_attribute("data-playing", "")
            expect(push).to_have_attribute("aria-pressed", "false")
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
                check_webgpu_depth_and_layers(page)
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
            check_enhancement(page, fixture)
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


def run_masked_export():
    with TemporaryDirectory() as temp:
        folder = Path(temp)
        size = (160, 90)
        image = folder / "shot.jpg"
        Image.new("RGB", size, (50, 50, 50)).save(image)
        masks = folder / ".effect-masks"
        masks.mkdir()
        left_mask = Image.new("RGBA", size)
        ImageDraw.Draw(left_mask).rectangle((0, 0, 49, 89), fill="white")
        left_mask.save(masks / "left.png")
        right_mask = Image.new("RGBA", size)
        ImageDraw.Draw(right_mask).rectangle((110, 0, 159, 89), fill="white")
        right_mask.save(masks / "right.png")

        def render_frames(effects):
            timeline = [
                {"index": 0, "start": 0, "end": 1 / FPS, "fade": 0,
                 "effects": [layer["preset"] for layer in effects], "effect_layers": effects},
                {"index": 1, "start": 1 / FPS, "end": 4 / FPS, "fade": 2 / FPS, "effects": []},
            ]
            frames = effects_export.capture_frames(folder, timeline, [], [image, image], size, 4)
            try:
                stream, count = next(frames)
                assert count == 4
                return [Image.open(BytesIO(b"\xff\xd8" + frame)).copy().convert("RGB")
                        for frame in stream.read_bytes().split(b"\xff\xd8")[1:]]
            finally:
                frames.close()

        inside_images = render_frames([
            {"preset": "push_in", "mode": "full_frame"},
            *({"preset": "cold_tint", "mode": "inside", "mask_files": ["left.png", "right.png"], "feather": 0}
              for _ in range(3)),
        ])
        left = [frame.getpixel((18, 45))[2] for frame in inside_images]
        center = [frame.getpixel((80, 45))[2] for frame in inside_images]
        right = [frame.getpixel((140, 45))[2] for frame in inside_images]
        assert left[0] > left[2] + 5 and left[2] > left[3] + 5, left
        assert right[0] > right[2] + 5 and right[2] > right[3] + 5, right
        assert max(center) - min(center) <= 4, center

        outside_images = render_frames([
            *({"preset": "cold_tint", "mode": "outside", "mask_files": ["left.png", "right.png"], "feather": 0}
              for _ in range(3)),
        ])
        out_left = outside_images[0].getpixel((18, 45))[2]
        out_center = outside_images[0].getpixel((80, 45))[2]
        out_right = outside_images[0].getpixel((140, 45))[2]
        assert out_center > out_left + 10 and out_center > out_right + 10, (out_left, out_center, out_right)
    print("PASS: stacked effects share multi-region union masks and fade with their source shot during export")


def run_depth_export():
    with TemporaryDirectory() as temp:
        folder = Path(temp)
        size = (160, 90)
        image = folder / "shot.png"
        Image.new("RGB", size, (24, 24, 24)).save(image)
        depth_img, depth_meta = _encode_split_depth_png(size, z_left=1.5, z_right=25.0)
        depth_meta = {**depth_meta, "subject_z": 3.2}
        depth_dir = folder / ".effect-depth"
        depth_dir.mkdir()
        depth_file = f"{image.name}.{'a' * 16}.depth.png"
        depth_img.save(depth_dir / depth_file)
        timeline = [{
            "index": 0, "start": 0, "end": 2 / FPS, "fade": 0, "effects": ["fog"],
            "depth_file": depth_file, "depth_meta": depth_meta,
        }]
        frames = effects_export.capture_frames(folder, timeline, [], [image], size, 2)
        try:
            stream, count = next(frames)
            assert count == 2
            exported = [Image.open(BytesIO(b"\xff\xd8" + frame)).copy().convert("RGB")
                        for frame in stream.read_bytes().split(b"\xff\xd8")[1:]]
        finally:
            frames.close()
        rgb = exported[1]
        left_luma = sum(sum(rgb.getpixel((x, y))) / 3.0 for x in range(20, 60, 4) for y in range(45, 80, 4))
        right_luma = sum(sum(rgb.getpixel((x, y))) / 3.0 for x in range(100, 140, 4) for y in range(45, 80, 4))
        assert right_luma > left_luma + 50, (left_luma, right_luma)
    print("PASS: depth-enhanced export serves /depth maps and applies volumetric fog")


if __name__ == "__main__":
    run(webgpu="--webgpu" in sys.argv)
    run_export()
    run_masked_export()
    run_depth_export()
