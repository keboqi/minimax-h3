import test from "node:test";
import assert from "node:assert/strict";
import {cameraTransform, effectsEnabled, paintEffects} from "../h3_ui/drama3/page/effects.js";
import {ATMO_IDS, LIGHT_IDS} from "../h3_ui/drama3/page/effects_webgpu.js";
import {createVideo} from "../h3_ui/drama3/page/video.js";

test("camera moves follow the displayed stills across an undrawn shot and a dissolve", (t) => {
  const globals = {window: {}, document: {querySelectorAll: () => []}, Image: class {}};
  for (const [name, value] of Object.entries(globals)) {
    const previous = Object.getOwnPropertyDescriptor(globalThis, name);
    Object.defineProperty(globalThis, name, {value, configurable: true});
    t.after(() => previous ? Object.defineProperty(globalThis, name, previous) : delete globalThis[name]);
  }
  const back = {dataset: {}, style: {}}, front = {dataset: {}, style: {}};
  const $ = (id) => ({"still-back": back, "still-front": front})[id];
  const state = {project: {video: {
    plan: {rows: [
      {index: 0, start: 0, image: "shot-a.png", effects: ["push_in"]},
      {index: 1, start: 4, image: null, effects: ["pan_left"]},
      {index: 2, start: 8, image: "shot-c.png", effects: ["pull_out"]},
    ]},
    stills: {shots: [
      {start: 0, end: 4, fade: 0, url: "shot-a.png"},
      {start: 8, end: 12, fade: 1, url: "shot-c.png"},
    ]},
  }}};
  const {paintStills} = createVideo({state, $});

  paintStills(6);
  assert.equal(back.src, "shot-a.png");
  assert.equal(back.style.transform, "scale(1.08)");
  paintStills(8.5);
  assert.equal(back.style.transform, "scale(1.08)");
  assert.equal(front.src, "shot-c.png");
  assert.equal(front.style.opacity, "0.5");
  assert.equal(front.style.transform, "scale(1.07)");
});

test("reduced motion pauses effects unless explicitly overridden for the project", () => {
  assert.equal(effectsEnabled({}, false), true);
  assert.equal(effectsEnabled({}, true), false);
  assert.equal(effectsEnabled({effects_enabled: true}, true), false);
  assert.equal(effectsEnabled({effects_override_reduced_motion: true}, true), true);
  assert.equal(effectsEnabled({effects_enabled: false, effects_override_reduced_motion: true}, true), false);
});

test("camera moves clamp to shot boundaries and old plans remain still", () => {
  assert.equal(cameraTransform([], 0.5), "none");
  assert.equal(cameraTransform(["push_in"], -1), "scale(1)");
  assert.equal(cameraTransform(["push_in"], 2), "scale(1.08)");
  assert.equal(cameraTransform(["pull_out"], 0), "scale(1.08)");
  assert.match(cameraTransform(["pan_left"], 1), /translateX\(-3%\)/);
  assert.match(cameraTransform(["pan_up"], 0), /translateY\(-3%\)/);
  assert.match(cameraTransform(["pan_up"], 1), /translateY\(3%\)/);
  assert.match(cameraTransform(["pan_down"], 0), /translateY\(3%\)/);
  assert.match(cameraTransform(["pan_down"], 1), /translateY\(-3%\)/);
  assert.match(cameraTransform(["subtle_sway"], 0.5, 1.2), /translate\(.+\) scale\(1\.04\)/);
  assert.notEqual(cameraTransform(["impact_shake"], 0.1, 0.05), "scale(1)");
  assert.equal(cameraTransform(["impact_shake"], 0.5, 0.6), "scale(1)");
});

test("overlays reconstruct the same frame after seeking and clear when removed", () => {
  const calls = [];
  const gradient = {addColorStop() {}};
  const ctx = new Proxy({}, {get: (_, name) => (...args) => {
    calls.push([name, ...args]);
    if (name === "createRadialGradient") return gradient;
  }, set: (_, name, value) => {calls.push([name, value]); return true;}});
  const canvas = {clientWidth: 640, clientHeight: 360, getContext: () => ctx};
  for (const effect of [
    "rain", "snow", "dust", "embers", "fog", "sunbeams", "smoke", "petals",
    "lightning", "flash", "warm_glow", "cold_tint", "flicker", "vignette", "film_grain", "dream_bloom", "scanlines",
  ]) {
    calls.length = 0; paintEffects(canvas, [effect], 2.4); const first = [...calls];
    paintEffects(canvas, [effect], 7);
    calls.length = 0; paintEffects(canvas, [effect], 2.4);
    assert.deepEqual(calls, first);
  }
  calls.length = 0; paintEffects(canvas, [], 3);
  assert.deepEqual(calls, [["clearRect", 0, 0, 640, 360]]);
});

test("lightning reaches short shots and repeats on the same media clock", () => {
  const ctx = {clearRect() {}, fillRect() {}};
  const canvas = {clientWidth: 640, clientHeight: 360, getContext: () => ctx};
  paintEffects(canvas, ["lightning"], 0.12);
  const peak = ctx.fillStyle;
  assert.equal(peak, "rgba(225,237,255,0.32)");
  paintEffects(canvas, ["lightning"], 0.6);
  assert.equal(ctx.fillStyle, "rgba(225,237,255,0)");
  paintEffects(canvas, ["lightning"], 7.12);
  assert.ok(Number(ctx.fillStyle.match(/,([^,]+)\)$/)[1]) > 0.31);
});

test("layered atmosphere and lighting overlays both render in one pass", () => {
  const calls = [];
  const gradient = {addColorStop() {}};
  const ctx = new Proxy({}, {get: (_, name) => (...args) => {
    calls.push(name);
    if (name === "createRadialGradient" || name === "createLinearGradient") return gradient;
  }, set: () => true});
  const canvas = {clientWidth: 640, clientHeight: 360, getContext: () => ctx};
  paintEffects(canvas, ["rain", "lightning"], 0.12);
  assert.ok(calls.includes("moveTo"), "expected rain strokes");
  assert.ok(calls.includes("fillRect"), "expected lightning fill");
});

test("shader IDs cover all overlays", () => {
  assert.deepEqual(Object.keys(ATMO_IDS).sort(), ["dust", "embers", "fog", "petals", "rain", "smoke", "snow", "sunbeams"]);
  assert.deepEqual(Object.keys(LIGHT_IDS).sort(),
    ["cold_tint", "dream_bloom", "film_grain", "flash", "flicker", "lightning", "scanlines", "vignette", "warm_glow"]);
});
