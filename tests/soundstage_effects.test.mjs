import test from "node:test";
import assert from "node:assert/strict";
import {cameraFilter, cameraMotion, cameraTransform, cueElapsed, effectCues, effectsEnabled, paintEffects} from "../h3_ui/drama3/page/effects.js";
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
  assert.equal(front.style.transform, "scale(1.077)");
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
    "rain", "snow", "dust", "embers", "fog", "sunbeams", "smoke", "petals", "underwater", "fireflies",
    "lightning", "flash", "warm_glow", "cold_tint", "flicker", "vignette", "film_grain", "dream_bloom", "scanlines", "spotlight",
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
  assert.deepEqual(Object.keys(ATMO_IDS).sort(),
    ["dust", "embers", "fireflies", "fog", "petals", "rain", "smoke", "snow", "sunbeams", "underwater"]);
  assert.deepEqual(Object.keys(LIGHT_IDS).sort(),
    ["cold_tint", "dream_bloom", "film_grain", "flash", "flicker", "lightning", "scanlines", "spotlight", "vignette", "warm_glow"]);
});

test("dissolve crossfades outgoing and incoming overlays instead of cutting at shot start", () => {
  const alphas = [];
  const ops = [];
  const gradient = {addColorStop() {}};
  let currentAlpha = 1;
  const ctx = {
    get globalAlpha() { return currentAlpha; },
    set globalAlpha(v) { currentAlpha = v; alphas.push(Math.round(v * 100) / 100); },
    save() { this._saved = currentAlpha; },
    restore() { currentAlpha = this._saved ?? 1; },
    clearRect() {},
    beginPath() {},
    moveTo() { ops.push(["moveTo", Math.round(currentAlpha * 100) / 100]); },
    lineTo() {},
    stroke() {},
    fillRect() { ops.push(["fillRect", Math.round(currentAlpha * 100) / 100]); },
    createRadialGradient() { return gradient; },
    createLinearGradient() { return gradient; },
  };
  const canvas = {clientWidth: 640, clientHeight: 360, getContext: () => ctx};
  // Outgoing shot has rain; incoming shot has vignette; 35% through dissolve.
  paintEffects(canvas, ["vignette"], 0.35, null, {
    fading: true,
    fadeMix: 0.35,
    backEffects: ["rain"],
    backElapsed: 4.35,
    shotIndex: 1,
    backShotIndex: 0,
  });
  assert.ok(ops.some(([op, a]) => op === "moveTo" && a === 0.65), "outgoing rain fades out at 0.65 alpha");
  assert.ok(ops.some(([op, a]) => op === "fillRect" && a === 0.35), "incoming vignette fades in at 0.35 alpha");

  // Outgoing shot has rain; incoming shot has no effects; rain still fades out during dissolve.
  ops.length = 0;
  let backend = null;
  paintEffects(canvas, [], 0.25, (b) => { backend = b; }, {
    fading: true,
    fadeMix: 0.25,
    backEffects: ["rain"],
    backElapsed: 4.25,
  });
  assert.equal(backend, "Canvas 2D");
  assert.ok(ops.some(([op, a]) => op === "moveTo" && a === 0.75), "outgoing rain continues fading into a clean shot");

  ops.length = 0;
  paintEffects(canvas, ["vignette"], 0.25, null, {
    fading: true,
    fadeMix: 0.25,
    backEffects: [],
  });
  assert.deepEqual(ops, [["fillRect", 0.25]], "incoming vignette fades in from a clean shot");
});

test("shot-seeded lightning varies flash timing across shots while keeping shot 0 deterministic", () => {
  const ctx = {clearRect() {}, fillRect() {}};
  const canvas = {clientWidth: 640, clientHeight: 360, getContext: () => ctx};
  paintEffects(canvas, ["lightning"], 0.12, null, {shotIndex: 0});
  const shot0Style = ctx.fillStyle;
  assert.equal(shot0Style, "rgba(225,237,255,0.32)");

  paintEffects(canvas, ["lightning"], 0.12, null, {shotIndex: 2});
  const shot2Style = ctx.fillStyle;
  assert.notEqual(shot2Style, shot0Style, "subsequent storm shots do not all flash at 0.12s");
});

test("cameraMotion exposes coupled pan, zoom, and impact shake vectors", () => {
  assert.deepEqual(cameraMotion([], 0.5), {tx: 0, ty: 0, scale: 1, shake: 0, angle: 0, blur: 0});
  const pan = cameraMotion(["pan_left"], 0);
  assert.equal(pan.tx, 3);
  assert.equal(pan.scale, 1.08);
  const shakeStart = cameraMotion(["impact_shake"], 0.05, 0.05);
  assert.ok(shakeStart.shake > 0.7, "impact shake reports active decay energy");
  const shakeDone = cameraMotion(["impact_shake"], 0.8, 0.6);
  assert.equal(shakeDone.shake, 0);

  const tilt0 = cameraMotion(["dutch_tilt"], 0);
  const tilt1 = cameraMotion(["dutch_tilt"], 1);
  assert.equal(tilt0.angle, -1.5);
  assert.equal(tilt1.angle, 1.5);
  assert.match(cameraTransform(["dutch_tilt"], 0), /rotate\(-1\.5deg\) scale\(1\.04\)/);

  const focusStart = cameraMotion(["rack_focus"], 0.05, 0.05);
  assert.ok(focusStart.blur > 0.8, "rack focus starts softly defocused");
  assert.match(cameraFilter(["rack_focus"], 0.05, 0.05), /^blur\(.+px\)$/);
  const focusEnd = cameraMotion(["rack_focus"], 0.8, 1.2);
  assert.equal(focusEnd.blur, 0);
  assert.equal(cameraFilter(["rack_focus"], 0.8, 1.2), "none");
});

test("one-shot sound cues inside a shot deterministically trigger impact_shake and flash on the cue timestamp", (t) => {
  const sounds = [
    {name: "wind_bed", start: 0, end: 10, loop: true},
    {name: "prism_shatter", start: 6.0, end: 6.8, loop: false},
  ];
  // Looped beds never re-trigger; before the 6.0s one-shot cue, elapsed from shotStart (4.0) is used.
  const [{shake}] = effectCues([{start: 4, end: 8, effects: ["impact_shake"]}], sounds);
  assert.ok(Math.abs(cueElapsed(shake, 5.2) - 1.2) < 1e-6);
  // At 6.05s (elapsed = 2.05s into the [4, 8) shot), the 6.0s one-shot cue just fired 0.05s ago.
  assert.ok(Math.abs(cueElapsed(shake, 6.05) - 0.05) < 1e-6);

  // Verify paintStills re-triggers impact_shake and flash at t = 6.05s when seeking.
  const globals = {window: {}, document: {querySelectorAll: () => []}, Image: class {}};
  for (const [name, value] of Object.entries(globals)) {
    const previous = Object.getOwnPropertyDescriptor(globalThis, name);
    Object.defineProperty(globalThis, name, {value, configurable: true});
    t.after(() => previous ? Object.defineProperty(globalThis, name, previous) : delete globalThis[name]);
  }
  const back = {dataset: {}, style: {}}, front = {dataset: {}, style: {}};
  const ctx = {clearRect() {}, fillRect() {}};
  const overlay = {clientWidth: 640, clientHeight: 360, getContext: () => ctx};
  const $ = (id) => ({"still-back": back, "still-front": front, "still-effects": overlay})[id];
  const state = {project: {
    result: {sounds},
    video: {
      plan: {rows: [
        {index: 0, start: 0, image: "shot-a.png", effects: []},
        {index: 1, start: 4, image: "shot-b.png", effects: ["impact_shake", "flash"]},
      ]},
      stills: {shots: [
        {start: 0, end: 4, fade: 0, url: "shot-a.png"},
        {start: 4, end: 8, fade: 0, url: "shot-b.png"},
      ]},
    },
  }};
  const {paintStills} = createVideo({state, $});

  // At 5.2s (1.2s after shot start, before the 6.0s sound cue), impact_shake and flash have settled.
  paintStills(5.2);
  assert.equal(back.style.transform, "scale(1)");
  assert.equal(ctx.fillStyle, "rgba(225,237,255,0)");

  paintStills(6);
  assert.equal(cameraMotion(["impact_shake"], 0.5, 0).shake, 1);
  assert.equal(back.style.transform, "translate(0%, 1.1%) scale(1.03)");

  // At 6.05s (0.05s after prism_shatter one-shot sound cue), both impact_shake and flash fire.
  paintStills(6.05);
  assert.notEqual(back.style.transform, "scale(1)");
  const flashAlpha = Number(ctx.fillStyle.match(/,([^,]+)\)$/)[1]);
  assert.ok(flashAlpha > 0.25, `expected flash pulse on sound cue, got ${ctx.fillStyle}`);
});

test("explicit seconds=0 is respected for rack_focus and subtle_sway", () => {
  assert.equal(cameraFilter(["rack_focus"], 0.5, 0), "blur(4.5px)");
  assert.equal(cameraFilter(["rack_focus"], 0.5), "none");
  assert.equal(cameraMotion(["subtle_sway"], 0.5, 0).tx, 0);
  assert.equal(cameraMotion(["subtle_sway"], 0.5, 0).ty, 0.55);
});

test("dense cues retrigger shake and space flashes across a cut", () => {
  const effects = ["impact_shake", "flash"];
  const shots = [{start: 0, end: 1, effects}, {start: 1, end: 2, effects}];
  const [first, second] = effectCues(shots, [
    {start: 1.75}, {start: 0.75}, {start: 0.25}, {start: 1.25}, {start: 1.5, loop: true},
  ]);
  assert.deepEqual(first, {shake: [0, 0.25, 0.75], flash: [0, 0.75]});
  assert.deepEqual(second, {shake: [1, 1.25, 1.75], flash: [1.25, 1.75]});
  assert.ok(cueElapsed(second.flash, 1.1) >= 0.45);
  assert.equal(cueElapsed(second.shake, 1.125), 0.125);
  assert.equal(cueElapsed(second.flash, 1.25), 0);
});

test("reused still URLs resolve effects for the matching shot start time", (t) => {
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
      {index: 0, start: 0, image: "shared.png", effects: ["push_in"]},
      {index: 1, start: 4, image: "shared.png", effects: ["pull_out"]},
    ]},
    stills: {shots: [
      {start: 0, end: 4, fade: 0, url: "shared.png"},
      {start: 4, end: 8, fade: 0, url: "shared.png"},
    ]},
  }}};
  const {paintStills} = createVideo({state, $});

  paintStills(0);
  assert.equal(back.style.transform, "scale(1)");
  paintStills(4);
  assert.equal(back.style.transform, "scale(1.08)");
});

