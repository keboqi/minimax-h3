import test from "node:test";
import assert from "node:assert/strict";
import {
  cameraFilter,
  cameraMotion,
  cameraTransform,
  colorizeDepthRgba,
  cueElapsed,
  effectCues,
  effectsEnabled,
  isSubtleFoleySound,
  isThunderSound,
  lightningCueElapsed,
  paintEffects,
  preloadEffectDepths,
  resolveFocusDepthMeta,
} from "../h3_ui/drama3/page/effects.js";
import {createEffectsPass, layersToSets, setsToLayers} from "../h3_ui/drama3/page/effects_pass.js";
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
      {index: 0, start: 0, end: 4, fade: 0, url: "shot-a.png", effects: ["push_in"]},
      {index: 2, start: 8, end: 12, fade: 1, url: "shot-c.png", effects: ["pull_out"]},
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
  const ops = [];
  const gradient = {addColorStop() {}};
  const layer = {
    beginPath() {},
    moveTo() { this.effect = "rain"; },
    lineTo() {},
    stroke() {},
    clearRect() { this.effect = null; },
    fillRect() { this.effect ??= "vignette"; },
    createRadialGradient() { return gradient; },
    createLinearGradient() { return gradient; },
  };
  layer.canvas = {getContext: () => layer};
  const ctx = {
    save() {},
    restore() { this.globalAlpha = 1; this.globalCompositeOperation = "source-over"; },
    clearRect() {},
    drawImage(buffer) {
      assert.equal(buffer, layer.canvas);
      ops.push([layer.effect, this.globalAlpha, this.globalCompositeOperation]);
    },
  };
  const canvas = {clientWidth: 640, clientHeight: 360, getContext: () => ctx,
    ownerDocument: {createElement: () => layer.canvas}};
  // Outgoing shot has rain; incoming shot has vignette; 35% through dissolve.
  paintEffects(canvas, ["vignette"], 0.35, null, {
    fading: true,
    fadeMix: 0.35,
    backEffects: ["rain"],
    backElapsed: 4.35,
    shotIndex: 1,
    backShotIndex: 0,
  });
  assert.deepEqual(ops, [["rain", 0.65, "lighter"], ["vignette", 0.35, "lighter"]]);

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
  assert.deepEqual(ops, [["rain", 0.75, "lighter"]]);

  ops.length = 0;
  paintEffects(canvas, ["vignette"], 0.25, null, {
    fading: true,
    fadeMix: 0.25,
    backEffects: [],
  });
  assert.deepEqual(ops, [["vignette", 0.25, "lighter"]]);
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
        {index: 0, start: 0, end: 4, fade: 0, url: "shot-a.png", effects: []},
        {index: 1, start: 4, end: 8, fade: 0, url: "shot-b.png", effects: ["impact_shake", "flash"]},
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

test("reused still URLs can have different effects on each shot", (t) => {
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
      {index: 0, start: 0, end: 4, fade: 0, url: "shared.png", effects: ["push_in"]},
      {index: 1, start: 4, end: 8, fade: 0, url: "shared.png", effects: ["pull_out"]},
    ]},
  }}};
  const {paintStills} = createVideo({state, $});

  paintStills(0);
  assert.equal(back.style.transform, "scale(1)");
  paintStills(4);
  assert.equal(back.style.transform, "scale(1.08)");
});

test("effect sets group shared targets/settings, separate differing ones, and enforce full-frame camera", () => {
  const catalog = {
    push_in: {kind: "camera", modes: ["full_frame"]},
    pull_out: {kind: "camera", modes: ["full_frame"]},
    rain: {kind: "atmosphere", modes: ["full_frame", "inside", "outside", "behind"]},
    fog: {kind: "atmosphere", modes: ["full_frame", "inside", "outside", "behind"]},
    lightning: {kind: "lighting", modes: ["full_frame", "inside", "outside"]},
    cold_tint: {kind: "lighting", modes: ["full_frame", "inside", "outside"]},
  };

  const fullFrameCombo = [
    {preset: "push_in", mode: "full_frame"},
    {preset: "lightning", mode: "full_frame", intensity: 0.7},
    {preset: "rain", mode: "full_frame", intensity: 0.7},
  ];
  const sets1 = layersToSets(fullFrameCombo, catalog);
  assert.deepEqual(sets1, [
    {mode: "full_frame", region_ids: [], presets: ["push_in", "lightning", "rain"], intensity: 0.7, feather: undefined},
  ]);
  assert.deepEqual(setsToLayers(sets1, catalog), fullFrameCombo);

  const differentIntensitiesAndFeathers = [
    {preset: "rain", mode: "full_frame", intensity: 0.7},
    {preset: "fog", mode: "full_frame", intensity: 0.4},
    {preset: "rain", mode: "inside", region_id: "region_1", feather: 1},
    {preset: "cold_tint", mode: "inside", region_id: "region_1", feather: 8},
  ];
  const sets2 = layersToSets(differentIntensitiesAndFeathers, catalog);
  assert.equal(sets2.length, 4);
  assert.deepEqual(setsToLayers(sets2, catalog), differentIntensitiesAndFeathers);

  const multiRegion = [
    {preset: "rain", mode: "inside", region_ids: ["region_1", "region_2"]},
    {preset: "cold_tint", mode: "inside", region_ids: ["region_1", "region_2"]},
  ];
  const sets3 = layersToSets(multiRegion, catalog);
  assert.deepEqual(sets3, [
    {mode: "inside", region_ids: ["region_1", "region_2"], presets: ["rain", "cold_tint"], intensity: undefined, feather: undefined},
  ]);
  assert.deepEqual(setsToLayers(sets3, catalog), multiRegion);

  // Camera presets always flatten as full_frame with no region_id, region_ids, intensity, or feather.
  assert.deepEqual(
    setsToLayers([{mode: "inside", region_ids: ["region_1", "region_2"], presets: ["pull_out"], intensity: 0.5, feather: 10}], catalog),
    [{preset: "pull_out", mode: "full_frame"}],
  );
  // Single region canonicalizes to region_id.
  assert.deepEqual(
    setsToLayers([{mode: "inside", region_ids: ["region_1"], presets: ["rain"]}], catalog),
    [{preset: "rain", mode: "inside", region_id: "region_1"}],
  );
  // Empty region_ids on a masked set falls back cleanly to full_frame without feather.
  assert.deepEqual(
    setsToLayers([{mode: "inside", region_ids: [], presets: ["rain"], feather: 6}], catalog),
    [{preset: "rain", mode: "full_frame"}],
  );
});

test("preloadEffectDepths caches URLs and settles cleanly on decode errors", async (t) => {
  const created = [];
  class FakeImage {
    set src(value) {
      this._src = value;
      created.push(value);
    }
    get src() { return this._src; }
    decode() {
      return this._src.includes("broken") ? Promise.reject(new Error("bad depth png")) : Promise.resolve();
    }
  }
  const previous = Object.getOwnPropertyDescriptor(globalThis, "Image");
  Object.defineProperty(globalThis, "Image", {value: FakeImage, configurable: true});
  t.after(() => previous ? Object.defineProperty(globalThis, "Image", previous) : delete globalThis.Image);

  await preloadEffectDepths([
    {depth_url: "/depth/shot_1.png"},
    {depth_url: "/depth/shot_1.png"},
    {depth_url: "/depth/broken.png"},
    {depth_url: null},
  ]);
  assert.deepEqual(created, ["/depth/shot_1.png", "/depth/broken.png"]);
});

test("camera-only shots with depth stay Camera only when WebGPU is unavailable", () => {
  const clears = [];
  const ctx = {clearRect: (...args) => clears.push(args)};
  const canvas = {clientWidth: 320, clientHeight: 180, width: 320, height: 180, getContext: () => ctx};
  const backends = [];
  paintEffects(canvas, ["push_in"], 0.5, (backend) => backends.push(backend), {
    cam: {tx: 0, ty: 0, scale: 1.04, shake: 0, angle: 0, blur: 0},
    depthUrl: "/depth/shot_1.png",
    depthMeta: {inv_min: 0, inv_max: 0.5, depth_strength: 1},
  });
  assert.deepEqual(backends, [null]);
});

test("colorizeDepthRgba maps near inverse depth to bright warm tones and far depth to dark indigo", () => {
  const pixels = new Uint8ClampedArray([
    255, 255, 0, 255, // nearest (u = 1)
    0, 0, 255, 255,   // farthest / sky (u = 0)
  ]);
  colorizeDepthRgba(pixels);
  // Nearest pixel: bright warm yellow
  assert.ok(pixels[0] > 240 && pixels[1] > 230 && pixels[3] === 255);
  // Farthest pixel: deep dark indigo
  assert.ok(pixels[4] < 25 && pixels[5] < 20 && pixels[6] > 30 && pixels[7] === 255);
});

test("shot effects edit popup toggles Show depth / Hide depth alongside Show segments", (t) => {
  const makeNode = (tag) => ({
    tag, style: {}, dataset: {}, children: [],
    removeAttribute() {}, setAttribute(k, v) { this[k] = v; },
    decode: () => Promise.resolve(),
    getContext: () => null,
    replaceChildren(...nodes) { this.children = nodes; },
    querySelector: (selector) => selector === ".style-body" ? {scrollTop: 0} : null,
    showModal() { this.open = true; },
    close() { this.open = false; },
  });
  const previousDoc = Object.getOwnPropertyDescriptor(globalThis, "document");
  const previousImg = Object.getOwnPropertyDescriptor(globalThis, "Image");
  Object.defineProperty(globalThis, "document", {value: {createElement: makeNode}, configurable: true});
  Object.defineProperty(globalThis, "Image", {value: class { set src(v) { this._src = v; } decode() { return Promise.resolve(); } }, configurable: true});
  t.after(() => {
    previousDoc ? Object.defineProperty(globalThis, "document", previousDoc) : delete globalThis.document;
    previousImg ? Object.defineProperty(globalThis, "Image", previousImg) : delete globalThis.Image;
  });

  const h = (tag, attrs = {}, ...children) => ({tag, ...attrs,
    children: children.flat(Infinity).filter((child) => child != null && child !== false)});
  const walk = (node) => typeof node === "object" ? [node, ...(node.children || []).flatMap(walk)] : [];
  const text = (node) => typeof node === "object" ? (node.children || []).map(text).join("") : String(node);
  const dialog = makeNode("dialog");
  const $ = (id) => id === "effect-pass-dialog" ? dialog : null;
  const shot = {
    shot_id: "shot_1", index: 0, image: "shot_1.jpg", segmented: true,
    regions: [{id: "r1", label: "person", position: "center", url: "/mask/r1.png"}],
    effects: [], default: [],
  };
  const state = {
    boot: {gpu: true, video: {enhancement_presets: {}, enhancement_max_layers: 16}},
    project: {
      name: "History",
      result: {sounds: []},
      video: {
        orientation: "landscape",
        plan: {rows: [{index: 0, start: 0, end: 4, image: "/still/shot_1.jpg"}]},
        stills: {shots: [{index: 0, shot_id: "shot_1", depth_url: "/depth/shot_1.png", depth_meta: {inv_min: 0, inv_max: 0.5}}]},
        effect_enhancement: {shots: [shot]},
      },
    },
  };
  const pass = createEffectsPass({state, $, h, call: () => {}, copy: () => {}, writer: () => ({}), interrupt: () => {}, busy: () => false});
  pass.open(shot);

  const depthBtn = () => walk(dialog).find((n) => n.tag === "button" && (text(n) === "Show depth" || text(n) === "Hide depth"));
  const stage = () => walk(dialog).find((n) => typeof n.class === "string" && n.class.includes("video-stage"));
  assert.equal(text(depthBtn()), "Show depth");
  assert.equal(stage()["data-show-depth"], "false");
  assert.ok(walk(stage()).some((n) => n.class === "pass-depth-overlay"));

  depthBtn().onclick();
  assert.equal(text(depthBtn()), "Hide depth");
  assert.equal(stage()["data-show-depth"], "true");

  depthBtn().onclick();
  assert.equal(text(depthBtn()), "Show depth");
  assert.equal(stage()["data-show-depth"], "false");
});

test("subtle Foley and quiet sounds are suppressed while thunder cues sync lightning", () => {
  assert.equal(isSubtleFoleySound({name: "soft_footsteps", start: 1.0}), true);
  assert.equal(isSubtleFoleySound({name: "脚步声", start: 1.5}), true);
  assert.equal(isSubtleFoleySound({name: "distant_knock", start: 2.0, volume: 0.5}), true);
  assert.equal(isSubtleFoleySound({name: "door_slam", start: 2.5, volume: 0.6}), false);
  assert.equal(isThunderSound({name: "thunderclap", start: 3.0}), true);
  assert.equal(isThunderSound({name: "惊雷", start: 3.5}), true);
  assert.equal(isThunderSound({name: "thunder_bed", start: 0, loop: true}), false);

  const shots = [
    {start: 0, end: 5, effects: ["impact_shake", "flash", "lightning"]},
    {start: 5, end: 9, effects: ["lightning"]},
  ];
  const sounds = [
    {name: "soft_footsteps", start: 1.0, loop: false},
    {name: "page_turn", description: "paper rustle", start: 1.8, loop: false},
    {name: "thunderclap", start: 2.5, loop: false},
    {name: "gunshot", start: 3.6, loop: false},
  ];
  const [first, second] = effectCues(shots, sounds);
  assert.deepEqual(first.shake, [0, 2.5, 3.6]);
  assert.deepEqual(first.flash, [0, 2.5, 3.6]);
  assert.deepEqual(first.lightning, [2.5]);
  assert.equal("lightning" in second, false, "shots without thunder fall back to periodic lightning");

  assert.equal(lightningCueElapsed(undefined, 6.0), -1);
  assert.equal(lightningCueElapsed(first.lightning, 1.0), 99, "stays dark prior to first thunder strike");
  assert.ok(Math.abs(lightningCueElapsed(first.lightning, 2.49) - 0.05) < 1e-6, "optical flash leads acoustic thunder by 60ms");
});

test("rack_focus S-curve holds initial defocus plateau and resolveFocusDepthMeta pulls focus across speakers", () => {
  assert.equal(cameraMotion(["rack_focus"], 0.02, 0.05).blur, 1);
  const midBlur = cameraMotion(["rack_focus"], 0.25, 0.59).blur;
  assert.ok(midBlur > 0.45 && midBlur < 0.55, `expected cosine midpoint ~0.5, got ${midBlur}`);
  assert.equal(cameraMotion(["rack_focus"], 0.8, 1.1).blur, 0);

  const baseMeta = {subject_z: 3.0, subject_uv: [0.5, 0.45], rack_start_z: 10.0};
  const focusCues = [
    {time: 0, z: 2.0, uv: [0.25, 0.5]},
    {time: 2.0, z: 8.0, uv: [0.75, 0.4]},
  ];
  const startMeta = resolveFocusDepthMeta(baseMeta, focusCues, 0.5);
  assert.equal(startMeta.subject_z, 2.0);
  assert.deepEqual(startMeta.subject_uv, [0.25, 0.5]);

  const midPull = resolveFocusDepthMeta(baseMeta, focusCues, 2.425);
  // Midpoint in diopters between 1/2.0 (0.5) and 1/8.0 (0.125) is 0.3125 -> z = 3.2m.
  assert.ok(Math.abs(midPull.subject_z - 3.2) < 0.05, `expected diopter midpoint ~3.2m, got ${midPull.subject_z}`);
  assert.ok(Math.abs(midPull.subject_uv[0] - 0.5) < 0.02);

  const settledPull = resolveFocusDepthMeta(baseMeta, focusCues, 3.2);
  assert.equal(settledPull.subject_z, 8.0);
  assert.deepEqual(settledPull.subject_uv, [0.75, 0.4]);
});

