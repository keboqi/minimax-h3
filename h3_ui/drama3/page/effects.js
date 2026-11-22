import { ATMO_IDS, LIGHT_IDS, clearWebGPU, gpuStatus, probeWebGPU, renderWebGPU } from "./effects_webgpu.js";

// Animation derives from media time: pausing freezes it; seeking reconstructs it.
const fraction = (value) => value - Math.floor(value);
const clamp01 = (value) => Math.max(0, Math.min(1, value));
const easeInOut = (p) => 0.5 - 0.5 * Math.cos(Math.PI * clamp01(p));
const round3 = (value) => Math.round(value * 1000) / 1000;

const IDENTITY_CAMERA = Object.freeze({ tx: 0, ty: 0, scale: 1, shake: 0, angle: 0, blur: 0 });

function rackBlurDecay(p, t) {
  const time = t ?? p * 4;
  if (time < 0 || time >= 0.95) return 0;
  return round3(Math.pow(1 - time / 0.95, 2));
}

// Single source of truth for camera kinematics; drives both CSS transforms and WebGPU/Canvas2D camera coupling.
const CAMERA_MOVES = {
  push_in: (p) => {
    const scale = round3(1 + easeInOut(p) * 0.08);
    return { css: `scale(${scale})`, tx: 0, ty: 0, scale, shake: 0, angle: 0, blur: 0 };
  },
  pull_out: (p) => {
    const scale = round3(1.08 - easeInOut(p) * 0.08);
    return { css: `scale(${scale})`, tx: 0, ty: 0, scale, shake: 0, angle: 0, blur: 0 };
  },
  pan_left: (p) => {
    const tx = round3(3 - easeInOut(p) * 6);
    return { css: `translateX(${tx}%) scale(1.08)`, tx, ty: 0, scale: 1.08, shake: 0, angle: 0, blur: 0 };
  },
  pan_right: (p) => {
    const tx = round3(-3 + easeInOut(p) * 6);
    return { css: `translateX(${tx}%) scale(1.08)`, tx, ty: 0, scale: 1.08, shake: 0, angle: 0, blur: 0 };
  },
  pan_up: (p) => {
    const ty = round3(-3 + easeInOut(p) * 6);
    return { css: `translateY(${ty}%) scale(1.08)`, tx: 0, ty, scale: 1.08, shake: 0, angle: 0, blur: 0 };
  },
  pan_down: (p) => {
    const ty = round3(3 - easeInOut(p) * 6);
    return { css: `translateY(${ty}%) scale(1.08)`, tx: 0, ty, scale: 1.08, shake: 0, angle: 0, blur: 0 };
  },
  subtle_sway: (p, t) => {
    const time = t ?? p * 4;
    const tx = round3(Math.sin(time * 0.9) * 0.85);
    const ty = round3(Math.cos(time * 0.65) * 0.55);
    return { css: `translate(${tx}%, ${ty}%) scale(1.04)`, tx, ty, scale: 1.04, shake: 0, angle: 0, blur: 0 };
  },
  impact_shake: (p, t) => {
    const time = t ?? p * 4;
    if (time < 0 || time >= 0.45) {
      return { css: "scale(1)", tx: 0, ty: 0, scale: 1, shake: 0, angle: 0, blur: 0 };
    }
    const decay = Math.pow(1 - time / 0.45, 2);
    const tx = round3(Math.sin(time * 58) * 1.6 * decay);
    const ty = round3(Math.cos(time * 47) * 1.1 * decay);
    const scale = round3(1 + 0.03 * decay);
    return {
      css: `translate(${tx}%, ${ty}%) scale(${scale})`,
      tx,
      ty,
      scale,
      shake: round3(decay),
      angle: 0,
      blur: 0,
    };
  },
  dutch_tilt: (p) => {
    const angle = round3(-1.5 + easeInOut(p) * 3);
    const scale = round3(1.04 + p * 0.04);
    return { css: `rotate(${angle}deg) scale(${scale})`, tx: 0, ty: 0, scale, shake: 0, angle, blur: 0 };
  },
  rack_focus: (p, t) => {
    const scale = round3(1.05 - easeInOut(p) * 0.05);
    return { css: `scale(${scale})`, tx: 0, ty: 0, scale, shake: 0, angle: 0, blur: rackBlurDecay(p, t) };
  },
};

function evaluateCamera(effects, progress, seconds) {
  const p = clamp01(progress);
  const move = effects?.find?.((id) => Object.hasOwn(CAMERA_MOVES, id));
  return move ? CAMERA_MOVES[move](p, seconds) : null;
}

export function cameraFrame(effects, progress, seconds, cueSeconds = seconds) {
  const camera = evaluateCamera(effects, progress, effects?.includes("impact_shake") ? cueSeconds : seconds);
  if (!camera) return STATIC_FRAME;
  return {...camera, filter: camera.blur > 0 ? `blur(${round3(camera.blur * 4.5)}px)` : "none"};
}

const STATIC_FRAME = Object.freeze({...IDENTITY_CAMERA, css: "none", filter: "none"});

const atmosphereEffects = new Set(Object.keys(ATMO_IDS));
const lightEffects = new Set(Object.keys(LIGHT_IDS));

// Precomputed Float32Array seed tables for deterministic, zero-hash particle loops.
const MAX_PARTICLES = 120;
const SEEDS = new Float32Array(MAX_PARTICLES);
const DEPTHS = new Float32Array(MAX_PARTICLES);
const PHASES = new Float32Array(MAX_PARTICLES);
for (let i = 0; i < MAX_PARTICLES; i++) {
  const s = fraction(Math.sin(i * 127.1 + 7) * 43758.5453);
  SEEDS[i] = s;
  DEPTHS[i] = 0.35 + s * 0.65;
  PHASES[i] = fraction(i * 0.618034);
}

let sprites = null;
function makeSprite(size, paint) {
  const c = globalThis.document?.createElement?.("canvas");
  if (!c) return null;
  c.width = size;
  c.height = size;
  const ctx = c.getContext("2d");
  if (!ctx) return null;
  paint(ctx, size);
  return c;
}

function radialSprite(size, stops) {
  return makeSprite(size, (ctx, s) => {
    const g = ctx.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
    for (const [offset, color] of stops) g.addColorStop(offset, color);
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, s, s);
  });
}

function getSprites() {
  if (sprites !== null || !globalThis.document?.createElement) return sprites;
  const fog = radialSprite(128, [[0, "rgba(218,230,235,0.28)"], [0.55, "rgba(218,230,235,0.10)"], [1, "rgba(218,230,235,0)"]]);
  const smoke = radialSprite(128, [[0, "rgba(128,136,146,0.28)"], [0.5, "rgba(116,124,134,0.12)"], [1, "rgba(110,118,128,0)"]]);
  const snow = radialSprite(32, [[0, "rgba(248,252,255,0.95)"], [0.45, "rgba(240,248,255,0.45)"], [1, "rgba(240,248,255,0)"]]);
  const ember = radialSprite(32, [[0, "rgba(255,220,120,0.98)"], [0.35, "rgba(255,130,45,0.65)"], [1, "rgba(255,80,20,0)"]]);
  const firefly = radialSprite(32, [[0, "rgba(255,248,165,0.96)"], [0.42, "rgba(182,250,105,0.52)"], [1, "rgba(150,240,80,0)"]]);
  const caustic = radialSprite(64, [[0, "rgba(105,228,255,0.22)"], [0.55, "rgba(85,205,240,0.08)"], [1, "rgba(65,180,225,0)"]]);
  const bubble = makeSprite(24, (ctx, s) => {
    const r = s * 0.42, cx = s / 2, cy = s / 2;
    const g = ctx.createRadialGradient(cx, cy, r * 0.4, cx, cy, r);
    g.addColorStop(0, "rgba(205,246,255,0.05)");
    g.addColorStop(0.75, "rgba(205,246,255,0.28)");
    g.addColorStop(0.95, "rgba(235,252,255,0.65)");
    g.addColorStop(1, "rgba(205,246,255,0)");
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "rgba(255,255,255,0.85)";
    ctx.beginPath();
    ctx.arc(cx - r * 0.32, cy - r * 0.32, r * 0.16, 0, Math.PI * 2);
    ctx.fill();
  });
  const grain = makeSprite(128, (ctx, s) => {
    const img = ctx.createImageData(s, s);
    for (let i = 0; i < img.data.length; i += 4) {
      const idx = i >> 2;
      const rnd = fraction(Math.sin(idx * 12.9898 + 78.233) * 43758.5453);
      const bright = rnd >= 0.5;
      const v = bright ? 255 : 0;
      const a = Math.round(Math.abs(rnd - 0.5) * 2 * 22);
      img.data[i] = v; img.data[i + 1] = v; img.data[i + 2] = v; img.data[i + 3] = a;
    }
    ctx.putImageData(img, 0, 0);
  });
  sprites = fog && snow && ember ? { fog, smoke, snow, ember, firefly, caustic, bubble, grain } : false;
  return sprites;
}

export function effectTrack(h, {shots, presets, place, pick, picked}) {
  const clips = shots.filter((shot) => shot.effects?.length).map((shot) => {
    const label = shot.effects.map((id) => presets?.[id]?.label || id).join(" · ");
    const title = `Shot ${shot.index + 1} · ${label}`;
    return h("button", {
      type: "button", class: "clip effect-clip", "data-shot": shot.index,
      "aria-label": title, "aria-pressed": picked == null ? null : String(shot.index === picked),
      title, style: place(shot.start, shot.end),
      onclick: (event) => { event.stopPropagation(); pick(shot); },
    }, h("span", {}, label));
  });
  return clips.length ? {
    name: h("span", {class: "lane-name effect-lane-name"}, "Visual effects"),
    lane: h("div", {class: "lane effect-lane", role: "group", "aria-label": "Visual effects track"}, clips),
  } : null;
}

export function effectsEnabled(video, reducedMotion) {
  return video.effects_enabled !== false && (!reducedMotion || video.effects_override_reduced_motion === true);
}

export function lastAtOrBefore(items, time, startOf) {
  let low = 0, high = items.length;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (startOf(items[mid]) <= time) low = mid + 1;
    else high = mid;
  }
  return low - 1;
}

const timestamp = (time) => time;
const SETTLED_CUE_SECONDS = 1;
export function effectCues(shots, sounds) {
  const onsets = (sounds || []).filter((sound) => sound?.start != null && !sound.loop)
    .map((sound) => Number(sound.start)).filter(Number.isFinite).sort((a, b) => a - b);
  let lastFlash = -Infinity;
  return shots.map(({start, end, effects}) => {
    const shake = effects.includes("impact_shake"), flash = effects.includes("flash");
    const candidates = shake || flash ? [start] : [];
    if (candidates.length) {
      for (let i = lastAtOrBefore(onsets, start, timestamp) + 1; i < onsets.length && onsets[i] < end; i++) {
        candidates.push(onsets[i]);
      }
    }
    return {
      shake: shake ? candidates : [],
      // Coalesce dense cues across cuts as well as within a shot. The pulse lasts 0.45s.
      flash: flash ? candidates.filter((onset) => {
        if (onset < lastFlash + 0.5) return false;
        lastFlash = onset;
        return true;
      }) : [],
    };
  });
}

export function cueElapsed(cues, time) {
  const index = lastAtOrBefore(cues, time, timestamp);
  return index < 0 ? SETTLED_CUE_SECONDS : time - cues[index];
}

export function cameraTransform(effects, progress, seconds) {
  return evaluateCamera(effects, progress, seconds)?.css ?? "none";
}

export function cameraFilter(effects, progress, seconds) {
  return cameraFrame(effects, progress, seconds).filter;
}

export function cameraMotion(effects, progress, seconds) {
  const evaluated = evaluateCamera(effects, progress, seconds);
  if (!evaluated) return { ...IDENTITY_CAMERA };
  const { tx, ty, scale, shake, angle, blur } = evaluated;
  return { tx, ty, scale, shake, angle, blur };
}

function withComposite(ctx, mode, draw) {
  if (!ctx.save || !ctx.restore) return draw();
  ctx.save();
  ctx.globalCompositeOperation = mode;
  try { draw(); } finally { ctx.restore(); }
}

function sizeCanvas(canvas, w, h) {
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
}

const emptyEffects = {atmospheres: [], lights: []};
const effectLayers = new WeakMap();
function splitEffects(list) {
  if (!Array.isArray(list) || !list.length) return emptyEffects;
  const cached = effectLayers.get(list);
  if (cached) return cached;
  const atmospheres = [];
  const lights = [];
  for (const id of list) {
    if (atmosphereEffects.has(id)) atmospheres.push(id);
    else if (lightEffects.has(id)) lights.push(id);
  }
  const layers = {atmospheres, lights};
  effectLayers.set(list, layers);
  return layers;
}

// Preview canvases stay independent of the player's WebGPU surface and clock.
export function paintEffectPreview(canvas, image, effects, seconds, duration) {
  const {atmospheres, lights} = splitEffects(effects);
  const camera = cameraFrame(effects, seconds / duration, seconds);
  image.style.transform = camera.css;
  image.style.filter = camera.filter;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const {width, height} = canvas;
  ctx.clearRect(0, 0, width, height);
  paintLayer2D(ctx, atmospheres, lights, seconds, seconds, width, height, 0, camera);
}

let lastPaintArgs = null;
const paintedOverlays = new WeakSet();
const dissolveCanvases = new WeakMap();
const maskImages = new Map();
const stackBuffers = new WeakMap();

export function preloadEffectMasks(shots) {
  const urls = new Set(shots.flatMap((shot) => (shot.effect_layers || [])
    .map((layer) => layer.mask_url).filter(Boolean)));
  const ready = [...urls].map((url) => {
    let entry = maskImages.get(url);
    if (entry) {
      maskImages.delete(url);
    } else {
      const image = new Image();
      image.src = url;
      entry = {image, ready: image.decode()};
    }
    maskImages.set(url, entry);
    return entry.ready;
  });
  // The player can seek across a long project without holding all decoded masks.
  while (maskImages.size > 48) maskImages.delete(maskImages.keys().next().value);
  return Promise.all(ready);
}

function drawInShot(ctx, image, cam, w, h) {
  const scale = Math.min(w / image.naturalWidth, h / image.naturalHeight);
  const width = image.naturalWidth * scale;
  const height = image.naturalHeight * scale;
  ctx.translate(w * (0.5 + cam.tx / 100), h * (0.5 + cam.ty / 100));
  ctx.rotate(cam.angle * Math.PI / 180);
  ctx.scale(cam.scale, cam.scale);
  ctx.drawImage(image, -width / 2, -height / 2, width, height);
}

function paintShotStack(target, scratch, matte, image, layers, seconds, cue, seed, cam) {
  const {width: w, height: h} = target.canvas;
  target.fillStyle = "#000";
  target.fillRect(0, 0, w, h);
  if (!image?.naturalWidth) return;
  target.save();
  if (cam.blur) target.filter = `blur(${cam.blur * 4.5}px)`;
  drawInShot(target, image, cam, w, h);
  target.restore();
  for (const layer of layers) {
    const {preset, mode, mask_url: url} = layer;
    const atmosphere = atmosphereEffects.has(preset);
    if (!atmosphere && !lightEffects.has(preset)) continue;
    const mask = maskImages.get(url)?.image;
    if (mode !== "full_frame" && !mask?.naturalWidth) continue;
    scratch.clearRect(0, 0, w, h);
    paintLayer2D(scratch, atmosphere ? [preset] : [], atmosphere ? [] : [preset], seconds, cue, w, h, seed, cam);
    if (mask) {
      scratch.save();
      scratch.globalCompositeOperation = mode === "inside" ? "destination-in" : "destination-out";
      const feather = layer.feather ?? 1;
      if (feather) scratch.filter = `blur(${feather * Math.max(w, h) / 1280}px)`;
      // Draw through a full-frame matte so destination-in also clears outside the image rectangle.
      matte.clearRect(0, 0, w, h);
      matte.save();
      drawInShot(matte, mask, cam, w, h);
      matte.restore();
      scratch.drawImage(matte.canvas, 0, 0);
      scratch.restore();
    }
    target.save();
    target.globalAlpha = layer.intensity ?? 1;
    target.drawImage(scratch.canvas, 0, 0);
    target.restore();
  }
}

function paintEnhancedScene(canvas, scene, effects, t, cue, fading) {
  clearWebGPU(canvas);
  const ctx = canvas.getContext("2d");
  if (!ctx) return false;
  const width = Math.max(1, Math.round(canvas.clientWidth));
  const height = Math.max(1, Math.round(canvas.clientHeight));
  const scale = Math.min(1, 1280 / Math.max(width, height));
  const w = Math.max(1, Math.round(width * scale)), h = Math.max(1, Math.round(height * scale));
  if (!stackBuffers.has(canvas)) {
    stackBuffers.set(canvas, Array.from({length: 4}, () => canvas.ownerDocument.createElement("canvas").getContext("2d")));
  }
  const [back, front, layer, matte] = stackBuffers.get(canvas);
  for (const context of [ctx, back, front, layer, matte]) sizeCanvas(context.canvas, w, h);
  const fullFrame = (presets) => presets.map((preset) => ({preset, mode: "full_frame"}));
  paintShotStack(back, layer, matte, scene.stillBack,
    fading ? scene.backLayers ?? fullFrame(scene.backEffects) : scene.layers ?? fullFrame(effects),
    fading ? scene.backElapsed : t, fading ? scene.backCueElapsed : cue,
    fading ? scene.backShotIndex : scene.shotIndex, fading ? scene.backCam : scene.cam);
  ctx.drawImage(back.canvas, 0, 0);
  if (fading) {
    paintShotStack(front, layer, matte, scene.stillFront, scene.layers ?? fullFrame(effects),
      t, cue, scene.shotIndex, scene.cam);
    ctx.save();
    ctx.globalAlpha = clamp01(scene.fadeMix);
    ctx.drawImage(front.canvas, 0, 0);
    ctx.restore();
  }
  return true;
}

export function paintEffects(canvas, effects, seconds, showBackend, scene = null) {
  const { atmospheres, lights } = splitEffects(effects);
  const fading = Boolean(scene?.fading && (scene?.fadeMix ?? 1) < 1);
  const { atmospheres: backAtmospheres, lights: backLights } = splitEffects(fading ? scene?.backEffects : null);
  if (scene?.layers || (fading && scene?.backLayers)) {
    lastPaintArgs = null;
    paintedOverlays.add(canvas);
    const painted = paintEnhancedScene(canvas, scene, effects, Math.max(0, seconds),
      Math.max(0, scene.cueElapsed ?? seconds), fading);
    showBackend?.(painted ? "Canvas 2D" : "Unavailable");
    return;
  }

  if (!atmospheres.length && !lights.length && !backAtmospheres.length && !backLights.length) {
    lastPaintArgs = null;
    if (paintedOverlays.delete(canvas)) {
      clearWebGPU(canvas);
      canvas.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
    }
    showBackend?.(null);
    return;
  }
  const width = Math.max(1, Math.round(canvas.clientWidth));
  const height = Math.max(1, Math.round(canvas.clientHeight));
  paintedOverlays.add(canvas);
  const t = Math.max(0, seconds);
  const cueT = Math.max(0, scene?.cueElapsed ?? t);
  lastPaintArgs = [canvas, effects, seconds, showBackend, scene];
  if (gpuStatus() === "idle") {
    probeWebGPU(() => {
      if (lastPaintArgs?.[0].isConnected) paintEffects(...lastPaintArgs);
    });
  }
  const atmoId = ATMO_IDS[atmospheres[0]] || 0;
  const lightId = LIGHT_IDS[lights[0]] || 0;
  const backAtmoId = ATMO_IDS[backAtmospheres[0]] || 0;
  const backLightId = LIGHT_IDS[backLights[0]] || 0;
  const gpuScene = scene ? {
    ...scene,
    fading,
    cueElapsed: cueT,
    backAtmoId,
    backLightId,
  } : null;
  if (
    atmospheres.length <= 1 &&
    lights.length <= 1 &&
    backAtmospheres.length <= 1 &&
    backLights.length <= 1 &&
    (!atmospheres.length || atmoId) &&
    (!lights.length || lightId) &&
    (!backAtmospheres.length || backAtmoId) &&
    (!backLights.length || backLightId) &&
    renderWebGPU(canvas, atmoId, lightId, t, width, height, gpuScene)
  ) {
    showBackend?.("WebGPU");
    return;
  }
  clearWebGPU(canvas);
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    showBackend?.("Unavailable");
    return;
  }
  const scale = Math.min(1, 960 / Math.max(width, height));
  const w = Math.max(1, Math.round(width * scale)), h = Math.max(1, Math.round(height * scale));
  sizeCanvas(canvas, w, h);
  ctx.clearRect(0, 0, w, h);

  const shotSeed = scene?.shotIndex ?? 0;
  const cam = scene?.cam || null;
  if (fading) {
    let buffer = dissolveCanvases.get(canvas);
    if (!buffer) {
      buffer = canvas.ownerDocument.createElement("canvas");
      dissolveCanvases.set(canvas, buffer);
    }
    sizeCanvas(buffer, w, h);
    const layer = buffer.getContext("2d");
    const fadeMix = clamp01(scene.fadeMix ?? 1);
    const backT = Math.max(0, scene.backElapsed ?? 0);
    const backCueT = Math.max(0, scene.backCueElapsed ?? backT);
    // Add the weighted premultiplied layers so identical effects retain their opacity.
    withComposite(ctx, "lighter", () => {
      blendLayer2D(ctx, layer, backAtmospheres, backLights, backT, backCueT, 1 - fadeMix,
        scene.backShotIndex ?? 0, scene.backCam);
      blendLayer2D(ctx, layer, atmospheres, lights, t, cueT, fadeMix, shotSeed, cam);
    });
  } else {
    paintLayer2D(ctx, atmospheres, lights, t, cueT, w, h, shotSeed, cam);
  }
  showBackend?.("Canvas 2D");
}

const ATMO_PAINTERS = {
  fog: paintCloud,
  smoke: paintCloud,
  sunbeams: (_effect, ctx, t, w, h, sx, sy) => paintSunbeams(ctx, t, w, h, sx, sy),
  petals: (_effect, ctx, t, w, h, sx, sy) => paintPetals(ctx, t, w, h, sx, sy),
  underwater: (_effect, ctx, t, w, h, sx, sy) => paintUnderwater(ctx, t, w, h, sx, sy),
  fireflies: (_effect, ctx, t, w, h, sx, sy) => paintFireflies(ctx, t, w, h, sx, sy),
};

function blendLayer2D(ctx, layer, atmospheres, lights, t, cueT, weight, shotSeed, cam) {
  if (!weight || (!atmospheres.length && !lights.length)) return;
  const {width, height} = layer.canvas;
  layer.clearRect(0, 0, width, height);
  paintLayer2D(layer, atmospheres, lights, t, cueT, width, height, shotSeed, cam);
  ctx.globalAlpha = weight;
  ctx.drawImage(layer.canvas, 0, 0);
}

function paintLayer2D(ctx, atmospheres, lights, t, cueT, w, h, shotSeed, cam) {
  const shiftX = cam ? (cam.tx || 0) * 0.01 * w : 0;
  const shiftY = cam ? (cam.ty || 0) * 0.01 * h : 0;
  for (const effect of atmospheres) {
    const painter = ATMO_PAINTERS[effect] || paintParticles;
    painter(effect, ctx, t, w, h, shiftX, shiftY);
  }
  for (const effect of lights) {
    paintLight(ctx, effect, t, cueT, w, h, shotSeed, shiftX, shiftY);
  }
}

function paintVignette(ctx, _t, _cueT, w, h) {
  const gradient = ctx.createRadialGradient(w / 2, h / 2, Math.min(w, h) * 0.22,
    w / 2, h / 2, Math.max(w, h) * 0.68);
  gradient.addColorStop(0, "rgba(0,0,0,0)");
  gradient.addColorStop(0.55, "rgba(0,0,0,0.14)");
  gradient.addColorStop(1, "rgba(0,0,0,0.54)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, w, h);
}

function paintSpotlight(ctx, t, _cueT, w, h, _shotSeed, shiftX, shiftY) {
  const cx = w * (0.5 + Math.sin(t * 0.28) * 0.035) + shiftX;
  const cy = h * 0.48 + shiftY;
  if (ctx.createRadialGradient) {
    const shade = ctx.createRadialGradient(cx, cy, Math.min(w, h) * 0.12, cx, cy, Math.max(w, h) * 0.62);
    shade.addColorStop(0, "rgba(6,8,14,0)");
    shade.addColorStop(0.58, "rgba(6,8,14,0.22)");
    shade.addColorStop(1, "rgba(6,8,14,0.56)");
    ctx.fillStyle = shade;
    ctx.fillRect(0, 0, w, h);
  }
  withComposite(ctx, "screen", () => {
    if (ctx.createRadialGradient) {
      const beam = ctx.createRadialGradient(cx, h * 0.36 + shiftY, 0, cx, cy, Math.max(w, h) * 0.48);
      beam.addColorStop(0, "rgba(255,240,204,0.15)");
      beam.addColorStop(0.55, "rgba(255,232,184,0.06)");
      beam.addColorStop(1, "rgba(255,225,170,0)");
      ctx.fillStyle = beam;
    } else {
      ctx.fillStyle = "rgba(255,240,204,0.12)";
    }
    ctx.fillRect(0, 0, w, h);
  });
}

function paintWarmGlow(ctx, t, _cueT, w, h, _shotSeed, shiftX, shiftY) {
  const pulse = 0.085 + Math.sin(t * 0.8) * 0.03;
  withComposite(ctx, "screen", () => {
    if (ctx.createRadialGradient) {
      const cx = w * (0.5 + Math.sin(t * 0.22) * 0.06) + shiftX;
      const cy = h * (0.42 + Math.cos(t * 0.18) * 0.04) + shiftY;
      const gradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(w, h) * 0.72);
      gradient.addColorStop(0, `rgba(255,196,112,${round3(pulse * 1.6)})`);
      gradient.addColorStop(0.55, `rgba(255,162,72,${round3(pulse * 0.75)})`);
      gradient.addColorStop(1, "rgba(255,140,50,0)");
      ctx.fillStyle = gradient;
    } else {
      ctx.fillStyle = `rgba(255,177,85,${pulse})`;
    }
    ctx.fillRect(0, 0, w, h);
  });
}

function paintColdTint(ctx, t, _cueT, w, h) {
  const pulse = 0.09 + Math.sin(t * 0.5) * 0.02;
  if (ctx.createRadialGradient) {
    const gradient = ctx.createRadialGradient(w * 0.5, h * 0.35, Math.min(w, h) * 0.1,
      w * 0.5, h * 0.5, Math.max(w, h) * 0.75);
    gradient.addColorStop(0, `rgba(110,165,235,${round3(pulse * 0.45)})`);
    gradient.addColorStop(1, `rgba(48,96,175,${round3(pulse * 1.45)})`);
    ctx.fillStyle = gradient;
  } else {
    ctx.fillStyle = `rgba(80,130,210,${pulse})`;
  }
  ctx.fillRect(0, 0, w, h);
}

function paintFlicker(ctx, t, _cueT, w, h, _shotSeed, shiftX, shiftY) {
  const wave = 0.5 + 0.3 * Math.sin(t * 7.3) + 0.2 * Math.sin(t * 13.1);
  const alpha = round3(0.035 + wave * 0.075);
  withComposite(ctx, "screen", () => {
    if (ctx.createRadialGradient) {
      const cx = w * (0.5 + Math.sin(t * 5.1) * 0.02) + shiftX;
      const cy = h * 0.62 + shiftY;
      const gradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(w, h) * 0.65);
      gradient.addColorStop(0, `rgba(255,168,76,${round3(alpha * 1.7)})`);
      gradient.addColorStop(1, "rgba(255,120,40,0)");
      ctx.fillStyle = gradient;
    } else {
      ctx.fillStyle = `rgba(255,160,72,${alpha})`;
    }
    ctx.fillRect(0, 0, w, h);
  });
}

function paintFilmGrain(ctx, t, _cueT, w, h) {
  const tile = getSprites()?.grain;
  const frame = Math.floor(t * 24);
  if (tile && ctx.createPattern && ctx.save && ctx.restore) {
    const ox = Math.floor(fraction(Math.sin(frame * 12.9898) * 43758.5453) * 128);
    const oy = Math.floor(fraction(Math.cos(frame * 78.233) * 19341.123) * 128);
    const pattern = ctx.createPattern(tile, "repeat");
    if (!pattern) return;
    ctx.save();
    ctx.translate?.(-ox, -oy);
    ctx.fillStyle = pattern;
    ctx.fillRect(ox, oy, w + ox, h + oy);
    ctx.restore();
    return;
  }
  ctx.fillStyle = `rgba(128,128,128,${round3(0.03 + (frame % 3) * 0.01)})`;
  ctx.fillRect(0, 0, w, h);
}

function paintDreamBloom(ctx, t, _cueT, w, h, _shotSeed, shiftX, shiftY) {
  const pulse = 0.14 + Math.sin(t * 0.55) * 0.03;
  withComposite(ctx, "screen", () => {
    if (ctx.createRadialGradient) {
      const cx = w * 0.5 + shiftX;
      const cy = h * 0.45 + shiftY;
      const gradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(w, h) * 0.72);
      gradient.addColorStop(0, `rgba(250,225,245,${round3(pulse)})`);
      gradient.addColorStop(0.6, `rgba(245,215,240,${round3(pulse * 0.55)})`);
      gradient.addColorStop(1, "rgba(240,210,235,0)");
      ctx.fillStyle = gradient;
    } else {
      ctx.fillStyle = `rgba(250,225,245,${round3(pulse * 0.6)})`;
    }
    ctx.fillRect(0, 0, w, h);
  });
}

function paintScanlines(ctx, t, _cueT, w, h) {
  ctx.fillStyle = "rgba(10,22,32,0.14)";
  for (let y = 0; y < h; y += 3) {
    ctx.fillRect(0, y, w, 1);
  }
  const barY = fraction(t * 0.08) * h;
  ctx.fillStyle = "rgba(90,215,242,0.06)";
  ctx.fillRect(0, barY, w, Math.max(4, h * 0.04));
}

function paintSurgeFlash(effect, ctx, t, cueT, w, h, shotSeed, shiftX, shiftY) {
  // Lightning pulses once every 7s, below three flashes per second, with a brief secondary echo.
  // Flash uses cueT so it can trigger on one-shot sound cues as well as at shot start.
  const seedOffset = shotSeed > 0 ? fraction(Math.sin(shotSeed * 91.7 + 13.3) * 43758.5453) * 4.5 : 0;
  const boltX = shotSeed > 0 ? 0.22 + fraction(Math.sin(shotSeed * 47.3) * 43758.5453) * 0.56 : 0.52;
  const phase = effect === "flash" ? cueT : (t + seedOffset) % 7;
  const primary = effect === "flash"
    ? Math.max(0, 1 - phase / 0.45) * 0.35
    : Math.max(0, 1 - Math.abs(phase - 0.12) / 0.12) * 0.32;
  const echo = effect === "lightning"
    ? Math.max(0, 1 - Math.abs(phase - 0.30) / 0.08) * 0.14
    : 0;
  const alpha = Math.max(primary, echo);
  withComposite(ctx, "screen", () => {
    if (effect === "lightning" && alpha > 0 && ctx.createRadialGradient) {
      const bx = w * boltX + shiftX;
      const by = h * 0.15 + shiftY;
      const bloom = ctx.createRadialGradient(bx, by, 0, bx, h * 0.2 + shiftY, Math.max(w, h) * 0.75);
      bloom.addColorStop(0, `rgba(235,245,255,${round3(alpha * 0.75)})`);
      bloom.addColorStop(1, "rgba(180,210,255,0)");
      ctx.fillStyle = bloom;
      ctx.fillRect(0, 0, w, h);
    }
    ctx.fillStyle = `rgba(225,237,255,${alpha})`;
    ctx.fillRect(0, 0, w, h);
  });
}

const LIGHT_PAINTERS = {
  vignette: paintVignette,
  spotlight: paintSpotlight,
  warm_glow: paintWarmGlow,
  cold_tint: paintColdTint,
  flicker: paintFlicker,
  film_grain: paintFilmGrain,
  dream_bloom: paintDreamBloom,
  scanlines: paintScanlines,
  lightning: (ctx, t, cueT, w, h, shotSeed, shiftX, shiftY) => paintSurgeFlash("lightning", ctx, t, cueT, w, h, shotSeed, shiftX, shiftY),
  flash: (ctx, t, cueT, w, h, shotSeed, shiftX, shiftY) => paintSurgeFlash("flash", ctx, t, cueT, w, h, shotSeed, shiftX, shiftY),
};

function paintLight(ctx, effect, t, cueT, w, h, shotSeed = 0, shiftX = 0, shiftY = 0) {
  LIGHT_PAINTERS[effect]?.(ctx, t, cueT, w, h, shotSeed, shiftX, shiftY);
}

function paintCloud(effect, ctx, t, w, h, shiftX = 0, shiftY = 0) {
  const smoke = effect === "smoke";
  const sprite = smoke ? getSprites()?.smoke : getSprites()?.fog;
  for (let i = 0; i < 6; i++) {
    const dir = i % 2 === 0 ? 1 : -1;
    const speed = 0.012 + (i % 3) * 0.006;
    const x = (smoke
      ? (fraction(i * 0.193 + Math.sin(t * 0.25 + i) * 0.05 + dir * t * 0.008) * 1.3 - 0.15) * w
      : (fraction(i * 0.213 + dir * t * speed) * 1.9 - 0.45) * w) + shiftX;
    const y = (smoke
      ? (1.05 - fraction(i * 0.17 + t * (0.045 + (i % 3) * 0.015)) * 1.15) * h
      : (0.38 + i * 0.11 + Math.sin(t * 0.18 + i) * 0.025) * h) + shiftY;
    const radius = w * (smoke ? 0.32 + (i % 3) * 0.08 : 0.42 + (i % 3) * 0.1);
    if (sprite && ctx.drawImage && ctx.save && ctx.restore) {
      ctx.save();
      ctx.globalAlpha = (ctx.globalAlpha ?? 1) * (smoke ? 0.55 : 0.65);
      ctx.drawImage(sprite, x - radius, y - radius, radius * 2, radius * 2);
      ctx.restore();
    } else {
      const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
      const base = smoke ? "128,136,146" : "218,230,235";
      gradient.addColorStop(0, `rgba(${base},${0.12 + (i % 2) * 0.04})`);
      gradient.addColorStop(0.6, `rgba(${base},0.05)`);
      gradient.addColorStop(1, `rgba(${base},0)`);
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, w, h);
    }
  }
}

function paintSunbeams(ctx, t, w, h, shiftX = 0, shiftY = 0) {
  withComposite(ctx, "screen", () => {
    const originX = w * 0.22 + shiftX;
    for (let i = 0; i < 5; i++) {
      const sway = Math.sin(t * 0.18 + i * 1.35) * 0.045;
      const topX = originX + (i * 0.13 + sway) * w;
      const bottomSpread = (0.18 + i * 0.22 + sway * 1.6) * w;
      const beamW = w * (0.12 + (i % 2) * 0.055);
      const alpha = round3(0.058 + 0.028 * Math.sin(t * 0.42 + i * 1.1));
      const g = ctx.createLinearGradient?.(topX, 0, originX + bottomSpread, h);
      if (g) {
        g.addColorStop(0, `rgba(255,238,192,${alpha})`);
        g.addColorStop(0.68, `rgba(255,230,175,${round3(alpha * 0.32)})`);
        g.addColorStop(1, "rgba(255,225,165,0)");
        ctx.fillStyle = g;
      } else {
        ctx.fillStyle = `rgba(255,236,188,${alpha})`;
      }
      ctx.beginPath();
      ctx.moveTo(topX - beamW * 0.32, 0);
      ctx.lineTo(topX + beamW * 0.32, 0);
      ctx.lineTo(originX + bottomSpread + beamW * 0.95, h);
      ctx.lineTo(originX + bottomSpread - beamW * 0.55, h);
      ctx.closePath();
      ctx.fill();
    }
  });
}

function paintPetals(ctx, t, w, h, shiftX = 0, shiftY = 0) {
  const scale = Math.max(0.5, w / 960);
  const count = 36;
  for (let i = 0; i < count; i++) {
    const seed = SEEDS[i];
    const depth = DEPTHS[i];
    const y = fraction(seed + t * 0.11 * depth) * (h + 36) - 18 + shiftY * depth;
    const x = fraction(PHASES[i] + t * 0.06 * depth + Math.sin(t * 1.1 + i) * 0.035) * w + shiftX * depth;
    const r = (4.2 + seed * 3.0) * depth * scale;
    const angle = t * (1.2 + seed * 1.4) + i;
    const tumble = 0.28 + 0.72 * Math.abs(Math.cos(t * (1.5 + seed * 1.3) + i * 0.9));
    ctx.beginPath();
    ctx.fillStyle = i % 3 === 0
      ? `rgba(245,154,108,${round3(depth * 0.72)})`
      : `rgba(250,188,204,${round3(depth * 0.75)})`;
    if (ctx.ellipse) {
      ctx.ellipse(x, y, r, r * 0.62 * tumble, angle, 0, Math.PI * 2);
    } else {
      ctx.arc(x, y, r * 0.7, 0, Math.PI * 2);
    }
    ctx.fill();
  }
}

function paintUnderwater(ctx, t, w, h, shiftX = 0, shiftY = 0) {
  const scale = Math.max(0.5, w / 960);
  const causticSprite = getSprites()?.caustic;
  const bubbleSprite = getSprites()?.bubble;
  withComposite(ctx, "screen", () => {
    const baseAlpha = ctx.globalAlpha ?? 1;
    for (let i = 0; i < 4; i++) {
      const cx = (0.18 + i * 0.22 + Math.sin(t * 0.45 + i * 1.3) * 0.06) * w + shiftX;
      const cy = (0.18 + Math.cos(t * 0.35 + i) * 0.05) * h + shiftY;
      const pulse = 0.23 + 0.11 * Math.sin(t * 0.8 + i);
      if (causticSprite && ctx.drawImage && ctx.save && ctx.restore) {
        const radius = w * 0.34;
        ctx.save();
        ctx.globalAlpha = baseAlpha * pulse;
        ctx.drawImage(causticSprite, cx - radius, cy - radius, radius * 2, radius * 2);
        ctx.restore();
      } else {
        const alpha = round3(0.05 + 0.025 * Math.sin(t * 0.8 + i));
        if (ctx.createRadialGradient) {
          const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, w * 0.34);
          g.addColorStop(0, `rgba(105,228,255,${alpha})`);
          g.addColorStop(1, "rgba(65,180,225,0)");
          ctx.fillStyle = g;
        } else {
          ctx.fillStyle = `rgba(105,228,255,${alpha})`;
        }
        ctx.fillRect(0, 0, w, h);
      }
    }
    const count = 44;
    for (let i = 0; i < count; i++) {
      const seed = SEEDS[i], depth = DEPTHS[i];
      const y = fraction(seed - t * 0.14 * depth) * (h + 32) - 16 + shiftY * depth;
      const wobble = Math.sin(t * (1.2 + seed) + i * 1.5) * 0.028;
      const x = fraction(PHASES[i] + wobble) * w + shiftX * depth;
      const radius = (1.6 + seed * 2.2) * depth * scale;
      if (bubbleSprite && ctx.drawImage && ctx.save && ctx.restore) {
        const size = radius * 2.6;
        ctx.save();
        ctx.globalAlpha = baseAlpha * clamp01(depth * 0.85);
        ctx.drawImage(bubbleSprite, x - size / 2, y - size / 2, size, size);
        ctx.restore();
      } else {
        ctx.beginPath();
        ctx.fillStyle = `rgba(205,246,255,${round3(depth * 0.62)})`;
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  });
}

function paintFireflies(ctx, t, w, h, shiftX = 0, shiftY = 0) {
  const scale = Math.max(0.5, w / 960);
  const count = 32;
  const sprite = getSprites()?.firefly;
  withComposite(ctx, "screen", () => {
    const baseAlpha = ctx.globalAlpha ?? 1;
    for (let i = 0; i < count; i++) {
      const seed = SEEDS[i], depth = DEPTHS[i];
      const phase = PHASES[i] * Math.PI * 2;
      const x = fraction(PHASES[i] + Math.sin(t * (0.35 + seed * 0.3) + phase) * 0.08) * w + shiftX * depth;
      const y = fraction(seed + Math.cos(t * (0.28 + seed * 0.25) + i) * 0.07 - t * 0.015 * depth) * h + shiftY * depth;
      const pulse = 0.2 + 0.8 * Math.pow(0.5 + 0.5 * Math.sin(t * (1.7 + seed * 2.0) + phase), 2);
      const radius = (2.0 + seed * 1.8) * depth * scale;
      const alpha = round3(depth * pulse * 0.88);
      if (sprite && ctx.drawImage && ctx.save && ctx.restore) {
        const size = radius * 7.2;
        ctx.save();
        ctx.globalAlpha = baseAlpha * clamp01(alpha);
        ctx.drawImage(sprite, x - size / 2, y - size / 2, size, size);
        ctx.restore();
      } else if (ctx.createRadialGradient) {
        const g = ctx.createRadialGradient(x, y, 0, x, y, radius * 3.6);
        g.addColorStop(0, `rgba(255,248,165,${alpha})`);
        g.addColorStop(0.45, `rgba(182,250,105,${round3(alpha * 0.55)})`);
        g.addColorStop(1, "rgba(150,240,80,0)");
        ctx.fillStyle = g;
        ctx.fillRect(x - radius * 3.6, y - radius * 3.6, radius * 7.2, radius * 7.2);
      } else {
        ctx.beginPath();
        ctx.fillStyle = `rgba(215,250,125,${alpha})`;
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  });
}

const RAIN_LAYERS = [
  { minDepth: 0, maxDepth: 0.58, width: 0.55, minWidth: 0.7, style: "rgba(195,220,240,0.18)", speed: 0.76, slant: 4.5 },
  { minDepth: 0.58, maxDepth: 0.8, width: 0.9, minWidth: 0.9, style: "rgba(195,220,240,0.28)", speed: 0.94, slant: 5.5 },
  { minDepth: 0.8, maxDepth: 2, width: 1.25, minWidth: 1.15, style: "rgba(205,228,248,0.38)", speed: 1.16, slant: 6.5 },
].map((layer) => ({
  ...layer,
  indices: Array.from({length: 115}, (_, index) => index)
    .filter((index) => DEPTHS[index] >= layer.minDepth && DEPTHS[index] < layer.maxDepth),
}));

function paintRain(ctx, t, w, h, shiftX, shiftY, scale) {
  for (const layer of RAIN_LAYERS) {
    ctx.beginPath();
    ctx.strokeStyle = layer.style;
    ctx.lineWidth = Math.max(layer.minWidth, layer.width * scale);
    for (const i of layer.indices) {
      const depth = DEPTHS[i];
      const seed = SEEDS[i];
      const y = fraction(seed + t * layer.speed * depth) * (h + 40) - 20 + shiftY * depth;
      const x = fraction(PHASES[i] - t * 0.13 * depth) * w + shiftX * depth;
      ctx.moveTo(x, y);
      ctx.lineTo(x - layer.slant * depth * scale, y + (15 + seed * 12) * depth * scale);
    }
    ctx.stroke();
  }
  if (ctx.createLinearGradient) {
    const mist = ctx.createLinearGradient(0, h * 0.72, 0, h);
    if (mist) {
      mist.addColorStop(0, "rgba(185,205,220,0)");
      mist.addColorStop(1, "rgba(185,205,220,0.12)");
      ctx.fillStyle = mist;
      ctx.fillRect(0, h * 0.72, w, h * 0.28);
    }
  }
}

function paintParticles(effect, ctx, t, w, h, shiftX = 0, shiftY = 0) {
  const scale = Math.max(0.5, w / 960);
  if (effect === "rain") {
    paintRain(ctx, t, w, h, shiftX, shiftY, scale);
    return;
  }
  const ember = effect === "embers", dust = effect === "dust";
  const count = dust ? 38 : ember ? 56 : 64;
  const sprite = ember ? getSprites()?.ember : !dust ? getSprites()?.snow : null;
  const draw = () => {
    const baseAlpha = ctx.globalAlpha ?? 1;
    for (let i = 0; i < count; i++) {
      const seed = SEEDS[i], depth = DEPTHS[i];
      const speed = ember ? -0.17 : dust ? -0.022 : 0.085;
      const y = fraction(seed + t * speed * depth) * (h + 36) - 18 + shiftY * depth;
      const drift = Math.sin(t * (ember ? 0.9 : 0.35) + i * 1.7) * (dust ? 0.025 : 0.04);
      const x = fraction(PHASES[i] + drift) * w + shiftX * depth;
      const radius = (dust ? 1.5 : ember ? 2.1 : 2.6) * depth * scale;
      if (sprite && ctx.drawImage && ctx.save && ctx.restore) {
        const size = radius * (ember ? 5.2 : 4.4);
        const twinkle = ember ? (0.68 + 0.32 * Math.sin(t * 9.5 + i * 2.3)) : 1;
        ctx.save();
        ctx.globalAlpha = baseAlpha * clamp01(depth * (ember ? 0.95 * twinkle : 0.85));
        ctx.drawImage(sprite, x - size / 2, y - size / 2, size, size);
        ctx.restore();
      } else {
        ctx.beginPath();
        ctx.fillStyle = ember ? `rgba(255,${Math.round(135 + seed * 95)},75,${round3(depth * 0.85)})`
          : dust ? `rgba(255,235,185,${round3(depth * 0.42)})`
          : `rgba(245,250,255,${round3(depth * 0.78)})`;
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  };
  if (ember || dust) withComposite(ctx, "screen", draw);
  else draw();
}
