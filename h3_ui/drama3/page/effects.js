import { ATMO_IDS, LIGHT_IDS, clearWebGPU, gpuStatus, probeWebGPU, renderWebGPU } from "./effects_webgpu.js";

// Animation derives from media time: pausing freezes it; seeking reconstructs it.
const fraction = (value) => value - Math.floor(value);
const clamp01 = (value) => Math.max(0, Math.min(1, value));
const easeInOut = (p) => 0.5 - 0.5 * Math.cos(Math.PI * clamp01(p));
const round3 = (value) => Math.round(value * 1000) / 1000;

const IDENTITY_CAMERA = Object.freeze({ tx: 0, ty: 0, scale: 1, shake: 0, angle: 0, blur: 0 });

function rackBlurDecay(p, t) {
  const time = t > 0 ? t : p * 4;
  if (time < 0 || time >= 0.95) return 0;
  return round3(Math.pow(1 - time / 0.95, 2));
}

// Single source of truth for camera kinematics; drives both CSS transforms and WebGPU/Canvas2D camera coupling.
const CAMERA_MOVES = {
  push_in: (p) => {
    const scale = 1 + p * 0.08;
    return { css: `scale(${scale})`, tx: 0, ty: 0, scale, shake: 0, angle: 0, blur: 0 };
  },
  pull_out: (p) => {
    const scale = 1.08 - p * 0.08;
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
    const time = t > 0 ? t : p * 4;
    const tx = round3(Math.sin(time * 0.9) * 0.85);
    const ty = round3(Math.cos(time * 0.65) * 0.55);
    return { css: `translate(${tx}%, ${ty}%) scale(1.04)`, tx, ty, scale: 1.04, shake: 0, angle: 0, blur: 0 };
  },
  impact_shake: (p, t) => {
    const time = t > 0 ? t : p * 4;
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

function evaluateCamera(effects, progress, seconds = 0) {
  const p = clamp01(progress);
  const move = effects?.find?.((id) => Object.hasOwn(CAMERA_MOVES, id));
  return move ? CAMERA_MOVES[move](p, seconds) : null;
}

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
  const grain = makeSprite(128, (ctx, s) => {
    const img = ctx.createImageData(s, s);
    for (let i = 0; i < img.data.length; i += 4) {
      const idx = i >> 2;
      const v = Math.floor(fraction(Math.sin(idx * 12.9898 + 78.233) * 43758.5453) * 255);
      img.data[i] = v; img.data[i + 1] = v; img.data[i + 2] = v; img.data[i + 3] = 28;
    }
    ctx.putImageData(img, 0, 0);
  });
  sprites = fog && snow && ember ? { fog, smoke, snow, ember, grain } : false;
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

/**
 * Deterministic timeline sound-cue onset time: when a shot contains one-shot sound cues
 * (`!sound.loop`) inside `(shotStart, shotEnd)`, transient effects (`impact_shake`, `flash`)
 * trigger at shot start and re-trigger at each one-shot sound cue's timestamp.
 */
export function cueTriggerElapsed(elapsed, shotStart = 0, shotEnd = 0, sounds = null) {
  if (elapsed < 0 || !Array.isArray(sounds) || !sounds.length || !(shotEnd > shotStart)) {
    return elapsed;
  }
  const time = shotStart + elapsed;
  let best = elapsed;
  for (const sound of sounds) {
    if (!sound || sound.loop) continue;
    const start = Number(sound.start);
    if (Number.isFinite(start) && start > shotStart && start < shotEnd && start <= time) {
      best = Math.min(best, time - start);
    }
  }
  return best;
}

export function cameraTransform(effects, progress, seconds = 0) {
  return evaluateCamera(effects, progress, seconds)?.css ?? "none";
}

export function cameraFilter(effects, progress, seconds = 0) {
  if (!effects?.includes?.("rack_focus")) return "none";
  const decay = rackBlurDecay(clamp01(progress), seconds);
  return decay > 0 ? `blur(${round3(decay * 4.5)}px)` : "none";
}

export function cameraMotion(effects, progress, seconds = 0) {
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

function withLayerAlpha(ctx, alpha, draw) {
  if (alpha >= 0.999 || !ctx.save || !ctx.restore) return draw();
  ctx.save();
  ctx.globalAlpha = (ctx.globalAlpha ?? 1) * clamp01(alpha);
  try { draw(); } finally { ctx.restore(); }
}

function sizeCanvas(canvas, w, h) {
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
}

let lastPaintArgs = null;
export function paintEffects(canvas, effects, seconds, showBackend, scene = null) {
  const atmospheres = effects.filter((id) => atmosphereEffects.has(id));
  const lights = effects.filter((id) => lightEffects.has(id));
  const fading = Boolean(scene?.fading && (scene?.fadeMix ?? 1) < 0.999);
  const backEffects = fading && Array.isArray(scene?.backEffects) ? scene.backEffects : [];
  const backAtmospheres = backEffects.filter((id) => atmosphereEffects.has(id));
  const backLights = backEffects.filter((id) => lightEffects.has(id));

  const width = Math.max(1, Math.round(canvas.clientWidth));
  const height = Math.max(1, Math.round(canvas.clientHeight));
  const scale = Math.min(1, 960 / width);
  const w = Math.max(1, Math.round(width * scale)), h = Math.max(1, Math.round(height * scale));
  if (!atmospheres.length && !lights.length && !backAtmospheres.length && !backLights.length) {
    lastPaintArgs = null;
    clearWebGPU(canvas);
    const ctx = canvas.getContext("2d");
    if (ctx) {
      sizeCanvas(canvas, w, h);
      ctx.clearRect(0, 0, w, h);
    }
    showBackend?.(null);
    return;
  }
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
    renderWebGPU(canvas, atmoId, lightId, t, w, h, gpuScene)
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
  sizeCanvas(canvas, w, h);
  ctx.clearRect(0, 0, w, h);

  const shotSeed = scene?.shotIndex ?? 0;
  const cam = scene?.cam || null;
  if (fading && (backAtmospheres.length || backLights.length)) {
    const fadeMix = clamp01(scene.fadeMix ?? 1);
    const backT = Math.max(0, scene.backElapsed ?? 0);
    const backCueT = Math.max(0, scene.backCueElapsed ?? backT);
    const backSeed = scene.backShotIndex ?? 0;
    paintLayer2D(ctx, backAtmospheres, backLights, backT, backCueT, w, h, 1 - fadeMix, backSeed, scene.backCam || null);
    if (atmospheres.length || lights.length) {
      paintLayer2D(ctx, atmospheres, lights, t, cueT, w, h, fadeMix, shotSeed, cam);
    }
  } else {
    paintLayer2D(ctx, atmospheres, lights, t, cueT, w, h, 1, shotSeed, cam);
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

function paintLayer2D(ctx, atmospheres, lights, t, cueT, w, h, alpha, shotSeed, cam) {
  withLayerAlpha(ctx, alpha, () => {
    const shiftX = cam ? (cam.tx || 0) * 0.01 * w : 0;
    const shiftY = cam ? (cam.ty || 0) * 0.01 * h : 0;
    for (const effect of atmospheres) {
      const painter = ATMO_PAINTERS[effect] || paintParticles;
      painter(effect, ctx, t, w, h, shiftX, shiftY);
    }
    for (const effect of lights) {
      paintLight(ctx, effect, t, cueT, w, h, shotSeed, shiftX, shiftY);
    }
  });
}

function paintLight(ctx, effect, t, cueT, w, h, shotSeed = 0, shiftX = 0, shiftY = 0) {
  if (effect === "vignette") {
    const gradient = ctx.createRadialGradient(w / 2, h / 2, Math.min(w, h) * 0.22,
      w / 2, h / 2, Math.max(w, h) * 0.68);
    gradient.addColorStop(0, "rgba(0,0,0,0)");
    gradient.addColorStop(0.55, "rgba(0,0,0,0.14)");
    gradient.addColorStop(1, "rgba(0,0,0,0.54)");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, w, h);
    return;
  }
  if (effect === "spotlight") {
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
    return;
  }
  if (effect === "warm_glow") {
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
    return;
  }
  if (effect === "cold_tint") {
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
    return;
  }
  if (effect === "flicker") {
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
    return;
  }
  if (effect === "film_grain") {
    const tile = getSprites()?.grain;
    const frame = Math.floor(t * 24);
    const ox = Math.floor(fraction(Math.sin(frame * 12.9898) * 43758.5453) * 128);
    const oy = Math.floor(fraction(Math.cos(frame * 78.233) * 19341.123) * 128);
    if (tile && ctx.createPattern) {
      withComposite(ctx, "overlay", () => {
        const pattern = ctx.createPattern(tile, "repeat");
        if (!pattern) return;
        ctx.translate?.(-ox, -oy);
        ctx.fillStyle = pattern;
        ctx.fillRect(ox, oy, w + ox, h + oy);
      });
    } else {
      ctx.fillStyle = `rgba(128,128,128,${round3(0.03 + (frame % 3) * 0.01)})`;
      ctx.fillRect(0, 0, w, h);
    }
    return;
  }
  if (effect === "dream_bloom") {
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
    return;
  }
  if (effect === "scanlines") {
    ctx.fillStyle = "rgba(10,22,32,0.14)";
    for (let y = 0; y < h; y += 3) {
      ctx.fillRect(0, y, w, 1);
    }
    const barY = fraction(t * 0.08) * h;
    ctx.fillStyle = "rgba(90,215,242,0.06)";
    ctx.fillRect(0, barY, w, Math.max(4, h * 0.04));
    return;
  }
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
  withComposite(ctx, "screen", () => {
    for (let i = 0; i < 4; i++) {
      const cx = (0.18 + i * 0.22 + Math.sin(t * 0.45 + i * 1.3) * 0.06) * w + shiftX;
      const cy = (0.18 + Math.cos(t * 0.35 + i) * 0.05) * h + shiftY;
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
    const count = 44;
    for (let i = 0; i < count; i++) {
      const seed = SEEDS[i], depth = DEPTHS[i];
      const y = fraction(seed - t * 0.14 * depth) * (h + 32) - 16 + shiftY * depth;
      const wobble = Math.sin(t * (1.2 + seed) + i * 1.5) * 0.028;
      const x = fraction(PHASES[i] + wobble) * w + shiftX * depth;
      const radius = (1.6 + seed * 2.2) * depth * scale;
      ctx.beginPath();
      ctx.fillStyle = `rgba(205,246,255,${round3(depth * 0.62)})`;
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
    }
  });
}

function paintFireflies(ctx, t, w, h, shiftX = 0, shiftY = 0) {
  const scale = Math.max(0.5, w / 960);
  const count = 32;
  withComposite(ctx, "screen", () => {
    for (let i = 0; i < count; i++) {
      const seed = SEEDS[i], depth = DEPTHS[i];
      const phase = PHASES[i] * Math.PI * 2;
      const x = fraction(PHASES[i] + Math.sin(t * (0.35 + seed * 0.3) + phase) * 0.08) * w + shiftX * depth;
      const y = fraction(seed + Math.cos(t * (0.28 + seed * 0.25) + i) * 0.07 - t * 0.015 * depth) * h + shiftY * depth;
      const pulse = 0.2 + 0.8 * Math.pow(0.5 + 0.5 * Math.sin(t * (1.7 + seed * 2.0) + phase), 2);
      const radius = (2.0 + seed * 1.8) * depth * scale;
      const alpha = round3(depth * pulse * 0.88);
      if (ctx.createRadialGradient) {
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

function paintParticles(effect, ctx, t, w, h, shiftX = 0, shiftY = 0) {
  const rain = effect === "rain", ember = effect === "embers", dust = effect === "dust";
  const count = rain ? 115 : dust ? 38 : ember ? 56 : 64;
  const scale = Math.max(0.5, w / 960);
  if (rain) {
    // Batch rain streaks into 3 depth layers (far fine sheet, mid rain, foreground fast streaks).
    const buckets = [
      { width: Math.max(0.7, 0.55 * scale), style: "rgba(195,220,240,0.18)", speed: 0.76, slant: 4.5, indices: [] },
      { width: Math.max(0.9, 0.9 * scale), style: "rgba(195,220,240,0.28)", speed: 0.94, slant: 5.5, indices: [] },
      { width: Math.max(1.15, 1.25 * scale), style: "rgba(205,228,248,0.38)", speed: 1.16, slant: 6.5, indices: [] },
    ];
    for (let i = 0; i < count; i++) {
      const depth = DEPTHS[i];
      const bucket = depth < 0.58 ? buckets[0] : depth < 0.8 ? buckets[1] : buckets[2];
      bucket.indices.push(i);
    }
    for (const bucket of buckets) {
      if (!bucket.indices.length) continue;
      ctx.beginPath();
      ctx.strokeStyle = bucket.style;
      ctx.lineWidth = bucket.width;
      for (const i of bucket.indices) {
        const seed = SEEDS[i], depth = DEPTHS[i];
        const y = fraction(seed + t * bucket.speed * depth) * (h + 40) - 20 + shiftY * depth;
        const x = fraction(PHASES[i] - t * 0.13 * depth) * w + shiftX * depth;
        ctx.moveTo(x, y);
        ctx.lineTo(x - bucket.slant * depth * scale, y + (15 + seed * 12) * depth * scale);
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
    return;
  }

  const sprite = ember ? getSprites()?.ember : !dust ? getSprites()?.snow : null;
  const draw = () => {
    for (let i = 0; i < count; i++) {
      const seed = SEEDS[i], depth = DEPTHS[i];
      const speed = ember ? -0.17 : dust ? -0.022 : 0.085;
      const y = fraction(seed + t * speed * depth) * (h + 36) - 18 + shiftY * depth;
      const drift = Math.sin(t * (ember ? 0.9 : 0.35) + i * 1.7) * (dust ? 0.025 : 0.04);
      const x = fraction(PHASES[i] + drift) * w + shiftX * depth;
      const radius = (dust ? 1.5 : ember ? 2.1 : 2.6) * depth * scale;
      if (sprite && ctx.drawImage && ctx.save && ctx.restore) {
        const size = radius * (ember ? 5.2 : 4.4);
        ctx.save();
        ctx.globalAlpha = (ctx.globalAlpha ?? 1) * clamp01(depth * (ember ? 0.95 : 0.85));
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
