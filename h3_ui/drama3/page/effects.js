// Animation derives from media time: pausing freezes it; seeking reconstructs it.
const fraction = (value) => value - Math.floor(value);
const cameraMoves = {
  push_in: (p) => `scale(${1 + p * 0.08})`,
  pull_out: (p) => `scale(${1.08 - p * 0.08})`,
  pan_left: (p) => `translateX(${3 - p * 6}%) scale(1.08)`,
  pan_right: (p) => `translateX(${-3 + p * 6}%) scale(1.08)`,
};
const lightEffects = new Set(["lightning", "flash", "warm_glow", "vignette"]);
const particleEffects = new Set(["rain", "snow", "dust", "embers"]);

export function effectsEnabled(video, reducedMotion) {
  return video.effects_enabled !== false && (!reducedMotion || video.effects_override_reduced_motion === true);
}

export function cameraTransform(effects, progress) {
  const p = Math.max(0, Math.min(1, progress));
  const move = effects.find((id) => Object.hasOwn(cameraMoves, id));
  return move ? cameraMoves[move](p) : "none";
}

export function paintEffects(canvas, effects, seconds) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  // Bound the drawing surface to keep overlays inexpensive on large screens.
  const width = Math.max(1, Math.round(canvas.clientWidth));
  const height = Math.max(1, Math.round(canvas.clientHeight));
  const scale = Math.min(1, 960 / width);
  const w = Math.max(1, Math.round(width * scale)), h = Math.max(1, Math.round(height * scale));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  ctx.clearRect(0, 0, w, h);
  const effect = effects.find((id) => lightEffects.has(id) || particleEffects.has(id) || id === "fog");
  if (!effect) return;
  const t = Math.max(0, seconds);
  if (lightEffects.has(effect)) paintLight(ctx, effect, t, w, h);
  else if (effect === "fog") paintFog(ctx, t, w, h);
  else paintParticles(ctx, effect, t, w, h);
}

function paintLight(ctx, effect, t, w, h) {
  if (effect === "vignette") {
    const gradient = ctx.createRadialGradient(w / 2, h / 2, Math.min(w, h) * 0.2,
      w / 2, h / 2, Math.max(w, h) * 0.65);
    gradient.addColorStop(0, "rgba(0,0,0,0)");
    gradient.addColorStop(1, "rgba(0,0,0,0.5)");
    ctx.fillStyle = gradient;
  } else if (effect === "warm_glow") {
    ctx.fillStyle = `rgba(255,177,85,${0.055 + Math.sin(t * 0.8) * 0.025})`;
  } else {
    // Lightning pulses once every 7s, below three flashes per second.
    const phase = effect === "flash" ? t : t % 7;
    const alpha = effect === "flash" ? Math.max(0, 1 - phase / 0.45) * 0.35
      : Math.max(0, 1 - Math.abs(phase - 0.12) / 0.12) * 0.32;
    ctx.fillStyle = `rgba(225,237,255,${alpha})`;
  }
  ctx.fillRect(0, 0, w, h);
}

function paintFog(ctx, t, w, h) {
  for (let i = 0; i < 5; i++) {
    const x = (fraction(i * 0.237 + t * 0.018) * 1.8 - 0.4) * w;
    const y = (0.3 + i * 0.15) * h;
    const gradient = ctx.createRadialGradient(x, y, 0, x, y, w * 0.55);
    gradient.addColorStop(0, "rgba(218,230,235,0.15)");
    gradient.addColorStop(1, "rgba(218,230,235,0)");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, w, h);
  }
}

function paintParticles(ctx, effect, t, w, h) {
  const rain = effect === "rain", ember = effect === "embers", dust = effect === "dust";
  const count = rain ? 100 : dust ? 32 : 60;
  for (let i = 0; i < count; i++) {
    // Fixed seeds preserve particle positions across redraws, pauses and seeks.
    const seed = fraction(Math.sin(i * 127.1 + 7) * 43758.5453);
    const depth = 0.4 + seed * 0.6;
    const speed = rain ? 0.85 : ember ? -0.16 : dust ? -0.025 : 0.09;
    const y = fraction(seed + t * speed * depth) * (h + 30) - 15;
    const x = fraction(i * 0.618034 + (rain ? -t * 0.12 : Math.sin(t * 0.3 + i) * 0.035)) * w;
    ctx.beginPath();
    if (rain) {
      ctx.strokeStyle = `rgba(190,215,235,${0.15 + seed * 0.25})`;
      ctx.lineWidth = depth;
      ctx.moveTo(x, y); ctx.lineTo(x - 5 * depth, y + 20 * depth); ctx.stroke();
    } else {
      ctx.fillStyle = ember ? `rgba(255,${Math.round(110 + seed * 100)},65,${depth * 0.8})`
        : dust ? `rgba(255,232,180,${depth * 0.45})` : `rgba(240,248,255,${depth * 0.8})`;
      ctx.arc(x, y, (dust ? 1.4 : 2.5) * depth * Math.max(0.5, w / 960), 0, Math.PI * 2);
      ctx.fill();
    }
  }
}
