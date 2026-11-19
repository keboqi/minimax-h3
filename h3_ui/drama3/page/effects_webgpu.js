// WebGPU WGSL unified texture compositor & procedural atmospheric/lighting shader backend for Sound Stage.
// Automatically probed when navigator.gpu is available; falls back to Canvas 2D otherwise.

export const ATMO_IDS = {
  rain: 1, snow: 2, fog: 3, dust: 4, embers: 5, sunbeams: 6, smoke: 7, petals: 8,
  underwater: 9, fireflies: 10,
};

export const LIGHT_IDS = {
  lightning: 1, flash: 2, warm_glow: 3, cold_tint: 4, flicker: 5,
  vignette: 6, film_grain: 7, dream_bloom: 8, scanlines: 9, spotlight: 10,
};

const WGSL = /* wgsl */ `
struct Uniforms {
  // x: width, y: height, z: time, w: backTime
  timeParams: vec4f,
  // x: fadeMix (0..1), y: shotSeed, z: backShotSeed, w: hasTexture (0 = overlay, 1 = single tex, 2 = dissolve pair)
  mixParams: vec4f,
  // x: tx (fraction), y: ty (fraction), z: scale, w: shake (0..1)
  cam: vec4f,
  // x: backTx, y: backTy, z: backScale, w: backShake
  backCam: vec4f,
  // x: backTexW, y: backTexH, z: frontTexW, w: frontTexH
  texSizes: vec4f,
  // x: camAngleRad, y: camBlur (0..1), z: backCamAngleRad, w: backCamBlur (0..1)
  camExtra: vec4f,
  // x: cueTime, y: backCueTime, z: reserved, w: reserved
  cueParams: vec4f,
  // x: atmo, y: light, z: backAtmo, w: backLight
  effectIds: vec4u,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var texSampler: sampler;
@group(0) @binding(2) var backTex: texture_2d<f32>;
@group(0) @binding(3) var frontTex: texture_2d<f32>;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) idx: u32) -> VSOut {
  var pts = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f( 3.0, -1.0),
    vec2f(-1.0,  3.0)
  );
  let p = pts[idx];
  var out: VSOut;
  out.pos = vec4f(p, 0.0, 1.0);
  // Flip Y so uv.y = 0 is top, 1 is bottom (matching Canvas 2D coordinates).
  out.uv = vec2f(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
  return out;
}

fn hash21(p: vec2f) -> f32 {
  var q = fract(p * vec2f(123.34, 456.21));
  q += dot(q, q + 45.32);
  return fract(q.x * q.y);
}

fn hash22(p: vec2f) -> vec2f {
  let n = vec2f(dot(p, vec2f(127.1, 311.7)), dot(p, vec2f(269.5, 183.3)));
  return fract(sin(n) * 43758.5453);
}

fn noise2(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let s = f * f * (3.0 - 2.0 * f);
  let a = hash21(i);
  let b = hash21(i + vec2f(1.0, 0.0));
  let c = hash21(i + vec2f(0.0, 1.0));
  let d = hash21(i + vec2f(1.0, 1.0));
  return mix(mix(a, b, s.x), mix(c, d, s.x), s.y);
}

fn fbm(p_in: vec2f) -> f32 {
  var p = p_in;
  var v = 0.0;
  var a = 0.5;
  let rot = mat2x2f(0.8, 0.6, -0.6, 0.8);
  for (var i = 0; i < 4; i++) {
    v += a * noise2(p);
    p = rot * p * 2.02 + vec2f(17.1, 31.7);
    a *= 0.5;
  }
  return v;
}

fn blendOver(dst: vec4f, srcRgb: vec3f, srcA: f32) -> vec4f {
  let a = clamp(srcA, 0.0, 1.0);
  let rgb = srcRgb * a + dst.rgb * (1.0 - a);
  let outA = a + dst.a * (1.0 - a);
  return vec4f(rgb, outA);
}

fn blendScreen(dst: vec4f, srcRgb: vec3f, srcA: f32) -> vec4f {
  let a = clamp(srcA, 0.0, 1.0);
  let rgb = dst.rgb + srcRgb * a * (1.0 - dst.rgb);
  // Keep the screen blend premultiplied for both the shader and the browser compositor.
  let outA = a + dst.a * (1.0 - a);
  return vec4f(rgb, outA);
}

fn invSmooth(low: f32, high: f32, x: f32) -> f32 {
  return 1.0 - smoothstep(low, high, x);
}

// Map viewport UV into camera-transformed world UV (including dutch_tilt rotation) so scene-anchored effects move with the shot.
fn cameraUv(uv: vec2f, cam: vec4f, angleRad: f32, aspect: f32) -> vec2f {
  let s = max(0.5, cam.z);
  var rel = (uv - vec2f(0.5)) - cam.xy;
  if (abs(angleRad) > 0.0001) {
    let cs = cos(-angleRad);
    let sn = sin(-angleRad);
    let metric = vec2f(rel.x * aspect, rel.y);
    let rot = vec2f(metric.x * cs - metric.y * sn, metric.x * sn + metric.y * cs);
    rel = vec2f(rot.x / max(0.1, aspect), rot.y);
  }
  return vec2f(0.5) + rel / s;
}

// Compute object-fit: contain UV and inside-frame mask for a texture of size texSize inside resolution.
fn containUv(uv: vec2f, res: vec2f, texSize: vec2f) -> vec3f {
  let stageAspect = res.x / max(1.0, res.y);
  let texAspect = texSize.x / max(1.0, texSize.y);
  var scale = vec2f(1.0, 1.0);
  if (texAspect > stageAspect) {
    scale.y = stageAspect / texAspect;
  } else {
    scale.x = texAspect / stageAspect;
  }
  let cuv = (uv - vec2f(0.5)) / scale + vec2f(0.5);
  let inside = select(0.0, 1.0, cuv.x >= 0.0 && cuv.x <= 1.0 && cuv.y >= 0.0 && cuv.y <= 1.0);
  return vec3f(clamp(cuv, vec2f(0.0), vec2f(1.0)), inside);
}

fn rainLayer(uv: vec2f, t: f32, scale: vec2f, speed: f32, slantAmt: f32) -> f32 {
  let slant = vec2f(uv.x + uv.y * slantAmt, uv.y);
  let st = slant * scale + vec2f(0.0, -t * speed);
  let id = floor(st);
  let f = fract(st);
  let rnd = hash21(id);
  if (rnd < 0.58) { return 0.0; }
  let x = fract(rnd * 17.3) * 0.62 + 0.19;
  let yStart = fract(rnd * 31.7) * 0.30;
  let dx = abs(f.x - x);
  // Tapered droplet: gentle fading tail at top, bright leading tip at bottom.
  let streak = invSmooth(0.0, 0.044, dx)
    * smoothstep(yStart, yStart + 0.34, f.y)
    * invSmooth(yStart + 0.50, yStart + 0.60, f.y);
  return streak * (0.45 + 0.55 * rnd);
}

fn particleDots(uv: vec2f, aspect: f32, t: f32, grid: f32, speedY: f32, sway: f32, radius: f32, soft: f32) -> vec2f {
  let p = vec2f(uv.x * aspect, uv.y) * grid;
  var acc = 0.0;
  var tint = 0.0;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let cell = floor(p + vec2f(0.0, -t * speedY)) + vec2f(f32(dx), f32(dy));
      let rnd = hash22(cell);
      if (rnd.x > 0.38) {
        let phase = rnd.y * 6.2831;
        let offset = vec2f(
          0.5 + sin(t * 0.9 + phase) * sway * 0.35,
          0.5 + (rnd.x - 0.5) * 0.55
        );
        let pos = cell + offset;
        let cur = p + vec2f(0.0, -t * speedY);
        let d = length(cur - pos);
        let r = radius * (0.55 + 0.45 * rnd.y);
        let alpha = invSmooth(r * 0.2, r * soft, d) * (0.45 + 0.55 * rnd.y);
        acc = max(acc, alpha);
        tint = max(tint, rnd.y * alpha);
      }
    }
  }
  return vec2f(acc, tint);
}

// 3x3 neighbor-searched 3D tumbling petals so blossoms never clip at grid cell edges.
fn petalLayer(uv: vec2f, aspect: f32, t: f32, grid: f32, speedX: f32, speedY: f32, sizeScale: f32) -> vec4f {
  let wind = vec2f(-t * speedX, -t * speedY);
  let p = vec2f(uv.x * aspect + uv.y * 0.28, uv.y) * grid + wind;
  let baseCell = floor(p);
  var out = vec4f(0.0);
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let cell = baseCell + vec2f(f32(dx), f32(dy));
      let rnd = hash22(cell);
      if (rnd.x > 0.48) {
        let phase = rnd.y * 6.2831;
        let sway = vec2f(
          sin(t * (1.1 + rnd.x * 0.7) + phase) * 0.34,
          cos(t * (0.85 + rnd.y * 0.5) + phase) * 0.22
        );
        let center = cell + vec2f(0.5) + sway;
        let rel = p - center;
        let ang = t * (1.4 + rnd.y * 1.9) + rnd.x * 6.2831;
        let cs = cos(ang);
        let sn = sin(ang);
        let q = vec2f(rel.x * cs - rel.y * sn, rel.x * sn + rel.y * cs);
        let tumble = 0.28 + 0.72 * abs(cos(t * (1.6 + rnd.x * 1.4) + phase));
        let shaped = vec2f(q.x * 1.05, (q.y - q.x * q.x * 0.35) / max(0.25, tumble) * 1.95);
        let d = length(shaped) / sizeScale;
        let petalA = invSmooth(0.11, 0.22, d) * (0.62 + 0.25 * rnd.y);
        if (petalA > 0.001) {
          let baseRgb = mix(vec3f(0.99, 0.75, 0.82), vec3f(0.96, 0.58, 0.40), rnd.y);
          let petalRgb = baseRgb * (0.82 + 0.22 * tumble);
          out = blendOver(out, petalRgb, petalA);
        }
      }
    }
  }
  return out;
}

// 3x3 neighbor-searched bioluminescent fireflies drifting on curved paths with independent breathing pulses.
fn fireflyLayer(uv: vec2f, aspect: f32, t: f32, grid: f32, radius: f32) -> vec4f {
  let p = vec2f(uv.x * aspect, uv.y) * grid + vec2f(0.0, t * 0.08);
  let baseCell = floor(p);
  var out = vec4f(0.0);
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let cell = baseCell + vec2f(f32(dx), f32(dy));
      let rnd = hash22(cell);
      if (rnd.x > 0.46) {
        let phase = rnd.y * 6.2831;
        let drift = vec2f(
          sin(t * (0.55 + rnd.x * 0.45) + phase) * 0.42,
          cos(t * (0.48 + rnd.y * 0.40) + phase * 1.3) * 0.38
        );
        let center = cell + vec2f(0.5) + drift;
        let d = length(p - center);
        let pulse = 0.22 + 0.78 * pow(0.5 + 0.5 * sin(t * (1.7 + rnd.x * 2.1) + phase), 2.0);
        let r = radius * (0.7 + 0.45 * rnd.y);
        let core = invSmooth(0.0, r * 0.65, d) * pulse;
        let halo = invSmooth(0.0, r * 2.6, d) * pulse * 0.58;
        if (halo > 0.002) {
          let glowColor = mix(vec3f(0.68, 0.98, 0.42), vec3f(1.0, 0.92, 0.52), clamp(core * 1.35 + rnd.y * 0.3, 0.0, 1.0));
          out = blendScreen(out, glowColor, clamp(core + halo, 0.0, 0.95));
        }
      }
    }
  }
  return out;
}

fn lightningEnvelope(t: f32, shotSeed: f32) -> vec2f {
  let seedOffset = select(0.0, fract(sin(shotSeed * 91.7 + 13.3) * 43758.5453) * 4.5, shotSeed > 0.0);
  let boltX = select(0.52, 0.22 + fract(sin(shotSeed * 47.3) * 43758.5453) * 0.56, shotSeed > 0.0);
  let phase = (t + seedOffset) % 7.0;
  let primary = max(0.0, 1.0 - abs(phase - 0.12) / 0.12) * 0.32;
  let echo = max(0.0, 1.0 - abs(phase - 0.30) / 0.08) * 0.14;
  return vec2f(max(primary, echo), boltX);
}

// Compute UV refraction and chromatic aberration strength for a layer before sampling its still texture.
fn computeDistortion(sceneUv: vec2f, aspect: f32, t: f32, cueT: f32, atmo: u32, light: u32, shake: f32, shotSeed: f32) -> vec3f {
  var offset = vec2f(0.0);
  var ca = shake * 0.0065;
  let auv = vec2f(sceneUv.x * aspect, sceneUv.y);

  if (atmo == 1u) { // rain refraction droplets/streaks
    let r1 = rainLayer(sceneUv, t, vec2f(22.0 * aspect, 6.5), 4.8, 0.26);
    let r2 = rainLayer(sceneUv + vec2f(0.37, 0.19), t, vec2f(36.0 * aspect, 11.0), 3.5, 0.24);
    let ripple = (r1 * 0.7 + r2 * 0.4);
    offset += vec2f(-0.0025, 0.004) * ripple;
  } else if (atmo == 3u || atmo == 7u) { // fog or smoke atmospheric shimmer
    let n = fbm(auv * 3.5 + vec2f(t * 0.14, -t * 0.18)) - 0.5;
    offset += vec2f(n * 0.0018, n * 0.0012);
  } else if (atmo == 9u) { // underwater aquatic wave refraction
    let wX = sin(sceneUv.y * 9.5 + t * 1.35) * 0.0032 + (fbm(auv * 2.8 + vec2f(t * 0.22, -t * 0.16)) - 0.5) * 0.0025;
    let wY = cos(sceneUv.x * 8.0 + t * 1.1) * 0.0024;
    offset += vec2f(wX, wY);
    ca = max(ca, 0.0014);
  }

  if (light == 1u) { // lightning surge chromatic aberration
    let env = lightningEnvelope(t, shotSeed);
    ca = max(ca, env.x * 0.011);
  } else if (light == 2u) { // flash impact chromatic aberration (sound-cue aware)
    let a = max(0.0, 1.0 - cueT / 0.45);
    ca = max(ca, a * 0.008);
  } else if (light == 5u) { // firelight heat shimmer
    let heat = (fbm(auv * 5.0 + vec2f(sin(t * 1.3) * 0.2, t * 0.85)) - 0.5) * smoothstep(0.25, 1.0, sceneUv.y);
    offset += vec2f(heat * 0.0022, heat * 0.0016);
  } else if (light == 8u) { // dream_bloom subtle edge refraction
    let fromCenter = sceneUv - vec2f(0.5);
    let r = length(fromCenter);
    let wave = sin(t * 0.85 + r * 6.0) * smoothstep(0.18, 0.75, r) * 0.003;
    offset += fromCenter * wave;
    ca = max(ca, smoothstep(0.22, 0.8, r) * 0.0028);
  } else if (light == 9u) { // scanlines horizontal CRT micro-jitter
    let row = floor(sceneUv.y * 240.0);
    let glitch = (hash21(vec2f(row, floor(t * 18.0))) - 0.5) * 0.0012;
    offset.x += glitch;
    ca = max(ca, 0.0015);
  }

  return vec3f(offset, ca);
}

// Sample a still texture with camera transform, rotation, rack-focus blur, UV refraction, chromatic aberration, and highlight bloom.
fn sampleScene(
  tex: texture_2d<f32>,
  texSize: vec2f,
  uv: vec2f,
  res: vec2f,
  cam: vec4f,
  angleRad: f32,
  camBlur: f32,
  t: f32,
  cueT: f32,
  atmo: u32,
  light: u32,
  shotSeed: f32
) -> vec4f {
  let aspect = res.x / max(1.0, res.y);
  let camUv = cameraUv(uv, cam, angleRad, aspect);
  let dist = computeDistortion(camUv, aspect, t, cueT, atmo, light, cam.w, shotSeed);
  let warpedUv = camUv + dist.xy;
  let mapped = containUv(warpedUv, res, texSize);
  let tuv = mapped.xy;
  let inside = mapped.z;

  // Radial chromatic aberration around frame center.
  let radial = (uv - vec2f(0.5)) * dist.z;
  var rgb = textureSampleLevel(tex, texSampler, tuv, 0.0).rgb;
  if (dist.z > 0.0001) {
    rgb.r = textureSampleLevel(tex, texSampler, tuv + radial, 0.0).r;
    rgb.b = textureSampleLevel(tex, texSampler, tuv - radial, 0.0).b;
  }

  // Real image-sampled optical highlight bloom & 12-tap Vogel-disk rack-focus defocus kernel.
  let needsBloom = ((light == 1u && dist.z > 0.0) || (light == 2u && cueT < 0.45) ||
    light == 3u || light == 5u || light == 8u || light == 10u || atmo == 6u || atmo == 9u);
  if ((needsBloom || camBlur > 0.001) && inside > 0.0) {
    let texel = vec2f(1.0) / max(vec2f(64.0), res * 0.45);
    let maxRadius = texel * (6.5 + camBlur * 16.5);
    var vogel = array<vec2f, 12>(
      vec2f( 0.204,  0.000), vec2f(-0.261,  0.239), vec2f( 0.039, -0.455),
      vec2f( 0.340,  0.427), vec2f(-0.618, -0.105), vec2f( 0.564, -0.391),
      vec2f(-0.181,  0.719), vec2f(-0.426, -0.663), vec2f( 0.831,  0.145),
      vec2f(-0.768,  0.449), vec2f( 0.299, -0.884), vec2f( 0.483,  0.854)
    );
    var blur = vec3f(0.0);
    for (var i = 0; i < 12; i++) {
      blur += textureSampleLevel(tex, texSampler, clamp(tuv + vogel[i] * maxRadius, vec2f(0.0), vec2f(1.0)), 0.0).rgb;
    }
    blur *= (1.0 / 12.0);
    if (camBlur > 0.001) {
      rgb = mix(rgb, blur, clamp(camBlur * 0.90, 0.0, 0.90));
    }
    let luma = dot(blur, vec3f(0.299, 0.587, 0.114));
    let highlight = blur * smoothstep(0.42, 0.92, luma);

    if (light == 8u) { // dream_bloom: ethereal halation + soft pastel lift
      let pulse = 0.34 + sin(t * 0.55) * 0.06;
      rgb = mix(rgb, blur, 0.14);
      rgb += highlight * vec3f(1.05, 0.88, 1.02) * pulse;
    } else if (light == 3u) { // warm_glow: golden highlight bloom
      let pulse = 0.24 + sin(t * 0.8) * 0.06;
      rgb += highlight * vec3f(1.08, 0.82, 0.52) * pulse;
    } else if (light == 5u) { // flicker: organic ember halation
      let wave = 0.5 + 0.3 * sin(t * 7.3) + 0.2 * sin(t * 13.1);
      rgb += highlight * vec3f(1.1, 0.72, 0.35) * (0.12 + wave * 0.22);
    } else if (light == 1u) { // lightning: cool storm highlight bloom
      let env = lightningEnvelope(t, shotSeed);
      rgb += highlight * vec3f(0.85, 0.94, 1.15) * (env.x * 1.4);
    } else if (light == 2u) { // flash: reveal highlight bloom (sound-cue aware)
      let a = max(0.0, 1.0 - cueT / 0.45);
      rgb += highlight * vec3f(1.0, 0.98, 0.95) * (a * 0.36);
    } else if (light == 10u) { // spotlight: warm stage highlight bloom inside cone
      rgb += highlight * vec3f(1.05, 0.95, 0.78) * 0.22;
    } else if (atmo == 6u) { // sunbeams: warm light scatter
      rgb += highlight * vec3f(1.05, 0.92, 0.68) * 0.18;
    } else if (atmo == 9u) { // underwater: cyan caustic glow on highlights
      rgb += highlight * vec3f(0.45, 0.92, 1.05) * 0.16;
    }
  }

  // Image-level color grading & midtone-weighted grain when sampling scene pixels directly.
  if (inside > 0.0) {
    if (light == 4u) { // cold_tint: moonlight shadow/midtone split-toning preserving skin/highlight luma
      let luma = dot(rgb, vec3f(0.299, 0.587, 0.114));
      let cool = mix(vec3f(0.14, 0.28, 0.52), vec3f(0.74, 0.86, 1.02), luma);
      let shadowWeight = 0.22 - 0.08 * smoothstep(0.55, 0.95, luma);
      rgb = mix(rgb, cool, shadowWeight);
    } else if (light == 7u) { // film_grain: midtone-weighted 35mm emulsion response
      let luma = dot(rgb, vec3f(0.299, 0.587, 0.114));
      let midWeight = 0.35 + 2.6 * luma * (1.0 - luma);
      let frame = floor(t * 24.0);
      let g = hash21(floor(uv * res * 0.5) + vec2f(frame * 13.7, frame * 7.3)) - 0.5;
      rgb = clamp(rgb + vec3f(g * 0.11 * midWeight), vec3f(0.0), vec3f(1.0));
    }
  }

  return vec4f(clamp(rgb, vec3f(0.0), vec3f(1.0)) * inside, inside);
}

// Render one shot's procedural atmosphere + lighting layer (using camera-coupled UVs for world-space effects).
fn renderLayer(
  screenUv: vec2f,
  res: vec2f,
  t: f32,
  cueT: f32,
  atmo: u32,
  light: u32,
  cam: vec4f,
  angleRad: f32,
  shotSeed: f32,
  hasBaseTex: bool
) -> vec4f {
  let aspect = res.x / max(1.0, res.y);
  let worldUv = cameraUv(screenUv, cam, angleRad, aspect);
  let wAuv = vec2f(worldUv.x * aspect, worldUv.y);
  var out = vec4f(0.0);

  // --- Atmosphere Pass (world-space camera-coupled) ---
  if (atmo == 1u) { // 3-layer rain (foreground fast, midground steady, background fine sheet)
    let r1 = rainLayer(worldUv, t, vec2f(22.0 * aspect, 6.5), 4.8, 0.26);
    let r2 = rainLayer(worldUv + vec2f(0.37, 0.19), t, vec2f(36.0 * aspect, 11.0), 3.5, 0.24);
    let r3 = rainLayer(worldUv + vec2f(0.71, 0.43), t, vec2f(62.0 * aspect, 19.0), 2.3, 0.22);
    let rainA = clamp(r1 * 0.34 + r2 * 0.24 + r3 * 0.15, 0.0, 0.62);
    out = blendOver(out, vec3f(0.77, 0.87, 0.97), rainA);
    let mistNoise = 0.85 + 0.3 * noise2(vec2f(wAuv.x * 3.0 + t * 0.25, worldUv.y * 4.0));
    let mist = smoothstep(0.66, 1.0, screenUv.y) * 0.13 * mistNoise;
    out = blendOver(out, vec3f(0.72, 0.80, 0.88), mist);
  } else if (atmo == 2u) { // 3-layer snow (foreground soft bokeh, midground, distant)
    let s0 = particleDots(worldUv + vec2f(0.19, 0.71), aspect, t, 6.5, 0.78, 0.75, 0.21, 2.7);
    let s1 = particleDots(worldUv, aspect, t, 11.0, 0.55, 0.55, 0.16, 2.2);
    let s2 = particleDots(worldUv + vec2f(0.41, 0.23), aspect, t, 18.0, 0.35, 0.4, 0.12, 2.0);
    let snowA = clamp(s0.x * 0.42 + s1.x * 0.70 + s2.x * 0.42, 0.0, 0.88);
    out = blendOver(out, vec3f(0.95, 0.98, 1.0), snowA);
  } else if (atmo == 3u) { // fog
    let n1 = fbm(wAuv * 2.1 + vec2f(t * 0.06, sin(t * 0.12) * 0.05));
    let n2 = fbm(wAuv * 3.6 + vec2f(-t * 0.04, 2.4));
    let fogMask = smoothstep(0.20, 0.95, worldUv.y);
    let density = smoothstep(0.32, 0.82, n1 * 0.65 + n2 * 0.35) * fogMask * 0.42;
    out = blendOver(out, vec3f(0.84, 0.90, 0.93), density);
  } else if (atmo == 4u) { // dust
    let d1 = particleDots(worldUv, aspect, t, 9.5, -0.14, 0.65, 0.11, 2.3);
    out = blendScreen(out, vec3f(1.0, 0.91, 0.72), d1.x * 0.45);
  } else if (atmo == 5u) { // embers with organic twinkle
    let e1 = particleDots(worldUv, aspect, t, 10.5, -0.85, 0.75, 0.15, 2.6);
    let e2 = particleDots(worldUv + vec2f(0.31, 0.17), aspect, t, 16.0, -0.62, 0.6, 0.11, 2.4);
    let flicker = 0.72 + 0.28 * sin(t * 9.5 + (e1.y + e2.y) * 12.0);
    let emberA = clamp((e1.x * 0.9 + e2.x * 0.6) * flicker, 0.0, 1.0);
    let emberRgb = mix(vec3f(1.0, 0.38, 0.10), vec3f(1.0, 0.84, 0.38), clamp(e1.y + e2.y, 0.0, 1.0));
    out = blendScreen(out, emberRgb, emberA);
  } else if (atmo == 6u) { // radial volumetric sunbeams
    let sunPos = vec2f(0.24 * aspect, -0.18);
    let delta = wAuv - sunPos;
    let dist = length(delta);
    let angle = atan2(delta.y, delta.x);
    let r1 = fbm(vec2f(angle * 7.5 + sin(t * 0.16) * 0.22, dist * 1.15 - t * 0.04));
    let r2 = fbm(vec2f(angle * 14.5 - t * 0.06, dist * 1.9 - t * 0.05));
    let cone = smoothstep(0.12, 0.52, sin(angle)) * invSmooth(0.08, 1.65, dist / max(1.0, aspect * 0.82));
    let shafts = (smoothstep(0.38, 0.82, r1) * 0.72 + smoothstep(0.45, 0.88, r2) * 0.34) * cone;
    let sourceGlow = invSmooth(0.0, 0.95 * aspect, dist) * 0.14;
    out = blendScreen(out, vec3f(1.0, 0.92, 0.74), clamp(shafts * 0.34 + sourceGlow, 0.0, 0.52));
  } else if (atmo == 7u) { // smoke
    let s1 = fbm(wAuv * 2.6 + vec2f(sin(t * 0.25) * 0.15, t * 0.18));
    let s2 = fbm(wAuv * 4.8 + vec2f(-t * 0.08, t * 0.26));
    let density = smoothstep(0.38, 0.86, s1 * 0.6 + s2 * 0.4) * smoothstep(0.0, 0.85, worldUv.y) * 0.38;
    out = blendOver(out, vec3f(0.46, 0.49, 0.54), density);
  } else if (atmo == 8u) { // 2-layer neighbor-searched 3D tumbling petals (no cell boundary clipping)
    let pMid = petalLayer(worldUv, aspect, t, 8.2, 0.34, 0.42, 1.0);
    let pNear = petalLayer(worldUv + vec2f(0.37, 0.21), aspect, t, 5.4, 0.46, 0.56, 1.18);
    out = blendOver(out, pMid.rgb, pMid.a);
    out = blendOver(out, pNear.rgb, pNear.a);
  } else if (atmo == 9u) { // underwater caustics + rising silvery bubbles
    let c1 = fbm(wAuv * 3.8 + vec2f(t * 0.16, -t * 0.11));
    let c2 = fbm(wAuv * 4.2 + vec2f(-t * 0.14, t * 0.18));
    let causticRidge = pow(clamp(1.0 - abs(c1 - c2) * 2.6, 0.0, 1.0), 3.0);
    let causticMask = invSmooth(0.05, 0.92, worldUv.y) * 0.26;
    out = blendScreen(out, vec3f(0.42, 0.90, 1.0), causticRidge * causticMask);
    let b1 = particleDots(worldUv, aspect, t, 10.0, -0.58, 0.72, 0.14, 1.85);
    let b2 = particleDots(worldUv + vec2f(0.33, 0.27), aspect, t, 16.5, -0.42, 0.55, 0.10, 1.75);
    let bubbleA = clamp(b1.x * 0.68 + b2.x * 0.45, 0.0, 0.82);
    out = blendScreen(out, vec3f(0.80, 0.96, 1.0), bubbleA);
  } else if (atmo == 10u) { // bioluminescent fireflies
    let f1 = fireflyLayer(worldUv, aspect, t, 6.2, 0.16);
    let f2 = fireflyLayer(worldUv + vec2f(0.43, 0.29), aspect, t * 0.85, 9.5, 0.12);
    out = blendScreen(out, f2.rgb, f2.a);
    out = blendScreen(out, f1.rgb, f1.a);
  }

  // --- Lighting & Framing Pass ---
  if (light == 1u) { // shot-seeded lightning
    let env = lightningEnvelope(t, shotSeed);
    let a = env.x;
    let boltX = env.y;
    let skyDist = length((worldUv - vec2f(boltX, 0.16)) * vec2f(aspect * 0.75, 1.0));
    let cloudMod = 0.75 + 0.35 * fbm(wAuv * 3.2 + vec2f(shotSeed * 1.7, 0.0));
    let sky = invSmooth(0.0, 1.15, skyDist) * a * 0.82 * cloudMod;
    out = blendScreen(out, vec3f(0.88, 0.93, 1.0), clamp(a + sky, 0.0, 0.58));
  } else if (light == 2u) { // flash (triggers at shot start and on one-shot sound cues)
    let a = max(0.0, 1.0 - cueT / 0.45) * 0.35;
    out = blendScreen(out, vec3f(0.94, 0.96, 1.0), a);
  } else if (light == 3u) { // warm_glow
    let pulse = 0.11 + sin(t * 0.8) * 0.035;
    let center = vec2f(0.5 + sin(t * 0.22) * 0.06, 0.42 + cos(t * 0.18) * 0.04);
    let glow = invSmooth(0.0, 0.95, length((worldUv - center) * vec2f(aspect, 1.0)));
    out = blendScreen(out, vec3f(1.0, 0.72, 0.38), glow * pulse * 1.45);
  } else if (light == 4u) { // cold_tint (sampleScene grades shadows/midtones when hasBaseTex; add outer edge wash)
    let pulse = 0.09 + sin(t * 0.5) * 0.02;
    let edge = smoothstep(0.15, 1.05, length(screenUv - vec2f(0.5, 0.4)));
    let strength = select(mix(0.35, 1.0, edge) * 1.35, edge * 0.65, hasBaseTex);
    out = blendOver(out, vec3f(0.24, 0.44, 0.78), strength * pulse);
  } else if (light == 5u) { // flicker
    let wave = 0.5 + 0.3 * sin(t * 7.3) + 0.2 * sin(t * 13.1);
    let a = 0.04 + wave * 0.08;
    let center = vec2f(0.5 + sin(t * 5.1) * 0.02, 0.62);
    let glow = invSmooth(0.0, 0.85, length(worldUv - center));
    out = blendScreen(out, vec3f(1.0, 0.64, 0.28), glow * a * 1.6);
  } else if (light == 6u) { // vignette (screen-space)
    let d = length((screenUv - 0.5) * vec2f(1.15, 1.0));
    let vig = smoothstep(0.28, 0.82, d) * 0.56;
    out = blendOver(out, vec3f(0.0, 0.0, 0.0), vig);
  } else if (light == 7u && !hasBaseTex) { // film_grain (screen-space fallback when not already applied in sampleScene)
    let frame = floor(t * 24.0);
    let g = hash21(floor(screenUv * res * 0.5) + vec2f(frame * 13.7, frame * 7.3)) - 0.5;
    let grainRgb = select(vec3f(0.0), vec3f(1.0), g > 0.0);
    out = blendOver(out, grainRgb, abs(g) * 0.14);
  } else if (light == 8u) { // dream_bloom (multi-lobed halation + soft dreamy edge)
    let pulse = 0.14 + sin(t * 0.55) * 0.03;
    let c1 = vec2f(0.5 + sin(t * 0.25) * 0.05, 0.44 + cos(t * 0.2) * 0.04);
    let c2 = vec2f(0.46 - cos(t * 0.3) * 0.06, 0.52 + sin(t * 0.22) * 0.05);
    let halo1 = invSmooth(0.0, 0.92, length((worldUv - c1) * vec2f(aspect * 0.8, 1.0)));
    let halo2 = invSmooth(0.0, 0.78, length((worldUv - c2) * vec2f(aspect * 0.8, 1.0))) * 0.55;
    let rim = smoothstep(0.35, 0.92, length(screenUv - vec2f(0.5))) * 0.45;
    out = blendScreen(out, vec3f(0.98, 0.87, 0.96), clamp((halo1 + halo2 + rim) * pulse, 0.0, 0.38));
  } else if (light == 9u) { // scanlines (screen-space)
    let line = 0.5 - 0.5 * cos(screenUv.y * res.y * 3.14159);
    let bar = invSmooth(0.0, 0.08, abs(fract(screenUv.y - t * 0.08) - 0.5));
    let crtVig = smoothstep(0.42, 0.92, length((screenUv - 0.5) * vec2f(1.1, 1.05))) * 0.22;
    out = blendOver(out, vec3f(0.03, 0.07, 0.11), clamp(line * 0.16 + crtVig, 0.0, 0.45));
    out = blendScreen(out, vec3f(0.35, 0.85, 0.95), bar * 0.07);
  } else if (light == 10u) { // spotlight (overhead cone + darkened surround)
    let swayX = 0.5 + sin(t * 0.28) * 0.035;
    let coneHalfWidth = mix(0.11, 0.44, clamp(worldUv.y, 0.0, 1.0));
    let dx = abs(worldUv.x - swayX) / max(0.05, coneHalfWidth);
    let beam = invSmooth(0.35, 1.18, dx) * invSmooth(0.0, 1.15, worldUv.y * 0.72);
    let stagePool = invSmooth(0.0, 0.52, length((worldUv - vec2f(swayX, 0.54)) * vec2f(aspect * 0.75, 1.15)));
    let lit = clamp(max(beam, stagePool), 0.0, 1.0);
    let shade = (1.0 - lit) * 0.56;
    out = blendOver(out, vec3f(0.02, 0.03, 0.05), shade);
    out = blendScreen(out, vec3f(1.0, 0.94, 0.80), lit * 0.14);
  }

  return vec4f(out.rgb, clamp(out.a, 0.0, 1.0));
}

fn compositeOverBase(base: vec4f, layer: vec4f) -> vec4f {
  // The sampled shot includes the stage's black matte, including any contain bars.
  return vec4f(layer.rgb + base.rgb * (1.0 - layer.a), 1.0);
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  let uv = in.uv;
  let res = u.timeParams.xy;
  let t = max(0.0, u.timeParams.z);
  let backT = max(0.0, u.timeParams.w);
  let fadeMix = clamp(u.mixParams.x, 0.0, 1.0);
  let shotSeed = u.mixParams.y;
  let backShotSeed = u.mixParams.z;
  let texMode = u.mixParams.w;
  let camAngle = u.camExtra.x;
  let camBlur = u.camExtra.y;
  let backCamAngle = u.camExtra.z;
  let backCamBlur = u.camExtra.w;
  let cueT = max(0.0, u.cueParams.x);
  let backCueT = max(0.0, u.cueParams.y);
  let atmo = u.effectIds.x;
  let light = u.effectIds.y;
  let backAtmo = u.effectIds.z;
  let backLight = u.effectIds.w;

  let hasBaseTex = texMode >= 0.5;
  let frontLayer = renderLayer(uv, res, t, cueT, atmo, light, u.cam, camAngle, shotSeed, hasBaseTex);
  if (texMode >= 1.5) {
    let backLayer = renderLayer(uv, res, backT, backCueT, backAtmo, backLight, u.backCam, backCamAngle, backShotSeed, true);
    let backBase = sampleScene(backTex, u.texSizes.xy, uv, res, u.backCam, backCamAngle, backCamBlur, backT, backCueT, backAtmo, backLight, backShotSeed);
    let frontBase = sampleScene(frontTex, u.texSizes.zw, uv, res, u.cam, camAngle, camBlur, t, cueT, atmo, light, shotSeed);
    return mix(compositeOverBase(backBase, backLayer), compositeOverBase(frontBase, frontLayer), fadeMix);
  }
  if (hasBaseTex) {
    let base = sampleScene(backTex, u.texSizes.xy, uv, res, u.cam, camAngle, camBlur, t, cueT, atmo, light, shotSeed);
    return compositeOverBase(base, frontLayer);
  }
  if (fadeMix < 1.0) {
    let backLayer = renderLayer(uv, res, backT, backCueT, backAtmo, backLight, u.backCam, backCamAngle, backShotSeed, false);
    return mix(backLayer, frontLayer, fadeMix);
  }

  return frontLayer;
}
`;

let gpuState = "idle"; // "idle" | "probing" | "ready" | "unavailable"
let gpuDevice = null;
let gpuPipeline = null;
let gpuUniformBuffer = null;
let gpuSampler = null;
let gpuDummyTexture = null;
let gpuBindGroup = null;
let boundBackView = null;
let boundFrontView = null;
let gpuFormat = "bgra8unorm";
let gpuDummyView = null;

// 32 x 4 bytes = 128 bytes (8 vec4 blocks)
const uniformData = new ArrayBuffer(128);
const uniformFloats = new Float32Array(uniformData);
const uniformUints = new Uint32Array(uniformData);

const textureSlots = {
  back: null,
  front: null,
};

function resetTextureSlots() {
  for (const key of ["back", "front"]) {
    textureSlots[key]?.texture.destroy();
    textureSlots[key] = null;
  }
  gpuBindGroup = null;
  boundBackView = null;
  boundFrontView = null;
}

export function gpuStatus() {
  return gpuState;
}

export async function probeWebGPU(onChange) {
  if (gpuState !== "idle") return gpuState === "ready";
  const gpu = globalThis.navigator?.gpu;
  if (!gpu?.requestAdapter) {
    gpuState = "unavailable";
    return false;
  }
  gpuState = "probing";
  try {
    const adapter = await gpu.requestAdapter();
    if (!adapter) { gpuState = "unavailable"; return false; }
    const device = await adapter.requestDevice();
    gpuFormat = gpu.getPreferredCanvasFormat?.() || "bgra8unorm";
    const module = device.createShaderModule({ code: WGSL });
    const pipeline = await device.createRenderPipelineAsync({
      layout: "auto",
      vertex: { module, entryPoint: "vs" },
      fragment: { module, entryPoint: "fs", targets: [{ format: gpuFormat }] },
      primitive: { topology: "triangle-list" },
    });
    const uniformBuffer = device.createBuffer({
      size: 128,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const sampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });
    const dummyView = device.createTexture({
      size: [1, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    }).createView();
    device.lost.then(() => {
      resetTextureSlots();
      try { gpuUniformBuffer?.destroy?.(); } catch { /* ignore */ }
      gpuDevice = null;
      gpuPipeline = null;
      gpuUniformBuffer = null;
      gpuSampler = null;
      gpuDummyView = null;
      configuredCanvas = null;
      gpuState = "unavailable";
      onChange?.();
    });
    gpuDevice = device;
    gpuPipeline = pipeline;
    gpuUniformBuffer = uniformBuffer;
    gpuSampler = sampler;
    gpuDummyView = dummyView;
    resetTextureSlots();
    gpuState = "ready";
  } catch {
    gpuState = "unavailable";
    return false;
  }
  onChange?.();
  return true;
}

function readyStill(img) {
  return img?.complete && img.naturalWidth > 0 && img.naturalHeight > 0 && (img.currentSrc || img.src);
}

function syncStillTexture(name, img, src) {
  // Once a dissolve ends, its incoming image is already resident in the other slot.
  const other = name === "back" ? "front" : "back";
  if (textureSlots[name]?.src !== src && textureSlots[other]?.src === src) {
    [textureSlots[name], textureSlots[other]] = [textureSlots[other], textureSlots[name]];
  }
  let slot = textureSlots[name];
  if (slot?.src === src) return slot;
  const width = img.naturalWidth, height = img.naturalHeight;
  if (Math.max(width, height) > gpuDevice.limits.maxTextureDimension2D) return null;
  if (!slot || slot.width !== width || slot.height !== height) {
    const texture = gpuDevice.createTexture({
      size: [width, height],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    slot?.texture.destroy();
    slot = textureSlots[name] = {texture, view: texture.createView(), width, height, src: null};
  }
  slot.src = null;
  gpuDevice.queue.copyExternalImageToTexture({source: img}, {texture: slot.texture}, [width, height]);
  slot.src = src;
  return slot;
}

function ensureBindGroup(back, front) {
  const backView = back?.view || gpuDummyView, frontView = front?.view || gpuDummyView;
  if (!gpuBindGroup || backView !== boundBackView || frontView !== boundFrontView) {
    gpuBindGroup = gpuDevice.createBindGroup({
      layout: gpuPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gpuUniformBuffer } },
        { binding: 1, resource: gpuSampler },
        { binding: 2, resource: backView },
        { binding: 3, resource: frontView },
      ],
    });
    boundBackView = backView;
    boundFrontView = frontView;
  }
  return gpuBindGroup;
}

function getGpuCanvas(hostCanvas, create) {
  const parent = hostCanvas.parentElement;
  if (!parent) return null;
  let gpuCanvas = parent.querySelector(".still-effects-gpu");
  if (!gpuCanvas && create) {
    gpuCanvas = hostCanvas.ownerDocument.createElement("canvas");
    gpuCanvas.className = "still-effects still-effects-gpu";
    gpuCanvas.setAttribute("aria-hidden", "true");
    hostCanvas.after(gpuCanvas);
  }
  return gpuCanvas;
}

export function clearWebGPU(hostCanvas) {
  const gpuCanvas = getGpuCanvas(hostCanvas, false);
  if (gpuCanvas) gpuCanvas.style.visibility = "hidden";
  if (hostCanvas.style) hostCanvas.style.visibility = "visible";
}

const DEG_TO_RAD = Math.PI / 180;
const IMAGE_ATMOSPHERE = new Set(["rain", "fog", "sunbeams", "smoke", "underwater"].map((id) => ATMO_IDS[id]));
const IMAGE_LIGHT = new Set(Object.values(LIGHT_IDS).filter((id) => id !== LIGHT_IDS.vignette));
const usesImage = (atmo, light, cam) => IMAGE_ATMOSPHERE.has(atmo) || IMAGE_LIGHT.has(light) || cam?.blur || cam?.shake;
let configuredCanvas = null;
export function renderWebGPU(hostCanvas, atmoId, lightId, seconds, w, h, scene = null) {
  if (gpuState !== "ready" || !gpuDevice || !gpuPipeline) return false;
  const gpuCanvas = getGpuCanvas(hostCanvas, true);
  if (!gpuCanvas) return false;
  const ctx = gpuCanvas.getContext("webgpu");
  if (!ctx) return false;
  const fading = Boolean(scene?.fading);
  const backAtmoId = fading ? (scene.backAtmoId || 0) : 0;
  const backLightId = fading ? (scene.backLightId || 0) : 0;
  const backSrc = readyStill(scene?.stillBack);
  const frontSrc = fading && readyStill(scene.stillFront);
  let back = null, front = null;
  if (backSrc && (!fading || frontSrc) && (usesImage(atmoId, lightId, scene.cam) || usesImage(backAtmoId, backLightId, scene.backCam))) {
    try {
      back = syncStillTexture("back", scene.stillBack, backSrc);
      if (fading && back) {
        front = frontSrc === backSrc ? back : syncStillTexture("front", scene.stillFront, frontSrc);
        if (!front) back = null;
      }
    } catch {
      // The overlay can still run if the browser rejects a source texture.
      back = null;
    }
  }
  const texMode = front ? 2 : back ? 1 : 0;
  if (texMode) {
    const ratio = Math.min(globalThis.devicePixelRatio || 1, 1280 / Math.max(w, h));
    w = Math.max(1, Math.round(w * ratio));
    h = Math.max(1, Math.round(h * ratio));
  } else {
    const ratio = Math.min(1, 960 / Math.max(w, h));
    w = Math.max(1, Math.round(w * ratio));
    h = Math.max(1, Math.round(h * ratio));
  }
  const resized = gpuCanvas.width !== w || gpuCanvas.height !== h;
  if (resized) {
    gpuCanvas.width = w;
    gpuCanvas.height = h;
  }
  if (configuredCanvas !== gpuCanvas || resized) {
    ctx.configure({
      device: gpuDevice,
      format: gpuFormat,
      alphaMode: "premultiplied",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    configuredCanvas = gpuCanvas;
  }

  const bindGroup = ensureBindGroup(back, front);

  const cam = scene?.cam || { tx: 0, ty: 0, scale: 1, shake: 0, angle: 0, blur: 0 };
  const backCam = scene?.backCam || { tx: 0, ty: 0, scale: 1, shake: 0, angle: 0, blur: 0 };
  const fadeMix = fading ? Math.max(0, Math.min(1, scene?.fadeMix ?? 1)) : 1;

  // vec4f[0]: timeParams (w, h, time, backTime)
  uniformFloats[0] = w;
  uniformFloats[1] = h;
  uniformFloats[2] = seconds;
  uniformFloats[3] = scene?.backElapsed ?? 0;

  // vec4f[1]: mixParams (fadeMix, shotSeed, backShotSeed, texMode)
  uniformFloats[4] = fadeMix;
  uniformFloats[5] = scene?.shotIndex ?? 0;
  uniformFloats[6] = scene?.backShotIndex ?? 0;
  uniformFloats[7] = texMode;

  // vec4f[2]: cam (tx_frac, ty_frac, scale, shake)
  uniformFloats[8] = (cam.tx || 0) / 100;
  uniformFloats[9] = (cam.ty || 0) / 100;
  uniformFloats[10] = cam.scale || 1;
  uniformFloats[11] = cam.shake || 0;

  // vec4f[3]: backCam (tx_frac, ty_frac, scale, shake)
  uniformFloats[12] = (backCam.tx || 0) / 100;
  uniformFloats[13] = (backCam.ty || 0) / 100;
  uniformFloats[14] = backCam.scale || 1;
  uniformFloats[15] = backCam.shake || 0;

  // vec4f[4]: texSizes (backW, backH, frontW, frontH)
  uniformFloats[16] = back?.width ?? w;
  uniformFloats[17] = back?.height ?? h;
  uniformFloats[18] = front?.width ?? w;
  uniformFloats[19] = front?.height ?? h;

  // vec4f[5]: camExtra (camAngleRad, camBlur, backCamAngleRad, backCamBlur)
  uniformFloats[20] = (cam.angle || 0) * DEG_TO_RAD;
  uniformFloats[21] = cam.blur || 0;
  uniformFloats[22] = (backCam.angle || 0) * DEG_TO_RAD;
  uniformFloats[23] = backCam.blur || 0;

  // vec4f[6]: cueParams (cueTime, backCueTime, 0, 0)
  uniformFloats[24] = scene?.cueElapsed ?? seconds;
  uniformFloats[25] = scene?.backCueElapsed ?? (scene?.backElapsed ?? 0);
  uniformFloats[26] = 0;
  uniformFloats[27] = 0;

  // vec4u[7]: effectIds (atmoId, lightId, backAtmoId, backLightId)
  uniformUints[28] = atmoId;
  uniformUints[29] = lightId;
  uniformUints[30] = backAtmoId;
  uniformUints[31] = backLightId;

  gpuDevice.queue.writeBuffer(gpuUniformBuffer, 0, uniformData);

  const encoder = gpuDevice.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: ctx.getCurrentTexture().createView(),
      clearValue: { r: 0, g: 0, b: 0, a: 0 },
      loadOp: "clear",
      storeOp: "store",
    }],
  });
  pass.setPipeline(gpuPipeline);
  pass.setBindGroup(0, bindGroup);
  pass.draw(3);
  pass.end();
  gpuDevice.queue.submit([encoder.finish()]);

  gpuCanvas.style.visibility = "visible";
  if (hostCanvas.style) hostCanvas.style.visibility = "hidden";
  return true;
}
