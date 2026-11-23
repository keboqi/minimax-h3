// WebGPU WGSL unified texture compositor, Depth Pro 2 volumetric/optical backend, and SAM-masked layer compositor.
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
  // x: fadeMix (0..1), y: shotSeed, z: backShotSeed, w: texMode (0 = overlay, 1 = single tex, 2 = dissolve pair)
  mixParams: vec4f,
  // x: tx (fraction), y: ty (fraction), z: scale, w: shake (0..1)
  cam: vec4f,
  // x: backTx, y: backTy, z: backScale, w: backShake
  backCam: vec4f,
  // x: backTexW, y: backTexH, z: frontTexW, w: frontTexH
  texSizes: vec4f,
  // x: camAngleRad, y: camBlur (0..1), z: backCamAngleRad, w: backCamBlur (0..1)
  camExtra: vec4f,
  // x: cueTime, y: backCueTime, z: rackActive, w: backRackActive
  cueParams: vec4f,
  // x: atmo, y: light, z: backAtmo, w: backLight
  effectIds: vec4u,
  // x: hasDepth, y: depthStrength, z: invMin, w: invMax
  depthMeta0: vec4f,
  // x: subjectZ, y: rackStartZ, z: fPxNorm, w: parallaxScale
  depthMeta1: vec4f,
  // x: zP10, y: zP25, z: zP50, w: zP90
  depthMeta2: vec4f,
  // x: warmU, y: warmV, z: warmZ, w: warmStrength
  depthAnchors0: vec4f,
  // x: skyU, y: skyV, z: subjectU, w: subjectV
  depthAnchors1: vec4f,
  // Back-shot equivalents during dissolves
  backDepthMeta0: vec4f,
  backDepthMeta1: vec4f,
  backDepthMeta2: vec4f,
  backDepthAnchors0: vec4f,
  backDepthAnchors1: vec4f,
  // x: intensity, y: maskMode (0=full_frame,1=inside,2=outside,3=behind), z: maskRefZ, w: hasMask
  layerParams: vec4f,
};

struct DepthParams {
  hasDepth: f32,
  depthStrength: f32,
  invMin: f32,
  invMax: f32,
  subjectZ: f32,
  rackStartZ: f32,
  fPxNorm: f32,
  parallaxScale: f32,
  zP10: f32,
  zP25: f32,
  zP50: f32,
  zP90: f32,
  warmAnchor: vec4f,
  skyDir: vec2f,
  subjectUv: vec2f,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var texSampler: sampler;
@group(0) @binding(2) var backTex: texture_2d<f32>;
@group(0) @binding(3) var frontTex: texture_2d<f32>;
@group(0) @binding(4) var backDepthTex: texture_2d<f32>;
@group(0) @binding(5) var frontDepthTex: texture_2d<f32>;
@group(0) @binding(6) var maskTex: texture_2d<f32>;

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

fn frontDepthParams() -> DepthParams {
  return DepthParams(
    u.depthMeta0.x, u.depthMeta0.y, u.depthMeta0.z, u.depthMeta0.w,
    u.depthMeta1.x, u.depthMeta1.y, u.depthMeta1.z, u.depthMeta1.w,
    u.depthMeta2.x, u.depthMeta2.y, u.depthMeta2.z, u.depthMeta2.w,
    u.depthAnchors0, u.depthAnchors1.xy, u.depthAnchors1.zw
  );
}

fn backDepthParams() -> DepthParams {
  return DepthParams(
    u.backDepthMeta0.x, u.backDepthMeta0.y, u.backDepthMeta0.z, u.backDepthMeta0.w,
    u.backDepthMeta1.x, u.backDepthMeta1.y, u.backDepthMeta1.z, u.backDepthMeta1.w,
    u.backDepthMeta2.x, u.backDepthMeta2.y, u.backDepthMeta2.z, u.backDepthMeta2.w,
    u.backDepthAnchors0, u.backDepthAnchors1.xy, u.backDepthAnchors1.zw
  );
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
  for (var i: i32 = 0; i < 4; i++) {
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
  let outA = a + dst.a * (1.0 - a);
  return vec4f(rgb, outA);
}

fn invSmooth(low: f32, high: f32, x: f32) -> f32 {
  return 1.0 - smoothstep(low, high, x);
}

// Exact integer tap decoder for 16-bit packed RG8 inverse depth + B sky mask (never hardware-linear filtered).
fn decodeDepthTap(depthTex: texture_2d<f32>, coord: vec2i, dims: vec2i, dp: DepthParams) -> vec2f {
  let c = clamp(coord, vec2i(0), dims - vec2i(1));
  let px = textureLoad(depthTex, c, 0);
  let rByte = round(px.r * 255.0);
  let gByte = round(px.g * 255.0);
  let q = rByte * 256.0 + gByte;
  let uNorm = clamp(q / 65535.0, 0.0, 1.0);
  let invSpan = max(1e-4, dp.invMax - dp.invMin);
  let invZ = dp.invMin + uNorm * invSpan;
  return vec2f(invZ, px.b);
}

// Bilinearly interpolated depth sample -> (zMeters, invZ, uNorm, skyMask).
fn sampleDepthInfo(depthTex: texture_2d<f32>, uv: vec2f, dp: DepthParams) -> vec4f {
  let dims = vec2i(textureDimensions(depthTex));
  let fc = clamp(uv, vec2f(0.0), vec2f(1.0)) * vec2f(dims) - vec2f(0.5);
  let i0 = vec2i(floor(fc));
  let f = fract(fc);
  let t00 = decodeDepthTap(depthTex, i0, dims, dp);
  let t10 = decodeDepthTap(depthTex, i0 + vec2i(1, 0), dims, dp);
  let t01 = decodeDepthTap(depthTex, i0 + vec2i(0, 1), dims, dp);
  let t11 = decodeDepthTap(depthTex, i0 + vec2i(1, 1), dims, dp);
  let m0 = mix(t00, t10, f.x);
  let m1 = mix(t01, t11, f.x);
  let res = mix(m0, m1, f.y);
  let invZ = res.x;
  let sky = res.y;
  let z = clamp(1.0 / max(invZ, 1e-4), 0.1, 150.0);
  let uNorm = clamp((invZ - dp.invMin) / max(1e-4, dp.invMax - dp.invMin), 0.0, 1.0);
  return vec4f(z, invZ, uNorm, sky);
}

// Fast single-tap depth lookup -> (zMeters, skyMask) for inner Vogel/ray-march loops.
fn decodeDepthFast(depthTex: texture_2d<f32>, uv: vec2f, dp: DepthParams) -> vec2f {
  let dims = vec2i(textureDimensions(depthTex));
  let c = clamp(vec2i(clamp(uv, vec2f(0.0), vec2f(1.0)) * vec2f(dims)), vec2i(0), dims - vec2i(1));
  let tap = decodeDepthTap(depthTex, c, dims, dp);
  let z = clamp(1.0 / max(tap.x, 1e-4), 0.1, 150.0);
  return vec2f(z, tap.y);
}

fn unprojectPos(uv: vec2f, z: f32, texSize: vec2f, fPxNorm: f32) -> vec3f {
  let maxDim = max(texSize.x, texSize.y);
  let fPx = max(16.0, fPxNorm * maxDim);
  let xy = (uv - vec2f(0.5)) * texSize * (z / fPx);
  return vec3f(xy, z);
}

fn reconstructPosNormal(depthTex: texture_2d<f32>, uv: vec2f, texSize: vec2f, dp: DepthParams) -> vec3f {
  let dims = max(vec2f(16.0), texSize);
  let stepUv = vec2f(2.0) / dims;
  let z0 = decodeDepthFast(depthTex, uv, dp).x;
  let zL = decodeDepthFast(depthTex, uv - vec2f(stepUv.x, 0.0), dp).x;
  let zR = decodeDepthFast(depthTex, uv + vec2f(stepUv.x, 0.0), dp).x;
  let zU = decodeDepthFast(depthTex, uv - vec2f(0.0, stepUv.y), dp).x;
  let zD = decodeDepthFast(depthTex, uv + vec2f(0.0, stepUv.y), dp).x;
  let p0 = unprojectPos(uv, z0, dims, dp.fPxNorm);
  let pL = unprojectPos(uv - vec2f(stepUv.x, 0.0), zL, dims, dp.fPxNorm);
  let pR = unprojectPos(uv + vec2f(stepUv.x, 0.0), zR, dims, dp.fPxNorm);
  let pU = unprojectPos(uv - vec2f(0.0, stepUv.y), zU, dims, dp.fPxNorm);
  let pD = unprojectPos(uv + vec2f(0.0, stepUv.y), zD, dims, dp.fPxNorm);
  let pX = select(p0 - pL, pR - p0, abs(zR - z0) < abs(z0 - zL));
  let pY = select(p0 - pU, pD - p0, abs(zD - z0) < abs(z0 - zU));
  let cN = cross(pY, pX);
  let nLen = length(cN);
  var n = select(vec3f(0.0, 0.0, -1.0), cN / max(1e-5, nLen), nLen > 1e-6);
  if (n.z > 0.0) { n = -n; }
  return n;
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

// 4-step iterative subject_z-pivoted 2.5D parallax relief mapping.
fn parallaxContainUv(
  camUv: vec2f,
  res: vec2f,
  texSize: vec2f,
  depthTex: texture_2d<f32>,
  dp: DepthParams,
  cam: vec4f
) -> vec3f {
  let base = containUv(camUv, res, texSize);
  let strength = dp.depthStrength * dp.parallaxScale;
  if (dp.hasDepth <= 0.5 || strength <= 0.01 || base.z <= 0.5) {
    return base;
  }
  let panVec = -cam.xy;
  let dolly = cam.z - 1.0;
  if (abs(panVec.x) + abs(panVec.y) + abs(dolly) <= 0.0005) {
    return base;
  }
  let invSpan = max(0.08, dp.invMax - dp.invMin);
  let invPivot = 1.0 / max(dp.subjectZ, 0.25);
  var pUv = base.xy;
  let maxShift = 0.022 * strength;
  for (var step: i32 = 0; step < 4; step++) {
    let z = decodeDepthFast(depthTex, pUv, dp).x;
    let invZ = 1.0 / max(z, 0.2);
    let relInv = clamp((invZ - invPivot) / invSpan, -0.65, 1.0);
    let radial = (base.xy - dp.subjectUv) * (dolly * 0.35);
    var shift = (panVec * 0.72 - radial) * relInv * strength;
    let lenS = length(shift);
    if (lenS > maxShift) {
      shift = shift * (maxShift / lenS);
    }
    pUv = clamp(base.xy + shift, vec2f(0.001), vec2f(0.999));
  }
  return vec3f(pUv, base.z);
}

fn particleVis(zScene: f32, planeZ: f32, dp: DepthParams) -> f32 {
  if (dp.hasDepth <= 0.5 || dp.depthStrength <= 0.01) {
    return 1.0;
  }
  let occ = smoothstep(planeZ * 0.68, planeZ * 1.12, zScene);
  return mix(1.0, occ, dp.depthStrength * 0.88);
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
  let streak = invSmooth(0.0, 0.044, dx)
    * smoothstep(yStart, yStart + 0.34, f.y)
    * invSmooth(yStart + 0.50, yStart + 0.60, f.y);
  return streak * (0.45 + 0.55 * rnd);
}

fn particleDots(uv: vec2f, aspect: f32, t: f32, grid: f32, speedY: f32, sway: f32, radius: f32, soft: f32) -> vec2f {
  let p = vec2f(uv.x * aspect, uv.y) * grid;
  var acc = 0.0;
  var tint = 0.0;
  for (var dy: i32 = -1; dy <= 1; dy++) {
    for (var dx: i32 = -1; dx <= 1; dx++) {
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

fn petalLayer(uv: vec2f, aspect: f32, t: f32, grid: f32, speedX: f32, speedY: f32, sizeScale: f32) -> vec4f {
  let wind = vec2f(-t * speedX, -t * speedY);
  let p = vec2f(uv.x * aspect + uv.y * 0.28, uv.y) * grid + wind;
  let baseCell = floor(p);
  var out = vec4f(0.0);
  for (var dy: i32 = -1; dy <= 1; dy++) {
    for (var dx: i32 = -1; dx <= 1; dx++) {
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

fn fireflyLayer(uv: vec2f, aspect: f32, t: f32, grid: f32, radius: f32) -> vec4f {
  let p = vec2f(uv.x * aspect, uv.y) * grid + vec2f(0.0, t * 0.08);
  let baseCell = floor(p);
  var out = vec4f(0.0);
  for (var dy: i32 = -1; dy <= 1; dy++) {
    for (var dx: i32 = -1; dx <= 1; dx++) {
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

fn computeDistortion(sceneUv: vec2f, aspect: f32, t: f32, cueT: f32, atmo: u32, light: u32, shake: f32, shotSeed: f32) -> vec3f {
  var offset = vec2f(0.0);
  var ca = shake * 0.0065;
  let auv = vec2f(sceneUv.x * aspect, sceneUv.y);

  if (atmo == 1u) {
    let r1 = rainLayer(sceneUv, t, vec2f(22.0 * aspect, 6.5), 4.8, 0.26);
    let r2 = rainLayer(sceneUv + vec2f(0.37, 0.19), t, vec2f(36.0 * aspect, 11.0), 3.5, 0.24);
    let ripple = (r1 * 0.7 + r2 * 0.4);
    offset += vec2f(-0.0025, 0.004) * ripple;
  } else if (atmo == 3u || atmo == 7u) {
    let n = fbm(auv * 3.5 + vec2f(t * 0.14, -t * 0.18)) - 0.5;
    offset += vec2f(n * 0.0018, n * 0.0012);
  } else if (atmo == 9u) {
    let wX = sin(sceneUv.y * 9.5 + t * 1.35) * 0.0032 + (fbm(auv * 2.8 + vec2f(t * 0.22, -t * 0.16)) - 0.5) * 0.0025;
    let wY = cos(sceneUv.x * 8.0 + t * 1.1) * 0.0024;
    offset += vec2f(wX, wY);
    ca = max(ca, 0.0014);
  }

  if (light == 1u) {
    let env = lightningEnvelope(t, shotSeed);
    ca = max(ca, env.x * 0.011);
  } else if (light == 2u) {
    let a = max(0.0, 1.0 - cueT / 0.45);
    ca = max(ca, a * 0.008);
  } else if (light == 5u) {
    let heat = (fbm(auv * 5.0 + vec2f(sin(t * 1.3) * 0.2, t * 0.85)) - 0.5) * smoothstep(0.25, 1.0, sceneUv.y);
    offset += vec2f(heat * 0.0022, heat * 0.0016);
  } else if (light == 8u) {
    let fromCenter = sceneUv - vec2f(0.5);
    let r = length(fromCenter);
    let wave = sin(t * 0.85 + r * 6.0) * smoothstep(0.18, 0.75, r) * 0.003;
    offset += fromCenter * wave;
    ca = max(ca, smoothstep(0.22, 0.8, r) * 0.0028);
  } else if (light == 9u) {
    let row = floor(sceneUv.y * 240.0);
    let glitch = (hash21(vec2f(row, floor(t * 18.0))) - 0.5) * 0.0012;
    offset.x += glitch;
    ca = max(ca, 0.0015);
  }

  return vec3f(offset, ca);
}

// Sample a still texture with 2.5D parallax, optical thin-lens rack-focus bokeh, UV refraction, CA, bloom, and grading.
fn sampleScene(
  tex: texture_2d<f32>,
  depthTex: texture_2d<f32>,
  texSize: vec2f,
  uv: vec2f,
  res: vec2f,
  cam: vec4f,
  angleRad: f32,
  camBlur: f32,
  rackActive: f32,
  t: f32,
  cueT: f32,
  atmo: u32,
  light: u32,
  shotSeed: f32,
  dp: DepthParams,
  applyGrading: bool
) -> vec4f {
  let aspect = res.x / max(1.0, res.y);
  let camUv = cameraUv(uv, cam, angleRad, aspect);
  let dist = computeDistortion(camUv, aspect, t, cueT, atmo, light, cam.w, shotSeed);
  let warpedUv = camUv + dist.xy;
  let mapped = parallaxContainUv(warpedUv, res, texSize, depthTex, dp, cam);
  let tuv = mapped.xy;
  let inside = mapped.z;

  let radial = (uv - vec2f(0.5)) * dist.z;
  var rgb = textureSampleLevel(tex, texSampler, tuv, 0.0).rgb;
  if (dist.z > 0.0001) {
    rgb.r = textureSampleLevel(tex, texSampler, clamp(tuv + radial, vec2f(0.0), vec2f(1.0)), 0.0).r;
    rgb.b = textureSampleLevel(tex, texSampler, clamp(tuv - radial, vec2f(0.0), vec2f(1.0)), 0.0).b;
  }

  let hasDepth = dp.hasDepth > 0.5 && dp.depthStrength > 0.01;
  let opticalDof = hasDepth && (camBlur > 0.001 || rackActive > 0.5);
  let needsBloom = applyGrading && ((light == 1u && dist.z > 0.0) || (light == 2u && cueT < 0.45) ||
    light == 3u || light == 5u || light == 8u || light == 10u || atmo == 6u || atmo == 9u);

  if ((needsBloom || camBlur > 0.001 || opticalDof) && inside > 0.0) {
    let texel = vec2f(1.0) / max(vec2f(64.0), res * 0.45);
    var vogel = array<vec2f, 12>(
      vec2f( 0.204,  0.000), vec2f(-0.261,  0.239), vec2f( 0.039, -0.455),
      vec2f( 0.340,  0.427), vec2f(-0.618, -0.105), vec2f( 0.564, -0.391),
      vec2f(-0.181,  0.719), vec2f(-0.426, -0.663), vec2f( 0.831,  0.145),
      vec2f(-0.768,  0.449), vec2f( 0.299, -0.884), vec2f( 0.483,  0.854)
    );
    var blur = vec3f(0.0);
    if (opticalDof) {
      let zCenter = decodeDepthFast(depthTex, tuv, dp).x;
      let blurAmt = clamp(camBlur, 0.0, 1.0);
      let zFocus = mix(dp.subjectZ, dp.rackStartZ, blurAmt);
      let invC = 1.0 / max(zCenter, 0.1);
      let invF = 1.0 / max(zFocus, 0.1);
      let invSpread = max(0.08, (dp.invMax - dp.invMin) * 0.42);
      let coc = clamp(abs(invC - invF) / invSpread, 0.0, 1.0) * dp.depthStrength;
      let dofRadius = texel * coc * mix(11.0, 19.0, blurAmt);
      let centerFocused = coc < 0.18;
      var dofSum = vec3f(0.0);
      var dofWeight = 0.0001;
      for (var i: i32 = 0; i < 12; i++) {
        let sUv = clamp(tuv + vogel[i] * dofRadius, vec2f(0.0), vec2f(1.0));
        let tapRgb = textureSampleLevel(tex, texSampler, sUv, 0.0).rgb;
        let tapZ = decodeDepthFast(depthTex, sUv, dp).x;
        let tapCoc = clamp(abs(1.0 / max(tapZ, 0.1) - invF) / invSpread, 0.0, 1.0);
        let w = select(0.25 + 0.75 * tapCoc, 0.08, centerFocused && tapZ > zCenter * 1.15);
        dofSum += tapRgb * w;
        dofWeight += w;
      }
      blur = dofSum / dofWeight;
      let dofMix = smoothstep(0.14, 0.50, coc) * 0.92;
      rgb = mix(rgb, blur, dofMix);
    } else {
      let maxRadius = texel * (6.5 + camBlur * 16.5);
      for (var i: i32 = 0; i < 12; i++) {
        blur += textureSampleLevel(tex, texSampler, clamp(tuv + vogel[i] * maxRadius, vec2f(0.0), vec2f(1.0)), 0.0).rgb;
      }
      blur *= (1.0 / 12.0);
      if (camBlur > 0.001) {
        rgb = mix(rgb, blur, clamp(camBlur * 0.90, 0.0, 0.90));
      }
    }

    if (applyGrading) {
      let luma = dot(blur, vec3f(0.299, 0.587, 0.114));
      let highlight = blur * smoothstep(0.42, 0.92, luma);
      if (light == 8u) {
        let pulse = 0.34 + sin(t * 0.55) * 0.06;
        rgb = mix(rgb, blur, 0.14);
        rgb += highlight * vec3f(1.05, 0.88, 1.02) * pulse;
      } else if (light == 3u) {
        let pulse = 0.24 + sin(t * 0.8) * 0.06;
        rgb += highlight * vec3f(1.08, 0.82, 0.52) * pulse;
      } else if (light == 5u) {
        let wave = 0.5 + 0.3 * sin(t * 7.3) + 0.2 * sin(t * 13.1);
        rgb += highlight * vec3f(1.1, 0.72, 0.35) * (0.12 + wave * 0.22);
      } else if (light == 1u) {
        let env = lightningEnvelope(t, shotSeed);
        rgb += highlight * vec3f(0.85, 0.94, 1.15) * (env.x * 1.4);
      } else if (light == 2u) {
        let a = max(0.0, 1.0 - cueT / 0.45);
        rgb += highlight * vec3f(1.0, 0.98, 0.95) * (a * 0.36);
      } else if (light == 10u) {
        rgb += highlight * vec3f(1.05, 0.95, 0.78) * 0.22;
      } else if (atmo == 6u) {
        rgb += highlight * vec3f(1.05, 0.92, 0.68) * 0.18;
      } else if (atmo == 9u) {
        rgb += highlight * vec3f(0.45, 0.92, 1.05) * 0.16;
      }
    }
  }

  if (inside > 0.0 && applyGrading) {
    if (light == 4u) {
      let luma = dot(rgb, vec3f(0.299, 0.587, 0.114));
      let cool = mix(vec3f(0.14, 0.28, 0.52), vec3f(0.74, 0.86, 1.02), luma);
      let shadowWeight = 0.22 - 0.08 * smoothstep(0.55, 0.95, luma);
      rgb = mix(rgb, cool, shadowWeight);
    } else if (light == 7u) {
      let luma = dot(rgb, vec3f(0.299, 0.587, 0.114));
      let midWeight = 0.35 + 2.6 * luma * (1.0 - luma);
      let frame = floor(t * 24.0);
      let g = hash21(floor(uv * res * 0.5) + vec2f(frame * 13.7, frame * 7.3)) - 0.5;
      rgb = clamp(rgb + vec3f(g * 0.11 * midWeight), vec3f(0.0), vec3f(1.0));
    }
  }

  return vec4f(clamp(rgb, vec3f(0.0), vec3f(1.0)) * inside, inside);
}

// Render one shot's procedural atmosphere + lighting layer with optional Depth Pro 2 spatial/volumetric/normal enhancement.
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
  hasBaseTex: bool,
  depthTex: texture_2d<f32>,
  texSize: vec2f,
  dp: DepthParams
) -> vec4f {
  let aspect = res.x / max(1.0, res.y);
  let worldUv = cameraUv(screenUv, cam, angleRad, aspect);
  let wAuv = vec2f(worldUv.x * aspect, worldUv.y);
  var out = vec4f(0.0);

  let ds = select(0.0, dp.depthStrength, dp.hasDepth > 0.5);
  var zScene = dp.subjectZ;
  var skyScene = 0.0;
  var dTuv = vec2f(0.5);
  var dInside = 0.0;
  if (ds > 0.01 && (atmo > 0u || (light > 0u && light != 6u && light != 9u))) {
    let dMap = parallaxContainUv(worldUv, res, texSize, depthTex, dp, cam);
    dTuv = dMap.xy;
    dInside = dMap.z;
    if (dInside > 0.5) {
      let info = sampleDepthInfo(depthTex, dTuv, dp);
      zScene = info.x;
      skyScene = info.w;
    }
  }
  let useD = ds > 0.01 && dInside > 0.5;
  let zNear = max(0.25, dp.zP10 * 0.72);
  let zMid = max(zNear + 0.35, dp.subjectZ * 1.12);
  let zFar = max(zMid + 0.6, dp.zP90 * 0.90);

  // --- Atmosphere Pass ---
  if (atmo == 1u) { // Z-stratified 3-plane rain
    let r1 = rainLayer(worldUv, t, vec2f(22.0 * aspect, 6.5), 4.8, 0.26) * select(1.0, particleVis(zScene, zNear, dp), useD);
    let r2 = rainLayer(worldUv + vec2f(0.37, 0.19), t, vec2f(36.0 * aspect, 11.0), 3.5, 0.24) * select(1.0, particleVis(zScene, zMid, dp), useD);
    let r3 = rainLayer(worldUv + vec2f(0.71, 0.43), t, vec2f(62.0 * aspect, 19.0), 2.3, 0.22) * select(1.0, particleVis(zScene, zFar, dp), useD);
    let rainA = clamp(r1 * 0.34 + r2 * 0.24 + r3 * 0.15, 0.0, 0.62);
    out = blendOver(out, vec3f(0.77, 0.87, 0.97), rainA);
    let mistNoise = 0.85 + 0.3 * noise2(vec2f(wAuv.x * 3.0 + t * 0.25, worldUv.y * 4.0));
    let groundFog = select(1.0, 0.35 + 0.65 * smoothstep(zNear, zFar, zScene), useD);
    let mist = smoothstep(0.66, 1.0, screenUv.y) * 0.13 * mistNoise * groundFog;
    out = blendOver(out, vec3f(0.72, 0.80, 0.88), mist);
  } else if (atmo == 2u) { // Z-stratified 3-plane snow
    let s0 = particleDots(worldUv + vec2f(0.19, 0.71), aspect, t, 6.5, 0.78, 0.75, 0.21, 2.7);
    let s1 = particleDots(worldUv, aspect, t, 11.0, 0.55, 0.55, 0.16, 2.2);
    let s2 = particleDots(worldUv + vec2f(0.41, 0.23), aspect, t, 18.0, 0.35, 0.4, 0.12, 2.0);
    let v0 = select(1.0, particleVis(zScene, zNear, dp), useD);
    let v1 = select(1.0, particleVis(zScene, zMid, dp), useD);
    let v2 = select(1.0, particleVis(zScene, zFar, dp), useD);
    let snowA = clamp(s0.x * 0.42 * v0 + s1.x * 0.70 * v1 + s2.x * 0.42 * v2, 0.0, 0.88);
    out = blendOver(out, vec3f(0.95, 0.98, 1.0), snowA);
  } else if (atmo == 3u) { // Beer-Lambert volumetric fog
    let n1 = fbm(wAuv * 2.1 + vec2f(t * 0.06, sin(t * 0.12) * 0.05));
    let n2 = fbm(wAuv * 3.6 + vec2f(-t * 0.04, 2.4));
    let fogMask = smoothstep(0.20, 0.95, worldUv.y);
    let legacyDensity = smoothstep(0.32, 0.82, n1 * 0.65 + n2 * 0.35) * fogMask * 0.42;
    var density = legacyDensity;
    if (useD) {
      let zStart = max(0.25, dp.subjectZ * 0.55);
      let depthSpan = max(2.0, dp.zP90 - zStart);
      let beer = 1.0 - exp(-2.2 * max(0.0, zScene - zStart) / depthSpan);
      let skyAtten = mix(1.0, 0.28, smoothstep(0.35, 0.9, skyScene));
      let volumetric = smoothstep(0.22, 0.78, n1 * 0.65 + n2 * 0.35) * (0.25 + 0.75 * smoothstep(0.10, 0.95, worldUv.y)) * beer * skyAtten * 0.62;
      density = mix(legacyDensity, volumetric, ds);
    }
    out = blendOver(out, vec3f(0.84, 0.90, 0.93), density);
  } else if (atmo == 4u) { // Z-stratified floating dust
    let d1 = particleDots(worldUv, aspect, t, 9.5, -0.14, 0.65, 0.11, 2.3);
    let d2 = particleDots(worldUv + vec2f(0.29, 0.53), aspect, t, 14.5, -0.10, 0.50, 0.08, 2.1);
    let vis1 = select(1.0, particleVis(zScene, zNear, dp), useD);
    let vis2 = select(1.0, particleVis(zScene, zFar, dp), useD);
    out = blendScreen(out, vec3f(1.0, 0.91, 0.72), clamp(d1.x * 0.42 * vis1 + d2.x * 0.30 * vis2, 0.0, 0.6));
  } else if (atmo == 5u) { // Z-occluded glowing embers
    let e1 = particleDots(worldUv, aspect, t, 10.5, -0.85, 0.75, 0.15, 2.6);
    let e2 = particleDots(worldUv + vec2f(0.31, 0.17), aspect, t, 16.0, -0.62, 0.6, 0.11, 2.4);
    let vis1 = select(1.0, particleVis(zScene, zNear, dp), useD);
    let vis2 = select(1.0, particleVis(zScene, zMid, dp), useD);
    let flicker = 0.72 + 0.28 * sin(t * 9.5 + (e1.y + e2.y) * 12.0);
    let emberA = clamp((e1.x * 0.9 * vis1 + e2.x * 0.6 * vis2) * flicker, 0.0, 1.0);
    let emberRgb = mix(vec3f(1.0, 0.38, 0.10), vec3f(1.0, 0.84, 0.38), clamp(e1.y + e2.y, 0.0, 1.0));
    out = blendScreen(out, emberRgb, emberA);
  } else if (atmo == 6u) { // Depth-occluded ray-marched sunbeams
    let skyOrigin = select(vec2f(0.24, 0.05), dp.skyDir, useD);
    let sunPos = select(vec2f(0.24 * aspect, -0.18), vec2f(skyOrigin.x * aspect, skyOrigin.y - 0.10), useD);
    let delta = wAuv - sunPos;
    let dist = length(delta);
    let angle = atan2(delta.y, delta.x);
    let r1 = fbm(vec2f(angle * 7.5 + sin(t * 0.16) * 0.22, dist * 1.15 - t * 0.04));
    let r2 = fbm(vec2f(angle * 14.5 - t * 0.06, dist * 1.9 - t * 0.05));
    let cone = smoothstep(0.08, 0.52, max(0.2, sin(angle))) * invSmooth(0.06, 1.75, dist / max(1.0, aspect * 0.82));
    let baseShafts = (smoothstep(0.38, 0.82, r1) * 0.72 + smoothstep(0.45, 0.88, r2) * 0.34) * cone;
    var shafts = baseShafts;
    if (useD) {
      var occAccum = 0.0;
      let startTexUv = containUv(skyOrigin, res, texSize).xy;
      let zFarMid = mix(dp.zP50, dp.zP90, 0.5);
      let skyThresh = max(zFarMid * 0.95, dp.subjectZ * 1.25);
      for (var s: i32 = 1; s <= 12; s++) {
        let frac = f32(s) / 12.0;
        let rayUv = mix(dTuv, startTexUv, frac);
        let rayTap = decodeDepthFast(depthTex, rayUv, dp);
        let rayZ = rayTap.x;
        let raySky = rayTap.y;
        let passOpen = smoothstep(skyThresh * 0.65, skyThresh * 1.15, rayZ);
        occAccum += max(raySky, passOpen);
      }
      let transmittance = occAccum / 12.0;
      let depthMix = smoothstep(dp.zP10 * 0.8, zFarMid, zScene) * (0.35 + 0.65 * transmittance);
      shafts = baseShafts * mix(1.0, depthMix, ds * 0.85);
    }
    let sourceGlow = invSmooth(0.0, 0.95 * aspect, dist) * 0.14;
    out = blendScreen(out, vec3f(1.0, 0.92, 0.74), clamp(shafts * 0.36 + sourceGlow, 0.0, 0.56));
  } else if (atmo == 7u) { // Depth-anchored plumes of smoke
    let s1 = fbm(wAuv * 2.6 + vec2f(sin(t * 0.25) * 0.15, t * 0.18));
    let s2 = fbm(wAuv * 4.8 + vec2f(-t * 0.08, t * 0.26));
    let legacyDensity = smoothstep(0.38, 0.86, s1 * 0.6 + s2 * 0.4) * smoothstep(0.0, 0.85, worldUv.y) * 0.38;
    var density = legacyDensity;
    if (useD) {
      let zStart = max(0.3, dp.subjectZ * 0.65);
      let beer = 1.0 - exp(-1.8 * max(0.0, zScene - zStart) / max(2.5, dp.zP90 - zStart));
      let midPlume = exp(-abs(zScene - dp.subjectZ * 1.35) / max(1.5, dp.subjectZ)) * 0.35;
      let vol = smoothstep(0.30, 0.82, s1 * 0.6 + s2 * 0.4) * smoothstep(0.0, 0.88, worldUv.y) * clamp(beer + midPlume, 0.0, 1.0) * 0.50;
      density = mix(legacyDensity, vol, ds);
    }
    out = blendOver(out, vec3f(0.46, 0.49, 0.54), density);
  } else if (atmo == 8u) { // Z-stratified 3D tumbling petals
    let pMid = petalLayer(worldUv, aspect, t, 8.2, 0.34, 0.42, 1.0);
    let pNear = petalLayer(worldUv + vec2f(0.37, 0.21), aspect, t, 5.4, 0.46, 0.56, 1.18);
    let visMid = select(1.0, particleVis(zScene, zMid, dp), useD);
    let visNear = select(1.0, particleVis(zScene, zNear, dp), useD);
    out = blendOver(out, pMid.rgb, pMid.a * visMid);
    out = blendOver(out, pNear.rgb, pNear.a * visNear);
  } else if (atmo == 9u) { // Beer-Lambert underwater attenuation + upward-normal caustics + Z-occluded bubbles
    let c1 = fbm(wAuv * 3.8 + vec2f(t * 0.16, -t * 0.11));
    let c2 = fbm(wAuv * 4.2 + vec2f(-t * 0.14, t * 0.18));
    let causticRidge = pow(clamp(1.0 - abs(c1 - c2) * 2.6, 0.0, 1.0), 3.0);
    var causticWeight = invSmooth(0.05, 0.92, worldUv.y) * 0.26;
    if (useD) {
      let n = reconstructPosNormal(depthTex, dTuv, texSize, dp);
      let upMask = mix(0.35, 1.0, clamp(-n.y, 0.0, 1.0));
      let shallowFade = exp(-max(0.0, zScene - dp.zP10) / max(4.0, dp.zP90 * 0.9));
      causticWeight *= mix(1.0, upMask * (0.4 + 0.8 * shallowFade), ds);
      let absT = 1.0 - exp(-0.95 * max(0.0, zScene - dp.zP10) / max(3.0, mix(dp.zP50, dp.zP90, 0.5)));
      out = blendOver(out, vec3f(0.04, 0.32, 0.46), absT * ds * 0.44);
    }
    out = blendScreen(out, vec3f(0.42, 0.90, 1.0), causticRidge * causticWeight);
    let b1 = particleDots(worldUv, aspect, t, 10.0, -0.58, 0.72, 0.14, 1.85);
    let b2 = particleDots(worldUv + vec2f(0.33, 0.27), aspect, t, 16.5, -0.42, 0.55, 0.10, 1.75);
    let vis1 = select(1.0, particleVis(zScene, zNear, dp), useD);
    let vis2 = select(1.0, particleVis(zScene, zMid, dp), useD);
    let bubbleA = clamp(b1.x * 0.68 * vis1 + b2.x * 0.45 * vis2, 0.0, 0.82);
    out = blendScreen(out, vec3f(0.80, 0.96, 1.0), bubbleA);
  } else if (atmo == 10u) { // Z-stratified fireflies
    let f1 = fireflyLayer(worldUv, aspect, t, 6.2, 0.16);
    let f2 = fireflyLayer(worldUv + vec2f(0.43, 0.29), aspect, t * 0.85, 9.5, 0.12);
    let vis1 = select(1.0, particleVis(zScene, zNear, dp), useD);
    let vis2 = select(1.0, particleVis(zScene, zMid, dp), useD);
    out = blendScreen(out, f2.rgb, f2.a * vis2);
    out = blendScreen(out, f1.rgb, f1.a * vis1);
  }

  // --- Lighting & Framing Pass ---
  if (light == 1u) { // Sky-masked storm lightning + cool foreground rim
    let env = lightningEnvelope(t, shotSeed);
    let a = env.x;
    let boltX = env.y;
    let skyAnchor = select(vec2f(boltX, 0.16), vec2f(mix(boltX, dp.skyDir.x, 0.45), dp.skyDir.y), useD);
    let skyDist = length((worldUv - skyAnchor) * vec2f(aspect * 0.75, 1.0));
    let cloudMod = 0.75 + 0.35 * fbm(wAuv * 3.2 + vec2f(shotSeed * 1.7, 0.0));
    var lightA = clamp(a + invSmooth(0.0, 1.15, skyDist) * a * 0.82 * cloudMod, 0.0, 0.58);
    if (useD) {
      let n = reconstructPosNormal(depthTex, dTuv, texSize, dp);
      let topRim = clamp(-n.y, 0.0, 1.0) * 0.6 + 0.25 * (1.0 - abs(n.z));
      let skyBoost = mix(0.45, 1.25, max(skyScene, smoothstep(dp.subjectZ, dp.zP90, zScene)));
      lightA = clamp(lightA * mix(1.0, skyBoost + topRim * 0.4, ds), 0.0, 0.72);
    }
    out = blendScreen(out, vec3f(0.88, 0.93, 1.0), lightA);
  } else if (light == 2u) { // Flash (distance inverse-square falloff when depth present)
    let baseA = max(0.0, 1.0 - cueT / 0.45) * 0.35;
    var falloff = 1.0;
    if (useD) {
      let refZ = max(0.5, dp.subjectZ);
      let ratio = refZ / max(0.4, zScene);
      falloff = mix(1.0, clamp(0.3 + 0.85 * ratio * ratio, 0.25, 1.35), ds);
    }
    out = blendScreen(out, vec3f(0.94, 0.96, 1.0), clamp(baseA * falloff, 0.0, 0.55));
  } else if (light == 3u || light == 5u) { // warm_glow / flicker: 3D Lambertian + 6-step contact shadows around warm_anchor
    let isFlicker = light == 5u;
    let wave = 0.5 + 0.3 * sin(t * 7.3) + 0.2 * sin(t * 13.1);
    let baseAmp = select(0.11 + sin(t * 0.8) * 0.035, 0.04 + wave * 0.08, isFlicker);
    let legacyCenter = select(
      vec2f(0.5 + sin(t * 0.22) * 0.06, 0.42 + cos(t * 0.18) * 0.04),
      vec2f(0.5 + sin(t * 5.1) * 0.02, 0.62),
      isFlicker
    );
    var center = legacyCenter;
    var contrib = invSmooth(0.0, select(0.95, 0.85, isFlicker), length((worldUv - center) * vec2f(aspect, 1.0))) * baseAmp * select(1.45, 1.6, isFlicker);
    if (useD) {
      center = mix(legacyCenter, dp.warmAnchor.xy, dp.warmAnchor.w * ds);
      let pos = unprojectPos(dTuv, zScene, texSize, dp.fPxNorm);
      let normal = reconstructPosNormal(depthTex, dTuv, texSize, dp);
      let lightTexUv = containUv(center, res, texSize).xy;
      let lightPos = unprojectPos(lightTexUv, dp.warmAnchor.z, texSize, dp.fPxNorm);
      let toLight = lightPos - pos;
      let dist3D = max(0.25, length(toLight));
      let lDir = toLight / dist3D;
      let lambert = 0.32 + 0.68 * max(0.0, dot(normal, lDir));
      let distFalloff = exp(-dist3D / max(2.0, dp.zP50 * 0.85));
      var occ = 0.0;
      for (var s: i32 = 1; s <= 6; s++) {
        let frac = f32(s) / 7.0;
        let sampleUv = mix(dTuv, lightTexUv, frac);
        let zRay = mix(zScene, dp.warmAnchor.z, frac);
        let zSample = decodeDepthFast(depthTex, sampleUv, dp).x;
        let blocked = smoothstep(0.15, 0.55, (zRay - zSample) / max(0.5, zSample));
        occ = max(occ, blocked * 0.72);
      }
      let relit = lambert * distFalloff * (1.0 - occ) * baseAmp * (1.4 + 0.9 * dp.warmAnchor.w);
      let halo = invSmooth(0.0, 0.88, length((worldUv - center) * vec2f(aspect, 1.0))) * baseAmp * 1.15;
      contrib = mix(contrib, clamp(halo * 0.45 + relit * 1.15, 0.0, 0.48), ds);
    }
    let tint = select(vec3f(1.0, 0.72, 0.38), vec3f(1.0, 0.64, 0.28), isFlicker);
    out = blendScreen(out, tint, contrib);
  } else if (light == 4u) { // cold_tint (preserves full wash when !hasBaseTex for masked layers, adds depth cool wash)
    let pulse = 0.09 + sin(t * 0.5) * 0.02;
    let edge = smoothstep(0.15, 1.05, length(screenUv - vec2f(0.5, 0.4)));
    var strength = select(mix(0.35, 1.0, edge) * 1.35, edge * 0.65, hasBaseTex);
    if (useD) {
      let farGain = mix(0.85, 1.45, smoothstep(dp.subjectZ * 0.8, dp.zP90, zScene));
      strength *= mix(1.0, farGain, ds * 0.55);
    }
    out = blendOver(out, vec3f(0.24, 0.44, 0.78), strength * pulse);
  } else if (light == 6u) { // vignette (screen-space)
    let d = length((screenUv - 0.5) * vec2f(1.15, 1.0));
    let vig = smoothstep(0.28, 0.82, d) * 0.56;
    out = blendOver(out, vec3f(0.0, 0.0, 0.0), vig);
  } else if (light == 7u && !hasBaseTex) { // film_grain
    let frame = floor(t * 24.0);
    let g = hash21(floor(screenUv * res * 0.5) + vec2f(frame * 13.7, frame * 7.3)) - 0.5;
    let grainRgb = select(vec3f(0.0), vec3f(1.0), g > 0.0);
    out = blendOver(out, grainRgb, abs(g) * 0.14);
  } else if (light == 8u) { // dream_bloom (depth-graduated dreamy halation behind foreground)
    let pulse = 0.14 + sin(t * 0.55) * 0.03;
    let c1 = vec2f(0.5 + sin(t * 0.25) * 0.05, 0.44 + cos(t * 0.2) * 0.04);
    let c2 = vec2f(0.46 - cos(t * 0.3) * 0.06, 0.52 + sin(t * 0.22) * 0.05);
    let halo1 = invSmooth(0.0, 0.92, length((worldUv - c1) * vec2f(aspect * 0.8, 1.0)));
    let halo2 = invSmooth(0.0, 0.78, length((worldUv - c2) * vec2f(aspect * 0.8, 1.0))) * 0.55;
    let rim = smoothstep(0.35, 0.92, length(screenUv - vec2f(0.5))) * 0.45;
    var dream = clamp((halo1 + halo2 + rim) * pulse, 0.0, 0.38);
    if (useD) {
      let bgBloom = mix(0.65, 1.35, smoothstep(dp.subjectZ * 0.85, dp.zP90, zScene));
      dream = clamp(dream * mix(1.0, bgBloom, ds * 0.75), 0.0, 0.44);
    }
    out = blendScreen(out, vec3f(0.98, 0.87, 0.96), dream);
  } else if (light == 9u) { // scanlines (screen-space)
    let line = 0.5 - 0.5 * cos(screenUv.y * res.y * 3.14159);
    let bar = invSmooth(0.0, 0.08, abs(fract(screenUv.y - t * 0.08) - 0.5));
    let crtVig = smoothstep(0.42, 0.92, length((screenUv - 0.5) * vec2f(1.1, 1.05))) * 0.22;
    out = blendOver(out, vec3f(0.03, 0.07, 0.11), clamp(line * 0.16 + crtVig, 0.0, 0.45));
    out = blendScreen(out, vec3f(0.35, 0.85, 0.95), bar * 0.07);
  } else if (light == 10u) { // spotlight anchored near subjectUv with Z-pool falloff
    let baseX = 0.5 + sin(t * 0.28) * 0.035;
    let swayX = select(baseX, mix(baseX, dp.subjectUv.x + sin(t * 0.28) * 0.025, 0.75), useD);
    let targetY = select(0.54, dp.subjectUv.y + 0.06, useD);
    let coneHalfWidth = mix(0.11, 0.44, clamp(worldUv.y, 0.0, 1.0));
    let dx = abs(worldUv.x - swayX) / max(0.05, coneHalfWidth);
    let beam = invSmooth(0.35, 1.18, dx) * invSmooth(0.0, 1.15, worldUv.y * 0.72);
    let stagePool = invSmooth(0.0, 0.52, length((worldUv - vec2f(swayX, targetY)) * vec2f(aspect * 0.75, 1.15)));
    var lit = clamp(max(beam, stagePool), 0.0, 1.0);
    if (useD) {
      let zPool = exp(-abs(zScene - dp.subjectZ) / max(1.6, dp.subjectZ * 1.15));
      lit *= mix(1.0, 0.45 + 0.55 * zPool, ds * 0.8);
    }
    let shade = (1.0 - lit) * 0.56;
    out = blendOver(out, vec3f(0.02, 0.03, 0.05), shade);
    out = blendScreen(out, vec3f(1.0, 0.94, 0.80), lit * 0.14);
  }

  return vec4f(out.rgb, clamp(out.a, 0.0, 1.0));
}

fn compositeOverBase(base: vec4f, layer: vec4f) -> vec4f {
  return vec4f(layer.rgb + base.rgb * (1.0 - layer.a), 1.0);
}

// Entry 1: Single-pass standard shots and cross-dissolves.
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
  let rackActive = u.cueParams.z;
  let backRackActive = u.cueParams.w;
  let atmo = u.effectIds.x;
  let light = u.effectIds.y;
  let backAtmo = u.effectIds.z;
  let backLight = u.effectIds.w;

  let frontDp = frontDepthParams();
  let hasBaseTex = texMode >= 0.5;

  if (texMode >= 1.5) {
    let backDp = backDepthParams();
    let backLayer = renderLayer(
      uv, res, backT, backCueT, backAtmo, backLight, u.backCam, backCamAngle,
      backShotSeed, true, backDepthTex, u.texSizes.xy, backDp
    );
    let frontLayer = renderLayer(
      uv, res, t, cueT, atmo, light, u.cam, camAngle,
      shotSeed, true, frontDepthTex, u.texSizes.zw, frontDp
    );
    let backBase = sampleScene(
      backTex, backDepthTex, u.texSizes.xy, uv, res, u.backCam, backCamAngle, backCamBlur, backRackActive,
      backT, backCueT, backAtmo, backLight, backShotSeed, backDp, true
    );
    let frontBase = sampleScene(
      frontTex, frontDepthTex, u.texSizes.zw, uv, res, u.cam, camAngle, camBlur, rackActive,
      t, cueT, atmo, light, shotSeed, frontDp, true
    );
    return mix(compositeOverBase(backBase, backLayer), compositeOverBase(frontBase, frontLayer), fadeMix);
  }

  let frontLayer = renderLayer(
    uv, res, t, cueT, atmo, light, u.cam, camAngle,
    shotSeed, hasBaseTex, backDepthTex, u.texSizes.xy, frontDp
  );
  if (hasBaseTex) {
    let base = sampleScene(
      backTex, backDepthTex, u.texSizes.xy, uv, res, u.cam, camAngle, camBlur, rackActive,
      t, cueT, atmo, light, shotSeed, frontDp, true
    );
    return compositeOverBase(base, frontLayer);
  }
  if (fadeMix < 1.0) {
    let backDp = backDepthParams();
    let backLayer = renderLayer(
      uv, res, backT, backCueT, backAtmo, backLight, u.backCam, backCamAngle,
      backShotSeed, false, backDepthTex, u.texSizes.xy, backDp
    );
    return mix(backLayer, frontLayer, fadeMix);
  }

  return frontLayer;
}

// Entry 2 (Unified SAM Layer Pass 1): Base still with 2.5D parallax and optical rack-focus.
@fragment
fn fsBase(in: VSOut) -> @location(0) vec4f {
  let dp = frontDepthParams();
  let res = u.timeParams.xy;
  let t = max(0.0, u.timeParams.z);
  let base = sampleScene(
    backTex, backDepthTex, u.texSizes.xy, in.uv, res, u.cam, u.camExtra.x, u.camExtra.y, u.cueParams.z,
    t, max(0.0, u.cueParams.x), 0u, 0u, u.mixParams.y, dp, false
  );
  return vec4f(base.rgb, 1.0);
}

// Entry 3 (Unified SAM Layer Pass 2): Per-layer masked & depth-occluded effect blended over shot RT.
@fragment
fn fsLayer(in: VSOut) -> @location(0) vec4f {
  let res = u.timeParams.xy;
  let aspect = res.x / max(1.0, res.y);
  let dp = frontDepthParams();
  let camUv = cameraUv(in.uv, u.cam, u.camExtra.x, aspect);
  let mapped = parallaxContainUv(camUv, res, u.texSizes.xy, backDepthTex, dp, u.cam);
  let tuv = mapped.xy;
  let inside = mapped.z;

  let intensity = clamp(u.layerParams.x, 0.0, 1.0);
  let maskMode = u.layerParams.y;
  let maskRefZ = u.layerParams.z;
  let hasMask = u.layerParams.w > 0.5;

  var maskW = 1.0;
  if (maskMode > 0.5) {
    let rawMask = select(0.0, textureSampleLevel(maskTex, texSampler, tuv, 0.0).a * inside, hasMask);
    if (maskMode < 1.5) { // inside
      maskW = rawMask;
    } else if (maskMode < 2.5) { // outside
      maskW = select(1.0, 1.0 - rawMask, hasMask);
    } else { // behind: outside silhouette + depth-occluded behind object's metric depth
      var baseOut = select(1.0, 1.0 - rawMask, hasMask);
      if (dp.hasDepth > 0.5 && dp.depthStrength > 0.01 && maskRefZ > 0.05 && inside > 0.5) {
        let z = decodeDepthFast(backDepthTex, tuv, dp).x;
        let occ = smoothstep(maskRefZ * 0.78, maskRefZ * 1.08, z);
        baseOut *= mix(1.0, occ, dp.depthStrength * 0.90);
      }
      maskW = baseOut;
    }
  }

  let weight = maskW * intensity;
  if (weight <= 0.001) {
    return vec4f(0.0);
  }

  let layer = renderLayer(
    in.uv, res, max(0.0, u.timeParams.z), max(0.0, u.cueParams.x),
    u.effectIds.x, u.effectIds.y, u.cam, u.camExtra.x, u.mixParams.y,
    false, backDepthTex, u.texSizes.xy, dp
  );
  return layer * weight;
}

// Entry 4 (Unified SAM Layer Pass 3): Present single RT or cross-dissolve backRT -> frontRT to swapchain.
@fragment
fn fsComposite(in: VSOut) -> @location(0) vec4f {
  let backSample = textureSampleLevel(backTex, texSampler, in.uv, 0.0);
  if (u.mixParams.w >= 1.5) {
    let frontSample = textureSampleLevel(frontTex, texSampler, in.uv, 0.0);
    return mix(backSample, frontSample, clamp(u.mixParams.x, 0.0, 1.0));
  }
  return backSample;
}
`;

const UNIFORM_STRUCT_BYTES = 304;
const UNIFORM_STRIDE = 512;
const UNIFORM_SLOT_FLOATS = UNIFORM_STRIDE / 4;
const MAX_UNIFORM_SLOTS = 48;

let gpuState = "idle"; // "idle" | "probing" | "ready" | "unavailable"
let gpuDevice = null;
let gpuBindGroupLayout = null;
let gpuPipelineLayout = null;
let gpuShaderModule = null;
let gpuPipeline = null;
let gpuBasePipeline = null;
let gpuLayerPipeline = null;
let gpuCompositePipeline = null;
let gpuUniformBuffer = null;
let gpuSampler = null;
let gpuFormat = "bgra8unorm";
let gpuDummyView = null;
let gpuDummyDepthView = null;
let gpuDummyMaskView = null;
let gpuDummyTextures = [];
let maskScratchA = null;
let maskScratchB = null;
let probePromise = null;
let layerPipelinesPromise = null;
const probeListeners = new Set();

const uniformData = new ArrayBuffer(MAX_UNIFORM_SLOTS * UNIFORM_STRIDE);
const uniformFloats = new Float32Array(uniformData);
const uniformUints = new Uint32Array(uniformData);
const uniformBytes = new Uint8Array(uniformData);

const textureSlots = {
  back: null,
  front: null,
  backDepth: null,
  frontDepth: null,
};

const renderTargets = {
  back: null,
  front: null,
};

const maskTextureCache = new Map();
let cachedSingleBindGroup = null;
let cachedSingleViews = [null, null, null, null];

function resetGpuResources() {
  for (const key of Object.keys(textureSlots)) {
    textureSlots[key]?.texture?.destroy();
    textureSlots[key] = null;
  }
  for (const key of Object.keys(renderTargets)) {
    renderTargets[key]?.texture?.destroy();
    renderTargets[key] = null;
  }
  for (const entry of maskTextureCache.values()) {
    entry.texture?.destroy();
  }
  maskTextureCache.clear();
  cachedSingleBindGroup = null;
  cachedSingleViews = [null, null, null, null];
  for (const tex of gpuDummyTextures) {
    try { tex?.destroy(); } catch { /* ignore */ }
  }
  gpuDummyTextures = [];
  gpuDummyView = null;
  gpuDummyDepthView = null;
  gpuDummyMaskView = null;
}

export function gpuStatus() {
  return gpuState;
}

function createSolidTexture(device, rgba) {
  const texture = device.createTexture({
    size: [1, 1],
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  device.queue.writeTexture(
    { texture },
    new Uint8Array(rgba),
    { bytesPerRow: 4, rowsPerImage: 1 },
    { width: 1, height: 1, depthOrArrayLayers: 1 }
  );
  gpuDummyTextures.push(texture);
  return texture.createView();
}

function notifyProbeListeners() {
  for (const fn of probeListeners) {
    try { fn?.(); } catch { /* ignore */ }
  }
}

async function ensureLayerPipelines() {
  if (gpuBasePipeline && gpuLayerPipeline && gpuCompositePipeline) return true;
  if (!gpuDevice || !gpuShaderModule || !gpuPipelineLayout || gpuState !== "ready") return false;
  if (!layerPipelinesPromise) {
    const device = gpuDevice;
    const module = gpuShaderModule;
    const layout = gpuPipelineLayout;
    layerPipelinesPromise = Promise.all([
      device.createRenderPipelineAsync({
        layout,
        vertex: { module, entryPoint: "vs" },
        fragment: { module, entryPoint: "fsBase", targets: [{ format: "rgba8unorm" }] },
        primitive: { topology: "triangle-list" },
      }),
      device.createRenderPipelineAsync({
        layout,
        vertex: { module, entryPoint: "vs" },
        fragment: {
          module,
          entryPoint: "fsLayer",
          targets: [{
            format: "rgba8unorm",
            blend: {
              color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
              alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
            },
          }],
        },
        primitive: { topology: "triangle-list" },
      }),
      device.createRenderPipelineAsync({
        layout,
        vertex: { module, entryPoint: "vs" },
        fragment: { module, entryPoint: "fsComposite", targets: [{ format: gpuFormat }] },
        primitive: { topology: "triangle-list" },
      }),
    ]).then(([basePipeline, layerPipeline, compositePipeline]) => {
      if (gpuDevice !== device || gpuState !== "ready") return false;
      gpuBasePipeline = basePipeline;
      gpuLayerPipeline = layerPipeline;
      gpuCompositePipeline = compositePipeline;
      notifyProbeListeners();
      return true;
    }).catch(() => {
      layerPipelinesPromise = null;
      return false;
    });
  }
  return layerPipelinesPromise;
}

export async function probeWebGPU(onChange, { layers = false } = {}) {
  if (onChange) probeListeners.add(onChange);
  if (gpuState === "unavailable") return false;
  if (gpuState === "ready") {
    return layers ? ensureLayerPipelines() : true;
  }
  if (!probePromise) {
    const gpu = globalThis.navigator?.gpu;
    if (!gpu?.requestAdapter) {
      gpuState = "unavailable";
      return false;
    }
    gpuState = "probing";
    probePromise = (async () => {
      try {
        const adapter = await gpu.requestAdapter();
        if (!adapter) { gpuState = "unavailable"; return false; }
        const device = await adapter.requestDevice();
        gpuFormat = gpu.getPreferredCanvasFormat?.() || "bgra8unorm";
        const module = device.createShaderModule({ code: WGSL });

        const bindGroupLayout = device.createBindGroupLayout({
          entries: [
            {
              binding: 0,
              visibility: GPUShaderStage.FRAGMENT,
              buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: UNIFORM_STRUCT_BYTES },
            },
            { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
            { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
            { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
            { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
            { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
            { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
          ],
        });
        const layout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

        const pipeline = await device.createRenderPipelineAsync({
          layout,
          vertex: { module, entryPoint: "vs" },
          fragment: { module, entryPoint: "fs", targets: [{ format: gpuFormat }] },
          primitive: { topology: "triangle-list" },
        });

        const uniformBuffer = device.createBuffer({
          size: MAX_UNIFORM_SLOTS * UNIFORM_STRIDE,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const sampler = device.createSampler({
          magFilter: "linear",
          minFilter: "linear",
          addressModeU: "clamp-to-edge",
          addressModeV: "clamp-to-edge",
        });

        const dummyView = createSolidTexture(device, [0, 0, 0, 0]);
        const dummyDepthView = createSolidTexture(device, [0, 0, 0, 255]);
        const dummyMaskView = createSolidTexture(device, [255, 255, 255, 255]);

        device.lost.then(() => {
          resetGpuResources();
          try { gpuUniformBuffer?.destroy?.(); } catch { /* ignore */ }
          gpuDevice = null;
          gpuBindGroupLayout = null;
          gpuPipelineLayout = null;
          gpuShaderModule = null;
          gpuPipeline = null;
          gpuBasePipeline = null;
          gpuLayerPipeline = null;
          gpuCompositePipeline = null;
          layerPipelinesPromise = null;
          gpuUniformBuffer = null;
          gpuSampler = null;
          configuredCanvas = null;
          gpuState = "unavailable";
          notifyProbeListeners();
        });

        gpuDevice = device;
        gpuBindGroupLayout = bindGroupLayout;
        gpuPipelineLayout = layout;
        gpuShaderModule = module;
        gpuPipeline = pipeline;
        gpuUniformBuffer = uniformBuffer;
        gpuSampler = sampler;
        gpuDummyView = dummyView;
        gpuDummyDepthView = dummyDepthView;
        gpuDummyMaskView = dummyMaskView;
        gpuState = "ready";
        notifyProbeListeners();
        return true;
      } catch {
        gpuState = "unavailable";
        return false;
      }
    })();
  }
  const ok = await probePromise;
  return ok && layers ? ensureLayerPipelines() : ok;
}

function readyStill(img) {
  return img?.complete && img.naturalWidth > 0 && img.naturalHeight > 0 && (img.currentSrc || img.src);
}

function syncImageSlot(name, otherName, img, src) {
  if (textureSlots[name]?.src !== src && textureSlots[otherName]?.src === src) {
    [textureSlots[name], textureSlots[otherName]] = [textureSlots[otherName], textureSlots[name]];
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
    slot = textureSlots[name] = { texture, view: texture.createView(), width, height, src: null };
  }
  slot.src = null;
  gpuDevice.queue.copyExternalImageToTexture({ source: img }, { texture: slot.texture }, [width, height]);
  slot.src = src;
  return slot;
}

function syncStillTexture(name, img, src) {
  return syncImageSlot(name, name === "back" ? "front" : "back", img, src);
}

function syncDepthTexture(name, img, src) {
  return syncImageSlot(name, name === "backDepth" ? "frontDepth" : "backDepth", img, src);
}

function syncRenderTarget(name, width, height) {
  let rt = renderTargets[name];
  if (rt && rt.width === width && rt.height === height) return rt;
  rt?.texture.destroy();
  const texture = gpuDevice.createTexture({
    size: [width, height],
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  rt = renderTargets[name] = { texture, view: texture.createView(), width, height };
  return rt;
}

function syncMaskTexture(hostDoc, maskImages, feather, maskUrls) {
  const readyMasks = (maskImages || []).filter((img) => readyStill(img));
  if (!readyMasks.length) return null;
  const first = readyMasks[0];
  const mw = Math.min(1280, first.naturalWidth);
  const mh = Math.min(1280, first.naturalHeight);
  const fVal = feather ?? 1;
  const urlKey = maskUrls?.length ? maskUrls.join("|") : readyMasks.map((m) => m.currentSrc || m.src).join("|");
  const key = `${urlKey}@${fVal}@${mw}x${mh}`;
  const cached = maskTextureCache.get(key);
  if (cached) {
    maskTextureCache.delete(key);
    maskTextureCache.set(key, cached);
    return cached;
  }

  if (!maskScratchA) {
    maskScratchA = hostDoc.createElement("canvas").getContext("2d");
    maskScratchB = hostDoc.createElement("canvas").getContext("2d");
  }
  if (!maskScratchA || !maskScratchB) return null;
  maskScratchA.canvas.width = mw;
  maskScratchA.canvas.height = mh;
  maskScratchA.clearRect(0, 0, mw, mh);
  for (const img of readyMasks) {
    maskScratchA.drawImage(img, 0, 0, mw, mh);
  }
  let uploadSource = maskScratchA.canvas;
  if (fVal > 0) {
    maskScratchB.canvas.width = mw;
    maskScratchB.canvas.height = mh;
    maskScratchB.clearRect(0, 0, mw, mh);
    maskScratchB.save();
    maskScratchB.filter = `blur(${fVal * Math.max(mw, mh) / 1280}px)`;
    maskScratchB.drawImage(maskScratchA.canvas, 0, 0, mw, mh);
    maskScratchB.restore();
    uploadSource = maskScratchB.canvas;
  }

  const texture = gpuDevice.createTexture({
    size: [mw, mh],
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  gpuDevice.queue.copyExternalImageToTexture({ source: uploadSource }, { texture }, [mw, mh]);
  const entry = { texture, view: texture.createView(), width: mw, height: mh };
  maskTextureCache.set(key, entry);
  while (maskTextureCache.size > 32) {
    const oldest = maskTextureCache.keys().next().value;
    maskTextureCache.get(oldest)?.texture?.destroy();
    maskTextureCache.delete(oldest);
  }
  return entry;
}

function createLayerBindGroup(backView, frontView, backDepthView, frontDepthView, maskView) {
  return gpuDevice.createBindGroup({
    layout: gpuBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: gpuUniformBuffer, offset: 0, size: UNIFORM_STRUCT_BYTES } },
      { binding: 1, resource: gpuSampler },
      { binding: 2, resource: backView || gpuDummyView },
      { binding: 3, resource: frontView || gpuDummyView },
      { binding: 4, resource: backDepthView || gpuDummyDepthView },
      { binding: 5, resource: frontDepthView || gpuDummyDepthView },
      { binding: 6, resource: maskView || gpuDummyMaskView },
    ],
  });
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

let configuredCanvas = null;
function prepareContext(hostCanvas, w, h) {
  const gpuCanvas = getGpuCanvas(hostCanvas, true);
  if (!gpuCanvas) return null;
  const ctx = gpuCanvas.getContext("webgpu");
  if (!ctx) return null;
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
  return { gpuCanvas, ctx };
}

function writeDepthUniforms(baseFloatIdx, meta, hasDepth) {
  const valid = Boolean(hasDepth && meta && (meta.inv_max ?? 0) > (meta.inv_min ?? 0));
  uniformFloats[baseFloatIdx + 0] = valid ? 1 : 0;
  uniformFloats[baseFloatIdx + 1] = valid ? (meta.depth_strength ?? 1) : 0;
  uniformFloats[baseFloatIdx + 2] = meta?.inv_min ?? 0;
  uniformFloats[baseFloatIdx + 3] = meta?.inv_max ?? 1;

  const subjectZ = meta?.subject_z ?? 4;
  const fPxNorm = (meta?.f_px && meta?.width && meta?.height)
    ? meta.f_px / Math.max(meta.width, meta.height)
    : 1.1;
  uniformFloats[baseFloatIdx + 4] = subjectZ;
  uniformFloats[baseFloatIdx + 5] = meta?.rack_start_z ?? 12;
  uniformFloats[baseFloatIdx + 6] = fPxNorm;
  uniformFloats[baseFloatIdx + 7] = meta?.parallax_scale ?? 1;

  uniformFloats[baseFloatIdx + 8] = meta?.z_p10 ?? 2;
  uniformFloats[baseFloatIdx + 9] = meta?.z_p25 ?? 4;
  uniformFloats[baseFloatIdx + 10] = meta?.z_p50 ?? 8;
  uniformFloats[baseFloatIdx + 11] = meta?.z_p90 ?? 24;

  uniformFloats[baseFloatIdx + 12] = meta?.warm_anchor?.u ?? 0.5;
  uniformFloats[baseFloatIdx + 13] = meta?.warm_anchor?.v ?? 0.62;
  uniformFloats[baseFloatIdx + 14] = meta?.warm_anchor?.z ?? subjectZ;
  uniformFloats[baseFloatIdx + 15] = meta?.warm_anchor?.strength ?? 0.45;

  uniformFloats[baseFloatIdx + 16] = meta?.sky_dir?.u ?? 0.26;
  uniformFloats[baseFloatIdx + 17] = meta?.sky_dir?.v ?? 0.12;
  uniformFloats[baseFloatIdx + 18] = meta?.subject_uv?.[0] ?? 0.5;
  uniformFloats[baseFloatIdx + 19] = meta?.subject_uv?.[1] ?? 0.46;
}

const DEG_TO_RAD = Math.PI / 180;
const IDENTITY_CAM = { tx: 0, ty: 0, scale: 1, shake: 0, angle: 0, blur: 0 };

function writeShotUniformSlot(slotIdx, {
  w, h, seconds, backSeconds = 0, fadeMix = 1, shotSeed = 0, backShotSeed = 0, texMode = 0,
  cam = IDENTITY_CAM, backCam = IDENTITY_CAM, backW = w, backH = h, frontW = w, frontH = h,
  cueTime = seconds, backCueTime = backSeconds, rackActive = 0, backRackActive = 0,
  atmoId = 0, lightId = 0, backAtmoId = 0, backLightId = 0,
  depthMeta = null, hasDepth = false, backDepthMeta = null, hasBackDepth = false,
  intensity = 1, maskMode = 0, maskRefZ = 0, hasMask = false,
}) {
  const b = slotIdx * UNIFORM_SLOT_FLOATS;
  uniformFloats[b + 0] = w;
  uniformFloats[b + 1] = h;
  uniformFloats[b + 2] = seconds;
  uniformFloats[b + 3] = backSeconds;

  uniformFloats[b + 4] = fadeMix;
  uniformFloats[b + 5] = shotSeed;
  uniformFloats[b + 6] = backShotSeed;
  uniformFloats[b + 7] = texMode;

  uniformFloats[b + 8] = (cam.tx || 0) / 100;
  uniformFloats[b + 9] = (cam.ty || 0) / 100;
  uniformFloats[b + 10] = cam.scale || 1;
  uniformFloats[b + 11] = cam.shake || 0;

  uniformFloats[b + 12] = (backCam.tx || 0) / 100;
  uniformFloats[b + 13] = (backCam.ty || 0) / 100;
  uniformFloats[b + 14] = backCam.scale || 1;
  uniformFloats[b + 15] = backCam.shake || 0;

  uniformFloats[b + 16] = backW;
  uniformFloats[b + 17] = backH;
  uniformFloats[b + 18] = frontW;
  uniformFloats[b + 19] = frontH;

  uniformFloats[b + 20] = (cam.angle || 0) * DEG_TO_RAD;
  uniformFloats[b + 21] = cam.blur || 0;
  uniformFloats[b + 22] = (backCam.angle || 0) * DEG_TO_RAD;
  uniformFloats[b + 23] = backCam.blur || 0;

  uniformFloats[b + 24] = cueTime;
  uniformFloats[b + 25] = backCueTime;
  uniformFloats[b + 26] = rackActive ? 1 : 0;
  uniformFloats[b + 27] = backRackActive ? 1 : 0;

  uniformUints[b + 28] = atmoId;
  uniformUints[b + 29] = lightId;
  uniformUints[b + 30] = backAtmoId;
  uniformUints[b + 31] = backLightId;

  writeDepthUniforms(b + 32, depthMeta, hasDepth);
  writeDepthUniforms(b + 52, backDepthMeta, hasBackDepth);

  uniformFloats[b + 72] = intensity;
  uniformFloats[b + 73] = maskMode;
  uniformFloats[b + 74] = maskRefZ;
  uniformFloats[b + 75] = hasMask ? 1 : 0;
}

const IMAGE_ATMOSPHERE = new Set(["rain", "fog", "sunbeams", "smoke", "underwater"].map((id) => ATMO_IDS[id]));
const IMAGE_LIGHT = new Set(Object.values(LIGHT_IDS).filter((id) => id !== LIGHT_IDS.vignette));
const camMoves = (cam) => Boolean(
  cam && (
    cam.blur || cam.shake ||
    Math.abs(cam.tx || 0) > 0.001 ||
    Math.abs(cam.ty || 0) > 0.001 ||
    Math.abs((cam.scale ?? 1) - 1) > 0.001 ||
    Math.abs(cam.angle || 0) > 0.001
  )
);
const usesImage = (atmo, light, cam, hasDepth, rackActive) =>
  IMAGE_ATMOSPHERE.has(atmo) ||
  IMAGE_LIGHT.has(light) ||
  cam?.blur ||
  cam?.shake ||
  rackActive ||
  (Boolean(hasDepth) && (atmo > 0 || light > 0 || camMoves(cam)));

export function renderWebGPU(hostCanvas, atmoId, lightId, seconds, w, h, scene = null) {
  if (gpuState !== "ready" || !gpuDevice || !gpuPipeline) return false;
  const fading = Boolean(scene?.fading);
  const backAtmoId = fading ? (scene.backAtmoId || 0) : 0;
  const backLightId = fading ? (scene.backLightId || 0) : 0;
  const backSrc = readyStill(scene?.stillBack);
  const frontSrc = fading && readyStill(scene.stillFront);

  const depthBackImg = fading ? (scene?.backDepthImage ?? scene?.depthImage) : scene?.depthImage;
  const depthFrontImg = fading ? scene?.depthImage : null;
  const depthBackSrc = readyStill(depthBackImg);
  const depthFrontSrc = readyStill(depthFrontImg);
  const depthBackMeta = fading ? (scene?.backDepthMeta ?? scene?.depthMeta) : scene?.depthMeta;
  const depthFrontMeta = fading ? scene?.depthMeta : null;
  const hasBackDepth = Boolean(depthBackSrc && depthBackMeta && (depthBackMeta.depth_strength ?? 1) > 0.01);
  const hasFrontDepth = Boolean(depthFrontSrc && depthFrontMeta && (depthFrontMeta.depth_strength ?? 1) > 0.01);
  const rackActive = Boolean(scene?.rackActive);
  const backRackActive = Boolean(scene?.backRackActive);

  let back = null, front = null, backDepth = null, frontDepth = null;
  if (
    backSrc && (!fading || frontSrc) && (
      usesImage(atmoId, lightId, scene?.cam, fading ? hasFrontDepth : hasBackDepth, rackActive) ||
      (fading && usesImage(backAtmoId, backLightId, scene?.backCam, hasBackDepth, backRackActive))
    )
  ) {
    try {
      back = syncStillTexture("back", scene.stillBack, backSrc);
      if (back && hasBackDepth) {
        backDepth = syncDepthTexture("backDepth", depthBackImg, depthBackSrc);
      }
      if (fading && back) {
        front = frontSrc === backSrc ? back : syncStillTexture("front", scene.stillFront, frontSrc);
        if (!front) {
          back = null;
        } else if (hasFrontDepth) {
          frontDepth = depthFrontSrc === depthBackSrc && backDepth
            ? backDepth
            : syncDepthTexture("frontDepth", depthFrontImg, depthFrontSrc);
        }
      }
    } catch {
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

  const target = prepareContext(hostCanvas, w, h);
  if (!target) return false;
  const { gpuCanvas, ctx } = target;

  const fadeMix = fading ? Math.max(0, Math.min(1, scene?.fadeMix ?? 1)) : 1;
  writeShotUniformSlot(0, {
    w, h,
    seconds,
    backSeconds: scene?.backElapsed ?? 0,
    fadeMix,
    shotSeed: scene?.shotIndex ?? 0,
    backShotSeed: scene?.backShotIndex ?? 0,
    texMode,
    cam: scene?.cam || IDENTITY_CAM,
    backCam: scene?.backCam || IDENTITY_CAM,
    backW: back?.width ?? w,
    backH: back?.height ?? h,
    frontW: front?.width ?? w,
    frontH: front?.height ?? h,
    cueTime: scene?.cueElapsed ?? seconds,
    backCueTime: scene?.backCueElapsed ?? (scene?.backElapsed ?? 0),
    rackActive,
    backRackActive,
    atmoId,
    lightId,
    backAtmoId,
    backLightId,
    depthMeta: fading ? depthFrontMeta : depthBackMeta,
    hasDepth: fading ? Boolean(frontDepth) : Boolean(backDepth),
    backDepthMeta: depthBackMeta,
    hasBackDepth: Boolean(backDepth),
  });
  gpuDevice.queue.writeBuffer(gpuUniformBuffer, 0, uniformBytes, 0, UNIFORM_STRUCT_BYTES);

  const v0 = back?.view || gpuDummyView;
  const v1 = front?.view || gpuDummyView;
  const v2 = backDepth?.view || gpuDummyDepthView;
  const v3 = frontDepth?.view || gpuDummyDepthView;
  if (
    !cachedSingleBindGroup ||
    cachedSingleViews[0] !== v0 ||
    cachedSingleViews[1] !== v1 ||
    cachedSingleViews[2] !== v2 ||
    cachedSingleViews[3] !== v3
  ) {
    cachedSingleBindGroup = createLayerBindGroup(v0, v1, v2, v3, gpuDummyMaskView);
    cachedSingleViews = [v0, v1, v2, v3];
  }
  const bindGroup = cachedSingleBindGroup;
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
  pass.setBindGroup(0, bindGroup, [0]);
  pass.draw(3);
  pass.end();
  gpuDevice.queue.submit([encoder.finish()]);

  gpuCanvas.style.visibility = "visible";
  if (hostCanvas.style) hostCanvas.style.visibility = "hidden";
  return true;
}

const MASK_MODE_IDS = { full_frame: 0, inside: 1, outside: 2, behind: 3 };

export function renderWebGPULayers(hostCanvas, layers, seconds, w, h, scene = null) {
  if (gpuState !== "ready" || !gpuDevice) return false;
  if (!gpuBasePipeline || !gpuLayerPipeline || !gpuCompositePipeline) {
    ensureLayerPipelines();
    return false;
  }
  const fading = Boolean(scene?.fading && (scene?.fadeMix ?? 1) < 1);
  const backSrc = readyStill(scene?.stillBack);
  const frontSrc = fading && readyStill(scene?.stillFront);
  if (!backSrc || (fading && !frontSrc)) return false;

  const ratio = Math.min(globalThis.devicePixelRatio || 1, 1280 / Math.max(w, h));
  w = Math.max(1, Math.round(w * ratio));
  h = Math.max(1, Math.round(h * ratio));
  const target = prepareContext(hostCanvas, w, h);
  if (!target) return false;
  const { gpuCanvas, ctx } = target;

  let stillBack, stillFront = null, backDepth = null, frontDepth = null;
  try {
    stillBack = syncStillTexture("back", scene.stillBack, backSrc);
    if (!stillBack) return false;
    const backDepthImg = fading ? (scene.backDepthImage ?? scene.depthImage) : scene.depthImage;
    const backDepthMeta = fading ? (scene.backDepthMeta ?? scene.depthMeta) : scene.depthMeta;
    if (backDepthMeta && readyStill(backDepthImg)) {
      backDepth = syncDepthTexture("backDepth", backDepthImg, readyStill(backDepthImg));
    }
    if (fading) {
      stillFront = frontSrc === backSrc ? stillBack : syncStillTexture("front", scene.stillFront, frontSrc);
      if (!stillFront) return false;
      const fDepthSrc = readyStill(scene.depthImage);
      if (scene.depthMeta && fDepthSrc) {
        frontDepth = fDepthSrc === readyStill(backDepthImg) && backDepth
          ? backDepth
          : syncDepthTexture("frontDepth", scene.depthImage, fDepthSrc);
      }
    }
  } catch {
    return false;
  }

  const rtBack = syncRenderTarget("back", w, h);
  const rtFront = fading ? syncRenderTarget("front", w, h) : null;
  let nextSlot = 0;
  const passOps = [];

  function scheduleShot(rt, stillSlot, depthSlot, depthMeta, cam, shotT, shotCueT, shotSeed, shotLayers) {
    const rackActive = Boolean((shotLayers || []).some((l) => l.preset === "rack_focus"));
    const baseSlot = nextSlot++;
    if (baseSlot >= MAX_UNIFORM_SLOTS) return false;
    writeShotUniformSlot(baseSlot, {
      w, h,
      seconds: shotT,
      fadeMix: 1,
      shotSeed,
      texMode: 1,
      cam: cam || IDENTITY_CAM,
      backW: stillSlot.width,
      backH: stillSlot.height,
      cueTime: shotCueT,
      rackActive,
      depthMeta,
      hasDepth: Boolean(depthSlot),
    });
    const baseBind = createLayerBindGroup(stillSlot.view, null, depthSlot?.view, null, null);
    passOps.push({ kind: "base", rtView: rt.view, slot: baseSlot, bindGroup: baseBind });

    for (const layer of shotLayers || []) {
      const preset = layer.preset;
      const atmoId = ATMO_IDS[preset] || 0;
      const lightId = LIGHT_IDS[preset] || 0;
      if (!atmoId && !lightId) continue;
      const slot = nextSlot++;
      if (slot >= MAX_UNIFORM_SLOTS) return false;
      const maskModeName = layer.mode || "full_frame";
      const maskMode = MASK_MODE_IDS[maskModeName] ?? 0;
      let maskEntry = null;
      if (maskMode > 0 && layer.maskImages?.length) {
        maskEntry = syncMaskTexture(hostCanvas.ownerDocument, layer.maskImages, layer.feather, layer.maskUrls);
      }
      writeShotUniformSlot(slot, {
        w, h,
        seconds: shotT,
        fadeMix: 1,
        shotSeed,
        texMode: 1,
        cam: cam || IDENTITY_CAM,
        backW: stillSlot.width,
        backH: stillSlot.height,
        cueTime: shotCueT,
        rackActive,
        atmoId,
        lightId,
        depthMeta,
        hasDepth: Boolean(depthSlot),
        intensity: layer.intensity ?? 1,
        maskMode,
        maskRefZ: layer.region_depth_m ?? 0,
        hasMask: Boolean(maskEntry),
      });
      const layerBind = createLayerBindGroup(stillSlot.view, null, depthSlot?.view, null, maskEntry?.view);
      passOps.push({ kind: "layer", rtView: rt.view, slot, bindGroup: layerBind });
    }
    return true;
  }

  const backLayers = fading ? (scene.backLayers ?? []) : (layers ?? []);
  const backCam = fading ? (scene.backCam || IDENTITY_CAM) : (scene?.cam || IDENTITY_CAM);
  const backT = fading ? Math.max(0, scene.backElapsed ?? 0) : Math.max(0, seconds);
  const backCueT = fading ? Math.max(0, scene.backCueElapsed ?? backT) : Math.max(0, scene?.cueElapsed ?? seconds);
  const backSeed = fading ? (scene.backShotIndex ?? 0) : (scene?.shotIndex ?? 0);
  const backMeta = fading ? (scene.backDepthMeta ?? scene.depthMeta) : scene?.depthMeta;

  if (!scheduleShot(rtBack, stillBack, backDepth, backMeta, backCam, backT, backCueT, backSeed, backLayers)) {
    return false;
  }
  if (fading) {
    if (!scheduleShot(rtFront, stillFront, frontDepth, scene?.depthMeta, scene?.cam, Math.max(0, seconds),
      Math.max(0, scene?.cueElapsed ?? seconds), scene?.shotIndex ?? 0, layers ?? [])) {
      return false;
    }
  }

  const presentSlot = nextSlot++;
  if (presentSlot >= MAX_UNIFORM_SLOTS) return false;
  writeShotUniformSlot(presentSlot, {
    w, h,
    seconds,
    fadeMix: fading ? Math.max(0, Math.min(1, scene?.fadeMix ?? 1)) : 1,
    texMode: fading ? 2 : 1,
  });
  const presentBind = createLayerBindGroup(rtBack.view, fading ? rtFront.view : null, null, null, null);

  gpuDevice.queue.writeBuffer(gpuUniformBuffer, 0, uniformBytes, 0, nextSlot * UNIFORM_STRIDE);

  const encoder = gpuDevice.createCommandEncoder();
  for (const op of passOps) {
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: op.rtView,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: op.kind === "base" ? "clear" : "load",
        storeOp: "store",
      }],
    });
    pass.setPipeline(op.kind === "base" ? gpuBasePipeline : gpuLayerPipeline);
    pass.setBindGroup(0, op.bindGroup, [op.slot * UNIFORM_STRIDE]);
    pass.draw(3);
    pass.end();
  }

  const finalPass = encoder.beginRenderPass({
    colorAttachments: [{
      view: ctx.getCurrentTexture().createView(),
      clearValue: { r: 0, g: 0, b: 0, a: 0 },
      loadOp: "clear",
      storeOp: "store",
    }],
  });
  finalPass.setPipeline(gpuCompositePipeline);
  finalPass.setBindGroup(0, presentBind, [presentSlot * UNIFORM_STRIDE]);
  finalPass.draw(3);
  finalPass.end();
  gpuDevice.queue.submit([encoder.finish()]);

  gpuCanvas.style.visibility = "visible";
  if (hostCanvas.style) hostCanvas.style.visibility = "hidden";
  return true;
}
