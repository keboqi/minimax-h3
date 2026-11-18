// WebGPU WGSL atmospheric & lighting shader backend for Sound Stage.
// Automatically probed when navigator.gpu is available; falls back to Canvas 2D otherwise.

export const ATMO_IDS = {
  rain: 1, snow: 2, fog: 3, dust: 4, embers: 5, sunbeams: 6, smoke: 7, petals: 8,
};

export const LIGHT_IDS = {
  lightning: 1, flash: 2, warm_glow: 3, cold_tint: 4, flicker: 5,
  vignette: 6, film_grain: 7, dream_bloom: 8, scanlines: 9,
};

const WGSL = /* wgsl */ `
struct Uniforms {
  resolution: vec2f,
  time: f32,
  atmo: u32,
  light: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

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
  let outA = clamp(dst.a + a * (1.0 - dst.a), 0.0, 1.0);
  return vec4f(rgb, outA);
}

fn invSmooth(low: f32, high: f32, x: f32) -> f32 {
  return 1.0 - smoothstep(low, high, x);
}

fn rainLayer(uv: vec2f, t: f32, scale: vec2f, speed: f32) -> f32 {
  let slant = vec2f(uv.x + uv.y * 0.24, uv.y);
  let st = slant * scale + vec2f(0.0, -t * speed);
  let id = floor(st);
  let f = fract(st);
  let rnd = hash21(id);
  if (rnd < 0.62) { return 0.0; }
  let x = fract(rnd * 17.3) * 0.6 + 0.2;
  let yStart = fract(rnd * 31.7) * 0.35;
  let dx = abs(f.x - x);
  let streak = invSmooth(0.0, 0.045, dx) * smoothstep(yStart, yStart + 0.12, f.y) * invSmooth(yStart + 0.45, yStart + 0.72, f.y);
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

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  let uv = in.uv;
  let aspect = u.resolution.x / max(1.0, u.resolution.y);
  let auv = vec2f(uv.x * aspect, uv.y);
  let t = max(0.0, u.time);
  var out = vec4f(0.0);

  // --- Atmosphere Pass ---
  if (u.atmo == 1u) { // rain
    let r1 = rainLayer(uv, t, vec2f(32.0 * aspect, 10.0), 3.6);
    let r2 = rainLayer(uv + vec2f(0.37, 0.19), t, vec2f(52.0 * aspect, 16.0), 2.5);
    let rainA = clamp(r1 * 0.36 + r2 * 0.22, 0.0, 0.55);
    out = blendOver(out, vec3f(0.76, 0.86, 0.96), rainA);
    let mist = smoothstep(0.68, 1.0, uv.y) * 0.13;
    out = blendOver(out, vec3f(0.72, 0.80, 0.88), mist);
  } else if (u.atmo == 2u) { // snow
    let s1 = particleDots(uv, aspect, t, 11.0, 0.55, 0.55, 0.16, 2.2);
    let s2 = particleDots(uv + vec2f(0.41, 0.23), aspect, t, 18.0, 0.35, 0.4, 0.12, 2.0);
    let snowA = clamp(s1.x * 0.72 + s2.x * 0.45, 0.0, 0.85);
    out = blendOver(out, vec3f(0.95, 0.98, 1.0), snowA);
  } else if (u.atmo == 3u) { // fog
    let n1 = fbm(auv * 2.1 + vec2f(t * 0.06, sin(t * 0.12) * 0.05));
    let n2 = fbm(auv * 3.6 + vec2f(-t * 0.04, 2.4));
    let fogMask = smoothstep(0.22, 0.95, uv.y);
    let density = smoothstep(0.32, 0.82, n1 * 0.65 + n2 * 0.35) * fogMask * 0.42;
    out = blendOver(out, vec3f(0.84, 0.90, 0.93), density);
  } else if (u.atmo == 4u) { // dust
    let d1 = particleDots(uv, aspect, t, 9.5, -0.14, 0.65, 0.11, 2.3);
    out = blendScreen(out, vec3f(1.0, 0.91, 0.72), d1.x * 0.45);
  } else if (u.atmo == 5u) { // embers
    let e1 = particleDots(uv, aspect, t, 10.5, -0.85, 0.75, 0.15, 2.6);
    let e2 = particleDots(uv + vec2f(0.31, 0.17), aspect, t, 16.0, -0.62, 0.6, 0.11, 2.4);
    let emberA = clamp(e1.x * 0.9 + e2.x * 0.6, 0.0, 1.0);
    let emberRgb = mix(vec3f(1.0, 0.38, 0.10), vec3f(1.0, 0.82, 0.36), clamp(e1.y + e2.y, 0.0, 1.0));
    out = blendScreen(out, emberRgb, emberA);
  } else if (u.atmo == 6u) { // sunbeams
    let diag = (auv.x * 0.82 + uv.y * 0.56);
    let rays = fbm(vec2f(diag * 4.5 + t * 0.05, uv.y * 1.2 - t * 0.03));
    let beam = smoothstep(0.42, 0.84, rays) * invSmooth(0.05, 1.0, uv.y) * smoothstep(0.0, 0.35, uv.x);
    out = blendScreen(out, vec3f(1.0, 0.92, 0.74), beam * 0.32);
  } else if (u.atmo == 7u) { // smoke
    let s1 = fbm(auv * 2.6 + vec2f(sin(t * 0.25) * 0.15, t * 0.18));
    let s2 = fbm(auv * 4.8 + vec2f(-t * 0.08, t * 0.26));
    let density = smoothstep(0.38, 0.86, s1 * 0.6 + s2 * 0.4) * smoothstep(0.0, 0.85, uv.y) * 0.38;
    out = blendOver(out, vec3f(0.46, 0.49, 0.54), density);
  } else if (u.atmo == 8u) { // petals
    let p = vec2f(uv.x * aspect + uv.y * 0.35, uv.y) * 8.5 + vec2f(-t * 0.38, -t * 0.45);
    let cell = floor(p);
    let f = fract(p) - 0.5;
    let rnd = hash22(cell);
    if (rnd.x > 0.56) {
      let ang = t * (1.8 + rnd.y * 2.0) + rnd.x * 6.28;
      let cs = cos(ang); let sn = sin(ang);
      let q = vec2f(f.x * cs - f.y * sn, f.x * sn + f.y * cs);
      let d = length(vec2f(q.x * 1.1, q.y * 2.1));
      let petalA = invSmooth(0.08, 0.16, d) * 0.75;
      let petalRgb = mix(vec3f(0.98, 0.74, 0.80), vec3f(0.96, 0.60, 0.42), rnd.y);
      out = blendOver(out, petalRgb, petalA);
    }
  }

  // --- Lighting & Framing Pass ---
  if (u.light == 1u) { // lightning
    let phase = t % 7.0;
    let primary = max(0.0, 1.0 - abs(phase - 0.12) / 0.12) * 0.32;
    let echo = max(0.0, 1.0 - abs(phase - 0.30) / 0.08) * 0.14;
    let a = max(primary, echo);
    let sky = invSmooth(0.0, 1.1, length(uv - vec2f(0.52, 0.16))) * a * 0.75;
    out = blendScreen(out, vec3f(0.88, 0.93, 1.0), clamp(a + sky, 0.0, 0.55));
  } else if (u.light == 2u) { // flash
    let a = max(0.0, 1.0 - t / 0.45) * 0.35;
    out = blendScreen(out, vec3f(0.94, 0.96, 1.0), a);
  } else if (u.light == 3u) { // warm_glow
    let pulse = 0.11 + sin(t * 0.8) * 0.035;
    let center = vec2f(0.5 + sin(t * 0.22) * 0.06, 0.42 + cos(t * 0.18) * 0.04);
    let glow = invSmooth(0.0, 0.95, length((uv - center) * vec2f(aspect, 1.0)));
    out = blendScreen(out, vec3f(1.0, 0.72, 0.38), glow * pulse * 1.45);
  } else if (u.light == 4u) { // cold_tint
    let pulse = 0.09 + sin(t * 0.5) * 0.02;
    let edge = smoothstep(0.15, 1.05, length(uv - vec2f(0.5, 0.4)));
    out = blendOver(out, vec3f(0.24, 0.44, 0.78), mix(0.35, 1.0, edge) * pulse * 1.35);
  } else if (u.light == 5u) { // flicker
    let wave = 0.5 + 0.3 * sin(t * 7.3) + 0.2 * sin(t * 13.1);
    let a = 0.04 + wave * 0.08;
    let center = vec2f(0.5 + sin(t * 5.1) * 0.02, 0.62);
    let glow = invSmooth(0.0, 0.85, length(uv - center));
    out = blendScreen(out, vec3f(1.0, 0.64, 0.28), glow * a * 1.6);
  } else if (u.light == 6u) { // vignette
    let d = length((uv - 0.5) * vec2f(1.15, 1.0));
    let vig = smoothstep(0.28, 0.82, d) * 0.56;
    out = blendOver(out, vec3f(0.0, 0.0, 0.0), vig);
  } else if (u.light == 7u) { // film_grain
    let frame = floor(t * 24.0);
    let g = hash21(floor(uv * u.resolution * 0.5) + vec2f(frame * 13.7, frame * 7.3)) - 0.5;
    let grainRgb = select(vec3f(0.0), vec3f(1.0), g > 0.0);
    out = blendOver(out, grainRgb, abs(g) * 0.14);
  } else if (u.light == 8u) { // dream_bloom
    let pulse = 0.14 + sin(t * 0.55) * 0.03;
    let halo = invSmooth(0.0, 0.95, length(uv - vec2f(0.5, 0.45)));
    out = blendScreen(out, vec3f(0.98, 0.88, 0.96), halo * pulse);
  } else if (u.light == 9u) { // scanlines
    let line = 0.5 - 0.5 * cos(uv.y * u.resolution.y * 3.14159);
    let bar = invSmooth(0.0, 0.08, abs(fract(uv.y - t * 0.08) - 0.5));
    out = blendOver(out, vec3f(0.04, 0.08, 0.12), line * 0.16);
    out = blendScreen(out, vec3f(0.35, 0.85, 0.95), bar * 0.07);
  }

  return vec4f(out.rgb, clamp(out.a, 0.0, 1.0));
}
`;

let gpuState = "idle"; // "idle" | "probing" | "ready" | "unavailable"
let gpuDevice = null;
let gpuPipeline = null;
let gpuUniformBuffer = null;
let gpuBindGroup = null;
let gpuFormat = "bgra8unorm";
const uniformData = new ArrayBuffer(32);
const uniformFloats = new Float32Array(uniformData);
const uniformUints = new Uint32Array(uniformData);

export function gpuStatus() {
  return gpuState;
}

export async function probeWebGPU(onReady) {
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
    const pipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs" },
      fragment: { module, entryPoint: "fs", targets: [{ format: gpuFormat }] },
      primitive: { topology: "triangle-list" },
    });
    const uniformBuffer = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
    });
    device.lost.then(() => {
      gpuDevice = null;
      gpuPipeline = null;
      configuredCanvas = null;
      gpuState = "unavailable";
    });
    gpuDevice = device;
    gpuPipeline = pipeline;
    gpuUniformBuffer = uniformBuffer;
    gpuBindGroup = bindGroup;
    gpuState = "ready";
    onReady?.();
    return true;
  } catch {
    gpuState = "unavailable";
    return false;
  }
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
export function renderWebGPU(hostCanvas, atmoId, lightId, seconds, w, h) {
  if (gpuState !== "ready" || !gpuDevice || !gpuPipeline) return false;
  const gpuCanvas = getGpuCanvas(hostCanvas, true);
  if (!gpuCanvas) return false;
  const ctx = gpuCanvas.getContext("webgpu");
  if (!ctx) return false;
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
  uniformFloats[0] = w;
  uniformFloats[1] = h;
  uniformFloats[2] = seconds;
  uniformUints[3] = atmoId;
  uniformUints[4] = lightId;
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
  pass.setBindGroup(0, gpuBindGroup);
  pass.draw(3);
  pass.end();
  gpuDevice.queue.submit([encoder.finish()]);

  gpuCanvas.style.visibility = "visible";
  if (hostCanvas.style) hostCanvas.style.visibility = "hidden";
  return true;
}
