import test from "node:test";
import assert from "node:assert/strict";
import {cameraTransform, paintEffects} from "../h3_ui/drama3/page/effects.js";

test("camera moves clamp to shot boundaries and old plans remain still", () => {
  assert.equal(cameraTransform([], 0.5), "none");
  assert.equal(cameraTransform(["push_in"], -1), "scale(1)");
  assert.equal(cameraTransform(["push_in"], 2), "scale(1.08)");
  assert.equal(cameraTransform(["pull_out"], 0), "scale(1.08)");
  assert.match(cameraTransform(["pan_left"], 1), /translateX\(-3%\)/);
});

test("overlays reconstruct the same frame after seeking and clear when removed", () => {
  const calls = [];
  const gradient = {addColorStop() {}};
  const ctx = new Proxy({}, {get: (_, name) => (...args) => {
    calls.push([name, ...args]);
    if (name === "createRadialGradient") return gradient;
  }, set: (_, name, value) => {calls.push([name, value]); return true;}});
  const canvas = {clientWidth: 640, clientHeight: 360, getContext: () => ctx};
  for (const effect of ["rain", "snow", "dust", "embers", "fog", "lightning", "flash", "warm_glow", "vignette"]) {
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
