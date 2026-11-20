import test from "node:test";
import assert from "node:assert/strict";
import {createVideo} from "../h3_ui/drama3/page/video.js";

test("the video file button follows the current effects and reduced motion", (t) => {
  const oldWindow = globalThis.window;
  const motion = {matches: false, addEventListener() {}};
  globalThis.window = {matchMedia: () => motion};
  t.after(() => oldWindow === undefined ? delete globalThis.window : globalThis.window = oldWindow);
  const video = {subtitles: true, subtitle_size: 100, stills: {}, effects_revision: "new",
    latest: {current: true, subtitles: 100, effects_revision: "old"}};
  const requested = [];
  const {videoFileButton} = createVideo({state: {project: {video}},
    h: (tag, attrs) => ({tag, ...attrs}), saveVideoToMedia: (enabled) => requested.push(enabled)});

  videoFileButton().onclick();
  video.latest.effects_revision = "new";
  assert.equal(videoFileButton().tag, "a");
  motion.matches = true;
  videoFileButton().onclick();
  video.latest.effects_revision = null;
  assert.equal(videoFileButton().tag, "a");
  assert.deepEqual(requested, [true, false]);
});

test("Illustrate reloads untimed project plans and returns from Characters & places before speech", () => {
  const oldWindow = globalThis.window, oldDocument = globalThis.document;
  globalThis.window = {};
  globalThis.document = {activeElement: null};
  try {
    const h = (tag, attrs = {}, ...children) => ({tag, ...attrs,
      children: children.flat(Infinity).filter((child) => child != null && child !== false)});
    const surfaces = new Map();
    const $ = (id) => {
      if (!surfaces.has(id)) surfaces.set(id, {children: [], open: false,
        contains: () => false,
        setAttribute(key, value) { this[key] = value; },
        replaceChildren(...nodes) { this.children = nodes; }});
      return surfaces.get(id);
    };
    const walk = (node) => typeof node === "object" ? [node, ...(node.children || []).flatMap(walk)] : [];
    const text = (node) => typeof node === "object" ? (node.children || []).map(text).join("") : String(node);
    const project = (name, count, engine) => ({name, result: null, video: {
      engine, timing_pending: true, preset: "Custom", orientation: "landscape", limit: count,
      plan: {subjects: [], rows: Array.from({length: count}, (_, index) => ({
        index, status: "todo", beat: `${name} beat ${index}`, prompt: `${name} prompt`,
        transition: "Cut", subjects: [], start: index, end: index + 1,
      }))},
    }});
    const state = {boot: {gpu: true, video: {style_uses: {}, styles: {}, engines: []}},
      project: project("A", 2, "local"), screen: "illustrate", illustrateView: "shots",
      shotPick: 0, shotFilter: "all"};
    const video = createVideo({state, $, h});
    const button = (label) => walk($("illustrate-main")).find((node) => node.tag === "button" && text(node) === label);

    for (const next of [state.project, project("B", 137, "gpt_image"), project("A", 2, "local")]) {
      state.project = next;
      video.renderVideo();
      const count = next.video.plan.rows.length;
      assert.ok(text($("illustrate-main")).includes(`${next.name} beat 0`));
      assert.ok(!text($("illustrate-main")).includes(`${next.name === "A" ? "B" : "A"} beat 0`));
      assert.equal(button("Draw every shot again").disabled, next.video.engine !== "gpt_image");
      button("Characters & places").onclick();
      assert.equal($("illustrate-main")["aria-label"], "Characters and places");
      button(`Shot plan · ${count}`).onclick();
      assert.equal($("illustrate-main")["aria-label"], "Shot plan");
      assert.equal(walk($("illustrate-main")).filter((node) => node.class === "shot-row").length, count);
    }
  } finally {
    if (oldWindow === undefined) delete globalThis.window; else globalThis.window = oldWindow;
    if (oldDocument === undefined) delete globalThis.document; else globalThis.document = oldDocument;
  }
});
