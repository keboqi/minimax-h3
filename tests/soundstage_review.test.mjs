import test from "node:test";
import assert from "node:assert/strict";
import {createMix, lineTakes} from "../h3_ui/drama3/page/mix.js";
import {interruptedRun} from "../h3_ui/drama3/page/transport.js";
import {createProjects} from "../h3_ui/drama3/page/project.js";
import {createLibrary} from "../h3_ui/drama3/page/library.js";

test("overlapping dialogue selects only its own speaker's take", () => {
  const a = {key: "a", speaker: "A", text: "Opening", start: 0, end: 5};
  const b = {key: "b", speaker: "B", text: "Reply", start: 1, end: 2};
  assert.deepEqual(lineTakes({lines: [{speaker: "B", text: "Reply", start: 1}], takes: [a, b]}), [b]);
  assert.deepEqual(lineTakes({lines: [{speaker: "C", text: "Reply", start: 1}], takes: [a, b]}), [null]);
});

test("interrupted runs are distinct from completion and failure", () => {
  assert.equal(interruptedRun({active: false, status: "Interrupted."}), true);
  assert.equal(interruptedRun({active: false, status: "Auto interrupted."}), true);
  assert.equal(interruptedRun({active: false, stop_requested: true}), true);
  assert.equal(interruptedRun({active: true, stop_requested: true}), false);
  assert.equal(interruptedRun({active: false, stop_requested: true, error: "failed"}), false);
  assert.equal(interruptedRun({active: false, status: "Done."}), false);
});

test("saving video settings redraws and synchronizes the player's shot lane", async () => {
  const calls = [], noop = () => {};
  const state = {project: {name: "A", script: {writer: {}}, video: {limit: 2}}};
  const projects = createProjects({state, scope: () => "A", named: (path) => path,
    api: async () => ({name: "A", script: {writer: {}}, video: {limit: 3}}),
    renderSaveState: noop, renderHeader: noop, renderSteps: noop, renderSettings: noop,
    renderScript: noop, queueEstimate: noop, renderVideo: () => calls.push("video"),
    renderActions: noop, renderPlayer: () => calls.push("player"), sync: () => calls.push("sync"),
    castAnswered: noop, toast: noop});
  projects.saves.queue("A", "video", {video_limit: 3});
  await projects.flushEdits();
  assert.equal(state.project.video.limit, 3);
  assert.deepEqual(calls, ["video", "player", "sync"]);
});

test("voice library drafts survive generation and polling rerenders", async () => {
  const h = (tag, attrs = {}, ...children) => ({tag, ...attrs, children: children.flat(Infinity).filter(Boolean)});
  const surfaces = new Map();
  const $ = (id) => {
    if (!surfaces.has(id)) surfaces.set(id, {open: true, children: [], replaceChildren(...nodes) { this.children = nodes; }});
    return surfaces.get(id);
  };
  const picked = {key: "mia", name: "Mia", group: "Global", description: "Old description", transcript: "Old words"};
  const data = {groups: ["Global"], speakers: [picked], sounds: []};
  const run = {id: "run", key: "mia", active: false, takes: [], status: "Done"};
  const noop = () => {};
  let api = async () => ({takes_run: run});
  const lib = createLibrary({state: {project: {name: "A", cast_settings: {speaker_group: "Global"}}}, $, h,
    API: "/api", api: (...args) => api(...args), castAnswered: noop, loadLibrary: noop,
    renderSlotEditor: noop, field: (label, input) => h("label", {}, label, input), toast: noop,
    poller: (callback) => () => callback(noop)});
  Object.assign(lib.libraryState, {data, picked: "mia"});
  lib.renderLibrary();
  const walk = (node) => typeof node === "object" ? [node, ...(node.children || []).flatMap(walk)] : [];
  let fields = walk($("library-body")).filter((node) => node.tag === "textarea");
  fields[0].oninput({target: {value: "New description"}});
  fields[1].oninput({target: {value: "New words"}});
  await lib.tryVoices(picked, 1, "New description", "New words");
  await lib.pollTakes();
  fields = walk($("library-body")).filter((node) => node.tag === "textarea");
  assert.deepEqual(fields.map((node) => node.value), ["New description", "New words"]);

  // A concurrent save/rename acknowledges submitted fields but preserves newer typing.
  let release;
  const response = new Promise((resolve) => { release = resolve; });
  api = async (path) => path === "/library/speaker/update" ? response : data;
  const saving = lib.libraryCall("/library/speaker/update", {
    key: "mia", name: "Mia", description: "New description", transcript: "New words"});
  fields[0].oninput({target: {value: "Edited while saving"}});
  picked.key = "renamed-mia";
  picked.description = "New description";
  picked.transcript = "New words";
  release({key: picked.key});
  await saving;
  assert.equal(lib.libraryState.drafts.has("mia"), false);
  assert.deepEqual(lib.libraryState.drafts.get(picked.key), {description: "Edited while saving"});
});

test("renderLines includes start-end timestamps formatted like illustrate", () => {
  const prevDoc = globalThis.document;
  globalThis.document = {fullscreenEnabled: false, fullscreenElement: null};
  try {
    const h = (tag, attrs = {}, ...children) => ({tag, ...attrs, children: children.flat(Infinity).filter(Boolean)});
    const surfaces = new Map();
    const $ = (id) => {
      if (!surfaces.has(id)) surfaces.set(id, {
        children: [],
        addEventListener() {},
        replaceChildren(...nodes) { this.children = nodes; },
      });
      return surfaces.get(id);
    };
    const clock = (seconds) => {
      const whole = Math.floor(seconds);
      return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
    };
    const state = {
      filter: "all",
      picked: 0,
      project: {
        result: {
          lines: [
            {speaker: "narrator", text: "First line.", start: 0, end: 2.4},
            {speaker: "lin", text: "Second line.", start: 11.2, end: 18.9},
          ],
          takes: [],
        },
      },
    };
    const mix = createMix({
      state, $, h, API: "/api", STATUS_LABELS: {new: "New"}, FILTERS: [["all", "All"]], ICONS: {},
      clock, lineTakes, speakerTone: () => "narrator", toneColor: () => "#fff",
      pick: () => {}, svg: () => null,
    });
    mix.renderLines();
    const walk = (node) => (node && typeof node === "object" ? [node, ...(node.children || []).flatMap(walk)] : []);
    const times = walk($("lines")).filter((node) => node.class === "line-time num");
    assert.deepEqual(times.map((node) => node.children.join("")), ["0:00–0:02", "0:11–0:18"]);
  } finally {
    globalThis.document = prevDoc;
  }
});
