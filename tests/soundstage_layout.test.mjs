import test from "node:test";
import assert from "node:assert/strict";
import {bindTablist, createLayout, readPref, writePref} from "../h3_ui/drama3/page/layout.js";

function makeElement(attrs = {}) {
  const listeners = new Map();
  const classes = new Set();
  return {
    ...attrs,
    hidden: Boolean(attrs.hidden),
    textContent: "",
    dataset: {},
    attributes: {...attrs},
    classList: {
      toggle(name, force) {
        if (force) classes.add(name);
        else classes.delete(name);
      },
      has(name) {
        return classes.has(name);
      },
    },
    setAttribute(name, value) {
      this.attributes[name] = String(value);
    },
    getAttribute(name) {
      return this.attributes[name] ?? null;
    },
    addEventListener(type, fn) {
      if (!listeners.has(type)) listeners.set(type, []);
      listeners.get(type).push(fn);
    },
    dispatch(type, event = {}) {
      for (const fn of listeners.get(type) || []) fn(event);
    },
    focus() {
      this.focused = true;
    },
  };
}

function withStorage(storage, fn) {
  const had = Object.prototype.hasOwnProperty.call(globalThis, "localStorage");
  const prev = globalThis.localStorage;
  if (storage === undefined) delete globalThis.localStorage;
  else Object.defineProperty(globalThis, "localStorage", {value: storage, configurable: true, writable: true});
  try {
    return fn();
  } finally {
    if (had) Object.defineProperty(globalThis, "localStorage", {value: prev, configurable: true, writable: true});
    else delete globalThis.localStorage;
  }
}

test("readPref and writePref round-trip values and survive missing or throwing storage", () => {
  withStorage(undefined, () => {
    writePref("k", "v");
    assert.equal(readPref("k", "fallback"), "fallback");
  });

  withStorage({
    getItem() { throw new Error("blocked"); },
    setItem() { throw new Error("blocked"); },
  }, () => {
    writePref("k", "v");
    assert.equal(readPref("k", "fallback"), "fallback");
  });

  const store = new Map();
  withStorage({
    getItem: (key) => (store.has(key) ? store.get(key) : null),
    setItem: (key, value) => store.set(key, value),
  }, () => {
    writePref("h3-soundstage-cast-view", "voices");
    assert.equal(readPref("h3-soundstage-cast-view", "split"), "voices");
    assert.equal(readPref("missing", "split"), "split");
  });
});

test("bindTablist handles click and keyboard navigation across tabs and panels", () => {
  const els = {
    "tab-a": makeElement({"aria-controls": "panel-a", "aria-selected": "true"}),
    "tab-b": makeElement({"aria-controls": "panel-b", "aria-selected": "false"}),
    "tab-c": makeElement({"aria-controls": "panel-c", "aria-selected": "false"}),
    "panel-a": makeElement({hidden: false}),
    "panel-b": makeElement({hidden: true}),
    "panel-c": makeElement({hidden: true}),
  };
  const $ = (id) => els[id] || null;
  const controller = bindTablist($, ["tab-a", "tab-b", "tab-c"]);

  els["tab-b"].dispatch("click");
  assert.equal(els["tab-a"].getAttribute("aria-selected"), "false");
  assert.equal(els["tab-a"].tabIndex, -1);
  assert.equal(els["panel-a"].hidden, true);
  assert.equal(els["tab-b"].getAttribute("aria-selected"), "true");
  assert.equal(els["tab-b"].tabIndex, 0);
  assert.equal(els["panel-b"].hidden, false);

  els["tab-b"].dispatch("keydown", {key: "ArrowRight", preventDefault() {}});
  assert.equal(els["panel-c"].hidden, false);
  assert.equal(els["tab-c"].focused, true);

  els["tab-c"].dispatch("keydown", {key: "Home", preventDefault() {}});
  assert.equal(els["panel-a"].hidden, false);

  els["tab-a"].dispatch("keydown", {key: "ArrowLeft", preventDefault() {}});
  assert.equal(els["panel-c"].hidden, false);

  els["tab-c"].dispatch("keydown", {key: "End", preventDefault() {}});
  assert.equal(els["panel-c"].hidden, false);

  controller.select("tab-b");
  assert.equal(els["panel-b"].hidden, false);
  assert.equal(els["panel-c"].hidden, true);
});

test("createLayout switches cast views, restores split when adding a hidden kind, and renders overviews", () => {
  const store = new Map();
  const buttons = ["split", "voices", "sounds"].map((view) => {
    const btn = makeElement({"data-view": view, "aria-pressed": "false"});
    btn.dataset.view = view;
    return btn;
  });
  const filter = makeElement();
  filter.querySelectorAll = () => buttons;

  const els = {
    "screen-script": makeElement(),
    "toggle-outline": makeElement(),
    "toggle-writer": makeElement(),
    "writer-tab-built": makeElement({"aria-controls": "writer-built"}),
    "writer-tab-paste": makeElement({"aria-controls": "writer-paste"}),
    "writer-built": makeElement({hidden: false}),
    "writer-paste": makeElement({hidden: true}),
    "mix-tab-line": makeElement({"aria-controls": "inspector"}),
    "mix-tab-sound": makeElement({"aria-controls": "mix-sound-panel"}),
    "inspector": makeElement({hidden: false}),
    "mix-sound-panel": makeElement({hidden: true}),
    "cast-view-filter": filter,
    "cast-voices-card": makeElement({hidden: false}),
    "cast-sounds-card": makeElement({hidden: false}),
    "overview-mode": makeElement(),
    "overview-lines": makeElement(),
    "overview-assets": makeElement(),
    "overview-duration": makeElement(),
    "cast-overview-voices": makeElement(),
    "cast-overview-sounds": makeElement(),
  };
  const $ = (id) => els[id] || null;
  const clock = (sec) => `${Math.floor(sec / 60)}:${String(Math.floor(sec % 60)).padStart(2, "0")}`;
  const layout = createLayout({$, shortMode: (m) => m, clock});

  withStorage({
    getItem: (key) => (store.has(key) ? store.get(key) : null),
    setItem: (key, value) => store.set(key, value),
  }, () => {
    layout.bind();
    layout.setCastView("voices");
    assert.equal(els["cast-voices-card"].hidden, false);
    assert.equal(els["cast-sounds-card"].hidden, true);
    assert.equal(buttons[1].getAttribute("aria-pressed"), "true");

    layout.ensureCastSection("sound");
    assert.equal(els["cast-voices-card"].hidden, false);
    assert.equal(els["cast-sounds-card"].hidden, false);
    assert.equal(buttons[0].getAttribute("aria-pressed"), "true");
  });

  const project = {
    lines: 12,
    script: {script_mode: "Advanced"},
    cast: [{ready: true}, {ready: false}],
    speakers: [{ready: true}, {ready: false}],
    sounds: [{ready: true}, {ready: true}],
    result: {duration: 75},
  };

  layout.renderScriptOverview(project, {lines: [{end: 62}], duration: 62, actual: false});
  assert.equal(els["overview-mode"].textContent, "Advanced");
  assert.equal(els["overview-lines"].textContent, "12");
  assert.equal(els["overview-assets"].textContent, "1/2");
  assert.equal(els["overview-assets"].dataset.state, "needs");
  assert.equal(els["overview-duration"].textContent, "≈ 1:02");

  layout.renderScriptOverview({...project, cast: [{ready: true}, {ready: true}]}, {lines: [{end: 75}], duration: 75, actual: true});
  assert.equal(els["overview-assets"].dataset.state, "ready");
  assert.equal(els["overview-duration"].textContent, "1:15");

  layout.renderScriptOverview({...project, result: null}, null);
  assert.equal(els["overview-duration"].textContent, "—");

  layout.renderCastOverview(project);
  assert.equal(els["cast-overview-voices"].textContent, "Voices 1/2");
  assert.equal(els["cast-overview-voices"].dataset.state, "needs");
  assert.equal(els["cast-overview-sounds"].textContent, "Sounds 2/2");
  assert.equal(els["cast-overview-sounds"].dataset.state, "ready");

  layout.selectMixTab("mix-tab-sound");
  assert.equal(els["inspector"].hidden, true);
  assert.equal(els["mix-sound-panel"].hidden, false);
});
