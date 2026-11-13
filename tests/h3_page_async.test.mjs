import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import vm from "node:vm";

const source = readFileSync(new URL("../h3_ui/h3/page/app.js", import.meta.url), "utf8").replace(/\r/g, "");

function page(send = async () => ({})) {
  const nodes = new Map(), calls = [];
  const node = (id) => {
    if (!nodes.has(id)) nodes.set(id, {
      value: "A new prompt", style: {}, listeners: {},
      addEventListener(name, callback) { this.listeners[name] = callback; },
    });
    return nodes.get(id);
  };
  const context = vm.createContext({
    document: {getElementById: node, addEventListener() {}},
    window: {frameElement: null}, innerHeight: 900,
    setTimeout() {}, clearTimeout() {},
    fetch: async (url, options) => {
      const path = url.replace("/h3/api", "");
      const body = options.body && JSON.parse(options.body);
      calls.push({path, body});
      const answer = await send(path, body);
      return {ok: true, json: async () => answer};
    },
  });
  // Run the actual page handlers without starting its browser layout or network boot.
  const script = source.replace("  const render = () => {", "  let render = () => {")
    .replace("\n  start();\n", `
  render = () => {};
  globalThis.page = {app, take, startRun, savePrompt};
  bind();
`);
  assert.notEqual(script, source);
  vm.runInContext(script, context);
  context.page.app.s = {
    values: {mode: "Text to video", width: 1280, height: 768, duration: 6}, media: {},
  };
  return {...context.page, node, calls};
}

test("a failed prompt save blocks generation and can be retried", async () => {
  let fail = true;
  const p = page(async (path) => {
    if (path === "/prompt" && fail) throw new Error("Prompt save failed");
    return {};
  });
  await p.startRun("/generate");
  assert.deepEqual(p.calls.map(({path}) => path), ["/prompt"]);
  assert.equal(p.node("toast").textContent, "Prompt save failed");

  fail = false;
  await p.startRun("/generate");
  assert.deepEqual(p.calls.map(({path}) => path), ["/prompt", "/prompt", "/generate"]);
  assert.equal(p.calls[1].body.prompt, "A new prompt");
});

test("prompt enhancement also stops on a save failure", async () => {
  const p = page(async () => { throw new Error("Offline"); });
  await p.node("enhance").listeners.click();
  assert.deepEqual(p.calls.map(({path}) => path), ["/prompt"]);
  assert.equal(p.node("toast").textContent, "Offline");
});

test("a blur autosave reports its failure without rejecting the event handler", async () => {
  const p = page(async () => { throw new Error("Offline"); });
  await p.node("prompt").listeners.blur();
  assert.equal(p.node("toast").textContent, "Offline");
  await assert.rejects(p.savePrompt(), /Offline/);
});

test("a settings response delivers a completed suggestion exactly once", () => {
  const p = page();
  p.app.enhanceContext = "original inputs";
  p.take({runs: {enhance: {id: "writer-1", active: true}}});
  const completed = {id: "writer-1", active: false, original: "An idea", suggestion: "A finished prompt"};
  // This settings response arrives while the poller is waiting for its next request.
  p.take({state: p.app.s, runs: {enhance: completed}});
  assert.equal(p.app.suggestion.text, "A finished prompt");
  assert.equal(p.app.suggestion.context, "original inputs");

  p.node("keep").listeners.click();
  p.take({runs: {enhance: completed}});
  assert.equal(p.app.suggestion, null);
  assert.equal(p.app.enhanceNote, "Original prompt kept.");

  p.app.enhanceContext = "new inputs";
  p.take({runs: {enhance: {...completed, id: "writer-2", suggestion: "Another prompt"}}});
  assert.equal(p.app.suggestion.text, "Another prompt");
  assert.equal(p.app.suggestion.context, "new inputs");
});

test("a writer that finishes before its start response still delivers its suggestion", async () => {
  const p = page(async (path) => path === "/enhance"
    ? {runs: {enhance: {id: "fast-writer", active: false, original: "A new prompt", suggestion: "Enhanced"}}}
    : {});
  await p.node("enhance").listeners.click();
  assert.equal(p.app.suggestion.text, "Enhanced");
  assert.equal(p.app.suggestion.original, "A new prompt");
});

test("reopening a completed run does not resurrect a suggestion without its draft context", () => {
  const p = page();
  const completed = {id: "old-writer", active: false, suggestion: "An old suggestion"};
  p.take({runs: {enhance: completed}});
  assert.equal(p.app.suggestion, null);
  p.app.enhanceContext = "new inputs";
  p.take({runs: {enhance: completed}});
  assert.equal(p.app.suggestion, null);
});
