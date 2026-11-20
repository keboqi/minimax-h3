// Run with: node --test tests/storyboard_page.test.cjs
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const commonSource = fs.readFileSync(path.join(__dirname, "../h3_ui/page_static/common.js"), "utf8");
const source = fs.readFileSync(path.join(__dirname, "../h3_ui/storyboard/page/app.js"), "utf8");

for (const kind of ["picture", "video"]) {
  test(`a delayed ${kind} upload leaves the newly opened storyboard selected`, async () => {
    let finishUpload;
    let uploadUrl;
    const toast = {style: {}};
    const context = vm.createContext({
      FormData, Blob,
      document: {addEventListener() {}, getElementById: () => toast},
      window: {frameElement: null}, innerHeight: 1000,
      setTimeout() {}, clearTimeout() {},
      fetch: (url) => {
        uploadUrl = url;
        return new Promise((resolve) => { finishUpload = resolve; });
      },
    });
    // Expose the actual page actions without starting its DOM-dependent boot sequence.
    vm.runInContext(commonSource, context);
    vm.runInContext(source.replace(/  start\(\);\s*\}\)\(\);\s*$/,
      "  globalThis.page = {state, uploadMedia, uploadVideo};\n})();"), context);
    const page = context.page;
    page.state.board = {id: "board-A"};
    page.state.sceneId = "scene-A";
    const file = new Blob(["media"]);
    const pending = kind === "picture" ? page.uploadMedia("ref_images", [file]) : page.uploadVideo(file);
    assert.match(uploadUrl, /\/boards\/board-A\/scenes\/scene-A\//);

    const newlyOpened = {id: "board-B", scenes: [{id: "scene-B", prompt: "Keep my edits"}]};
    page.state.board = newlyOpened;
    page.state.sceneId = "scene-B";
    finishUpload({ok: true, json: async () => ({id: "board-A", scenes: [{id: "scene-A"}]})});
    await pending;

    assert.equal(page.state.board, newlyOpened);
    assert.equal(page.state.sceneId, "scene-B");
    assert.equal(page.state.board.scenes[0].prompt, "Keep my edits");
  });
}

function inspectorFixture(fetch) {
  const find = (root, matches) => matches(root) ? root
    : root?.children?.map((child) => find(child, matches)).find(Boolean);
  const h = (tag, props, ...children) => ({
    tag, props, ...props, style: {}, classList: {toggle() {}},
    textContent: children.filter((child) => typeof child === "string").join(""),
    children: children.flat(Infinity).filter(Boolean),
    replaceChildren(...nodes) { this.children = nodes; },
    replaceWith(node) { Object.assign(this, node); },
    setAttribute(key, value) { this.props[key] = value; },
    querySelector() { return find(this, (node) => node?.props?.["aria-pressed"] === "true"); },
    setSelectionRange(start, end) { this.selectionStart = start; this.selectionEnd = end; },
    focus() { this.focused = true; },
    showModal() { this.open = true; },
    close() { this.open = false; },
  });
  const inspector = h("aside", {});
  const library = h("dialog", {});
  const $ = (id) => ({inspector, library}[id] || find(inspector, (node) => node?.id === id));
  const context = vm.createContext({
    H3PageCommon: {$, h, visibleArea: () => ({top: 0, bottom: 800}), createToast: () => ({toast() {}})},
    document: {addEventListener() {}, querySelector: (selector) => selector === "#inspector textarea"
      ? find(inspector, (node) => node?.tag === "textarea") : null},
    setTimeout() {}, clearTimeout() {}, fetch,
  });
  vm.runInContext(source.replace(/  start\(\);\s*\}\)\(\);\s*$/,
    "  globalThis.page = {state, renderInspector, renderInspectorDerived, followRun, browseGlobalMedia};\n})();"), context);
  const page = context.page;
  const item = {
    id: "scene-A", number: 1, mode: "Reference media", prompt: "An opening shot", seconds: 5,
    issues: [], takes: [], pictures: [{tag: "image 1 = <Picture 1>"}], tags: ["image 1 = <Picture 1>"],
    cast: [], ref_images: [{name: "Sample image", path: "/sample.png"}], ref_audios: [], ref_videos: [],
  };
  Object.assign(page.state, {
    board: {id: "board-A", scenes: [item], cast: []}, sceneId: item.id,
    boot: {gpu: true, seconds: [3, 12], overlaps: [4], max_takes: 1},
  });
  return {page, item, $, find, inspector, library};
}

test("the inspector Generate button follows readiness, takes, and run state", () => {
  const {page, item, $} = inspectorFixture();
  item.issues = ["Write a prompt"];
  page.renderInspector();
  const generate = $("inspector-generate-scene");
  assert.equal(generate.disabled, true);

  item.issues = [];
  page.renderInspectorDerived();
  assert.equal($("inspector-generate-scene"), generate);
  assert.equal(generate.disabled, false);

  page.followRun({id: "run-A", active: true});
  assert.equal(generate.disabled, true);
  item.takes = [{id: "take-A"}];
  page.followRun(null);
  assert.equal(generate.disabled, false);
  assert.equal(generate.textContent, "Generate another take");
});

test("a picture tag in Media & Cast opens the prompt and inserts the tag", () => {
  const {page, item, find, inspector} = inspectorFixture();
  page.state.inspectorTab = "media";
  page.renderInspector();
  find(inspector, (node) => node?.tag === "button" && node.textContent === "<Picture 1>").onclick();

  const textarea = find(inspector, (node) => node?.tag === "textarea");
  assert.equal(page.state.inspectorTab, "prompt");
  assert.equal(item.prompt, "An opening shot <Picture 1>");
  assert.equal(textarea.value, item.prompt);
  assert.equal(textarea.focused, true);
  assert.equal(textarea.selectionStart, item.prompt.length);
});

test("global Media only attaches an image after explicit confirmation for a reference scene", async () => {
  const writes = [];
  const {page, item, find, library} = inspectorFixture(async (url, {method, body}) => {
    if (method === "PUT") {
      writes.push({url, body: JSON.parse(body)});
      return {ok: true, json: async () => null};
    }
    return {ok: true, json: async () => ({total: 1, items: [{caption: "Sample image", path: "/sample.png"}]})};
  });
  const load = () => new Promise(setImmediate);
  item.mode = "Text to video";
  page.browseGlobalMedia();
  await load();
  assert.ok(find(library, (node) => node?.tag === "img" && node.alt === "Sample image"));
  assert.equal(find(library, (node) => node?.textContent === "Add image to Scene 1"), undefined);
  library.close();

  item.mode = "Reference media";
  page.browseGlobalMedia();
  await load();
  const attach = find(library, (node) => node?.textContent === "Add image to Scene 1");
  const thumbnail = find(library, (node) => node?.props?.["aria-label"] === "Sample image");
  assert.equal(attach.disabled, true);
  thumbnail.onclick({currentTarget: thumbnail});
  assert.equal(attach.disabled, false);
  assert.equal(library.open, true);
  assert.deepEqual(writes, []);

  attach.onclick();
  assert.equal(library.open, false);
  assert.deepEqual(writes, [{url: "/storyboard/api/boards/board-A/scenes/scene-A/media/ref_images", body: {source: "/sample.png"}}]);
});
