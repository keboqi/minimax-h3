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

test("missing video posters fall back to a deduplicated client-side frame and hide broken icons", async () => {
  const findAll = (root, matches, out = []) => {
    if (!root || typeof root !== "object") return out;
    if (matches(root)) out.push(root);
    for (const child of root.children || []) findAll(child, matches, out);
    return out;
  };
  const h = (tag, props = {}, ...children) => ({
    tag, props, ...props, style: {}, classList: {toggle() {}},
    children: children.flat(Infinity).filter(Boolean),
    replaceChildren(...nodes) { this.children = nodes.filter(Boolean); },
    getAttribute(key) { return this[key] ?? this.props?.[key] ?? null; },
    removeAttribute(key) { delete this[key]; },
    querySelectorAll() { return []; },
  });
  const strip = h("div", {id: "strip"});
  const stripHeading = h("span", {id: "strip-heading"});
  const takeRow = h("div", {id: "take-row"});
  const takeConfirm = h("div", {id: "take-confirm"});
  const takeReview = h("div", {id: "take-review"});
  const takesPanel = h("div", {id: "takes-panel"});
  const stageVideo = h("video", {src: "/storyboard/api/boards/b1/file?path=t1.mp4", poster: "/video-posters?path=t1.mp4"});
  const nodes = {
    strip, "strip-heading": stripHeading, "take-row": takeRow,
    "take-confirm": takeConfirm, "take-review": takeReview, "takes-panel": takesPanel,
  };
  const createdVideos = [];
  const context = vm.createContext({
    H3PageCommon: {
      $: (id) => nodes[id], h,
      visibleArea: () => ({top: 0, bottom: 800}),
      createToast: () => ({toast() {}}),
      scoreTone: () => "", decisionScoreChips: () => [], llmReviewNodes: () => [], copyToClipboard() {},
    },
    document: {
      addEventListener() {},
      querySelector: (selector) => (selector === "#preview video" ? stageVideo : null),
      createElementNS(_ns, tag) {
        const node = h(tag);
        node.setAttribute = (k, v) => { node.props[k] = v; };
        node.append = (...kids) => { node.children.push(...kids); };
        return node;
      },
      createElement(tag) {
        if (tag === "video") {
          const v = {
            duration: 10, currentTime: 0, readyState: 2, seeking: false,
            videoWidth: 960, videoHeight: 540,
            pause() {}, load() {}, removeAttribute(key) { delete this[key]; },
          };
          Object.defineProperty(v, "src", {
            get() { return this._src; },
            set(value) {
              this._src = value;
              createdVideos.push(v);
              setImmediate(() => {
                if (value.includes("broken.mp4")) {
                  v.onerror?.();
                  return;
                }
                v.onloadedmetadata?.();
                if (v.currentTime >= 0.1) v.onseeked?.();
              });
            },
            configurable: true,
          });
          return v;
        }
        if (tag === "canvas") {
          return {
            width: 0, height: 0,
            getContext() { return {drawImage() {}}; },
            toDataURL() { return "data:image/jpeg;base64,FALLBACK_FRAME"; },
          };
        }
        return h(tag);
      },
    },
    setTimeout, clearTimeout,
  });
  vm.runInContext(source.replace(/  start\(\);\s*\}\)\(\);\s*$/,
    "  globalThis.page = {state, renderStrip, renderTakes};\n})();"), context);
  const page = context.page;
  const sceneItem = {
    id: "s1", number: 1, title: "Scene 1", seconds: 10, added_seconds: 10,
    flow: false, kind: "generated", state: "done", state_label: "Done", summary: "",
    thumb: "/video-posters?path=t1.mp4", in_use: "t1",
    takes: [
      {id: "t1", number: 1, seed: 1, video: "/storyboard/api/boards/b1/file?path=t1.mp4", poster: "/video-posters?path=t1.mp4", missing: false},
      {id: "t2", number: 2, seed: 2, video: "/storyboard/api/boards/b1/file?path=broken.mp4", poster: "/video-posters?path=broken.mp4", missing: false},
    ],
  };
  page.state.boot = {gpu: false, writer: {gemini_models: []}};
  page.state.board = {id: "b1", scenes: [sceneItem]};
  page.state.sceneId = "s1";

  page.renderStrip();
  page.renderTakes();
  const [stripImg] = findAll(strip, (n) => n.tag === "img");
  const [take1Img, take2Img] = findAll(takeRow, (n) => n.tag === "img");

  // Trigger 404 onerror on both the timeline scene thumb and take cards.
  stripImg.onerror();
  take1Img.onerror();
  take2Img.onerror();
  assert.equal(stripImg.hidden, true);
  assert.equal(take1Img.hidden, true);
  assert.equal(take2Img.hidden, true);

  await new Promise((resolve) => setTimeout(resolve, 25));

  // Shared video t1.mp4 is decoded only once; both strip & take cards and stage video poster update.
  assert.equal(createdVideos.filter((v) => v._src.includes("t1.mp4")).length, 1);
  assert.equal(stripImg.src, "data:image/jpeg;base64,FALLBACK_FRAME");
  assert.equal(stripImg.hidden, false);
  assert.equal(take1Img.src, "data:image/jpeg;base64,FALLBACK_FRAME");
  assert.equal(take1Img.hidden, false);
  assert.equal(stageVideo.poster, "data:image/jpeg;base64,FALLBACK_FRAME");
  // Unreadable video stays hidden instead of showing a broken image icon.
  assert.equal(take2Img.hidden, true);

  // Subsequent re-renders reuse the cached fallback directly and omit permanently failed imgs.
  page.renderStrip();
  page.renderTakes();
  const [stripImg2] = findAll(strip, (n) => n.tag === "img");
  const takeImgs2 = findAll(takeRow, (n) => n.tag === "img");
  assert.equal(stripImg2.src, "data:image/jpeg;base64,FALLBACK_FRAME");
  assert.equal(takeImgs2.length, 1);
  assert.equal(takeImgs2[0].src, "data:image/jpeg;base64,FALLBACK_FRAME");
});
