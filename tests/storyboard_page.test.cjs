// Run with: node --test tests/storyboard_page.test.cjs
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const commonSource = fs.readFileSync(path.join(__dirname, "../h3_ui/page_static/common.js"), "utf8");
const source = fs.readFileSync(path.join(__dirname, "../h3_ui/storyboard/page/app.js"), "utf8");

for (const kind of ["picture", "video", "upload-take", "library-take"]) {
  test(`a delayed ${kind} upload leaves the newly opened storyboard selected`, async () => {
    let finishUpload;
    let uploadUrl;
    const toast = {style: {}};
    const nodeStub = () => ({style: {}, children: [], replaceChildren(...n) { this.children = n; }, setAttribute() {}, addEventListener() {}, append(...n) { this.children.push(...n); }, classList: {toggle() {}}});
    const nodes = {toast, "take-row": nodeStub(), "take-confirm": nodeStub(), "take-review": nodeStub(), "takes-panel": nodeStub()};
    const context = vm.createContext({
      FormData, Blob,
      document: {addEventListener() {}, getElementById: (id) => nodes[id] || nodeStub(), createElement: () => nodeStub(), createTextNode: () => nodeStub()},
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
      "  globalThis.page = {state, uploadMedia, uploadVideo, uploadTake, libraryTake};\n})();"), context);
    const page = context.page;
    page.state.boot = {gpu: false, library: true, writer: {gemini_models: []}};
    page.state.board = {id: "board-A", scenes: [{id: "scene-A", kind: "generated", takes: [], in_use: null, overlap: 39}]};
    page.state.sceneId = "scene-A";
    const file = new Blob(["media"]);
    const pending = kind === "picture" ? page.uploadMedia("ref_images", [file])
      : kind === "video" ? page.uploadVideo(file)
      : kind === "upload-take" ? page.uploadTake(file)
      : page.libraryTake("/media/clip.mp4");
    await new Promise(setImmediate);
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

test("renderTakes offers upload and Media library take actions and renders manual take metadata", () => {
  const findAll = (root, matches, out = []) => {
    if (!root || typeof root !== "object") return out;
    if (matches(root)) out.push(root);
    for (const child of root.children || []) findAll(child, matches, out);
    return out;
  };
  const h = (tag, props = {}, ...children) => ({
    tag, props, ...props, style: {}, classList: {toggle() {}},
    textContent: children.filter((c) => typeof c === "string").join(""),
    children: children.flat(Infinity).filter(Boolean),
    replaceChildren(...nodes) { this.children = nodes.filter(Boolean); },
  });
  const takeRow = h("div", {id: "take-row"});
  const takeConfirm = h("div", {id: "take-confirm"});
  const takeReview = h("div", {id: "take-review"});
  const takesPanel = h("div", {id: "takes-panel"});
  const nodes = {"take-row": takeRow, "take-confirm": takeConfirm, "take-review": takeReview, "takes-panel": takesPanel};
  const context = vm.createContext({
    H3PageCommon: {
      $: (id) => nodes[id], h,
      visibleArea: () => ({top: 0, bottom: 800}),
      createToast: () => ({toast() {}}),
      scoreTone: () => "", decisionScoreChips: () => [], llmReviewNodes: () => [], copyToClipboard() {},
    },
    document: {addEventListener() {}},
    setTimeout() {}, clearTimeout() {},
  });
  vm.runInContext(source.replace(/  start\(\);\s*\}\)\(\);\s*$/,
    "  globalThis.page = {state, renderTakes};\n})();"), context);
  const page = context.page;
  const emptyScene = {id: "s1", number: 1, kind: "generated", flow: true, overlap: 39, takes: [], in_use: null};
  page.state.boot = {gpu: false, library: false, writer: {gemini_models: []}};
  page.state.board = {id: "b1", scenes: [emptyScene]};
  page.state.sceneId = "s1";

  page.renderTakes();
  assert.equal(takesPanel.hidden, false);
  assert.ok(findAll(takeRow, (n) => n.id === "upload-take").length === 1);
  assert.ok(findAll(takeRow, (n) => n.id === "library-take").length === 0);

  page.state.boot.library = true;
  emptyScene.takes = [
    {id: "t1", number: 1, source: "upload", source_name: "edit.mp4", duration: 4.5, frames: 108, prefix_frames: 39, missing: false},
    {id: "t2", number: 2, source: "library", source_name: "clip.mp4", duration: 5.0, frames: 120, prefix_frames: 0, missing: false},
  ];
  emptyScene.in_use = "t1";
  page.renderTakes();
  assert.ok(findAll(takeRow, (n) => n.id === "library-take").length === 1);
  const labels = findAll(takeRow, (n) => n.className?.includes("take-card-meta")).map((n) => n.children.map((c) => c.textContent).join(" "));
  assert.deepEqual(labels, ["Take 1 Uploaded", "Take 2 Media"]);
  assert.equal(findAll(takeRow, (n) => n.textContent === "Use seed").length, 0);
  const overlapCheck = findAll(takeRow, (n) => n.tag === "input" && n.type === "checkbox")[0];
  assert.ok(overlapCheck);
  assert.equal(overlapCheck.checked, true);
});

test("settingsPanel renders the auto_score checkbox and saves changes to /score-settings", async () => {
  const find = (root, matches) => matches(root) ? root
    : root?.children?.map((child) => find(child, matches)).find(Boolean);
  const h = (tag, props = {}, ...children) => ({
    tag, props, ...props, style: {}, classList: {toggle() {}},
    textContent: children.filter((c) => typeof c === "string").join(""),
    children: children.flat(Infinity).filter(Boolean),
    replaceChildren(...nodes) { this.children = nodes.filter(Boolean); },
    append(...nodes) { this.children.push(...nodes); },
    replaceWith(node) { Object.assign(this, node); },
    setAttribute(key, value) { this.props[key] = value; },
    querySelector() { return null; },
    querySelectorAll() { return []; },
  });
  const nodes = {inspector: h("aside", {id: "inspector"}), library: h("dialog", {id: "library"})};
  const $ = (id) => {
    if (!nodes[id]) nodes[id] = find(nodes.inspector, (node) => node?.id === id) || h("div", {id});
    return nodes[id];
  };
  const requests = [];
  const context = vm.createContext({
    H3PageCommon: {
      $, h,
      visibleArea: () => ({top: 0, bottom: 800}),
      createToast: () => ({toast() {}}),
      scoreTone: () => "",
      decisionScoreChips: () => [],
      llmReviewNodes: () => [],
      copyToClipboard() {},
    },
    document: {
      addEventListener() {},
      querySelector: () => null,
      querySelectorAll: () => [],
      createElementNS: () => h("svg"),
    },
    setTimeout() {}, clearTimeout() {},
    fetch: async (url, {method, body}) => {
      const parsed = JSON.parse(body);
      requests.push({url, method, body: parsed});
      return {ok: true, json: async () => ({...page.state.board, auto_score: parsed.auto_score})};
    },
  });
  vm.runInContext(source.replace(/  start\(\);\s*\}\)\(\);\s*$/,
    "  globalThis.page = {state, renderInspector};\n})();"), context);
  const page = context.page;
  const item = {
    id: "scene-A", number: 1, title: "Opening", mode: "Text to video", prompt: "Shot",
    seconds: 5, added_seconds: 5, flow: false, kind: "generated", state: "draft", state_label: "Not generated",
    issues: [], takes: [], in_use: null, cast: [], ref_images: [], ref_audios: [], ref_videos: [],
  };
  page.state.boot = {
    gpu: true, seconds: [5, 15], overlaps: [39], max_takes: 4,
    boards: [{id: "board-A", name: "Board A", generated: false}],
    subtitle_languages: ["Spanish"], subtitle_models: ["gemma"],
    writer: {gemini_models: ["gemini-2.5-flash"], gemini_default: "gemini-2.5-flash", openai_models: ["gpt-4o"]},
  };
  page.state.board = {
    id: "board-A", name: "Board A", total: "0:05", done: 0, pending: 1, queue: [],
    auto_subtitles: false, auto_score: false, subtitle_language: "Spanish",
    scenes: [item], cast: [], versions: [],
  };
  page.state.sceneId = item.id;
  page.state.panel = "settings";
  page.renderInspector();

  const label = find(nodes.inspector, (node) => node?.tag === "label" && node.textContent?.includes("Automatically score each take in the background after generation"));
  assert.ok(label);
  const checkbox = find(label, (node) => node?.tag === "input" && node.type === "checkbox");
  assert.ok(checkbox);
  assert.equal(checkbox.checked, false);

  checkbox.checked = true;
  await checkbox.onchange({target: checkbox});
  assert.deepEqual(requests, [{
    url: "/storyboard/api/boards/board-A/score-settings",
    method: "PUT",
    body: {auto_score: true},
  }]);
  assert.equal(page.state.board.auto_score, true);
});

test("renderTakes shows pending pill, polls when another take is pending, and reports failed auto score", () => {
  const findAll = (root, matches, out = []) => {
    if (!root || typeof root !== "object") return out;
    if (matches(root)) out.push(root);
    for (const child of root.children || []) findAll(child, matches, out);
    return out;
  };
  const h = (tag, props = {}, ...children) => ({
    tag, props, ...props, style: {}, classList: {toggle() {}},
    textContent: children.filter((c) => typeof c === "string").join(""),
    children: children.flat(Infinity).filter(Boolean),
    replaceChildren(...nodes) { this.children = nodes.filter(Boolean); },
  });
  const takeRow = h("div", {id: "take-row"});
  const takeConfirm = h("div", {id: "take-confirm"});
  const takeReview = h("div", {id: "take-review"});
  const takesPanel = h("div", {id: "takes-panel"});
  const nodes = {"take-row": takeRow, "take-confirm": takeConfirm, "take-review": takeReview, "takes-panel": takesPanel};
  const scheduled = [];
  const context = vm.createContext({
    H3PageCommon: {
      $: (id) => nodes[id], h,
      visibleArea: () => ({top: 0, bottom: 800}),
      createToast: () => ({toast() {}}),
      scoreTone: () => "good",
      decisionScoreChips: (review) => [h("span", {className: "review-chip"}, `Take score ${review.score}`)],
      llmReviewNodes: () => [],
      copyToClipboard() {},
    },
    document: {addEventListener() {}},
    setTimeout(fn, delay) { scheduled.push(delay); return scheduled.length; },
    clearTimeout() {},
  });
  vm.runInContext(source.replace(/  start\(\);\s*\}\)\(\);\s*$/,
    "  globalThis.page = {state, renderTakes};\n})();"), context);
  const page = context.page;
  const sceneItem = {
    id: "s1", number: 1, kind: "generated", flow: false, overlap: 39, in_use: "t1",
    takes: [
      {id: "t1", number: 1, seed: 1, missing: false, review: {status: "failed", error: "Decision error"}},
      {id: "t2", number: 2, seed: 2, missing: false, review: {status: "pending"}},
    ],
  };
  page.state.boot = {gpu: false, library: false, writer: {gemini_models: []}};
  page.state.board = {id: "b1", scenes: [sceneItem]};
  page.state.sceneId = "s1";

  page.renderTakes();
  const pills = findAll(takeRow, (n) => n.className?.includes("take-score-pill"));
  assert.deepEqual(pills.map((p) => p.textContent), ["…"]);
  assert.ok(findAll(takeRow, (n) => n.textContent === "Auto score failed.").length === 1);
  assert.deepEqual(scheduled, [4000]);
});

test("decisionScoreRadar builds an interactive SVG radar chart for multi-aspect scores", () => {
  const makeNode = (tag, ns = null) => {
    const classes = new Set();
    const listeners = new Map();
    const node = {
      tag, ns, attrs: {}, children: [],
      classList: {
        toggle(name, force) {
          const add = force === undefined ? !classes.has(name) : Boolean(force);
          if (add) classes.add(name);
          else classes.delete(name);
          node.className = [...classes].join(" ");
          return add;
        },
        contains(name) { return classes.has(name); },
      },
      setAttribute(key, value) {
        this.attrs[key] = String(value);
        if (key === "class") {
          classes.clear();
          for (const part of String(value).split(/\s+/).filter(Boolean)) classes.add(part);
          node.className = [...classes].join(" ");
        }
      },
      addEventListener(type, fn) {
        if (!listeners.has(type)) listeners.set(type, []);
        listeners.get(type).push(fn);
      },
      dispatch(type) {
        for (const fn of listeners.get(type) || []) fn({currentTarget: node, target: node});
      },
      append(...items) {
        for (const item of items) {
          if (item === null || item === undefined || item === false) continue;
          const child = typeof item === "string" ? {textContent: item} : item;
          this.children.push(child);
        }
      },
    };
    Object.defineProperty(node, "className", {
      get() { return [...classes].join(" "); },
      set(v) {
        classes.clear();
        for (const part of String(v || "").split(/\s+/).filter(Boolean)) classes.add(part);
      },
      configurable: true,
    });
    return node;
  };
  const findAll = (root, matches, out = []) => {
    if (!root || typeof root !== "object") return out;
    if (matches(root)) out.push(root);
    for (const child of root.children || []) findAll(child, matches, out);
    return out;
  };
  const context = vm.createContext({
    document: {
      createElement: (tag) => makeNode(tag),
      createElementNS: (ns, tag) => makeNode(tag, ns),
      createTextNode: (text) => ({textContent: String(text)}),
    },
    window: {frameElement: null},
    innerHeight: 900,
  });
  vm.runInContext(commonSource, context);
  const {decisionScoreRadar, scoreAxes} = context.H3PageCommon;

  assert.equal(decisionScoreRadar({status: "done", fit: 75, scores: {prompt: 75, visuals: 80}}), null);

  const review = {
    status: "done",
    score: 78,
    score_label: "Good take",
    ref_count: 2,
    advice: "Keep warm lighting",
    tags: ["close-up", "dialogue"],
    scores: {
      prompt: 80, visuals: 90, style: 85, camera: 75,
      action: 80, continuity: 65, audio: 75, speech: 55,
      references: 70, advice: 60,
    },
  };
  assert.equal(scoreAxes(review).length, 10);

  const card = decisionScoreRadar(review, {chipClass: "review-chip", includeTags: true, includeHeadline: true});
  assert.ok(card);
  const rings = findAll(card, (n) => n.tag === "circle" && n.classList?.contains("radar-ring"));
  const spokes = findAll(card, (n) => n.tag === "line" && n.classList?.contains("radar-spoke"));
  const polygons = findAll(card, (n) => n.tag === "polygon" && n.classList?.contains("radar-area"));
  const axisGroups = findAll(card, (n) => n.tag === "g" && n.classList?.contains("radar-axis"));
  const labels = findAll(card, (n) => n.tag === "text" && n.classList?.contains("radar-label"));
  const hits = findAll(card, (n) => n.tag === "circle" && n.classList?.contains("radar-hit"));
  const listItems = findAll(card, (n) => n.classList?.contains("score-radar-item"));
  const readout = findAll(card, (n) => n.classList?.contains("score-radar-readout"))[0];
  const tags = findAll(card, (n) => n.classList?.contains("score-radar-tags"))[0];

  assert.equal(rings.length, 5);
  assert.equal(spokes.length, 10);
  assert.equal(polygons.length, 1);
  assert.equal(polygons[0].attrs.points.split(" ").length, 10);
  assert.equal(axisGroups.length, 10);
  assert.deepEqual(labels.map((l) => l.children[0].textContent),
    ["Prompt", "Visuals", "Style", "Camera", "Action", "Continuity", "Audio", "Speech", "Refs", "Advice"]);
  assert.equal(listItems.length, 10);
  assert.equal(tags.children.length, 2);

  // Hovering/focusing an SVG hit target highlights both the axis group and the side list item.
  hits[2].dispatch("pointerenter");
  assert.equal(axisGroups[2].classList.contains("is-active"), true);
  assert.equal(listItems[2].classList.contains("is-active"), true);
  assert.equal(readout.children[0].textContent, "85");
  assert.equal(readout.children[1].textContent, "Style");
  hits[2].dispatch("pointerleave");
  assert.equal(axisGroups[2].classList.contains("is-active"), false);
  assert.equal(readout.children[1].textContent, "Hover or focus any aspect");
});

test("renderTakes shows the radar chart under take-detail and keeps take-detail compact", () => {
  const findAll = (root, matches, out = []) => {
    if (!root || typeof root !== "object") return out;
    if (matches(root)) out.push(root);
    for (const child of root.children || []) findAll(child, matches, out);
    return out;
  };
  const h = (tag, props = {}, ...children) => ({
    tag, props, ...props, style: {}, classList: {toggle() {}, contains(c) { return (this.className || "").split(/\s+/).includes(c); }},
    textContent: children.filter((c) => typeof c === "string").join(""),
    children: children.flat(Infinity).filter(Boolean),
    replaceChildren(...nodes) { this.children = nodes.filter(Boolean); },
  });
  const takeRow = h("div", {id: "take-row"});
  const takeConfirm = h("div", {id: "take-confirm"});
  const takeReview = h("div", {id: "take-review"});
  const takesPanel = h("div", {id: "takes-panel"});
  const nodes = {"take-row": takeRow, "take-confirm": takeConfirm, "take-review": takeReview, "takes-panel": takesPanel};
  const context = vm.createContext({
    H3PageCommon: {
      $: (id) => nodes[id], h,
      visibleArea: () => ({top: 0, bottom: 800}),
      createToast: () => ({toast() {}}),
      scoreTone: () => "good",
      decisionScoreChips: (review, {includeBreakdown = true} = {}) => includeBreakdown
        ? [h("span", {className: "review-chip"}, `Take score ${review.score}`), h("span", {className: "review-chip"}, "Visuals 90")]
        : [h("span", {className: "review-chip"}, `Take score ${review.score}`)],
      decisionScoreRadar: () => h("div", {className: "score-radar-card"}, "RADAR"),
      llmReviewNodes: () => [],
      copyToClipboard() {},
    },
    document: {addEventListener() {}},
    setTimeout() {}, clearTimeout() {},
  });
  vm.runInContext(source.replace(/  start\(\);\s*\}\)\(\);\s*$/,
    "  globalThis.page = {state, renderTakes};\n})();"), context);
  const page = context.page;
  const sceneItem = {
    id: "s1", number: 1, kind: "generated", flow: false, overlap: 39, in_use: "t1",
    takes: [{
      id: "t1", number: 1, seed: 1, missing: false,
      review: {status: "done", score: 80, scores: {prompt: 80, visuals: 90, style: 85, camera: 75, action: 80, continuity: 65, audio: 75, speech: 55}},
    }],
  };
  page.state.boot = {gpu: false, library: false, writer: {gemini_models: []}};
  page.state.board = {id: "b1", scenes: [sceneItem]};
  page.state.sceneId = "s1";

  page.renderTakes();
  const detail = findAll(takeRow, (n) => n.className === "take-detail")[0];
  const radarCards = findAll(takeRow, (n) => n.className === "score-radar-card");
  assert.equal(radarCards.length, 1);
  assert.equal(findAll(detail, (n) => n.className === "review-chip").length, 1);
});

test("scoreLevel maps scores into five distinct color levels", () => {
  const context = vm.createContext({
    document: {createElement: () => ({}), createElementNS: () => ({}), createTextNode: () => ({})},
    window: {frameElement: null},
    innerHeight: 900,
  });
  vm.runInContext(commonSource, context);
  const {scoreLevel} = context.H3PageCommon;

  assert.equal(scoreLevel(95), "strong");
  assert.equal(scoreLevel(85), "strong");
  assert.equal(scoreLevel(84), "good");
  assert.equal(scoreLevel(70), "good");
  assert.equal(scoreLevel(69), "fair");
  assert.equal(scoreLevel(55), "fair");
  assert.equal(scoreLevel(54), "weak");
  assert.equal(scoreLevel(40), "weak");
  assert.equal(scoreLevel(39), "poor");
  assert.equal(scoreLevel(0), "poor");
  assert.equal(scoreLevel(null), "");
  assert.equal(scoreLevel(undefined), "");
  assert.equal(scoreLevel("bad"), "");
});

function stageStripFixture({fetch, gpu = false} = {}) {
  const findAll = (root, matches, out = []) => {
    if (!root || typeof root !== "object") return out;
    if (matches(root)) out.push(root);
    for (const child of root.children || []) findAll(child, matches, out);
    return out;
  };
  const h = (tag, props = {}, ...children) => ({
    tag, props, ...props, style: {}, classList: {toggle() {}, add() {}, remove() {}},
    textContent: children.filter((c) => typeof c === "string" || typeof c === "number").map(String).join(""),
    children: children.flat(Infinity).filter(Boolean),
    replaceChildren(...nodes) { this.children = nodes.filter(Boolean); },
    append(...nodes) { this.children.push(...nodes.flat(Infinity).filter(Boolean)); },
    setAttribute(k, v) { this.props[k] = v; },
    getAttribute(k) { return this[k] ?? this.props?.[k] ?? null; },
    querySelectorAll() { return []; },
  });
  const nodes = {
    strip: h("div", {id: "strip"}),
    "strip-heading": h("span", {id: "strip-heading"}),
    "strip-actions": h("div", {id: "strip-actions"}),
    "take-row": h("div", {id: "take-row"}),
    "take-confirm": h("div", {id: "take-confirm"}),
    "take-review": h("div", {id: "take-review"}),
    "takes-panel": h("div", {id: "takes-panel"}),
    "subtitle-language": h("select", {id: "subtitle-language"}),
    "show-subtitles": h("input", {id: "show-subtitles"}),
    "show-translation": h("input", {id: "show-translation"}),
    "translate-scene": h("button", {id: "translate-scene"}),
    "translate-board": h("button", {id: "translate-board"}),
    "export-translation": h("input", {id: "export-translation"}),
    "burn-subtitles": h("input", {id: "burn-subtitles"}),
    "subtitle-status": h("p", {id: "subtitle-status"}),
    "stage-head": h("div", {id: "stage-head"}),
    preview: h("div", {id: "preview"}),
    "stage-notes": h("div", {id: "stage-notes"}),
    topbar: h("header", {id: "topbar"}),
    inspector: h("aside", {id: "inspector"}),
    editor: h("div", {id: "editor", hidden: false}),
    toast: h("div", {id: "toast"}),
  };
  const scheduled = [];
  const toasts = [];
  const context = vm.createContext({
    H3PageCommon: {
      $: (id) => nodes[id], h,
      visibleArea: () => ({top: 0, bottom: 800}),
      createToast: () => ({toast(msg, err) { toasts.push({msg, err: Boolean(err)}); }}),
      scoreTone: (v) => (v >= 70 ? "good" : v < 45 ? "warn" : ""),
      scoreLevel: (v) => (v >= 85 ? "strong" : v >= 70 ? "good" : v >= 55 ? "fair" : v >= 40 ? "weak" : "poor"),
      decisionScoreChips: (review) => [h("span", {className: "review-chip"}, `Take score ${review.score}`)],
      decisionScoreRadar: () => null,
      llmReviewNodes: () => [],
      copyToClipboard() {},
    },
    document: {
      addEventListener() {},
      querySelector: () => null,
      querySelectorAll: () => [],
      createElementNS: () => h("svg"),
    },
    setTimeout(fn, delay) { scheduled.push(delay); return scheduled.length; },
    clearTimeout() {},
    fetch,
  });
  vm.runInContext(source.replace(/  start\(\);\s*\}\)\(\);\s*$/,
    "  globalThis.page = {state, renderStrip, renderStripActions, renderTakes};\n})();"), context);
  const page = context.page;
  page.state.boot = {
    gpu, library: false, subtitle_languages: ["Spanish"], subtitle_models: ["gemma"],
    boards: [{id: "b1", name: "Board", generated: true}],
    writer: {gemini_models: []},
  };
  return {page, nodes, findAll, scheduled, toasts};
}

test("renderStrip shows level-colored score badges on scene timeline thumbnails", () => {
  const {page, nodes, findAll} = stageStripFixture();
  page.state.board = {
    id: "b1",
    scenes: [
      {id: "s1", number: 1, title: "One", seconds: 5, added_seconds: 5, flow: false, kind: "generated", state: "done", state_label: "Done", summary: "", in_use: "t1",
        takes: [{id: "t1", number: 1, missing: false, review: {status: "done", score: 90, score_label: "Strong take"}}]},
      {id: "s2", number: 2, title: "Two", seconds: 5, added_seconds: 5, flow: false, kind: "generated", state: "done", state_label: "Done", summary: "", in_use: "t2",
        takes: [{id: "t2", number: 1, missing: false, review: {status: "done", score: 75, score_label: "Good take"}}]},
      {id: "s3", number: 3, title: "Three", seconds: 5, added_seconds: 5, flow: false, kind: "generated", state: "done", state_label: "Done", summary: "", in_use: "t3",
        takes: [{id: "t3", number: 1, missing: false, review: {status: "done", score: 60}}]},
      {id: "s4", number: 4, title: "Four", seconds: 5, added_seconds: 5, flow: false, kind: "generated", state: "done", state_label: "Done", summary: "", in_use: "t4",
        takes: [{id: "t4", number: 1, missing: false, review: {status: "done", score: 45}}]},
      {id: "s5", number: 5, title: "Five", seconds: 5, added_seconds: 5, flow: false, kind: "generated", state: "done", state_label: "Done", summary: "", in_use: "t5",
        takes: [{id: "t5", number: 1, missing: false, review: {status: "done", score: 25}}]},
      {id: "s6", number: 6, title: "Six", seconds: 5, added_seconds: 5, flow: false, kind: "generated", state: "done", state_label: "Done", summary: "", in_use: "t6",
        takes: [{id: "t6", number: 1, missing: false, review: {status: "pending"}}]},
      {id: "s7", number: 7, title: "Seven", seconds: 5, added_seconds: 5, flow: false, kind: "generated", state: "done", state_label: "Done", summary: "", in_use: "t7",
        takes: [{id: "t7", number: 1, missing: false, review: null}]},
      {id: "s8", number: 8, title: "Clip", seconds: 5, added_seconds: 5, flow: false, kind: "media", state: "done", state_label: "Done", summary: "", in_use: "t8",
        takes: [{id: "t8", number: 1, missing: false, review: {status: "done", score: 88}}]},
    ],
  };
  page.state.sceneId = "s1";
  page.renderStrip();

  const frames = findAll(nodes.strip, (n) => n.className === "frame");
  assert.equal(frames.length, 8);
  const badgeFor = (frame) => findAll(frame, (n) => n.className?.includes("thumb-score"))[0];
  assert.equal(badgeFor(frames[0]).textContent, "90");
  assert.ok(badgeFor(frames[0]).className.includes("score-level-strong"));
  assert.equal(badgeFor(frames[1]).textContent, "75");
  assert.ok(badgeFor(frames[1]).className.includes("score-level-good"));
  assert.equal(badgeFor(frames[2]).textContent, "60");
  assert.ok(badgeFor(frames[2]).className.includes("score-level-fair"));
  assert.equal(badgeFor(frames[3]).textContent, "45");
  assert.ok(badgeFor(frames[3]).className.includes("score-level-weak"));
  assert.equal(badgeFor(frames[4]).textContent, "25");
  assert.ok(badgeFor(frames[4]).className.includes("score-level-poor"));
  assert.equal(badgeFor(frames[5]).textContent, "…");
  assert.equal(badgeFor(frames[6]), undefined);
  assert.equal(badgeFor(frames[7]), undefined);
});

test("renderTakes and renderStripActions wire batch scoring and take thumbnail score levels", async () => {
  const calls = [];
  const {page, nodes, findAll, scheduled} = stageStripFixture({
    gpu: false,
    fetch: async (url, options = {}) => {
      calls.push({url, method: options.method || "GET", body: options.body ? JSON.parse(options.body) : null});
      return {
        ok: true,
        json: async () => ({
          ...page.state.board,
          scheduled: 2, already_scored: 1, pending: 1, skipped: 0,
        }),
      };
    },
  });
  page.state.board = {
    id: "b1", name: "Board", total: "0:15", done: 3, pending: 0, queue: [],
    subtitle_language: "Spanish", take_note: "", cast: [], versions: [],
    scenes: [
      {
        id: "s1", number: 1, title: "One", seconds: 5, added_seconds: 5, flow: false, kind: "generated",
        state: "done", state_label: "Done", summary: "", issues: [], in_use: "t2",
        takes: [
          {id: "t1", number: 1, missing: false, review: {status: "done", score: 88}},
          {id: "t2", number: 2, missing: false, review: null},
          {id: "t3", number: 3, missing: false, review: {status: "failed", error: "Oops"}},
          {id: "t4", number: 4, missing: false, review: {status: "pending"}},
          {id: "t5", number: 5, missing: true, review: null},
        ],
      },
      {
        id: "s2", number: 2, title: "Two", seconds: 5, added_seconds: 5, flow: false, kind: "generated",
        state: "done", state_label: "Done", summary: "", issues: [], in_use: "t20",
        takes: [{id: "t20", number: 1, missing: false, review: {status: "failed", error: "Err"}}],
      },
      {
        id: "s3", number: 3, title: "Media", seconds: 5, added_seconds: 5, flow: false, kind: "media",
        state: "done", state_label: "Done", summary: "", issues: [], in_use: "t30",
        takes: [{id: "t30", number: 1, missing: false, review: null}],
      },
    ],
  };
  page.state.sceneId = "s1";

  page.renderTakes();
  page.renderStripActions();

  const pills = findAll(nodes["take-row"], (n) => n.className?.includes("take-score-pill"));
  assert.equal(pills[0].textContent, "88");
  assert.ok(pills[0].className.includes("score-level-strong"));
  assert.equal(pills[1].textContent, "…");

  const sceneBatchBtn = findAll(nodes["take-row"], (n) => n.id === "score-scene-takes")[0];
  assert.ok(sceneBatchBtn);
  assert.equal(sceneBatchBtn.textContent, "Score unscored takes (2)");
  assert.equal(sceneBatchBtn.disabled, false);

  const boardBatchBtn = findAll(nodes["strip-actions"], (n) => n.id === "score-board-takes")[0];
  assert.ok(boardBatchBtn);
  assert.equal(boardBatchBtn.textContent, "Score in-use takes (2)");
  assert.equal(boardBatchBtn.disabled, false);

  await sceneBatchBtn.onclick();
  await boardBatchBtn.onclick();
  assert.deepEqual(calls.map((c) => [c.method, c.url]), [
    ["POST", "/storyboard/api/boards/b1/scenes/s1/score-takes"],
    ["POST", "/storyboard/api/boards/b1/score-takes"],
  ]);

  // Media scenes still keep the 4000ms review poll alive when another scene has a pending review.
  scheduled.length = 0;
  page.state.sceneId = "s3";
  page.renderTakes();
  assert.deepEqual(scheduled, [4000]);

  // Scenes with only uploaded/Media manual takes omit the batch-score button.
  page.state.board.scenes.push({
    id: "s4", number: 4, title: "Manual only", seconds: 5, added_seconds: 5, flow: false, kind: "generated",
    state: "done", state_label: "Done", summary: "", issues: [], in_use: "t40",
    takes: [{id: "t40", number: 1, source: "upload", missing: false, review: null}],
  });
  page.state.sceneId = "s4";
  page.renderTakes();
  assert.equal(findAll(nodes["take-row"], (n) => n.id === "score-scene-takes").length, 0);
});
