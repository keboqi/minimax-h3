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
