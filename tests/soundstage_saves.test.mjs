import test from "node:test";
import assert from "node:assert/strict";
import {createSaves} from "../h3_ui/drama3/page/saves.js";
import {createApi, createPoller, StaleResponse} from "../h3_ui/drama3/page/transport.js";

const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
};

test("flush waits for in-flight saves and drains edits in order", async () => {
  const first = deferred(), calls = [];
  const saves = createSaves({send: async (name, key, body) => {
    calls.push({name, key, body});
    if (calls.length === 1) await first.promise;
    return body;
  }});
  saves.queue("A", "script", {text: "old"});
  const running = saves.flush("A");
  await Promise.resolve();
  assert.equal(saves.revision("A"), 0);
  saves.queue("A", "script", {text: "new"});
  assert.equal(saves.flush("A"), running);
  assert.equal(calls.length, 1);
  first.resolve();
  await running;
  assert.equal(saves.revision("A"), 2);
  assert.deepEqual(calls.map((call) => call.body.text), ["old", "new"]);
  assert.equal(saves.status("A"), "saved");
});

test("failed edits survive and newer edits win when retried", async () => {
  const first = deferred(), calls = [];
  const saves = createSaves({send: async (name, key, body) => {
    calls.push({...body});
    if (calls.length === 1) await first.promise;
  }});
  saves.queue("A", "script", {text: "old", title: "Keep"});
  const running = saves.flush("A");
  await Promise.resolve();
  saves.queue("A", "script", {text: "new"});
  first.reject(new Error("offline"));
  await assert.rejects(running, /offline/);
  assert.equal(saves.revision("A"), 0);
  assert.equal(saves.status("A"), "failed");
  assert.deepEqual(saves.edits("A").script, {text: "new", title: "Keep"});
  await saves.flush("A");
  assert.equal(saves.revision("A"), 1);
  assert.deepEqual(calls[1], {text: "new", title: "Keep"});
});

test("settings and video edits retain their originating project", async () => {
  const calls = [];
  const saves = createSaves({send: async (...args) => calls.push(args)});
  saves.queue("A", "video", {video_style: "A style"});
  saves.queue("A", "settings", {parallel: 3});
  saves.queue("B", "video", {video_style: "B style"});
  await saves.flush("A");
  await saves.flush("B");
  assert.deepEqual(calls.map(([name, key]) => [name, key]), [["A", "video"], ["A", "settings"], ["B", "video"]]);
});

test("a save failure rejects the barrier before a dependent action", async () => {
  let started = false;
  const saves = createSaves({send: async () => { throw new Error("offline"); }});
  saves.queue("A", "slot:voice:0", {description: "changed"});
  await assert.rejects(async () => { await saves.flush("A"); started = true; });
  assert.equal(started, false);
});

test("transport rejects old answers after switching away and back", async (t) => {
  const reply = deferred();
  const original = globalThis.fetch;
  t.after(() => { globalThis.fetch = original; });
  globalThis.fetch = async () => reply.promise;
  let scope = "A:1";
  const api = createApi("/api", () => scope);
  const request = api("/rewrite");
  scope = "A:3";
  reply.resolve({ok: true, json: async () => ({name: "A"})});
  await assert.rejects(request, StaleResponse);
});

test("stopping a poller cancels its scheduled request", async () => {
  let calls = 0;
  const poll = createPoller(async () => calls++, {delay: 1});
  poll(); poll.stop();
  await new Promise((resolve) => setTimeout(resolve, 10));
  assert.equal(calls, 0);
});
