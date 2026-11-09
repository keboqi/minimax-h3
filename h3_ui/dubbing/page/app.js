// The Dubbing page: four steps over one dubbing session (Source, Review lines, Dub, Result).
// The session and every run live on the server; this keeps what the viewer is looking at and the
// settings of the next analysis and dub, and polls the runs it started.

import {api} from "./api.js";
import {esc, icon, plural, toast} from "./ui.js";
import {renderSource, bindSource} from "./source.js";
import {renderReview, bindReview} from "./review.js";
import {renderDub, bindDub} from "./dub.js";
import {renderResult, bindResult, afterResult} from "./result.js";

const STORE = "h3-dubbing-page";
const SCREENS = [["source", "Source"], ["review", "Review lines"], ["dub", "Dub"], ["result", "Result"]];
const SAVED = ["settings", "sessionId", "project", "screen", "sourceTab", "source", "pendingRuns"];

export const app = {
  boot: null,
  settings: {},
  apiKey: "",
  manualReply: "",
  source: null,
  sourceTab: "upload",
  media: [],
  recent: [],
  session: null,
  sessionId: null,
  project: null,
  projects: [],
  screen: "source",
  runs: {},
  pendingRuns: {},
  focus: 0,
  picks: new Set(),
  filter: "all",
  table: false,
  linesRequest: null,
  plan: [],
  optionsOpen: false,
  search: "",
  follow: true,
  previewRun: null,
  originalLine: null,
  render,
  refresh,
  go,
  startRun,
  saveSettings,
  setSession,
};

function defaults(boot) {
  const local = (key, cloud) => (boot.gpu ? key : cloud);
  return {
    source_language: "Auto-detect", target_language: "English", style: boot.default_style, notes: "",
    pipeline: local(boot.pipelines[0].key, boot.pipelines.find((p) => p.cloud)?.key),
    translate: true, translator: local(boot.translators[0].key, boot.translators[1].key),
    gemini_model: boot.default_gemini_model, separate: true, separator_model: boot.default_separator, reuse: true,
    aligner: true, merge_gap: 0.35, refine_timestamps: true, translation_context: false,
    voice_mode: boot.voice_modes[0], preset: "Balance", timing: boot.timings[0].key, max_speed: 1.35,
    speech_volume: 100, background_volume: 100, keep_original: true, regenerate: "", parallel: 16,
    word_timing: false, rewrite: false, rewrite_style: boot.default_rewrite_style, rewrite_by_hand: false,
  };
}

function restore() {
  try {
    return JSON.parse(localStorage.getItem(STORE) || "{}") || {};
  } catch (_) {
    return {};
  }
}

export function saveSettings() {
  try {
    localStorage.setItem(STORE, JSON.stringify(Object.fromEntries(SAVED.map((key) => [key, app[key]]))));
  } catch (_) { /* Private windows keep the page working without it. */ }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function attention(session) {
  if (!session) return 0;
  return session.board.lines.filter((line) => line.flags.some((flag) => flag !== "off")).length;
}

function renderSteps() {
  const session = app.session;
  const result = session?.result;
  const done = {source: !!session, review: !!session && !!result, dub: !!result, result: false};
  const notes = {
    source: app.source?.name || session?.source?.name || "",
    review: session ? (attention(session) ? `<span class="chip chip-warn">${attention(session)} need attention</span>` : "") : "",
    dub: "", result: "",
  };
  document.getElementById("steps").innerHTML = SCREENS.map(([key, title], index) => `
    <button type="button" class="step" data-screen="${key}" ${app.screen === key ? 'aria-current="step"' : ""}
      ${!session && key !== "source" ? "disabled" : ""}>
      <span class="step-mark ${done[key] && app.screen !== key ? "done" : ""}">${done[key] && app.screen !== key ? icon("check", 13) : index + 1}</span>
      ${title}${key === "source" && notes.source ? `<span class="muted small step-note">${esc(notes.source)}</span>` : ""}${notes[key] && key !== "source" ? notes[key] : ""}
    </button>${index < SCREENS.length - 1 ? '<span class="step-sep" aria-hidden="true">›</span>' : ""}`).join("");
}

function renderTop() {
  const list = document.getElementById("project-list");
  const current = app.project?.name || "";
  list.innerHTML = `<option value="">${app.projects.length ? "Choose a project…" : "No saved projects"}</option>`
    + app.projects.map((name) => `<option ${name === current ? "selected" : ""}>${esc(name)}</option>`).join("");
  document.getElementById("project-status").textContent = current ? `Project: ${current}` : "Unsaved draft · API keys are never saved";
  const mode = document.getElementById("mode");
  mode.innerHTML = `<span class="dot ${app.boot.gpu ? "ready" : "prepare"}"></span>${app.boot.gpu ? "GPU ready" : "Prepare only · voices run in Studio"}`;
  document.getElementById("banner").hidden = app.boot.gpu;
}

export function render() {
  renderTop();
  renderSteps();
  const screen = document.getElementById("screen");
  const renderers = {source: renderSource, review: renderReview, dub: renderDub, result: renderResult};
  if (app.screen !== "source" && !app.session) app.screen = "source";
  screen.innerHTML = renderers[app.screen](app);
  screen.dataset.screen = app.screen;
  if (app.screen === "result") afterResult(app);
  updateRunViews();
  saveSettings();
}

export function go(screen) {
  app.screen = screen;
  render();
  window.scrollTo({top: 0});
}

// ---------------------------------------------------------------------------
// Sessions, projects and runs
// ---------------------------------------------------------------------------

export function setSession(session, {focus, keepSettings = false} = {}) {
  const changed = !app.session || app.session.id !== session?.id;
  app.session = session;
  app.sessionId = session?.id || null;
  if (changed) {
    app.picks = new Set();
    app.focus = 0;
    app.linesRequest = null;
    if (session && !keepSettings) app.settings.voice_mode = session.voice_default;
  }
  if (typeof focus === "number") app.focus = focus;
  const count = session?.board.lines.length || 0;
  app.focus = count ? Math.min(Math.max(0, app.focus), count - 1) : 0;
  app.picks = new Set([...app.picks].filter((index) => index < count));
}

export async function refresh() {
  if (!app.sessionId) return;
  try {
    setSession((await api.get("session", {id: app.sessionId})).session);
  } catch (error) {
    toast(error.message, {error: true});
    app.session = null;
    app.sessionId = null;
  }
  render();
}

const POLL_MS = 1000;
let polling = null;

async function poll() {
  polling = null;
  let data;
  try {
    data = await api.get("run", {session: app.sessionId || ""});
  } catch (_) {
    polling = setTimeout(poll, POLL_MS * 3);
    return;
  }
  app.runs = data.runs;
  for (const [kind, run] of Object.entries(app.runs)) {
    await handleCompletion(kind, run);
  }
  updateRunViews();
  if (Object.values(app.runs).some((run) => run && run.active)) polling = setTimeout(poll, POLL_MS);
}

function watch() {
  if (!polling) polling = setTimeout(poll, 150);
}

// A pending ID survives reloads until its completion is consumed.
async function handleCompletion(kind, run) {
  if (!run || run.active || app.pendingRuns[kind] !== run.id) return;
  delete app.pendingRuns[kind];
  saveSettings();
  await finished(kind, run);
}

async function finished(kind, run) {
  if (run.error) toast(run.error, {error: true});
  else if (run.status && run.status !== "Done.") toast(run.status);
  if (kind === "analyze" && run.task === "analyze" && run.session && !run.error) {
    app.project = run.project ? {name: run.project, session: run.session} : null;
    app.sessionId = run.session;
    await refresh();
    go("review");
    return;
  }
  if (kind === "lines" && (run.task === "preview" || run.task === "take") && run.audio) app.previewRun = run.id;
  if (kind === "dub" && !run.error) {
    await refresh();
    go("result");
    return;
  }
  if (["lines", "dub", "words", "export"].includes(kind)) await refresh();
  else render();
}

// The run cards update their status in place, so typing elsewhere on the screen is never interrupted.
function updateRunViews() {
  for (const box of document.querySelectorAll("[data-run]")) {
    const run = app.runs[box.dataset.run];
    const task = box.dataset.task;
    const show = run && (!task || task.split(",").includes(run.task));
    box.hidden = !show || (!run.active && !run.error && !run.status);
    if (!show) continue;
    box.querySelector("[data-run-status]").textContent = run.error || run.status || "";
    box.classList.toggle("error", !!run.error);
    box.classList.toggle("active", !!run.active);
  }
  for (const button of document.querySelectorAll("[data-busy]")) {
    const run = app.runs[button.dataset.busy];
    button.disabled = !!(run && run.active) || button.dataset.locked === "true";
  }
  for (const button of document.querySelectorAll("[data-stop]")) {
    const run = app.runs[button.dataset.stop];
    button.disabled = !(run && run.active);
  }
  if (app.screen === "source") {
    const run = app.runs.analyze;
    const active = run?.active && run.task === "analyze" ? run.step : null;
    for (const item of document.querySelectorAll("[data-plan-step]")) {
      const keys = [...document.querySelectorAll("[data-plan-step]")].map((node) => node.dataset.planStep);
      const at = active ? keys.indexOf(active) : -1;
      const mine = keys.indexOf(item.dataset.planStep);
      item.dataset.state = at < 0 ? "" : mine < at ? "done" : mine === at ? "current" : "";
    }
  }
}

export async function startRun(path, body) {
  try {
    const data = await api.post(path, body);
    const run = data.run;
    app.runs = {...app.runs, [run.kind]: run};
    app.pendingRuns[run.kind] = run.id;
    saveSettings();
    updateRunViews();
    await handleCompletion(run.kind, run);
    watch();
    return run;
  } catch (error) {
    toast(error.message, {error: true});
    return null;
  }
}

async function loadProject() {
  const name = document.getElementById("project-list").value;
  if (!name) {
    toast("Choose a saved project first.", {error: true});
    return;
  }
  try {
    const data = await api.post("project/load", {name});
    app.project = data.project;
    app.settings = {...defaults(app.boot), ...data.settings};
    app.source = data.session.source;
    setSession(data.session, {keepSettings: true});
    toast(`Opened ${name}.`);
    go(data.session.board.lines.length ? "review" : "source");
  } catch (error) {
    toast(error.message, {error: true});
  }
}

async function saveProject(name) {
  try {
    const data = await api.post("project/save", {
      name, session: app.sessionId, source: app.source?.path || null, settings: app.settings,
    });
    app.project = data.project;
    app.projects = data.projects;
    app.source = data.session.source;
    setSession(data.session, {keepSettings: true});
    toast(`Saved ${name}. Line edits save automatically.`);
    render();
  } catch (error) {
    toast(error.message, {error: true});
  }
}

function askName() {
  const dialog = document.getElementById("name-dialog");
  const input = document.getElementById("name-input");
  input.value = app.project?.name || app.session?.name || app.source?.name?.replace(/\.[^.]+$/, "") || "";
  dialog.showModal();
  input.select();
}

function bindChrome() {
  document.getElementById("steps").addEventListener("click", (event) => {
    const step = event.target.closest("[data-screen]");
    if (step && !step.disabled) go(step.dataset.screen);
  });
  document.getElementById("project-load").addEventListener("click", loadProject);
  document.getElementById("project-save").addEventListener("click", () => {
    if (app.project?.name) saveProject(app.project.name); else askName();
  });
  document.getElementById("project-save-as").addEventListener("click", askName);
  const dialog = document.getElementById("name-dialog");
  document.getElementById("name-cancel").addEventListener("click", () => dialog.close("cancel"));
  dialog.addEventListener("close", () => {
    const name = document.getElementById("name-input").value.trim();
    if (dialog.returnValue === "save" && name) saveProject(name);
  });
  // Segmented controls write the setting they name, then the screen draws itself again.
  document.getElementById("screen").addEventListener("click", (event) => {
    const button = event.target.closest("[data-seg]");
    if (!button || button.disabled || !(button.dataset.seg in app.settings)) return;
    if (button.dataset.value === "__custom") return;  // "Your own…" opens a field (source.js).
    const raw = button.dataset.value;
    const current = app.settings[button.dataset.seg];
    app.settings[button.dataset.seg] = typeof current === "boolean" ? raw === "true"
      : typeof current === "number" ? Number(raw) : raw;
    render();
  });
  // Setting fields: [data-setting] inputs save as they change, without drawing the screen again.
  document.getElementById("screen").addEventListener("input", (event) => {
    const field = event.target.closest("[data-setting]");
    if (!field) return;
    const key = field.dataset.setting;
    app.settings[key] = field.type === "checkbox" ? field.checked
      : field.type === "range" || field.type === "number" ? Number(field.value) : field.value;
    const out = document.querySelector(`[data-output="${key}"]`);
    if (out) out.textContent = field.dataset.format === "x" ? `${Number(field.value).toFixed(2)}×`
      : field.dataset.format === "%" ? `${field.value}%` : field.dataset.format === "s" ? `${Number(field.value).toFixed(2)} s` : field.value;
    saveSettings();
  });
  document.getElementById("screen").addEventListener("change", (event) => {
    const field = event.target.closest("[data-setting][data-redraw]");
    if (field) render();
  });
  bindSource(app);
  bindReview(app);
  bindDub(app);
  bindResult(app);
}

async function start() {
  try {
    app.boot = await api.get("boot");
  } catch (error) {
    document.getElementById("screen").innerHTML = `<p class="card empty">${esc(error.message)}</p>`;
    return;
  }
  const saved = restore();
  app.settings = {...defaults(app.boot), ...(saved.settings || {})};
  if (!app.boot.gpu && !app.boot.pipelines.some((p) => p.key === app.settings.pipeline)) {
    app.settings.pipeline = app.boot.pipelines[0].key;
  }
  app.projects = app.boot.projects;
  app.pendingRuns = saved.pendingRuns || {};
  app.project = saved.project || null;
  app.sourceTab = saved.sourceTab || "upload";
  app.source = saved.source || null;
  app.sessionId = saved.sessionId || null;
  app.screen = saved.screen || "source";
  bindChrome();
  if (app.sessionId) await refresh(); else render();
  poll();
  document.title = `Dubbing${app.project ? ` · ${app.project.name}` : ""}`;
}

export const summaryLine = (session) => {
  const lines = session.board.lines;
  return `${plural(lines.length, "line")} · ${plural(session.board.speakers.length, "speaker")}`;
};

start();
