// The Dubbing page: five steps over one dubbing session (Source, Review lines, Dub, Lip sync, Result).
// The session and every run live on the server; this keeps what the viewer is looking at and the
// settings of the next analysis and dub, and polls the runs it started.

import {api} from "./api.js";
import {esc, icon, plural, toast} from "./ui.js";
import {renderSource, bindSource, autoBody} from "./source.js";
import {bindBatch} from "./batch.js";
import {renderReview, bindReview} from "./review.js";
import {renderDub, bindDub, selectedTiming} from "./dub.js";
import {renderLipSync, bindLipSync, afterLipSync, loadLipSyncPlan} from "./lipsync.js";
import {renderResult, bindResult, afterResult, pauseResultPlayer} from "./result.js";

const STORE = "h3-dubbing-page";
const SCREENS = [["source", "Source"], ["review", "Review lines"], ["dub", "Dub"], ["lipsync", "Lip sync"], ["result", "Result"]];
const SAVED = ["settings", "sessionId", "project", "projectSettings", "projectSource",
  "screen", "sourceTab", "source", "pendingRuns", "completedAuto"];

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
  projectSettings: null,
  projectSource: null,
  projects: [],
  screen: "source",
  runs: {},
  pendingRuns: {},
  completedAuto: null,
  focus: 0,
  picks: new Set(),
  mergeGroups: [],
  // The lines as they were when the picks and mergeGroups were made; see lineSignature.
  linesSignature: "",
  mergeRelabel: false,
  checking: false,
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
  watchRuns: watch,
  openProject: loadProject,
};

function defaults(boot) {
  const local = (key, cloud) => (boot.gpu ? key : cloud);
  return {
    source_language: "Auto-detect", target_language: "English", style: boot.default_style, notes: "",
    pipeline: local(boot.pipelines[0].key, boot.pipelines.find((p) => p.cloud)?.key),
    moss_backend: boot.moss_backends[0].key,
    translate: true, translator: local(boot.translators[0].key, boot.translators[1].key),
    gemini_model: boot.default_gemini_model, separate: true, separator_model: boot.default_separator, reuse: true,
    aligner: true, merge_gap: 0.35, refine_timestamps: true, translation_context: false,
    // Gemini prompt options, off by default.
    shorten_lines: false, readings: false,
    voice_mode: boot.voice_modes[0], preset: "Balance", timing: boot.timings[0].key, max_speed: 1.35,
    speech_volume: 100, background_volume: 100, keep_original: true, regenerate: "", parallel: 16,
    word_timing: true, rewrite: false, rewrite_style: boot.default_rewrite_style, rewrite_by_hand: false,
    // Auto-detect combine confirms neighbouring lines by context with Gemini Decision.
    combine_context: true, auto_mode: false,
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
  const hasLipSync = !!result?.lip_sync;
  const done = {source: !!session, review: !!session && !!result, dub: !!result, lipsync: !!result?.lip_sync?.output, result: false};
  const notes = {
    source: app.source?.name || session?.source?.name || "",
    review: session ? (attention(session) ? `<span class="chip chip-warn">${attention(session)} need attention</span>` : "") : "",
    dub: "",
    lipsync: hasLipSync && !result?.lip_sync?.output ? '<span class="muted small">Optional</span>' : "",
    result: "",
  };
  document.getElementById("steps").innerHTML = SCREENS.map(([key, title], index) => {
    const optional = key === "lipsync" && hasLipSync && !result?.lip_sync?.output;
    const isDone = done[key] && app.screen !== key;
    const stepBtn = `
      <button type="button" class="step${optional ? " optional" : ""}" data-screen="${key}" ${app.screen === key ? 'aria-current="step"' : ""}
        ${(!session && key !== "source") || (key === "lipsync" && !hasLipSync) ? "disabled" : ""}>
        <span class="step-mark ${isDone ? "done" : ""}" aria-hidden="true">${isDone ? icon("check", 13) : index + 1}</span>
        ${title}${key === "source" && notes.source ? `<span class="muted small step-note">${esc(notes.source)}</span>` : ""}${notes[key] && key !== "source" ? notes[key] : ""}
      </button>`;
    const rail = index < SCREENS.length - 1 ? `<span class="step-rail${done[key] ? " done" : ""}" aria-hidden="true"></span>` : "";
    return stepBtn + rail;
  }).join("");

  const langEl = document.getElementById("step-lang");
  const target = session?.target_language || app.settings.target_language;
  const spoken = session?.spoken || app.settings.source_language;
  if (target === app.boot.same) {
    langEl.textContent = "Same language";
    langEl.hidden = false;
  } else if (spoken || target) {
    langEl.textContent = `${spoken || "Auto-detect"} → ${target || "…"}`;
    langEl.hidden = false;
  } else {
    langEl.hidden = true;
  }

  const ctaEl = document.getElementById("step-cta");
  const ctaByScreen = {
    source: session ? {label: "Review lines →", go: "review", primary: false} : null,
    review: session ? {label: "Next: Dub →", go: "dub", primary: true} : null,
    dub: result ? {label: "Watch result →", go: "result", primary: true} : null,
    lipsync: result ? {label: "Skip to Result →", go: "result", primary: false} : null,
    result: session ? {label: "Edit lines", go: "review", primary: false} : null,
  };
  const cta = ctaByScreen[app.screen] || null;
  ctaEl.hidden = !cta;
  if (cta) {
    ctaEl.textContent = cta.label;
    ctaEl.dataset.go = cta.go;
    ctaEl.className = cta.primary ? "btn btn-small btn-primary" : "btn btn-small";
  }
}

function renderTop() {
  const list = document.getElementById("project-list");
  const current = app.project?.name || "";
  list.innerHTML = `<option value="">${app.projects.length ? "Choose a project…" : "No saved projects"}</option>`
    + app.projects.map((name) => `<option ${name === current ? "selected" : ""}>${esc(name)}</option>`).join("");
  document.getElementById("project-status").textContent = current ? `Project: ${current}` : "Unsaved draft · API keys are never saved";
  const mode = document.getElementById("mode");
  mode.dataset.mode = app.boot.gpu ? "studio" : "workshop";
  mode.innerHTML = `<span class="dot ${app.boot.gpu ? "ready" : "prepare"}"></span>${app.boot.gpu ? "GPU ready" : "Workshop · Prepare only"}`;
  document.getElementById("banner").hidden = app.boot.gpu;
  document.getElementById("auto-dub").disabled = !app.boot.gpu || !!app.runs.auto?.active;
}

function savedProjectSettings() {
  return {...defaults(app.boot), ...JSON.parse(app.project ? app.projectSettings : null)};
}

function renderProjectSave() {
  const source = app.source?.path;
  const sourceChanged = app.project
    ? source && source !== app.projectSource && source !== app.session?.source?.path
    : source || app.sessionId;
  const saved = savedProjectSettings();
  const settingsChanged = app.boot.project_keys.some((key) => app.settings[key] !== saved[key]);
  const dirty = Boolean(sourceChanged || settingsChanged);
  document.getElementById("project-unsaved").hidden = !dirty;
  document.getElementById("project-save").classList.toggle("is-dirty", dirty);
}

export function render() {
  renderTop();
  renderSteps();
  const screen = document.getElementById("screen");
  const renderers = {source: renderSource, review: renderReview, dub: renderDub, lipsync: renderLipSync, result: renderResult};
  if (app.screen !== "source" && !app.session) app.screen = "source";
  if (app.screen !== "result") pauseResultPlayer();
  screen.innerHTML = renderers[app.screen](app);
  screen.dataset.screen = app.screen;
  if (app.screen === "result") afterResult(app);
  afterLipSync(app);
  updateRunViews();
  renderProjectSave();
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

// What identifies each line by position: its start, speaker and original text. Picks and combine
// suggestions hold line numbers, so they are only valid while this is unchanged; translations,
// Dub switches and end times may still be edited.
export function lineSignature(session) {
  return JSON.stringify((session?.board.lines || []).map((line) => [line.s, line.speakerId, line.source]));
}

export function setSession(session, {focus, keepSettings = false} = {}) {
  const changed = !app.session || app.session.id !== session?.id;
  app.session = session;
  app.sessionId = session?.id || null;
  const signature = lineSignature(session);
  if (changed || signature !== app.linesSignature) {
    // A new session, or lines added, deleted, moved or changed: picked and detected line numbers no longer fit.
    app.picks = new Set();
    app.mergeGroups = [];
    if (app.filter === "combine") app.filter = "all";
  }
  if (changed) {
    if (app.session?.id && session?.id && app.session.id !== session.id) {
      for (const kind of Object.keys(app.pendingRuns)) {
        if (kind !== "analyze" && kind !== "auto") delete app.pendingRuns[kind];
      }
    }
    app.focus = 0;
    app.linesRequest = null;
    if (session && !keepSettings) app.settings.voice_mode = session.voice_default;
  }
  if (session?.pipeline && session.voice_modes && !session.voice_modes.includes(app.settings.voice_mode)) {
    app.settings.voice_mode = session.voice_default;
  }
  app.linesSignature = signature;
  if (typeof focus === "number") app.focus = focus;
  const count = session?.board.lines.length || 0;
  app.focus = count ? Math.min(Math.max(0, app.focus), count - 1) : 0;
  app.picks = new Set([...app.picks].filter((index) => index < count));
}

export async function refresh({includePlayer = true, followProject = false, preserveSession = false} = {}) {
  const {sessionId, project, source} = app;
  if (!sessionId) return false;
  const isCurrent = () => sessionId === app.sessionId && project === app.project && source === app.source;
  try {
    if (followProject && project?.session === sessionId && app.projects.includes(project.name)) {
      const data = await api.post("project/load", {name: project.name, include_player: includePlayer});
      if (!isCurrent()) return false;
      if (data.session.id !== sessionId) {
        applyProject(data);
      } else {
        const saved = projectSettingsKey(data.settings);
        // Refresh saved values while keeping fields the user has edited locally.
        if (app.projectSettings !== null && saved !== app.projectSettings) {
          const previous = savedProjectSettings();
          const edited = Object.entries(app.settings).filter(([key, value]) => value !== previous[key]);
          app.settings = {...defaults(app.boot), ...data.settings, ...Object.fromEntries(edited)};
        }
        app.projectSettings = saved;
        if (app.source?.path === app.projectSource) app.source = data.session.source;
        app.projectSource = data.session.source?.path || null;
        setSession(data.session, {keepSettings: true});
      }
    } else {
      const data = await api.get("session", {id: sessionId, include_player: includePlayer});
      if (!isCurrent()) return false;
      setSession(data.session);
    }
  } catch (error) {
    if (!isCurrent()) return false;
    if (preserveSession) throw error;
    toast(error.message, {error: true});
    app.session = null;
    app.sessionId = null;
  }
  render();
  return true;
}

const POLL_MS = 1000;
let polling = null;
let pollInFlight = false;

async function poll() {
  if (pollInFlight) return;
  clearTimeout(polling);
  polling = null;
  pollInFlight = true;
  let retry = false;
  try {
    const requestedSession = app.sessionId || "";
    const data = await api.get("run", {session: requestedSession});
    app.runs = data.runs;
    // A batch can finish between polls, or while this page is closed.
    const auto = app.runs.auto;
    if (auto && auto.id !== app.completedAuto) app.pendingRuns.auto = auto.id;
    if ((app.sessionId || "") === requestedSession) {
      for (const kind of Object.keys(app.pendingRuns)) {
        if (!(kind in app.runs) || app.runs[kind] === null) {
          delete app.pendingRuns[kind];
          toast("The server restarted while that step was running.", {error: true});
        }
      }
    }
    saveSettings();
    for (const [kind, run] of Object.entries(app.runs)) {
      await handleCompletion(kind, run);
    }
    updateRunViews();
  } catch (_) {
    retry = true;
  } finally {
    pollInFlight = false;
    if (retry || Object.values(app.runs).some((run) => run?.active)) {
      watch(retry ? POLL_MS * 3 : POLL_MS);
    }
  }
}

function watch(delay = 150) {
  if (!polling && !pollInFlight) polling = setTimeout(poll, delay);
}

// A pending ID survives reloads until its completion is consumed.
async function handleCompletion(kind, run) {
  if (!run || run.active || app.pendingRuns[kind] !== run.id) return;
  await finished(kind, run);
  delete app.pendingRuns[kind];
  if (kind === "auto") app.completedAuto = run.id;
  saveSettings();
}

async function restoreAutoSession(run) {
  const keepSettings = !run.batch && run.project && app.project?.name === run.project;
  app.sessionId = run.session;
  app.project = null;
  if (run.project) {
    const data = await api.post("project/load", {name: run.project});
    if (data.session.id === run.session) {
      applyProject(data, {keepSettings});
      return;
    }
  }
  setSession((await api.get("session", {id: run.session})).session);
  app.source = app.session.source;
}

async function finished(kind, run) {
  if (run.error) toast(run.error, {error: true});
  else if (run.status && run.status !== "Done.") toast(run.status);
  if (kind === "analyze" && run.task === "analyze" && run.session && !run.error) {
    if (app.project?.name !== run.project) {
      app.projectSettings = null;
      app.projectSource = null;
    }
    app.project = run.project ? {name: run.project, session: run.session} : null;
    app.sessionId = run.session;
    await refresh({followProject: true});
    go("review");
    return;
  }
  if (kind === "lines" && (run.task === "preview" || run.task === "take") && run.audio) app.previewRun = run.id;
  if (kind === "auto" && run.session) await restoreAutoSession(run);
  if (["dub", "auto"].includes(kind) && !run.error && !run.stop_requested && run.status !== "Interrupted.") {
    if (kind !== "auto") await refresh();
    go("result");
    return;
  }
  if (kind === "lip-sync-detect" && !run.error && !run.stop_requested) {
    if (run.session === app.session?.id && run.dub_video === app.session.result?.lip_sync?.video) {
      await loadLipSyncPlan(app, {silent: true});
    }
    return;
  }
  if (kind === "lip-sync" && !run.error && !run.stop_requested && run.status !== "Interrupted.") {
    if (run.session !== app.session?.id || run.dub_video !== app.session.result?.lip_sync?.video) return;
    await refresh();
    if (run.session !== app.session?.id || run.dub_video !== app.session.result?.lip_sync?.video) return;
    await loadLipSyncPlan(app, {silent: true});
    app.playerView = {...(app.playerView || {voice: "dub", music: "on", subtitles: "translation"}), video: "lipsync"};
    go("result");
    return;
  }
  if (["lines", "dub", "words", "export", "lip-sync"].includes(kind)) await refresh();
  else render();
}

// The run cards update their status in place, so typing elsewhere on the screen is never interrupted.
function updateRunViews() {
  const auto = app.runs.auto;
  const status = document.getElementById("auto-status");
  status.hidden = !auto?.active && !auto?.error;
  status.classList.toggle("active", !!auto?.active);
  status.classList.toggle("error", !!auto?.error);
  document.getElementById("auto-message").textContent = auto ? `Auto · ${auto.error || auto.status}` : "";
  document.getElementById("auto-stop").disabled = !auto?.active;
  const autoDub = document.getElementById("auto-dub");
  autoDub.disabled = !app.boot.gpu || !!auto?.active;
  autoDub.hidden = !!auto?.active;
  for (const box of document.querySelectorAll("[data-run]")) {
    const run = app.runs[box.dataset.run];
    const task = box.dataset.task;
    const show = run && (!task || task.split(",").includes(run.task))
      && (!box.dataset.dubVideo || box.dataset.dubVideo === run.dub_video);
    box.hidden = !show || (!run.active && !run.error && !run.status);
    if (!show) continue;
    box.querySelector("[data-run-status]").textContent = run.error || run.status || "";
    box.classList.toggle("error", !!run.error);
    box.classList.toggle("active", !!run.active);
  }
  for (const button of document.querySelectorAll("[data-busy]")) {
    const run = app.runs[button.dataset.busy];
    const lipSyncBusy = ["lip-sync", "lip-sync-detect"].includes(button.dataset.busy)
      && (app.runs["lip-sync"]?.active || app.runs["lip-sync-detect"]?.active);
    button.disabled = !!auto?.active || !!run?.active || !!lipSyncBusy || button.dataset.locked === "true";
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

function projectSettingsKey(settings) {
  return JSON.stringify(settings, Object.keys(settings).sort());
}

function applyProject(data, {keepSettings = false} = {}) {
  app.project = data.project;
  app.projectSettings = projectSettingsKey(data.settings);
  app.projectSource = data.session.source?.path || null;
  if (!keepSettings) app.settings = {...defaults(app.boot), ...data.settings};
  app.source = data.session.source;
  setSession(data.session, {keepSettings: true});
}

async function loadProject(name = document.getElementById("project-list").value) {
  if (!name) {
    toast("Choose a saved project first.", {error: true});
    return;
  }
  try {
    const data = await api.post("project/load", {name, include_player: false});
    applyProject(data);
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
      expected_session: app.project?.name === name ? app.project.session : undefined,
    });
    app.projects = data.projects;
    applyProject(data, {keepSettings: true});
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

let studioSyncInFlight = false;

async function syncFromStudio({opening = false, showRecent = false} = {}) {
  if (studioSyncInFlight) return;
  studioSyncInFlight = true;
  const retry = document.getElementById("reload-studio");
  retry.disabled = true;
  const refreshOptions = {includePlayer: app.screen === "result", followProject: true, preserveSession: true};
  try {
    try {
      await api.post("reload-volume");
    } catch (error) {
      if (error.status !== 409 && error.status !== 503) throw error;
      await new Promise((resolve) => setTimeout(resolve, 600));
      await api.post("reload-volume");
    }
    app.projects = (await api.get("projects")).projects;
    app.recentLoaded = false;
    if (app.sessionId) {
      const previousId = app.sessionId;
      const wasEmpty = app.session?.board.lines.length === 0;
      const refreshed = await refresh(refreshOptions);
      if (refreshed && app.screen === "source" && app.session?.board.lines.length
          && (wasEmpty || app.sessionId !== previousId)) {
        go("review");
      }
    } else {
      if (showRecent && !app.source) {
        app.recent = (await api.get("recent")).sessions;
        app.recentLoaded = true;
        if (app.recent.length) app.sourceTab = "sessions";
      }
      render();
    }
    retry.hidden = true;
  } catch (error) {
    retry.hidden = false;
    if (opening) {
      try {
        if (app.sessionId) await refresh(refreshOptions);
        else render();
      } catch (_) {
        // The current mount may not contain the remembered session yet.
        render();
      }
    }
    toast(error.message, {error: true});
  } finally {
    studioSyncInFlight = false;
    retry.disabled = false;
  }
}

function watchWorkshop(showRecent) {
  const frame = window.frameElement;
  let visible = false;
  let opened = false;
  const check = () => {
    const shown = !document.hidden && (!frame || frame.getClientRects().length > 0);
    const entered = shown && !visible;
    visible = shown;
    if (!entered) return;
    const opening = !opened;
    opened = true;
    syncFromStudio({opening, showRecent: opening && showRecent}).then(() => {
      if (opening) poll();
    });
  };
  if (frame) new window.parent.ResizeObserver(check).observe(frame);
  document.addEventListener("visibilitychange", check);
  check();
}

function bindChrome() {
  bindBatch(app);
  document.getElementById("reload-studio").addEventListener("click", () => syncFromStudio({
    showRecent: !app.sessionId && !app.source && app.sourceTab === "sessions",
  }));
  document.getElementById("auto-dub").addEventListener("click", () => {
    const body = autoBody(app);
    if (app.screen !== "source" && app.sessionId) {
      body.session = app.sessionId;
      body.timing = selectedTiming(app);
    }
    app.startRun("auto", body);
  });
  document.getElementById("auto-stop").addEventListener("click", async () => {
    try { toast((await api.post("interrupt", {kind: "auto"})).message); }
    catch (error) { toast(error.message, {error: true}); }
  });
  document.getElementById("steps").addEventListener("click", (event) => {
    const step = event.target.closest("[data-screen]");
    if (step && !step.disabled) go(step.dataset.screen);
  });
  document.getElementById("project-load").addEventListener("click", () => loadProject());
  document.getElementById("project-new").addEventListener("click", () => {
    if (!app.project && (app.source || app.sessionId || app.manualReply)
        && !window.confirm("Leave this unsaved draft and start a new project?")) return;
    try {
      localStorage.removeItem(STORE);
      localStorage.setItem(STORE, JSON.stringify({sourceTab: "upload", completedAuto: app.runs.auto?.id || app.completedAuto}));
    } catch (_) { /* Storage can be unavailable in private windows. */ }
    window.location.reload();
  });
  document.getElementById("project-save").addEventListener("click", () => {
    if (app.project?.name) saveProject(app.project.name); else askName();
  });
  const saveMenu = document.getElementById("save-menu");
  document.getElementById("project-save-as").addEventListener("click", () => {
    if (saveMenu) saveMenu.open = false;
    askName();
  });
  document.addEventListener("click", (event) => {
    if (saveMenu?.open && !saveMenu.contains(event.target)) saveMenu.open = false;
  });
  document.getElementById("step-lang").addEventListener("click", () => go("source"));
  const stepCta = document.getElementById("step-cta");
  stepCta.addEventListener("click", () => { if (stepCta.dataset.go) go(stepCta.dataset.go); });
  const chrome = document.getElementById("dub-chrome");
  const syncChromeHeight = () => {
    if (chrome) document.documentElement.style.setProperty("--chrome-h", `${Math.ceil(chrome.getBoundingClientRect().height)}px`);
  };
  syncChromeHeight();
  if (chrome && typeof ResizeObserver !== "undefined") new ResizeObserver(syncChromeHeight).observe(chrome);
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
    renderProjectSave();
    saveSettings();
  });
  document.getElementById("screen").addEventListener("change", (event) => {
    const field = event.target.closest("[data-setting][data-redraw]");
    if (field) render();
  });
  bindSource(app);
  bindReview(app);
  bindDub(app);
  bindLipSync(app);
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
  app.projects = app.boot.projects;
  app.pendingRuns = saved.pendingRuns || {};
  app.completedAuto = saved.completedAuto || null;
  app.project = saved.project || null;
  app.projectSettings = saved.projectSettings || null;
  app.projectSource = saved.projectSource || null;
  app.sourceTab = saved.sourceTab || "upload";
  app.source = saved.source || null;
  app.sessionId = saved.sessionId || null;
  app.screen = saved.screen || "source";
  bindChrome();
  if (app.boot.reload_volume) {
    watchWorkshop(!saved.sourceTab);
  } else {
    if (app.sessionId) await refresh({includePlayer: app.screen === "result", followProject: true});
    else render();
    poll();
  }
  document.title = `Dubbing${app.project ? ` · ${app.project.name}` : ""}`;
}

export const summaryLine = (session) => {
  const lines = session.board.lines;
  return `${plural(lines.length, "line")} · ${plural(session.board.speakers.length, "speaker")}`;
};

start();
