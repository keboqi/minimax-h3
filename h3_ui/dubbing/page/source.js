// Step 1, Source: the file, its languages, how to find the lines, and the analysis.

import {api} from "./api.js";
import {clock, copyText, esc, icon, promptOptions, rich, seg, size, toast} from "./ui.js";

const CUSTOM = "__custom";
const SAME_HINT = "No translation: fix lines, or rewrite them for an audience.";
let planKey = "";
let planTimer = null;

const isSame = (app) => app.settings.target_language === app.boot.same;
const gemini = (app) => app.boot.translators[1].key;

function fileCard(source, {replace = true} = {}) {
  if (!source) return "";
  const length = source.seconds ? clock(source.seconds, false) : "";
  return `
    <div class="file-card">
      <div class="file-thumb" aria-hidden="true">${icon("play", 30)}${length ? `<span>${length}</span>` : ""}</div>
      <div class="file-info">
        <strong>${esc(source.name)}</strong>
        <span class="muted small">${[source.kind, length && `${length}`, size(source.size)].filter(Boolean).map(esc).join(" · ")}</span>
        ${replace ? '<div class="row"><button type="button" class="btn btn-small" data-action="replace-file">Replace file</button></div>' : ""}
      </div>
    </div>`;
}

function sourceTabs(app) {
  const tab = app.sourceTab;
  const tabs = [["upload", "Upload"], ["media", "From Media"], ["sessions", "Earlier session"]];
  let body = "";
  if (tab === "upload") {
    body = app.source ? fileCard(app.source) : `
      <label class="drop" data-drop>
        <input type="file" accept="video/*,audio/*,.mkv,.m4v,.flac,.m4a" data-action="upload" hidden>
        ${icon("download", 22)}
        <strong>Drop a video or recording, or choose a file</strong>
        <span class="muted small">MP4, MOV, MKV, WEBM, MP3, WAV, FLAC, OGG, M4A</span>
      </label>`;
  } else if (tab === "media") {
    body = `
      <div class="row row-end-align">
        <label class="field grow">Video or audio in Media
          <select data-action="media">
            <option value="">${app.media.length ? "Choose a file…" : "Nothing in Media yet"}</option>
            ${app.media.map((item) => `<option value="${esc(item.path)}" ${app.source?.path === item.path ? "selected" : ""}>${esc(item.label)}</option>`).join("")}
          </select>
        </label>
        <button type="button" class="btn" data-action="refresh-media">Refresh</button>
      </div>
      <p class="muted small">Online videos: download them first in <strong>Media → Download from a URL</strong>; they
        appear here when done. Picking one replaces an upload.</p>
      ${app.source ? fileCard(app.source, {replace: false}) : ""}`;
  } else {
    body = `
      <p class="muted small">Analyzed sessions are kept for 14 days. Opening one restores its edited lines and dub
        language, and opens Review lines.</p>
      <div class="stack-8">${app.recent.length ? app.recent.map((item) => `
        <button type="button" class="session-row" data-action="open-session" data-id="${esc(item.id)}">
          <span class="grow"><strong>${esc(item.name)}</strong><span class="muted small">${esc(item.detail)}</span></span>
          <span class="muted small">${esc(item.when)}</span>
        </button>`).join("") : '<p class="muted">No analyzed sessions yet.</p>'}</div>`;
  }
  return `
    <section class="card" aria-labelledby="src-h">
      <div class="card-head">
        <h2 id="src-h">What should be dubbed?</h2>
        <div class="tabs" role="tablist">${tabs.map(([key, label]) => `
          <button type="button" role="tab" aria-selected="${tab === key}" data-action="source-tab" data-tab="${key}">${label}</button>`).join("")}
        </div>
      </div>
      <div class="card-body">${body}</div>
    </section>`;
}

function styleField(app, {rewrite}) {
  const key = rewrite ? "rewrite_style" : "style";
  const names = rewrite ? app.boot.rewrite_styles : app.boot.styles;
  const current = app.settings[key];
  const custom = app.customStyle?.[key] || (current && !names.includes(current));
  return `
    <div class="field">${rewrite ? "Rewrite style" : "Translation style"}
      <span class="muted small">${rewrite ? "Who the new version is for. Translation notes still keep your terms."
        : "How Hy-MT and Gemini should phrase it. Applies to lines translated from now on."}</span>
      ${seg(key, [...names.map((name) => [name, name]), [CUSTOM, "Your own…"]], custom ? CUSTOM : current,
        {label: rewrite ? "Rewrite style" : "Translation style"})}
      ${custom ? `<input data-setting="${key}" value="${esc(names.includes(current) ? "" : current)}"
        placeholder="${rewrite ? "e.g. for 8-year-olds learning science" : "e.g. for adults over 60, warm and unhurried"}"
        aria-label="Your own style">` : ""}
    </div>`;
}

function rewriter(app) {
  const cards = [
    [gemini(app), "Gemini", "Recommended", "Best at retelling for an audience."],
    ["hand", "By hand", "Recommended", "Copy the prompt into any chat model in Review lines. No key."],
    ...(app.boot.gpu ? [[app.boot.translators[0].key, "Hy-MT2", "Experimental",
      "On this server. A translation model; it may change little."]] : []),
  ];
  const current = app.settings.rewrite_by_hand ? "hand" : app.settings.translator;
  return `
    <div class="field">Rewriter <span class="muted small">Who retells the lines.</span>
      <div class="pick-grid three" role="radiogroup" aria-label="Rewriter">${cards.map(([key, label, badge, desc]) => `
        <button type="button" role="radio" class="pick" aria-checked="${current === key}" data-action="rewriter" data-key="${esc(key)}">
          <span class="dot"></span><span><strong>${label}</strong>
          <span class="badge ${badge === "Experimental" ? "warn" : ""}">${badge}</span>
          <span class="muted small">${desc}</span></span>
        </button>`).join("")}
      </div>
    </div>`;
}

function languages(app) {
  const same = isSame(app);
  const options = [app.boot.same, ...app.boot.target_languages];
  const target = app.settings.target_language;
  const custom = !options.includes(target);
  return `
    <section class="card card-pad stack-16" aria-labelledby="lang-h">
      <h2 id="lang-h">Languages and tone</h2>
      <div class="row row-wrap align-start">
        <label class="field grow">Spoken language
          <select data-setting="source_language">${app.boot.source_languages.map((name) => `
            <option ${name === app.settings.source_language ? "selected" : ""}>${esc(name)}</option>`).join("")}</select>
        </label>
        <span class="arrow" aria-hidden="true">→</span>
        <div class="field grow">
          <label for="dub-into">Dub into</label>
          <select id="dub-into" data-action="target-language">
            ${options.map((name) => `<option value="${esc(name)}" ${name === target ? "selected" : ""}>${esc(name)}</option>`).join("")}
            <option value="" ${custom ? "selected" : ""}>Other language…</option>
          </select>
          ${custom ? `<label for="custom-target">Language name</label>
            <input id="custom-target" data-setting="target_language" value="${esc(target)}" autocomplete="off">` : ""}
          <span class="muted small">${same ? SAME_HINT : `Choose <button type="button" class="linkish" data-action="same">Same as spoken</button> to fix or rewrite lines.`}</span>
        </div>
      </div>
      ${same ? `
        <div class="sub stack-8">
          <span class="field-label">New text starts as</span>
          ${seg("rewrite", [[false, "A copy of what was said"], [true, "A rewrite for an audience"]], app.settings.rewrite,
            {label: "New text starts as"})}
          <span class="muted small">${app.settings.rewrite
            ? "Every line is retold in the same language, in the style below. Lines the rewrite leaves unchanged keep their audio."
            : "The translate step is skipped. Edit only the lines you want to fix; only edited lines are voiced."}</span>
        </div>
        ${app.settings.rewrite ? styleField(app, {rewrite: true}) + rewriter(app) : ""}`
        : styleField(app, {rewrite: false})}
      <label class="field">Translation notes
        <span class="muted small">Names and terms to keep. Sent to the translator and to Gemini's transcription.</span>
        <textarea data-setting="notes" rows="2" placeholder="e.g. Keep the name Mei.">${esc(app.settings.notes)}</textarea>
      </label>
    </section>`;
}

function transcription(app) {
  const pipe = app.settings.pipeline;
  const manual = pipe === app.boot.pipelines.find((p) => p.label === "Gemini by hand")?.key;
  const same = isSame(app);
  const translator = app.settings.translator;
  const cloud = pipe === app.boot.pipelines.find((p) => p.label === "Gemini (cloud)")?.key;
  const needsGemini = cloud
    || (translator === gemini(app) && (!same || app.settings.rewrite) && !app.settings.rewrite_by_hand);
  // Any Gemini prompt that writes the new text: transcription that translates, the translator or rewriter.
  const prompted = same ? app.settings.rewrite && (translator === gemini(app) || app.settings.rewrite_by_hand)
    : app.settings.translate && (cloud || manual || translator === gemini(app));
  const prepare = app.runs.analyze;
  const speech = prepare && prepare.task === "prepare" && prepare.file && !prepare.active
    ? `<a class="btn btn-small" href="${api.url("run-file", {id: prepare.id})}">${icon("download", 14)} ${esc(prepare.file)}</a>` : "";
  return `
    <section class="card card-pad stack-14" aria-labelledby="tr-h">
      <div><h2 id="tr-h">How to find the lines</h2>
        <p class="muted">Who speaks when, every line transcribed${same ? "" : ", then translated"}.</p></div>
      <div class="pick-grid" role="radiogroup" aria-label="Transcription">${app.boot.pipelines.map((p) => `
        <button type="button" role="radio" class="pick" aria-checked="${p.key === pipe}" data-action="pipeline" data-key="${esc(p.key)}">
          <span class="dot"></span><span><strong>${esc(p.label)}</strong><span class="muted small">${esc(p.desc)}</span></span>
        </button>`).join("")}
      </div>
      ${manual ? `
        <div class="sub stack-12">
          <strong>Run it yourself in a chat model</strong>
          <p class="muted small">Copy the prompt into Gemini or any chat model and attach the audio. Your languages,
            style and notes are in it. Keep timestamps exact if you tune it. Nothing here calls a model.</p>
          <div class="row row-wrap">
            <button type="button" class="btn btn-small" data-action="copy-source-prompt">Copy prompt</button>
            <button type="button" class="btn btn-small" data-action="prepare-audio" data-busy="analyze">Prepare audio</button>
            ${speech}
          </div>
          <label class="field">Reply
            <textarea class="mono" rows="5" data-action="manual-reply" placeholder='{"speakers": [...], "segments": [...]}'>${esc(app.manualReply)}</textarea>
            <span class="muted small">Checked before anything slow starts. Lines left without new text wait in Review lines.</span>
          </label>
        </div>` : ""}
      ${same ? "" : `
        <div class="row row-wrap align-center">
          <label class="field inline">Translator
            <select data-setting="translator" data-redraw>${app.boot.translators.filter((t) => app.boot.gpu || t.key === gemini(app)).map((t) => `
              <option value="${esc(t.key)}" ${t.key === translator ? "selected" : ""}>${esc(t.label)}</option>`).join("")}</select>
          </label>
          <label class="check"><input type="checkbox" data-setting="translate" ${app.settings.translate ? "checked" : ""}>
            <span>Translate now <span class="muted small">(off: transcribe only)</span></span></label>
          ${translator !== gemini(app) ? `<label class="check"><input type="checkbox" data-setting="translation_context"
            ${app.settings.translation_context ? "checked" : ""}><span>Include previous lines <span class="muted small">(Hy-MT, slower)</span></span></label>` : ""}
        </div>`}
      ${needsGemini ? `
        <div class="sub row row-wrap">
          <label class="field grow">Gemini model
            <select data-setting="gemini_model">${app.boot.gemini_models.map((name) => `
              <option ${name === app.settings.gemini_model ? "selected" : ""}>${esc(name)}</option>`).join("")}</select>
          </label>
          <label class="field grow-2">Gemini API key for this session
            <input type="password" data-action="api-key" value="${esc(app.apiKey)}" placeholder="Uses GEMINI_API_KEY when empty" autocomplete="off">
          </label>
        </div>` : ""}
      ${prompted ? `<div class="sub stack-8"><strong>Gemini prompt</strong>
        ${promptOptions(app.settings, {same})}</div>` : ""}
    </section>`;
}

function options(app) {
  const s = app.settings;
  const local = app.boot.gpu && s.pipeline === app.boot.pipelines[0].key;
  const summary = [s.separate ? `Separate vocals · ${s.separator_model.split(" (")[0]}` : "No vocal separation",
    s.reuse ? "Reuse earlier results" : "Redo earlier results",
    ...(local ? [s.aligner ? "Word timing" : "No word timing", `Join < ${Number(s.merge_gap).toFixed(2)} s`,
      ...(s.refine_timestamps ? ["Align"] : [])] : [])].join(" · ");
  return `
    <section class="card">
      <button type="button" class="disclosure" aria-expanded="${app.optionsOpen}" data-action="toggle-options">
        ${icon("chevron")}<strong>More options</strong><span class="muted small">${esc(summary)}</span>
      </button>
      ${app.optionsOpen ? `
        <div class="card-body stack-14 top-line">
          <label class="check"><input type="checkbox" data-setting="separate" data-redraw ${s.separate ? "checked" : ""}>
            <span><strong>Separate vocals from the background</strong><br><span class="muted small">Keeps music and
            effects under the dub and gives clean voice references. Off: the original is ducked under the dub.</span></span></label>
          ${s.separate ? `<div class="field indent">Separation model
            ${seg("separator_model", app.boot.separators.map((name) => [name, name.replace(" (", " · ").replace(")", "")]), s.separator_model,
              {label: "Separation model"})}</div>` : ""}
          <label class="check"><input type="checkbox" data-setting="reuse" ${s.reuse ? "checked" : ""}>
            <span><strong>Reuse earlier results for this file</strong><br><span class="muted small">Same file and
            settings lend their separated vocals and transcript. Untick to redo both, e.g. to rerun a failed refinement.</span></span></label>
          ${local ? `
            <div class="sub stack-12">
              <strong>Qwen3-ASR</strong>
              <label class="check"><input type="checkbox" data-setting="aligner" ${s.aligner ? "checked" : ""}>
                <span>Word timing with Qwen3-ForcedAligner <span class="muted small">· splits long speech at sentence ends</span></span></label>
              <label class="slider-row"><span>Join pauses shorter than</span>
                <input type="range" min="0" max="1.5" step="0.05" value="${s.merge_gap}" data-setting="merge_gap" data-format="s">
                <output class="mono" data-output="merge_gap">${Number(s.merge_gap).toFixed(2)} s</output></label>
              <label class="check"><input type="checkbox" data-setting="refine_timestamps" ${s.refine_timestamps ? "checked" : ""}>
                <span>Refine speech boundaries with <a href="https://desertant.com/models/align/" target="_blank" rel="noopener">Align</a>
                (experimental)<br><span class="muted small">Keeps the original timing on failure. Needs Linux and Node.js 22+.</span></span></label>
            </div>` : ""}
        </div>` : ""}
    </section>`;
}

function analyzeCard(app) {
  return `
    <aside class="rail">
      <section class="card card-pad stack-16" aria-labelledby="an-h">
        <div><h2 id="an-h">Analyze</h2><p class="muted">Review lines opens when it is done.</p></div>
        <ol class="plan" id="plan">${planItems(app.plan)}</ol>
        <div class="run-status" data-run="analyze" hidden><span class="spinner" aria-hidden="true"></span>
          <span data-run-status role="status"></span></div>
        <div class="row">
          <button type="button" class="btn btn-primary grow" data-action="analyze" data-busy="analyze">Analyze source</button>
          <button type="button" class="btn" data-stop="analyze" data-action="interrupt" data-kind="analyze" disabled>Interrupt</button>
        </div>
      </section>
      ${app.session ? '<p class="muted small rail-note">Already analyzed? <button type="button" class="linkish" data-action="to-review">Go to Review lines →</button></p>' : ""}
    </aside>`;
}

function planItems(steps) {
  return steps.map((step, index) => `
    <li data-plan-step="${esc(step.key)}"><span class="plan-mark" aria-hidden="true">${index + 1}</span>
      <span><strong>${esc(step.title)}</strong><span class="muted small">${esc(step.detail)}</span></span></li>`).join("");
}

function schedulePlan(app) {
  const s = app.settings;
  const body = {separate: s.separate, separator_model: s.separator_model, reuse: s.reuse, pipeline: s.pipeline,
    translate: s.translate, translator: s.translator, target_language: s.target_language, rewrite: s.rewrite,
    rewrite_style: s.rewrite_style};
  const key = JSON.stringify(body);
  if (key === planKey) return;
  planKey = key;
  clearTimeout(planTimer);
  planTimer = setTimeout(async () => {
    try {
      app.plan = (await api.post("plan", body)).steps;
      const list = document.getElementById("plan");
      if (list) list.innerHTML = planItems(app.plan);
    } catch (_) { /* The plan is a preview; the analysis checks its own settings. */ }
  }, 150);
}

export function renderSource(app) {
  schedulePlan(app);
  if (app.sourceTab === "media" && !app.media.length && !app.mediaLoaded) loadMedia(app);
  if (app.sourceTab === "sessions" && !app.recentLoaded) loadRecent(app);
  return `
    <div class="layout">
      <div class="main-col stack-20">
        ${sourceTabs(app)}${languages(app)}${transcription(app)}${options(app)}
      </div>
      ${analyzeCard(app)}
    </div>`;
}

async function loadMedia(app) {
  app.mediaLoaded = true;
  try {
    app.media = (await api.get("media")).media;
    app.render();
  } catch (error) {
    toast(error.message, {error: true});
  }
}

async function loadRecent(app) {
  app.recentLoaded = true;
  try {
    app.recent = (await api.get("recent")).sessions;
    app.render();
  } catch (error) {
    toast(error.message, {error: true});
  }
}

async function upload(app, file) {
  if (!file) return;
  toast(`Uploading ${file.name}…`);
  try {
    app.source = (await api.upload(file)).source;
    toast(`${file.name} is ready to analyze.`);
    app.render();
  } catch (error) {
    toast(error.message, {error: true});
  }
}

function analyzeBody(app, extra = {}) {
  const s = app.settings;
  const same = s.target_language === app.boot.same;
  const style = same && s.rewrite ? s.rewrite_style : s.style;
  return {...s, style, source: app.source?.path || null, api_key: app.apiKey, manual_reply: app.manualReply,
    translator: same && s.rewrite_by_hand ? gemini(app) : s.translator,
    rewrite: same && s.rewrite && !s.rewrite_by_hand, project: app.project, ...extra};
}

export function bindSource(app) {
  const screen = document.getElementById("screen");
  screen.addEventListener("click", async (event) => {
    if (app.screen !== "source") return;
    const target = event.target.closest("[data-action]");
    if (!target) return;
    const action = target.dataset.action;
    const s = app.settings;
    if (action === "source-tab") {
      app.sourceTab = target.dataset.tab;
      if (app.sourceTab === "sessions") app.recentLoaded = false;
      app.render();
    } else if (action === "replace-file") {
      app.source = null;
      app.sourceTab = "upload";
      app.render();
    } else if (action === "refresh-media") {
      app.mediaLoaded = false;
      app.media = [];
      app.render();
    } else if (action === "open-session") {
      app.sessionId = target.dataset.id;
      app.project = null;
      await app.refresh();
      if (app.session) {
        app.source = app.session.source;
        s.target_language = app.session.target_language || s.target_language;
        s.rewrite = app.session.rewrite;
        if (app.session.rewrite_style) s.rewrite_style = app.session.rewrite_style;
        app.go("review");
      }
    } else if (action === "same") {
      s.target_language = app.boot.same;
      app.render();
    } else if (action === "pipeline") {
      s.pipeline = target.dataset.key;
      app.render();
    } else if (action === "rewriter") {
      s.rewrite_by_hand = target.dataset.key === "hand";
      if (!s.rewrite_by_hand) s.translator = target.dataset.key;
      app.render();
    } else if (action === "toggle-options") {
      app.optionsOpen = !app.optionsOpen;
      app.render();
    } else if (action === "copy-source-prompt") {
      try {
        const {prompt} = await api.post("source-prompt", {target_language: s.target_language, translate: s.translate,
          notes: s.notes, style: s.style, shorten_lines: s.shorten_lines, readings: s.readings});
        toast(await copyText(prompt) ? "Prompt copied. Paste it with the audio in your chat model." : "Copy failed.");
      } catch (error) {
        toast(error.message, {error: true});
      }
    } else if (action === "prepare-audio") {
      app.startRun("prepare-audio", analyzeBody(app));
    } else if (action === "analyze") {
      if (!app.source) {
        toast("Upload a video or audio file, or pick one from Media.", {error: true});
        return;
      }
      app.startRun("analyze", analyzeBody(app));
    } else if (action === "interrupt") {
      try {
        const data = await api.post("interrupt", {kind: target.dataset.kind, session: app.sessionId});
        toast(data.message);
      } catch (error) {
        toast(error.message, {error: true});
      }
    } else if (action === "to-review") {
      app.go("review");
    }
  });
  // Segmented "Your own…" opens a field instead of choosing a preset.
  screen.addEventListener("click", (event) => {
    if (app.screen !== "source") return;
    const button = event.target.closest('[data-seg="style"], [data-seg="rewrite_style"]');
    if (!button) return;
    const key = button.dataset.seg;
    app.customStyle = {...(app.customStyle || {}), [key]: button.dataset.value === CUSTOM};
    if (button.dataset.value === CUSTOM) {
      app.settings[key] = "";
      app.render();
      setTimeout(() => document.querySelector(`[data-setting="${key}"]`)?.focus(), 0);
    }
  }, true);
  screen.addEventListener("input", (event) => {
    if (app.screen !== "source") return;
    const action = event.target.dataset.action;
    if (action === "api-key") app.apiKey = event.target.value;
    if (action === "manual-reply") app.manualReply = event.target.value;
    if (event.target.id === "custom-target") schedulePlan(app);
  });
  screen.addEventListener("change", async (event) => {
    if (app.screen !== "source") return;
    const action = event.target.dataset.action;
    if (action === "target-language") {
      const target = event.target.value;
      app.settings.target_language = target;
      app.render();
      document.getElementById(target ? "dub-into" : "custom-target")?.focus();
    }
    if (event.target.id === "custom-target" && (isSame(app) || app.boot.target_languages.includes(event.target.value))) {
      app.render();
    }
    if (action === "upload") upload(app, event.target.files[0]);
    if (action === "media") {
      const path = event.target.value;
      if (!path) return;
      try {
        app.source = (await api.get("source", {path})).source;
        app.render();
      } catch (error) {
        toast(error.message, {error: true});
      }
    }
  });
  screen.addEventListener("dragover", (event) => {
    if (app.screen === "source" && event.target.closest("[data-drop]")) event.preventDefault();
  });
  screen.addEventListener("drop", (event) => {
    if (app.screen !== "source" || !event.target.closest("[data-drop]")) return;
    event.preventDefault();
    upload(app, event.dataTransfer.files[0]);
  });
}

export const sourceNote = (text) => rich(text);
