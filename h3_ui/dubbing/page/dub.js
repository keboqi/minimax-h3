// Step 3, Dub: speakers, voices, timing and mix, and what the next dub will do.

import {api} from "./api.js";
import {esc, icon, plural, ranges, seg, toast} from "./ui.js";

const TONES = ["#c92a2a", "#1864ab", "#2b8a3e", "#6741d9", "#d9480f", "#9c36b5", "#c2255c", "#a07800"];

function latest(app) {
  const result = app.session.result;
  if (!result) return "";
  return `
    <button type="button" class="card latest" data-action="to-result">
      <span class="latest-thumb" aria-hidden="true">${icon("play", 18)}</span>
      <span class="grow"><strong>Latest dub · ${esc(result.when)}</strong>
        <span class="muted small">${esc(result.label)}${result.failed.length ? ` · ${plural(result.failed.length, "line")} kept the original voice` : ""}</span></span>
      <span class="latest-go">Watch result ${icon("arrow", 15)}</span>
    </button>`;
}

function speakers(app) {
  const panel = app.session.speakers;
  if (!panel) return "";
  return `
    <section class="card card-pad stack-14" aria-labelledby="cast-h">
      <div class="card-title"><h2 id="cast-h">Speakers</h2>
        <span class="muted small">Cloned from the original audio with Drama 3. Give a speaker a saved voice instead, or make
        it louder or quieter. Changes save at once.</span></div>
      <div class="speaker-grid">${panel.cards.map((card, index) => `
        <div class="speaker-card">
          <div class="row align-center">
            <span class="avatar" style="background:${TONES[card.tone]}">${esc(card.initials)}</span>
            <span class="grow"><strong>${esc(card.label)}</strong><span class="muted small">${esc(card.detail)}</span></span>
          </div>
          <label class="field small-label">Voice
            <select data-speaker="${index}" data-speaker-field="voice">${panel.voices.map(([label, key]) => `
              <option value="${esc(key)}" ${key === card.voice ? "selected" : ""}>${esc(label)}</option>`).join("")}</select>
          </label>
          <label class="slider-row compact"><span>Volume</span>
            <input type="range" min="0" max="200" step="5" value="${card.volume}" data-speaker="${index}" data-speaker-field="volume"
              aria-label="${esc(card.label)} volume">
            <output class="mono" data-volume-out="${index}">${Math.round(card.volume)}%</output></label>
        </div>`).join("")}
      </div>
    </section>`;
}

function slider(key, label, min, max, step, value, format, hint = "") {
  const shown = format === "x" ? `${Number(value).toFixed(2)}×` : format === "%" ? `${value}%` : `${value}`;
  return `
    <label class="slider-row"><span>${label}</span>
      <input type="range" min="${min}" max="${max}" step="${step}" value="${value}" data-setting="${key}" data-format="${format}">
      <output class="mono" data-output="${key}">${shown}</output></label>
    ${hint ? `<span class="muted small slider-hint">${hint}</span>` : ""}`;
}

// The timings in the order page.py lists them: Smart, Match the original, Natural.
const matchKey = (app) => app.boot.timings[1].key;

// An automatic timing default applies to generation without changing the saved settings.
export function selectedTiming(app) {
  return app.session?.same && !app.session.rewrite && !app.timingTouched ? matchKey(app) : app.settings.timing;
}

// Slot, take, next slot: how each timing fits a take that runs long.
const TIMING_PICTURES = ["take-smart", "take-match", "take-natural"];

function timingPicture(index) {
  return `<span class="timing-pic ${TIMING_PICTURES[index] || ""}" aria-hidden="true"><span class="slot"></span><span class="take"></span><span class="slot next"></span></span>`;
}

function settings(app) {
  const s = app.settings;
  const voiceModes = app.session?.voice_modes || app.boot.voice_modes;
  const match = matchKey(app);
  const timing = selectedTiming(app);
  return `
    ${speakers(app)}
    <section class="card" aria-labelledby="set-h">
      <div class="card-pad card-title"><h2 id="set-h">Voices and timing</h2>
        <span class="muted small">Changes apply to the next dub. Unchanged lines keep their takes.</span></div>
      <div class="sec">
        <h3>Voices</h3>
        <div class="row row-wrap gap-32">
          <div class="field">Clone from
            ${seg("voice_mode", voiceModes.map((mode) => [mode, mode === "One voice per speaker" ? "One per speaker" : mode]), s.voice_mode, {label: "Clone from"})}
            <span class="muted small">${{
              "Each line's own voice": "Keeps each line's delivery. Lines under 2 s use their speaker's clip.",
              "Nearest clean voice": "Clones overlapping or short lines from the same speaker's closest solo passage. Needs accurate speaker labels.",
              "Separate overlapping voices": "Uses isolated speech from the line when it can verify the speaker. Otherwise uses the nearest clean voice.",
              "One voice per speaker": "Each speaker's best clip (3–15 s). Steadier.",
            }[s.voice_mode] || ""}</span>
          </div>
          <div class="field">Delivery ${seg("preset", app.boot.presets.map((p) => [p, p]), s.preset, {label: "Delivery"})}</div>
        </div>
      </div>
      <div class="sec">
        <h3>Timing</h3>
        <div class="pick-grid three" role="radiogroup" aria-label="Timing">${app.boot.timings.map((t, index) => `
          <button type="button" role="radio" class="pick pick-timing" aria-checked="${t.key === timing}" data-action="timing" data-key="${esc(t.key)}">
            <span class="dot"></span><span><strong>${esc(t.label)}</strong>${timingPicture(index)}<span class="muted small">${esc(t.desc)}</span></span>
          </button>`).join("")}
        </div>
        ${timing !== match ? slider("max_speed", "Maximum speed-up", 1, 2, 0.05, s.max_speed, "x",
          "Takes still too long at this speed overlap the next line.") : ""}
      </div>
    </section>
    <section class="card" aria-labelledby="mix-h">
      <div class="sec first">
        <h2 id="mix-h">Mix and output</h2>
        ${slider("speech_volume", "Dub volume", 0, 200, 5, s.speech_volume, "%")}
        ${slider("background_volume", "Music and effects", 0, 200, 5, s.background_volume, "%")}
        <div class="row row-wrap gap-28">
          <label class="check"><input type="checkbox" data-setting="keep_original" ${s.keep_original ? "checked" : ""}>
            <span>Keep the original audio as a second track</span></label>
          <label class="check"><input type="checkbox" data-setting="word_timing" ${s.word_timing ? "checked" : ""}>
            <span>Word-by-word subtitles <span class="muted small">· to add it to a finished dub, turn it on and dub again</span></span></label>
        </div>
      </div>
      <details class="sec">
        <summary><strong>Advanced</strong> <span class="muted small">${s.parallel} parallel requests ·
          ${String(s.regenerate || "").trim() ? `fresh takes for ${esc(s.regenerate)}` : "no forced fresh takes"}</span></summary>
        <div class="stack-14 top-gap">
          <label class="field narrow">Fresh takes for lines
            <input data-setting="regenerate" value="${esc(s.regenerate)}" placeholder="e.g. 4, 9-10">
            <span class="muted small">These lines get new takes even when unchanged.</span></label>
          ${slider("parallel", "Parallel requests", 1, 32, 1, s.parallel, "")}
        </div>
      </details>
    </section>`;
}

function readiness(app) {
  const r = app.session.readiness;
  const same = app.session.same;
  const s = app.settings;
  const timing = app.boot.timings.find((t) => t.key === selectedTiming(app));
  const warnings = [];
  const rushed = [...new Set([...r.long, ...r.rushed])].sort((a, b) => a - b);
  if (rushed.length) warnings.push(["warn", `${rushed.length === 1 ? "Line" : "Lines"} <b>${ranges(rushed)}</b> may sound rushed`, true]);
  if (r.failed.length) warnings.push(["fail", `${r.failed.length === 1 ? "Line" : "Lines"} <b>${ranges(r.failed)}</b>: take failed; ${r.failed.length === 1 ? "it gets" : "they get"} a fresh take`, false]);
  if (r.untranslated && !same) warnings.push(["warn", `${plural(r.untranslated, "line")} untranslated; ${r.untranslated === 1 ? "it keeps its" : "they keep their"} voice`, true]);
  if (r.issues.length) warnings.push(["warn", `Check ${r.issues.length === 1 ? "line" : "lines"} <b>${ranges(r.issues)}</b>: may contain explanations`, true]);
  return `
    <aside class="rail">
      <section class="card card-pad stack-14" aria-labelledby="go-h">
        <h2 id="go-h">Dub it</h2>
        <p class="muted small">${same ? `Keeping <strong>${esc(app.session.spoken || "the spoken language")}</strong>: only edited lines are voiced.`
          : `Dubbing into <strong>${esc(app.session.target_language || "the target language")}</strong>.`}</p>
        <div class="stats">
          <div class="stat ok"><b>${r.voiced}</b><span>${same ? "lines voiced" : "voiced"}</span></div>
          <div class="stat"><b>${r.kept}</b><span>${same ? "original audio" : "original voice"}</span></div>
          <div class="stat"><b>${r.speakers}</b><span>${r.speakers === 1 ? "speaker" : "speakers"}</span></div>
        </div>
        ${warnings.length ? `<span class="rail-label">Before you dub</span><ul class="warnings">${warnings.map(([kind, text, fix]) => `
          <li class="${kind}">${icon("warn", 18)}<span class="grow">${text}</span>
            ${fix ? '<button type="button" class="linkish" data-action="to-review">Fix</button>' : ""}</li>`).join("")}</ul>` : ""}
        <div class="recap small">
          <span><strong>${esc(s.voice_mode)}</strong> · ${esc(s.preset)} delivery</span>
          <span>${esc(timing?.label || "")}${timing?.key !== matchKey(app) ? ` · up to ${Number(s.max_speed).toFixed(2)}×` : ""} · mix ${s.speech_volume}% / ${s.background_volume}%</span>
          <span>${s.keep_original ? "Original audio kept as a second track" : "No original track"} · word subtitles ${s.word_timing ? "on" : "off"}</span>
        </div>
        ${app.boot.gpu ? `
          <div class="run-status" data-run="dub" hidden><span class="spinner" aria-hidden="true"></span><span data-run-status role="status"></span></div>
          <div class="row">
            <button type="button" class="btn btn-primary btn-large grow" data-action="dub" data-busy="dub"
              ${r.voiced ? "" : 'data-locked="true" disabled'}>${same ? `Generate ${plural(r.voiced, "line")}` : "Generate dub"}</button>
            <button type="button" class="btn" data-stop="dub" data-action="interrupt" disabled>Interrupt</button>
          </div>
          <p class="muted small">Only changed lines are voiced; every other line reuses its saved take. Every dub is also saved to Media.</p>`
        : '<p class="alert alert-info">Voices are generated in Studio. Save the project, then load it in Studio to dub.</p>'}
      </section>
      ${latest(app)}
    </aside>`;
}

export function renderDub(app) {
  if (!app.session) return "";
  return `
    <div class="stack-24 wide">
      <div class="layout">
        <div class="main-col stack-16">${settings(app)}</div>
        ${readiness(app)}
      </div>
    </div>`;
}

async function saveSpeaker(app, index) {
  const voice = document.querySelector(`[data-speaker="${index}"][data-speaker-field="voice"]`).value;
  const volume = Number(document.querySelector(`[data-speaker="${index}"][data-speaker-field="volume"]`).value);
  try {
    const data = await api.post("speaker", {session: app.session.id, index, voice, volume});
    app.setSession(data.session);
  } catch (error) {
    toast(error.message, {error: true});
    app.render();
  }
}

export function bindDub(app) {
  const screen = document.getElementById("screen");
  screen.addEventListener("click", async (event) => {
    if (app.screen !== "dub" || !app.session) return;
    const target = event.target.closest("[data-action]");
    if (!target) return;
    const action = target.dataset.action;
    const s = app.settings;
    if (action === "timing") {
      s.timing = target.dataset.key;
      app.timingTouched = true;
      app.render();
    } else if (action === "to-result") {
      app.go("result");
    } else if (action === "to-review") {
      app.filter = "attention";
      app.go("review");
    } else if (action === "dub") {
      app.startRun("dub", {session: app.session.id, voice_mode: s.voice_mode, preset: s.preset, timing: selectedTiming(app),
        max_speed: s.max_speed, parallel: s.parallel, speech_volume: s.speech_volume,
        background_volume: s.background_volume, keep_original: s.keep_original, regenerate: s.regenerate,
        word_timing: s.word_timing});
    } else if (action === "interrupt") {
      try {
        toast((await api.post("interrupt", {kind: "dub", session: app.session.id})).message);
      } catch (error) {
        toast(error.message, {error: true});
      }
    }
  });
  screen.addEventListener("input", (event) => {
    if (app.screen !== "dub") return;
    const field = event.target;
    if (field.dataset.speakerField === "volume") {
      document.querySelector(`[data-volume-out="${field.dataset.speaker}"]`).textContent = `${field.value}%`;
    }
  });
  screen.addEventListener("change", (event) => {
    if (app.screen !== "dub") return;
    if (event.target.dataset.speakerField) saveSpeaker(app, Number(event.target.dataset.speaker));
  });
}
