// Step 2, Review lines: the board, a timeline per speaker, the inspector and the line actions.
// Edits save as soon as a field is left; the server answers with the session as saved.

import {api} from "./api.js";
import {clock, copyText, esc, icon, plural, promptOptions, ranges, rich, toast} from "./ui.js";

const TONES = ["#c92a2a", "#1864ab", "#2b8a3e", "#6741d9", "#d9480f", "#9c36b5", "#c2255c", "#a07800"];

function editTime(seconds) {
  const ms = Math.round(seconds * 1000);
  return `${Math.floor(ms / 60000)}:${((ms % 60000) / 1000).toFixed(3).padStart(6, "0")}`;
}

function flagOf(line, same) {
  const flags = line.flags;
  if (flags.includes("failed")) return ["Take failed", "flag-fail"];
  if (flags.includes("empty")) return [same ? "No new text" : "Not translated", "flag-warn"];
  if (flags.includes("translation")) return ["Check translation", "flag-warn"];
  if (flags.includes("rushed")) return ["Too long", "flag-warn"];
  if (flags.includes("long")) return ["May be too long", "flag-warn"];
  if (flags.includes("off")) return [same ? "Original audio" : "Original voice", "flag-quiet"];
  if (same) return ["Voiced · edited", "flag-ok"];
  return null;
}

const needsAttention = (line) => line.flags.some((flag) => flag !== "off");

function filters(app) {
  const session = app.session;
  const same = session.same;
  const list = [["all", "All"], ["attention", "Needs attention"]];
  if (same) list.push(["edited", "Edited"]);
  else list.push(["untranslated", "Untranslated"]);
  if (session.board.lines.some((line) => line.flags.includes("failed"))) list.push(["failed", "Failed takes"]);
  if (app.mergeGroups?.length) list.push(["combine", "To combine"]);
  if (session.board.speakers.length > 1) list.push(...session.board.speakers.map((s) => [`speaker:${s.k}`, s.label]));
  return list;
}

function visible(app, line, index) {
  const f = app.filter;
  if (f === "attention") return needsAttention(line);
  if (f === "untranslated") return !line.translation;
  if (f === "edited") return line.dub && line.translation && line.edited;
  if (f === "failed") return line.flags.includes("failed");
  if (f === "combine") return groupOf(app, index) !== null;
  if (f.startsWith("speaker:")) return line.speaker === Number(f.slice(8));
  return true;
}

function summary(app) {
  const lines = app.session.board.lines;
  const r = app.session.readiness;
  if (app.session.same) {
    return `
      <span><b class="big ok">${r.voiced}</b> edited · will be voiced</span>
      <span><b class="big">${r.kept}</b> unchanged · original audio</span>
      ${r.failed.length ? `<span><b class="big fail">${r.failed.length}</b> take failed</span>` : ""}
      ${r.long.length ? `<span><b class="big warn">${r.long.length}</b> may be too long</span>` : ""}
      ${r.issues.length ? `<span><b class="big warn">${r.issues.length}</b> to check</span>` : ""}`;
  }
  return `
    <span><b class="big">${r.voiced}</b> ready</span>
    <span><b class="big">${app.session.board.speakers.length}</b> ${app.session.board.speakers.length === 1 ? "speaker" : "speakers"}</span>
    ${r.untranslated ? `<span><b class="big warn">${r.untranslated}</b> untranslated</span>` : ""}
    ${r.long.length ? `<span><b class="big warn">${r.long.length}</b> may be too long</span>` : ""}
    ${r.issues.length ? `<span><b class="big warn">${r.issues.length}</b> to check</span>` : ""}
    ${r.failed.length ? `<span><b class="big fail">${r.failed.length}</b> take failed</span>` : ""}
    <span class="muted">${plural(lines.length, "line")}</span>`;
}

function timeline(app) {
  const board = app.session.board;
  const duration = Math.max(1, board.duration);
  const ticks = 5;
  return `
    <section class="card timeline" aria-label="Speaker timeline">
      <div class="lane-grid">
        <span></span>
        <div class="ticks mono">${Array.from({length: ticks}, (_, i) => `<span>${clock(duration * i / (ticks - 1), false)}</span>`).join("")}</div>
        ${board.speakers.map((speaker) => `
          <span class="lane-name"><span class="swatch" style="background:${TONES[speaker.tone]}"></span>${esc(speaker.label)}</span>
          <div class="lane">${board.lines.map((line, index) => line.speaker !== speaker.k ? "" : `
            <button type="button" class="block ${needsAttention(line) ? "stripe" : ""} ${line.dub ? "" : "faded"} ${index === app.focus ? "focused" : ""}"
              style="left:${line.s / duration * 100}%;width:${Math.max(0.3, (line.e - line.s) / duration * 100)}%;background-color:${TONES[speaker.tone]}"
              data-action="focus" data-line="${index}" aria-label="Line ${index + 1}"></button>`).join("")}
          </div>`).join("")}
      </div>
    </section>`;
}

// The detected group (its position in app.mergeGroups) a line belongs to, or null.
function groupOf(app, index) {
  const found = (app.mergeGroups || []).findIndex((group) => group.includes(index));
  return found < 0 ? null : found;
}

// The detected groups as the viewer left them: each keeps its picked lines that are still next to
// each other, so unticking a line splits or drops its group. Runs of fewer than two lines are dropped.
function pickedGroups(app) {
  const groups = [];
  for (const group of app.mergeGroups || []) {
    let run = [];
    for (const index of group) {
      if (app.picks.has(index) && (!run.length || index === run.at(-1) + 1)) run.push(index);
      else {
        if (run.length > 1) groups.push(run);
        run = app.picks.has(index) ? [index] : [];
      }
    }
    if (run.length > 1) groups.push(run);
  }
  return groups;
}

// Two or more picked lines next to each other, all of one speaker, can become one line.
function mergeable(app) {
  const picks = [...app.picks].sort((a, b) => a - b);
  const lines = app.session.board.lines;
  if (picks.length < 2 || picks.some((n, i) => i && n !== picks[i - 1] + 1)) return false;
  return picks.every((n) => lines[n] && lines[n].speakerId === lines[picks[0]].speakerId);
}

function board(app) {
  const lines = app.session.board.lines;
  const same = app.session.same;
  const picks = [...app.picks];
  const shown = lines.map((line, index) => [line, index]).filter(([line, index]) => visible(app, line, index));
  return `
    <section class="card board" aria-label="Lines">
      ${picks.length ? `
        <div class="picks-bar">
          <strong>${picks.length} picked</strong>
          <span class="muted small">Line actions use these; Listen and Preview play the first.</span>
          <span class="spacer"></span>
          ${pickedGroups(app).length ? `<button type="button" class="btn btn-small btn-primary" data-action="merge-groups" title="Join each detected group into one line">Combine ${plural(pickedGroups(app).length, "group")}</button>`
            : mergeable(app) ? '<button type="button" class="btn btn-small" data-action="merge-lines" title="Join these lines into one line">Combine</button>' : ""}
          <button type="button" class="btn btn-small" data-action="translate" data-busy="lines">${againLabel(app)}</button>
          ${app.boot.gpu ? '<button type="button" class="btn btn-small" data-action="take" data-busy="lines">New takes</button>' : ""}
          <button type="button" class="btn btn-small" data-action="copy-lines-prompt">Copy ${same ? "rewrite" : "translation"} prompt</button>
          <button type="button" class="btn btn-small" data-action="clear-picks">Clear</button>
        </div>` : ""}
      <div class="line-row head">
        <span><input type="checkbox" data-action="pick-all" aria-label="Pick every shown line" ${shown.length && shown.every(([, i]) => app.picks.has(i)) ? "checked" : ""}></span>
        <span>#</span><span>Time</span><span>${same ? "What was said → new text" : "Line"}</span><span></span><span>Dub</span>
      </div>
      <div class="line-list">${shown.length ? shown.map(([line, index]) => {
        const flag = flagOf(line, same);
        return `
        <div class="line-row ${index === app.focus ? "selected" : ""} ${groupOf(app, index) !== null ? `combine-group combine-${groupOf(app, index) % 2}` : ""}">
          <span><input type="checkbox" data-action="pick" data-line="${index}" ${app.picks.has(index) ? "checked" : ""} aria-label="Pick line ${index + 1}"></span>
          <span class="mono muted">${index + 1}</span>
          <span class="mono muted small">${clock(line.s)}<br>${(line.e - line.s).toFixed(1)} s</span>
          <button type="button" class="line-text" data-action="focus" data-line="${index}" aria-label="Edit line ${index + 1}">
            <span class="line-meta"><span class="speaker" style="color:${TONES[line.tone]}">${esc(line.label)}</span>
              <span class="muted small ellipsis">${esc(line.source)}</span></span>
            <span class="${line.translation ? (line.dub ? "" : "muted") : "missing"}">${esc(line.translation || (same ? "No new text" : "Not translated yet"))}</span>
            ${line.problem ? `<span class="small fail-text">${esc(line.problem)}</span>` : ""}
            ${line.check ? `<span class="small warn-text">${esc(line.check)}</span>` : ""}
          </button>
          <span>${flag ? `<span class="flag ${flag[1]}">${flag[0]}</span>` : ""}</span>
          <button type="button" class="switch" role="switch" aria-checked="${line.dub}" aria-label="Dub line ${index + 1}"
            data-action="dub" data-line="${index}"></button>
        </div>`;
      }).join("") : '<p class="empty muted">No line matches this filter.</p>'}</div>
    </section>`;
}

function table(app) {
  const same = app.session.same;
  return `
    <section class="card board" aria-label="Every line as a table">
      <div class="card-head"><h2>Edit as a table</h2>
        <span class="muted small">Changes save as you leave a cell. A new row saves once it has a start, an end and text.</span>
        <span class="spacer"></span>
        <button type="button" class="btn btn-small" data-action="table-add">Add row</button>
        <button type="button" class="btn btn-small" data-action="table-close">Back to the board</button></div>
      <div class="table-wrap"><table class="lines-table">
        <thead><tr><th>#</th><th>Start</th><th>End</th><th>Speaker</th><th>${same ? "What was said" : "Original"}</th>
          <th>${same ? "New text" : "Translation"}</th><th>Dub</th></tr></thead>
        <tbody>${app.session.board.lines.map((line, index) => `
          <tr data-row="${index}">
            <td class="mono">${index + 1}</td>
            <td><input class="mono" data-cell="1" value="${esc(editTime(line.s))}" aria-label="Start of line ${index + 1}"></td>
            <td><input class="mono" data-cell="2" value="${esc(editTime(line.e))}" aria-label="End of line ${index + 1}"></td>
            <td><input data-cell="3" value="${esc(line.speakerId)}" aria-label="Speaker of line ${index + 1}"></td>
            <td><textarea data-cell="4" rows="2" aria-label="Original of line ${index + 1}">${esc(line.source)}</textarea></td>
            <td><textarea data-cell="5" rows="2" aria-label="New text of line ${index + 1}">${esc(line.translation)}</textarea></td>
            <td><input type="checkbox" data-cell="6" ${line.dub ? "checked" : ""} aria-label="Dub line ${index + 1}"></td>
          </tr>`).join("")}
          ${(app.newRows || []).map((row, n) => `
          <tr data-new="${n}">
            <td class="mono muted">new</td>
            <td><input class="mono" data-cell="1" value="${esc(row[1])}" placeholder="0:00.0" aria-label="Start"></td>
            <td><input class="mono" data-cell="2" value="${esc(row[2])}" placeholder="0:00.0" aria-label="End"></td>
            <td><input data-cell="3" value="${esc(row[3])}" aria-label="Speaker"></td>
            <td><textarea data-cell="4" rows="2" aria-label="Original">${esc(row[4])}</textarea></td>
            <td><textarea data-cell="5" rows="2" aria-label="New text">${esc(row[5])}</textarea></td>
            <td><input type="checkbox" data-cell="6" ${row[6] ? "checked" : ""} aria-label="Dub"></td>
          </tr>`).join("")}
        </tbody></table></div>
    </section>`;
}

const againLabel = (app) => (app.session.same ? (app.settings.rewrite ? "Rewrite again" : "Copy again") : "Translate again");

function fit(line, same) {
  const match = /about ([\d.]+)×/.exec(line.fit || "");
  const speed = match ? Number(match[1]) : null;
  const pct = speed ? Math.min(speed / 2.56, 1) * 100 : 0;
  return `
    <div class="fit">
      <p class="small">${rich(line.fit || "")}</p>
      ${speed ? `<div class="meter" aria-hidden="true"><div class="meter-fill ${speed > 1.6 ? "over" : ""}" style="width:${pct}%"></div>
        <div class="meter-mark" title="1.6×"></div></div>
        <span class="muted small">Mark = 1.6×, where lines start to sound rushed${same ? "" : ""}</span>` : ""}
    </div>`;
}

function inspector(app) {
  const lines = app.session.board.lines;
  const line = lines[app.focus];
  const same = app.session.same;
  if (!line) return '<aside class="card inspector empty"><p class="muted">Click a line or the timeline.</p></aside>';
  const flag = flagOf(line, same);
  const speakers = [...new Set(lines.map((item) => item.speakerId))];
  const previewSrc = app.previewRun ? api.url("run-audio", {id: app.previewRun}) : "";
  const originalSrc = app.originalLine === app.focus ? api.url("original", {session: app.session.id, line: app.focus, t: Date.now()}) : "";
  return `
    <aside class="card inspector" aria-label="Line inspector">
      <div class="card-head">
        <h2>Line ${app.focus + 1}</h2>
        <span class="speaker small" style="color:${TONES[line.tone]}">${esc(line.label)}</span>
        ${flag ? `<span class="flag ${flag[1]}">${flag[0]}</span>` : ""}
        <span class="spacer"></span>
        <button type="button" class="btn btn-icon" data-action="step" data-step="-1" aria-label="Previous line">${icon("prev")}</button>
        <button type="button" class="btn btn-icon" data-action="step" data-step="1" aria-label="Next line">${icon("next")}</button>
      </div>
      <div class="card-body stack-14" data-inspector="${app.focus}">
        ${line.problem ? `<div class="alert alert-fail" role="alert"><strong>Take failed, so the original voice was kept.</strong>
          ${esc(line.problem)} Use New take, then dub again.</div>` : ""}
        ${line.check ? `<div class="alert alert-warn" role="status">${esc(line.check)} Edit the line or translate it again; editing clears this.</div>` : ""}
        <div class="grid-3">
          <label class="field">Start<input class="mono" data-line-field="start" value="${esc(editTime(line.s))}"></label>
          <label class="field">End<input class="mono" data-line-field="end" value="${esc(editTime(line.e))}"></label>
          <label class="field">Speaker<input data-line-field="speaker" list="speaker-list" value="${esc(line.speakerId)}">
            <datalist id="speaker-list">${speakers.map((s) => `<option value="${esc(s)}"></option>`).join("")}</datalist></label>
        </div>
        <span class="muted small">Fields save when you leave them; times also on Enter. Pick a speaker or type a new name.</span>
        <label class="field">${same ? "What was said" : "Original"}<textarea data-line-field="original" rows="2">${esc(line.source)}</textarea></label>
        <label class="field">${same ? "New text" : "Translation"}<textarea data-line-field="translation" rows="3"
          placeholder="${same ? "Same as what was said" : "Not translated yet"}">${esc(line.translation)}</textarea></label>
        ${same && line.edited && line.translation ? '<button type="button" class="linkish small" data-action="revert">Revert to what was said</button>' : ""}
        ${fit(line, same)}
        <div class="grid-2">
          <button type="button" class="btn" data-action="listen">${icon("speaker")} Original</button>
          ${app.boot.gpu ? `<button type="button" class="btn btn-primary" data-action="preview" data-busy="lines">${icon("play", 14)} Preview dub</button>
          <button type="button" class="btn" data-action="take" data-busy="lines">New take</button>` : ""}
          <button type="button" class="btn" data-action="translate" data-busy="lines">${againLabel(app)}</button>
        </div>
        <div class="row align-center">
          <div class="run-status grow" data-run="lines" hidden><span class="spinner" aria-hidden="true"></span><span data-run-status role="status"></span></div>
          <button type="button" class="btn btn-small" data-stop="lines" data-action="interrupt" data-kind="lines" disabled>Interrupt</button>
        </div>
        ${app.boot.gpu ? "" : '<p class="small info-text">Preview and New take run in Studio.</p>'}
        <div class="players">
          <label class="player-row"><span>Original segment</span><audio controls preload="none" src="${originalSrc}" data-player="original"></audio></label>
          ${app.boot.gpu ? `<label class="player-row"><span>Dub preview</span><audio controls preload="none" src="${previewSrc}" data-player="preview"></audio></label>` : ""}
        </div>
        <label class="check">
          <button type="button" class="switch" role="switch" aria-checked="${line.dub}" data-action="dub" data-line="${app.focus}" aria-label="Dub this line"></button>
          <span>Dub this line <span class="muted">· off keeps the original ${same ? "audio" : "voice"}</span></span>
        </label>
        <details class="sub">
          <summary><strong>${same ? "Rewrite" : "Translate"} by hand in a chat model</strong></summary>
          <div class="stack-12 top-gap">
            <p class="muted small">Copies the picked lines, or every line, with their times, style and notes. No API key.
              Replies reach the copied lines even after other edits.</p>
            ${promptOptions(app.settings, {same})}
            <button type="button" class="btn btn-small" data-action="copy-lines-prompt">Copy ${same ? "rewrite" : "translation"} prompt</button>
            <label class="field">Reply<textarea class="mono" rows="4" data-action="lines-reply" placeholder='["First line", "Second line"]'>${esc(app.linesReply || "")}</textarea></label>
            <button type="button" class="btn btn-small" data-action="apply-lines-reply">Apply pasted ${same ? "lines" : "translations"}</button>
          </div>
        </details>
        <div class="row top-line">
          <button type="button" class="btn btn-small" data-action="add-line">Add a line after</button>
          <span class="spacer"></span>
          <button type="button" class="btn btn-small danger" data-action="delete-line">Delete line</button>
        </div>
      </div>
    </aside>`;
}

export function renderReview(app) {
  if (!app.session) return "";
  const session = app.session;
  const same = session.same;
  const fillable = !same && session.readiness.untranslated;
  return `
    <div class="stack-16 wide">
      <div class="summary-row">
        <div class="counts">${summary(app)}</div>
        <span class="spacer"></span>
        <div class="seg" role="group" aria-label="Show">${filters(app).map(([key, label]) => `
          <button type="button" data-action="filter" data-filter="${esc(key)}" aria-pressed="${app.filter === key}">${esc(label)}</button>`).join("")}</div>
        ${fillable ? `<button type="button" class="btn btn-small" data-action="fill" data-busy="lines">Fill untranslated (${session.readiness.untranslated})</button>` : ""}
        ${session.board.lines.some((line) => line.translation && line.dub) ? `<button type="button" class="btn btn-small" data-action="check-translations" ${app.checking ? "disabled" : ""}
          title="Ask Gemini Decision whether each ${same ? "rewritten" : "translated"} line keeps its meaning and sounds natural. Uses the picked lines, or every line.">${app.checking ? "Checking…" : same ? "Check rewrites" : "Check translations"}</button>` : ""}
        <span class="combine-tools">
          <button type="button" class="btn btn-small" data-action="detect-combine" ${app.detecting ? "disabled" : ""}
            title="Pick neighbouring lines that read as one sentence, to review and combine">${app.detecting ? "Detecting…" : "Auto-detect combine"}</button>
          <label class="check small" title="Confirm each pair with Gemini Decision from the lines around it, so missing or wrong speaker labels don't decide. Off, or without a Gemini key, fixed rules decide and trust the speaker labels.">
            <input type="checkbox" data-action="combine-context" ${app.settings.combine_context ? "checked" : ""}> By context</label>
        </span>
        <button type="button" class="btn btn-small" data-action="table-open">Edit as a table</button>
        <button type="button" class="btn btn-small btn-primary" data-action="to-dub">Next: dub →</button>
      </div>
      ${timeline(app)}
      <div class="layout">
        <div class="main-col">${app.table ? table(app) : board(app)}</div>
        ${inspector(app)}
      </div>
    </div>`;
}

// ---------------------------------------------------------------------------

function keyOf(line, index) {
  return {index, start: line.s, speaker: line.speakerId, source: line.source};
}

// Serialize blur saves. While newer typing is still in the DOM, leave that editor mounted.
let saveQueue = Promise.resolve();
let editVersion = 0;
const savedInspectorLines = new WeakMap();

function enqueueSave(save) {
  const version = ++editVersion;
  saveQueue = saveQueue.then(() => save(() => version === editVersion))
    .catch((error) => toast(error.message, {error: true}));
}

function saveLine(app, fields) {
  const box = document.querySelector("[data-inspector]");
  if (!box) return;
  const index = Number(box.dataset.inspector);
  const session = app.session.id;
  const initial = {line: app.session.board.lines[index], index};
  enqueueSave(async (canRedraw) => {
    // A preceding save can change the source, speaker or position used to identify this line.
    const saved = savedInspectorLines.get(box) || initial;
    const {line} = saved;
    const key = keyOf(line, saved.index);
    const data = await api.patch("line", {session, key, start: line.s, end: line.e, speaker: line.speakerId,
      original: line.source, translation: line.translation, dub: line.dub, ...fields});
    savedInspectorLines.set(box, {index: data.focus, line: data.session.board.lines[data.focus]});
    if (app.sessionId !== session) return;
    app.setSession(data.session, {focus: box.isConnected ? data.focus : undefined});
    if (!canRedraw()) return;
    const next = document.activeElement?.dataset?.lineField;
    const caret = document.activeElement?.selectionStart;
    app.render();
    const field = next && document.querySelector(`[data-line-field="${next}"]`);
    if (field) {
      field.focus();
      if (typeof caret === "number") field.setSelectionRange(caret, caret);
    }
  });
}

// Returns whether the edit was saved; a refusal is reported and leaves the page as it was.
async function edit(app, path, body) {
  try {
    // The revision names the lines these positions were drawn from; the server refuses stale ones.
    const data = await api.post(path, {session: app.session.id, rev: app.session.board.rev, ...body});
    app.setSession(data.session, {focus: data.focus});
    if (data.status) toast(data.status.replace(/\*\*/g, ""));
    app.render();
    return true;
  } catch (error) {
    toast(error.message, {error: true});
    // Show the lines as saved now, so a refused action can be retried on what is really there.
    await app.refresh();
    return false;
  }
}

function lineNumbers(app) {
  return app.picks.size ? [...app.picks].sort((a, b) => a - b) : [app.focus];
}

function translateBody(app, picked) {
  const s = app.settings;
  const same = app.session.same;
  return {session: app.session.id, picked, target_language: app.session.target_language || s.target_language,
    translator: s.translator, gemini_model: s.gemini_model, api_key: app.apiKey, notes: s.notes,
    translation_context: s.translation_context, style: same && s.rewrite ? s.rewrite_style : s.style,
    rewrite: same && s.rewrite && !s.rewrite_by_hand, shorten_lines: s.shorten_lines, readings: s.readings};
}

function tableRows(app) {
  const rows = [];
  for (const tr of document.querySelectorAll(".lines-table tbody tr")) {
    const cell = (n) => tr.querySelector(`[data-cell="${n}"]`);
    rows.push([tr.dataset.row !== undefined ? Number(tr.dataset.row) + 1 : null, cell(1).value, cell(2).value,
      cell(3).value, cell(4).value, cell(5).value, cell(6).checked]);
  }
  return rows;
}

function saveTable(app) {
  const rows = tableRows(app);
  const session = app.session.id;
  enqueueSave(async (canRedraw) => {
    const data = await api.put("lines", {session, rows});
    if (app.sessionId !== session) return;
    app.setSession(data.session);
    app.newRows = data.unfinished;
    if (data.status) toast(data.status.replace(/\*\*/g, ""));
    if (!canRedraw()) return;
    const next = document.activeElement?.closest(".lines-table [data-cell]");
    const row = next?.closest("tr").dataset;
    app.render();
    if (next) {
      const selector = row.row !== undefined ? `[data-row="${row.row}"]` : `[data-new="${row.new}"]`;
      document.querySelector(`.lines-table tr${selector} [data-cell="${next.dataset.cell}"]`)?.focus();
    }
  });
}

export function bindReview(app) {
  const screen = document.getElementById("screen");
  screen.addEventListener("click", async (event) => {
    if (app.screen !== "review" || !app.session) return;
    const target = event.target.closest("[data-action]");
    if (!target) return;
    const action = target.dataset.action;
    const index = Number(target.dataset.line);
    if (action === "focus") {
      app.focus = index;
      app.render();
    } else if (action === "pick") {
      if (target.checked) app.picks.add(index); else app.picks.delete(index);
      app.render();
    } else if (action === "pick-all") {
      const shown = app.session.board.lines.map((line, i) => [line, i]).filter(([line, i]) => visible(app, line, i)).map(([, i]) => i);
      if (target.checked) shown.forEach((i) => app.picks.add(i)); else shown.forEach((i) => app.picks.delete(i));
      app.render();
    } else if (action === "clear-picks") {
      app.picks.clear();
      app.mergeGroups = [];
      if (app.filter === "combine") app.filter = "all";
      app.render();
    } else if (action === "combine-context") {
      app.settings.combine_context = target.checked;
    } else if (action === "detect-combine") {
      app.detecting = true;
      app.render();
      try {
        const data = await api.post("combine-suggestions", {session: app.session.id,
          context: app.settings.combine_context, api_key: app.apiKey});
        app.mergeGroups = data.groups;
        app.mergeRelabel = data.context;
        app.picks = new Set(data.groups.flat());
        if (data.groups.length) app.filter = "combine";
        toast(data.groups.length ? `${data.status} Untick any to keep apart, then Combine.` : data.status);
      } catch (error) {
        toast(error.message, {error: true});
      } finally {
        app.detecting = false;
        app.render();
      }
    } else if (action === "merge-groups") {
      // The picks and groups are cleared only once the lines are combined, so a refusal keeps them.
      if (await edit(app, "line/merge-groups", {groups: pickedGroups(app), relabel: app.mergeRelabel})) {
        app.picks.clear();
        app.mergeGroups = [];
        if (app.filter === "combine") app.filter = "all";
        app.render();
      }
    } else if (action === "check-translations") {
      app.checking = true;
      app.render();
      try {
        const data = await api.post("check-translations", {session: app.session.id, picked: [...app.picks], api_key: app.apiKey});
        app.setSession(data.session);
        toast(data.status);
      } catch (error) {
        toast(error.message, {error: true});
      } finally {
        app.checking = false;
        app.render();
      }
    } else if (action === "filter") {
      app.filter = target.dataset.filter;
      app.render();
    } else if (action === "dub") {
      edit(app, "line/dub", {line: index, dub: target.getAttribute("aria-checked") !== "true"});
    } else if (action === "step") {
      app.focus = Math.min(Math.max(0, app.focus + Number(target.dataset.step)), app.session.board.lines.length - 1);
      app.render();
    } else if (action === "revert") {
      saveLine(app, {translation: app.session.board.lines[app.focus].source});
    } else if (action === "listen") {
      app.originalLine = lineNumbers(app)[0];
      app.focus = app.originalLine;
      app.render();
      document.querySelector('[data-player="original"]')?.play().catch(() => {});
    } else if (action === "preview" || action === "take") {
      const s = app.settings;
      app.previewRun = null;
      app.startRun("preview", {session: app.session.id, line: lineNumbers(app)[0], fresh: action === "take",
        voice_mode: s.voice_mode, preset: s.preset, timing: s.timing, max_speed: s.max_speed, parallel: s.parallel});
    } else if (action === "translate") {
      app.startRun("translate", translateBody(app, [...app.picks]));
    } else if (action === "fill") {
      app.startRun("translate", translateBody(app, []));
    } else if (action === "interrupt") {
      try {
        toast((await api.post("interrupt", {kind: "lines", session: app.session.id})).message);
      } catch (error) {
        toast(error.message, {error: true});
      }
    } else if (action === "copy-lines-prompt") {
      const s = app.settings;
      try {
        const data = await api.post("lines-prompt", {session: app.session.id, picked: [...app.picks],
          target_language: app.session.target_language || s.target_language, notes: s.notes,
          style: app.session.same && s.rewrite ? s.rewrite_style : s.style, shorten_lines: s.shorten_lines,
          readings: s.readings});
        app.linesRequest = data.request;
        toast(await copyText(data.prompt) ? data.status : "Copy failed.");
      } catch (error) {
        toast(error.message, {error: true});
      }
    } else if (action === "apply-lines-reply") {
      if (!app.linesRequest) {
        toast("Copy the prompt for these lines first.", {error: true});
        return;
      }
      try {
        const data = await api.post("lines-reply", {session: app.session.id, request: app.linesRequest,
          reply: app.linesReply || ""});
        app.setSession(data.session);
        app.linesReply = "";
        toast(data.status.replace(/\*\*/g, ""));
        app.render();
      } catch (error) {
        toast(error.message, {error: true});
      }
    } else if (action === "merge-lines") {
      const picked = [...app.picks].sort((a, b) => a - b);
      if (await edit(app, "line/merge", {picked})) {
        app.picks.clear();
        app.mergeGroups = [];
        app.render();
      }
    } else if (action === "add-line") {
      edit(app, "line/add", {focus: app.focus});
    } else if (action === "delete-line") {
      if (confirm(`Delete line ${app.focus + 1}?`)) edit(app, "line/delete", {focus: app.focus});
    } else if (action === "table-open") {
      app.table = true;
      app.render();
    } else if (action === "table-close") {
      app.table = false;
      app.newRows = [];
      app.render();
    } else if (action === "table-add") {
      app.newRows = [...(app.newRows || []), [null, "", "", app.session.board.lines.at(-1)?.speakerId || "speaker1", "", "", true]];
      app.render();
    } else if (action === "to-dub") {
      app.go("dub");
    }
  });
  screen.addEventListener("change", (event) => {
    if (app.screen !== "review" || !app.session) return;
    const field = event.target.dataset.lineField;
    if (field) saveLine(app, {[field]: event.target.value});
    else if (event.target.matches(".lines-table [data-cell]")) saveTable(app);
  });
  screen.addEventListener("keydown", (event) => {
    if (app.screen !== "review") return;
    const field = event.target.dataset?.lineField;
    if (event.key === "Enter" && (field === "start" || field === "end")) {
      event.preventDefault();
      saveLine(app, {[field]: event.target.value});
    }
  });
  screen.addEventListener("input", (event) => {
    if (event.target.matches("[data-line-field], [data-cell]")) editVersion++;
    if (app.screen === "review" && event.target.dataset.action === "lines-reply") app.linesReply = event.target.value;
  });
}

export const reviewRanges = ranges;
