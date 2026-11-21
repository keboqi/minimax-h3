// Step 4, Result: the latest dub's player with its lines, downloads, subtitles export, word timing and share.
// The player is the same one the share page plays; the page draws its own controls over it.

import {api} from "./api.js";
import {copyText, esc, icon, plural, size, toast} from "./ui.js";

let mounted = {key: null, node: null};
let bindPlayers = null;
const playerRequests = new WeakMap();
let lipSyncOverrides = {video: null, picks: {}};

function overridesFor(video) {
  if (lipSyncOverrides.video !== video) {
    lipSyncOverrides = {video, picks: {}};
  }
  return lipSyncOverrides.picks;
}

function shotKey(shot) {
  return `${shot.shot_index}:${shot.speaker}`;
}

function chips(result) {
  const s = result.settings || {};
  const list = [s.timing, s.preset, s.voice_mode, s.word_timing ? "Word timing" : null].filter(Boolean);
  return list.map((item) => `<span class="chip chip-quiet">${esc(item)}</span>`).join("");
}

function controls(app, result) {
  const same = app.session.same;
  const view = app.playerView || {voice: "dub", music: "on", subtitles: "translation"};
  const button = (group, value, label) => `<button type="button" data-player-control="${group}" data-value="${value}"
    aria-pressed="${view[group] === value}">${label}</button>`;
  return `
    <div class="player-controls">
      <div class="control"><span class="field-label">Voice</span><div class="seg" role="group" aria-label="Voice">
        ${button("voice", "dub", same ? "Edited" : "Dub")}${button("voice", "original", "Original")}</div></div>
      <div class="control"><span class="field-label">Music</span>${result.separated
        ? `<div class="seg" role="group" aria-label="Background music">${button("music", "on", "On")}${button("music", "off", "Off")}</div>`
        : '<span class="muted small">Part of the mix (needs vocal separation)</span>'}</div>
      <div class="control"><span class="field-label">Subtitles</span><div class="seg" role="group" aria-label="Subtitles">
        ${button("subtitles", "translation", same ? "New text" : esc(app.session.target_language || "Translation"))}
        ${button("subtitles", "source", same ? "What was said" : esc(app.session.spoken || "Original"))}
        ${button("subtitles", "both", "Both")}${button("subtitles", "off", "Off")}</div></div>
    </div>`;
}

function downloads(result) {
  const badge = {video: ["MP4", "badge-blue"], audio: ["MP3", "badge-green"], subtitles: ["SRT", "badge-grey"]};
  return `
    <section class="card card-pad" aria-labelledby="dl-h">
      <div class="row align-baseline"><h2 id="dl-h">Downloads</h2><span class="muted small">Also in Media</span></div>
      <div class="files">${result.files.map((file) => `
        <div class="file-row">
          <span class="file-badge ${badge[file.kind][1]}">${badge[file.kind][0]}</span>
          <span class="grow"><strong>${esc(file.label)}</strong><span class="muted small ellipsis">${esc(file.name)} · ${size(file.size)}</span></span>
          <a class="btn btn-icon" href="${esc(file.url)}" download aria-label="Download ${esc(file.label)}">${icon("download")}</a>
        </div>`).join("")}
      </div>
    </section>`;
}

// Which subtitles the export burns in: chosen in its card, at first what the player shows.
function exportChoice(app) {
  const shown = app.playerView?.subtitles;
  return app.exportSubtitles || (shown && shown !== "off" ? shown : "translation");
}

function showExportChoice(app) {
  const choice = exportChoice(app);
  for (const button of document.querySelectorAll("[data-export-subtitles]")) {
    button.setAttribute("aria-pressed", String(button.dataset.exportSubtitles === choice));
  }
}

function exportCard(app, result) {
  const run = app.runs.export;
  const same = app.session.same;
  const file = run && !run.active && run.file && !run.error
    ? `<a class="btn btn-small" href="${api.url("run-file", {id: run.id})}">${icon("download", 14)} ${esc(run.file)}</a>` : "";
  const choice = exportChoice(app);
  const option = (value, label) => `<button type="button" data-export-subtitles="${value}"
    aria-pressed="${choice === value}">${label}</button>`;
  return `
    <section class="card card-pad stack-10" aria-labelledby="bi-h">
      <h2 id="bi-h">Video with subtitles</h2>
      <p class="muted small">Re-encodes the picture with the subtitles burned in as the player shows them, as a new MP4
        in Media; aligned words light up as they are spoken. Audio tracks are copied; voices are not generated again.
        The dub itself stays uncaptioned.</p>
      <div class="control"><span class="field-label">Burn in</span><div class="seg" role="group" aria-label="Subtitles to burn in">
        ${option("translation", same ? "New text" : esc(app.session.target_language || "Translation"))}
        ${option("source", same ? "What was said" : esc(app.session.spoken || "Original"))}
        ${option("both", "Both")}</div></div>
      <div class="run-status" data-run="export" hidden><span class="spinner" aria-hidden="true"></span><span data-run-status role="status"></span></div>
      <div class="row row-wrap">
        <button type="button" class="btn btn-small" data-action="export" data-busy="export" ${result.export ? "" : 'data-locked="true" disabled'}>Export video</button>
        ${file}
      </div>
      ${result.export ? "" : '<span class="muted small">Needs a video dub with subtitles.</span>'}
    </section>`;
}

function lipSyncCard(app, result) {
  const ls = result.lip_sync;
  if (!ls) return "";
  const out = ls.output;
  const report = ls.report;
  const shots = ls.review_shots || [];
  const picks = overridesFor(ls.video);
  const canRun = !!(app.boot.lip_sync && ls.ready);
  const stats = report
    ? `<span class="chip chip-quiet">${plural(report.total_windows, "window")}${report.cached_windows ? ` · ${report.cached_windows} cached` : ""}</span>`
    : "";
  const reviewHtml = shots.length && canRun ? `
    <details class="sec lip-sync-review">
      <summary>Review ${plural(shots.length, "uncertain shot")} (optional)</summary>
      <div class="stack-10">
        <p class="muted small">Uncertain shots kept the original picture. Pick the speaking face or mark off-screen, then re-render only those shots.</p>
        ${shots.map((shot) => {
          const key = shotKey(shot);
          const picked = Object.prototype.hasOwnProperty.call(picks, key) ? picks[key] : undefined;
          return `
            <div class="lip-sync-shot stack-8">
              <div class="row align-baseline row-wrap">
                <strong>${esc(shot.speaker_label)}</strong>
                <span class="muted small">Shot ${shot.shot_index + 1} · ${shot.time.toFixed(1)}s</span>
              </div>
              ${shot.reason ? `<span class="muted small">${esc(shot.reason)}</span>` : ""}
              <div class="lip-sync-frame-wrap">
                <img src="${esc(shot.frame_url)}" alt="${esc(shot.speaker_label)} frame at ${shot.time.toFixed(1)}s" loading="lazy">
                ${shot.tracks.map((track, idx) => {
                  const [x, y, w, h] = track.box;
                  const active = picked === track.track_id;
                  return `<button type="button" class="lip-sync-box"
                    style="left:${(x * 100).toFixed(2)}%;top:${(y * 100).toFixed(2)}%;width:${(w * 100).toFixed(2)}%;height:${(h * 100).toFixed(2)}%"
                    data-lip-sync-pick="${esc(key)}" data-track-id="${track.track_id}"
                    aria-pressed="${active}">Face ${idx + 1}</button>`;
                }).join("")}
              </div>
              <div class="row row-wrap">
                <button type="button" class="btn btn-small" data-lip-sync-pick="${esc(key)}" data-track-id="off"
                  aria-pressed="${picked === null}">Off-screen / skip</button>
              </div>
            </div>`;
        }).join("")}
        <div class="row">
          <button type="button" class="btn btn-small btn-primary" data-action="lip-sync-apply" data-busy="lip-sync">Apply &amp; re-render affected shots</button>
        </div>
      </div>
    </details>` : "";
  return `
    <section class="card card-pad stack-10" aria-labelledby="ls-h">
      <div class="row align-center row-wrap"><h2 id="ls-h">Lip sync</h2>${stats}</div>
      <p class="muted small">Automatically tracks faces, repaints the lower face to match the fitted dub with H3, and keeps the approved audio and subtitles.</p>
      ${app.boot.lip_sync ? `
        <div class="run-status" data-run="lip-sync" data-dub-video="${esc(ls.video)}" hidden>
          <span class="spinner" aria-hidden="true"></span><span data-run-status role="status"></span></div>
        <div class="row row-wrap">
          <button type="button" class="btn btn-small btn-primary" data-action="lip-sync" data-busy="lip-sync"
            ${ls.ready ? "" : 'data-locked="true" disabled'}>${out ? "Re-run lip sync" : "Lip sync video"}</button>
          <button type="button" class="btn btn-small" data-stop="lip-sync" data-action="interrupt" data-kind="lip-sync" disabled>Interrupt</button>
          ${out ? `<a class="btn btn-small" href="${esc(out.url)}" download>${icon("download", 14)} Download (${size(out.size)})</a>` : ""}
        </div>
        ${ls.ready ? "" : '<span class="muted small">Generate this dub again to save fitted dialogue for lip sync.</span>'}`
        : `<span class="small info-text">Lip sync runs in Studio.</span>${out ? `<div class="row"><a class="btn btn-small" href="${esc(out.url)}" download>${icon("download", 14)} Download (${size(out.size)})</a></div>` : ""}`}
      ${out ? `<video controls playsinline preload="metadata" class="lip-sync-preview" aria-label="Lip-synced video" src="${esc(out.inline_url)}"></video>` : ""}
      ${reviewHtml}
    </section>`;
}

function wordsCard(app, result) {
  const missing = result.cues - result.timed;
  return `
    <section class="card card-pad stack-10" aria-labelledby="wt-h">
      <h2 id="wt-h">Word timing</h2>
      ${result.word_timing ? `
        <div class="row align-baseline"><b class="big">${result.timed}</b><span class="muted small">of ${plural(result.cues, "line")} light word by word</span></div>
        <div class="word-bar" aria-hidden="true"><span style="width:${result.cues ? result.timed / result.cues * 100 : 0}%"></span></div>
        <p class="muted small">${missing ? `${plural(missing, "line")} ${missing === 1 ? "shows" : "show"} plain subtitles. ` : ""}Repair aligns the latest dub
          again without new voices: timed lines are reused, missing ones retried.</p>`
        : '<p class="muted small">This dub has plain subtitles. Generate or repair word alignment without voicing again.</p>'}
      ${app.boot.gpu ? `
        <div class="run-status" data-run="words" hidden><span class="spinner" aria-hidden="true"></span><span data-run-status role="status"></span></div>
        <div class="row"><button type="button" class="btn btn-small" data-action="align" data-busy="words">${result.word_timing ? "Repair word timing" : "Generate word timing"}</button>
          <button type="button" class="btn btn-small" data-stop="words" data-action="interrupt" data-kind="words" disabled>Interrupt</button></div>`
        : '<span class="small info-text">Word alignment runs in Studio.</span>'}
    </section>`;
}

function shareCard(app) {
  if (app.boot.gpu) {
    return `
      <section class="card card-pad stack-8"><h2>Share</h2>
        <p class="muted small">Share links are made and served by Workshop. Open this project there to create a view-only link.</p>
      </section>`;
  }
  const url = app.session.share;
  return `
    <section class="card card-pad stack-10" aria-labelledby="sh-h">
      <div class="row align-center"><h2 id="sh-h">Share</h2>${url ? '<span class="chip">Live</span>' : ""}</div>
      <p class="muted small">A view-only page with the player and synced subtitles, for any screen. Anyone with the link
        can watch it, without signing in.</p>
      ${url ? `
        <div class="row"><input class="mono grow" readonly value="${esc(url)}" aria-label="Share link">
          <button type="button" class="btn btn-small btn-primary" data-action="copy-share">Copy</button></div>
        <div class="row"><a class="btn btn-small" href="${esc(url)}" target="_blank" rel="noopener">Open page</a>
          <button type="button" class="btn btn-small danger" data-action="stop-share">Stop sharing</button></div>
        <span class="muted small">Follows this dub: a later word alignment shows up. Deleting the dub from Media ends the link.</span>`
        : '<button type="button" class="btn btn-small btn-primary" data-action="create-share">Create share link</button>'}
    </section>`;
}

export function renderResult(app) {
  const session = app.session;
  const result = session?.result;
  if (!result) {
    return `
      <div class="card card-pad empty-state">
        <h2>No dub yet</h2>
        <p class="muted">${app.boot.gpu ? "Generate the dub on the Dub step; it opens here when it is done."
          : "Generate the dub in Studio; it opens here once this project is loaded again."}</p>
        <button type="button" class="btn btn-primary" data-action="to-dub">Go to Dub</button>
      </div>`;
  }
  const title = `${session.source?.name?.replace(/\.[^.]+$/, "") || session.name || "Dub"} · ${result.label}`;
  return `
    <div class="stack-16 wider">
      <div class="result-head">
        <h1>${esc(title)}</h1>
        <span class="muted small">${esc(result.when)}</span>
        <div class="chips">${chips(result)}</div>
        <span class="spacer"></span>
        <button type="button" class="btn btn-small" data-action="to-dub">Change settings</button>
        <button type="button" class="btn btn-small" data-action="to-review">Edit lines</button>
      </div>
      ${result.failed.length ? `
        <div class="alert alert-fail row row-wrap align-center" role="alert">
          <span class="grow"><strong>Incomplete dub:</strong> ${result.failed.length === 1 ? "line" : "lines"}
            ${result.failed.join(", ")} failed and kept the original voice.</span>
          <button type="button" class="btn btn-small" data-action="jump-failed">Play line ${result.failed[0]}</button>
          <button type="button" class="btn btn-small btn-fail" data-action="retry-failed">Retry in Review lines</button>
        </div>` : ""}
      <section class="card result-player" aria-label="Player">
        ${controls(app, result)}
        <div class="player-slot" id="player-slot"></div>
      </section>
      <div class="output-grid">
        ${downloads(result)}${lipSyncCard(app, result)}${exportCard(app, result)}${wordsCard(app, result)}${shareCard(app)}
      </div>
    </div>`;
}

function playerRoot() {
  return mounted.node?.querySelector(".h3-dub-player") || null;
}

function applyView(app) {
  const root = playerRoot();
  if (!root) return;
  const view = app.playerView || {voice: "dub", music: "on", subtitles: "translation"};
  const voices = root.querySelectorAll("[data-voice-choice]");
  const voice = voices[view.voice === "original" ? 1 : 0];
  if (voice && !voice.checked) {
    voice.checked = true;
    voice.dispatchEvent(new Event("change", {bubbles: true}));
  }
  const music = root.querySelector("[data-music-toggle]");
  if (music && music.checked !== (view.music === "on")) {
    music.checked = view.music === "on";
    music.dispatchEvent(new Event("change", {bubbles: true}));
  }
  const subtitles = root.querySelector("select[data-subtitles]");
  if (subtitles && subtitles.value !== view.subtitles) {
    subtitles.value = view.subtitles;
    subtitles.dispatchEvent(new Event("change", {bubbles: true}));
  }
  root.dataset.follow = app.follow ? "on" : "off";
}

function filterCues(app) {
  const root = playerRoot();
  if (!root) return;
  const query = (app.search || "").trim().toLowerCase();
  for (const cue of root.querySelectorAll("[data-cues] li")) {
    cue.hidden = !!query && !cue.textContent.toLowerCase().includes(query);
  }
}

// The player keeps playing across redraws of the rest of the screen: its node moves into each new slot.
export async function afterResult(app) {
  const slot = document.getElementById("player-slot");
  const result = app.session?.result;
  if (!slot || !result) return;
  if (result.html === null) {
    slot.textContent = "Loading player…";
    if (!playerRequests.has(result)) playerRequests.set(result, api.get("result-player", {session: app.session.id}));
    try {
      result.html = (await playerRequests.get(result)).html;
    } catch (error) {
      playerRequests.delete(result);
      if (slot.isConnected) slot.textContent = error.message;
      return;
    }
    if (!slot.isConnected) return;
  }
  const key = result.html;
  if (mounted.key !== key) {
    if (!bindPlayers) bindPlayers = (await import("./player.js")).bindPlayers;
    if (!slot.isConnected) return;
    for (const media of mounted.node?.querySelectorAll("audio, video") || []) media.pause();
    const node = document.createElement("div");
    node.className = "player-host";
    node.innerHTML = result.html;
    const cues = node.querySelector("[data-cues]");
    if (cues) {
      const tools = document.createElement("div");
      tools.className = "cue-tools";
      tools.innerHTML = `
        <label class="search">${icon("search")}<input type="search" placeholder="Search lines" aria-label="Search lines"
          data-cue-search value="${esc(app.search || "")}"></label>
        <label class="check small"><input type="checkbox" data-cue-follow ${app.follow ? "checked" : ""}> Follow playback</label>`;
      cues.before(tools);
    }
    mounted = {key, node};
    slot.replaceChildren(node);
    bindPlayers(node);
  } else {
    slot.append(mounted.node);
  }
  applyView(app);
  filterCues(app);
}

function jumpTo(lineNumber) {
  const root = playerRoot();
  const cue = root && [...root.querySelectorAll("[data-cues] li")][lineNumber - 1];
  if (cue) cue.click();
  if (root?.querySelector("[data-voice]")?.paused) root.querySelector("[data-play]")?.click();
}

export function bindResult(app) {
  const screen = document.getElementById("screen");
  screen.addEventListener("click", async (event) => {
    if (app.screen !== "result" || !app.session) return;
    const control = event.target.closest("[data-player-control]");
    if (control) {
      app.playerView = {...(app.playerView || {voice: "dub", music: "on", subtitles: "translation"}),
        [control.dataset.playerControl]: control.dataset.value};
      for (const button of document.querySelectorAll(`[data-player-control="${control.dataset.playerControl}"]`)) {
        button.setAttribute("aria-pressed", String(button === control));
      }
      applyView(app);
      showExportChoice(app);  // until one is picked there, the export follows the player
      return;
    }
    const burn = event.target.closest("[data-export-subtitles]");
    if (burn) {
      app.exportSubtitles = burn.dataset.exportSubtitles;
      showExportChoice(app);
      return;
    }
    const pick = event.target.closest("[data-lip-sync-pick]");
    if (pick && app.session.result?.lip_sync) {
      const picks = overridesFor(app.session.result.lip_sync.video);
      const key = pick.dataset.lipSyncPick;
      const raw = pick.dataset.trackId;
      picks[key] = raw === "off" ? null : Number(raw);
      for (const btn of document.querySelectorAll(`[data-lip-sync-pick="${key}"]`)) {
        const val = btn.dataset.trackId === "off" ? null : Number(btn.dataset.trackId);
        btn.setAttribute("aria-pressed", String(picks[key] === val));
      }
      return;
    }
    const target = event.target.closest("[data-action]");
    if (!target) return;
    const action = target.dataset.action;
    const session = app.session;
    if (action === "to-dub") app.go("dub");
    else if (action === "to-review") app.go("review");
    else if (action === "retry-failed") {
      app.filter = "failed";
      app.focus = Math.max(0, (session.result.failed[0] || 1) - 1);
      app.go("review");
    } else if (action === "jump-failed") jumpTo(session.result.failed[0]);
    else if (action === "export") app.startRun("export", {session: session.id, subtitles: exportChoice(app)});
    else if (action === "align") app.startRun("align", {session: session.id});
    else if (action === "lip-sync") app.startRun("lip-sync", {session: session.id});
    else if (action === "lip-sync-apply") {
      const ls = session.result?.lip_sync;
      const picks = ls ? overridesFor(ls.video) : {};
      const overrides = Object.entries(picks).map(([key, trackId]) => {
        const sep = key.indexOf(":");
        return {shot_index: Number(key.slice(0, sep)), speaker: key.slice(sep + 1), track_id: trackId};
      });
      if (!overrides.length) {
        toast("Pick a face or Off-screen for at least one shot first.", {error: true});
        return;
      }
      app.startRun("lip-sync", {session: session.id, overrides});
    }
    else if (action === "interrupt") {
      try {
        toast((await api.post("interrupt", {kind: target.dataset.kind, session: session.id})).message);
      } catch (error) {
        toast(error.message, {error: true});
      }
    } else if (action === "create-share") {
      try {
        app.setSession((await api.post("share", {session: session.id})).session);
        toast("Link ready. It lasts until you stop sharing or delete this dub.");
        app.render();
      } catch (error) {
        toast(error.message, {error: true});
      }
    } else if (action === "stop-share") {
      try {
        app.setSession((await api.del("share", {session: session.id})).session);
        toast("Sharing stopped; the link no longer works.");
        app.render();
      } catch (error) {
        toast(error.message, {error: true});
      }
    } else if (action === "copy-share") {
      toast(await copyText(session.share) ? "Link copied." : "Copy failed.");
    }
  });
  screen.addEventListener("input", (event) => {
    if (app.screen !== "result") return;
    if (event.target.matches("[data-cue-search]")) {
      app.search = event.target.value;
      filterCues(app);
    }
  });
  screen.addEventListener("change", (event) => {
    if (app.screen !== "result") return;
    if (event.target.matches("[data-cue-follow]")) {
      app.follow = event.target.checked;
      applyView(app);
    }
  });
}
