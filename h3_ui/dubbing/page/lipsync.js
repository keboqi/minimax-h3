// Step 4, Lip sync (optional): 1. Auto detect -> 2. Play & confirm face alignment -> 3. Confirm & render.

import {api} from "./api.js";
import {esc, icon, plural, toast} from "./ui.js";

const lsState = {
  targetKey: null,
  data: null,
  loading: false,
  request: null,
  saving: false,
  error: null,
  filter: "all",
  selectedShot: 0,
  selectedSpeaker: null,
  strength: 0.55,
  feather: 6,
  rafId: null,
  observer: null,
};

function stopOverlayLoop() {
  if (lsState.rafId !== null) {
    cancelAnimationFrame(lsState.rafId);
    lsState.rafId = null;
  }
  lsState.observer?.disconnect();
  lsState.observer = null;
}

function reviewTarget(app) {
  const ls = app.session?.result?.lip_sync;
  return ls ? {session: app.session.id, video: ls.video, video_version: ls.video_version} : null;
}

export async function loadLipSyncPlan(app, {silent = false} = {}) {
  const target = reviewTarget(app);
  if (!target) return;
  const key = JSON.stringify(target);
  if (lsState.targetKey !== key) {
    Object.assign(lsState, {targetKey: key, data: null, saving: false, selectedShot: 0, selectedSpeaker: null});
  }
  const request = Symbol();
  Object.assign(lsState, {request, loading: true, error: null});
  const isCurrent = () => lsState.request === request && JSON.stringify(reviewTarget(app)) === key;
  if (!silent) app.render();
  try {
    const data = await api.get("lip-sync-plan", target);
    if (isCurrent()) lsState.data = data;
  } catch (error) {
    if (isCurrent()) lsState.error = error.message;
  } finally {
    if (lsState.request === request) lsState.loading = false;
    if (isCurrent() && app.screen === "lipsync") app.render();
  }
}

function speakerLabel(speaker) {
  return (lsState.data?.speakers && lsState.data.speakers[speaker]) || speaker || "Speaker";
}

function formatClock(sec) {
  const s = Math.max(0, Number(sec) || 0);
  const m = Math.floor(s / 60);
  const r = (s - m * 60).toFixed(1).padStart(4, "0");
  return `${m}:${r}`;
}

function assignmentMap(plan) {
  const map = new Map();
  for (const a of plan?.assignments || []) {
    map.set(`${a.shot_index}:${a.speaker}`, a);
  }
  return map;
}

function shotItems(plan) {
  if (!plan?.shots) return [];
  const assignments = plan.assignments || [];
  const turns = plan.turns || [];
  const items = [];
  for (const shot of plan.shots) {
    const shotAssignments = assignments.filter((a) => a.shot_index === shot.index);
    const shotTurns = turns.filter((t) => Math.min(t.end, shot.end) - Math.max(t.start, shot.start) >= 0.04);
    if (!shotAssignments.length) continue;
    for (const a of shotAssignments) {
      const firstTurn = shotTurns.find((t) => t.speaker === a.speaker);
      items.push({
        shot,
        assignment: a,
        speaker: a.speaker,
        seekTime: firstTurn
          ? (Math.max(shot.start, firstTurn.start) + Math.min(shot.end, firstTurn.end)) / 2
          : Math.min(shot.end, Math.max(shot.start, a.reference_time ?? shot.start)),
        line: firstTurn?.text || "",
        key: `${shot.index}:${a.speaker}`,
      });
    }
  }
  return items;
}

function assignmentLabel(shot, speaker) {
  return shot && speaker
    ? `Shot ${shot.index + 1} (${formatClock(shot.start)}–${formatClock(shot.end)}) · <strong>${esc(speakerLabel(speaker))}</strong>`
    : "Choose a shot with a speaker to review.";
}

function assignmentButtons(shot, assignment, overrides) {
  if (!shot || !assignment) return "";
  const key = `${shot.index}:${assignment.speaker}`;
  const button = (id, label, selected) => `<button type="button" class="btn btn-small ${selected ? "btn-primary" : ""}"
    data-ls-assign="${esc(key)}" data-track-id="${id}">${label}</button>`;
  const buttons = (shot.tracks || []).map((track, idx) =>
    button(track.track_id, `Face #${idx + 1}`, track.track_id === assignment.track_id));
  buttons.push(button("off", "Off-screen / Skip", assignment.status === "off_screen"));
  if (Object.hasOwn(overrides || {}, key)) buttons.push(button("auto", "Restore automatic match", false));
  return buttons.join("");
}

function boxAtTime(track, time) {
  const obs = track.obs || [];
  if (!obs.length) return track.box || null;
  let best = obs[0];
  let bestDt = Math.abs(obs[0][0] - time);
  for (let i = 1; i < obs.length; i++) {
    const dt = Math.abs(obs[i][0] - time);
    if (dt < bestDt) {
      bestDt = dt;
      best = obs[i];
    }
  }
  if (bestDt > 1.2) return null;
  return [best[1], best[2], best[3], best[4], best[5]];
}

function statusBadge(a) {
  if (a.status === "manual") return {label: "Confirmed", cls: "chip"};
  if (a.status === "matched") return {label: "Auto matched", cls: "chip"};
  if (a.status === "off_screen") return {label: "Off-screen", cls: "chip chip-quiet"};
  return {label: "Needs check", cls: "chip chip-warn"};
}

export function renderLipSync(app) {
  const result = app.session?.result;
  const ls = result?.lip_sync;
  if (!result || !ls) {
    return `
      <div class="card card-pad stack-12">
        <h1>Lip sync</h1>
        <p class="muted">Generate a video dub first, then return here to detect speakers, confirm face alignment on the video, and render lip sync.</p>
        <div class="row"><button type="button" class="btn btn-primary" data-ls-action="to-dub">Back to Dub</button></div>
      </div>`;
  }

  const data = lsState.targetKey === JSON.stringify(reviewTarget(app)) ? lsState.data : null;
  const plan = data?.plan || null;
  const items = shotItems(plan);
  const visible = items.filter((it) => {
    if (lsState.filter === "review") return it.assignment.status === "uncertain";
    if (lsState.filter === "confirmed") return it.assignment.status === "matched" || it.assignment.status === "manual";
    return true;
  });
  const activeItem = items.find((it) => it.shot.index === lsState.selectedShot && (!lsState.selectedSpeaker || it.speaker === lsState.selectedSpeaker))
    || items[0] || null;
  if (activeItem) {
    lsState.selectedShot = activeItem.shot.index;
    lsState.selectedSpeaker = activeItem.speaker;
  }

  const uncertainCount = items.filter((it) => it.assignment.status === "uncertain").length;
  const confirmedCount = items.filter((it) => it.assignment.status === "matched" || it.assignment.status === "manual").length;
  const offCount = items.filter((it) => it.assignment.status === "off_screen").length;
  const unavailable = !ls.ready ? "Generate this dub again to save its fitted dialogue." : "";
  const renderReason = unavailable || (!app.boot.lip_sync ? "Open this dub in Studio to render lip sync."
    : plan && !plan.windows_count ? "No render windows. Review speaker assignments and choose a visible face." : "");
  const canRender = !!(plan?.revision && !renderReason && !lsState.loading);

  return `
    <div class="ls-workspace stack-16">
      <section class="card card-pad ls-stage-bar">
        <div class="row align-center row-wrap gap-28">
          <div class="ls-phases row align-center row-wrap">
            <span class="chip ${plan ? "" : "chip-warn"}">1. Auto detect${plan ? ` (${esc(plan.backend)})` : ""}</span>
            <span class="muted">→</span>
            <span class="chip ${plan ? "" : "chip-quiet"}">2. Play &amp; confirm faces</span>
            <span class="muted">→</span>
            <span class="chip ${data?.output ? "" : "chip-quiet"}">3. Confirm &amp; render</span>
          </div>
          <span class="spacer"></span>
          <div class="row row-wrap">
            <button type="button" class="btn btn-small" data-ls-action="detect" data-busy="lip-sync-detect" data-locked="${!!unavailable}">
              ${plan ? "Re-run auto detect" : "Auto detect faces & speakers"}
            </button>
            <button type="button" class="btn btn-small" data-ls-action="to-result">Skip to Result →</button>
          </div>
        </div>
        <div class="run-status top-gap" data-run="lip-sync-detect" data-dub-video="${esc(ls.video)}" hidden>
          <span class="spinner" aria-hidden="true"></span>
          <span data-run-status role="status"></span>
          <button type="button" class="btn btn-small" data-ls-action="interrupt" data-kind="lip-sync-detect">Interrupt</button>
        </div>
        <div class="run-status top-gap" data-run="lip-sync" data-dub-video="${esc(ls.video)}" hidden>
          <span class="spinner" aria-hidden="true"></span>
          <span data-run-status role="status"></span>
          <button type="button" class="btn btn-small" data-ls-action="interrupt" data-kind="lip-sync">Interrupt</button>
        </div>
      </section>

      ${renderReason ? `<p class="muted small">${esc(renderReason)}</p>` : ""}
      ${lsState.loading ? '<div class="card card-pad muted">Loading lip-sync plan…</div>' : ""}
      ${lsState.error ? `<div class="card card-pad fail-text">${esc(lsState.error)}</div>` : ""}

      ${!plan && !lsState.loading ? `
        <section class="card card-pad stack-14">
          <h2>Step 1 · Auto detect faces &amp; speaking speakers</h2>
          <p class="muted">Runs <code>face_yolov8m.pt</code> face tracking, aligns speakers using your Gemini <code>visual_speakers</code> hints, and uses MediaPipe mouth-motion (MAR) as backup. No H3 video rendering starts until you review and confirm.</p>
          <div class="row row-wrap">
            <button type="button" class="btn btn-primary" data-ls-action="detect" data-busy="lip-sync-detect" data-locked="${!!unavailable}">
              Auto detect faces &amp; speakers
            </button>
            <button type="button" class="btn" data-ls-action="to-result">Skip to Result</button>
          </div>
        </section>
      ` : ""}

      ${plan ? `
        <div class="ls-grid">
          <section class="ls-main stack-14">
            <div class="card ls-player-card">
              <div class="ls-video-stage" id="ls-video-stage">
                <video id="ls-video" src="${esc(data.video_url)}" controls playsinline preload="metadata"></video>
                <div class="ls-overlay" id="ls-overlay" aria-label="Tracked face boxes"></div>
                <div class="ls-subtitle-bar" id="ls-subtitle" hidden></div>
              </div>
              <div class="ls-assign-bar">
                <div class="row align-center row-wrap">
                  <button type="button" class="btn btn-small" data-ls-action="prev-shot">← Prev shot</button>
                  <button type="button" class="btn btn-small" data-ls-action="next-shot">Next shot →</button>
                  <span class="muted small" id="ls-active-info">
                    ${assignmentLabel(activeItem?.shot, activeItem?.speaker)}
                  </span>
                </div>
                <div class="row row-wrap" id="ls-quick-buttons" data-key="${esc(activeItem?.key || "")}">
                  ${assignmentButtons(activeItem?.shot, activeItem?.assignment, data.overrides)}
                </div>
              </div>
            </div>

            <section class="card card-pad stack-12">
              <div class="row align-center row-wrap">
                <h3>Render settings</h3>
                <span class="spacer"></span>
                <span class="muted small">Output keeps <strong>Dubbed audio only</strong> · MediaPipe landmark mouth mask</span>
              </div>
              <div class="ls-settings-row">
                <label class="field">
                  <span class="field-label">Denoise strength <span class="mono" data-ls-out="strength">${lsState.strength.toFixed(2)}</span></span>
                  <input type="range" min="0.25" max="0.85" step="0.05" value="${lsState.strength}" data-ls-setting="strength">
                </label>
                <label class="field">
                  <span class="field-label">Mask feather <span class="mono" data-ls-out="feather">${lsState.feather} px</span></span>
                  <input type="range" min="2" max="16" step="1" value="${lsState.feather}" data-ls-setting="feather">
                </label>
              </div>
              <div class="row align-center row-wrap top-line">
                <span class="small">
                  <strong>${confirmedCount}</strong> matched · <strong>${uncertainCount}</strong> needs check · <strong>${offCount}</strong> off-screen
                  (${plural(plan.windows_count || 0, "render window")})
                </span>
                <span class="spacer"></span>
                <button type="button" class="btn btn-primary" data-ls-action="render" data-busy="lip-sync" data-locked="${!canRender}">
                  ${data?.output ? "Confirm & re-render lip sync" : "Confirm & render lip sync"}
                </button>
              </div>
            </section>
          </section>

          <aside class="card ls-queue">
            <div class="ls-queue-head stack-10">
              <div class="row align-center">
                <h2>Shots (${items.length})</h2>
                <span class="spacer"></span>
                <span class="muted small">Click to jump video</span>
              </div>
              <div class="seg" role="group" aria-label="Filter shots">
                <button type="button" data-ls-filter="all" aria-pressed="${lsState.filter === "all"}">All (${items.length})</button>
                <button type="button" data-ls-filter="review" aria-pressed="${lsState.filter === "review"}">Needs check (${uncertainCount})</button>
                <button type="button" data-ls-filter="confirmed" aria-pressed="${lsState.filter === "confirmed"}">Matched (${confirmedCount})</button>
              </div>
            </div>
            <div class="ls-queue-list">
              ${visible.map((it) => {
                const badge = statusBadge(it.assignment);
                const selected = activeItem && it.key === activeItem.key;
                const trackIdx = (it.shot.tracks || []).findIndex((t) => t.track_id === it.assignment.track_id);
                const targetDesc = it.assignment.status === "off_screen"
                  ? "Skipped (off-screen)"
                  : trackIdx >= 0 ? `Face #${trackIdx + 1} · ${it.assignment.source}` : it.assignment.reason || it.assignment.source;
                return `
                  <button type="button" class="ls-shot-row ${selected ? "selected" : ""}" data-ls-shot="${it.shot.index}" data-ls-speaker="${esc(it.speaker)}" data-ls-time="${it.seekTime}">
                    <div class="row align-center">
                      <strong>Shot ${it.shot.index + 1}</strong>
                      <span class="mono muted">${formatClock(it.shot.start)}–${formatClock(it.shot.end)}</span>
                      <span class="spacer"></span>
                      <span class="${badge.cls}">${badge.label}</span>
                    </div>
                    <div class="small ellipsis"><strong>${esc(speakerLabel(it.speaker))}</strong>${it.line ? ` · ${esc(it.line)}` : ""}</div>
                    <div class="small muted ellipsis">${esc(targetDesc)}</div>
                  </button>`;
              }).join("") || '<p class="muted small card-pad">No shots in this filter.</p>'}
            </div>
          </aside>
        </div>
      ` : ""}
    </div>`;
}

function drawOverlay(app) {
  const video = document.getElementById("ls-video");
  const overlay = document.getElementById("ls-overlay");
  const subtitle = document.getElementById("ls-subtitle");
  const plan = lsState.data?.plan;
  if (!video || !overlay || !plan) return;
  if (!video.videoWidth || !video.videoHeight) return;
  const scale = Math.min(video.clientWidth / video.videoWidth, video.clientHeight / video.videoHeight);
  const width = video.videoWidth * scale;
  const height = video.videoHeight * scale;
  Object.assign(overlay.style, {
    left: `${(video.clientWidth - width) / 2}px`, top: `${(video.clientHeight - height) / 2}px`,
    width: `${width}px`, height: `${height}px`,
  });

  const t = Number(video.currentTime) || 0;
  const shot = (plan.shots || []).find((s) => t >= s.start && t < s.end);
  const turn = (plan.turns || []).find((ln) => t >= ln.start && t < ln.end) || null;
  if (subtitle) {
    if (turn) {
      subtitle.hidden = false;
      subtitle.innerHTML = `<span class="ls-sub-spk">${esc(speakerLabel(turn.speaker))}</span> <span>${esc(turn.text)}</span>`;
    } else {
      subtitle.hidden = true;
    }
  }
  const aMap = assignmentMap(plan);
  const selected = shot?.index === lsState.selectedShot ? lsState.selectedSpeaker : null;
  const activeAssign = shot && (aMap.get(`${shot.index}:${selected}`) || aMap.get(`${shot.index}:${turn?.speaker}`)
    || (plan.assignments || []).find((a) => a.shot_index === shot.index));
  const activeSpeaker = activeAssign?.speaker;
  const key = activeSpeaker ? `${shot.index}:${activeSpeaker}` : "";
  const buttons = document.getElementById("ls-quick-buttons");
  if (buttons && buttons.dataset.key !== key) {
    buttons.dataset.key = key;
    buttons.innerHTML = assignmentButtons(shot, activeAssign, lsState.data.overrides);
    document.getElementById("ls-active-info").innerHTML = assignmentLabel(shot, activeSpeaker);
  }
  if (!shot) {
    if (overlay.childElementCount) overlay.textContent = "";
    return;
  }
  const activeTrackId = activeAssign && activeAssign.status !== "off_screen" ? activeAssign.track_id : null;

  // Reuse existing box buttons keyed by shot/track so clicks complete across animation frames.
  const existing = new Map();
  for (const child of overlay.children) {
    if (child.dataset?.boxId) existing.set(child.dataset.boxId, child);
  }
  const seen = new Set();
  (shot.tracks || []).forEach((track, idx) => {
    const b = boxAtTime(track, t);
    if (!b) return;
    const [x, y, w, h] = b;
    const isSpeaking = activeTrackId !== null && track.track_id === activeTrackId;
    const label = isSpeaking
      ? `${speakerLabel(activeSpeaker)} · Assigned`
      : `Face #${idx + 1}`;
    const boxId = `${shot.index}:${track.track_id}`;
    seen.add(boxId);
    let btn = existing.get(boxId);
    if (!btn) {
      btn = document.createElement("button");
      btn.type = "button";
      btn.className = "ls-face-box";
      btn.dataset.boxId = boxId;
      const tag = document.createElement("span");
      tag.className = "ls-face-tag";
      btn.appendChild(tag);
      overlay.appendChild(btn);
    }
    btn.classList.toggle("speaking", isSpeaking);
    btn.style.left = `${(x * 100).toFixed(2)}%`;
    btn.style.top = `${(y * 100).toFixed(2)}%`;
    btn.style.width = `${(w * 100).toFixed(2)}%`;
    btn.style.height = `${(h * 100).toFixed(2)}%`;
    if (key) {
      btn.dataset.lsAssign = key;
      btn.dataset.trackId = String(track.track_id);
    } else {
      delete btn.dataset.lsAssign;
      delete btn.dataset.trackId;
    }
    btn.disabled = !activeSpeaker;
    btn.title = `Click to assign ${speakerLabel(activeSpeaker)} to Face #${idx + 1}`;
    const tag = btn.firstElementChild;
    if (tag && tag.textContent !== label) tag.textContent = label;
  });
  for (const [boxId, node] of existing) {
    if (!seen.has(boxId)) node.remove();
  }
}

export function afterLipSync(app) {
  stopOverlayLoop();
  const target = reviewTarget(app);
  if (app.screen !== "lipsync" || !target) return;
  if (lsState.targetKey !== JSON.stringify(target) || (!lsState.data && !lsState.loading && !lsState.error)) {
    loadLipSyncPlan(app);
    return;
  }
  const video = document.getElementById("ls-video");
  if (!video || !lsState.data?.plan) return;
  lsState.observer = new ResizeObserver(() => drawOverlay(app));
  lsState.observer.observe(video);
  const tick = () => {
    if (!video.isConnected || app.screen !== "lipsync") {
      stopOverlayLoop();
      return;
    }
    drawOverlay(app);
    if (!video.paused && !video.ended) {
      lsState.rafId = requestAnimationFrame(tick);
    } else {
      lsState.rafId = null;
    }
  };
  video.addEventListener("play", () => { if (lsState.rafId === null) lsState.rafId = requestAnimationFrame(tick); });
  video.addEventListener("pause", () => drawOverlay(app));
  video.addEventListener("seeked", () => drawOverlay(app));
  video.addEventListener("timeupdate", () => { if (lsState.rafId === null) drawOverlay(app); });
  video.addEventListener("loadedmetadata", () => drawOverlay(app));
  video.addEventListener("resize", () => drawOverlay(app));
  drawOverlay(app);
}

async function saveOverride(app, key, trackId) {
  const target = reviewTarget(app);
  const targetKey = JSON.stringify(target);
  const revision = lsState.data?.plan?.revision;
  if (!target || lsState.targetKey !== targetKey || !revision || lsState.loading || lsState.saving) return;
  const sep = key.indexOf(":");
  const shotIndex = Number(key.slice(0, sep));
  const speaker = key.slice(sep + 1);
  lsState.saving = true;
  try {
    const resp = await api.post("lip-sync-overrides", {
      ...target,
      plan_revision: revision,
      overrides: [{shot_index: shotIndex, speaker, ...(trackId === "auto" ? {reset: true} : {track_id: trackId})}],
    });
    if (JSON.stringify(reviewTarget(app)) !== targetKey) return;
    lsState.data = resp;
    lsState.selectedShot = shotIndex;
    lsState.selectedSpeaker = speaker;
    const video = document.getElementById("ls-video");
    const savedTime = video ? video.currentTime : null;
    const wasPaused = video ? video.paused : true;
    app.render();
    const nextVideo = document.getElementById("ls-video");
    if (nextVideo && savedTime !== null) {
      const restore = () => {
        nextVideo.currentTime = savedTime;
        if (!wasPaused) nextVideo.play().catch(() => {});
        drawOverlay(app);
      };
      if (nextVideo.readyState >= 1) restore();
      else nextVideo.addEventListener("loadedmetadata", restore, {once: true});
    }
    toast(trackId === "auto" ? "Automatic assignment restored." : trackId === null ? `Shot ${shotIndex + 1} marked off-screen.` : `Shot ${shotIndex + 1} aligned to Face.`);
  } catch (error) {
    if (JSON.stringify(reviewTarget(app)) !== targetKey) return;
    toast(error.message, {error: true});
    await loadLipSyncPlan(app, {silent: true});
  } finally {
    if (lsState.targetKey === targetKey) lsState.saving = false;
  }
}

export function bindLipSync(app) {
  const screen = document.getElementById("screen");
  screen.addEventListener("click", async (event) => {
    if (app.screen !== "lipsync" || !app.session) return;
    const filter = event.target.closest("[data-ls-filter]");
    if (filter) {
      lsState.filter = filter.dataset.lsFilter;
      app.render();
      return;
    }
    const shotRow = event.target.closest("[data-ls-shot]");
    if (shotRow) {
      lsState.selectedShot = Number(shotRow.dataset.lsShot);
      lsState.selectedSpeaker = shotRow.dataset.lsSpeaker || null;
      const jumpTime = Number(shotRow.dataset.lsTime) || 0;
      app.render();
      const video = document.getElementById("ls-video");
      if (video) {
        const seek = () => { video.currentTime = jumpTime; drawOverlay(app); };
        if (video.readyState >= 1) seek();
        else video.addEventListener("loadedmetadata", seek, {once: true});
      }
      return;
    }
    const assign = event.target.closest("[data-ls-assign]");
    if (assign) {
      const raw = assign.dataset.trackId;
      await saveOverride(app, assign.dataset.lsAssign, raw === "auto" ? "auto" : raw === "off" ? null : Number(raw));
      return;
    }
    const btn = event.target.closest("[data-ls-action]");
    if (!btn) return;
    const action = btn.dataset.lsAction;
    if (action === "to-result") {
      stopOverlayLoop();
      app.go("result");
    } else if (action === "to-dub") {
      stopOverlayLoop();
      app.go("dub");
    } else if (action === "detect") {
      if (!lsState.saving) app.startRun("lip-sync-detect", {...reviewTarget(app), api_key: app.apiKey, force: !!lsState.data?.plan});
    } else if (action === "render") {
      const target = reviewTarget(app);
      const revision = lsState.data?.plan?.revision;
      if (lsState.targetKey !== JSON.stringify(target) || !revision || lsState.saving || lsState.loading) return;
      app.startRun("lip-sync", {
        ...target,
        plan_revision: revision,
        api_key: app.apiKey,
        strength: lsState.strength,
        feather: lsState.feather,
      });
    } else if (action === "interrupt") {
      try {
        toast((await api.post("interrupt", {kind: btn.dataset.kind, session: app.session.id})).message);
      } catch (error) {
        toast(error.message, {error: true});
      }
    } else if (action === "prev-shot" || action === "next-shot") {
      const items = shotItems(lsState.data?.plan);
      if (!items.length) return;
      const idx = Math.max(0, items.findIndex((it) => it.shot.index === lsState.selectedShot && it.speaker === lsState.selectedSpeaker));
      const next = items[(idx + (action === "next-shot" ? 1 : -1) + items.length) % items.length];
      lsState.selectedShot = next.shot.index;
      lsState.selectedSpeaker = next.speaker;
      app.render();
      const video = document.getElementById("ls-video");
      if (video) {
        const seek = () => { video.currentTime = next.seekTime; drawOverlay(app); };
        if (video.readyState >= 1) seek();
        else video.addEventListener("loadedmetadata", seek, {once: true});
      }
    }
  });
  screen.addEventListener("input", (event) => {
    if (app.screen !== "lipsync") return;
    const el = event.target.closest("[data-ls-setting]");
    if (!el) return;
    const key = el.dataset.lsSetting;
    lsState[key] = Number(el.value);
    const out = document.querySelector(`[data-ls-out="${key}"]`);
    if (out) out.textContent = key === "strength" ? lsState[key].toFixed(2) : `${lsState[key]} px`;
  });
}
