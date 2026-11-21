// Step 4, Lip sync (optional): 1. Auto detect -> 2. Play & confirm face alignment -> 3. Confirm & render.

import {api} from "./api.js";
import {esc, icon, plural, toast} from "./ui.js";

const lsState = {
  sessionId: null,
  data: null,
  loading: false,
  error: null,
  filter: "all",
  selectedShot: 0,
  selectedSpeaker: null,
  strength: 0.55,
  feather: 6,
  rafId: null,
};

function stopOverlayLoop() {
  if (lsState.rafId !== null) {
    cancelAnimationFrame(lsState.rafId);
    lsState.rafId = null;
  }
}

export function resetLipSyncState() {
  stopOverlayLoop();
  lsState.sessionId = null;
  lsState.data = null;
  lsState.loading = false;
  lsState.error = null;
}

export async function loadLipSyncPlan(app, {silent = false} = {}) {
  const sid = app.session?.id;
  if (!sid || !app.session?.result?.lip_sync) return;
  lsState.sessionId = sid;
  lsState.loading = !silent;
  lsState.error = null;
  if (!silent) app.render();
  try {
    lsState.data = await api.get("lip-sync-plan", {session: sid});
  } catch (error) {
    lsState.error = error.message;
  } finally {
    lsState.loading = false;
    if (app.screen === "lipsync") app.render();
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
      const firstTurn = shotTurns.find((t) => t.speaker === a.speaker) || shotTurns[0];
      items.push({
        shot,
        assignment: a,
        speaker: a.speaker,
        line: firstTurn?.text || "",
        key: `${shot.index}:${a.speaker}`,
      });
    }
  }
  return items;
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

  const data = lsState.sessionId === app.session.id ? lsState.data : null;
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
  const canRender = !!(app.boot.lip_sync && ls.ready && plan);

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
            <button type="button" class="btn btn-small" data-ls-action="detect" data-busy="lip-sync-detect">
              ${plan ? "Re-run auto detect" : "Auto detect faces & speakers"}
            </button>
            <button type="button" class="btn btn-small" data-ls-action="to-result">Skip to Result →</button>
          </div>
        </div>
        <div class="run-status top-gap" data-run="lip-sync-detect" hidden>
          <span class="spinner" aria-hidden="true"></span>
          <span data-run-status role="status"></span>
          <button type="button" class="btn btn-small" data-ls-action="interrupt" data-kind="lip-sync-detect">Interrupt</button>
        </div>
        <div class="run-status top-gap" data-run="lip-sync" hidden>
          <span class="spinner" aria-hidden="true"></span>
          <span data-run-status role="status"></span>
          <button type="button" class="btn btn-small" data-ls-action="interrupt" data-kind="lip-sync">Interrupt</button>
        </div>
      </section>

      ${lsState.loading ? '<div class="card card-pad muted">Loading lip-sync plan…</div>' : ""}
      ${lsState.error ? `<div class="card card-pad fail-text">${esc(lsState.error)}</div>` : ""}

      ${!plan && !lsState.loading ? `
        <section class="card card-pad stack-14">
          <h2>Step 1 · Auto detect faces &amp; speaking speakers</h2>
          <p class="muted">Runs <code>face_yolov8m.pt</code> face tracking, aligns speakers using your Gemini <code>visual_speakers</code> hints, and uses MediaPipe mouth-motion (MAR) as backup. No H3 video rendering starts until you review and confirm.</p>
          <div class="row row-wrap">
            <button type="button" class="btn btn-primary" data-ls-action="detect" data-busy="lip-sync-detect">
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
                    ${activeItem ? `Shot ${activeItem.shot.index + 1} (${formatClock(activeItem.shot.start)}–${formatClock(activeItem.shot.end)}) · <strong>${esc(speakerLabel(activeItem.speaker))}</strong>` : "Play the video and click any face rectangle to align."}
                  </span>
                </div>
                <div class="row row-wrap" id="ls-quick-buttons">
                  ${activeItem ? `
                    ${(activeItem.shot.tracks || []).map((t, idx) => `
                      <button type="button" class="btn btn-small ${activeItem.assignment.track_id === t.track_id && activeItem.assignment.status !== "off_screen" ? "btn-primary" : ""}"
                        data-ls-assign="${esc(activeItem.key)}" data-track-id="${t.track_id}">
                        Face #${idx + 1}
                      </button>`).join("")}
                    <button type="button" class="btn btn-small ${activeItem.assignment.status === "off_screen" ? "btn-primary" : ""}"
                      data-ls-assign="${esc(activeItem.key)}" data-track-id="off">
                      Off-screen / Skip
                    </button>
                  ` : ""}
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
                <button type="button" class="btn btn-primary" data-ls-action="render" data-busy="lip-sync" ${canRender ? "" : "disabled"}>
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
                  <button type="button" class="ls-shot-row ${selected ? "selected" : ""}" data-ls-shot="${it.shot.index}" data-ls-speaker="${esc(it.speaker)}" data-ls-time="${it.shot.start}">
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

  const t = Number(video.currentTime) || 0;
  const shot = (plan.shots || []).find((s) => t >= s.start - 0.05 && t < s.end + 0.05)
    || (plan.shots || []).find((s) => s.index === lsState.selectedShot);
  const turn = (plan.turns || []).find((ln) => t >= ln.start && t <= ln.end) || null;
  if (subtitle) {
    if (turn) {
      subtitle.hidden = false;
      subtitle.innerHTML = `<span class="ls-sub-spk">${esc(speakerLabel(turn.speaker))}</span> <span>${esc(turn.text)}</span>`;
    } else {
      subtitle.hidden = true;
    }
  }
  if (!shot) {
    overlay.innerHTML = "";
    return;
  }

  const aMap = assignmentMap(plan);
  const activeSpeaker = turn?.speaker || (shot.index === lsState.selectedShot ? lsState.selectedSpeaker : null)
    || (plan.assignments || []).find((a) => a.shot_index === shot.index)?.speaker || null;
  const activeAssign = activeSpeaker ? aMap.get(`${shot.index}:${activeSpeaker}`) : null;
  const activeTrackId = activeAssign && activeAssign.status !== "off_screen" ? activeAssign.track_id : null;

  const boxes = [];
  (shot.tracks || []).forEach((track, idx) => {
    const b = boxAtTime(track, t);
    if (!b) return;
    const [x, y, w, h] = b;
    const isSpeaking = activeTrackId !== null && track.track_id === activeTrackId;
    const label = isSpeaking
      ? `${speakerLabel(activeSpeaker)} · Speaking`
      : `Face #${idx + 1}`;
    const assignKey = activeSpeaker ? `${shot.index}:${activeSpeaker}` : "";
    boxes.push(`
      <button type="button" class="ls-face-box ${isSpeaking ? "speaking" : ""}"
        style="left:${(x * 100).toFixed(2)}%;top:${(y * 100).toFixed(2)}%;width:${(w * 100).toFixed(2)}%;height:${(h * 100).toFixed(2)}%"
        ${assignKey ? `data-ls-assign="${esc(assignKey)}" data-track-id="${track.track_id}"` : ""}
        title="Click to assign ${esc(speakerLabel(activeSpeaker))} to Face #${idx + 1}">
        <span class="ls-face-tag">${esc(label)}</span>
      </button>`);
  });
  overlay.innerHTML = boxes.join("");
}

export function afterLipSync(app) {
  stopOverlayLoop();
  if (!app.session?.result?.lip_sync) return;
  if (lsState.sessionId !== app.session.id || (!lsState.data && !lsState.loading && !lsState.error)) {
    loadLipSyncPlan(app);
    return;
  }
  const video = document.getElementById("ls-video");
  if (!video || !lsState.data?.plan) return;
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
  drawOverlay(app);
}

async function saveOverride(app, key, trackId) {
  const sep = key.indexOf(":");
  const shotIndex = Number(key.slice(0, sep));
  const speaker = key.slice(sep + 1);
  try {
    const resp = await api.post("lip-sync-overrides", {
      session: app.session.id,
      overrides: [{shot_index: shotIndex, speaker, track_id: trackId}],
    });
    lsState.data = resp;
    lsState.selectedShot = shotIndex;
    lsState.selectedSpeaker = speaker;
    const video = document.getElementById("ls-video");
    const savedTime = video ? video.currentTime : null;
    const wasPaused = video ? video.paused : true;
    app.render();
    const nextVideo = document.getElementById("ls-video");
    if (nextVideo && savedTime !== null) {
      nextVideo.addEventListener("loadedmetadata", () => {
        nextVideo.currentTime = savedTime;
        if (!wasPaused) nextVideo.play().catch(() => {});
        drawOverlay(app);
      }, {once: true});
    }
    toast(trackId === null ? `Shot ${shotIndex + 1} marked off-screen.` : `Shot ${shotIndex + 1} aligned to Face.`);
  } catch (error) {
    toast(error.message, {error: true});
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
        const seek = () => { video.currentTime = jumpTime + 0.08; drawOverlay(app); };
        if (video.readyState >= 1) seek();
        else video.addEventListener("loadedmetadata", seek, {once: true});
      }
      return;
    }
    const assign = event.target.closest("[data-ls-assign]");
    if (assign) {
      const raw = assign.dataset.trackId;
      await saveOverride(app, assign.dataset.lsAssign, raw === "off" ? null : Number(raw));
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
      app.startRun("lip-sync-detect", {session: app.session.id, force: !!lsState.data?.plan});
    } else if (action === "render") {
      app.startRun("lip-sync", {
        session: app.session.id,
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
        const seek = () => { video.currentTime = next.shot.start + 0.08; drawOverlay(app); };
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
