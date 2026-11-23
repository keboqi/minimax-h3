// Step 4, Lip sync (optional): 1. Auto detect -> 2. Play & confirm face alignment -> 3. Confirm & render.

import {api} from "./api.js";
import {esc, icon, plural, toast} from "./ui.js";

const SIZE_PRESETS = [[0, 100, "All"], [8, 100, "8%+"], [15, 100, "15%+"], [25, 100, "25%+"]];
const ENGINE_OPTIONS = [
  ["h3", "MiniMax H3", "Preserves dubbed audio &amp; subtitles · MediaPipe landmark mouth mask"],
  ["musetalk", "MuseTalk 1.5 · Fast", "Preserves dubbed audio &amp; subtitles · Fast single-step 256×256 inpainting"],
  ["latentsync", "LatentSync 1.6 · Quality", "Preserves dubbed audio &amp; subtitles · 512×512 Whisper-conditioned diffusion"],
];

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
  engine: "h3",
  strength: 0.58,
  feather: 6,
  bboxShift: 0,
  inferenceSteps: 20,
  guidanceScale: 1.5,
  minFaceSize: 0,
  maxFaceSize: 100,
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
    Object.assign(lsState, {
      targetKey: key, data: null, saving: false, selectedShot: 0, selectedSpeaker: null,
      minFaceSize: 0, maxFaceSize: 100,
    });
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

function trackMedianHeight(track) {
  const obs = track?.obs || [];
  if (obs.length) {
    const heights = obs.map((o) => Number(o[4])).filter(Number.isFinite).sort((a, b) => a - b);
    if (heights.length) return heights[Math.floor(heights.length / 2)];
  }
  const boxH = Number(track?.box?.[3]);
  return Number.isFinite(boxH) ? boxH : null;
}

function windowFaceSize(win) {
  return typeof win.face_size === "number" && Number.isFinite(win.face_size) ? win.face_size : null;
}

function isWindowManual(win, plan) {
  if (typeof win?.manual === "boolean") return win.manual;
  const a = (plan?.assignments || []).find((item) => item.shot_index === win.shot_index && item.speaker === win.speaker);
  return a?.status === "manual";
}

function sizeInRange(size) {
  if (size === null) return true;
  const min = lsState.minFaceSize / 100;
  const max = lsState.maxFaceSize / 100;
  return size >= min - 1e-6 && size <= max + 1e-6;
}

function windowInSizeRange(win, plan) {
  if (isWindowManual(win, plan)) return true;
  return sizeInRange(windowFaceSize(win));
}

function filteredWindowsInfo(plan) {
  const wins = plan?.windows || [];
  const total = plan?.windows_count ?? wins.length;
  if (!wins.length) return {total, included: total, filtered: 0};
  const included = wins.filter((w) => windowInSizeRange(w, plan)).length;
  return {total: wins.length, included, filtered: wins.length - included};
}

function itemFaceSize(it, plan) {
  const wins = (plan?.windows || []).filter((w) => w.shot_index === it.shot.index && w.speaker === it.speaker);
  const winSizes = wins.map(windowFaceSize).filter((v) => v !== null);
  if (winSizes.length) {
    winSizes.sort((a, b) => a - b);
    return winSizes[Math.floor(winSizes.length / 2)];
  }
  if (it.assignment.track_id !== null && it.assignment.track_id !== undefined) {
    const track = (it.shot.tracks || []).find((t) => t.track_id === it.assignment.track_id);
    if (track) return trackMedianHeight(track);
  }
  return null;
}

function isItemSizeFiltered(it, plan) {
  if (it.assignment.status !== "matched") return false;
  const wins = (plan?.windows || []).filter((w) => w.shot_index === it.shot.index && w.speaker === it.speaker);
  return wins.length > 0 && wins.every((w) => !windowInSizeRange(w, plan));
}

function shotCounts(items) {
  let uncertainCount = 0;
  let confirmedCount = 0;
  let offCount = 0;
  for (const it of items) {
    if (it.assignment.status === "uncertain") uncertainCount++;
    else if (it.assignment.status === "off_screen") offCount++;
    else confirmedCount++;
  }
  return {uncertainCount, confirmedCount, offCount};
}

function rowBadgeFor(it, plan) {
  return isItemSizeFiltered(it, plan) ? {label: "Size ignored", cls: "chip chip-quiet"} : statusBadge(it.assignment);
}

function formatFacePct(size) {
  return size === null ? "" : `${Math.round(size * 100)}%`;
}

function faceSizeLabel(plan) {
  const minPct = lsState.minFaceSize;
  const maxPct = lsState.maxFaceSize;
  const h = Number(plan?.height);
  const px = Number.isFinite(h) && h > 0
    ? ` (~${Math.round((minPct / 100) * h)}–${Math.round((maxPct / 100) * h)} px)`
    : "";
  return `${minPct}% – ${maxPct}%${px}`;
}

function shotTargetDescription(it, plan) {
  if (it.assignment.status === "off_screen") return "Skipped (off-screen)";
  const trackIdx = (it.shot.tracks || []).findIndex((t) => t.track_id === it.assignment.track_id);
  if (trackIdx < 0) return it.assignment.reason || it.assignment.source;
  const faceSize = itemFaceSize(it, plan);
  const pct = formatFacePct(faceSize);
  const manualOutside = it.assignment.status === "manual" && !sizeInRange(faceSize);
  return `Face #${trackIdx + 1}${pct ? ` (${pct})` : ""} · ${manualOutside ? "Manual override" : it.assignment.source}`;
}

function renderSummaryHtml(plan, confirmedCount, uncertainCount, offCount) {
  const info = filteredWindowsInfo(plan);
  const winText = info.filtered > 0
    ? `${info.included} of ${info.total} render windows (${info.filtered} size-ignored)`
    : plural(info.total, "render window");
  return `<strong>${confirmedCount}</strong> matched · <strong>${uncertainCount}</strong> needs check · <strong>${offCount}</strong> off-screen (${esc(winText)})`;
}

function formatSettingOut(key, value) {
  if (key === "strength") return Number(value).toFixed(2);
  if (key === "guidanceScale") return Number(value).toFixed(1);
  if (key === "bboxShift") return `${value > 0 ? "+" : ""}${value} px`;
  if (key === "inferenceSteps") return `${value} steps`;
  return `${value} px`;
}

function engineSettingsHtml() {
  const featherLabel = lsState.engine === "musetalk" ? "Fallback mask feather" : "Mask feather";
  const feather = `
    <label class="field">
      <span class="field-label">${featherLabel} <span class="mono" data-ls-out="feather">${formatSettingOut("feather", lsState.feather)}</span></span>
      <input type="range" min="2" max="16" step="1" value="${lsState.feather}" data-ls-setting="feather">
    </label>`;
  if (lsState.engine === "musetalk") {
    return `
      <label class="field">
        <span class="field-label">Mouth box shift <span class="mono" data-ls-out="bboxShift">${formatSettingOut("bboxShift", lsState.bboxShift)}</span></span>
        <input type="range" min="-15" max="15" step="1" value="${lsState.bboxShift}" data-ls-setting="bboxShift">
      </label>
      ${feather}`;
  }
  if (lsState.engine === "latentsync") {
    return `
      <label class="field">
        <span class="field-label">Diffusion steps <span class="mono" data-ls-out="inferenceSteps">${formatSettingOut("inferenceSteps", lsState.inferenceSteps)}</span></span>
        <input type="range" min="10" max="40" step="5" value="${lsState.inferenceSteps}" data-ls-setting="inferenceSteps">
      </label>
      <label class="field">
        <span class="field-label">Guidance scale <span class="mono" data-ls-out="guidanceScale">${formatSettingOut("guidanceScale", lsState.guidanceScale)}</span></span>
        <input type="range" min="1.0" max="3.0" step="0.1" value="${lsState.guidanceScale}" data-ls-setting="guidanceScale">
      </label>
      ${feather}`;
  }
  return `
    <label class="field">
      <span class="field-label">Denoise strength <span class="mono" data-ls-out="strength">${formatSettingOut("strength", lsState.strength)}</span></span>
      <input type="range" min="0.25" max="0.90" step="0.05" value="${lsState.strength}" data-ls-setting="strength">
    </label>
    ${feather}`;
}

function renderLockedReason(ls, app, plan) {
  if (!ls.ready) return "Generate this dub again to save its fitted dialogue.";
  if (!app.boot.lip_sync) return "Open this dub in Studio to render lip sync.";
  if (plan && !plan.windows_count) return "No render windows. Review speaker assignments and choose a visible face.";
  if (plan && filteredWindowsInfo(plan).included === 0) {
    return "No windows match the face size range. Widen the range or manually select a face.";
  }
  return "";
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

  const {uncertainCount, confirmedCount, offCount} = shotCounts(items);
  const unavailable = !ls.ready ? "Generate this dub again to save its fitted dialogue." : "";
  const renderReason = renderLockedReason(ls, app, plan);
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

      <p class="muted small" id="ls-render-reason" ${renderReason ? "" : "hidden"}>${esc(renderReason)}</p>
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
                <span class="muted small">${(ENGINE_OPTIONS.find(([id]) => id === lsState.engine) || ENGINE_OPTIONS[0])[2]}</span>
              </div>
              <div class="row align-center row-wrap">
                <span class="field-label">Engine</span>
                <div class="seg" role="group" aria-label="Lip-sync engine">
                  ${ENGINE_OPTIONS.map(([id, label]) => `
                    <button type="button" data-ls-engine="${id}" aria-pressed="${lsState.engine === id}">${esc(label)}</button>
                  `).join("")}
                </div>
              </div>
              <div class="ls-settings-row">
                ${engineSettingsHtml()}
              </div>
              <div class="field ls-range-field">
                <div class="row align-center row-wrap">
                  <span class="field-label">Face size range <span class="mono" data-ls-out="face-size">${esc(faceSizeLabel(plan))}</span></span>
                  <span class="spacer"></span>
                  <div class="ls-size-presets row row-wrap">
                    ${SIZE_PRESETS.map(([lo, hi, label]) => `
                      <button type="button" class="btn btn-small ${lsState.minFaceSize === lo && lsState.maxFaceSize === hi ? "btn-primary" : ""}"
                        data-ls-size-preset="${lo},${hi}">${label}</button>
                    `).join("")}
                  </div>
                </div>
                <div class="ls-range-dual">
                  <div class="ls-range-track">
                    <div class="ls-range-fill" id="ls-range-fill" style="left:${lsState.minFaceSize}%;width:${Math.max(0, lsState.maxFaceSize - lsState.minFaceSize)}%"></div>
                  </div>
                  <input type="range" min="0" max="100" step="1" value="${lsState.minFaceSize}" data-ls-range="min" aria-label="Minimum face height percent" style="z-index:${lsState.minFaceSize >= 95 ? 4 : 3}">
                  <input type="range" min="0" max="100" step="1" value="${lsState.maxFaceSize}" data-ls-range="max" aria-label="Maximum face height percent" style="z-index:${lsState.minFaceSize >= 95 ? 3 : 4}">
                </div>
                <span class="muted small">Auto-ignores faces outside this height range; manual face selections are always included.</span>
              </div>
              <div class="row align-center row-wrap top-line">
                <span class="small" data-ls-render-summary>
                  ${renderSummaryHtml(plan, confirmedCount, uncertainCount, offCount)}
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
                const badge = rowBadgeFor(it, plan);
                const selected = activeItem && it.key === activeItem.key;
                const targetDesc = shotTargetDescription(it, plan);
                return `
                  <button type="button" class="ls-shot-row ${selected ? "selected" : ""}" data-ls-shot="${it.shot.index}" data-ls-speaker="${esc(it.speaker)}" data-ls-time="${it.seekTime}">
                    <div class="row align-center">
                      <strong>Shot ${it.shot.index + 1}</strong>
                      <span class="mono muted">${formatClock(it.shot.start)}–${formatClock(it.shot.end)}</span>
                      <span class="spacer"></span>
                      <span class="${badge.cls}" data-ls-row-badge="${esc(it.key)}">${badge.label}</span>
                    </div>
                    <div class="small ellipsis"><strong>${esc(speakerLabel(it.speaker))}</strong>${it.line ? ` · ${esc(it.line)}` : ""}</div>
                    <div class="small muted ellipsis" data-ls-row-desc="${esc(it.key)}">${esc(targetDesc)}</div>
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
  const activeWindow = (plan.windows || []).find((win) =>
    win.shot_index === shot.index && win.speaker === activeSpeaker && t >= win.start && t < win.end
  );

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
    const speakingWindow = isSpeaking ? activeWindow : null;
    const trackSize = (speakingWindow && windowFaceSize(speakingWindow)) ?? trackMedianHeight(track) ?? h;
    const pct = formatFacePct(trackSize);
    const sizeIgnored = !!speakingWindow && !windowInSizeRange(speakingWindow, plan);
    const manualKept = !!speakingWindow && isWindowManual(speakingWindow, plan)
      && !sizeInRange(windowFaceSize(speakingWindow));
    const label = isSpeaking
      ? `${speakerLabel(activeSpeaker)} · ${pct}${sizeIgnored ? " (size ignored)" : manualKept ? " (manual)" : ""}`
      : `Face #${idx + 1} · ${pct}`;
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
    btn.classList.toggle("size-ignored", sizeIgnored);
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

function syncFaceSizeUI(app) {
  const data = lsState.targetKey === JSON.stringify(reviewTarget(app)) ? lsState.data : null;
  const plan = data?.plan;
  if (!plan) return;
  const fill = document.getElementById("ls-range-fill");
  if (fill) {
    fill.style.left = `${lsState.minFaceSize}%`;
    fill.style.width = `${Math.max(0, lsState.maxFaceSize - lsState.minFaceSize)}%`;
  }
  const minInput = document.querySelector('[data-ls-range="min"]');
  const maxInput = document.querySelector('[data-ls-range="max"]');
  if (minInput) {
    if (Number(minInput.value) !== lsState.minFaceSize) minInput.value = String(lsState.minFaceSize);
    minInput.style.zIndex = lsState.minFaceSize >= 95 ? "4" : "3";
  }
  if (maxInput) {
    if (Number(maxInput.value) !== lsState.maxFaceSize) maxInput.value = String(lsState.maxFaceSize);
    maxInput.style.zIndex = lsState.minFaceSize >= 95 ? "3" : "4";
  }
  const out = document.querySelector('[data-ls-out="face-size"]');
  if (out) out.textContent = faceSizeLabel(plan);
  for (const btn of document.querySelectorAll("[data-ls-size-preset]")) {
    const [lo, hi] = btn.dataset.lsSizePreset.split(",").map(Number);
    btn.classList.toggle("btn-primary", lsState.minFaceSize === lo && lsState.maxFaceSize === hi);
  }
  const items = shotItems(plan);
  const {uncertainCount, confirmedCount, offCount} = shotCounts(items);
  const summary = document.querySelector("[data-ls-render-summary]");
  if (summary) summary.innerHTML = renderSummaryHtml(plan, confirmedCount, uncertainCount, offCount);
  const ls = app.session?.result?.lip_sync;
  const reason = ls ? renderLockedReason(ls, app, plan) : "";
  const reasonEl = document.getElementById("ls-render-reason");
  if (reasonEl) {
    reasonEl.textContent = reason;
    reasonEl.hidden = !reason;
  }
  const renderBtn = document.querySelector('[data-ls-action="render"]');
  if (renderBtn) {
    const locked = !(plan?.revision && !reason && !lsState.loading);
    renderBtn.dataset.locked = String(locked);
    const busy = !!app.runs?.auto?.active || !!app.runs?.["lip-sync"]?.active || !!app.runs?.["lip-sync-detect"]?.active;
    renderBtn.disabled = busy || locked;
  }
  for (const it of items) {
    const badgeEl = document.querySelector(`[data-ls-row-badge="${CSS.escape(it.key)}"]`);
    if (badgeEl) {
      const badge = rowBadgeFor(it, plan);
      badgeEl.className = badge.cls;
      badgeEl.textContent = badge.label;
    }
    const descEl = document.querySelector(`[data-ls-row-desc="${CSS.escape(it.key)}"]`);
    if (descEl) {
      descEl.textContent = shotTargetDescription(it, plan);
    }
  }
  drawOverlay(app);
}

export function bindLipSync(app) {
  const screen = document.getElementById("screen");
  screen.addEventListener("click", async (event) => {
    if (app.screen !== "lipsync" || !app.session) return;
    const engineBtn = event.target.closest("[data-ls-engine]");
    if (engineBtn) {
      const nextEngine = engineBtn.dataset.lsEngine;
      if (nextEngine && nextEngine !== lsState.engine) {
        const video = document.getElementById("ls-video");
        const savedTime = video ? video.currentTime : null;
        const wasPaused = video ? video.paused : true;
        lsState.engine = nextEngine;
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
      }
      return;
    }
    const preset = event.target.closest("[data-ls-size-preset]");
    if (preset) {
      const [lo, hi] = preset.dataset.lsSizePreset.split(",").map(Number);
      lsState.minFaceSize = lo;
      lsState.maxFaceSize = hi;
      syncFaceSizeUI(app);
      return;
    }
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
        engine: lsState.engine,
        strength: lsState.strength,
        feather: lsState.feather,
        bbox_shift: lsState.bboxShift,
        inference_steps: lsState.inferenceSteps,
        guidance_scale: lsState.guidanceScale,
        min_face_size: lsState.minFaceSize / 100,
        max_face_size: lsState.maxFaceSize / 100,
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
    const rangeEl = event.target.closest("[data-ls-range]");
    if (rangeEl) {
      const val = Math.max(0, Math.min(100, Math.round(Number(rangeEl.value) || 0)));
      if (rangeEl.dataset.lsRange === "min") {
        lsState.minFaceSize = Math.min(val, lsState.maxFaceSize);
      } else {
        lsState.maxFaceSize = Math.max(val, lsState.minFaceSize);
      }
      syncFaceSizeUI(app);
      return;
    }
    const el = event.target.closest("[data-ls-setting]");
    if (!el) return;
    const key = el.dataset.lsSetting;
    lsState[key] = Number(el.value);
    const out = document.querySelector(`[data-ls-out="${key}"]`);
    if (out) out.textContent = formatSettingOut(key, lsState[key]);
  });
}
