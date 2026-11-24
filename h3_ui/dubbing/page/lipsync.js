// Step 4, Lip sync (optional): 1. Auto detect -> 2. Play & confirm face alignment -> 3. Confirm & render.

import {api} from "./api.js";
import {esc, icon, plural, toast} from "./ui.js";

const SIZE_PRESETS = [[0, 100, "All"], [8, 100, "8%+"], [15, 100, "15%+"], [25, 100, "25%+"]];
const ENGINE_OPTIONS = [
  ["h3", "MiniMax H3", "Preserves dubbed audio &amp; subtitles · MediaPipe landmark mouth mask"],
  ["musetalk", "MuseTalk 1.5 · Fast", "Preserves dubbed audio &amp; subtitles · Fast single-step 256×256 inpainting"],
  ["latentsync", "LatentSync 1.6 · Quality", "Preserves dubbed audio &amp; subtitles · 512×512 Whisper-conditioned diffusion"],
];

const LS_SETTINGS_KEY = "h3-dubbing-lipsync-settings";

function loadSavedLsSettings() {
  try {
    const raw = JSON.parse(localStorage.getItem(LS_SETTINGS_KEY) || "{}");
    return raw && typeof raw === "object" ? raw : {};
  } catch {
    return {};
  }
}

function saveLsSettings() {
  try {
    localStorage.setItem(LS_SETTINGS_KEY, JSON.stringify({
      engine: lsState.engine,
      strength: lsState.strength,
      feather: lsState.feather,
      bboxShift: lsState.bboxShift,
      inferenceSteps: lsState.inferenceSteps,
      guidanceScale: lsState.guidanceScale,
      minFaceSize: lsState.minFaceSize,
      maxFaceSize: lsState.maxFaceSize,
    }));
  } catch {
    // Ignore storage errors in private browsing.
  }
}

const savedLs = loadSavedLsSettings();
const savedMinFaceSize = Number.isFinite(savedLs.minFaceSize)
  ? Math.max(0, Math.min(100, Math.round(savedLs.minFaceSize)))
  : 0;
const savedMaxFaceSize = Number.isFinite(savedLs.maxFaceSize)
  ? Math.max(savedMinFaceSize, Math.min(100, Math.round(savedLs.maxFaceSize)))
  : 100;

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
  stageMode: "align",
  loopShot: true,
  checkedKeys: new Set(),
  lastRenderedKeys: [],
  autoplayPreview: false,
  queueScrollTop: 0,
  videoTime: null,
  engine: savedLs.engine || "h3",
  strength: Number.isFinite(savedLs.strength) ? savedLs.strength : 0.58,
  feather: Number.isFinite(savedLs.feather) ? savedLs.feather : 6,
  bboxShift: Number.isFinite(savedLs.bboxShift) ? savedLs.bboxShift : 0,
  inferenceSteps: Number.isFinite(savedLs.inferenceSteps) ? savedLs.inferenceSteps : 20,
  guidanceScale: Number.isFinite(savedLs.guidanceScale) ? savedLs.guidanceScale : 1.5,
  minFaceSize: savedMinFaceSize,
  maxFaceSize: savedMaxFaceSize,
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

export async function loadLipSyncPlan(app, {silent = false, openPreview = false} = {}) {
  const target = reviewTarget(app);
  if (!target) return;
  const key = JSON.stringify(target);
  if (lsState.targetKey !== key) {
    Object.assign(lsState, {
      targetKey: key, data: null, saving: false, selectedShot: 0, selectedSpeaker: null,
      stageMode: "align", checkedKeys: new Set(), lastRenderedKeys: [], autoplayPreview: false,
      queueScrollTop: 0, videoTime: null,
    });
  }
  const request = Symbol();
  Object.assign(lsState, {request, loading: true, error: null});
  const isCurrent = () => lsState.request === request && JSON.stringify(reviewTarget(app)) === key;
  if (!silent) app.render();
  try {
    const data = await api.get("lip-sync-plan", target);
    if (isCurrent()) {
      lsState.data = data;
      if (openPreview) {
        const focusKey = lsState.lastRenderedKeys[0]
          || (lsState.selectedSpeaker ? `${lsState.selectedShot}:${lsState.selectedSpeaker}` : null);
        lsState.lastRenderedKeys = [];
        if (focusKey) {
          const sep = focusKey.indexOf(":");
          if (sep > 0) {
            lsState.selectedShot = Number(focusKey.slice(0, sep));
            lsState.selectedSpeaker = focusKey.slice(sep + 1);
          }
          const take = activeTakeFor(data, focusKey);
          if (take?.preview_url) {
            lsState.stageMode = "preview";
            lsState.autoplayPreview = true;
          }
        }
      }
    }
  } catch (error) {
    if (isCurrent()) lsState.error = error.message;
  } finally {
    if (lsState.request === request) lsState.loading = false;
    if (isCurrent() && app.screen === "lipsync") app.render();
  }
}

function shotTakeEntry(data, key) {
  const entry = data?.takes?.shots?.[key];
  return entry && typeof entry === "object"
    ? {active_take_id: entry.active_take_id || "original", takes: Array.isArray(entry.takes) ? entry.takes : []}
    : {active_take_id: "original", takes: []};
}

function activeTakeFor(data, key) {
  const entry = shotTakeEntry(data, key);
  if (entry.active_take_id === "original") return null;
  return entry.takes.find((tk) => tk.id === entry.active_take_id) || null;
}

function assembledTakeId(data, key) {
  return data?.takes?.assembled?.shot_takes?.[key] || null;
}

function shotTakeStatus(it, data) {
  const active = activeTakeFor(data, it.key);
  const assembledId = assembledTakeId(data, it.key);
  const hasAssembled = !!data?.takes?.assembled;
  if (active) {
    if (it.assignment.status === "off_screen") return "updated";
    if (active.track_id !== null && active.track_id !== undefined
        && it.assignment.track_id !== null && it.assignment.track_id !== undefined
        && Number(active.track_id) !== Number(it.assignment.track_id)) {
      return "outdated";
    }
    if (hasAssembled && assembledId === active.id) return "in_export";
    return "updated";
  }
  if (hasAssembled && assembledId) return "updated";
  if (it.assignment.status === "off_screen") return "off_screen";
  return "unrendered";
}

function itemHasWindows(it, plan) {
  return !!(it && (getPlanIndex(plan)?.windowsByKey.get(it.key) || []).length > 0);
}

function takeStatusBadge(it, plan, data) {
  const active = activeTakeFor(data, it.key);
  const st = shotTakeStatus(it, data);
  if (st === "outdated") return {label: `Take ${active.number} · Outdated`, cls: "chip chip-warn"};
  if (st === "updated") {
    return active
      ? {label: `Take ${active.number} · Updated`, cls: "chip chip-warn"}
      : {label: "Original · Updated", cls: "chip chip-warn"};
  }
  if (st === "in_export") return {label: `Take ${active.number} · In export`, cls: "chip"};
  if (st === "off_screen") return {label: "Off-screen", cls: "chip chip-quiet"};
  if (isItemSizeFiltered(it, plan)) return {label: "Size ignored", cls: "chip chip-quiet"};
  if (it.assignment.status === "uncertain") return {label: "Needs check", cls: "chip chip-warn"};
  if (!itemHasWindows(it, plan)) return {label: "No speaking frames", cls: "chip chip-quiet"};
  return {label: "Not rendered", cls: "chip chip-quiet"};
}

function formatEngineShort(engine) {
  if (engine === "musetalk") return "MuseTalk";
  if (engine === "latentsync") return "LatentSync";
  return "H3";
}

function formatTakeMeta(take, shot) {
  const eng = formatEngineShort(take.engine);
  const idx = (shot?.tracks || []).findIndex((t) => Number(t.track_id) === Number(take.track_id));
  const face = idx >= 0 ? `Face #${idx + 1}` : "Face";
  const detail = take.engine === "h3" && take.settings?.strength != null
    ? `${eng} ${Number(take.settings.strength).toFixed(2)}`
    : take.engine === "latentsync" && take.settings?.inference_steps != null
      ? `${eng} ${take.settings.inference_steps}st`
      : eng;
  return `${detail} · ${face}`;
}

function assemblySummary(items, data, plan = null) {
  let reused = 0;
  let updated = 0;
  let outdated = 0;
  let unrendered = 0;
  let off = 0;
  let activeTakesCount = 0;
  const changedShots = [];
  const unrenderedKeys = [];
  for (const it of items) {
    const active = activeTakeFor(data, it.key);
    if (active) activeTakesCount++;
    const st = shotTakeStatus(it, data);
    if (st === "in_export") reused++;
    else if (st === "updated") { updated++; changedShots.push(it); }
    else if (st === "outdated") { outdated++; changedShots.push(it); }
    else if (st === "off_screen") off++;
    else if (plan && it.assignment.status !== "uncertain" && !itemHasWindows(it, plan)) off++;
    else { unrendered++; if (it.assignment.status !== "off_screen") unrenderedKeys.push(it.key); }
  }
  return {
    reused, updated, outdated, unrendered, off, activeTakesCount,
    hasAssembled: !!data?.takes?.assembled,
    changedCount: updated + outdated,
    changedShots,
    unrenderedKeys,
  };
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

const planIndexes = new WeakMap();

function getPlanIndex(plan) {
  if (!plan) return null;
  let idx = planIndexes.get(plan);
  if (idx) return idx;
  const assignments = new Map();
  const firstAssignmentByShot = new Map();
  const assignmentsByShot = new Map();
  for (const a of plan.assignments || []) {
    assignments.set(`${a.shot_index}:${a.speaker}`, a);
    if (!firstAssignmentByShot.has(a.shot_index)) firstAssignmentByShot.set(a.shot_index, a);
    let shotList = assignmentsByShot.get(a.shot_index);
    if (!shotList) {
      shotList = [];
      assignmentsByShot.set(a.shot_index, shotList);
    }
    shotList.push(a);
  }
  const shotsByIndex = new Map();
  let shotsSorted = true;
  let prevShotStart = -Infinity;
  for (const shot of plan.shots || []) {
    shotsByIndex.set(shot.index, shot);
    if (shot.start < prevShotStart) shotsSorted = false;
    prevShotStart = shot.start;
  }
  const windowsByKey = new Map();
  for (const win of plan.windows || []) {
    const key = `${win.shot_index}:${win.speaker}`;
    let list = windowsByKey.get(key);
    if (!list) {
      list = [];
      windowsByKey.set(key, list);
    }
    list.push(win);
  }
  let turnsSorted = true;
  let prevTurnStart = -Infinity;
  let maxTurnDuration = 0;
  for (const ln of plan.turns || []) {
    if (ln.start < prevTurnStart) turnsSorted = false;
    prevTurnStart = ln.start;
    const dur = ln.end - ln.start;
    if (dur > maxTurnDuration) maxTurnDuration = dur;
  }
  idx = {
    assignments,
    firstAssignmentByShot,
    assignmentsByShot,
    shotsByIndex,
    shotsSorted,
    windowsByKey,
    turnsSorted,
    maxTurnDuration,
    trackMedians: new WeakMap(),
  };
  planIndexes.set(plan, idx);
  return idx;
}

function assignmentMap(plan) {
  return getPlanIndex(plan)?.assignments || new Map();
}

function findShotAtTime(plan, idx, t) {
  const shots = plan?.shots || [];
  if (!shots.length) return null;
  if (!idx?.shotsSorted) return shots.find((s) => t >= s.start && t < s.end) || null;
  let lo = 0;
  let hi = shots.length - 1;
  let candidate = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (shots[mid].start <= t) {
      candidate = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  if (candidate >= 0 && t >= shots[candidate].start && t < shots[candidate].end) return shots[candidate];
  return null;
}

function findTurnAtTime(plan, idx, t) {
  const turns = plan?.turns || [];
  if (!turns.length) return null;
  if (!idx?.turnsSorted) return turns.find((ln) => t >= ln.start && t < ln.end) || null;
  let lo = 0;
  let hi = turns.length - 1;
  let candidate = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (turns[mid].start <= t) {
      candidate = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  if (candidate < 0) return null;
  const minStart = t - idx.maxTurnDuration;
  let firstMatch = null;
  for (let i = candidate; i >= 0 && turns[i].start >= minStart; i--) {
    const ln = turns[i];
    if (t >= ln.start && t < ln.end) firstMatch = ln;
  }
  return firstMatch;
}

function shotItems(plan) {
  if (!plan?.shots) return [];
  const idx = getPlanIndex(plan);
  const turns = plan.turns || [];
  const items = [];
  for (const shot of plan.shots) {
    const shotAssignments = idx?.assignmentsByShot.get(shot.index) || [];
    if (!shotAssignments.length) continue;
    const shotTurns = turns.filter((t) => Math.min(t.end, shot.end) - Math.max(t.start, shot.start) >= 0.04);
    for (const a of shotAssignments) {
      const firstTurn = shotTurns.find((t) => t.speaker === a.speaker);
      items.push({
        shot,
        assignment: a,
        speaker: a.speaker,
        seekTime: firstTurn
          ? (Math.max(shot.start, firstTurn.start) + Math.min(shot.end, firstTurn.end)) / 2
          : Math.max(shot.start, Math.min(shot.end - 0.001, a.reference_time ?? shot.start)),
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

function assignedTrackId(assignment) {
  if (!assignment || assignment.status === "off_screen") return null;
  return assignment.track_id !== null && assignment.track_id !== undefined ? assignment.track_id : null;
}

function isTrackInSizeRange(track, index = null, fallbackHeight = null) {
  return sizeInRange(trackMedianHeight(track, index) ?? fallbackHeight);
}

function assignmentButtons(shot, assignment, overrides, plan = null) {
  if (!shot || !assignment) return "";
  const index = getPlanIndex(plan);
  const key = `${shot.index}:${assignment.speaker}`;
  const activeTrackId = assignedTrackId(assignment);
  const button = (id, label, selected) => `<button type="button" class="btn btn-small ${selected ? "btn-primary" : ""}"
    data-ls-assign="${esc(key)}" data-track-id="${id}">${label}</button>`;
  const buttons = [];
  let hiddenCount = 0;
  (shot.tracks || []).forEach((track, idx) => {
    const selected = activeTrackId !== null && track.track_id === activeTrackId;
    if (!selected && !isTrackInSizeRange(track, index)) {
      hiddenCount++;
      return;
    }
    buttons.push(button(track.track_id, `Face #${idx + 1}`, selected));
  });
  buttons.push(button("off", "Off-screen / Skip", assignment.status === "off_screen"));
  if (Object.hasOwn(overrides || {}, key)) buttons.push(button("auto", "Restore automatic match", false));
  if (hiddenCount > 0) {
    buttons.push(`<span class="muted small" data-ls-hidden-faces="${hiddenCount}">${plural(hiddenCount, "face")} hidden by size</span>`);
  }
  return buttons.join("");
}

function boxAtTime(track, time) {
  const obs = track.obs || [];
  if (!obs.length) return track.box || null;
  let lo = 0;
  let hi = obs.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (obs[mid][0] < time) lo = mid + 1;
    else hi = mid;
  }
  let best = obs[lo];
  let bestDt = Math.abs(best[0] - time);
  if (lo > 0) {
    const prev = obs[lo - 1];
    const dt = Math.abs(prev[0] - time);
    if (dt < bestDt) {
      bestDt = dt;
      best = prev;
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

function trackMedianHeight(track, index = null) {
  if (!track) return null;
  if (index?.trackMedians.has(track)) return index.trackMedians.get(track);
  let median = null;
  const obs = track.obs || [];
  if (obs.length) {
    const heights = obs.map((o) => Number(o[4])).filter(Number.isFinite).sort((a, b) => a - b);
    if (heights.length) median = heights[Math.floor(heights.length / 2)];
  }
  if (median === null) {
    const boxH = Number(track.box?.[3]);
    if (Number.isFinite(boxH)) median = boxH;
  }
  index?.trackMedians.set(track, median);
  return median;
}

function windowFaceSize(win) {
  return typeof win.face_size === "number" && Number.isFinite(win.face_size) ? win.face_size : null;
}

function isWindowManual(win, plan) {
  if (typeof win?.manual === "boolean") return win.manual;
  const a = getPlanIndex(plan)?.assignments.get(`${win.shot_index}:${win.speaker}`);
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
  const idx = getPlanIndex(plan);
  const wins = idx?.windowsByKey.get(it.key) || [];
  const winSizes = wins.map(windowFaceSize).filter((v) => v !== null);
  if (winSizes.length) {
    winSizes.sort((a, b) => a - b);
    return winSizes[Math.floor(winSizes.length / 2)];
  }
  if (it.assignment.track_id !== null && it.assignment.track_id !== undefined) {
    const track = (it.shot.tracks || []).find((t) => t.track_id === it.assignment.track_id);
    if (track) return trackMedianHeight(track, idx);
  }
  return null;
}

function isItemSizeFiltered(it, plan) {
  if (it.assignment.status !== "matched") return false;
  const wins = getPlanIndex(plan)?.windowsByKey.get(it.key) || [];
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

function detectLockedReason(ls, app) {
  if (!ls.ready) return "Generate this dub again to save its fitted dialogue.";
  if (!app.boot.lip_sync) return "Open this dub in Studio to detect faces and speakers.";
  return "";
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

function stageTopHtml(activeItem, activeEntry, activeTake, isPreviewMode, showingSyncedTake) {
  const hasAnyTake = activeEntry.takes.length > 0;
  return `
    <div class="seg" role="group" aria-label="Stage view mode">
      <button type="button" data-ls-stage="align" aria-pressed="${!isPreviewMode}">Edit face</button>
      <button type="button" data-ls-stage="preview" aria-pressed="${isPreviewMode}" ${hasAnyTake ? "" : 'disabled title="Render this shot first to preview a take"'}>
        Preview take${activeTake ? ` (T${activeTake.number})` : ""}
      </button>
    </div>
    ${isPreviewMode ? `
      <span class="chip ${showingSyncedTake ? "" : "chip-quiet"}">
        ${showingSyncedTake ? `Take ${activeTake.number} · Lip-synced` : "Original picture"}
      </span>
      <button type="button" class="btn btn-small ${lsState.loopShot ? "btn-primary" : ""}" data-ls-action="toggle-loop" aria-pressed="${lsState.loopShot}">Loop</button>
    ` : ""}
    <span class="spacer"></span>
    ${activeItem ? `<span class="mono muted small">Shot ${activeItem.shot.index + 1} · ${formatClock(activeItem.shot.start)}–${formatClock(activeItem.shot.end)}</span>` : ""}`;
}

function takesPanelHtml(activeItem, activeEntry, activeStatus, data) {
  if (!activeItem) return "";
  return `
    <div class="row align-center row-wrap">
      <strong class="small">Takes for Shot ${activeItem.shot.index + 1}</strong>
      <span class="muted small">${activeEntry.takes.length ? "Click a take to preview &amp; use in assembly" : "No take rendered yet — click Render shot above."}</span>
      <span class="spacer"></span>
      ${activeStatus === "outdated" ? '<span class="chip chip-warn">Face changed — click New take</span>' : ""}
    </div>
    ${activeEntry.takes.length ? `
      <div class="ls-takes-row">
        <div class="ls-take-card ${activeEntry.active_take_id === "original" ? "active" : ""}">
          <button type="button" class="ls-take-pick" data-ls-pick-take="original" data-shot-key="${esc(activeItem.key)}">
            <div class="row align-center">
              <strong class="small">Original</strong>
              <span class="spacer"></span>
              <span class="chip ${activeEntry.active_take_id === "original" ? "" : "chip-quiet"}">${activeEntry.active_take_id === "original" ? "Active" : "No sync"}</span>
            </div>
            <span class="muted small">Compare source picture</span>
          </button>
        </div>
        ${activeEntry.takes.map((tk) => {
          const isActive = activeEntry.active_take_id === tk.id;
          const inExport = assembledTakeId(data, activeItem.key) === tk.id;
          const pillLabel = isActive ? "Active ★" : (inExport ? "In export" : "Saved");
          const pillCls = isActive ? "chip" : "chip chip-quiet";
          return `
            <div class="ls-take-card ${isActive ? "active" : ""}">
              <button type="button" class="ls-take-pick" data-ls-pick-take="${esc(tk.id)}" data-shot-key="${esc(activeItem.key)}">
                <div class="row align-center">
                  <strong class="small">Take ${tk.number}</strong>
                  <span class="spacer"></span>
                  <span class="${pillCls}">${pillLabel}</span>
                </div>
                <span class="muted small">${esc(formatTakeMeta(tk, activeItem.shot))}</span>
              </button>
              <button type="button" class="ls-take-del" data-ls-delete-take="${esc(tk.id)}" data-shot-key="${esc(activeItem.key)}" title="Delete Take ${tk.number}" aria-label="Delete Take ${tk.number}">×</button>
            </div>`;
        }).join("")}
      </div>
    ` : ""}`;
}

function queueBulkHtml(checkedCount, canRender, busy) {
  if (!checkedCount) return "";
  return `
    <div class="ls-queue-bulk">
      <strong class="small">${checkedCount} selected</strong>
      <button type="button" class="btn btn-small btn-primary" data-ls-action="render-selected" data-busy="lip-sync" data-locked="${!canRender}" ${canRender && !busy ? "" : "disabled"}>
        Render selected (${checkedCount})
      </button>
    </div>`;
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
  const asm = assemblySummary(items, data, plan);
  const visible = items.filter((it) => {
    if (lsState.filter === "review") return it.assignment.status === "uncertain";
    if (lsState.filter === "pending") {
      const st = shotTakeStatus(it, data);
      return st === "updated" || st === "outdated" || st === "unrendered";
    }
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
  const detectReason = detectLockedReason(ls, app);
  const renderReason = renderLockedReason(ls, app, plan);
  const canRender = !!(plan?.revision && !renderReason && !lsState.loading);
  const canRenderShot = !!(
    plan?.revision && app.boot.lip_sync && ls.ready && !lsState.loading
    && activeItem && activeItem.assignment.status !== "off_screen"
    && itemHasWindows(activeItem, plan)
  );
  const canAssemble = !!(plan && asm.activeTakesCount > 0 && !lsState.loading);

  const activeEntry = activeItem ? shotTakeEntry(data, activeItem.key) : {active_take_id: "original", takes: []};
  const activeTake = activeItem ? activeTakeFor(data, activeItem.key) : null;
  const activeStatus = activeItem ? shotTakeStatus(activeItem, data) : "unrendered";
  const hasAnyTake = activeEntry.takes.length > 0;
  const isPreviewMode = lsState.stageMode === "preview" && !!activeItem && hasAnyTake;
  const showingSyncedTake = isPreviewMode && !!activeTake?.preview_url;
  const stageVideoSrc = showingSyncedTake ? activeTake.preview_url : (data?.video_url || "");

  const validKeys = new Set(items.map((it) => it.key));
  for (const k of Array.from(lsState.checkedKeys)) {
    if (!validKeys.has(k)) lsState.checkedKeys.delete(k);
  }
  const checkedCount = lsState.checkedKeys.size;
  const pendingCount = asm.changedCount + asm.unrendered;

  const failedWindows = Array.isArray(ls.report?.failed_windows) ? ls.report.failed_windows : [];
  const failedByKey = new Map();
  for (const f of failedWindows) {
    const k = `${f.shot_index}:${f.speaker}`;
    if (!failedByKey.has(k)) failedByKey.set(k, f);
  }
  const failedBanner = failedWindows.length ? `
    <section class="card card-pad stack-8">
      <div class="row align-center row-wrap">
        <span class="chip chip-warn">${plural(failedWindows.length, "window")} failed on last render (kept original frames)</span>
        <span class="muted small">Click a window to jump straight to that shot.</span>
      </div>
      <div class="stack-6">
        ${failedWindows.map((f) => `
          <div class="row align-center row-wrap small">
            <button type="button" class="btn btn-small" data-ls-shot="${Number(f.shot_index) || 0}" data-ls-speaker="${esc(f.speaker || "")}" data-ls-time="${Number(f.start) || 0}">
              Shot ${(Number(f.shot_index) || 0) + 1} (${formatClock(f.start)}–${formatClock(f.end)}) · ${esc(speakerLabel(f.speaker))}
            </button>
            <span class="fail-text">${esc(f.error || "Render failed")}</span>
          </div>`).join("")}
      </div>
    </section>` : "";

  const assembleLabel = asm.hasAssembled && asm.changedCount > 0
    ? `Update full video (${plural(asm.changedCount, "shot")})`
    : asm.hasAssembled && asm.changedCount === 0
      ? "Re-assemble full video"
      : `Assemble full video (${plural(asm.activeTakesCount, "take")})`;

  return `
    <div class="ls-workspace stack-16">
      <section class="card card-pad ls-stage-bar">
        <div class="row align-center row-wrap gap-28">
          <div class="ls-phases row align-center row-wrap">
            <span class="chip ${plan ? "" : "chip-warn"}">1. Auto detect${plan ? ` (${esc(plan.backend)})` : ""}</span>
            <span class="muted">→</span>
            <span class="chip ${plan ? "" : "chip-quiet"}">2. Confirm faces</span>
            <span class="muted">→</span>
            <span class="chip ${asm.activeTakesCount > 0 ? "" : "chip-quiet"}">3. Shot takes &amp; preview${plan ? ` (${asm.activeTakesCount}/${Math.max(0, items.length - offCount)})` : ""}</span>
            <span class="muted">→</span>
            <span class="chip ${asm.changedCount > 0 ? "chip-warn" : data?.output ? "" : "chip-quiet"}">4. Assemble${asm.changedCount > 0 ? ` (${asm.changedCount} changed)` : ""}</span>
          </div>
          <span class="spacer"></span>
          <div class="row row-wrap">
            <button type="button" class="btn btn-small" data-ls-action="detect" data-busy="lip-sync-detect" data-locked="${!!detectReason}" ${detectReason ? `disabled title="${esc(detectReason)}"` : ""}>
              ${plan ? "Re-run auto detect" : "Auto detect faces & speakers"}
            </button>
            <button type="button" class="btn btn-small" data-ls-action="to-result">Skip to Result →</button>
          </div>
        </div>
        <div class="run-status top-gap" data-run="lip-sync-detect" data-dub-video="${esc(ls.video)}" hidden>
          <span class="spinner" aria-hidden="true"></span>
          <span data-run-status role="status"></span>
          <button type="button" class="btn btn-small" data-ls-action="interrupt" data-stop="lip-sync-detect" data-kind="lip-sync-detect">Interrupt</button>
        </div>
        <div class="run-status top-gap" data-run="lip-sync" data-dub-video="${esc(ls.video)}" hidden>
          <span class="spinner" aria-hidden="true"></span>
          <span data-run-status role="status"></span>
          <button type="button" class="btn btn-small" data-ls-action="interrupt" data-stop="lip-sync" data-kind="lip-sync">Interrupt</button>
        </div>
      </section>

      <p class="muted small" id="ls-render-reason" ${renderReason ? "" : "hidden"}>${esc(renderReason)}</p>
      ${failedBanner}
      ${lsState.loading ? '<div class="card card-pad muted">Loading lip-sync plan…</div>' : ""}
      ${lsState.error ? `<div class="card card-pad fail-text">${esc(lsState.error)}</div>` : ""}

      ${!plan && !lsState.loading ? `
        <section class="card card-pad stack-14">
          <h2>Step 1 · Auto detect faces &amp; speaking speakers</h2>
          <p class="muted">Runs <code>face_yolov8m.pt</code> face tracking, aligns speakers using your Gemini <code>visual_speakers</code> hints, and uses MediaPipe mouth-motion (MAR) as backup. No video rendering starts until you pick shots or confirm.</p>
          ${app.boot.lip_sync ? "" : '<p class="alert alert-info">Auto face and speaker detection runs in Studio. Open this project in Studio to detect faces or render lip sync.</p>'}
          <div class="row row-wrap">
            <button type="button" class="btn ${app.boot.lip_sync ? "btn-primary" : ""}" data-ls-action="detect" data-busy="lip-sync-detect" data-locked="${!!detectReason}" ${detectReason ? `disabled title="${esc(detectReason)}"` : ""}>
              Auto detect faces &amp; speakers
            </button>
            <button type="button" class="btn ${app.boot.lip_sync ? "" : "btn-primary"}" data-ls-action="to-result">Skip to Result</button>
          </div>
        </section>
      ` : ""}

      ${plan ? `
        <div class="ls-grid">
          <section class="ls-main stack-14">
            <div class="card ls-player-card">
              <div class="ls-stage-top" id="ls-stage-top">
                ${stageTopHtml(activeItem, activeEntry, activeTake, isPreviewMode, showingSyncedTake)}
              </div>

              <div class="ls-video-stage" id="ls-video-stage">
                <video id="ls-video" src="${esc(stageVideoSrc)}" controls playsinline ${showingSyncedTake && lsState.loopShot ? "loop" : ""} preload="metadata" data-stage-mode="${isPreviewMode ? "preview" : "align"}" data-seek-time="${activeItem ? activeItem.seekTime : 0}" data-clip-start="${activeItem ? activeItem.shot.start : 0}" data-clip-end="${activeItem ? activeItem.shot.end : 0}"></video>
                ${!isPreviewMode ? '<div class="ls-overlay" id="ls-overlay" aria-label="Tracked face boxes"></div><div class="ls-subtitle-bar" id="ls-subtitle" hidden></div>' : `
                  <div class="ls-preview-hud">
                    <span class="ls-preview-dot ${showingSyncedTake ? "" : "orig"}"></span>
                    <span>${showingSyncedTake ? `Take ${activeTake.number} · ${esc(formatTakeMeta(activeTake, activeItem.shot))}` : "Original picture (no lip sync)"}</span>
                  </div>
                `}
              </div>

              <div class="ls-assign-bar">
                <div class="row align-center row-wrap">
                  <button type="button" class="btn btn-small" data-ls-action="prev-shot">← Prev</button>
                  <button type="button" class="btn btn-small" data-ls-action="next-shot">Next →</button>
                  <span class="muted small" id="ls-active-info">
                    ${assignmentLabel(activeItem?.shot, activeItem?.speaker)}
                  </span>
                </div>
                <div class="row row-wrap align-center">
                  <div class="row row-wrap align-center" id="ls-quick-buttons" data-assign-key="${esc(activeItem?.key || "")}" data-range="${lsState.minFaceSize}-${lsState.maxFaceSize}">
                    ${assignmentButtons(activeItem?.shot, activeItem?.assignment, data.overrides, plan)}
                  </div>
                  ${activeItem ? `
                    <button type="button" id="ls-render-shot-btn" class="btn btn-small btn-primary" data-ls-action="render-shot" data-shot-key="${esc(activeItem.key)}" data-busy="lip-sync" data-locked="${!canRenderShot}" ${canRenderShot ? "" : "disabled"}>
                      ${activeEntry.takes.length ? "New take" : "Render shot"}
                    </button>
                  ` : ""}
                </div>
              </div>

              ${activeItem ? `<div class="ls-takes-panel" id="ls-takes-panel">${takesPanelHtml(activeItem, activeEntry, activeStatus, data)}</div>` : ""}
            </div>

            <section class="card card-pad stack-12 ls-assembly-card">
              <div class="row align-center row-wrap">
                <h3>Full video assembly</h3>
                ${asm.changedCount > 0 ? `<span class="chip chip-warn">${plural(asm.changedCount, "shot")} changed since last export</span>` : data?.output ? '<span class="chip">Up to date</span>' : '<span class="chip chip-quiet">Not assembled yet</span>'}
                <span class="spacer"></span>
                ${data?.output ? `<button type="button" class="btn btn-small" data-ls-action="to-result">Watch in Result →</button>` : ""}
              </div>
              <div class="ls-readiness-bar" role="img" aria-label="${asm.reused} reused, ${asm.changedCount} updated, ${asm.unrendered} unrendered, ${asm.off} off-screen">
                ${asm.reused > 0 ? `<span class="ls-seg-reused" style="flex:${asm.reused}"></span>` : ""}
                ${asm.changedCount > 0 ? `<span class="ls-seg-updated" style="flex:${asm.changedCount}"></span>` : ""}
                ${asm.unrendered > 0 ? `<span class="ls-seg-unrendered" style="flex:${asm.unrendered}"></span>` : ""}
                ${asm.off > 0 ? `<span class="ls-seg-off" style="flex:${asm.off}"></span>` : ""}
              </div>
              <div class="row row-wrap small muted">
                <span><strong>${asm.reused}</strong> reused</span>
                <span><strong>${asm.changedCount}</strong> updated</span>
                <span><strong>${asm.unrendered}</strong> unrendered</span>
                <span><strong>${asm.off}</strong> off-screen</span>
              </div>
              <div class="row align-center row-wrap top-line">
                <span class="small muted">Fast splice · updates only changed shots and reuses every unchanged take without GPU re-rendering.</span>
                <span class="spacer"></span>
                <button type="button" class="btn btn-primary" data-ls-action="assemble" data-busy="lip-sync" data-locked="${!canAssemble}" ${canAssemble ? "" : "disabled"}>
                  ${assembleLabel}
                </button>
              </div>
            </section>

            <section class="card card-pad stack-12">
              <div class="row align-center row-wrap">
                <h3>Settings for next take</h3>
                <span class="spacer"></span>
                <span class="muted small" id="ls-engine-desc">${(ENGINE_OPTIONS.find(([id]) => id === lsState.engine) || ENGINE_OPTIONS[0])[2]}</span>
              </div>
              <div class="row align-center row-wrap">
                <span class="field-label">Engine</span>
                <div class="seg" role="group" aria-label="Lip-sync engine">
                  ${ENGINE_OPTIONS.map(([id, label]) => `
                    <button type="button" data-ls-engine="${id}" aria-pressed="${lsState.engine === id}">${esc(label)}</button>
                  `).join("")}
                </div>
              </div>
              <div class="ls-settings-row" id="ls-settings-row">
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
                <span class="muted small">Filters face boxes, face buttons, and newly rendered shots; manual picks and existing takes stay untouched.</span>
              </div>
              <div class="row align-center row-wrap top-line">
                <span class="small" data-ls-render-summary>
                  ${renderSummaryHtml(plan, confirmedCount, uncertainCount, offCount)}
                </span>
                <span class="spacer"></span>
                <button type="button" class="btn" data-ls-action="render" data-busy="lip-sync" data-locked="${!canRender}" ${canRender ? "" : "disabled"}>
                  Render all &amp; assemble
                </button>
              </div>
            </section>
          </section>

          <aside class="card ls-queue">
            <div class="ls-queue-head stack-10">
              <div class="row align-center row-wrap">
                <h2>Shots &amp; takes (${items.length})</h2>
                <span class="spacer"></span>
                <button type="button" class="btn btn-small" data-ls-action="select-unrendered">Select unrendered</button>
                <button type="button" id="ls-clear-checked-btn" class="btn btn-small" data-ls-action="clear-checked" ${checkedCount > 0 ? "" : "hidden"}>Clear</button>
              </div>
              <div class="seg" role="group" aria-label="Filter shots">
                <button type="button" data-ls-filter="all" aria-pressed="${lsState.filter === "all"}">All (${items.length})</button>
                <button type="button" data-ls-filter="pending" aria-pressed="${lsState.filter === "pending"}">Pending (${pendingCount})</button>
                <button type="button" data-ls-filter="review" aria-pressed="${lsState.filter === "review"}">Needs check (${uncertainCount})</button>
              </div>
              <div id="ls-queue-bulk-slot">
                ${queueBulkHtml(checkedCount, canRender, !!app.runs?.auto?.active || !!app.runs?.["lip-sync"]?.active || !!app.runs?.["lip-sync-detect"]?.active)}
              </div>
            </div>
            <div class="ls-queue-list">
              ${visible.map((it) => {
                const failedEntry = failedByKey.get(it.key);
                const badge = failedEntry ? {label: "Failed", cls: "chip chip-warn"} : takeStatusBadge(it, plan, data);
                const selected = activeItem && it.key === activeItem.key;
                const checked = lsState.checkedKeys.has(it.key);
                const tk = activeTakeFor(data, it.key);
                const targetDesc = failedEntry
                  ? `Last render failed: ${failedEntry.error}`
                  : tk
                    ? `Take ${tk.number} · ${formatTakeMeta(tk, it.shot)}`
                    : shotTargetDescription(it, plan);
                return `
                  <div class="ls-shot-row ${selected ? "selected" : ""}" data-ls-row-key="${esc(it.key)}">
                    <input type="checkbox" class="ls-shot-check" data-ls-check="${esc(it.key)}" ${checked ? "checked" : ""} aria-label="Select Shot ${it.shot.index + 1}">
                    <button type="button" class="ls-shot-main" data-ls-shot="${it.shot.index}" data-ls-speaker="${esc(it.speaker)}" data-ls-time="${it.seekTime}">
                      <div class="row align-center">
                        <strong>Shot ${it.shot.index + 1}</strong>
                        <span class="mono muted">${formatClock(it.shot.start)}–${formatClock(it.shot.end)}</span>
                        <span class="spacer"></span>
                        <span class="${badge.cls}" data-ls-row-badge="${esc(it.key)}">${badge.label}</span>
                      </div>
                      <div class="small ellipsis"><strong>${esc(speakerLabel(it.speaker))}</strong>${it.line ? ` · ${esc(it.line)}` : ""}</div>
                      <div class="small muted ellipsis" data-ls-row-desc="${esc(it.key)}">${esc(targetDesc)}</div>
                    </button>
                  </div>`;
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
  const index = getPlanIndex(plan);
  const shot = findShotAtTime(plan, index, t);
  const turn = findTurnAtTime(plan, index, t);
  if (subtitle) {
    if (turn) {
      subtitle.hidden = false;
      subtitle.innerHTML = `<span class="ls-sub-spk">${esc(speakerLabel(turn.speaker))}</span> <span>${esc(turn.text)}</span>`;
    } else {
      subtitle.hidden = true;
    }
  }
  const aMap = index.assignments;
  const selected = shot?.index === lsState.selectedShot ? lsState.selectedSpeaker : null;
  const activeAssign = shot && (aMap.get(`${shot.index}:${selected}`) || aMap.get(`${shot.index}:${turn?.speaker}`)
    || index.firstAssignmentByShot.get(shot.index));
  const activeSpeaker = activeAssign?.speaker;
  const key = activeSpeaker ? `${shot.index}:${activeSpeaker}` : "";
  const buttons = document.getElementById("ls-quick-buttons");
  const rangeKey = `${lsState.minFaceSize}-${lsState.maxFaceSize}`;
  if (buttons && (buttons.dataset.assignKey !== key || buttons.dataset.range !== rangeKey)) {
    buttons.dataset.assignKey = key;
    buttons.dataset.range = rangeKey;
    buttons.innerHTML = assignmentButtons(shot, activeAssign, lsState.data.overrides, plan);
    const infoEl = document.getElementById("ls-active-info");
    if (infoEl) infoEl.innerHTML = assignmentLabel(shot, activeSpeaker);
  }
  if (!shot) {
    if (overlay.childElementCount) overlay.textContent = "";
    return;
  }
  const activeTrackId = assignedTrackId(activeAssign);
  const activeWindow = key
    ? (index.windowsByKey.get(key) || []).find((win) => t >= win.start && t < win.end) || null
    : null;

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
    if (!isSpeaking && !isTrackInSizeRange(track, index, h)) return;
    const speakingWindow = isSpeaking ? activeWindow : null;
    const trackSize = (speakingWindow && windowFaceSize(speakingWindow)) ?? trackMedianHeight(track, index) ?? h;
    const pct = formatFacePct(trackSize);
    const isManual = speakingWindow ? isWindowManual(speakingWindow, plan) : activeAssign?.status === "manual";
    const inRange = speakingWindow ? sizeInRange(windowFaceSize(speakingWindow)) : sizeInRange(trackSize);
    const sizeIgnored = isSpeaking && !isManual && !inRange;
    const manualKept = isSpeaking && isManual && !inRange;
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

function captureQueueScroll() {
  const list = document.querySelector(".ls-queue-list");
  if (list) lsState.queueScrollTop = list.scrollTop;
}

function rerenderLipSync(app, {preserveVideo = false} = {}) {
  captureQueueScroll();
  const video = preserveVideo ? document.getElementById("ls-video") : null;
  const savedTime = video && video.dataset.stageMode === "align" ? video.currentTime : null;
  const wasPaused = video ? video.paused : true;
  app.render();
  if (savedTime !== null) {
    const nextVideo = document.getElementById("ls-video");
    if (nextVideo && nextVideo.dataset.stageMode === "align") {
      const restore = () => {
        nextVideo.currentTime = savedTime;
        if (!wasPaused) nextVideo.play().catch(() => {});
        drawOverlay(app);
      };
      if (nextVideo.readyState >= 1) restore();
      else nextVideo.addEventListener("loadedmetadata", restore, {once: true});
    }
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
  const queueList = document.querySelector(".ls-queue-list");
  if (queueList) {
    queueList.scrollTop = lsState.queueScrollTop;
    queueList.addEventListener("scroll", () => {
      lsState.queueScrollTop = queueList.scrollTop;
    }, {passive: true});
  }
  const video = document.getElementById("ls-video");
  if (!video || !lsState.data?.plan) return;
  if (video.dataset.stageMode === "preview") {
    const clipStart = Number(video.dataset.clipStart) || 0;
    const clipEnd = Number(video.dataset.clipEnd) || 0;
    const isFullSource = !video.hasAttribute("loop") && clipEnd > clipStart;
    if (isFullSource) {
      const seekToStart = () => {
        if (video.currentTime < clipStart || video.currentTime >= clipEnd) {
          video.currentTime = clipStart;
        }
        if (lsState.autoplayPreview) {
          lsState.autoplayPreview = false;
          video.play().catch(() => {});
        }
      };
      if (video.readyState >= 1) seekToStart();
      else video.addEventListener("loadedmetadata", seekToStart, {once: true});
      video.addEventListener("timeupdate", () => {
        if (video.currentTime >= clipEnd) {
          if (lsState.loopShot) {
            video.currentTime = clipStart;
            video.play().catch(() => {});
          } else {
            video.pause();
            video.currentTime = clipStart;
          }
        }
      });
    } else if (lsState.autoplayPreview) {
      lsState.autoplayPreview = false;
      const startPlay = () => { video.play().catch(() => {}); };
      if (video.readyState >= 1) startPlay();
      else video.addEventListener("loadedmetadata", startPlay, {once: true});
    }
    return;
  }
  const restoreAlignTime = () => {
    const targetTime = lsState.videoTime !== null ? lsState.videoTime : (Number(video.dataset.seekTime) || 0);
    if (targetTime > 0 && Math.abs(video.currentTime - targetTime) > 0.05) {
      video.currentTime = targetTime;
    }
    drawOverlay(app);
  };
  if (video.readyState >= 1) restoreAlignTime();
  else video.addEventListener("loadedmetadata", restoreAlignTime, {once: true});
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
  const rememberTime = () => {
    if (video.readyState >= 1) lsState.videoTime = video.currentTime;
  };
  video.addEventListener("play", () => { if (lsState.rafId === null) lsState.rafId = requestAnimationFrame(tick); });
  video.addEventListener("pause", () => { rememberTime(); drawOverlay(app); });
  video.addEventListener("seeked", () => { rememberTime(); drawOverlay(app); });
  video.addEventListener("timeupdate", () => { rememberTime(); if (lsState.rafId === null) drawOverlay(app); });
  video.addEventListener("resize", () => drawOverlay(app));
}

async function selectTake(app, shotKey, takeId, {remove = false} = {}) {
  const target = reviewTarget(app);
  const targetKey = JSON.stringify(target);
  if (!target || lsState.targetKey !== targetKey || lsState.loading || lsState.saving) return;
  lsState.saving = true;
  try {
    const resp = await api.post("lip-sync-take-select", {
      ...target,
      shot_key: shotKey,
      take_id: takeId,
      delete: remove,
    });
    if (JSON.stringify(reviewTarget(app)) !== targetKey) return;
    lsState.data = resp;
    if (!remove) {
      lsState.stageMode = "preview";
      lsState.autoplayPreview = true;
    }
    rerenderLipSync(app);
    toast(remove ? "Take removed." : takeId === "original" ? "Switched shot to original picture." : "Active take updated.");
  } catch (error) {
    if (JSON.stringify(reviewTarget(app)) !== targetKey) return;
    toast(error.message, {error: true});
  } finally {
    if (lsState.targetKey === targetKey) lsState.saving = false;
  }
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
    rerenderLipSync(app, {preserveVideo: true});
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
      const badge = takeStatusBadge(it, plan, data);
      badgeEl.className = badge.cls;
      badgeEl.textContent = badge.label;
    }
    const descEl = document.querySelector(`[data-ls-row-desc="${CSS.escape(it.key)}"]`);
    if (descEl) {
      const tk = activeTakeFor(data, it.key);
      descEl.textContent = tk ? `Take ${tk.number} · ${formatTakeMeta(tk, it.shot)}` : shotTargetDescription(it, plan);
    }
  }
  const buttonsEl = document.getElementById("ls-quick-buttons");
  if (buttonsEl) {
    const assignKey = buttonsEl.dataset.assignKey || "";
    const sep = assignKey.indexOf(":");
    const idx = getPlanIndex(plan);
    const shot = sep > 0 ? idx?.shotsByIndex.get(Number(assignKey.slice(0, sep))) : null;
    const assign = assignKey ? idx?.assignments.get(assignKey) : null;
    buttonsEl.dataset.range = `${lsState.minFaceSize}-${lsState.maxFaceSize}`;
    buttonsEl.innerHTML = assignmentButtons(shot, assign, data.overrides, plan);
  }
  drawOverlay(app);
}

function patchBulkSelection(app) {
  const data = lsState.targetKey === JSON.stringify(reviewTarget(app)) ? lsState.data : null;
  const plan = data?.plan;
  const ls = app.session?.result?.lip_sync;
  const reason = ls && plan ? renderLockedReason(ls, app, plan) : "unavailable";
  const canRender = !!(plan?.revision && !reason && !lsState.loading);
  const busy = !!app.runs?.auto?.active || !!app.runs?.["lip-sync"]?.active || !!app.runs?.["lip-sync-detect"]?.active;
  const count = lsState.checkedKeys.size;
  const slot = document.getElementById("ls-queue-bulk-slot");
  if (slot) slot.innerHTML = queueBulkHtml(count, canRender, busy);
  const clearBtn = document.getElementById("ls-clear-checked-btn");
  if (clearBtn) clearBtn.hidden = count === 0;
  for (const chk of document.querySelectorAll("[data-ls-check]")) {
    const shouldCheck = lsState.checkedKeys.has(chk.dataset.lsCheck);
    if (chk.checked !== shouldCheck) chk.checked = shouldCheck;
  }
}

function patchEngineUI() {
  for (const btn of document.querySelectorAll("[data-ls-engine]")) {
    btn.setAttribute("aria-pressed", String(btn.dataset.lsEngine === lsState.engine));
  }
  const descEl = document.getElementById("ls-engine-desc");
  if (descEl) {
    descEl.innerHTML = (ENGINE_OPTIONS.find(([id]) => id === lsState.engine) || ENGINE_OPTIONS[0])[2];
  }
  const rowEl = document.getElementById("ls-settings-row");
  if (rowEl) rowEl.innerHTML = engineSettingsHtml();
}

function patchLoopUI() {
  const video = document.getElementById("ls-video");
  if (video && video.dataset.stageMode === "preview") {
    video.loop = lsState.loopShot;
  }
  const btn = document.querySelector('[data-ls-action="toggle-loop"]');
  if (btn) {
    btn.classList.toggle("btn-primary", lsState.loopShot);
    btn.setAttribute("aria-pressed", String(lsState.loopShot));
  }
}

function selectShotInPlace(app, shotIndex, speaker, seekTime, {scroll = false} = {}) {
  captureQueueScroll();
  lsState.selectedShot = shotIndex;
  lsState.selectedSpeaker = speaker || null;
  lsState.videoTime = seekTime;
  const data = lsState.targetKey === JSON.stringify(reviewTarget(app)) ? lsState.data : null;
  const plan = data?.plan;
  const key = `${shotIndex}:${speaker || ""}`;
  const entry = shotTakeEntry(data, key);
  if (lsState.stageMode === "preview" && !entry.takes.length) {
    lsState.stageMode = "align";
  }

  const video = document.getElementById("ls-video");
  const canPatchInPlace = !!(plan && video && lsState.stageMode === "align" && video.dataset.stageMode === "align");
  if (!canPatchInPlace) {
    app.render();
    if (scroll) {
      const row = document.querySelector(`.ls-shot-row[data-ls-row-key="${CSS.escape(key)}"]`);
      row?.scrollIntoView({block: "nearest"});
      captureQueueScroll();
    }
    return;
  }

  const items = shotItems(plan);
  const activeItem = items.find((it) => it.shot.index === shotIndex && (!speaker || it.speaker === speaker)) || null;
  const activeTake = activeItem ? activeTakeFor(data, activeItem.key) : null;
  const activeStatus = activeItem ? shotTakeStatus(activeItem, data) : "unrendered";
  const stageTop = document.getElementById("ls-stage-top");
  if (stageTop) stageTop.innerHTML = stageTopHtml(activeItem, entry, activeTake, false, false);
  const infoEl = document.getElementById("ls-active-info");
  if (infoEl) infoEl.innerHTML = assignmentLabel(activeItem?.shot, activeItem?.speaker);
  const buttonsEl = document.getElementById("ls-quick-buttons");
  if (buttonsEl) {
    buttonsEl.dataset.assignKey = activeItem?.key || "";
    buttonsEl.dataset.range = `${lsState.minFaceSize}-${lsState.maxFaceSize}`;
    buttonsEl.innerHTML = assignmentButtons(activeItem?.shot, activeItem?.assignment, data.overrides, plan);
  }
  const renderShotBtn = document.getElementById("ls-render-shot-btn");
  if (renderShotBtn && activeItem) {
    const ls = app.session?.result?.lip_sync;
    const canRenderShot = !!(
      plan?.revision && app.boot?.lip_sync && ls?.ready && !lsState.loading
      && activeItem.assignment.status !== "off_screen"
      && itemHasWindows(activeItem, plan)
    );
    const busy = !!app.runs?.auto?.active || !!app.runs?.["lip-sync"]?.active || !!app.runs?.["lip-sync-detect"]?.active;
    renderShotBtn.dataset.shotKey = activeItem.key;
    renderShotBtn.dataset.locked = String(!canRenderShot);
    renderShotBtn.disabled = busy || !canRenderShot;
    renderShotBtn.textContent = entry.takes.length ? "New take" : "Render shot";
  }
  const takesEl = document.getElementById("ls-takes-panel");
  if (takesEl) takesEl.innerHTML = takesPanelHtml(activeItem, entry, activeStatus, data);
  for (const row of document.querySelectorAll(".ls-shot-row")) {
    const isSelected = row.dataset.lsRowKey === key;
    row.classList.toggle("selected", isSelected);
    if (isSelected && scroll) {
      row.scrollIntoView({block: "nearest"});
      captureQueueScroll();
    }
  }
  video.dataset.seekTime = String(seekTime);
  const seek = () => {
    video.currentTime = seekTime;
    drawOverlay(app);
  };
  if (video.readyState >= 1) seek();
  else video.addEventListener("loadedmetadata", seek, {once: true});
}

function renderShotsPayload(app, shotKeys, {fresh = true} = {}) {
  const target = reviewTarget(app);
  const revision = lsState.data?.plan?.revision;
  if (lsState.targetKey !== JSON.stringify(target) || !revision || lsState.saving || lsState.loading) return null;
  if (!shotKeys.length) return null;
  return {
    ...target,
    plan_revision: revision,
    shot_keys: shotKeys,
    fresh,
    engine: lsState.engine,
    strength: lsState.strength,
    feather: lsState.feather,
    bbox_shift: lsState.bboxShift,
    inference_steps: lsState.inferenceSteps,
    guidance_scale: lsState.guidanceScale,
  };
}

export function bindLipSync(app) {
  const screen = document.getElementById("screen");
  screen.addEventListener("change", (event) => {
    if (app.screen !== "lipsync" || !app.session) return;
    const chk = event.target.closest("[data-ls-check]");
    if (chk) {
      const key = chk.dataset.lsCheck;
      if (chk.checked) lsState.checkedKeys.add(key);
      else lsState.checkedKeys.delete(key);
      patchBulkSelection(app);
    }
  });
  screen.addEventListener("click", async (event) => {
    if (app.screen !== "lipsync" || !app.session) return;
    const stageBtn = event.target.closest("[data-ls-stage]");
    if (stageBtn) {
      const nextMode = stageBtn.dataset.lsStage;
      if (nextMode && nextMode !== lsState.stageMode) {
        lsState.stageMode = nextMode;
        if (nextMode === "preview") lsState.autoplayPreview = true;
        rerenderLipSync(app);
      }
      return;
    }
    const pickTakeBtn = event.target.closest("[data-ls-pick-take]");
    if (pickTakeBtn) {
      await selectTake(app, pickTakeBtn.dataset.shotKey, pickTakeBtn.dataset.lsPickTake, {remove: false});
      return;
    }
    const delTakeBtn = event.target.closest("[data-ls-delete-take]");
    if (delTakeBtn) {
      await selectTake(app, delTakeBtn.dataset.shotKey, delTakeBtn.dataset.lsDeleteTake, {remove: true});
      return;
    }
    const engineBtn = event.target.closest("[data-ls-engine]");
    if (engineBtn) {
      const nextEngine = engineBtn.dataset.lsEngine;
      if (nextEngine && nextEngine !== lsState.engine) {
        lsState.engine = nextEngine;
        saveLsSettings();
        patchEngineUI();
      }
      return;
    }
    const preset = event.target.closest("[data-ls-size-preset]");
    if (preset) {
      const [lo, hi] = preset.dataset.lsSizePreset.split(",").map(Number);
      lsState.minFaceSize = lo;
      lsState.maxFaceSize = hi;
      saveLsSettings();
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
      selectShotInPlace(app, Number(shotRow.dataset.lsShot), shotRow.dataset.lsSpeaker || null, Number(shotRow.dataset.lsTime) || 0);
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
    } else if (action === "toggle-loop") {
      lsState.loopShot = !lsState.loopShot;
      patchLoopUI();
    } else if (action === "select-unrendered") {
      const plan = lsState.data?.plan;
      const items = shotItems(plan);
      const next = new Set();
      for (const it of items) {
        if (!itemHasWindows(it, plan)) continue;
        const st = shotTakeStatus(it, lsState.data);
        if (st === "outdated" || (st === "unrendered" && !isItemSizeFiltered(it, plan))) next.add(it.key);
      }
      lsState.checkedKeys = next;
      patchBulkSelection(app);
      toast(next.size ? `Selected ${plural(next.size, "shot")}.` : "All on-screen shots already have up-to-date takes.");
    } else if (action === "clear-checked") {
      lsState.checkedKeys.clear();
      patchBulkSelection(app);
    } else if (action === "render-shot") {
      const shotKey = btn.dataset.shotKey;
      const payload = shotKey ? renderShotsPayload(app, [shotKey], {fresh: true}) : null;
      if (payload) {
        lsState.lastRenderedKeys = payload.shot_keys;
        app.startRun("lip-sync-shots", payload);
      }
    } else if (action === "render-selected") {
      const payload = renderShotsPayload(app, Array.from(lsState.checkedKeys), {fresh: true});
      if (payload) {
        lsState.lastRenderedKeys = payload.shot_keys;
        app.startRun("lip-sync-shots", payload);
      }
    } else if (action === "assemble") {
      const target = reviewTarget(app);
      if (lsState.targetKey !== JSON.stringify(target) || lsState.saving || lsState.loading) return;
      app.startRun("lip-sync-assemble", {...target});
    } else if (action === "detect") {
      if (!lsState.saving && app.boot?.lip_sync && app.session?.result?.lip_sync?.ready) {
        app.startRun("lip-sync-detect", {...reviewTarget(app), api_key: app.apiKey, force: !!lsState.data?.plan});
      }
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
      selectShotInPlace(app, next.shot.index, next.speaker, next.seekTime, {scroll: true});
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
      saveLsSettings();
      syncFaceSizeUI(app);
      return;
    }
    const el = event.target.closest("[data-ls-setting]");
    if (!el) return;
    const key = el.dataset.lsSetting;
    lsState[key] = Number(el.value);
    saveLsSettings();
    const out = document.querySelector(`[data-ls-out="${key}"]`);
    if (out) out.textContent = formatSettingOut(key, lsState[key]);
  });
}
