import {
  cameraFrame,
  cueElapsed,
  effectCues,
  lastAtOrBefore,
  lightningCueElapsed,
  paintEffects,
  resolveFocusDepthMeta,
} from "./effects.js";

const EMPTY = Object.freeze([]);
const startOf = (shot) => shot.start;

export function stageLayers(onload) {
  const back = document.createElement("img"), front = document.createElement("img");
  back.id = "still-back";
  front.id = "still-front";
  for (const image of [back, front]) {
    image.className = "still";
    image.alt = "";
    if (onload) image.addEventListener("load", onload);
  }
  const overlay = document.createElement("canvas");
  overlay.id = overlay.className = "still-effects";
  overlay.setAttribute("aria-hidden", "true");
  return {back, front, overlay};
}

export function stageTimeline(shots, sounds) {
  const cues = effectCues(shots, sounds);
  return shots.map((shot, index) => ({...shot, cues: cues[index]}));
}

function paintCamera(image, shot, effects, time) {
  const elapsed = Math.max(0, Math.min(time, shot.end) - shot.start);
  const progress = elapsed / Math.max(0.001, shot.end - shot.start);
  const frame = cameraFrame(effects, progress, elapsed, cueElapsed(shot.cues.shake, time));
  if (image.style.transform !== frame.css) image.style.transform = frame.css;
  if (image.style.filter !== frame.filter && (image.style.filter || frame.filter !== "none")) image.style.filter = frame.filter;
  return frame;
}

function showStill(image, url) {
  if (image.dataset.src === url) return false;
  image.dataset.src = url;
  image.src = url;
  return true;
}

export function stageFrame(shots, time) {
  const index = Math.max(0, lastAtOrBefore(shots, time, startOf));
  const shot = shots[index];
  const fading = index > 0 && shot.fade > 0 && time < shot.start + shot.fade;
  const background = fading ? shots[index - 1] : shot;
  return {index, shot, background, fading, time};
}

export function showStageStills({back, front}, {shot, background, fading}) {
  const changed = [];
  if (showStill(back, background.url)) changed.push(back);
  if (fading && showStill(front, shot.url)) changed.push(front);
  return changed;
}

export function paintStage({back, front, overlay}, {shot, background, fading, time}, enabled, showBackend) {
  const elapsed = time - shot.start;
  const fadeMix = fading ? elapsed / shot.fade : 1;
  const opacity = fading ? String(fadeMix) : "0";
  if (front.style.opacity !== opacity) front.style.opacity = opacity;
  const effects = enabled && elapsed >= 0 ? shot.effects : EMPTY;
  const backEffects = enabled && fading ? background.effects : EMPTY;
  const backCam = paintCamera(back, background, fading ? backEffects : effects, time);
  const cam = fading ? paintCamera(front, shot, effects, time) : backCam;
  if (overlay) {
    const backElapsed = fading ? time - background.start : 0;
    const depthMeta = shot.depth_meta
      ? resolveFocusDepthMeta(shot.depth_meta, enabled && elapsed >= 0 && shot.focus_cues, elapsed)
      : null;
    const backDepthMeta = fading && background.depth_meta
      ? resolveFocusDepthMeta(background.depth_meta, enabled && background.focus_cues, backElapsed)
      : null;
    paintEffects(overlay, effects, elapsed, (backend) => {
      showBackend?.(backend, Boolean(effects.length || backEffects.length || depthMeta?.focus_z || backDepthMeta?.focus_z));
    }, {
      shotIndex: shot.index, cam, stillBack: back, stillFront: front, fading, fadeMix,
      cueElapsed: cueElapsed(shot.cues.flash, time),
      lightningCueElapsed: lightningCueElapsed(shot.cues.lightning, time),
      backEffects, backElapsed,
      backCueElapsed: cueElapsed(background.cues.flash, time),
      backLightningCueElapsed: fading ? lightningCueElapsed(background.cues.lightning, time) : -1,
      backShotIndex: background.index, backCam,
      layers: enabled && elapsed >= 0 ? shot.effect_layers : null,
      backLayers: enabled && fading ? background.effect_layers : null,
      depthUrl: shot.depth_url || null,
      depthMeta,
      backDepthUrl: fading ? (background.depth_url || null) : null,
      backDepthMeta,
    });
  }
}
