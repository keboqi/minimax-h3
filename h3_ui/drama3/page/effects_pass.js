import {preloadEffectDepths, preloadEffectMasks} from "./effects.js";
import {EFFECT_CATEGORIES, openEffectPresetPopup} from "./effects_picker.js";
import {paintStage, showStageStills, stageFrame, stageLayers, stageTimeline} from "./stage.js";

const FULL_FRAME = "full_frame";
const MAX_TOTAL_REGIONS = 24;
const MODE_LABELS = {full_frame: "Full frame", inside: "Inside", outside: "Outside", behind: "Behind"};
const SEG_COLORS = 12;
const regionColor = (index) => `var(--seg-${index % SEG_COLORS})`;
const regionColorMap = (shot) => new Map((shot?.regions || []).map((region, index) => [region.id, regionColor(index)]));
const maskUrl = (url) => `url("${String(url).replace(/["\\]/g, "\\$&")}")`;
const layerRegions = (layer) => layer.region_ids?.length ? layer.region_ids : (layer.region_id ? [layer.region_id] : []);
const sameRegions = (a, b) => a.length === b.length && a.every((id, index) => id === b[index]);

export function layersToSets(layers, catalog = {}) {
  const sets = [];
  for (const layer of layers || []) {
    const mode = layer.mode || FULL_FRAME;
    const regionIds = mode === FULL_FRAME ? [] : layerRegions(layer);
    const isCamera = catalog[layer.preset]?.kind === "camera";
    const prev = sets[sets.length - 1];
    if (prev && prev.mode === mode && sameRegions(prev.region_ids, regionIds)) {
      if (isCamera) {
        if (mode === FULL_FRAME && !prev.presets.some((preset) => catalog[preset]?.kind === "camera")) {
          prev.presets.push(layer.preset);
          continue;
        }
      } else {
        const prevOnlyCamera = prev.presets.every((preset) => catalog[preset]?.kind === "camera");
        if (prevOnlyCamera) {
          prev.presets.push(layer.preset);
          prev.intensity = layer.intensity;
          if (mode !== FULL_FRAME) prev.feather = layer.feather;
          continue;
        }
        if ((prev.intensity ?? 1) === (layer.intensity ?? 1)
            && (mode === FULL_FRAME || (prev.feather ?? 1) === (layer.feather ?? 1))) {
          prev.presets.push(layer.preset);
          continue;
        }
      }
    }
    sets.push({
      mode,
      region_ids: [...regionIds],
      presets: [layer.preset],
      intensity: isCamera ? undefined : layer.intensity,
      feather: isCamera || mode === FULL_FRAME ? undefined : layer.feather,
    });
  }
  return sets;
}

export function setsToLayers(sets, catalog = {}) {
  const layers = [];
  for (const set of sets || []) {
    const mode = set.mode || FULL_FRAME;
    const regionIds = mode === FULL_FRAME ? [] : (set.region_ids || []);
    const effectiveMode = mode !== FULL_FRAME && regionIds.length ? mode : FULL_FRAME;
    for (const preset of set.presets || []) {
      if (catalog[preset]?.kind === "camera") {
        layers.push({preset, mode: FULL_FRAME});
        continue;
      }
      const layer = {preset, mode: effectiveMode};
      if (effectiveMode !== FULL_FRAME) {
        if (regionIds.length === 1) layer.region_id = regionIds[0];
        else if (regionIds.length > 1) layer.region_ids = [...regionIds];
      }
      if (set.intensity !== undefined) layer.intensity = set.intensity;
      if (effectiveMode !== FULL_FRAME && set.feather !== undefined) layer.feather = set.feather;
      layers.push(layer);
    }
  }
  return layers;
}

function pruneSetFields(set, catalog) {
  if (set.mode === FULL_FRAME) {
    set.region_ids = [];
    set.feather = undefined;
  }
  if (!set.presets.some((preset) => catalog[preset]?.kind !== "camera")) {
    set.intensity = undefined;
  }
}

function setAccentColor(set, colors, catalog) {
  if (set.mode !== FULL_FRAME) {
    const setColors = set.region_ids.map((id) => colors.get(id)).filter(Boolean);
    return setColors.length === 1 ? setColors[0] : "var(--accent)";
  }
  const kindSet = new Set(set.presets.map((id) => catalog[id]?.kind).filter(Boolean));
  if (kindSet.size !== 1) return "var(--accent)";
  const [kind] = kindSet;
  if (kind === "camera") return "var(--new)";
  if (kind === "lighting") return "var(--warn)";
  if (kind === "atmosphere") return "var(--tok-effect)";
  return "var(--accent)";
}

export function createEffectsPass({state, $, h, call, copy, writer, interrupt, busy}) {
  let project, selected, mode, draftSets, pasted = "", includeManual = false;
  let before = false, refine, frameTime = 0, showSegments = false, playFrame = null;
  const dialog = () => $("effect-pass-dialog");
  const presetDialog = () => $("effect-preset-dialog");
  const panel = () => state.project.video?.effect_enhancement;
  const rowOf = (shot) => state.project.video.plan.rows[shot.index];
  const button = (label, onclick, disabled = false, extra = "") =>
    h("button", {type: "button", class: `btn btn-small ${extra}`.trim(), onclick, disabled}, label);
  const api = (action, body = {}) => call(`/project/video/effects/${action}`, body);
  const selectedWriter = () => writer();
  const totalEffects = () => (draftSets || []).reduce((sum, set) => sum + set.presets.length, 0);

  function stopPlayback() {
    if (playFrame !== null) {
      cancelAnimationFrame(playFrame);
      const playBtn = dialog()?.querySelector(".effect-timeline button");
      if (playBtn) playBtn.textContent = "Play preview";
    }
    playFrame = null;
  }

  function closePresetDialog() {
    const popup = presetDialog();
    if (popup?.open) popup.close();
  }

  function setFocusRegions(regionIds) {
    const stage = dialog()?.querySelector(".pass-preview");
    if (!stage) return;
    const list = Array.isArray(regionIds) ? regionIds : (regionIds ? [regionIds] : []);
    const active = new Set(list.length ? list : (refine?.region ? [refine.region] : []));
    stage.classList.toggle("has-focus-region", active.size > 0);
    stage.querySelectorAll(".pass-seg-overlay").forEach((overlay) => {
      overlay.classList.toggle("is-focused", active.has(overlay.dataset.regionId));
    });
  }

  function open(shot = null) {
    stopPlayback();
    closePresetDialog();
    if (project !== state.project.name) {
      project = state.project.name;
      pasted = "";
      includeManual = false;
    }
    selected = shot?.shot_id;
    mode = shot ? "edit" : "batch";
    draftSets = shot ? layersToSets(shot.effects ?? shot.default, state.boot.video.enhancement_presets) : null;
    before = false;
    refine = null;
    frameTime = 0;
    showSegments = false;
    if (!dialog().open) dialog().showModal();
    render(true);
  }

  function withMasks(shot, effects) {
    const urls = new Map(shot.regions.map((region) => [region.id, region.url]));
    const depths = new Map(shot.regions.filter((region) => region.depth_m != null).map((region) => [region.id, region.depth_m]));
    return effects.map((effect) => {
      const ids = layerRegions(effect);
      const depthVals = ids.map((id) => depths.get(id)).filter((value) => value != null);
      const depthExtra = depthVals.length ? {region_depth_m: Math.min(...depthVals)} : {};
      if (ids.length > 1) return {...effect, ...depthExtra, mask_urls: ids.map((id) => urls.get(id)).filter(Boolean)};
      if (ids.length === 1) return {...effect, ...depthExtra, mask_url: urls.get(ids[0])};
      return effect;
    });
  }

  function formatLayers(shot, effects, catalog) {
    const regions = new Map((shot.regions || []).map((r) => [r.id, r.label]));
    const colors = regionColorMap(shot);
    if (!effects?.length) return [h("span", {class: "muted small"}, "No effects")];
    return effects.map((layer) => {
      const label = catalog[layer.preset]?.label || layer.preset;
      const ids = layerRegions(layer);
      const names = ids.map((id) => regions.get(id)).filter(Boolean);
      const modeText = layer.mode && layer.mode !== FULL_FRAME
        ? ` · ${MODE_LABELS[layer.mode] || layer.mode}${names.length ? ` ${names.join(" + ")}` : ""}`
        : "";
      const dots = layer.mode !== FULL_FRAME
        ? ids.map((id) => colors.get(id)).filter(Boolean)
          .map((color) => h("span", {class: "pass-seg-dot", style: {"--seg-color": color}, "aria-hidden": "true"}))
        : [];
      return h("span", {class: "chip"}, ...dots, `${label}${modeText}`);
    });
  }

  function preview(shot, effects) {
    stopPlayback();
    const row = rowOf(shot);
    const orientation = state.project.video.orientation;
    const {back, front, overlay} = stageLayers();
    [back, front, overlay].forEach((element) => element.removeAttribute("id"));
    const activeRegion = refine?.region || "";
    const segOverlays = shot.regions?.length
      ? shot.regions.map((region, index) => h("div", {
        class: `pass-seg-overlay${region.id === activeRegion ? " is-focused" : ""}`,
        "data-region-id": region.id,
        "aria-hidden": "true",
        style: {"--seg-color": regionColor(index), "--seg-mask": maskUrl(region.url)},
      }))
      : [];
    const stage = h("div", {
      class: `video-stage pass-preview ${orientation === "portrait" ? "portrait" : ""}${activeRegion ? " has-focus-region" : ""}`,
      "data-show-segments": String(showSegments),
    }, back, front, overlay, ...segOverlays);
    const duration = Math.max(0.05, row.end - row.start);
    const sounds = (state.project.result?.sounds || [])
      .filter((sound) => sound.start >= row.start && sound.start < row.end)
      .map((sound) => ({...sound, start: sound.start - row.start}));
    const stillShot = (state.project.video?.stills?.shots || []).find((item) => item.index === row.index || item.shot_id === shot.shot_id);
    const [view] = stageTimeline([{start: 0, end: duration, index: row.index, fade: 0, url: row.image,
      effects: effects.map(({preset}) => preset), effect_layers: effects,
      depth_url: stillShot?.depth_url ?? null,
      depth_meta: stillShot?.depth_meta ?? null}], sounds);
    const layers = {back, front, overlay};
    const formatTime = (t) => `${t.toFixed(1)} s / ${duration.toFixed(1)} s`;
    const setTime = (t) => {
      frameTime = Math.min(duration, Math.max(0, t));
      slider.value = String(frameTime);
      timeReadout.textContent = formatTime(frameTime);
      paint();
    };
    const timeReadout = h("span", {class: "mono small muted effect-time"}, formatTime(Math.min(frameTime, duration)));
    const slider = h("input", {type: "range", min: 0, max: duration, step: 0.025, value: Math.min(frameTime, duration),
      "aria-label": "Preview time", oninput: (event) => {
        stopPlayback();
        setTime(Number(event.target.value));
      }});
    const playBtn = h("button", {type: "button", class: "btn btn-small", onclick: () => {
      if (playFrame !== null) {
        stopPlayback();
        return;
      }
      playBtn.textContent = "Pause";
      const startOffset = frameTime >= duration ? 0 : frameTime;
      const started = performance.now() - startOffset * 1000;
      const tick = (now) => {
        if (!dialog().open) { stopPlayback(); return; }
        setTime(((now - started) / 1000) % duration);
        playFrame = requestAnimationFrame(tick);
      };
      playFrame = requestAnimationFrame(tick);
    }}, "Play preview");
    const paint = () => paintStage(layers, stageFrame([view], Math.min(frameTime, duration)), true);
    showStageStills(layers, stageFrame([view], 0));
    Promise.all([back.decode(), preloadEffectMasks([view]), preloadEffectDepths([view])]).then(paint, paint);
    return h("div", {class: "pass-preview-wrap"}, stage,
      h("div", {class: "effect-timeline"}, playBtn, slider, timeReadout));
  }

  function batch() {
    const {shots, proposal} = panel();
    const catalog = state.boot.video.enhancement_presets || {};
    const segmented = shots.filter((shot) => shot.segmented).length;
    const eligible = shots.some((shot) => shot.regions.length && (!shot.manual || includeManual));
    const locked = busy();
    const items = proposal?.decisions || [];
    const changed = items.filter((item) => item.action === "replace");
    const kept = items.filter((item) => item.action === "keep").length;
    const skipped = shots.length - changed.length - kept;
    const target = changed.find((item) => item.shot_id === selected) || changed[0];
    const previewShot = target && shots.find((shot) => shot.shot_id === target.shot_id);
    const model = selectedWriter();
    const modelName = model.prompt_writer_backend === "OpenAI-compatible" ? model.openai_model : model.gemini_prompt_model;
    return [
      h("p", {class: "muted small"}, `${segmented} of ${shots.length} drawn shots segmented.`),
      h("div", {class: "pass-actions"},
        button(segmented ? "Segment missing shots" : "Segment shots", () => api("segment"),
          locked || segmented === shots.length || !state.boot.gpu),
        button(modelName ? `Generate with ${modelName}` : "Generate with selected LLM",
          () => api("generate", {include_manual: includeManual, writer: selectedWriter()}), locked || !eligible),
        button("Copy prompt", () => copy("prompt", {include_manual: includeManual}), locked || !eligible)),
      h("label", {class: "check"}, h("input", {type: "checkbox", checked: includeManual, disabled: locked,
        onchange: (event) => { includeManual = event.target.checked; render(true); }}), "Include manually edited shots"),
      h("label", {class: "field"}, h("span", {class: "label"}, "Pasted effects reply"),
        h("textarea", {rows: 3, value: pasted, disabled: locked || !eligible, oninput: (event) => {
          pasted = event.target.value;
          $("check-effect-reply").disabled = !pasted.trim();
        }})),
      h("button", {type: "button", class: "btn btn-small", id: "check-effect-reply", disabled: locked || !eligible || !pasted.trim(),
        onclick: () => api("reply", {include_manual: includeManual, reply: pasted})}, "Preview pasted result"),
      proposal ? h("section", {class: "pass-proposal"},
        h("div", {class: "pass-proposal-head"},
          h("strong", {}, `${changed.length} proposed · ${kept} kept · ${skipped} skipped`),
          button("Apply all", () => api("apply", {id: proposal.id}), locked || !changed.length, "btn-primary")),
        previewShot ? h("div", {class: "pass-actions"},
          h("select", {"aria-label": "Preview shot", onchange: (event) => { selected = event.target.value; frameTime = 0; render(true); }},
            changed.map((item) => h("option", {value: item.shot_id, selected: item === target},
              `Shot ${shots.find((shot) => shot.shot_id === item.shot_id).index + 1}`))),
          button(before ? "Show proposal" : "Show current", () => { before = !before; render(true); })) : null,
        previewShot ? preview(previewShot, before ? withMasks(previewShot, previewShot.effects ?? previewShot.default) : target.layers) : null,
        h("div", {class: "pass-proposal-cards"},
          ...items.map((item) => {
            const shot = shots.find((entry) => entry.shot_id === item.shot_id);
            const current = shot ? (shot.effects ?? shot.default) : [];
            return h("div", {class: "pass-proposal-card"},
              h("div", {class: "pass-proposal-card-head"},
                h("strong", {}, `Shot ${shot ? shot.index + 1 : item.shot_id}`),
                h("span", {class: `chip ${item.action === "replace" ? "" : "chip-quiet"}`},
                  item.action === "replace" ? "Proposed" : "Kept")),
              h("div", {class: "pass-proposal-row"},
                h("span", {class: "label"}, "Now"),
                h("span", {class: "effect-summary-chips"}, ...formatLayers(shot || {regions: []}, current, catalog))),
              item.action === "replace" ? h("div", {class: "pass-proposal-row"},
                h("span", {class: "label"}, "Suggest"),
                h("span", {class: "effect-summary-chips"}, ...formatLayers(shot || {regions: []}, item.layers, catalog))) : null,
              item.reason ? h("p", {class: "muted small"}, item.reason) : null);
          }))) : null,
    ];
  }

  function otherCameraLabel(catalog, excludeSet = -1, excludeEffect = -1) {
    for (let si = 0; si < draftSets.length; si++) {
      const presets = draftSets[si].presets;
      for (let ei = 0; ei < presets.length; ei++) {
        if (si === excludeSet && ei === excludeEffect) continue;
        const spec = catalog[presets[ei]];
        if (spec?.kind === "camera") return spec.label;
      }
    }
    return "";
  }

  function blockedKindsFor(catalog, setMode, excludeSet = -1, excludeEffect = -1) {
    const blocked = {};
    if (setMode !== FULL_FRAME) {
      blocked.camera = "Camera moves apply to Full frame only.";
    } else {
      const cam = otherCameraLabel(catalog, excludeSet, excludeEffect);
      if (cam) blocked.camera = `One camera move per shot — already using ${cam}. Change or remove that effect to pick another.`;
    }
    if (setMode === "behind") blocked.lighting = "Lighting effects do not support Behind mode.";
    return blocked;
  }

  function previewIcon() {
    const icon = h("span", {class: "pass-preview-pick-icon", "aria-hidden": "true"});
    icon.innerHTML = '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><rect x="1.5" y="1.5" width="5" height="5" rx="1"/><rect x="9.5" y="1.5" width="5" height="5" rx="1"/><rect x="1.5" y="9.5" width="5" height="5" rx="1"/><rect x="9.5" y="9.5" width="5" height="5" rx="1"/></svg>';
    return icon;
  }

  function openPresetPopup(shot, target, triggerEl) {
    const popup = presetDialog();
    if (!popup) return;
    stopPlayback();
    const catalog = state.boot.video.enhancement_presets;
    const maxLayers = state.boot.video.enhancement_max_layers || 16;
    const isNewSet = target.kind === "newSet";
    const isAddToSet = target.kind === "addToSet";
    const set = isNewSet ? null : draftSets[target.setIndex];
    const currentPreset = target.kind === "replace" ? (set?.presets[target.effectIndex] || "") : "";
    const usedPresets = new Set(draftSets.flatMap((entry, si) =>
      entry.presets.filter((_, ei) => target.kind !== "replace" || si !== target.setIndex || ei !== target.effectIndex)));
    const blockedKinds = blockedKindsFor(
      catalog,
      isNewSet ? FULL_FRAME : (set?.mode || FULL_FRAME),
      target.kind === "replace" ? target.setIndex : -1,
      target.kind === "replace" ? target.effectIndex : -1,
    );
    const firstUnblocked = ["camera", "atmosphere", "lighting"].find((kind) => !blockedKinds[kind]) || "atmosphere";
    const initialCategory = target.kind === "replace"
      ? (catalog[currentPreset]?.kind || firstUnblocked)
      : firstUnblocked;
    const title = isNewSet
      ? "Add effect set"
      : isAddToSet
        ? `Add effect · Set ${target.setIndex + 1}`
        : `Choose effect · Set ${target.setIndex + 1}`;
    const subtitle = isNewSet
      ? "Pick a first visual effect for this new set."
      : isAddToSet
        ? "Effects in the same set share the same target and intensity."
        : "Replace this effect in the set.";
    openEffectPresetPopup({
      h,
      dialog: popup,
      presets: catalog,
      title,
      subtitle,
      currentPreset,
      usedPresets,
      blockedKinds,
      shotImage: rowOf(shot)?.image || "",
      initialCategory,
      onSelect: (id) => {
        if (isNewSet) {
          if (totalEffects() < maxLayers) draftSets.push({mode: FULL_FRAME, region_ids: [], presets: [id]});
        } else if (isAddToSet) {
          if (totalEffects() < maxLayers && set) set.presets.push(id);
        } else if (set) {
          set.presets[target.effectIndex] = id;
          pruneSetFields(set, catalog);
        }
        render(true);
      },
      restoreFocus: (picked) => {
        const modal = dialog();
        if (!modal?.open) return;
        if (picked) {
          const targetSet = isNewSet ? draftSets.length - 1 : target.setIndex;
          const targetEffect = isNewSet
            ? 0
            : isAddToSet
              ? (draftSets[targetSet]?.presets.length ?? 1) - 1
              : target.effectIndex;
          modal.querySelector(
            `.pass-set-card[data-set-index="${targetSet}"] .pass-effect-row[data-effect-index="${targetEffect}"] .pass-preview-pick`,
          )?.focus();
        } else if (triggerEl?.isConnected) {
          triggerEl.focus();
        }
      },
    });
  }

  function maskEditor(shot) {
    const isAdd = refine.mode === "add";
    const region = isAdd ? null : shot.regions.find((item) => item.id === refine.region);
    if (!isAdd && !region) return null;
    const color = isAdd ? regionColor(shot.regions.length) : regionColorMap(shot).get(region.id);
    const portrait = state.project.video.orientation === "portrait";
    const image = h("img", {src: rowOf(shot).image, alt: "Shot to segment"});
    const addPoint = (event, type) => {
      if (busy()) return;
      const rect = image.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      const x = (event.clientX - rect.left) / rect.width;
      const y = (event.clientY - rect.top) / rect.height;
      if (x >= 0 && x <= 1 && y >= 0 && y <= 1 && refine[type].length < 32) {
        refine[type].push([x, y]);
        render(true);
      }
    };
    return h("div", {class: "pass-proposal"},
      h("div", {class: "pass-mask-head"},
        h("span", {class: "pass-mask-title", style: color ? {"--seg-color": color} : undefined},
          h("span", {class: "pass-seg-dot", "aria-hidden": "true"}),
          h("span", {class: "pass-seg-text"}, isAdd ? "New segment" : `Refining: ${region.label} · ${region.position}`)),
        button("Done", () => { refine = null; render(true); }, busy())),
      isAdd ? h("input", {
        type: "text",
        class: "pass-mask-label",
        "aria-label": "Segment label",
        placeholder: "Label (optional, e.g. lantern)",
        maxLength: 48,
        value: refine.label,
        disabled: busy(),
        oninput: (event) => { refine.label = event.target.value; },
      }) : null,
      h("div", {class: "pass-actions"},
        h("select", {"aria-label": "Point type", onchange: (event) => { refine.type = event.target.value; }},
          h("option", {value: "include", selected: refine.type === "include"}, "Include point"),
          h("option", {value: "exclude", selected: refine.type === "exclude"}, "Exclude point")),
        button("Clear points", () => { refine.include = []; refine.exclude = []; render(true); }, busy()),
        button(isAdd ? "Add segment" : "Update mask", async () => {
          if (isAdd) {
            const prevIds = new Set(shot.regions.map((item) => item.id));
            const result = await api("region", {
              shot_id: shot.shot_id,
              image: shot.image,
              label: refine.label,
              include: refine.include,
              exclude: refine.exclude,
            });
            if (result) {
              refine = {...refine, pendingFrom: prevIds, include: [], exclude: []};
              render(true);
            }
            return;
          }
          const result = await api("refine", {shot_id: shot.shot_id, image: shot.image, region_id: region.id,
            include: refine.include, exclude: refine.exclude});
          if (result) {
            refine = {...refine, include: [], exclude: []};
            render(true);
          }
        }, busy() || !refine.include.length, "btn-primary")),
      h("p", {class: "muted small"}, isAdd
        ? "Click to place selected point · right-click to exclude · then click Add segment."
        : "Click to place selected point · right-click to exclude · then click Update mask."),
      h("div", {
        class: `pass-mask${portrait ? " portrait" : ""}`,
        onclick: (event) => addPoint(event, refine.type),
        oncontextmenu: (event) => { event.preventDefault(); addPoint(event, "exclude"); },
      }, image, isAdd ? null : h("div", {class: "pass-mask-overlay", "aria-hidden": "true",
        style: {"--seg-color": color || "var(--accent)", "--seg-mask": maskUrl(region.url)}}),
      ...["include", "exclude"].flatMap((type) => refine[type].map(([x, y]) =>
        h("span", {class: `pass-point ${type}`, style: {left: `${x * 100}%`, top: `${y * 100}%`}}, type === "include" ? "+" : "−")))));
  }

  function renderRegionControls({shot, set, colors, regionMap, locked, select}) {
    if (set.mode === FULL_FRAME || !shot.regions.length) return null;
    const availableRegions = shot.regions.filter((region) => !set.region_ids.includes(region.id));
    let singleOrChips;
    if (set.region_ids.length === 1) {
      const regionId = set.region_ids[0];
      const regionSelect = select("Region", (event) => {
        set.region_ids = [event.target.value];
        render(true);
      }, shot.regions.map((region) => h("option", {
        value: region.id,
        selected: region.id === regionId,
      }, `${region.label} · ${region.position}`)));
      regionSelect.value = regionId;
      singleOrChips = h("div", {
        class: "pass-region-wrap",
        style: {"--seg-color": colors.get(regionId) || "var(--accent)"},
      }, h("span", {class: "pass-seg-dot", "aria-hidden": "true"}), regionSelect);
    } else {
      singleOrChips = h("div", {class: "pass-region-chips"},
        ...set.region_ids.map((rid) => {
          const region = regionMap.get(rid);
          if (!region) return null;
          return h("span", {
            class: "pass-region-chip",
            style: {"--seg-color": colors.get(rid) || "var(--accent)"},
            onmouseenter: () => setFocusRegions([rid]),
            onmouseleave: () => setFocusRegions(set.region_ids),
          },
          h("span", {class: "pass-seg-dot", "aria-hidden": "true"}),
          h("span", {class: "pass-seg-text"}, `${region.label} · ${region.position}`),
          h("button", {
            type: "button",
            class: "pass-region-remove",
            "aria-label": `Remove ${region.label}`,
            disabled: locked,
            onclick: () => {
              set.region_ids = set.region_ids.filter((id) => id !== rid);
              render(true);
            },
          }, "×"));
        }));
    }
    const addObjectSelect = availableRegions.length > 0
      ? select("Add object", (event) => {
        if (!event.target.value) return;
        set.region_ids.push(event.target.value);
        render(true);
      }, [
        h("option", {value: "", selected: true}, "+ Add object"),
        ...availableRegions.map((region) => h("option", {value: region.id}, `${region.label} · ${region.position}`)),
      ], {class: "pass-add-region"})
      : null;
    return h("div", {class: "pass-region-row"}, singleOrChips, addObjectSelect);
  }

  function renderEffectRow({shot, set, setIndex, presetId, effectIndex, catalog, locked, select}) {
    const spec = catalog[presetId];
    const blocked = blockedKindsFor(catalog, set.mode, setIndex, effectIndex);
    const effectSelect = select("Effect", (event) => {
      set.presets[effectIndex] = event.target.value;
      pruneSetFields(set, catalog);
      render(true);
    }, EFFECT_CATEGORIES.map((cat) => {
      const catBlocked = Boolean(blocked[cat.id]);
      return h("optgroup", {label: cat.label, disabled: catBlocked},
        Object.entries(catalog)
          .filter(([, item]) => item.kind === cat.id)
          .map(([id, item]) => h("option", {
            value: id,
            selected: id === presetId,
            disabled: catBlocked,
          }, item.label)));
    }));
    effectSelect.value = presetId;
    return h("div", {class: "pass-effect-row", "data-effect-index": String(effectIndex)},
      h("span", {class: `pass-kind-dot pass-kind-${spec.kind}`, "aria-hidden": "true"}),
      effectSelect,
      h("button", {
        type: "button",
        class: "btn btn-small pass-preview-pick",
        title: "Choose effect with visual previews",
        "aria-haspopup": "dialog",
        "aria-controls": "effect-preset-dialog",
        disabled: locked,
        onclick: (event) => openPresetPopup(shot, {kind: "replace", setIndex, effectIndex}, event.currentTarget),
      }, previewIcon(), "Previews"),
      set.presets.length > 1
        ? h("button", {
          type: "button",
          class: "btn btn-small btn-icon pass-effect-remove",
          "aria-label": `Remove ${spec.label}`,
          disabled: locked,
          onclick: () => {
            set.presets.splice(effectIndex, 1);
            pruneSetFields(set, catalog);
            render(true);
          },
        }, "×")
        : null);
  }

  function renderSetCard({shot, set, setIndex, catalog, colors, regionMap, locked, maxLayers, countEffects, swapSet}) {
    const hasCamera = set.presets.some((id) => catalog[id]?.kind === "camera");
    const hasLighting = set.presets.some((id) => catalog[id]?.kind === "lighting");
    const hasOverlay = set.presets.some((id) => catalog[id]?.kind !== "camera");
    const accentColor = setAccentColor(set, colors, catalog);
    const select = (name, onchange, options, extra = {}) =>
      h("select", {"aria-label": name, disabled: locked, onchange, ...extra}, options);

    const applyModes = [FULL_FRAME, ...(shot.regions.length ? ["inside", "outside", "behind"] : [])];
    const applySelect = select("Apply", (event) => {
      set.mode = event.target.value;
      pruneSetFields(set, catalog);
      if (set.mode !== FULL_FRAME && !set.region_ids.length && shot.regions.length) {
        set.region_ids = [shot.regions[0].id];
      }
      render(true);
    }, applyModes.map((way) => h("option", {
      value: way,
      selected: way === set.mode,
      disabled: (way !== FULL_FRAME && hasCamera) || (way === "behind" && hasLighting),
    }, MODE_LABELS[way])));
    applySelect.value = set.mode;

    return h("div", {
      class: "pass-layer pass-layer-card pass-set-card",
      "data-set-index": String(setIndex),
      style: {"--set-color": accentColor},
      onmouseenter: set.region_ids.length ? () => setFocusRegions(set.region_ids) : undefined,
      onmouseleave: set.region_ids.length ? () => setFocusRegions(null) : undefined,
    },
    h("div", {class: "pass-set-head"},
      applySelect,
      h("span", {class: "pass-set-title"}, `Set ${setIndex + 1}`),
      h("span", {class: "spacer"}),
      button("↑", () => swapSet(setIndex, setIndex - 1), locked || setIndex === 0),
      button("↓", () => swapSet(setIndex, setIndex + 1), locked || setIndex === draftSets.length - 1),
      button("Remove", () => { draftSets.splice(setIndex, 1); render(true); }, locked)),
    hasCamera && shot.regions.length > 0
      ? h("p", {class: "muted small pass-set-hint"}, "Camera moves apply to Full frame only.")
      : null,
    renderRegionControls({shot, set, colors, regionMap, locked, select}),
    h("div", {class: "pass-effect-list"},
      ...set.presets.map((presetId, effectIndex) =>
        renderEffectRow({shot, set, setIndex, presetId, effectIndex, catalog, locked, select})),
      h("div", {class: "pass-set-actions"},
        h("button", {
          type: "button",
          class: "btn btn-small",
          "aria-haspopup": "dialog",
          "aria-controls": "effect-preset-dialog",
          disabled: locked || countEffects >= maxLayers,
          onclick: (event) => openPresetPopup(shot, {kind: "addToSet", setIndex}, event.currentTarget),
        }, "+ Add effect"))),
    hasOverlay ? h("div", {class: "pass-layer-ranges"},
      ...[["intensity", 1, 0.05, 1, (v) => `${Math.round(v * 100)}%`],
        ...(set.mode !== FULL_FRAME && set.region_ids.length ? [["feather", 32, 1, 1, (v) => `${v} px`]] : [])]
        .map(([key, max, step, fallback, fmt]) => {
          const val = set[key] ?? fallback;
          const readout = h("span", {class: "mono small"}, fmt(val));
          return h("label", {class: "pass-range"},
            h("span", {class: "pass-range-head"}, h("span", {}, key), readout),
            h("input", {type: "range", min: 0, max, step, value: val, disabled: locked,
              oninput: (event) => { readout.textContent = fmt(Number(event.target.value)); },
              onchange: (event) => { set[key] = Number(event.target.value); render(true); }}));
        })) : null);
  }

  function edit(shot) {
    const catalog = state.boot.video.enhancement_presets;
    const maxLayers = state.boot.video.enhancement_max_layers || 16;
    const locked = busy();
    const canRefine = Boolean(state.boot.gpu);
    const colors = regionColorMap(shot);
    const regionMap = new Map((shot.regions || []).map((region) => [region.id, region]));
    const rowKey = {shot_id: shot.shot_id, image: shot.image};
    const swapSet = (index, other) => {
      [draftSets[index], draftSets[other]] = [draftSets[other], draftSets[index]];
      render(true);
    };
    const flatLayers = setsToLayers(draftSets, catalog);
    const countEffects = flatLayers.length;
    return [
      h("div", {class: "effects-picker-body pass-edit-grid"},
        h("section", {class: "effect-preview", "aria-label": "Layer preview"},
          h("div", {class: "effect-options-controls"},
            h("strong", {}, `Shot ${shot.index + 1}`),
            shot.regions.length ? h("button", {type: "button", class: "btn btn-small pass-seg-toggle", "aria-pressed": String(showSegments),
              onclick: () => { showSegments = !showSegments; render(true); }},
            showSegments ? `Hide segments (${shot.regions.length})` : `Show segments (${shot.regions.length})`) : null),
          preview(shot, withMasks(shot, flatLayers)),
          shot.regions.length || canRefine ? h("div", {class: "pass-segments"},
            h("div", {class: "pass-segments-head"},
              h("span", {class: "label"}, `Segments (${shot.regions.length})`),
              h("span", {class: "muted small"}, shot.regions.length
                ? (canRefine ? "Hover to highlight · click to refine" : "Hover to highlight")
                : "No auto-detected segments · add one with points")),
            h("div", {class: "pass-seg-list"},
              ...shot.regions.map((region, index) => h("button", {
                type: "button",
                class: "pass-seg-chip",
                style: {"--seg-color": regionColor(index)},
                title: `${region.label} · ${region.position}`,
                disabled: locked,
                "aria-pressed": canRefine ? String(refine?.mode === "refine" && refine?.region === region.id) : undefined,
                onmouseenter: () => setFocusRegions(region.id),
                onmouseleave: () => setFocusRegions(null),
                onfocus: () => setFocusRegions(region.id),
                onblur: () => setFocusRegions(null),
                onclick: canRefine ? () => {
                  refine = refine?.mode === "refine" && refine?.region === region.id
                    ? null
                    : {mode: "refine", region: region.id, type: "include", include: [], exclude: []};
                  render(true);
                } : undefined,
              },
              h("span", {class: "pass-seg-dot", "aria-hidden": "true"}),
              h("span", {class: "pass-seg-text"}, region.label),
              h("span", {class: "pass-seg-pos"}, `· ${region.position}`))),
              canRefine ? h("button", {
                type: "button",
                class: "pass-seg-chip pass-seg-add",
                disabled: locked || shot.regions.length >= MAX_TOTAL_REGIONS,
                "aria-pressed": String(refine?.mode === "add"),
                onclick: () => {
                  refine = refine?.mode === "add"
                    ? null
                    : {mode: "add", region: null, label: "", type: "include", include: [], exclude: []};
                  render(true);
                },
              }, "+ Add segment") : null)) : null,
          refine ? maskEditor(shot) : null),
        h("section", {class: "pass-layers-col", "aria-label": "Effect sets"},
          h("div", {class: "effect-options-controls"},
            h("span", {class: "muted small"},
              `${draftSets.length} ${draftSets.length === 1 ? "set" : "sets"} · ${countEffects} / ${maxLayers} effects`),
            h("button", {
              type: "button",
              class: "btn btn-small",
              "aria-haspopup": "dialog",
              "aria-controls": "effect-preset-dialog",
              disabled: locked || countEffects >= maxLayers,
              onclick: (event) => openPresetPopup(shot, {kind: "newSet"}, event.currentTarget),
            }, "+ Add effect set")),
          draftSets.length ? null : h("p", {class: "muted small"}, "No effect sets. The still plays as is."),
          ...draftSets.map((set, setIndex) =>
            renderSetCard({shot, set, setIndex, catalog, colors, regionMap, locked, maxLayers, countEffects, swapSet})),
          h("div", {class: "pass-actions pass-layer-foot"},
            button("Reset to planned effects", async () => { if (await api("shot", {...rowKey, reset: true})) dialog().close(); }, locked),
            button("Save shot effects", async () => {
              if (await api("shot", {...rowKey, effects: setsToLayers(draftSets, catalog)})) dialog().close();
            }, locked, "btn-primary"))))
    ];
  }

  function render(force = false) {
    const modal = dialog();
    if (!modal?.open) return;
    if (project !== state.project.name || !panel()) { stopPlayback(); closePresetDialog(); modal.close(); return; }
    if (!force && presetDialog()?.open) return;
    if (!force && modal.contains(document.activeElement) && document.activeElement.matches("textarea, input[type='text']")) return;
    const shot = panel().shots.find((item) => item.shot_id === selected);
    const run = state.videoRun;
    const running = run?.active && ["segment_effects", "enhance_effects", "refine_effect_mask"].includes(run.kind);
    if (mode === "edit" && shot && refine?.mode === "add" && refine.pendingFrom) {
      const added = shot.regions.find((region) => !refine.pendingFrom.has(region.id));
      if (added) {
        refine = {mode: "refine", region: added.id, type: "include", include: [], exclude: []};
      } else if (!running) {
        delete refine.pendingFrom;
      }
    }
    const scroll = modal.querySelector(".style-body")?.scrollTop ?? 0;
    modal.replaceChildren(h("div", {class: "library-head"},
      h("h2", {id: "effect-pass-title"}, mode === "edit" && shot ? `Shot ${shot.index + 1} effects` : "Enhance visual effects"),
      h("span", {class: "spacer"}),
      running ? button("Interrupt", interrupt) : null,
      button("Close", () => { stopPlayback(); closePresetDialog(); modal.close(); })),
    h("div", {class: "style-body pass-body"},
      running ? h("p", {role: "status"}, run.status) : null,
      ...(mode === "edit" && shot ? edit(shot) : batch())));
    modal.onclose = () => { stopPlayback(); closePresetDialog(); };
    modal.querySelector(".style-body").scrollTop = scroll;
  }

  return {open, render};
}
