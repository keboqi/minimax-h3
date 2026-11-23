import {preloadEffectMasks} from "./effects.js";
import {paintStage, showStageStills, stageFrame, stageLayers, stageTimeline} from "./stage.js";

const FULL_FRAME = "full_frame";
const MODE_LABELS = {full_frame: "Full frame", inside: "Inside", outside: "Outside", behind: "Behind"};
const SEG_COLORS = 12;
const regionColor = (index) => `var(--seg-${index % SEG_COLORS})`;
const regionColorMap = (shot) => new Map((shot?.regions || []).map((region, index) => [region.id, regionColor(index)]));
const maskUrl = (url) => `url("${String(url).replace(/["\\]/g, "\\$&")}")`;

export function createEffectsPass({state, $, h, call, copy, writer, interrupt, busy}) {
  let project, selected, mode, draft, pasted = "", includeManual = false;
  let before = false, refine, frameTime = 0, showSegments = false, playFrame = null;
  const dialog = () => $("effect-pass-dialog");
  const panel = () => state.project.video?.effect_enhancement;
  const rowOf = (shot) => state.project.video.plan.rows[shot.index];
  const button = (label, onclick, disabled = false, extra = "") =>
    h("button", {type: "button", class: `btn btn-small ${extra}`.trim(), onclick, disabled}, label);
  const api = (action, body = {}) => call(`/project/video/effects/${action}`, body);
  const selectedWriter = () => writer();

  function stopPlayback() {
    if (playFrame !== null) cancelAnimationFrame(playFrame);
    playFrame = null;
  }

  function setFocusRegion(regionId) {
    const stage = dialog()?.querySelector(".pass-preview");
    if (!stage) return;
    const active = regionId || refine?.region || "";
    stage.classList.toggle("has-focus-region", Boolean(active));
    stage.querySelectorAll(".pass-seg-overlay").forEach((overlay) => {
      overlay.classList.toggle("is-focused", overlay.dataset.regionId === active);
    });
  }

  function open(shot = null) {
    stopPlayback();
    if (project !== state.project.name) {
      project = state.project.name;
      pasted = "";
      includeManual = false;
    }
    selected = shot?.shot_id;
    mode = shot ? "edit" : "batch";
    draft = shot ? structuredClone(shot.effects ?? shot.default) : null;
    before = false;
    refine = null;
    frameTime = 0;
    showSegments = false;
    if (!dialog().open) dialog().showModal();
    render(true);
  }

  function withMasks(shot, effects) {
    const urls = new Map(shot.regions.map((region) => [region.id, region.url]));
    return effects.map((effect) => ({...effect, mask_url: urls.get(effect.region_id)}));
  }

  function formatLayers(shot, effects, catalog) {
    const regions = new Map((shot.regions || []).map((r) => [r.id, r.label]));
    const colors = regionColorMap(shot);
    if (!effects?.length) return [h("span", {class: "muted small"}, "No effects")];
    return effects.map((layer) => {
      const label = catalog[layer.preset]?.label || layer.preset;
      const regionName = regions.get(layer.region_id);
      const modeText = layer.mode && layer.mode !== FULL_FRAME
        ? ` · ${MODE_LABELS[layer.mode] || layer.mode}${regionName ? ` ${regionName}` : ""}`
        : "";
      const color = layer.mode !== FULL_FRAME ? colors.get(layer.region_id) : null;
      return h("span", {class: "chip"},
        color ? h("span", {class: "pass-seg-dot", style: {"--seg-color": color}, "aria-hidden": "true"}) : null,
        `${label}${modeText}`);
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
    const [view] = stageTimeline([{start: 0, end: duration, index: row.index, fade: 0, url: row.image,
      effects: effects.map(({preset}) => preset), effect_layers: effects}], sounds);
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
        playBtn.textContent = "Play preview";
        setTime(Number(event.target.value));
      }});
    const playBtn = h("button", {type: "button", class: "btn btn-small", onclick: () => {
      if (playFrame !== null) {
        stopPlayback();
        playBtn.textContent = "Play preview";
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
    Promise.all([back.decode(), preloadEffectMasks([view])]).then(paint, paint);
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

  function replaceLayer(index, values) {
    draft[index] = {...draft[index], ...values};
    if (draft[index].mode === FULL_FRAME) delete draft[index].region_id;
    render(true);
  }

  function maskEditor(shot) {
    const region = shot.regions.find((item) => item.id === refine.region);
    if (!region) return null;
    const color = regionColorMap(shot).get(region.id);
    const image = h("img", {src: rowOf(shot).image, alt: "Shot to segment"});
    return h("div", {class: "pass-proposal"},
      h("div", {class: "pass-mask-head"},
        h("span", {class: "pass-mask-title", style: color ? {"--seg-color": color} : undefined},
          h("span", {class: "pass-seg-dot", "aria-hidden": "true"}),
          h("span", {class: "pass-seg-text"}, `Refining: ${region.label} · ${region.position}`)),
        button("Done", () => { refine = null; render(true); }, busy())),
      h("div", {class: "pass-actions"},
        h("select", {"aria-label": "Point type", onchange: (event) => { refine.type = event.target.value; }},
          h("option", {value: "include", selected: refine.type === "include"}, "Include point"),
          h("option", {value: "exclude", selected: refine.type === "exclude"}, "Exclude point")),
        button("Clear points", () => { refine.include = []; refine.exclude = []; render(true); }, busy()),
        button("Update mask", async () => {
          const result = await api("refine", {shot_id: shot.shot_id, image: shot.image, region_id: region.id,
            include: refine.include, exclude: refine.exclude});
          if (result) { refine = null; render(true); }
        }, busy() || !refine.include.length, "btn-primary")),
      h("div", {class: "pass-mask", onclick: (event) => {
        if (busy()) return;
        const rect = image.getBoundingClientRect();
        if (!rect.width || !rect.height) return;
        const x = (event.clientX - rect.left) / rect.width, y = (event.clientY - rect.top) / rect.height;
        if (x >= 0 && x <= 1 && y >= 0 && y <= 1 && refine[refine.type].length < 32) {
          refine[refine.type].push([x, y]);
          render(true);
        }
      }}, image, h("div", {class: "pass-mask-overlay", "aria-hidden": "true",
        style: {"--seg-color": color || "var(--accent)", "--seg-mask": maskUrl(region.url)}}),
      ...["include", "exclude"].flatMap((type) => refine[type].map(([x, y]) =>
        h("span", {class: `pass-point ${type}`, style: {left: `${x * 100}%`, top: `${y * 100}%`}}, type === "include" ? "+" : "−")))));
  }

  function edit(shot) {
    const catalog = state.boot.video.enhancement_presets;
    const maxLayers = state.boot.video.enhancement_max_layers || 8;
    const locked = busy();
    const canRefine = Boolean(state.boot.gpu);
    const colors = regionColorMap(shot);
    const rowKey = {shot_id: shot.shot_id, image: shot.image};
    const swap = (index, other) => {
      [draft[index], draft[other]] = [draft[other], draft[index]];
      render(true);
    };
    return [
      h("div", {class: "effects-picker-body pass-edit-grid"},
        h("section", {class: "effect-preview", "aria-label": "Layer preview"},
          h("div", {class: "effect-options-controls"},
            h("strong", {}, `Shot ${shot.index + 1}`),
            shot.regions.length ? h("button", {type: "button", class: "btn btn-small pass-seg-toggle", "aria-pressed": String(showSegments),
              onclick: () => { showSegments = !showSegments; render(true); }},
            showSegments ? `Hide segments (${shot.regions.length})` : `Show segments (${shot.regions.length})`) : null),
          preview(shot, withMasks(shot, draft)),
          shot.regions.length ? h("div", {class: "pass-segments"},
            h("div", {class: "pass-segments-head"},
              h("span", {class: "label"}, `Segments (${shot.regions.length})`),
              h("span", {class: "muted small"}, canRefine ? "Hover to highlight · click to refine" : "Hover to highlight")),
            h("div", {class: "pass-seg-list"},
              ...shot.regions.map((region, index) => h("button", {
                type: "button",
                class: "pass-seg-chip",
                style: {"--seg-color": regionColor(index)},
                title: `${region.label} · ${region.position}`,
                disabled: locked,
                "aria-pressed": canRefine ? String(refine?.region === region.id) : undefined,
                onmouseenter: () => setFocusRegion(region.id),
                onmouseleave: () => setFocusRegion(null),
                onfocus: () => setFocusRegion(region.id),
                onblur: () => setFocusRegion(null),
                onclick: canRefine ? () => {
                  refine = refine?.region === region.id ? null : {region: region.id, type: "include", include: [], exclude: []};
                  render(true);
                } : undefined,
              },
              h("span", {class: "pass-seg-dot", "aria-hidden": "true"}),
              h("span", {class: "pass-seg-text"}, region.label),
              h("span", {class: "pass-seg-pos"}, `· ${region.position}`))))) : null,
          refine ? maskEditor(shot) : null),
        h("section", {class: "pass-layers-col", "aria-label": "Effect layers"},
          h("div", {class: "effect-options-controls"},
            h("span", {class: "muted small"}, `Layers ${draft.length} / ${maxLayers}`),
            button("+ Add effect", () => { draft.push({preset: "warm_glow", mode: FULL_FRAME}); render(true); }, locked || draft.length >= maxLayers)),
          draft.length ? null : h("p", {class: "muted small"}, "No layers. The still plays as is."),
          ...draft.map((layer, index) => {
            const spec = catalog[layer.preset];
            const layerColor = layer.region_id ? colors.get(layer.region_id) : null;
            const select = (name, onchange, options) => h("select", {"aria-label": name, disabled: locked, onchange}, options);
            return h("div", {
              class: `pass-layer pass-layer-card pass-kind-${spec.kind}`,
              onmouseenter: layer.region_id ? () => setFocusRegion(layer.region_id) : undefined,
              onmouseleave: layer.region_id ? () => setFocusRegion(null) : undefined,
            },
              h("div", {class: "pass-layer-head"},
                select("Effect", (event) => replaceLayer(index, {preset: event.target.value, mode: FULL_FRAME}),
                  Object.entries(catalog).map(([id, item]) => h("option", {value: id, selected: id === layer.preset}, item.label))),
                button("↑", () => swap(index, index - 1), locked || index === 0),
                button("↓", () => swap(index, index + 1), locked || index === draft.length - 1),
                button("Remove", () => { draft.splice(index, 1); render(true); }, locked)),
              h("div", {class: "pass-layer-row"},
                select("Apply", (event) => replaceLayer(index, {mode: event.target.value, region_id: layer.region_id || shot.regions[0]?.id}),
                  spec.modes.filter((way) => way === FULL_FRAME || shot.regions.length)
                    .map((way) => h("option", {value: way, selected: way === layer.mode}, MODE_LABELS[way]))),
                layer.region_id ? h("div", {class: "pass-region-wrap", style: {"--seg-color": layerColor || "var(--accent)"}},
                  h("span", {class: "pass-seg-dot", "aria-hidden": "true"}),
                  select("Region", (event) => replaceLayer(index, {region_id: event.target.value}),
                    shot.regions.map((region) => h("option", {value: region.id, selected: region.id === layer.region_id},
                      `${region.label} · ${region.position}`)))) : null),
              spec.kind !== "camera" ? h("div", {class: "pass-layer-ranges"},
                ...[["intensity", 1, 0.05, 1, (v) => `${Math.round(v * 100)}%`],
                  ...(layer.region_id ? [["feather", 32, 1, 1, (v) => `${v} px`]] : [])]
                  .map(([key, max, step, fallback, fmt]) => {
                    const val = layer[key] ?? fallback;
                    const readout = h("span", {class: "mono small"}, fmt(val));
                    return h("label", {class: "pass-range"},
                      h("span", {class: "pass-range-head"}, h("span", {}, key), readout),
                      h("input", {type: "range", min: 0, max, step, value: val, disabled: locked,
                        oninput: (event) => { readout.textContent = fmt(Number(event.target.value)); },
                        onchange: (event) => replaceLayer(index, {[key]: Number(event.target.value)})}));
                  })) : null);
          }),
          h("div", {class: "pass-actions pass-layer-foot"},
            button("Reset to planned effects", async () => { if (await api("shot", {...rowKey, reset: true})) dialog().close(); }, locked),
            button("Save shot effects", async () => { if (await api("shot", {...rowKey, effects: draft})) dialog().close(); }, locked, "btn-primary"))))
    ];
  }

  function render(force = false) {
    const modal = dialog();
    if (!modal?.open) return;
    if (project !== state.project.name || !panel()) { stopPlayback(); modal.close(); return; }
    if (!force && modal.contains(document.activeElement) && document.activeElement.matches("textarea")) return;
    const shot = panel().shots.find((item) => item.shot_id === selected);
    const run = state.videoRun;
    const running = run?.active && ["segment_effects", "enhance_effects", "refine_effect_mask"].includes(run.kind);
    const scroll = modal.querySelector(".style-body")?.scrollTop ?? 0;
    modal.replaceChildren(h("div", {class: "library-head"},
      h("h2", {id: "effect-pass-title"}, mode === "edit" && shot ? `Shot ${shot.index + 1} effects` : "Enhance visual effects"),
      h("span", {class: "spacer"}),
      running ? button("Interrupt", interrupt) : null,
      button("Close", () => { stopPlayback(); modal.close(); })),
    h("div", {class: "style-body pass-body"},
      running ? h("p", {role: "status"}, run.status) : null,
      ...(mode === "edit" && shot ? edit(shot) : batch())));
    modal.onclose = () => stopPlayback();
    modal.querySelector(".style-body").scrollTop = scroll;
  }

  return {open, render};
}
