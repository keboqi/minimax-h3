import {preloadEffectMasks} from "./effects.js";
import {paintStage, showStageStills, stageFrame, stageLayers, stageTimeline} from "./stage.js";

const FULL_FRAME = "full_frame";
const MODE_LABELS = {full_frame: "Full frame", inside: "Inside", outside: "Outside", behind: "Behind"};

export function createEffectsPass({state, $, h, call, copy, writer, interrupt, busy}) {
  let project, selected, mode, draft, pasted = "", includeManual = false;
  let before = false, refine, frameTime = 0;
  const dialog = () => $("effect-pass-dialog");
  const panel = () => state.project.video?.effect_enhancement;
  const rowOf = (shot) => state.project.video.plan.rows[shot.index];
  const button = (label, onclick, disabled = false) => h("button", {type: "button", class: "btn btn-small", onclick, disabled}, label);
  const api = (action, body = {}) => call(`/project/video/effects/${action}`, body);
  const selectedWriter = () => writer();

  function open(shot = null) {
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
    if (!dialog().open) dialog().showModal();
    render(true);
  }

  function withMasks(shot, effects) {
    const urls = new Map(shot.regions.map((region) => [region.id, region.url]));
    return effects.map((effect) => ({...effect, mask_url: urls.get(effect.region_id)}));
  }

  function preview(shot, effects) {
    const row = rowOf(shot);
    const orientation = state.project.video.orientation;
    const {back, front, overlay} = stageLayers();
    [back, front, overlay].forEach((element) => element.removeAttribute("id"));
    const stage = h("div", {class: `video-stage pass-preview ${orientation === "portrait" ? "portrait" : ""}`}, back, front, overlay);
    const duration = Math.max(0.05, row.end - row.start);
    const sounds = (state.project.result?.sounds || [])
      .filter((sound) => sound.start >= row.start && sound.start < row.end)
      .map((sound) => ({...sound, start: sound.start - row.start}));
    const [view] = stageTimeline([{start: 0, end: duration, index: row.index, fade: 0, url: row.image,
      effects: effects.map(({preset}) => preset), effect_layers: effects}], sounds);
    const layers = {back, front, overlay};
    const paint = () => paintStage(layers, stageFrame([view], Math.min(frameTime, duration)), true);
    showStageStills(layers, stageFrame([view], 0));
    Promise.all([back.decode(), preloadEffectMasks([view])]).then(paint, paint);
    return h("div", {class: "pass-preview-wrap"}, stage,
      h("input", {type: "range", min: 0, max: duration, step: 0.025, value: Math.min(frameTime, duration),
        "aria-label": "Preview time", oninput: (event) => { frameTime = Number(event.target.value); paint(); }}));
  }

  function batch() {
    const {shots, proposal} = panel();
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
        h("strong", {}, `${changed.length} proposed · ${kept} kept · ${skipped} skipped`),
        ...items.filter((item) => item.reason).map((item) => h("div", {class: "muted small"},
          `Shot ${shots.find((shot) => shot.shot_id === item.shot_id).index + 1}. ${item.reason}`)),
        previewShot ? h("div", {class: "pass-actions"},
          h("select", {"aria-label": "Preview shot", onchange: (event) => { selected = event.target.value; frameTime = 0; render(true); }},
            changed.map((item) => h("option", {value: item.shot_id, selected: item === target},
              `Shot ${shots.find((shot) => shot.shot_id === item.shot_id).index + 1}`))),
          button(before ? "Show proposal" : "Show current", () => { before = !before; render(true); })) : null,
        previewShot ? preview(previewShot, before ? withMasks(previewShot, previewShot.effects ?? previewShot.default) : target.layers) : null,
        button("Apply all", () => api("apply", {id: proposal.id}), locked || !changed.length)) : null,
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
    const image = h("img", {src: rowOf(shot).image, alt: "Shot to segment"});
    return h("div", {class: "pass-proposal"},
      h("div", {class: "pass-actions"},
        h("select", {"aria-label": "Point type", onchange: (event) => { refine.type = event.target.value; }},
          h("option", {value: "include", selected: refine.type === "include"}, "Include point"),
          h("option", {value: "exclude", selected: refine.type === "exclude"}, "Exclude point")),
        button("Clear points", () => { refine.include = []; refine.exclude = []; render(true); }, busy()),
        button("Update mask", async () => {
          const result = await api("refine", {shot_id: shot.shot_id, image: shot.image, region_id: region.id,
            include: refine.include, exclude: refine.exclude});
          if (result) { refine = null; render(true); }
        }, busy() || !refine.include.length)),
      h("div", {class: "pass-mask", onclick: (event) => {
        if (busy()) return;
        const rect = image.getBoundingClientRect();
        if (!rect.width || !rect.height) return;
        const x = (event.clientX - rect.left) / rect.width, y = (event.clientY - rect.top) / rect.height;
        if (x >= 0 && x <= 1 && y >= 0 && y <= 1 && refine[refine.type].length < 32) {
          refine[refine.type].push([x, y]);
          render(true);
        }
      }}, image, h("img", {src: region.url, class: "pass-mask-overlay", alt: ""}),
      ...["include", "exclude"].flatMap((type) => refine[type].map(([x, y]) =>
        h("span", {class: `pass-point ${type}`, style: {left: `${x * 100}%`, top: `${y * 100}%`}}, type === "include" ? "+" : "−")))));
  }

  function edit(shot) {
    const catalog = state.boot.video.enhancement_presets;
    const locked = busy();
    const rowKey = {shot_id: shot.shot_id, image: shot.image};
    const swap = (index, other) => {
      [draft[index], draft[other]] = [draft[other], draft[index]];
      render(true);
    };
    return [
      preview(shot, withMasks(shot, draft)),
      ...draft.map((layer, index) => {
        const spec = catalog[layer.preset];
        const select = (name, onchange, options) => h("select", {"aria-label": name, disabled: locked, onchange}, options);
        return h("div", {class: "pass-layer"},
          select("Effect", (event) => replaceLayer(index, {preset: event.target.value, mode: FULL_FRAME}),
            Object.entries(catalog).map(([id, item]) => h("option", {value: id, selected: id === layer.preset}, item.label))),
          select("Apply", (event) => replaceLayer(index, {mode: event.target.value, region_id: layer.region_id || shot.regions[0]?.id}),
            spec.modes.filter((way) => way === FULL_FRAME || shot.regions.length)
              .map((way) => h("option", {value: way, selected: way === layer.mode}, MODE_LABELS[way]))),
          layer.region_id ? select("Region", (event) => replaceLayer(index, {region_id: event.target.value}),
            shot.regions.map((region) => h("option", {value: region.id, selected: region.id === layer.region_id},
              `${region.label} · ${region.position}`))) : null,
          ...((spec.kind !== "camera") ? [["intensity", 1, 0.05, 1], ...(layer.region_id ? [["feather", 32, 1, 1]] : [])] : [])
            .map(([key, max, step, fallback]) => h("label", {class: "pass-range"}, key,
              h("input", {type: "range", min: 0, max, step, value: layer[key] ?? fallback, disabled: locked,
                onchange: (event) => replaceLayer(index, {[key]: Number(event.target.value)})}))),
          button("↑", () => swap(index, index - 1), locked || index === 0),
          button("↓", () => swap(index, index + 1), locked || index === draft.length - 1),
          button("Remove", () => { draft.splice(index, 1); render(true); }, locked));
      }),
      h("div", {class: "pass-actions"},
        button("Add effect", () => { draft.push({preset: "warm_glow", mode: FULL_FRAME}); render(true); }, locked || draft.length >= state.boot.video.enhancement_max_layers),
        button("Save shot effects", async () => { if (await api("shot", {...rowKey, effects: draft})) dialog().close(); }, locked),
        button("Reset to planned effects", async () => { if (await api("shot", {...rowKey, reset: true})) dialog().close(); }, locked)),
      shot.regions.length && state.boot.gpu ? h("div", {class: "pass-actions"}, "Refine region",
        ...shot.regions.map((region) => button(`${region.label} · ${region.position}`,
          () => { refine = {region: region.id, type: "include", include: [], exclude: []}; render(true); }, locked))) : null,
      refine ? maskEditor(shot) : null,
    ];
  }

  function render(force = false) {
    const modal = dialog();
    if (!modal?.open) return;
    if (project !== state.project.name || !panel()) { modal.close(); return; }
    if (!force && modal.contains(document.activeElement) && document.activeElement.matches("textarea")) return;
    const shot = panel().shots.find((item) => item.shot_id === selected);
    const run = state.videoRun;
    const running = run?.active && ["segment_effects", "enhance_effects", "refine_effect_mask"].includes(run.kind);
    const scroll = modal.querySelector(".style-body")?.scrollTop ?? 0;
    modal.replaceChildren(h("div", {class: "library-head"},
      h("h2", {id: "effect-pass-title"}, mode === "edit" && shot ? `Shot ${shot.index + 1} effects` : "Enhance visual effects"),
      h("span", {class: "spacer"}),
      running ? button("Interrupt", interrupt) : null,
      button("Close", () => modal.close())),
    h("div", {class: "style-body pass-body"},
      running ? h("p", {role: "status"}, run.status) : null,
      ...(mode === "edit" && shot ? edit(shot) : batch())));
    modal.querySelector(".style-body").scrollTop = scroll;
  }

  return {open, render};
}
