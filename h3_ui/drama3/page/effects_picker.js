import {paintEffectPreview} from "./effects.js";

const CATEGORIES = [
  {id: "camera", label: "Camera movement", scene: "Forest bridge"},
  {id: "atmosphere", label: "Atmosphere", scene: "Twilight street"},
  {id: "lighting", label: "Lighting & framing", scene: "Lantern-lit cabin"},
];
const DURATION = 3;
const FLASHING = new Set(["lightning", "flash"]);
const demo = (kind) => new URL(`./effect-demo-${kind}.webp`, import.meta.url).href;

function previewTime(effects) {
  if (effects.includes("flash")) return 0.04;
  if (effects.includes("lightning")) return 0.12;
  if (effects.includes("impact_shake")) return 0.07;
  if (effects.includes("rack_focus")) return 0.15;
  return 1.8;
}

export function openEffectsPicker({h, dialog, row, orientation, presets, save, restoreFocus}) {
  const initial = row.effects || [];
  const selected = new Map(initial.map((id) => [presets[id]?.kind, id]));
  const choices = Object.entries(presets);
  let category = CATEGORIES.find(({id}) => selected.has(id)) || CATEGORIES[0];
  let saving = false;
  let frame = null;
  let hover = null;
  let sourceReady = false;
  let sourceRequest = 0;
  const motion = window.matchMedia?.("(prefers-reduced-motion: reduce)");

  const title = h("h2", {id: "effects-title"}, `Visual effects · Shot ${row.index + 1}`);
  const close = h("button", {type: "button", class: "btn btn-icon", "aria-label": "Close",
    onclick: () => dialog.close()}, "×");
  const chips = h("div", {class: "effect-summary-chips", "aria-live": "polite"});
  const image = h("img", {alt: ""});
  const canvas = h("canvas", {width: 640, height: 360, "aria-hidden": "true"});
  const stage = h("div", {class: "effect-preview-frame"}, image, canvas);
  const shotLabel = `Shot ${row.index + 1}`;
  const shotOption = h("option", {value: "shot", selected: Boolean(row.image), disabled: !row.image},
    row.image ? shotLabel : `${shotLabel} · No image yet`);
  const source = h("select", {onchange: loadPreviewSource}, shotOption,
    ...CATEGORIES.map(({id, scene}) => h("option",
      {value: id, selected: !row.image && id === category.id}, `Demo · ${scene}`)));
  const play = h("button", {type: "button", class: "btn btn-small", onclick: playPreview}, "Play preview");
  const scrubber = h("input", {type: "range", min: 0, max: DURATION, step: 0.02, value: 0,
    "aria-label": "Preview position", oninput: scrubPreview});
  const timeLabel = h("span", {class: "mono small muted effect-time"}, `0.0 s / ${DURATION.toFixed(1)} s`);
  const timeline = h("div", {class: "effect-timeline"}, play, scrubber, timeLabel);
  const hint = h("p", {class: "muted small effect-preview-hint"});
  const sourceStatus = h("p", {class: "muted small effect-preview-hint", role: "status"});
  const grid = h("div", {class: "effect-picker-grid", role: "group"});
  const hoverToggle = h("input", {type: "checkbox", checked: true,
    onchange: () => { if (!hoverToggle.checked) stopHover(); }});
  const tabs = CATEGORIES.map((option) => h("button", {type: "button", "data-kind": option.id,
    "aria-label": option.label,
    onclick: () => {
      if (category === option) return;
      category = option;
      renderChoices();
    }}, h("span", {class: "effect-tab-label"}, option.label),
    h("span", {class: "effect-tab-sub", "aria-hidden": "true"}, "None")));
  const reset = h("button", {type: "button", class: "btn", disabled: true, onclick: resetEffects}, "Reset");
  const cancel = h("button", {type: "button", class: "btn", onclick: () => dialog.close()}, "Cancel");
  const apply = h("button", {type: "button", class: "btn btn-primary", disabled: true, onclick: applyEffects}, "Apply effects");
  const error = h("span", {class: "error small", role: "alert"});

  function effects() {
    return CATEGORIES.map(({id}) => selected.get(id)).filter(Boolean);
  }

  function stopPreview() {
    if (frame !== null) cancelAnimationFrame(frame);
    frame = null;
    play.textContent = "Play preview";
  }

  function setTime(seconds) {
    const clamped = Math.min(DURATION, Math.max(0, seconds));
    scrubber.value = String(clamped);
    timeLabel.textContent = `${clamped.toFixed(1)} s / ${DURATION.toFixed(1)} s`;
    if (sourceReady) paintEffectPreview(canvas, image, effects(), clamped, DURATION);
  }

  function scrubPreview() {
    stopPreview();
    setTime(Number(scrubber.value));
  }

  function playPreview() {
    const active = effects();
    if (frame !== null) {
      stopPreview();
      return;
    }
    const start = performance.now();
    const tick = (now) => {
      const elapsed = Math.min(DURATION, Math.max(0, (now - start) / 1000));
      setTime(elapsed);
      if (elapsed < DURATION) frame = requestAnimationFrame(tick);
      else stopPreview();
    };
    play.textContent = "Stop preview";
    setTime(0);
    frame = requestAnimationFrame(tick);
  }

  async function loadPreviewSource() {
    const request = ++sourceRequest;
    const isShot = source.value === "shot";
    const portrait = isShot && orientation === "portrait";
    stopPreview();
    sourceReady = false;
    updateControls();
    sourceStatus.textContent = row.image && shotOption.disabled ? "The shot image couldn’t load. Showing a demo." : "";
    stage.classList.toggle("effect-preview-portrait", portrait);
    stage.classList.toggle("effect-preview-shot", isShot);
    stage.classList.add("effect-preview-loading");
    canvas.width = portrait ? 360 : 640;
    canvas.height = portrait ? 640 : 360;
    image.src = isShot ? row.image : demo(source.value);
    try {
      await image.decode();
    } catch {
      if (request !== sourceRequest) return;
      if (isShot) {
        shotOption.disabled = true;
        shotOption.textContent = `${shotLabel} · Image unavailable`;
        source.value = category.id;
        return loadPreviewSource();
      }
      sourceStatus.textContent = "This demo couldn’t load. Choose another preview source.";
      return;
    }
    if (request !== sourceRequest) return;
    sourceReady = true;
    paintPreview();
    stage.classList.remove("effect-preview-loading");
    updateControls();
  }

  function paintPreview() {
    stopPreview();
    const active = effects();
    setTime(previewTime(active));
  }

  function updateControls() {
    const active = effects();
    const unchanged = active.length === initial.length && active.every((id) => initial.includes(id));
    play.disabled = saving || !sourceReady || !active.length;
    scrubber.disabled = saving || !sourceReady || !active.length;
    reset.disabled = saving || unchanged;
    apply.disabled = saving || unchanged;
  }

  function resetEffects() {
    if (saving) return;
    selected.clear();
    for (const id of initial) {
      if (presets[id]?.kind) selected.set(presets[id].kind, id);
    }
    error.textContent = "";
    renderSelection();
  }

  function renderSelection() {
    const active = effects();
    for (const tab of tabs) {
      const id = selected.get(tab.dataset.kind);
      const sub = tab.querySelector(".effect-tab-sub");
      if (sub) sub.textContent = id && presets[id] ? presets[id].label : "None";
    }
    chips.replaceChildren(...(active.length
      ? active.map((id) => {
        const removeBtn = h("button", {type: "button", class: "effect-chip-remove", "aria-label": `Remove ${presets[id].label}`,
          onclick: () => {
            if (saving) return;
            selected.delete(presets[id].kind);
            error.textContent = "";
            renderSelection();
          }});
        removeBtn.innerHTML = `<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>`;
        return h("span", {class: "chip effect-chip-removable"}, presets[id].label, removeBtn);
      })
      : [h("span", {class: "muted small"}, "No effects selected")]));
    hint.textContent = active.some((id) => FLASHING.has(id)) ? "Playing this preview includes brief flashes." : "";
    markSelection();
    paintPreview();
    updateControls();
  }

  function markSelection() {
    const selectedId = selected.get(category.id) || "";
    for (const card of grid.children) {
      card.setAttribute("aria-pressed", String(card.dataset.effect === selectedId));
    }
  }

  function stopHover() {
    if (!hover) return;
    cancelAnimationFrame(hover.frame);
    hover.card.removeAttribute("data-playing");
    hover.paint(hover.posterTime);
    hover = null;
  }

  async function playHover(event, still, paint, posterTime, {loops, once}) {
    if (event.pointerType === "touch" || !hoverToggle.checked || saving) return;
    stopHover();
    const preview = {card: event.currentTarget, paint, posterTime, frame: 0};
    hover = preview;
    try {
      if (!still.complete || !still.naturalWidth) await still.decode();
    } catch {
      if (hover === preview) stopHover();
      return;
    }
    if (hover !== preview) return;

    const started = performance.now();
    const tick = (now) => {
      const elapsed = Math.max(0, (now - started) / 1000);
      const time = loops ? elapsed % DURATION : elapsed;
      paint(once ? Math.min(DURATION, time) : time);
      if (!once || elapsed < DURATION) preview.frame = requestAnimationFrame(tick);
      else preview.card.removeAttribute("data-playing");
    };
    preview.card.setAttribute("data-playing", "");
    paint(0);
    preview.frame = requestAnimationFrame(tick);
  }

  function renderChoices() {
    stopHover();
    const {id: kind, label} = category;
    const scene = demo(kind);
    for (const tab of tabs) tab.setAttribute("aria-pressed", String(tab.dataset.kind === kind));
    grid.setAttribute("aria-label", label);
    const options = [["", {label: "None", description: `Skip ${label.toLowerCase()} for this shot.`}],
      ...choices.filter(([, preset]) => preset.kind === kind)];
    grid.replaceChildren(...options.map(([id, preset]) => {
      const still = h("img", {src: scene, alt: "", loading: "lazy"});
      const overlay = h("canvas", {width: 256, height: 144, "aria-hidden": "true"});
      const effect = id ? [id] : [];
      const posterTime = previewTime(effect);
      const paint = (time) => {
        paintEffectPreview(overlay, still, effect, time, DURATION);
        if (id === "vignette") overlay.style.opacity = String(Math.min(1, time / 0.6));
      };
      const playback = {loops: (kind === "camera" && id !== "subtle_sway") || id === "vignette", once: FLASHING.has(id)};
      if (id) paint(posterTime);
      return h("button", {type: "button", class: "style-card effect-choice", "data-effect": id,
        "aria-label": preset.label, "aria-description": preset.description,
        onpointerenter: id ? (event) => playHover(event, still, paint, posterTime, playback) : null,
        onpointerleave: id ? stopHover : null,
        onpointercancel: id ? stopHover : null,
        onclick: () => {
          if (saving || (selected.get(kind) || "") === id) return;
          if (id) selected.set(kind, id);
          else selected.delete(kind);
          error.textContent = "";
          renderSelection();
        }},
        h("span", {class: "effect-preview-frame", "aria-hidden": "true"}, still, overlay),
        h("strong", {class: "style-name"}, preset.label),
        h("span", {class: "muted small"}, preset.description));
    }));
    markSelection();
  }

  function setSaving(value) {
    stopPreview();
    stopHover();
    saving = value;
    for (const control of dialog.querySelectorAll("button, select, input")) control.disabled = value;
    apply.textContent = value ? "Saving…" : "Apply effects";
    updateControls();
  }

  async function applyEffects() {
    error.textContent = "";
    setSaving(true);
    if (await save(effects())) {
      dialog.close();
      return;
    }
    setSaving(false);
    error.textContent = "Couldn’t save the visual effects. Try again.";
    apply.focus();
  }

  dialog.replaceChildren(
    h("div", {class: "library-head"}, h("div", {class: "style-head-text"}, title), close),
    h("div", {class: "style-body effects-picker-body"},
      h("section", {class: "effect-preview", "aria-label": "Combined preview"},
        h("div", {class: "effect-preview-controls"},
          h("label", {class: "field"}, h("span", {class: "label"}, "Preview source"), source)),
        stage, timeline, chips, sourceStatus, hint),
      h("section", {class: "effect-options", "aria-label": "Effect options"},
        h("div", {class: "effect-options-controls"},
          h("div", {class: "seg effect-categories", role: "group", "aria-label": "Effect categories"}, ...tabs),
          h("label", {class: "check"}, hoverToggle, "Hover previews")),
        grid)),
    h("div", {class: "drawn-foot"}, error, h("span", {class: "spacer"}), reset, cancel, apply));
  dialog.oncancel = (event) => { if (saving) event.preventDefault(); };
  const updateMotion = (event) => { if (event.matches) stopPreview(); };
  motion?.addEventListener("change", updateMotion);
  dialog.onclose = () => {
    stopPreview();
    stopHover();
    sourceRequest++;
    motion?.removeEventListener("change", updateMotion);
    restoreFocus();
  };
  renderChoices();
  renderSelection();
  loadPreviewSource();
  dialog.showModal();
}
