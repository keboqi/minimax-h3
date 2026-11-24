import {clearEffectPreview, paintDepthVisualization, paintEffectPreview, preloadEffectDepths} from "./effects.js";

const CATEGORIES = [
  {id: "camera", label: "Camera movement", scene: "Forest bridge"},
  {id: "atmosphere", label: "Atmosphere", scene: "Twilight street"},
  {id: "lighting", label: "Lighting & framing", scene: "Lantern-lit cabin"},
];
export const EFFECT_CATEGORIES = CATEGORIES;
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

function paintCardPreview(overlay, still, id, time) {
  const effect = id ? [id] : [];
  paintEffectPreview(overlay, still, effect, time, DURATION);
  if (id === "vignette") overlay.style.opacity = String(Math.min(1, time / 0.6));
}

function cardPlayback(kind, id) {
  return {loops: (kind === "camera" && id !== "subtle_sway") || id === "vignette", once: FLASHING.has(id)};
}

function createHoverController(canPlay) {
  let hover = null;

  function stopHover() {
    if (!hover) return;
    cancelAnimationFrame(hover.frame);
    hover.card.removeAttribute("data-playing");
    hover.paint(hover.posterTime);
    hover = null;
  }

  async function playHover(event, still, paint, posterTime, {loops, once}) {
    if (event.pointerType === "touch" || !canPlay()) return;
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
      if (hover !== preview) return;
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

  return {playHover, stopHover};
}

export function openEffectsPicker({h, dialog, row, orientation, presets, depthUrl = null, depthMeta = null, openEnhanceEditor = null, save, restoreFocus}) {
  const initial = row.effects || [];
  const selected = new Map(initial.map((id) => [presets[id]?.kind, id]));
  const choices = Object.entries(presets);
  let category = CATEGORIES.find(({id}) => selected.has(id)) || CATEGORIES[0];
  let saving = false;
  let frame = null;
  let sourceReady = false;
  let sourceRequest = 0;
  let showDepth = false;
  let depthPainted = false;
  const motion = window.matchMedia?.("(prefers-reduced-motion: reduce)");

  const title = h("h2", {id: "effects-title"}, `Visual effects · Shot ${row.index + 1}`);
  const enhanceBtn = openEnhanceEditor ? h("button", {
    type: "button",
    class: "btn btn-small pass-shot-enhance-btn",
    onclick: () => {
      dialog.close();
      openEnhanceEditor();
    },
  }, "Enhance & layers…") : null;
  const close = h("button", {type: "button", class: "btn btn-icon", "aria-label": "Close",
    onclick: () => dialog.close()}, "×");
  const chips = h("div", {class: "effect-summary-chips", "aria-live": "polite"});
  const image = h("img", {alt: ""});
  const canvas = h("canvas", {width: 640, height: 360, "aria-hidden": "true"});
  const depthCanvas = depthUrl ? h("canvas", {class: "pass-depth-overlay", "aria-hidden": "true"}) : null;
  const stage = h("div", {class: "effect-preview-frame", "data-show-depth": "false"}, image, canvas, depthCanvas);
  const depthToggle = depthUrl ? h("button", {type: "button", class: "btn btn-small pass-seg-toggle",
    "aria-pressed": "false", onclick: toggleDepth}, "Show depth") : null;
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
  const {playHover, stopHover} = createHoverController(() => hoverToggle.checked && !saving);
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
    if (sourceReady) {
      const isShot = source.value === "shot";
      if (!isShot) clearEffectPreview(canvas);
      paintEffectPreview(canvas, image, effects(), clamped, DURATION, isShot ? {
        useWebGPU: true,
        shotIndex: row.index,
        depthUrl,
        depthMeta,
        onReady: () => { if (dialog.open && sourceReady && source.value === "shot") setTime(Number(scrubber.value)); },
      } : null);
    }
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

  function syncDepthOverlay() {
    if (!depthToggle || !depthCanvas) return;
    const isShot = source.value === "shot";
    const visible = Boolean(showDepth && isShot);
    stage.dataset.showDepth = String(visible);
    depthToggle.disabled = saving || !isShot;
    depthToggle.setAttribute("aria-pressed", String(visible));
    depthToggle.textContent = visible ? "Hide depth" : "Show depth";
    if (visible && !depthPainted) {
      depthPainted = true;
      paintDepthVisualization(depthCanvas, depthUrl);
    }
  }

  function toggleDepth() {
    showDepth = !showDepth;
    syncDepthOverlay();
  }

  function updateControls() {
    const active = effects();
    const unchanged = active.length === initial.length && active.every((id) => initial.includes(id));
    play.disabled = saving || !sourceReady || !active.length;
    scrubber.disabled = saving || !sourceReady || !active.length;
    reset.disabled = saving || unchanged;
    apply.disabled = saving || unchanged;
    syncDepthOverlay();
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
      const posterTime = previewTime(id ? [id] : []);
      const paint = (time) => paintCardPreview(overlay, still, id, time);
      const playback = cardPlayback(kind, id);
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
    h("div", {class: "library-head"}, h("div", {class: "style-head-text"}, title), enhanceBtn, close),
    h("div", {class: "style-body effects-picker-body"},
      h("section", {class: "effect-preview", "aria-label": "Combined preview"},
        h("div", {class: "effect-preview-controls"},
          h("label", {class: "field"}, h("span", {class: "label"}, "Preview source"), source),
          depthToggle),
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
    clearEffectPreview(canvas);
    sourceRequest++;
    motion?.removeEventListener("change", updateMotion);
    restoreFocus();
  };
  if (depthUrl) {
    preloadEffectDepths([{depth_url: depthUrl}]).then(() => {
      if (dialog.open && sourceReady && source.value === "shot") setTime(Number(scrubber.value));
    });
  }
  renderChoices();
  renderSelection();
  loadPreviewSource();
  dialog.showModal();
}

export function openEffectPresetPopup({
  h,
  dialog,
  presets,
  title,
  subtitle = "",
  currentPreset = "",
  usedPresets = new Set(),
  blockedKinds = {},
  shotImage = "",
  initialCategory = "",
  onSelect,
  restoreFocus = () => {},
}) {
  const choices = Object.entries(presets);
  const blockedReasons = {...blockedKinds};
  const defaultCategory = () => presets[currentPreset]?.kind
    || CATEGORIES.find(({id}) => !blockedReasons[id])?.id
    || "atmosphere";
  let activeCategory = initialCategory === "all" || CATEGORIES.some(({id}) => id === initialCategory)
    ? initialCategory
    : defaultCategory();
  let useShot = false;
  let picked = null;
  const motion = window.matchMedia?.("(prefers-reduced-motion: reduce)");

  const titleEl = h("h2", {id: "effect-preset-title"}, title);
  const close = h("button", {type: "button", class: "btn btn-icon", "aria-label": "Close",
    onclick: () => dialog.close()}, "×");
  const hoverToggle = h("input", {type: "checkbox", checked: true,
    onchange: () => { if (!hoverToggle.checked) stopHover(); }});
  const {playHover, stopHover} = createHoverController(() => hoverToggle.checked);
  const categoryButtons = [{id: "all", label: "All"}, ...CATEGORIES].map(({id, label}) =>
    h("button", {
      type: "button",
      "data-kind": id,
      "aria-pressed": String(activeCategory === id),
      onclick: () => {
        if (activeCategory === id) return;
        activeCategory = id;
        renderGrid();
      },
    }, label));
  const bgButtons = shotImage
    ? [
      h("button", {type: "button", "data-bg": "demo", "aria-pressed": "true", onclick: () => {
        if (!useShot) return;
        useShot = false;
        renderGrid();
      }}, "Demo scenes"),
      h("button", {type: "button", "data-bg": "shot", "aria-pressed": "false", onclick: () => {
        if (useShot) return;
        useShot = true;
        renderGrid();
      }}, "This shot"),
    ]
    : null;
  const content = h("div", {class: "effect-preset-content"});

  function renderCategoryNotice(kind) {
    if (!blockedReasons[kind]) return null;
    return h("p", {class: "effect-preset-note"}, blockedReasons[kind]);
  }

  function renderCard([id, preset]) {
    const isCurrent = id === currentPreset;
    const isUsed = !isCurrent && usedPresets.has(id);
    const disabled = Boolean(blockedReasons[preset.kind]);
    const still = h("img", {src: useShot && shotImage ? shotImage : demo(preset.kind), alt: "", loading: "lazy"});
    const overlay = h("canvas", {width: 256, height: 144, "aria-hidden": "true"});
    const posterTime = previewTime([id]);
    const paint = (time) => paintCardPreview(overlay, still, id, time);
    paint(posterTime);
    const badges = isCurrent || isUsed
      ? h("span", {class: "effect-preset-badges", "aria-hidden": "true"},
        isCurrent ? h("span", {class: "effect-preset-badge is-current"}, "Current") : null,
        isUsed ? h("span", {class: "effect-preset-badge"}, "In stack") : null)
      : null;
    const playback = cardPlayback(preset.kind, id);
    return h("button", {
      type: "button",
      class: `style-card effect-choice pass-preset-card pass-kind-${preset.kind}`,
      "data-effect": id,
      "aria-label": preset.label,
      "aria-description": preset.description,
      "aria-pressed": String(isCurrent),
      disabled,
      onpointerenter: disabled ? null : (event) => playHover(event, still, paint, posterTime, playback),
      onpointerleave: disabled ? null : stopHover,
      onpointercancel: disabled ? null : stopHover,
      onclick: () => {
        if (disabled) return;
        stopHover();
        picked = id;
        dialog.close();
        onSelect(id);
      },
    },
    h("span", {
      class: useShot && shotImage ? "effect-preview-frame effect-preview-shot" : "effect-preview-frame",
      "aria-hidden": "true",
    }, still, overlay, badges),
    h("strong", {class: "style-name"}, preset.label),
    h("span", {class: "muted small"}, preset.description));
  }

  function renderGrid() {
    stopHover();
    for (const button of categoryButtons) {
      button.setAttribute("aria-pressed", String(button.dataset.kind === activeCategory));
    }
    if (bgButtons) {
      bgButtons[0].setAttribute("aria-pressed", String(!useShot));
      bgButtons[1].setAttribute("aria-pressed", String(useShot));
    }
    if (activeCategory === "all") {
      content.replaceChildren(...CATEGORIES.map((cat) => h("section", {class: "style-group"},
        h("h4", {}, cat.label),
        renderCategoryNotice(cat.id),
        h("div", {class: "effect-preset-grid", role: "group", "aria-label": cat.label},
          ...choices.filter(([, preset]) => preset.kind === cat.id).map(renderCard)))));
      return;
    }
    const cat = CATEGORIES.find(({id}) => id === activeCategory) || CATEGORIES[0];
    content.replaceChildren(
      renderCategoryNotice(cat.id),
      h("div", {class: "effect-preset-grid", role: "group", "aria-label": cat.label},
        ...choices.filter(([, preset]) => preset.kind === cat.id).map(renderCard)));
  }

  dialog.replaceChildren(
    h("div", {class: "library-head"},
      h("div", {class: "style-head-text"}, titleEl, subtitle ? h("span", {class: "muted small"}, subtitle) : null),
      close),
    h("div", {class: "style-body effect-preset-body"},
      h("div", {class: "effect-preset-toolbar"},
        h("div", {class: "seg effect-preset-categories", role: "group", "aria-label": "Effect categories"}, ...categoryButtons),
        h("div", {class: "effect-preset-tools"},
          bgButtons ? h("div", {class: "seg effect-preset-bg", role: "group", "aria-label": "Preview background"}, ...bgButtons) : null,
          h("label", {class: "check"}, hoverToggle, "Hover previews"))),
      content),
    h("div", {class: "drawn-foot"},
      h("span", {class: "muted small"}, "Hover any preview to animate · click to choose"),
      h("span", {class: "spacer"}),
      h("button", {type: "button", class: "btn", onclick: () => dialog.close()}, "Cancel")));

  const updateMotion = (event) => { if (event.matches) stopHover(); };
  motion?.addEventListener("change", updateMotion);
  dialog.onclose = () => {
    stopHover();
    motion?.removeEventListener("change", updateMotion);
    restoreFocus(picked);
  };
  renderGrid();
  if (!dialog.open) dialog.showModal();
}
