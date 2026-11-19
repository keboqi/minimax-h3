// Shared viewport-shell chrome, tablists, persistent UI preferences and overview badges for Soundstage.

const LAYOUT_KEY = "h3-soundstage-layout";
const CAST_VIEW_KEY = "h3-soundstage-cast-view";

/** Read a saved UI preference string safely when storage may be blocked or absent. */
export function readPref(key, fallback = null) {
  try {
    const value = globalThis.localStorage?.getItem(key);
    return value ?? fallback;
  } catch {
    return fallback;
  }
}

/** Persist a UI preference string safely when storage may be blocked or absent. */
export function writePref(key, value) {
  try {
    globalThis.localStorage?.setItem(key, String(value));
  } catch { /* Storage may be blocked. */ }
}

/** Wire click and arrow-key navigation for an ARIA tablist whose tabs expose aria-controls. */
export function bindTablist($, tabIds) {
  const tabs = tabIds.map((id) => $(id)).filter(Boolean);
  const select = (tabOrId, focus = false) => {
    const target = typeof tabOrId === "string" ? $(tabOrId) : tabOrId;
    if (!target || !tabs.includes(target)) return;
    for (const tab of tabs) {
      const on = tab === target;
      tab.setAttribute("aria-selected", String(on));
      tab.tabIndex = on ? 0 : -1;
      const controls = tab.getAttribute?.("aria-controls") ?? tab["aria-controls"];
      const panel = controls ? $(controls) : null;
      if (panel) panel.hidden = !on;
    }
    if (focus) target.focus?.();
  };
  for (const tab of tabs) {
    tab.addEventListener("click", () => select(tab));
    tab.addEventListener("keydown", (event) => {
      const index = tabs.indexOf(tab);
      let next = -1;
      if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
      else if (event.key === "ArrowLeft") next = (index - 1 + tabs.length) % tabs.length;
      else if (event.key === "Home") next = 0;
      else if (event.key === "End") next = tabs.length - 1;
      if (next < 0) return;
      event.preventDefault?.();
      select(tabs[next], true);
    });
  }
  return {select};
}

/** Create the Soundstage shell layout controller for column toggles, tabs, cast filter and overviews. */
export function createLayout({$, shortMode, clock}) {
  let mixTablist = null;

  function bindScriptColumns() {
    const screen = $("screen-script");
    const outlineButton = $("toggle-outline");
    const writerButton = $("toggle-writer");
    if (!screen || !outlineButton || !writerButton) return;
    let layout = {outline: true, writer: true};
    try {
      layout = {...layout, ...JSON.parse(readPref(LAYOUT_KEY, "{}") || "{}")};
    } catch { /* Ignore corrupted JSON preferences. */ }
    const apply = () => {
      screen.classList.toggle("hide-outline", !layout.outline);
      screen.classList.toggle("hide-writer", !layout.writer);
      outlineButton.setAttribute("aria-pressed", String(layout.outline));
      writerButton.setAttribute("aria-pressed", String(layout.writer));
    };
    const toggle = (key) => () => {
      layout = {...layout, [key]: !layout[key]};
      writePref(LAYOUT_KEY, JSON.stringify(layout));
      apply();
    };
    outlineButton.addEventListener("click", toggle("outline"));
    writerButton.addEventListener("click", toggle("writer"));
    apply();
  }

  function setCastView(view) {
    const filter = $("cast-view-filter");
    const voices = $("cast-voices-card");
    const sounds = $("cast-sounds-card");
    if (!filter || !voices || !sounds) return;
    const next = view === "voices" || view === "sounds" ? view : "split";
    writePref(CAST_VIEW_KEY, next);
    for (const button of filter.querySelectorAll("button[data-view]")) {
      button.setAttribute("aria-pressed", String(button.dataset.view === next));
    }
    voices.hidden = next === "sounds";
    sounds.hidden = next === "voices";
  }

  function bindCastViewFilter() {
    const filter = $("cast-view-filter");
    if (!filter) return;
    setCastView(readPref(CAST_VIEW_KEY, "split"));
    filter.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-view]");
      if (button) setCastView(button.dataset.view);
    });
  }

  function bind() {
    bindScriptColumns();
    bindTablist($, ["writer-tab-built", "writer-tab-paste"]);
    mixTablist = bindTablist($, ["mix-tab-line", "mix-tab-sound"]);
    bindCastViewFilter();
  }

  function selectMixTab(tabId, focus = false) {
    mixTablist?.select(tabId, focus);
  }

  function ensureCastSection(kind) {
    const target = kind === "sound" ? $("cast-sounds-card") : $("cast-voices-card");
    if (target?.hidden) setCastView("split");
  }

  function renderScriptOverview(project, estimate) {
    if (!project) return;
    const mode = $("overview-mode");
    if (mode) mode.textContent = shortMode(project.script.script_mode);
    const lines = $("overview-lines");
    if (lines) lines.textContent = String(project.lines || 0);
    const assets = $("overview-assets");
    if (assets) {
      const total = project.cast.length;
      const ready = project.cast.filter((speaker) => speaker.ready).length;
      assets.textContent = total ? `${ready}/${total}` : "0/0";
      assets.dataset.state = total && ready === total ? "ready" : "needs";
    }
    const dur = $("overview-duration");
    if (dur) {
      if (estimate?.lines?.length) {
        const duration = estimate.duration || Math.max(...estimate.lines.map((line) => line.end));
        dur.textContent = estimate.actual ? clock(duration) : `≈ ${clock(duration)}`;
      } else {
        dur.textContent = project.result ? clock(project.result.duration) : "—";
      }
    }
  }

  function renderCastOverview(project) {
    if (!project) return;
    const speakers = project.speakers || [];
    const sounds = project.sounds || [];
    const voicesBadge = $("cast-overview-voices");
    if (voicesBadge) {
      const ready = speakers.filter((speaker) => speaker.ready).length;
      voicesBadge.textContent = speakers.length ? `Voices ${ready}/${speakers.length}` : "No voices";
      voicesBadge.dataset.state = speakers.length && ready === speakers.length ? "ready" : "needs";
    }
    const soundsBadge = $("cast-overview-sounds");
    if (soundsBadge) {
      const ready = sounds.filter((sound) => sound.ready).length;
      soundsBadge.textContent = sounds.length ? `Sounds ${ready}/${sounds.length}` : "No sounds";
      soundsBadge.dataset.state = !sounds.length || ready === sounds.length ? "ready" : "needs";
    }
  }

  return {bind, selectMixTab, setCastView, ensureCastSection, renderScriptOverview, renderCastOverview};
}
