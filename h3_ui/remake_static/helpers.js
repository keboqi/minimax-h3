// Pure UI, storage, and timing helpers for the Remake standalone page.

export const API = "/remake/api";
export const POLL_MS = 1500;
export const PULSE_MS = 2500;
export const CHECK_DELAY_MS = 200;
export const DRAG_PX = 4;
export const MIN_SELECTION_S = 0.5;
export const PREFS = "h3-remake-prefs";
export const SESSION = {project: "h3-remake-project", window: "h3-remake-window"};
export const BRIDGE = {edit: (model) => `remake-edit-bridge-${model}`, settings: "remake-settings-bridge"};
export const QUEUED = ["pending", "changed", "failed"];

export const CHECK = '<svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M2 6.5l2.8 2.8L10 3.5"/></svg>';
export const CROSS = '<svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="M2.5 2.5l7 7M9.5 2.5l-7 7"/></svg>';
export const CHEVRON = '<svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="M2 4l4 4 4-4"/></svg>';
export const PLAY_SVG = '<svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor" aria-hidden="true"><path d="M3 1.5v11l9-5.5z"/></svg>';
export const PAUSE_SVG = '<svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor" aria-hidden="true"><path d="M3 2h3v10H3zM8 2h3v10H8z"/></svg>';
export const VOLUME_SVG = '<svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><path d="M2 5.2h2.6L8 2.5v9L4.6 8.8H2z" fill="currentColor"/><path d="M10 4.5a3.5 3.5 0 0 1 0 5M11.6 2.8a6 6 0 0 1 0 8.4" stroke-linecap="round"/></svg>';
export const MUTE_SVG = '<svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><path d="M2 5.2h2.6L8 2.5v9L4.6 8.8H2z" fill="currentColor"/><path d="M9.8 5l3 4M12.8 5l-3 4" stroke-linecap="round"/></svg>';

export const svg = (markup) => {
  const holder = document.createElement("span");
  holder.innerHTML = markup;
  return holder.firstChild;
};

export const stored = {
  get(key) { try { return sessionStorage.getItem(key); } catch { return null; } },
  set(key, value) { try { sessionStorage.setItem(key, value); } catch { /* Private windows refuse it. */ } },
  remove(key) { try { sessionStorage.removeItem(key); } catch { /* Private windows refuse it. */ } },
};

export const loadPrefs = () => {
  try { return JSON.parse(localStorage.getItem(PREFS)) || {}; } catch { return {}; }
};

export const formatClock = (frames, fpsValue) => {
  const seconds = frames / fpsValue;
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${(seconds % 60).toFixed(1).padStart(4, "0")}`;
};

export const fileUrl = (path) => `${API}/file?path=${encodeURIComponent(path)}`;
