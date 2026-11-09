// Small helpers every screen shares: escaping, formatting, icons and the status toast.

export const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => (
  {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));

// Handler notes mark emphasis as **text**; everything else stays plain.
export const rich = (value) => esc(value).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/\n+/g, " ");

export const clock = (seconds, tenths = true) => {
  const total = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(total / 60);
  const rest = total - minutes * 60;
  return tenths ? `${minutes}:${rest.toFixed(1).padStart(4, "0")}` : `${minutes}:${String(Math.floor(rest)).padStart(2, "0")}`;
};

export const size = (bytes) => {
  if (!bytes && bytes !== 0) return "";
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
};

export const plural = (count, word) => `${count} ${word}${count === 1 ? "" : "s"}`;

export const ranges = (indices) => {
  const sorted = [...new Set(indices)].sort((a, b) => a - b);
  const parts = [];
  for (const index of sorted) {
    const last = parts.at(-1);
    if (last && index === last[1] + 1) last[1] = index; else parts.push([index, index]);
  }
  return parts.map(([low, high]) => (low === high ? `${low + 1}` : `${low + 1}-${high + 1}`)).join(", ");
};

const PATHS = {
  play: '<path d="M7 4v16l13-8z" fill="currentColor" stroke="none"></path>',
  check: '<path d="M5 12l5 5 9-10"></path>',
  prev: '<path d="M15 6l-6 6 6 6"></path>',
  next: '<path d="M9 6l6 6-6 6"></path>',
  speaker: '<path d="M11 5L6 9H3v6h3l5 4V5z"></path>',
  download: '<path d="M12 4v12M7 11l5 5 5-5M5 20h14"></path>',
  search: '<circle cx="11" cy="11" r="7"></circle><path d="M20 20l-4-4"></path>',
  swap: '<path d="M7 7h13l-4-4M17 17H4l4 4"></path>',
  warn: '<path d="M12 3l9 16H3zM12 10v4M12 17v.5"></path>',
  link: '<path d="M10 14a4 4 0 005.7 0l3-3a4 4 0 00-5.7-5.7l-1 1M14 10a4 4 0 00-5.7 0l-3 3a4 4 0 005.7 5.7l1-1"></path>',
  chevron: '<path d="M9 6l6 6-6 6"></path>',
};

export const icon = (name, px = 16) => `<svg width="${px}" height="${px}" viewBox="0 0 24 24" fill="none"
  stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${PATHS[name] || ""}</svg>`;

let toastTimer = null;
export function toast(message, {error = false} = {}) {
  const box = document.getElementById("toast");
  if (!message) {
    box.hidden = true;
    return;
  }
  box.textContent = message;
  box.classList.toggle("error", error);
  box.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { box.hidden = true; }, error ? 8000 : 4000);
}

export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (_) {
    const area = document.createElement("textarea");
    area.value = text;
    document.body.append(area);
    area.select();
    const done = document.execCommand("copy");
    area.remove();
    return done;
  }
}

// A segmented control: options are [value, label] pairs.
export const seg = (name, options, current, {label = "", disabled = false} = {}) => `
  <div class="seg" role="group" aria-label="${esc(label || name)}">${options.map(([value, text]) => `
    <button type="button" data-seg="${esc(name)}" data-value="${esc(value)}" aria-pressed="${String(value) === String(current)}"
      ${disabled ? "disabled" : ""}>${esc(text)}</button>`).join("")}</div>`;
