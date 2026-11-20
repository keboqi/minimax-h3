"use strict";
// Shared DOM, transport, toast, and control primitives for standalone workspace pages.
(() => {
  const $ = (id) => document.getElementById(id);

  const h = (tag, props = {}, ...children) => {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props)) {
      if (value === undefined || value === null || value === false) continue;
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
      else if (key in node && key !== "list") node[key] = value;
      else node.setAttribute(key, value === true ? "" : value);
    }
    node.append(...children.flat(Infinity).filter((child) => child !== null && child !== undefined && child !== false));
    return node;
  };

  const visibleArea = () => {
    const frame = window.frameElement;
    if (!frame) return {top: 0, bottom: innerHeight};
    try {
      const box = frame.getBoundingClientRect();
      return {top: Math.max(0, -box.top), bottom: Math.min(innerHeight, window.parent.innerHeight - box.top)};
    } catch {
      return {top: 0, bottom: innerHeight};
    }
  };

  const visibleBottom = () => visibleArea().bottom;

  const createToast = ({id = "toast", duration = 4000, errorDuration = 8000} = {}) => {
    let toastTimer;
    const toast = (message, error = false) => {
      const node = $(id);
      if (!node) return;
      node.textContent = message;
      node.className = error ? "toast error" : "toast";
      if (node.style) node.style.bottom = `${Math.max(20, innerHeight - visibleBottom() + 20)}px`;
      node.hidden = false;
      clearTimeout(toastTimer);
      toastTimer = setTimeout(() => { node.hidden = true; }, error ? errorDuration : duration);
    };
    const guard = (action) => async (...args) => {
      try { return await action(...args); } catch (error) { toast(error.message, true); return undefined; }
    };
    return {toast, guard};
  };

  const createClient = (apiBase) => {
    const request = async (method, path, body, {form = false, keepalive = false} = {}) => {
      const options = {method, credentials: "same-origin", ...(keepalive ? {keepalive: true} : {})};
      if (form) options.body = body;
      else if (body !== undefined) {
        options.headers = {"Content-Type": "application/json"};
        options.body = JSON.stringify(body);
      }
      const response = await fetch(`${apiBase}${path}`, options);
      let data = null;
      try { data = await response.json(); } catch { /* A proxy's error page has no JSON. */ }
      if (!response.ok) throw new Error(data?.error || data?.detail || `The server answered ${response.status}.`);
      return data;
    };
    const post = (path, body, extra) => request("POST", path, body, extra);
    return {request, post};
  };

  // Keep buttons in place while their labels match so a poll redraw between press and release preserves the click.
  const seg = (node, options, current, pick, {disabled = false, multi = false} = {}) => {
    const items = options.map((option) => Array.isArray(option) ? option : [String(option), option]);
    const children = node.children;
    const kept = children && children.length === items.length
      && items.every(([label], index) => children[index].textContent === label);
    if (!kept) {
      node.replaceChildren(...items.map(([label, value]) => {
        const on = multi ? current.includes(value) : value === current;
        return h("button", {type: "button", "aria-pressed": String(on), disabled, text: label, onclick: () => pick(value, on)});
      }));
      return node;
    }
    items.forEach(([, value], index) => {
      const on = multi ? current.includes(value) : value === current;
      const button = children[index];
      button.setAttribute("aria-pressed", String(on));
      button.disabled = disabled;
      button.onclick = () => pick(value, on);
    });
    return node;
  };

  const multiSeg = (node, options, current, pick, {disabled = false, onEmpty} = {}) => {
    const selected = Array.isArray(current) ? current : [];
    return seg(node, options, selected, (value, pressed) => {
      const next = pressed ? selected.filter((item) => item !== value) : [...selected, value];
      if (!next.length && onEmpty) return onEmpty();
      pick(next);
    }, {disabled, multi: true});
  };

  const rangeOf = (field) => field.minimum === null || field.minimum === undefined ? "" : `${field.minimum} – ${field.maximum}`;

  const presetText = (field, {booleanLabels = true} = {}) => {
    if (Array.isArray(field.preset)) return field.preset.join(", ");
    const found = field.choices?.find(([, choice]) => choice === field.preset);
    if (found) return found[0];
    if (booleanLabels && field.preset === true) return "On";
    if (booleanLabels && field.preset === false) return "Off";
    return String(field.preset ?? "blank");
  };

  const openStudio = (task, label, onFallback) => {
    try {
      const page = window.parent.document;
      const find = (selector, text) => [...page.querySelectorAll(selector)].find((item) => item.textContent.trim() === text)?.querySelector("input");
      if (window.parent === window || !page.querySelector(".h3-task-picker")) throw new Error();
      const taskInput = find(".h3-task-picker label", task);
      const create = page.querySelector('#h3-main-tabs [role="tab"][data-tab-id="create"]');
      const needsTask = Boolean(taskInput && (!taskInput.checked || (create && create.getAttribute("aria-selected") !== "true")));
      if (needsTask) taskInput.click();
      const choose = (tries) => {
        const found = find(".h3-engine-bar label", label);
        if (found) { if (!found.checked) found.click(); }
        else if (tries > 0) setTimeout(() => choose(tries - 1), 100);
      };
      if (needsTask) setTimeout(() => choose(50), 50);
      else choose(50);
    } catch {
      onFallback?.(`Open ${label} from the ${task} task.`);
    }
  };

  globalThis.H3PageCommon = {
    $, h, visibleArea, visibleBottom, createToast, createClient,
    seg, multiSeg, rangeOf, presetText, openStudio,
  };
})();
