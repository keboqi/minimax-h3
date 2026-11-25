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
      const ENGINE_TABS = {"MiniMax H3": "h3", "LTX 2.5": "ltx"};
      const TASK_TABS = {"Image": "image", "Song": "song", "Audio / Music": "song", "Remake": "remake",
                         "Storyboard": "storyboard", "Soundstage": "drama3", "TTS": "drama3", "Dubbing": "dub", "Dub": "dub"};
      const targetTab = (task === "Video" && ENGINE_TABS[label]) || TASK_TABS[task] || ENGINE_TABS[label];
      if (task === "Video" && ENGINE_TABS[label]) window.parent.__h3VideoEngine = ENGINE_TABS[label];
      if (targetTab) window.parent.__h3DesiredEngineTab = targetTab;
      const taskInput = find(".h3-task-picker label", task);
      const create = page.querySelector('#h3-main-tabs [role="tab"][data-tab-id="create"]');
      if (create && create.getAttribute("aria-selected") !== "true") create.click();
      const tab = targetTab && page.querySelector(`#h3-engine-tabs [role="tab"][data-tab-id="${targetTab}"]`);
      if (tab && tab.getAttribute("aria-selected") !== "true") tab.click();
      const needsTask = Boolean(taskInput && (!taskInput.checked || (create && create.getAttribute("aria-selected") !== "true")));
      if (needsTask) taskInput.click();
      const choose = (tries) => {
        if (tab && tab.getAttribute("aria-selected") !== "true") tab.click();
        const found = find(".h3-engine-bar label", label);
        if (found && !found.disabled) { if (!found.checked) found.click(); }
        else if (tries > 0) setTimeout(() => choose(tries - 1), 100);
      };
      if (needsTask) setTimeout(() => choose(50), 50);
      else choose(50);
    } catch {
      onFallback?.(`Open ${label} from the ${task} task.`);
    }
  };

  const openSystem = (onFallback) => {
    try {
      const page = window.parent.document;
      if (window.parent === window) throw new Error();
      const tab = page.querySelector('#h3-main-tabs [role="tab"][data-tab-id="system"]');
      if (!tab) throw new Error();
      if (tab.getAttribute("aria-selected") !== "true") tab.click();
    } catch {
      onFallback?.("Open the System tab to set a temporary Gemini API key for this session.");
    }
  };

  const SCORE_FIELDS = [
    ["prompt", "Prompt"],
    ["visuals", "Visuals"],
    ["style", "Style"],
    ["camera", "Camera"],
    ["action", "Action"],
    ["continuity", "Continuity"],
    ["audio", "Audio"],
    ["speech", "Speech"],
    ["references", "Refs"],
    ["advice", "Advice"],
  ];
  const scoreTone = (value) => (value >= 70 ? "good" : value < 45 ? "warn" : "");

  const scoreAxes = (review) => {
    const scores = review?.scores;
    if (!scores || typeof scores !== "object") return [];
    const axes = [];
    for (const [key, label] of SCORE_FIELDS) {
      const raw = scores[key];
      if (raw === undefined || raw === null || raw === "") continue;
      const numeric = Number(raw);
      if (!Number.isFinite(numeric)) continue;
      axes.push({key, label, value: Math.max(0, Math.min(100, Math.round(numeric)))});
    }
    return axes;
  };

  const reviewHeadline = (review, {includeTagsInMeta = false} = {}) => {
    const headline = review.score !== undefined && review.score !== null ? review.score : review.fit;
    const headlineLabel = review.score_label || review.label || "";
    const prefix = review.score !== undefined && review.score !== null ? "Take score" : "Prompt fit";
    const meta = [];
    if (review.ref_count) meta.push(`${review.ref_count} ref${review.ref_count === 1 ? "" : "s"}`);
    if (review.advice) meta.push("with advice");
    if (includeTagsInMeta && review.tags?.length) meta.push(`Tags: ${review.tags.join(", ")}`);
    return {
      headline,
      headlineLabel,
      prefix,
      meta,
      text: `${prefix} ${headline}${headlineLabel ? ` · ${headlineLabel}` : ""}`,
      title: `Scored by Gemini Decision${meta.length ? ` (${meta.join(" · ")})` : ""}`,
    };
  };

  const decisionScoreChips = (review, {chipClass = "chip", includeTags = false, includeBreakdown = true} = {}) => {
    const info = reviewHeadline(review, {includeTagsInMeta: !includeTags});
    const chips = [
      h("span", {
        class: `${chipClass} ${scoreTone(info.headline)}`.trim(),
        title: info.title,
        text: info.text,
      }),
    ];
    if (includeBreakdown && review.scores && typeof review.scores === "object") {
      for (const [key, label] of SCORE_FIELDS) {
        if (review.scores[key] !== undefined && review.scores[key] !== null) {
          chips.push(h("span", {class: `${chipClass} ${scoreTone(review.scores[key])}`.trim(), text: `${label} ${review.scores[key]}`}));
        }
      }
    }
    if (includeTags) {
      for (const tag of review.tags || []) {
        chips.push(h("span", {class: chipClass, title: "Added to Media library tags", text: tag}));
      }
    }
    return chips;
  };

  const SVG_NS = "http://www.w3.org/2000/svg";
  const svgEl = (tag, attrs = {}, ...children) => {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [key, value] of Object.entries(attrs)) {
      if (value === undefined || value === null || value === false) continue;
      node.setAttribute(key, value === true ? "" : String(value));
    }
    for (const child of children.flat(Infinity)) {
      if (child === null || child === undefined || child === false) continue;
      node.append(typeof child === "string" || typeof child === "number" ? document.createTextNode(String(child)) : child);
    }
    return node;
  };

  const decisionScoreRadar = (review, {chipClass = "chip", includeTags = false, includeHeadline = true} = {}) => {
    const axes = scoreAxes(review);
    if (axes.length < 3) return null;
    const info = reviewHeadline(review);
    const summaryParts = axes.map((axis) => `${axis.label} ${axis.value}`);
    const ariaSummary = info.headline !== undefined && info.headline !== null
      ? `${info.prefix} ${info.headline}${info.headlineLabel ? ` (${info.headlineLabel})` : ""}: ${summaryParts.join(", ")}`
      : `Aspect scores: ${summaryParts.join(", ")}`;

    const cx = 190, cy = 138, R = 86, LR = 102;
    const rings = [];
    const ticks = [];
    for (const pct of [20, 40, 60, 80, 100]) {
      const r = (R * pct) / 100;
      rings.push(svgEl("circle", {cx, cy, r: r.toFixed(2), class: pct === 100 ? "radar-ring radar-backdrop" : "radar-ring"}));
      ticks.push(svgEl("text", {x: cx - 4, y: (cy - r + 3).toFixed(2), "text-anchor": "end", class: "radar-tick mono"}, `${pct}%`));
    }

    const spokes = [];
    const polygonPoints = [];
    const axisGroups = [];
    const listItems = [];
    const readoutValue = h("strong", {class: "mono"});
    const readoutLabel = h("span", {class: "small muted", text: "Hover or focus any aspect"});

    const setActive = (activeIndex) => {
      axisGroups.forEach((group, idx) => group.classList?.toggle("is-active", idx === activeIndex));
      listItems.forEach((item, idx) => item.classList?.toggle("is-active", idx === activeIndex));
      if (activeIndex === null || activeIndex === undefined) {
        readoutValue.textContent = "";
        readoutLabel.textContent = "Hover or focus any aspect";
      } else {
        const picked = axes[activeIndex];
        readoutValue.textContent = String(picked.value);
        readoutLabel.textContent = picked.label;
      }
    };

    axes.forEach((axis, index) => {
      const angle = -Math.PI / 2 + (2 * Math.PI * index) / axes.length;
      const cos = Math.cos(angle);
      const sin = Math.sin(angle);
      spokes.push(svgEl("line", {
        x1: cx, y1: cy, x2: (cx + R * cos).toFixed(2), y2: (cy + R * sin).toFixed(2), class: "radar-spoke",
      }));
      const vr = (R * axis.value) / 100;
      const vx = cx + vr * cos;
      const vy = cy + vr * sin;
      polygonPoints.push(`${vx.toFixed(2)},${vy.toFixed(2)}`);

      const lx = cx + LR * cos;
      const ly = cy + LR * sin;
      const anchor = Math.abs(cos) < 0.2 ? "middle" : cos > 0 ? "start" : "end";
      const baseline = sin < -0.7 ? "auto" : sin > 0.7 ? "hanging" : "middle";

      const labelEl = svgEl("text", {
        x: lx.toFixed(2), y: ly.toFixed(2), "text-anchor": anchor, "dominant-baseline": baseline, class: "radar-label",
      }, axis.label);
      const dotEl = svgEl("circle", {cx: vx.toFixed(2), cy: vy.toFixed(2), r: "4.5", class: "radar-dot"});
      const hitEl = svgEl("circle", {
        cx: vx.toFixed(2), cy: vy.toFixed(2), r: "14", class: "radar-hit",
        tabindex: "0", role: "graphics-symbol", "aria-label": `${axis.label}: ${axis.value} out of 100`,
      });
      const group = svgEl("g", {class: "radar-axis"}, labelEl, dotEl, hitEl);

      const activate = () => setActive(index);
      const deactivate = () => setActive(null);
      hitEl.addEventListener("pointerenter", activate);
      hitEl.addEventListener("pointerleave", deactivate);
      hitEl.addEventListener("focus", activate);
      hitEl.addEventListener("blur", deactivate);
      labelEl.addEventListener("pointerenter", activate);
      labelEl.addEventListener("pointerleave", deactivate);
      axisGroups.push(group);

      const tone = scoreTone(axis.value);
      const item = h("div", {
        class: "score-radar-item", role: "listitem", tabIndex: 0,
        onpointerenter: activate, onpointerleave: deactivate, onfocus: activate, onblur: deactivate,
      },
      h("span", {class: `score-radar-dot ${tone}`.trim(), "aria-hidden": "true"}),
      h("span", {class: "score-radar-name", text: axis.label}),
      h("span", {class: "score-radar-val mono", text: String(axis.value)}));
      listItems.push(item);
    });

    const svg = svgEl("svg", {
      viewBox: "0 0 380 276", class: "score-radar-svg", role: "img", "aria-label": ariaSummary,
    },
    ...rings,
    ...spokes,
    ...ticks,
    svgEl("polygon", {points: polygonPoints.join(" "), class: "radar-area"}),
    svgEl("circle", {cx, cy, r: "2.5", class: "radar-origin"}),
    ...axisGroups);

    const sideChildren = [];
    if (includeHeadline && info.headline !== undefined && info.headline !== null) {
      sideChildren.push(h("div", {class: "score-radar-head"},
        h("span", {
          class: `${chipClass} ${scoreTone(info.headline)}`.trim(),
          title: info.title,
          text: info.text,
        }),
        info.meta.length ? h("span", {class: "small muted", text: info.meta.join(" · ")}) : null));
    }
    sideChildren.push(
      h("div", {class: "score-radar-readout", "aria-live": "polite"}, readoutValue, readoutLabel),
      h("div", {class: "score-radar-list", role: "list"}, ...listItems),
    );
    if (includeTags && review.tags?.length) {
      sideChildren.push(h("div", {class: "score-radar-tags"},
        review.tags.map((tag) => h("span", {class: chipClass, title: "Added to Media library tags", text: tag}))));
    }

    return h("div", {class: "score-radar-card"},
      h("div", {class: "score-radar-plot"}, svg),
      h("div", {class: "score-radar-side"}, ...sideChildren));
  };

  const copyToClipboard = async (text, toast, label = "Finetuned prompt copied.") => {
    try {
      await navigator.clipboard.writeText(text);
      toast?.(label);
    } catch {
      toast?.("Could not copy to clipboard.", true);
    }
  };

  const llmReviewNodes = (llm, {
    chipClass = "chip",
    headClass = "row between",
    rowClass = "row tightrow",
    listClass = "llm-review-list",
    issueListClass = "llm-review-list llm-review-issues",
    promptClass = "llm-review-prompt",
    primaryBtnClass = "btn primary",
    secondaryBtnClass = "btn",
    onApply,
    onCopy,
  } = {}) => {
    const metaChips = [
      llm.model ? h("span", {class: chipClass, text: llm.model}) : null,
      llm.ref_count ? h("span", {class: chipClass, text: `${llm.ref_count} ref${llm.ref_count === 1 ? "" : "s"}`}) : null,
      llm.advice ? h("span", {class: chipClass, title: llm.advice, text: "with advice"}) : null,
    ].filter(Boolean);
    const sections = [
      h("div", {class: headClass}, h("strong", {class: "small", text: "Gemini review"}), h("div", {class: rowClass}, ...metaChips)),
      h("p", {class: "small", text: llm.summary || ""}),
    ];
    if (llm.strengths?.length) {
      sections.push(
        h("strong", {class: "small muted", text: "What worked"}),
        h("ul", {class: listClass}, llm.strengths.map((item) => h("li", {text: item}))),
      );
    }
    if (llm.issues?.length) {
      sections.push(
        h("strong", {class: "small muted", text: "Issues spotted"}),
        h("ul", {class: issueListClass}, llm.issues.map((issue) => h("li", {},
          h("span", {class: `${chipClass} ${issue.severity === "major" ? "warn" : ""}`.trim(), text: `${issue.area || "prompt"} · ${issue.severity || "minor"}`}),
          issue.timestamp ? h("span", {class: `${chipClass} mono`, text: issue.timestamp}) : null,
          h("span", {text: issue.detail || ""}),
        ))),
      );
    }
    if (llm.prompt_changes?.length) {
      sections.push(
        h("strong", {class: "small muted", text: "Prompt adjustments"}),
        h("ul", {class: listClass}, llm.prompt_changes.map((item) => h("li", {text: item}))),
      );
    }
    if (llm.finetuned_prompt) {
      sections.push(
        h("div", {class: promptClass},
          h("strong", {class: "small", text: "Finetuned prompt"}),
          h("pre", {text: llm.finetuned_prompt}),
          h("div", {class: rowClass},
            h("button", {type: "button", class: primaryBtnClass, text: "Apply finetuned prompt", onclick: () => onApply?.(llm.finetuned_prompt)}),
            h("button", {type: "button", class: secondaryBtnClass, text: "Copy prompt", onclick: () => onCopy?.(llm.finetuned_prompt)}),
          )),
      );
    }
    return sections;
  };

  globalThis.H3PageCommon = {
    $, h, visibleArea, visibleBottom, createToast, createClient,
    seg, multiSeg, rangeOf, presetText, openStudio, openSystem,
    scoreTone, scoreAxes, decisionScoreChips, decisionScoreRadar, llmReviewNodes, copyToClipboard,
  };
})();
