// The Script step's two views (code and visual) and the estimated timeline that follows them.
// The timeline and the script locate each other by script line: the preview numbers every line and sound.
export function createScriptView({state, $, h, svg, toneColor, scriptSpeakers, paintEditor, NAME, clock, rows}) {
  const VIEW_KEY = "h3-soundstage-script-view";
  const CONTAINER = /(<\/?(?:parallel|sequential)\b[^<>]*>)/i;
  // Like the speech parser, a sound tag must open and close on one line; one split over lines is read as speech.
  const SOUND = /<sound\b([^>\n]*)>([^\n]*?)<\/sound\s*>/gi;
  const PAUSE = /^<#\s*(-?\d+(?:\.\d+)?)\s*#>$/;
  const SPEAKER_TAG = /^<\|speaker:(\d+)\|>\s*/;
  const EDIT_ICON = "M4 20h4L19 9l-4-4L4 16v4z";
  const TIMELINE_KEY = "h3-soundstage-timeline";
  let flashTimer = 0, visualTimer = 0, outlineTimer = 0, frame = 0;
  let timelineOpen = false;
  try { timelineOpen = localStorage.getItem(TIMELINE_KEY) === "open"; } catch { /* Storage may be blocked. */ }

  try { state.scriptView = localStorage.getItem(VIEW_KEY) === "visual" ? "visual" : "code"; } catch { state.scriptView = "code"; }
  state.scriptLine = 1;

  // -- Reading the script ------------------------------------------------------

  function attrs(text) {
    const out = {};
    for (const match of text.matchAll(/(\w+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))/g)) {
      out[match[1].toLowerCase()] = match[2] ?? match[3] ?? match[4] ?? "";
    }
    return out;
  }

  /** What a line says, as heard: no name, cues, tags, sounds or phonemes. */
  function plain(text) {
    return text.replace(SPEAKER_TAG, "").replace(NAME, "").replace(/<sound\b[^>]*>[\s\S]*?<\/sound\s*>/gi, "")
      .replace(/<[^>]*>/g, " ").replace(/\[[^\]]*\]/g, " ").replace(/\{[^{}]*\}/g, " ")
      .replace(/\s+/g, " ").replace(/\s+([,.!?;:…])/g, "$1").trim();
  }

  /** What a line says with its delivery cues kept apart, for the visual view's cue chips. */
  function spokenParts(text) {
    const kept = text.replace(SPEAKER_TAG, "").replace(NAME, "").replace(/<sound\b[^>]*>[\s\S]*?<\/sound\s*>/gi, "")
      .replace(/<[^>]*>/g, " ").replace(/\{[^{}]*\}/g, " ");
    const out = [];
    const said = (piece) => {
      const words = piece.replace(/\s+/g, " ").replace(/\s+([,.!?;:…])/g, "$1").trim();
      if (words) out.push({text: words});
    };
    let cursor = 0;
    for (const cue of kept.matchAll(/\[([^\]]*)\]/g)) {
      said(kept.slice(cursor, cue.index));
      if (cue[1].trim()) out.push({cue: cue[1].trim()});
      cursor = cue.index + cue[0].length;
    }
    said(kept.slice(cursor));
    return out;
  }

  function soundNode(n, raw, desc) {
    const a = attrs(raw);
    const loop = a.loop === "true";
    const name = a.name || desc.toLowerCase().match(/[a-z0-9]+/g)?.slice(0, 4).join(" ") || "sound";
    const meta = loop
      ? ["loop", a.volume ? `vol ${a.volume}` : ""]
      : [a.duration ? `${a.duration} s` : "", a.pan ? `pan ${a.pan}` : "", a.distance || ""];
    return {type: "sound", n, name, loop, desc: desc.replace(/\s+/g, " ").trim(), meta: meta.filter(Boolean).join(" · ")};
  }

  /** The script as a tree of groups, lines, sounds and pauses, each with the line it starts on. */
  function parse(text) {
    const source = text.replace(/\r\n/g, "\n");
    const sounds = [];
    // Each sound tag becomes a marker that splitting the line keeps whole.
    const body = source.replace(SOUND, (whole, raw, desc, offset) => {
      sounds.push(soundNode(source.slice(0, offset).split("\n").length, raw, desc));
      return `\u0000${sounds.length - 1}\u0000`;
    });
    const root = {type: "group", kind: "sequential", n: 0, end: 0, attrs: {}, children: []};
    const stack = [root];
    let speaker = "", last = 0;
    body.split("\n").forEach((line, index) => {
      const n = index + 1;
      for (let piece of line.split(/(\u0000\d+\u0000|<\/?(?:parallel|sequential)\b[^<>]*>)/i)) {
        piece = piece.trim();
        if (!piece) continue;
        last = n;
        const top = stack[stack.length - 1];
        const marker = /^\u0000(\d+)\u0000$/.exec(piece);
        if (marker) { top.children.push(sounds[Number(marker[1])]); continue; }
        if (CONTAINER.test(piece) && CONTAINER.exec(piece)[0] === piece) {
          const kind = /parallel/i.test(piece) ? "parallel" : "sequential";
          if (piece[1] !== "/") {
            const group = {type: "group", kind, n, end: n, attrs: attrs(piece), children: []};
            top.children.push(group);
            stack.push(group);
          } else if (stack.length > 1 && top.kind === kind) {
            top.end = n;
            stack.pop();
          }
          continue;
        }
        const pause = PAUSE.exec(piece);
        if (pause) { top.children.push({type: "pause", n, seconds: Number(pause[1])}); continue; }
        const tag = SPEAKER_TAG.exec(piece);
        const named = tag ? null : NAME.exec(piece);
        if (tag) speaker = `Speaker ${Number(tag[1])}`;
        else if (named) speaker = named[2].trim();
        const said = plain(piece);
        if (said) top.children.push({type: "line", n, speaker, text: said, parts: spokenParts(piece)});
      }
    });
    for (const group of stack.slice(1)) group.end = Math.max(group.end, last);
    return root;
  }

  const hasLine = (node) => node.type === "line" || (node.type === "group" && node.children.some(hasLine));
  const isScene = (node) => node.type === "group" && node.kind === "parallel"
    && Boolean(node.attrs.room || node.attrs.stage || node.children.some((child) => child.type === "sound" && child.loop));

  function beds(node, out = []) {
    for (const child of node.children) {
      if (child.type === "sound" && child.loop) out.push(child);
      else if (child.type === "group") beds(child, out);
    }
    return out;
  }

  /** Top-level parallels with a room, a stage or a bed are scenes; what lies between them is loose. */
  function sections(root) {
    const out = [];
    for (const node of root.children) {
      if (isScene(node)) out.push({scene: true, node});
      else if (out.length && !out[out.length - 1].scene) out[out.length - 1].nodes.push(node);
      else out.push({scene: false, nodes: [node]});
    }
    return out;
  }

  // -- Timing from the estimate ------------------------------------------------

  function timed() {
    const estimate = state.estimate;
    if (!estimate) return [];
    return [...estimate.lines, ...estimate.sounds.filter((sound) => !sound.loop)]
      .filter((item) => item.line).sort((a, b) => a.line - b.line || a.start - b.start);
  }

  /** When the script at line ``n`` starts playing: the first timed item on or after it. */
  function startOf(n) {
    const item = timed().find((entry) => entry.line >= n);
    return item ? item.start : state.estimate?.duration ?? 0;
  }

  /** When everything up to line ``n`` has finished. */
  function endOf(n) {
    return Math.max(0, ...timed().filter((entry) => entry.line <= n).map((entry) => entry.end));
  }

  // -- Visual view -------------------------------------------------------------

  function humanize(room) {
    const words = room.replace(/[_-]+/g, " ").trim();
    return words ? words[0].toUpperCase() + words.slice(1) : "";
  }

  function stageText(stage) {
    return stage.split(",").map((part) => {
      const [name, value] = part.split("=").map((bit) => bit.trim());
      if (!name) return "";
      const position = Number(value);
      const side = !Number.isFinite(position) ? "" : position < -0.15 ? "left" : position > 0.15 ? "right" : "centre";
      return side ? `${name} ${side}` : name;
    }).filter(Boolean).join(" · ");
  }

  function cueChip(sound) {
    return h("button", {type: "button", class: `vcue${sound.loop ? " loop" : ""}`, "data-line": sound.n,
                        title: `Line ${sound.n} · double-click to edit it in code`},
      h("span", {class: "vcue-head"}, svg("M11 5 6 9H2v6h4l5 4V5zM15.5 8.5a5 5 0 0 1 0 7", 12),
        h("strong", {}, sound.name), h("span", {class: "vcue-meta"}, sound.meta)),
      sound.desc ? h("span", {class: "vcue-desc"}, sound.desc) : null);
  }

  function lineRow(node, tones, {note = "", cues = []} = {}) {
    const tone = tones.get(node.speaker);
    // Words with the delivery cues as chips between them, as the voice will read them.
    const said = (node.parts || [{text: node.text}]).flatMap((part, index) => [
      index ? " " : null, part.cue ? h("span", {class: "vcue-inline"}, part.cue) : part.text]);
    return h("div", {class: "vrow", "data-row": node.n},
      h("button", {type: "button", class: "vline", "data-line": node.n, title: `Line ${node.n} · double-click to edit it in code`},
        h("span", {class: "vspeaker", style: {color: tone ? toneColor(tone) : ""}}, node.speaker || ""),
        h("span", {class: "vtext"}, note ? h("span", {class: "vnote"}, note) : null, h("span", {class: "vsaid"}, said)),
        h("span", {class: "vtime"}, timed().length ? clock(startOf(node.n)) : "")),
      h("button", {type: "button", class: "vedit", "data-line": node.n, "aria-label": `Edit line ${node.n} in code`,
                   title: "Edit in code"}, svg(EDIT_ICON, 15)),
      h("div", {class: "vcues"}, cues.map(cueChip)));
  }

  function pauseRow(node) {
    return h("button", {type: "button", class: "vpause", "data-line": node.n, title: `Line ${node.n}`},
      h("span", {"aria-hidden": "true"}), `${node.seconds} s pause`, h("span", {"aria-hidden": "true"}));
  }

  function soundRow(node) {
    return h("div", {class: "vrow vrow-sound", "data-row": node.n},
      h("span", {class: "vline vline-empty"}), h("span"), h("div", {class: "vcues"}, cueChip(node)));
  }

  /** Lines, pauses and sounds in order; a parallel with two or more spoken branches becomes a Together block. */
  function renderNodes(nodes, tones, ctx) {
    const out = [];
    for (const node of nodes) {
      if (node.type === "line") {
        out.push(lineRow(node, tones, {note: ctx.note}));
        ctx.note = "";
      } else if (node.type === "pause") {
        if (node.seconds > 0) out.push(pauseRow(node));
        else if (node.seconds < 0) ctx.note = `Cuts in ${Math.abs(node.seconds)} s early`;
      } else if (node.type === "sound") {
        if (!node.loop || !ctx.inScene) out.push(soundRow(node));
      } else if (node.kind === "sequential") {
        out.push(...renderNodes(node.children, tones, ctx));
      } else {
        out.push(...renderParallel(node, tones, ctx));
      }
    }
    return out;
  }

  function renderParallel(group, tones, ctx) {
    const branches = group.children.filter((child) => !(child.type === "sound" && child.loop && ctx.inScene));
    const spoken = branches.filter(hasLine);
    const sounds = branches.filter((child) => child.type === "sound");
    if (spoken.length < 2) {
      // One voice with its sounds: the sounds sit beside that voice's first line.
      const rendered = renderNodes(branches.filter((child) => child.type !== "sound"), tones, ctx);
      const first = rendered.find((element) => element.classList?.contains("vrow") && !element.classList.contains("vrow-sound"));
      if (first) first.querySelector(".vcues").append(...sounds.map(cueChip));
      else rendered.push(...sounds.map(soundRow));
      return rendered;
    }
    const lead = spoken[0];
    const leadSpeaker = firstLine(lead)?.speaker || "the first voice";
    const columns = spoken.map((branch, index) => {
      let children = branch.type === "group" && branch.kind === "sequential" ? [...branch.children] : [branch];
      let delay = 0;
      while (children[0]?.type === "pause") { delay += children[0].seconds; children = children.slice(1); }
      const note = index === 0 ? "" : delay > 0 ? `Starts ${+delay.toFixed(2)} s into ${leadSpeaker}'s line` : `With ${leadSpeaker}`;
      return h("div", {class: "vbranch"}, note ? h("span", {class: "vbranch-note"}, note) : null,
        renderNodes(children, tones, {...ctx, note: ""}));
    });
    return [h("div", {class: "vtogether", "data-row": group.n},
      h("button", {type: "button", class: "vtogether-head", "data-line": group.n, title: `Line ${group.n}`},
        svg("M8 4H5v16h3M16 4h3v16h-3", 14), `Together · ${spoken.length} voices at once`),
      h("div", {class: "vtogether-body"},
        h("div", {class: "vbranches"}, columns),
        sounds.length ? h("div", {class: "vcues"}, sounds.map(cueChip)) : null))];
  }

  function firstLine(node) {
    if (node.type === "line") return node;
    if (node.type !== "group") return null;
    for (const child of node.children) { const found = firstLine(child); if (found) return found; }
    return null;
  }

  function timeRange(from, to) {
    if (!timed().length) return "";
    return `${clock(startOf(from))} – ${clock(Math.max(endOf(to), startOf(from)))}`;
  }

  function renderVisual() {
    clearTimeout(visualTimer);
    const box = $("visual-view");
    if (box.hidden || !state.project) return;  // The first paint after loading draws it.
    const text = $("script-text").value;
    const tones = new Map(scriptSpeakers(text).map((speaker) => [speaker.name, speaker.tone]));
    const root = parse(text);
    const parts = sections(root);
    if (!root.children.length) {
      box.replaceChildren(h("p", {class: "visual-empty muted"}, "Nothing to show yet. Write a line in Code, or let the writer script it."));
      return;
    }
    const scenes = parts.filter((part) => part.scene).length;
    const anySound = /<sound\b/i.test(text);
    const basic = state.project.script.script_mode !== state.boot.advanced;
    const out = [];
    if (basic && anySound) {
      out.push(h("p", {class: "visual-note small"}, "Basic mode speaks the lines only: these sounds and scenes are left out until you switch to Advanced."));
    }
    let sceneNumber = 0;
    parts.forEach((part, index) => {
      if (!part.scene) {
        const label = !scenes ? "" : index === 0 ? "Opening" : index === parts.length - 1 ? "Closing" : "Between scenes";
        out.push(h("section", {class: "vloose"}, label ? h("span", {class: "vloose-label"}, label) : null,
          renderNodes(part.nodes, tones, {inScene: false, note: ""})));
        return;
      }
      sceneNumber += 1;
      const group = part.node;
      const range = timeRange(group.n, group.end);
      const sceneBeds = beds(group);
      out.push(h("article", {class: "vscene"},
        h("button", {type: "button", class: "vscene-head", "data-line": group.n, title: `Line ${group.n} · double-click to edit it in code`},
          h("strong", {}, `Scene ${sceneNumber}`),
          group.attrs.room ? h("span", {class: "chip"}, humanize(group.attrs.room)) : null,
          group.attrs.stage ? h("span", {class: "muted small"}, stageText(group.attrs.stage)) : null,
          h("span", {class: "spacer"}),
          range ? h("span", {class: "muted small num"}, range) : null),
        // The scene's beds run under all of it, so they sit across its top.
        sceneBeds.length ? h("div", {class: "vbeds"}, sceneBeds.map((bed) => h("button", {
          type: "button", class: "vbed", "data-line": bed.n, title: `Line ${bed.n} · double-click to edit it in code`,
          "aria-label": `Bed ${bed.name}, ${bed.meta}, line ${bed.n}${bed.desc ? `: ${bed.desc}` : ""}`,
        }, h("span", {class: "vbed-head"}, svg("M2 12c2-4 4-4 6 0s4 4 6 0 4-4 6 0", 13), h("strong", {}, bed.name),
             bed.meta ? h("span", {class: "vbed-meta"}, bed.meta) : null),
           bed.desc ? h("span", {class: "vbed-desc"}, bed.desc) : null))) : null,
        h("div", {class: "vscene-body"},
          h("div", {class: "vscene-rows"}, renderNodes(group.children, tones, {inScene: true, note: ""})))));
    });
    // Holding the old height while the content is swapped keeps the page from jumping.
    box.style.minHeight = `${box.offsetHeight}px`;
    box.replaceChildren(...out);
    box.style.minHeight = "";
    markPicked();
    updateWindow();
  }

  function queueVisual() {
    clearTimeout(visualTimer);
    if (!$("visual-view").hidden) visualTimer = setTimeout(renderVisual, 150);
    clearTimeout(outlineTimer);
    outlineTimer = setTimeout(renderOutline, 150);
  }

  // -- Outline -----------------------------------------------------------------

  /** The outline's Scenes list: each scene, and the loose lines around them, with where they start. */
  function renderOutline() {
    clearTimeout(outlineTimer);
    const box = $("outline-scenes");
    if (!box || !state.project) return;
    const parts = sections(parse($("script-text").value));
    if (!parts.length) {
      box.replaceChildren(h("p", {class: "rail-empty"}, "Scenes appear as you write."));
      return;
    }
    const scenes = parts.filter((part) => part.scene).length;
    const span = (node) => [node.n, node.type === "group" ? node.end : node.n];
    let sceneNumber = 0;
    const items = parts.map((part, index) => {
      const nodes = part.scene ? [part.node] : part.nodes;
      const from = span(nodes[0])[0];
      const to = Math.max(...nodes.map((node) => span(node)[1]));
      let name;
      if (part.scene) {
        sceneNumber += 1;
        name = `Scene ${sceneNumber}${part.node.attrs.room ? ` · ${humanize(part.node.attrs.room)}` : ""}`;
      } else {
        name = !scenes ? "Script" : index === 0 ? "Opening" : index === parts.length - 1 ? "Closing" : "Between scenes";
      }
      return {from, to, name};
    });
    box.replaceChildren(...items.map((item, index) => {
      // A section owns the blank lines up to the next one, so the caret always lands in some section.
      const until = index + 1 < items.length ? items[index + 1].from - 1 : Number.MAX_SAFE_INTEGER;
      return h("button", {type: "button", class: "rail-scene", "data-from": item.from, "data-to": until,
                          onclick: () => locate(item.from)},
        h("span", {class: "rail-scene-top"}, h("strong", {}, item.name),
          timed().length ? h("span", {}, clock(startOf(item.from))) : null),
        h("small", {}, item.from === item.to ? `Line ${item.from}` : `Lines ${item.from}–${item.to}`));
    }));
    markOutline();
  }

  function markOutline() {
    for (const item of $("outline-scenes").querySelectorAll(".rail-scene")) {
      item.classList.toggle("active", state.scriptLine >= Number(item.dataset.from) && state.scriptLine <= Number(item.dataset.to));
    }
  }

  function markPicked() {
    const box = $("visual-view");
    for (const element of box.querySelectorAll(".picked")) element.classList.remove("picked");
    for (const element of box.querySelectorAll(`[data-line="${state.scriptLine}"]`)) {
      (element.closest(".vrow") && element.classList.contains("vline") ? element.closest(".vrow") : element).classList.add("picked");
    }
  }

  // -- Code view ---------------------------------------------------------------

  function caretLine() {
    const area = $("script-text");
    return area.value.slice(0, area.selectionStart).split("\n").length;
  }

  function lineSpan(n) { return $("script-mirror").querySelector(`[data-line="${n}"]`); }

  function placeBand() {
    const band = $("line-band");
    const span = $("code-view").hidden ? null : lineSpan(state.scriptLine);
    band.hidden = !span;
    if (!span) return;
    band.style.top = `${span.offsetTop}px`;
    band.style.height = `${Math.max(span.offsetHeight, 20)}px`;
  }

  function onCaret() {
    const line = caretLine();
    if (line === state.scriptLine && !$("line-band").hidden) return;
    state.scriptLine = line;
    placeBand();
    markTimeline();
  }

  /** The part of the window the script can be seen in: under the script's sticky head, above the sticky action bar. */
  function viewport() {
    const bar = document.querySelector(".actionbar");
    const head = document.querySelector("#screen-script .script-head");
    return {top: head ? Math.max(0, head.getBoundingClientRect().bottom) : 0,
            bottom: bar ? bar.getBoundingClientRect().top : window.innerHeight};
  }

  /** Scroll the page so ``element`` sits in the middle of what can be seen. */
  function centre(element) {
    const rect = element.getBoundingClientRect();
    const view = viewport();
    const middle = (view.top + view.bottom) / 2;
    // A line taller than the view starts at its top instead.
    const delta = rect.height > view.bottom - view.top - 40 ? rect.top - view.top - 24 : rect.top + rect.height / 2 - middle;
    window.scrollBy({top: delta, behavior: "smooth"});
  }

  /** Open line ``n`` in the code view with the caret at its start. */
  function editLine(n, {flash = true} = {}) {
    if (state.scriptView !== "code") setView("code");
    const area = $("script-text");
    const lines = area.value.split("\n");
    n = Math.min(Math.max(1, n), lines.length);
    const offset = lines.slice(0, n - 1).reduce((sum, line) => sum + line.length + 1, 0);
    area.focus({preventScroll: true});
    area.setSelectionRange(offset, offset);
    state.scriptLine = n;
    placeBand();
    markTimeline();
    const span = lineSpan(n);
    if (span) centre(span);
    if (flash) {
      const band = $("line-band");
      band.classList.remove("flash");
      void band.offsetWidth;  // Restarts the animation.
      band.classList.add("flash");
      clearTimeout(flashTimer);
      flashTimer = setTimeout(() => band.classList.remove("flash"), 1500);
    }
  }

  /** Find line ``n`` in whichever view is showing. */
  function locate(n) {
    if (state.scriptView === "code") { editLine(n); return; }
    state.scriptLine = n;
    const box = $("visual-view");
    let target = null, best = 0;
    for (const element of box.querySelectorAll("[data-line]")) {
      const line = Number(element.dataset.line);
      if (line <= n && line > best) { best = line; target = element; }
    }
    markPicked();
    markTimeline();
    if (target) centre(target);
  }

  function pick(n) {
    state.scriptLine = n;
    markPicked();
    markTimeline();
  }

  // -- View switch -------------------------------------------------------------

  function setView(view) {
    state.scriptView = view;
    try { localStorage.setItem(VIEW_KEY, view); } catch { /* Storage may be blocked. */ }
    const code = view === "code";
    $("code-view").hidden = !code;
    $("visual-view").hidden = code;
    $("code-hint").hidden = !code;
    $("visual-hint").hidden = code;
    for (const button of $("script-view").querySelectorAll("button")) button.setAttribute("aria-pressed", String(button.dataset.view === view));
    if (code) {
      if (state.project) paintEditor();  // A hidden editor can't measure its height, so it is sized when shown.
      placeBand();
      const span = lineSpan(state.scriptLine);
      if (span && state.project) centre(span);
    } else {
      renderVisual();
      const target = $("visual-view").querySelector(`[data-line="${state.scriptLine}"]`);
      if (target) centre(target);
    }
    updateWindow();
  }

  // -- Timeline ----------------------------------------------------------------

  /** The first and last script lines on screen, and whether some of the script is off screen. */
  function visibleLines() {
    const code = state.scriptView === "code";
    const elements = [...(code ? $("script-mirror") : $("visual-view")).querySelectorAll("[data-line]")];
    if (!elements.length) return null;
    const view = viewport();
    let first = 0, last = 0, hidden = false;
    for (const element of elements) {
      const rect = element.getBoundingClientRect();
      if (rect.bottom > view.top + 4 && rect.top < view.bottom - 4) {
        const line = Number(element.dataset.line);
        first = first ? Math.min(first, line) : line;
        last = Math.max(last, line);
      } else {
        hidden = true;
      }
    }
    return first ? {first, last, hidden} : null;
  }

  function updateWindow() {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(() => {
      const marker = $("estimate-window");
      if (!marker) return;
      const range = visibleLines();
      const duration = state.estimate?.duration || 0;
      marker.hidden = !range || !range.hidden || !duration || !timed().length;
      if (marker.hidden) return;
      const start = startOf(range.first);
      const end = Math.max(endOf(range.last), start + 0.5);
      marker.style.left = `${start / duration * 100}%`;
      marker.style.width = `${Math.min(end - start, duration - start) / duration * 100}%`;
    });
  }

  function markTimeline() {
    markOutline();
    const lanes = $("estimate-lanes");
    if (!lanes) return;
    for (const clip of lanes.querySelectorAll(".clip[data-line]")) {
      clip.classList.toggle("active", Number(clip.dataset.line) === state.scriptLine);
    }
    for (const band of $("estimate").querySelectorAll(".scene-band")) {
      band.classList.toggle("active", state.scriptLine >= Number(band.dataset.from) && state.scriptLine <= Number(band.dataset.to));
    }
    const cursor = $("estimate-cursor");
    const duration = state.estimate?.duration || 0;
    cursor.hidden = !duration || !timed().length;
    if (!cursor.hidden) cursor.style.left = `${Math.min(startOf(state.scriptLine), duration) / duration * 100}%`;
  }

  /** The speech line playing at ``seconds``, or the nearest one. */
  function lineAt(seconds) {
    const lines = (state.estimate?.lines || []).filter((line) => line.line);
    const playing = lines.filter((line) => line.start <= seconds && seconds < line.end);
    if (playing.length) return playing[playing.length - 1].line;
    let best = null;
    for (const line of lines) if (!best || Math.abs(line.start - seconds) < Math.abs(best.start - seconds)) best = line;
    return best?.line ?? null;
  }

  function renderEstimate() {
    const box = $("estimate");
    const estimate = state.estimate;
    if (!estimate || !estimate.lines.length) {
      box.replaceChildren(h("div", {class: "estimate-head"},
        h("span", {}, h("strong", {}, "Estimated timeline"), " · add a spoken line to see it")));
      return;
    }
    const duration = estimate.duration || Math.max(...estimate.lines.map((line) => line.end));
    estimate.duration = duration;
    const place = (start, end) => ({left: `${start / duration * 100}%`, width: `${Math.max(end - start, 0) / duration * 100}%`});
    const where = (item) => (item.line ? ` · line ${item.line}` : "");
    const voices = rows(estimate.lines);
    const lanes = [h("div", {class: "lane", style: {"--rows": voices.count}}, estimate.lines.map((line, index) => h("span", {
      class: "clip", title: `${line.speaker}: ${line.text}${where(line)}`, "data-line": line.line || null,
      style: {...place(line.start, line.end), "--row": voices.placed[index], background: toneColor(
        estimate.speakers.find((speaker) => speaker.name === line.speaker)?.tone ?? "narrator")},
    })))];
    if (estimate.sounds.length) {
      const sounds = rows(estimate.sounds);
      lanes.push(h("div", {class: "lane", style: {"--rows": sounds.count}}, estimate.sounds.map((sound, index) => h("span", {
        class: `clip ${sound.loop ? "bed" : "cue"}`, title: `${sound.name}${where(sound)}`, "data-line": sound.line || null,
        style: {...place(sound.start, sound.end), "--row": sounds.placed[index]},
      }))));
    }
    const located = timed().length > 0;
    const scenes = located ? sections(parse($("script-text").value)).filter((part) => part.scene) : [];
    const bands = scenes.length ? h("div", {class: "scene-bands"}, scenes.map((part, index) => {
      const start = startOf(part.node.n), end = Math.max(endOf(part.node.end), start);
      const label = `Scene ${index + 1}${part.node.attrs.room ? ` · ${humanize(part.node.attrs.room)}` : ""}`;
      return h("button", {type: "button", class: "scene-band", "data-from": part.node.n, "data-to": part.node.end,
                          style: place(start, end), title: `Go to ${label}`, onclick: () => locate(part.node.n)}, label);
    })) : null;
    const lanesBox = h("div", {class: "lanes", id: "estimate-lanes"}, lanes,
      h("span", {class: "estimate-window", id: "estimate-window", hidden: true, "aria-hidden": "true"}),
      h("span", {class: "estimate-cursor", id: "estimate-cursor", hidden: true, "aria-hidden": "true"}));
    if (located) {
      lanesBox.classList.add("locating");
      lanesBox.addEventListener("click", (event) => {
        const clip = event.target.closest(".clip[data-line]");
        const rect = lanesBox.getBoundingClientRect();
        const line = clip ? Number(clip.dataset.line) : lineAt((event.clientX - rect.left) / rect.width * duration);
        if (line) locate(line);
      });
    }
    // Closed, the bar keeps one slim voice lane so it covers little of the script; open adds scenes and sounds.
    box.classList.toggle("open", timelineOpen);
    const toggle = h("button", {type: "button", class: "estimate-toggle", "aria-expanded": String(timelineOpen),
                                title: "Show scenes and sounds on the timeline", onclick: () => {
      timelineOpen = !timelineOpen;
      try { localStorage.setItem(TIMELINE_KEY, timelineOpen ? "open" : "closed"); } catch { /* Storage may be blocked. */ }
      box.classList.toggle("open", timelineOpen);
      toggle.setAttribute("aria-expanded", String(timelineOpen));
      updateWindow();
    }}, svg("m6 15 6-6 6 6", 14), "Estimated timeline");
    box.replaceChildren(
      h("div", {class: "estimate-head"},
        h("span", {}, toggle, ` · ${estimate.lines.length} lines, no audio yet`,
          located ? " · click to find it in the script" : ""),
        h("span", {class: "num"}, `≈ ${clock(duration)}`)),
      bands, lanesBox);
    markTimeline();
    updateWindow();
    // Line and scene times come from the estimate, so the views and the outline redraw with it.
    queueVisual();
  }

  // -- Wiring ------------------------------------------------------------------

  /** After the editor repaints: the band follows the caret line and the visual view follows the text. */
  function painted() {
    placeBand();
    queueVisual();
    updateWindow();
  }

  function bind() {
    const area = $("script-text");
    for (const type of ["keyup", "click", "select", "focus"]) area.addEventListener(type, onCaret);
    area.addEventListener("input", onCaret);
    // The script grows with its text and the page scrolls, so the In view window follows the page.
    window.addEventListener("scroll", updateWindow, {passive: true});
    window.addEventListener("resize", () => { placeBand(); updateWindow(); });
    $("script-view").replaceChildren(...[["visual", "Visual", "M3 4h18v6H3zM3 14h12v6H3z"], ["code", "Code", "m8 7-5 5 5 5M16 7l5 5-5 5"]]
      .map(([view, label, icon]) => h("button", {type: "button", "data-view": view, "aria-pressed": String(state.scriptView === view),
                                                 onclick: () => setView(view)}, svg(icon, 15), label)));
    const visual = $("visual-view");
    visual.addEventListener("click", (event) => {
      const edit = event.target.closest(".vedit");
      if (edit) { editLine(Number(edit.dataset.line)); return; }
      const target = event.target.closest("[data-line]");
      if (target) pick(Number(target.dataset.line));
    });
    visual.addEventListener("dblclick", (event) => {
      const target = event.target.closest("[data-line]");
      if (!target) return;
      event.preventDefault();
      window.getSelection()?.removeAllRanges();
      editLine(Number(target.dataset.line));
    });
    setView(state.scriptView);
  }

  return {bind, painted, renderEstimate, renderVisual, editLine, locate, setView, parse, sections, plain};
}
