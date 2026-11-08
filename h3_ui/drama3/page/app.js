// The Soundstage page: a saved project's mix, its lines and takes, played and reviewed in the browser.
// Playing, seeking and picking a line ask the server for nothing; speaking runs as a job the page polls.
(() => {
  "use strict";
  const API = "/soundstage/api";
  const POLL_MS = 1500;
  const LAST_PROJECT = "h3-soundstage-project";
  const FILTERS = [["all", "All"], ["new", "New"], ["reused", "Reused"], ["flagged", "Flagged"]];
  const STATUS_LABELS = {new: "New", reused: "Reused", flagged: "Flagged"};
  const $ = (id) => document.getElementById(id);
  const mix = $("mix-audio");
  const takeAudio = $("take-audio");

  const state = {boot: null, project: null, picked: 0, filter: "all", run: null, poll: null, busy: false};

  // -- Helpers ---------------------------------------------------------------
  function h(tag, attrs = {}, ...children) {
    const element = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
      if (value === undefined || value === null || value === false) continue;
      if (key === "class") element.className = value;
      else if (key === "style") {
        for (const [name, setting] of Object.entries(value)) {
          if (name.startsWith("--")) element.style.setProperty(name, String(setting));
          else element.style[name] = setting;
        }
      }
      else if (key.startsWith("on")) element.addEventListener(key.slice(2), value);
      else if (key in element && typeof value !== "string") element[key] = value;
      else element.setAttribute(key, value === true ? "" : value);
    }
    for (const child of children.flat()) {
      if (child === null || child === undefined || child === false) continue;
      element.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return element;
  }

  function svg(path, size = 16, fill = false) {
    const ns = "http://www.w3.org/2000/svg";
    const icon = document.createElementNS(ns, "svg");
    icon.setAttribute("width", size);
    icon.setAttribute("height", size);
    icon.setAttribute("viewBox", "0 0 24 24");
    icon.setAttribute("aria-hidden", "true");
    icon.setAttribute("fill", fill ? "currentColor" : "none");
    if (!fill) {
      icon.setAttribute("stroke", "currentColor");
      icon.setAttribute("stroke-width", "2");
      icon.setAttribute("stroke-linecap", "round");
    }
    const shape = document.createElementNS(ns, "path");
    shape.setAttribute("d", path);
    icon.append(shape);
    return icon;
  }
  const ICONS = {
    play: "M7 4v16l13-8z",
    pause: "M6 4h4v16H6zM14 4h4v16h-4z",
    prev: "m15 6-6 6 6 6",
    next: "m9 6 6 6-6 6",
  };

  function clock(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) seconds = 0;
    const whole = Math.floor(seconds);
    return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
  }

  async function api(path, options = {}) {
    const init = {...options, headers: {"Content-Type": "application/json", ...(options.headers || {})}};
    if (init.body && typeof init.body !== "string") init.body = JSON.stringify(init.body);
    const response = await fetch(`${API}${path}`, init);
    let data = {};
    try { data = await response.json(); } catch { /* Not JSON: an empty or failed reply. */ }
    if (!response.ok || data.error) throw new Error(data.error || data.detail || `Request failed (${response.status}).`);
    return data;
  }
  const named = (path) => `${path}${path.includes("?") ? "&" : "?"}name=${encodeURIComponent(state.project.name)}`;

  let toastTimer = null;
  function toast(message) {
    const box = $("toast");
    box.textContent = message;
    box.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { box.hidden = true; }, 5000);
  }

  const toneColor = (tone) => `var(--tone-${tone})`;
  function speakerTone(name) {
    const speaker = state.project?.result?.speakers.find((item) => item.name === name);
    return speaker ? speaker.tone : "narrator";
  }

  // -- Derived data ----------------------------------------------------------
  /** Each subtitle line's take: the take whose span holds the line's start. */
  function lineTakes(result) {
    return result.lines.map((line) => result.takes.find((take) =>
      take.start !== null && take.start !== undefined && take.start - 0.01 <= line.start && line.start < take.end)
      ?? result.takes.find((take) => take.speaker === line.speaker && take.text.includes(line.text)) ?? null);
  }

  /** Rows so that clips playing at the same time stack instead of overlapping. */
  function rows(spans) {
    const ends = [];
    const placed = spans.map(() => 0);
    [...spans.keys()].sort((a, b) => spans[a].start - spans[b].start).forEach((index) => {
      const {start, end} = spans[index];
      let row = ends.findIndex((last) => last <= start + 0.01);
      if (row < 0) { row = ends.length; ends.push(end); } else { ends[row] = end; }
      placed[index] = row;
    });
    return {placed, count: Math.max(ends.length, 1)};
  }

  // -- Rendering -------------------------------------------------------------
  function renderHeader() {
    const select = $("project");
    select.replaceChildren(...state.boot.projects.map((project) =>
      h("option", {value: project.name, selected: project.name === state.project.name},
        project.name + (project.generated ? "" : " · Draft"))));
    $("project-state").textContent = state.project.result ? "Generated" : "Draft";
    $("mode").textContent = `${state.project.script_mode} script`;
    const gpu = $("gpu");
    gpu.dataset.ready = String(state.boot.gpu);
    gpu.textContent = state.boot.gpu ? "GPU ready" : "No GPU here";
  }

  function renderSteps() {
    const project = state.project;
    $("script-summary").textContent = project.lines ? `${project.lines} lines` : "No script yet";
    $("step-script").dataset.state = project.lines ? "done" : "needs";
    const missing = project.cast.filter((speaker) => !speaker.ready).length;
    const sounds = project.sounds.length;
    const silent = project.sounds.filter((sound) => !sound.ready).length;
    const parts = [missing ? `${missing} voice${missing > 1 ? "s" : ""} missing`
                           : `${project.cast.length} voice${project.cast.length === 1 ? "" : "s"}`];
    if (sounds) parts.push(silent ? `${silent} sound${silent > 1 ? "s" : ""} to make` : `${sounds} sounds`);
    $("cast-summary").textContent = parts.join(" · ");
    $("step-cast").dataset.state = missing ? "needs" : "done";
    const result = project.result;
    $("mix-summary").textContent = result ? `${clock(result.duration)} · ${result.takes.length} takes` : "Not spoken yet";
  }

  function renderPlayer() {
    const box = $("player");
    const result = state.project.result;
    if (!result) {
      box.replaceChildren(h("div", {class: "not-spoken"},
        h("strong", {}, "Not spoken yet"),
        h("p", {class: "muted"}, "Speak renders the project as last saved in the Soundstage tab. "
          + "Its timeline, lines and takes appear here.")));
      return;
    }
    const playButton = h("button", {type: "button", class: "play", id: "play", "aria-label": "Play",
                                    onclick: () => (mix.paused ? mix.play() : mix.pause())}, svg(ICONS.play, 18, true));
    const head = h("div", {class: "player-head"},
      playButton,
      h("div", {class: "player-title"},
        h("strong", {}, result.title || state.project.title || state.project.name),
        h("span", {class: "muted small num", id: "clock"}, `0:00 / ${clock(result.duration)}`)),
      h("span", {class: "spacer"}),
      h("a", {class: "btn btn-small", href: `${API}${named("/audio")}`, download: result.file}, "Download"));

    const duration = result.duration || Math.max(0, ...result.lines.map((line) => line.end));
    const place = (start, end) => ({left: `${Math.min(start / duration, 1) * 100}%`,
                                    width: `${Math.max(end - start, 0) / duration * 100}%`});
    const voiceRows = rows(result.lines);
    const voices = h("div", {class: "lane", style: {"--rows": voiceRows.count}},
      result.lines.map((line, index) => h("button", {
        type: "button", class: "clip", "data-line": index,
        title: `${line.speaker}: ${line.text}`, "aria-label": `Play from ${line.speaker}: ${line.text}`,
        style: {...place(line.start, line.end), "--row": voiceRows.placed[index], background: toneColor(speakerTone(line.speaker))},
        onclick: (event) => { event.stopPropagation(); pick(index, {seek: true}); },
      })));
    const laneHeight = (count) => ({height: `${count * 18 + 6}px`});
    const lanes = [h("span", {class: "lane-name", style: laneHeight(voiceRows.count)}, "Voices")];
    const tracks = [voices];
    if (result.sounds.length) {
      const soundRows = rows(result.sounds);
      lanes.push(h("span", {class: "lane-name", style: laneHeight(soundRows.count)}, "Sounds"));
      tracks.push(h("div", {class: "lane", style: {"--rows": soundRows.count}},
        result.sounds.map((sound, index) => h("button", {
          type: "button", class: `clip ${sound.loop ? "bed" : "cue"}`,
          title: `${sound.name} (${sound.loop ? "looped" : "one-shot"})`, "aria-label": `Play from sound: ${sound.name}`,
          style: {...place(sound.start, sound.end), "--row": soundRows.placed[index]},
          onclick: (event) => { event.stopPropagation(); seek(sound.start); },
        }))));
    }
    const ticks = Math.min(6, Math.max(3, Math.floor(duration / 10)));
    const ruler = h("div", {class: "ruler num", "aria-hidden": "true"},
      Array.from({length: ticks}, (_, index) => {
        const at = duration * index / ticks;
        return h("span", {style: {left: `${at / duration * 100}%`}}, clock(at));
      }));
    const lanesBox = h("div", {class: "lanes", id: "lanes"}, tracks, h("span", {class: "playhead", id: "playhead", hidden: true}));
    // Clicking an empty spot on a lane seeks there.
    lanesBox.addEventListener("click", (event) => {
      const box = lanesBox.getBoundingClientRect();
      seek((event.clientX - box.left) / box.width * duration);
    });
    const timeline = h("div", {class: "timeline", role: "group", "aria-label": "Voices and sounds on the timeline"},
      h("span"), ruler, h("div", {class: "lane-names"}, lanes), lanesBox);
    const legend = h("div", {class: "legend"},
      result.speakers.map((speaker) => h("span", {}, h("i", {style: {background: toneColor(speaker.tone)}}), speaker.name)),
      result.sounds.length ? [h("span", {}, h("i", {style: {background: "var(--bed)"}}), "Looped bed"),
                              h("span", {}, h("i", {style: {background: "var(--cue)"}}), "One-shot effect")] : null,
      h("span", {class: "spacer"}),
      h("span", {}, "Click a clip or a line to jump there"));
    box.replaceChildren(head, timeline, legend);
  }

  function wordNodes(line) {
    if (!line.words) return [line.text];
    return [line.words.lead, ...line.words.items.map((word) =>
      h("span", {class: "word", "data-start": word.start, "data-end": word.end}, word.text))];
  }

  function renderFilters() {
    const result = state.project.result;
    const takes = result ? lineTakes(result) : [];
    const count = (status) => takes.filter((take) => take?.status === status).length;
    $("filters").replaceChildren(...FILTERS.map(([id, label]) => {
      const total = id === "all" ? takes.length : count(id);
      return h("button", {type: "button", "aria-pressed": String(state.filter === id), disabled: id !== "all" && !total,
                          onclick: () => { state.filter = id; renderFilters(); renderLines(); }},
               `${label} ${total}`);
    }));
  }

  function renderLines() {
    const list = $("lines");
    const result = state.project.result;
    if (!result || !result.lines.length) {
      list.replaceChildren(h("li", {class: "lines-empty"}, result
        ? "This mix was made before line timings were recorded. Remix it to get its lines."
        : "Lines appear here once the project is spoken."));
      return;
    }
    const takes = lineTakes(result);
    list.replaceChildren(...result.lines.map((line, index) => {
      const take = takes[index];
      if (state.filter !== "all" && take?.status !== state.filter) return null;
      return h("li", {},
        h("button", {type: "button", class: "line", "data-line": index, "aria-pressed": String(index === state.picked),
                     onclick: () => pick(index, {seek: true})},
          h("span", {class: "line-number num"}, index + 1),
          h("span", {class: "line-speaker", style: {color: toneColor(speakerTone(line.speaker))}}, line.speaker),
          h("span", {class: "line-text"}, wordNodes(line)),
          take ? h("span", {class: `status status-${take.status}`}, STATUS_LABELS[take.status]) : h("span")));
    }).filter(Boolean));
  }

  function renderInspector() {
    const box = $("inspector");
    const result = state.project.result;
    if (!result || !result.lines.length) {
      box.replaceChildren(h("h2", {}, "Picked line"),
        h("p", {class: "muted small"}, "Pick a line to hear its take before staging and effects, or to make a new take."));
      return;
    }
    const line = result.lines[state.picked];
    const take = lineTakes(result)[state.picked];
    const step = (by) => () => pick(Math.max(0, Math.min(result.lines.length - 1, state.picked + by)), {seek: true});
    const playTake = h("button", {type: "button", class: "btn btn-icon", id: "take-play", "aria-label": "Play this take",
                                  disabled: !take, onclick: () => toggleTake(take)}, svg(ICONS.play, 14, true));
    box.replaceChildren(...[
      h("div", {class: "inspector-head"},
        h("span", {}, `Line ${state.picked + 1}`), h("span", {"aria-hidden": "true"}, "·"),
        h("strong", {style: {color: toneColor(speakerTone(line.speaker))}}, line.speaker),
        h("span", {class: "spacer"}),
        h("button", {type: "button", class: "btn btn-icon", "aria-label": "Previous line", onclick: step(-1),
                     disabled: state.picked === 0}, svg(ICONS.prev)),
        h("button", {type: "button", class: "btn btn-icon", "aria-label": "Next line", onclick: step(1),
                     disabled: state.picked >= result.lines.length - 1}, svg(ICONS.next))),
      h("p", {class: "inspector-text"}, line.text),
      take ? h("div", {class: "take"},
        playTake,
        h("div", {class: "take-bar"}, h("span", {id: "take-progress"})),
        h("span", {class: "muted small num"}, take.seconds ? `${take.seconds} s` : ""),
        h("span", {class: `status status-${take.status}`}, STATUS_LABELS[take.status]))
        : h("p", {class: "muted small"}, "This line has no saved take."),
      take && take.text !== line.text ? h("p", {class: "muted small"}, `Spoken in one take with: “${take.text}”`) : null,
      take?.warnings.length ? h("ul", {class: "warnings"}, take.warnings.map((warning) => h("li", {}, warning))) : null,
      h("div", {class: "inspector-actions"},
        h("button", {type: "button", class: "btn btn-primary", id: "regenerate", disabled: !take || !state.boot.gpu || isRunning(),
                     onclick: () => speak("regenerate", take.key)}, "Regenerate line"),
        h("button", {type: "button", class: "btn", onclick: () => seek(line.start, true)}, "Play in mix")),
      h("p", {class: "muted small"}, "Regenerate makes a new take of this line, then remixes.")].filter(Boolean));
  }

  function renderSettings() {
    const settings = state.project.settings;
    const names = Object.keys(state.boot.presets);
    const buttons = names.map((name) => h("button", {
      type: "button", "aria-pressed": String(settings.preset === name),
      onclick: () => saveSettings({temperature: state.boot.presets[name][0], top_p: state.boot.presets[name][1],
                                   top_k: state.boot.presets[name][2]}),
    }, name));
    if (!settings.preset) buttons.push(h("button", {type: "button", "aria-pressed": "true", disabled: true}, "Custom"));
    $("presets").replaceChildren(...buttons);
    $("word-timing").checked = settings.word_timing;
  }

  function isRunning() { return Boolean(state.run?.active); }

  function renderActions() {
    const project = state.project;
    const result = project.result;
    const running = isRunning();
    const title = $("action-title");
    const detail = $("action-detail");
    detail.classList.remove("error");
    if (running) {
      title.textContent = "Speaking…";
      detail.textContent = state.run.status || "Waiting for the GPU…";
    } else if (state.run?.error) {
      title.textContent = "Speaking stopped";
      detail.textContent = state.run.error;
      detail.classList.add("error");
    } else if (!state.boot.gpu) {
      title.textContent = "Speaking needs the GPU";
      detail.textContent = "Open Soundstage in Studio to speak, remix or regenerate lines.";
    } else {
      const missing = project.cast.filter((speaker) => !speaker.ready).length;
      if (!result) {
        title.textContent = missing ? `${missing} speaker${missing > 1 ? "s have" : " has"} no voice yet` : "Ready to speak";
        detail.textContent = missing ? "Give them voices in the Soundstage tab's Cast & sounds, save the project, then speak."
                                     : "Speak renders every line with its speaker's voice, then mixes in the sounds.";
      } else {
        const takes = result.takes;
        const flagged = takes.filter((take) => take.status === "flagged").length;
        title.textContent = flagged ? `${flagged} take${flagged > 1 ? "s" : ""} flagged` : "Mix ready";
        detail.textContent = "Speak again reuses unchanged lines from the project as last saved. "
          + "Remix only rebuilds sounds, staging and effects.";
      }
    }
    $("speak").textContent = result ? "Speak again" : "Speak";
    $("speak").disabled = running || !state.boot.gpu;
    $("remix").disabled = running || !state.boot.gpu || !result;
    $("remix").hidden = !result;
    $("interrupt").hidden = !running;
    const regenerate = $("regenerate");
    if (regenerate) regenerate.disabled = running || !state.boot.gpu || !lineTakes(result)[state.picked];
  }

  function renderAll() {
    renderHeader();
    renderSteps();
    renderPlayer();
    renderFilters();
    renderLines();
    renderInspector();
    renderSettings();
    renderActions();
    sync();
  }

  // -- Playback --------------------------------------------------------------
  function seek(time, play = false) {
    if (!state.project?.result) return;
    mix.currentTime = Math.max(0, time);
    if (play) mix.play();
    sync();
  }

  function pick(index, {seek: andSeek = false} = {}) {
    const result = state.project.result;
    if (!result || !result.lines[index]) return;
    state.picked = index;
    takeAudio.pause();
    for (const button of $("lines").querySelectorAll(".line")) {
      button.setAttribute("aria-pressed", String(Number(button.dataset.line) === index));
    }
    renderInspector();
    renderActions();
    if (andSeek) seek(result.lines[index].start);
  }

  function toggleTake(take) {
    const url = `${API}${named(`/take?key=${encodeURIComponent(take.key)}`)}`;
    if (!takeAudio.paused && takeAudio.dataset.key === take.key) { takeAudio.pause(); return; }
    mix.pause();
    if (takeAudio.dataset.key !== take.key) {
      takeAudio.src = url;
      takeAudio.dataset.key = take.key;
    }
    takeAudio.play().catch((error) => toast(`The take could not play: ${error.message}`));
  }

  let lastActive = -1;
  function sync() {
    const result = state.project?.result;
    if (!result) return;
    const time = mix.currentTime || 0;
    const duration = result.duration || mix.duration || 0;
    const playhead = $("playhead");
    if (playhead && duration > 0) {
      playhead.hidden = false;
      playhead.style.left = `${Math.min(100, time / duration * 100)}%`;
    }
    const clockText = $("clock");
    if (clockText) clockText.textContent = `${clock(time)} / ${clock(duration)}`;
    const play = $("play");
    if (play) {
      const playing = !mix.paused;
      if (play.dataset.playing !== String(playing)) {
        play.dataset.playing = String(playing);
        play.setAttribute("aria-label", playing ? "Pause" : "Play");
        play.replaceChildren(svg(playing ? ICONS.pause : ICONS.play, 18, true));
      }
    }
    // Overlapping speech lights together; during a gap the last line started stays lit.
    const started = result.lines.map((line, index) => ({line, index})).filter(({line}) => line.start <= time + 0.05);
    const playing = started.filter(({line}) => line.end > time);
    const lit = new Set((playing.length ? playing : started.slice(-1)).map(({index}) => index));
    const active = Math.max(-1, ...lit);
    for (const button of $("lines").querySelectorAll(".line")) {
      const index = Number(button.dataset.line);
      const line = result.lines[index];
      button.classList.toggle("playing", lit.has(index));
      button.classList.toggle("past", line.end <= time && !lit.has(index));
      for (const word of button.querySelectorAll(".word")) {
        const start = Number(word.dataset.start), end = Number(word.dataset.end);
        word.classList.toggle("spoken", lit.has(index) && start <= time);
        word.classList.toggle("current", lit.has(index) && start <= time && time < end + 0.08);
      }
    }
    for (const clip of document.querySelectorAll("#lanes .clip[data-line]")) {
      clip.classList.toggle("active", lit.has(Number(clip.dataset.line)));
    }
    if (active >= 0 && active !== lastActive && !mix.paused && $("follow").checked) {
      const row = $("lines").querySelector(`.line[data-line="${active}"]`);
      const list = $("lines");
      if (row) list.scrollTo({top: row.offsetTop - list.offsetTop - (list.clientHeight - row.offsetHeight) / 2, behavior: "smooth"});
    }
    lastActive = active;
    const progress = $("take-progress");
    if (progress) {
      const current = lineTakes(result)[state.picked];
      const own = current && takeAudio.dataset.key === current.key && takeAudio.duration;
      progress.style.width = own ? `${takeAudio.currentTime / takeAudio.duration * 100}%` : "0";
      const button = $("take-play");
      const playing = own && !takeAudio.paused;
      if (button && button.dataset.playing !== String(Boolean(playing))) {
        button.dataset.playing = String(Boolean(playing));
        button.setAttribute("aria-label", playing ? "Pause this take" : "Play this take");
        button.replaceChildren(svg(playing ? ICONS.pause : ICONS.play, 14, true));
      }
    }
  }

  let following = false;
  function follow() {
    if (following) return;
    following = true;
    const frame = () => {
      sync();
      following = !mix.paused || !takeAudio.paused;
      if (following) requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }
  for (const element of [mix, takeAudio]) {
    element.addEventListener("play", () => {
      (element === mix ? takeAudio : mix).pause();
      follow();
    });
    for (const event of ["pause", "seeked", "loadedmetadata", "ended"]) element.addEventListener(event, sync);
  }

  // -- Projects and runs -----------------------------------------------------
  async function loadProject(name, {keepPick = false} = {}) {
    const project = await api(`/project?name=${encodeURIComponent(name)}`);
    const changed = state.project?.name !== project.name || state.project?.result?.version !== project.result?.version;
    state.project = project;
    state.run = project.run;
    if (!keepPick || !project.result || state.picked >= project.result.lines.length) state.picked = 0;
    try { localStorage.setItem(LAST_PROJECT, name); } catch { /* Storage may be blocked. */ }
    if (changed) {
      mix.pause();
      takeAudio.pause();
      takeAudio.removeAttribute("src");
      delete takeAudio.dataset.key;
      if (project.result) {
        mix.src = `${API}/audio?name=${encodeURIComponent(name)}&v=${project.result.version}`;
      } else {
        mix.removeAttribute("src");
      }
      lastActive = -1;
    }
    renderAll();
    if (isRunning()) poll();
  }

  async function saveSettings(values) {
    try {
      const project = await api(named("/project/settings"), {method: "PATCH", body: values});
      state.project.settings = project.settings;
      renderSettings();
    } catch (error) {
      toast(error.message);
    }
  }

  async function speak(action, line) {
    if (isRunning()) return;
    try {
      const data = await api(named("/speak"), {method: "POST", body: {action, line}});
      state.run = data.run;
      renderActions();
      renderInspector();
      poll();
    } catch (error) {
      toast(error.message);
    }
  }

  function poll() {
    clearTimeout(state.poll);
    state.poll = setTimeout(async () => {
      const name = state.project.name;
      try {
        const {run} = await api(named("/run"));
        if (name !== state.project.name) return;
        const before = state.run;
        state.run = run;
        if (run && !run.active) {
          if (run.error) toast(`Speaking stopped: ${run.error}`);
          else if (before?.active) toast("Speech ready.");
          await loadProject(name, {keepPick: true});
          return;
        }
        renderActions();
        if (run?.active) poll();
      } catch (error) {
        toast(error.message);
        poll();
      }
    }, POLL_MS);
  }

  async function interruptRun() {
    try {
      const data = await api(named("/interrupt"), {method: "POST"});
      toast(data.message);
      state.run = data.run;
      renderActions();
    } catch (error) {
      toast(error.message);
    }
  }

  // -- Start -----------------------------------------------------------------
  function applyTheme() {
    let dark = matchMedia("(prefers-color-scheme: dark)").matches;
    try {
      if (window.parent !== window) dark = window.parent.document.body.classList.contains("dark");
    } catch { /* Framed by another origin. */ }
    document.documentElement.dataset.theme = dark ? "dark" : "light";
  }

  function showEmpty() {
    $("loading").hidden = true;
    $("empty").hidden = false;
    $("empty").replaceChildren(
      h("h1", {}, "No Soundstage projects yet"),
      h("p", {}, "Write a script and cast it in the Soundstage tab, then click Save project. "
        + "Saved projects open here to speak, play and review."),
      h("p", {}, h("a", {href: "/", target: "_top"}, "Open the studio")));
  }

  async function start() {
    applyTheme();
    matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", applyTheme);
    $("project").addEventListener("change", (event) => loadProject(event.target.value).catch((error) => toast(error.message)));
    $("speak").addEventListener("click", () => speak("generate"));
    $("remix").addEventListener("click", () => speak("remix"));
    $("interrupt").addEventListener("click", interruptRun);
    $("word-timing").addEventListener("change", (event) => saveSettings({word_timing: event.target.checked}));
    document.addEventListener("keydown", (event) => {
      if (event.target.closest("input, select, textarea, button") || !state.project?.result) return;
      if (event.key === " ") { event.preventDefault(); if (mix.paused) mix.play(); else mix.pause(); }
    });
    try {
      state.boot = await api("/boot");
    } catch (error) {
      $("loading").textContent = `Soundstage could not open: ${error.message}`;
      return;
    }
    const projects = state.boot.projects;
    if (!projects.length) { showEmpty(); return; }
    let last = null;
    try { last = localStorage.getItem(LAST_PROJECT); } catch { /* Storage may be blocked. */ }
    const first = projects.find((project) => project.name === last) ?? projects.find((project) => project.generated) ?? projects[0];
    try {
      await loadProject(first.name);
    } catch (error) {
      $("loading").textContent = `Soundstage could not open ${first.name}: ${error.message}`;
      return;
    }
    $("loading").hidden = true;
    $("app").hidden = false;
  }

  start();
})();
