import {createScript} from "./script.js";
import {createCast} from "./cast.js";
import {createMix} from "./mix.js";
import {createVideo} from "./video.js";
import {createLibrary} from "./library.js";
import {createProjects} from "./project.js";
import {createApi, createPoller, StaleResponse} from "./transport.js";

// The Soundstage page: a saved project's mix, its lines and takes, played and reviewed in the browser.
// Playing, seeking and picking a line ask the server for nothing; speaking runs as a job the page polls.
export function startWorkspace() {
  "use strict";
  const API = "/soundstage/api";
  const POLL_MS = 1500;
  const LAST_PROJECT = "h3-soundstage-project";
  const FILTERS = [["all", "All"], ["new", "New"], ["reused", "Reused"], ["flagged", "Flagged"]];
  const STATUS_LABELS = {new: "New", reused: "Reused", flagged: "Flagged"};
  const $ = (id) => document.getElementById(id);
  const mixAudio = $("mix-audio");
  const mixVideo = $("mix-video");
  // The player plays the mix, or its illustrated video, which carries the same soundtrack.
  let mix = mixAudio;
  const takeAudio = $("take-audio");

  const state = {boot: null, project: null, picked: 0, filter: "all", run: null, busy: false,
                 screen: "script", saving: null, writing: false, writerResult: null,
                 estimate: null, estimateTimer: null, castRun: null, castPick: null, castSource: null,
                 videoRun: null, mediaMode: "audio"};
  const SCREENS = ["script", "cast", "mix"];
  const shortMode = (mode) => (mode === state.boot.advanced ? "Advanced" : "Basic");

  // -- Helpers ---------------------------------------------------------------
  const NO_GPU = "This needs the GPU: open the project in Studio to run it.";

  function h(tag, attrs = {}, ...children) {
    const element = document.createElement(tag);
    if (attrs["data-gpu"] !== undefined && state.boot && !state.boot.gpu) {
      attrs = {...attrs, disabled: true, title: NO_GPU};
    }
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
      // A textarea ignores a value attribute; every field takes the property.
      else if (key === "value") element.value = value;
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

  let projectEpoch = 0;
  const scope = () => `${state.project?.name || ""}:${projectEpoch}`;
  const api = createApi(API, scope);
  const named = (path, name = state.project.name) => `${path}${path.includes("?") ? "&" : "?"}name=${encodeURIComponent(name)}`;
  const poller = (step, options = {}) => createPoller(step, {
    delay: POLL_MS, scope, failed: (error) => toast(error.message), ...options,
  });

  /** Say how a run that was going ended. */
  const toastEnded = (run) => toast(run?.error ? `Stopped: ${run.error}` : (run?.status || "Done."));

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
    const mode = state.project.script.script_mode;
    $("mode").replaceChildren(...state.boot.script_modes.map((option) => h("button", {
      type: "button", "aria-pressed": String(option === mode),
      title: option === state.boot.advanced ? "Dialogue plus sound effects, music and vocal effects"
                                            : "Dialogue with expression cues; sound tags are kept but ignored",
      onclick: () => option !== mode && saveScript({script_mode: option}, {now: true}),
    }, shortMode(option))));
    const gpu = $("gpu");
    gpu.dataset.ready = String(state.boot.gpu);
    gpu.textContent = state.boot.gpu ? "GPU ready" : "Prepare only";
    gpu.title = state.boot.gpu ? "" : "No GPU here: write, cast and manage voices, then speak the project in Studio.";
  }

  function renderSteps() {
    const project = state.project;
    for (const screen of SCREENS) {
      const step = $(`step-${screen}`);
      if (screen === state.screen) step.setAttribute("aria-current", "step");
      else step.removeAttribute("aria-current");
    }
    $("script-summary").textContent = project.lines ? `${project.lines} lines` : "No script yet";
    $("step-script").dataset.state = project.lines ? "done" : "needs";
    const missing = project.cast.filter((speaker) => !speaker.ready).length;
    const sounds = project.sounds.length;
    const silent = project.sounds.filter((sound) => !sound.ready).length;
    const parts = [!project.cast.length ? "No cast yet"
                   : missing ? `${missing} voice${missing > 1 ? "s" : ""} missing`
                   : `${project.cast.length} voice${project.cast.length === 1 ? "" : "s"}`];
    if (sounds) parts.push(silent ? `${silent} sound${silent > 1 ? "s" : ""} to make` : `${sounds} sounds`);
    $("cast-summary").textContent = parts.join(" · ");
    $("step-cast").dataset.state = missing || !project.cast.length ? "needs" : "done";
    const result = project.result;
    $("mix-summary").textContent = result ? `${clock(result.duration)} · ${result.takes.length} takes` : "Not spoken yet";
  }

  // -- Subtitles on the video: shown or hidden (C), and their size ------------
  function saveVideo(body, {later = false} = {}) {
    saves.queue(state.project.name, "video", body, later ? 700 : 0);
  }
  const flushVideo = () => flushEdits();

  async function videoCall(path, body) {
    try {
      await flushVideo();
      const data = await api(named(path), {method: "POST", body: body ?? {}});
      if (data.video_run) state.videoRun = data.video_run;
      if (data.video) state.project.video = data.video;
      if (data.message) toast(data.message);
      renderVideo();
      if (state.videoRun?.active) pollVideo();
      return data;
    } catch (error) {
      toast(error.message);
      return null;
    }
  }

  const planShots = () => videoCall("/project/video/plan", {writer: writerService()});
  const drawImages = () => videoCall("/project/video/draw");
  async function saveVideoToMedia() {
    await videoCall("/project/video/save");
    renderPlayer();
    sync();  // A new player shows nothing until it is painted at the playhead.
  }
  const interruptVideo = () => videoCall("/project/video/interrupt");

  async function copyPlannerPrompt() {
    try {
      await flushVideo();
      const {prompt} = await api(named("/project/video/prompt"), {method: "POST"});
      await navigator.clipboard.writeText(prompt);
      toast("Planner prompt copied. Paste it into any chat model.");
    } catch (error) {
      toast(`Could not copy the prompt: ${error.message}`);
    }
  }

  async function applyPlannerReply() {
    const data = await videoCall("/project/video/reply", {reply: $("plan-reply")?.value || ""});
    if (data && $("plan-reply")) $("plan-reply").value = "";
  }

  const pollVideo = poller(async (again) => {
    const name = state.project.name;
    const project = await api(named("/project", name));
    if (state.project.name !== name) return;
    const before = state.videoRun;
    const hadStills = state.project.video?.stills?.revision;
    state.videoRun = project.video_run;
    reconcile(project);
    if (project.video_run?.active) { renderVideo(); again(); return; }
    if (before?.active) {
      toastEnded(project.video_run);
      if (before.kind === "draw" && !project.video_run?.error && project.video?.stills?.revision !== hadStills) {
        setMediaMode("video");
        window.scrollTo({top: 0, behavior: "smooth"});
      }
    }
    loadVideoSource();
    renderVideo();
    renderPlayer();
    sync();
  });

  /** Load the rendered video only for a mix that has no stills to slide; otherwise the page never fetches it. */
  function loadVideoSource() {
    const source = visualSource();
    if (source === "file") {
      const latest = state.project.video.latest;
      if (mixVideo.dataset.url !== latest.url) {
        mixVideo.src = latest.url;
        mixVideo.dataset.url = latest.url;
      }
    } else if (mixVideo.dataset.url) {
      mixVideo.pause();
      mixVideo.removeAttribute("src");
      delete mixVideo.dataset.url;
      mixVideo.load();
    }
    // Keep the player on what can show now: the stills play on the mix's audio.
    if (state.mediaMode === "video" && !source) state.mediaMode = "audio";
    const player = state.mediaMode === "video" && source === "file" ? mixVideo : mixAudio;
    if (player !== mix) {
      const time = mix.currentTime || 0;
      mix.pause();
      mix = player;
      try { mix.currentTime = time; } catch { /* Not loaded yet. */ }
    }
  }

  // -- Word timing -----------------------------------------------------------
  function renderWordRepair() {
    const box = $("word-repair");
    const result = state.project.result;
    if (!result || !result.lines.length) { box.replaceChildren(); return; }
    const timed = result.lines.filter((line) => line.words).length;
    const aligning = isRunning() && state.run.task === "words";
    box.replaceChildren(
      h("span", {class: "small"}, h("strong", {}, "Word timing: "),
        timed === result.lines.length ? `every line (${timed})` : `${timed} of ${result.lines.length} lines`),
      h("span", {class: "muted small"}, "Times this mix's words from its saved takes, without speaking or remixing. "
        + "Lines already timed are kept; missing ones are tried again."),
      h("button", {type: "button", class: "btn btn-small", style: {alignSelf: "flex-start"},
                   "data-gpu": true, disabled: isRunning() || !state.boot.gpu, onclick: repairWords},
        aligning ? "Aligning…" : timed ? "Repair word timing" : "Time every word"));
  }

  async function repairWords() {
    if (isRunning()) return;
    try {
      const data = await api(named("/words"), {method: "POST"});
      state.run = data.run;
      renderActions();
      renderWordRepair();
      poll();
    } catch (error) {
      toast(error.message);
    }
  }

  // -- Voice library ---------------------------------------------------------
  // -- Auto ------------------------------------------------------------------
  const AUTO_STAGES = {script: "Script", cast: "Voices and sounds", speak: "Speaking"};
  function autoRunning() { return Boolean(state.run?.active && state.run.auto); }

  function renderAuto() {
    const running = autoRunning();
    $("auto-status").hidden = !running;
    $("auto-split").hidden = running;
    if (running) {
      $("auto-status-text").textContent = `Auto · ${AUTO_STAGES[state.run.stage] || "Starting"}: ${state.run.status || "…"}`;
    }
    $("auto").disabled = !state.boot.gpu || isRunning() || castRunning();
    $("auto").title = !state.boot.gpu ? "Auto needs the GPU: open Soundstage in Studio."
      : "Rewrite the script if it needs it, make every missing voice and sound, then speak";
  }

  async function startAuto() {
    $("auto-menu").open = false;
    if (isRunning() || castRunning()) return;
    try {
      await flushEdits();
      const data = await api(named("/auto"), {method: "POST", body: {writer: writerService()}});
      state.run = data.run;
      autoRevision = data.run?.revision ?? 0;
      renderAuto();
      renderActions();
      poll();
    } catch (error) {
      toast(error.message);
    }
  }

  function isRunning() { return Boolean(state.run?.active); }
  let autoRevision = 0;

  function renderActions() {
    const onScript = state.screen === "script";
    const onCast = state.screen === "cast";
    $("estimate").hidden = !onScript;
    $("action-text").hidden = onScript;
    for (const id of ["speak", "remix", "interrupt"]) if (state.screen !== "mix") $(id).hidden = true;
    for (const id of ["to-script", "cast-interrupt", "make-missing"]) $(id).hidden = !onCast;
    $("to-mix").hidden = state.screen === "mix";
    $("to-cast").hidden = !onScript;
    if (onScript) {
      $("to-mix").className = "btn";
      $("to-mix").textContent = "Mix & review";
      renderEstimate();
      return;
    }
    if (onCast) { renderCastActions(); return; }
    $("speak").hidden = false;
    const project = state.project;
    const result = project.result;
    const running = isRunning();
    const title = $("action-title");
    const detail = $("action-detail");
    detail.classList.remove("error");
    if (running && state.run.task === "words") {
      title.textContent = "Timing every word…";
      detail.textContent = state.run.status || "Waiting for the GPU…";
    } else if (running && state.run.auto) {
      title.textContent = `Auto · ${AUTO_STAGES[state.run.stage] || "Starting"}`;
      detail.textContent = state.run.status || "Waiting for the GPU…";
    } else if (running) {
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
    $("speak").title = $("remix").title = state.boot.gpu ? "" : NO_GPU;
    $("remix").hidden = !result;
    $("interrupt").hidden = !running;
    const regenerate = $("regenerate");
    if (regenerate) regenerate.disabled = running || !state.boot.gpu || !lineTakes(result)[state.picked];
  }

  function renderAll() {
    $("screen-script").hidden = state.screen !== "script";
    $("screen-mix").hidden = state.screen !== "mix";
    $("screen-cast").hidden = state.screen !== "cast";
    renderHeader();
    renderSteps();
    renderScript();
    if (state.screen === "cast") renderCast();
    renderPlayer();
    renderFilters();
    renderLines();
    renderInspector();
    renderSettings();
    renderWordRepair();
    renderVideo();
    renderAuto();
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
    paintStills(time);
    const caption = $("video-caption");
    if (caption) {
      const line = active >= 0 ? result.lines[active] : null;
      const show = state.project.video?.subtitles && line && line.end + 0.6 > time;
      caption.hidden = !show;
      if (show && caption.dataset.line !== String(active)) {
        caption.dataset.line = String(active);
        caption.replaceChildren(h("strong", {}, `${line.speaker}: `), ...wordNodes(line));
      }
      if (show) {
        for (const word of caption.querySelectorAll(".word")) {
          const start = Number(word.dataset.start), end = Number(word.dataset.end);
          word.classList.toggle("spoken", start <= time);
          word.classList.toggle("current", start <= time && time < end + 0.08);
        }
      }
    }
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
      following = !mixAudio.paused || !mixVideo.paused || !takeAudio.paused;
      if (following) requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }
  const players = [mixAudio, mixVideo, takeAudio];
  for (const element of players) {
    // One thing plays at a time.
    element.addEventListener("play", () => {
      for (const other of players) if (other !== element) other.pause();
      follow();
    });
    for (const event of ["pause", "seeked", "loadedmetadata", "ended"]) element.addEventListener(event, sync);
  }

  // -- Projects and runs -----------------------------------------------------
  let projectLoad = 0;
  async function loadProject(name, {keepPick = false} = {}) {
    const load = ++projectLoad;
    let project, revision;
    do {
      await flushEdits();
      revision = saves.revision(name);
      project = await api(named("/project", name), {unscoped: true});
      if (load !== projectLoad) return;
      // A save during this read makes its snapshot obsolete, even once edits are clean.
      await flushEdits();
      if (load !== projectLoad) return;
    } while (revision !== saves.revision(name));
    const switched = state.project?.name !== project.name;
    const changed = switched || state.project?.result?.version !== project.result?.version;
    if (switched) {
      projectEpoch++;
      poll.stop(); pollCast.stop(); pollVideo.stop();
    }
    state.project = project;
    reconcile(project);
    state.saving = saves.status(name);
    state.run = project.run;
    state.castRun = project.cast_run;
    if (!keepPick || !project.result || state.picked >= project.result.lines.length) state.picked = 0;
    if (switched) {
      state.castPick = null;
      state.castSource = null;
      library = null;
      state.writerResult = null;
      state.writing = false;
      state.estimate = null;
      if (!location.hash) state.screen = project.result ? "mix" : "script";
    }
    try { localStorage.setItem(LAST_PROJECT, name); } catch { /* Storage may be blocked. */ }
    state.videoRun = project.video_run;
    if (switched) { state.mediaMode = "audio"; mix = mixAudio; }
    if (changed) {
      mixAudio.pause();
      mixVideo.pause();
      takeAudio.pause();
      takeAudio.removeAttribute("src");
      delete takeAudio.dataset.key;
      if (project.result) {
        mixAudio.src = `${API}/audio?name=${encodeURIComponent(name)}&v=${project.result.version}`;
      } else {
        mixAudio.removeAttribute("src");
      }
      lastActive = -1;
    }
    if (switched || !library) await loadLibrary(project.cast_settings.speaker_group);
    if (load !== projectLoad) return;
    loadVideoSource();
    renderAll();
    if (switched) queueEstimate(0);
    if (isRunning()) poll();
    if (castRunning()) pollCast();
    if (videoRunning()) pollVideo();
  }

  function show(screen) {
    if (!SCREENS.includes(screen) || !state.project) return;
    if (screen !== state.screen) window.scrollTo({top: 0});
    state.screen = screen;
    if (location.hash !== `#${screen}`) history.replaceState(null, "", `#${screen}`);
    if (screen !== "mix") { mixAudio.pause(); mixVideo.pause(); takeAudio.pause(); }
    if (screen !== "cast") flushSlots();
    renderAll();
    if (screen === "script") queueEstimate(0);
  }

  // -- Script ----------------------------------------------------------------
  function renderSaveState() {
    const labels = {pending: "Unsaved changes", saving: "Saving…", saved: "All changes saved", failed: "Not saved"};
    $("save-state").textContent = labels[state.saving] || "";
  }

  // Script edits save on their own, a moment after typing stops.
  function saveScript(body, {now = false} = {}) {
    saves.queue(state.project.name, "script", body, now ? 0 : 800);
  }
  const flushScript = () => flushEdits();

  const WRITER_KEY = "h3-soundstage-writer";
  function writerService() {
    try { return JSON.parse(sessionStorage.getItem(WRITER_KEY)) || {}; } catch { return {}; }
  }
  function storeWriterService(values) {
    try { sessionStorage.setItem(WRITER_KEY, JSON.stringify({...writerService(), ...values})); } catch { /* Private windows refuse it. */ }
  }

  /** Apply a writer, pasted-reply or example answer: the project changed on the server. */
  function writerAnswered(data, label) {
    const {status, failed, previous, ...project} = data;
    state.writerResult = {status, failed, previous, label};
    if (project.name !== state.project.name) return;
    reconcile(project);
    if (!failed) {
      $("script-text").value = state.project.script.text;
      queueEstimate(0);
    }
    renderAll();
  }

  async function rewrite() {
    if (state.writing) return;
    try { await flushScript(); } catch (error) { toast(error.message); return; }
    const writingScope = scope();
    state.writing = true;
    const button = $("rewrite");
    const started = Date.now();
    const tick = () => { button.textContent = `Rewriting… ${Math.round((Date.now() - started) / 1000)} s`; };
    tick();
    const timer = setInterval(tick, 1000);
    renderWriter();
    button.textContent = "Rewriting… 0 s";
    try {
      const data = await api(named("/project/rewrite"), {method: "POST", body: {writer: writerService()}});
      writerAnswered(data, `Rewritten in ${Math.round((Date.now() - started) / 1000)} s`);
    } catch (error) {
      if (writingScope !== scope()) return;
      state.writerResult = {status: error.message, failed: true};
      renderWriterResult();
    } finally {
      clearInterval(timer);
      if (writingScope === scope()) { state.writing = false; renderWriter(); }
    }
  }

  function proposalDialog(title, contents, apply) {
    const dialog = h("dialog", {}, h("h2", {}, title), contents,
      h("button", {type: "button", class: "btn", onclick: async () => {
        try { await flushEdits(); await apply(); dialog.close(); }
        catch (error) { toast(error.message); }
      }}, "Apply saved result"),
      h("button", {type: "button", class: "btn", onclick: () => dialog.close()}, "Close"));
    dialog.addEventListener("close", () => dialog.remove());
    document.body.append(dialog); dialog.showModal();
  }

  async function reviewWriterProposal() {
    try {
      const path = named("/project/proposal"), owner = scope();
      const answer = await api(path);
      proposalDialog("Saved writer answer", h("textarea", {readOnly: true, value: answer.text, rows: 16}), async () => {
        if (scope() !== owner) throw new StaleResponse("Open the original project to apply this answer.");
        writerAnswered(await api(path, {method: "POST", body: {id: answer.id}}), "Saved answer applied");
      });
    } catch (error) { toast(error.message); }
  }

  async function reviewSlotProposal(kind, slot) {
    try {
      const path = named(`/project/slot/proposal?kind=${kind}&slot=${slot}`), owner = scope();
      const answer = await api(path);
      proposalDialog("Saved generated result", h("div", {},
        answer.audio ? h("audio", {controls: true, src: `${API}${named(`/project/slot/proposal/audio?kind=${kind}&slot=${slot}&id=${encodeURIComponent(answer.id)}`)}`}) : null,
        h("p", {}, answer.transcript || "Listen before replacing this slot's clip.")), async () => {
        if (scope() !== owner) throw new StaleResponse("Open the original project to apply this result.");
        castAnswered(await api(path, {method: "POST", body: {id: answer.id}}), {editor: true});
      });
    } catch (error) { toast(error.message); }
  }

  async function undoWriter() {
    const previous = state.writerResult?.previous;
    if (previous === undefined) return;
    $("script-text").value = previous;
    state.project.script.text = previous;
    state.writerResult = null;
    saveScript({text: previous}, {now: true});
    renderScript();
    toast("Script put back. The cast keeps the writer's changes; edit them in Cast & sounds.");
  }

  async function copyPrompt() {
    try {
      await flushScript();
      const {prompt} = await api(named("/project/writer-prompt"), {method: "POST", body: {}});
      await navigator.clipboard.writeText(prompt);
      toast("Writer prompt copied. Paste it into any chat model.");
    } catch (error) {
      toast(`Could not copy the prompt: ${error.message}`);
    }
  }

  async function applyReply() {
    try {
      await flushScript();
      const data = await api(named("/project/reply"), {method: "POST", body: {reply: $("reply").value}});
      writerAnswered(data, "Reply applied");
      if (!data.failed) $("reply").value = "";
    } catch (error) {
      toast(error.message);
    }
  }

  async function loadExample(example) {
    $("examples").open = false;
    const text = $("script-text").value.trim();
    if (text && !confirm(`Replace this project's script and cast with “${example.label}”?`)) return;
    try {
      await flushScript();
      writerAnswered(await api(named("/project/example"), {method: "POST", body: {example: example.id}}), "Example loaded");
    } catch (error) {
      toast(error.message);
    }
  }

  // The estimated timeline follows the saved script.
  function queueEstimate(delay = 400) {
    clearTimeout(state.estimateTimer);
    state.estimateTimer = setTimeout(async () => {
      if (!state.project) return;
      const name = state.project.name;
      try {
        const estimate = await api("/preview", {method: "POST", body: {text: $("script-text").value,
                                                                         script_mode: state.project.script.script_mode}});
        if (state.project.name !== name) return;
        state.estimate = estimate;
        renderEstimate();
      } catch { /* The bar keeps its last estimate. */ }
    }, delay);
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
    const place = (start, end) => ({left: `${start / duration * 100}%`, width: `${Math.max(end - start, 0) / duration * 100}%`});
    const voices = rows(estimate.lines);
    const lanes = [h("div", {class: "lane", style: {"--rows": voices.count}}, estimate.lines.map((line, index) => h("span", {
      class: "clip", title: `${line.speaker}: ${line.text}`,
      style: {...place(line.start, line.end), "--row": voices.placed[index], background: toneColor(
        estimate.speakers.find((speaker) => speaker.name === line.speaker)?.tone ?? "narrator")},
    })))];
    if (estimate.sounds.length) {
      const sounds = rows(estimate.sounds);
      lanes.push(h("div", {class: "lane", style: {"--rows": sounds.count}}, estimate.sounds.map((sound, index) => h("span", {
        class: `clip ${sound.loop ? "bed" : "cue"}`, title: sound.name,
        style: {...place(sound.start, sound.end), "--row": sounds.placed[index]},
      }))));
    }
    box.replaceChildren(
      h("div", {class: "estimate-head"},
        h("span", {}, h("strong", {}, "Estimated timeline"), ` · ${estimate.lines.length} lines, no audio yet`),
        h("span", {class: "num"}, `≈ ${clock(duration)}`)),
      h("div", {class: "lanes"}, lanes));
  }

  async function newProject() {
    const dialog = $("new-dialog");
    $("new-name").value = "";
    dialog.showModal();
  }

  async function createProject(name) {
    try {
      const project = await api("/projects", {method: "POST", body: {name}});
      state.boot.projects.unshift({name: project.name, generated: false});
      $("empty").hidden = true;
      $("app").hidden = false;
      history.replaceState(null, "", "#script");
      state.screen = "script";
      await loadProject(project.name);
    } catch (error) {
      toast(error.message);
    }
  }

  // -- Cast & sounds ---------------------------------------------------------
  let library = null;
  async function loadLibrary(group) {
    try {
      library = await api(`/library?group=${encodeURIComponent(group || "Global")}`);
    } catch (error) {
      library = {groups: ["Global"], speakers: [], sounds: [], error: error.message};
    }
    return library;
  }

  function saveSlot(kind, slot, body, {now = false} = {}) {
    saves.queue(state.project.name, `slot:${kind}:${slot}`, body, now ? 0 : 700);
  }
  const flushSlots = () => flushEdits();

  /** The project changed on the server: keep the screen, refresh what it shows. */
  function castAnswered(project, {editor = false} = {}) {
    if (!project) return;
    const {message, ...rest} = project;
    if (!reconcile(rest)) return;
    if (project.cast_run !== undefined) state.castRun = project.cast_run;
    renderSteps();
    renderAuto();
    if (state.screen === "cast") {
      renderCastLists();
      renderSlotEditor(editor);
      renderCastActions();
    }
    if (message) toast(message);
  }

  async function castCall(path, options, {editor = true} = {}) {
    try {
      await flushSlots();
      const project = await api(named(path), options);
      castAnswered(project, {editor});
      return project;
    } catch (error) {
      toast(error.message);
      return null;
    }
  }

  async function saveCastSettings(body) {
    const project = await castCall("/project/cast-settings", {method: "PATCH", body});
    if (project && body.speaker_group) { await loadLibrary(body.speaker_group); renderCast(); }
    else if (project) renderSlotEditor(true);
  }

  function resizeSlots(kind, add) {
    castCall(`/project/slots?kind=${kind}`, {method: "POST", body: {add}}).then((project) => {
      if (!project) return;
      const list = kind === "voice" ? project.speakers : project.sounds;
      if (add && list.length) pickSlot(kind, list.at(-1).slot);
      else { state.castPick = null; renderCast(); }
    });
  }

  function useSaved(kind, slot, key) {
    state.castSource = {...(state.castSource || {}), [`${kind}:${slot}`]: "saved"};
    castCall(`/project/slot/use-saved?kind=${kind}&slot=${slot}`, {method: "POST", body: {key}});
  }

  function saveToLibrary(kind, slot) {
    castCall(`/project/slot/save?kind=${kind}&slot=${slot}`, {method: "POST", body: {}}).then(() => {
      loadLibrary(state.project.cast_settings.speaker_group).then(() => renderSlotEditor(true));
    });
  }

  function removeClip(kind, slot) {
    castCall(`/project/slot/clip?kind=${kind}&slot=${slot}`, {method: "DELETE"});
  }

  let clipTarget = null;
  function chooseClip(kind, slot) {
    clipTarget = {kind, slot};
    const input = $("clip-input");
    input.value = "";
    input.click();
  }

  async function uploadClip(file) {
    if (!clipTarget || !file) return;
    const {kind, slot} = clipTarget;
    await flushSlots();
    const form = new FormData();
    form.append("file", file);
    try {
      const response = await fetch(`${API}${named(`/project/slot/upload?kind=${kind}&slot=${slot}`)}`, {method: "POST", body: form});
      const data = await response.json();
      if (!response.ok || data.error) throw new Error(data.error || data.detail || "Upload failed.");
      castAnswered(data, {editor: true});
      toast(kind === "voice" ? "Clip uploaded. Add its exact transcript, or click Transcribe." : "Clip uploaded.");
    } catch (error) {
      toast(error.message);
    }
  }

  async function make(target, slot) {
    if (castRunning()) return;
    try {
      await flushSlots();
      const data = await api(named("/project/make"), {method: "POST", body: {target, slot}});
      state.castRun = data.cast_run;
      renderCastLists();
      renderSlotEditor(true);
      renderCastActions();
      pollCast();
    } catch (error) {
      toast(error.message);
    }
  }

  async function interruptCast() {
    try {
      toast((await api(named("/project/make/interrupt"), {method: "POST"})).message);
    } catch (error) {
      toast(error.message);
    }
  }

  const pollCast = poller(async (again) => {
    const name = state.project.name;
    const project = await api(named("/project", name));
    if (state.project.name !== name) return;
    const before = state.castRun;
    castAnswered(project);
    if (state.screen !== "cast") renderSteps();
    if (project.cast_run?.active) { again(); return; }
    if (before?.active) toastEnded(project.cast_run);
    renderSlotEditor(true);
  });

  async function newGroup() {
    const name = prompt("Name the speaker group, such as a series or book:");
    if (!name || !name.trim()) return;
    try {
      const data = await api("/library/groups", {method: "POST", body: {name: name.trim()}});
      library = {...(library || {}), groups: data.groups};
      await saveCastSettings({speaker_group: data.group});
    } catch (error) {
      toast(error.message);
    }
  }

  async function speak(action, line) {
    if (isRunning()) return;
    try {
      await flushEdits();
      if (isRunning()) return;
      const data = await api(named("/speak"), {method: "POST", body: {action, line}});
      state.run = data.run;
      renderActions();
      renderInspector();
      poll();
    } catch (error) {
      toast(error.message);
    }
  }

  const poll = poller(async (again) => {
    const name = state.project.name;
    const {run} = await api(named("/run", name));
    if (name !== state.project.name) return;
    const before = state.run;
    state.run = run;
    if (run && !run.active) {
      if (run.auto) {
        if (run.error) toast(`Auto stopped: ${run.error}`);
        else if (before?.active) toast(run.message ? `Auto finished. ${run.message}` : "Auto finished.");
        if (!run.error && before?.active) show("mix");
      } else if (run.task === "words") {
        if (run.error) toast(`Word timing stopped: ${run.error}`);
        else if (before?.active) toast(run.status || "Word timing ready.");
      } else if (run.error) toast(`Speaking stopped: ${run.error}`);
      else if (before?.active) toast("Speech ready.");
      await loadProject(name, {keepPick: true});
      return;
    }
    if (run?.auto && run.revision !== autoRevision) {
      autoRevision = run.revision;
      await loadProject(name, {keepPick: true});
    }
    renderAuto();
    renderActions();
    if (state.screen === "cast") { renderCastLists(); renderSlotEditor(false); }
    if (run?.active) again();
  });

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
      h("p", {}, "Start one here: write or paste a story, and the writer turns it into a script."),
      h("p", {}, h("button", {type: "button", class: "btn btn-primary", onclick: newProject}, "New project")));
  }

  async function start() {
    window.addEventListener("unhandledrejection", (event) => {
      if (event.reason instanceof Error) { event.preventDefault(); toast(event.reason.message); }
    });
    applyTheme();
    matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", applyTheme);
    // In the studio's Soundstage tab, follow its light/dark switch as it changes.
    try {
      if (window.parent !== window) {
        new MutationObserver(applyTheme).observe(window.parent.document.body, {attributes: true, attributeFilter: ["class"]});
      }
    } catch { /* Framed by another origin. */ }
    $("project").addEventListener("change", (event) => loadProject(event.target.value).catch((error) => {
      event.target.value = state.project?.name || "";
      toast(error.message);
    }));
    $("new-project").addEventListener("click", newProject);
    $("new-cancel").addEventListener("click", () => $("new-dialog").close());
    $("new-form").addEventListener("submit", (event) => {
      event.preventDefault();
      const name = $("new-name").value.trim();
      if (!name) return;
      $("new-dialog").close();
      createProject(name);
    });
    for (const screen of SCREENS) $(`step-${screen}`).addEventListener("click", () => show(screen));
    $("to-mix").addEventListener("click", () => show("mix"));
    $("to-cast").addEventListener("click", () => show("cast"));
    $("auto").addEventListener("click", startAuto);
    $("auto-interrupt").addEventListener("click", interruptRun);
    $("auto-save").addEventListener("change", (event) => saveSettings({auto_save_speakers: event.target.checked}));
    $("to-script").addEventListener("click", () => show("script"));
    $("make-missing").addEventListener("click", () => make("prepare"));
    $("cast-interrupt").addEventListener("click", interruptCast);
    $("new-group").addEventListener("click", newGroup);
    $("open-library").addEventListener("click", openLibrary);
    $("library-close").addEventListener("click", () => $("library-dialog").close());
    $("add-sound").addEventListener("click", () => resizeSlots("sound", true));
    $("speaker-group").addEventListener("change", (event) => saveCastSettings({speaker_group: event.target.value}));
    $("clip-input").addEventListener("change", (event) => uploadClip(event.target.files[0]));
    window.addEventListener("hashchange", () => show(location.hash.slice(1)));
    $("speak").addEventListener("click", () => speak("generate"));
    $("remix").addEventListener("click", () => speak("remix"));
    $("interrupt").addEventListener("click", interruptRun);
    $("word-timing").addEventListener("change", (event) => saveSettings({word_timing: event.target.checked}));
    const area = $("script-text");
    area.addEventListener("input", () => { paintEditor(); saveScript({text: area.value}); });
    area.addEventListener("blur", () => flushScript());
    $("title").addEventListener("input", (event) => saveScript({title: event.target.value}));
    $("strict").addEventListener("change", (event) => saveScript({strict_source: event.target.checked}, {now: true}));
    $("rewrite").addEventListener("click", rewrite);
    $("copy-prompt").addEventListener("click", copyPrompt);
    $("apply-reply").addEventListener("click", applyReply);
    document.addEventListener("click", (event) => {
      if (!event.target.closest("#examples")) $("examples").open = false;
      if (!event.target.closest("#auto-menu")) $("auto-menu").open = false;
    });
    document.addEventListener("visibilitychange", () => { if (document.hidden) { flushScript(); flushSlots(); } });
    document.addEventListener("keydown", (event) => {
      if (state.screen !== "mix" || event.target.closest("input, select, textarea, button") || !state.project?.result) return;
      if (event.key === " ") { event.preventDefault(); if (mix.paused) mix.play(); else mix.pause(); }
      if ((event.key === "c" || event.key === "C") && !event.ctrlKey && !event.metaKey && !event.altKey
          && state.mediaMode === "video" && state.project.video) {
        setSubtitles(!state.project.video.subtitles);
      }
    });
    try {
      state.boot = await api("/boot");
    } catch (error) {
      $("loading").textContent = `Soundstage could not open: ${error.message}`;
      return;
    }
    const projects = state.boot.projects;
    if (!projects.length) { showEmpty(); return; }
    if (SCREENS.includes(location.hash.slice(1))) state.screen = location.hash.slice(1);
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
    paintEditor();  // Sizes the editor now that it can be measured.
  }


  const {scriptSpeakers, highlight, paintEditor, renderWriter, renderWriterResult, renderScript, NAME, TOKEN} = createScript({
    state,
    $: (...args) => $(...args),
    h,
    toneColor,
    saveScript,
    writerService,
    storeWriterService,
    reviewWriterProposal,
    undoWriter,
    loadExample,
    renderSaveState
  });
  const {castRunning, castJob, slotNote, slotWorking, speakerLines, missingItems, pickSlot, ensurePick, renderCast, renderCastLists, field, renderSlotEditor, noteBox, sourceSwitch, voiceEditor, soundEditor, renderGeneratorSettings, renderCastActions} = createCast({
    state,
    $: (...args) => $(...args),
    h,
    toneColor,
    svg,
    NO_GPU,
    AUTO_STAGES,
    autoRunning,
    flushSlots,
    saveSlot,
    reviewSlotProposal,
    saveCastSettings,
    resizeSlots,
    useSaved,
    make,
    chooseClip,
    saveToLibrary,
    removeClip,
    scriptSpeakers,
    NAME,
    TOKEN,
    getLibrary: () => library
  });
  const {setSubtitles, sizeLimit, setSubtitleSize, subtitleControls, renderPlayer, wordNodes, renderFilters, renderLines, renderInspector, renderSettings} = createMix({
    state,
    $: (...args) => $(...args),
    h,
    API,
    STATUS_LABELS,
    FILTERS,
    ICONS,
    takeAudio,
    mixVideo,
    clock,
    isRunning,
    lineTakes,
    named,
    pick,
    rows,
    saveSettings: (...args) => saveSettings(...args),
    saveVideo,
    seek,
    setMediaMode: (...args) => setMediaMode(...args),
    speak,
    speakerTone,
    svg,
    sync,
    toggleTake,
    toneColor,
    videoFileButton: (...args) => videoFileButton(...args),
    visualSource: (...args) => visualSource(...args),
    getPlayer: () => mix
  });
  const {videoRunning, visualSource, setMediaMode, videoFileButton, paintStills, videoField, renderVideo} = createVideo({
    state,
    $: (...args) => $(...args),
    h,
    mixAudio,
    mixVideo,
    clock,
    saveVideo,
    seek,
    renderPlayer,
    sync,
    saveVideoToMedia,
    planShots,
    drawImages,
    interruptVideo,
    copyPlannerPrompt,
    applyPlannerReply,
    getPlayer: () => mix,
    setPlayer: (player) => { mix = player; }
  });
  const {openLibrary, refreshLibrary, libraryCall, renderLibrary, speakerLibrary, soundLibrary, tryVoices, libraryState, pollTakes} = createLibrary({
    state,
    $: (...args) => $(...args),
    h,
    API,
    api,
    castAnswered,
    loadLibrary,
    renderSlotEditor,
    field,
    toast,
    poller
  });

  const {saves, reconcile, saveSettings, flushEdits} = createProjects({
    state, api, named, scope, renderSaveState, renderHeader, renderSteps, renderSettings, renderScript, queueEstimate, renderVideo, castAnswered, toast
  });

  return start();
}
