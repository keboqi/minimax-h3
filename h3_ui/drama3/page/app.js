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
  const mixAudio = $("mix-audio");
  const mixVideo = $("mix-video");
  // The player plays the mix, or its illustrated video, which carries the same soundtrack.
  let mix = mixAudio;
  const takeAudio = $("take-audio");

  const state = {boot: null, project: null, picked: 0, filter: "all", run: null, poll: null, busy: false,
                 screen: "script", saving: null, saveTimer: null, writing: false, writerResult: null,
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
      state.project.video?.latest ? h("div", {class: "seg", role: "group", "aria-label": "Show"},
        ["audio", "video"].map((mode) => h("button", {type: "button", "aria-pressed": String(state.mediaMode === mode),
                                                       onclick: () => setMediaMode(mode)}, mode === "audio" ? "Audio" : "Video"))) : null,
      state.mediaMode === "video" ? h("label", {class: "check"}, h("input", {type: "checkbox", checked: state.project.video.subtitles,
        onchange: (event) => { state.project.video.subtitles = event.target.checked; saveVideo({video_subtitles: event.target.checked}); sync(); }}),
        "Subtitles on the video") : null,
      state.mediaMode === "video"
        ? h("a", {class: "btn btn-small", href: state.project.video.latest.url, download: state.project.video.latest.file}, "Download video")
        : h("a", {class: "btn btn-small", href: `${API}${named("/audio")}`, download: result.file}, "Download"));
    let stage = null;
    if (state.mediaMode === "video") {
      mixVideo.hidden = false;
      stage = h("div", {class: "video-stage"}, mixVideo, h("p", {class: "video-caption", id: "video-caption", "aria-live": "off"}));
      mixVideo.onclick = () => (mixVideo.paused ? mixVideo.play() : mixVideo.pause());
    } else {
      mixVideo.hidden = true;
      document.body.append(mixVideo);
    }

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
    box.replaceChildren(...[head, stage, timeline, legend].filter(Boolean));
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
        h("button", {type: "button", class: "btn btn-primary", id: "regenerate", "data-gpu": true, disabled: !take || !state.boot.gpu || isRunning(),
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
    $("auto-save").checked = settings.auto_save_speakers;
    // Leave the sliders alone while one is being dragged or typed in.
    if ($("speech-settings").contains(document.activeElement)) return;
    const sliders = [
      ["temperature", "Temperature", ""],
      ["top_p", "Top-p", ""],
      ["top_k", "Top-k", "0 keeps all candidates. Small caps can make speech stall at pauses."],
      ["parallel", "Parallel requests", "Lines go out in parallel with their speaker's clip."],
      ["retries", "Retries per chunk", "0 keeps the first take. Retries only near-silent audio."],
      ["max_frames", "Maximum audio frames per take", "21.5 frames a second; 8192 is about 6.3 minutes. Only limits a take that runs on."],
    ];
    $("speech-settings").replaceChildren(...sliders.map(([key, label, hint]) => {
      const [min, max, step] = state.boot.ranges[key];
      // The server keeps values inside their range; show what it kept.
      const commit = async (value) => {
        if (value === "") return;
        await saveSettings({[key]: Number(value)});
        number.value = range.value = state.project.settings[key];
      };
      const number = h("input", {type: "number", min, max, step, value: settings[key], "aria-label": label,
                                 onchange: (event) => commit(event.target.value)});
      const range = h("input", {type: "range", min, max, step, value: settings[key], "aria-label": label,
                                oninput: (event) => { number.value = event.target.value; },
                                onchange: (event) => commit(event.target.value)});
      return h("div", {class: "slider-row"}, h("span", {class: "label"}, label), range, number,
               hint ? h("span", {class: "hint"}, hint) : null);
    }), h("p", {class: "muted small"}, "Drama 3 runs on SGLang-Omni (Linux, CUDA); the checkpoint downloads on first use."));
  }

  // -- Illustrated video -----------------------------------------------------
  function videoRunning(kind) {
    const run = state.videoRun;
    return Boolean(run?.active && (!kind || run.kind === kind));
  }

  /** Audio or Video in the player: the video plays the same mix, with the lines over it. */
  function setMediaMode(mode) {
    const video = state.project.video?.latest;
    mode = mode === "video" && video ? "video" : "audio";
    if (mode === state.mediaMode) return;
    const time = mix.currentTime || 0;
    mix.pause();
    state.mediaMode = mode;
    mix = mode === "video" ? mixVideo : mixAudio;
    try { mix.currentTime = time; } catch { /* Not loaded yet; it starts from the top. */ }
    renderPlayer();
    sync();
  }

  function videoField(label, input, hint) {
    return h("label", {class: "field"}, h("span", {class: "label"}, label), input,
             hint ? h("span", {class: "muted small"}, hint) : null);
  }

  function renderVideo() {
    const box = $("video-panel");
    const boot = state.boot.video;
    const video = state.project.video;
    box.hidden = !boot || !video;
    if (box.hidden) return;
    // Leave the panel alone while one of its fields is being typed in.
    if (box.contains(document.activeElement) && document.activeElement.matches("input, textarea")) return;
    const result = state.project.result;
    const planning = videoRunning("plan");
    const drawing = videoRunning("render");
    const save = (body) => saveVideo(body);
    const presets = [...Object.keys(boot.styles), boot.custom_style];
    const rows = video.plan.rows;
    const run = state.videoRun;
    const note = run && (run.active || run.error || run.status)
      ? h("p", {class: `slot-note${run.error ? " error" : ""}`}, run.error || run.status) : null;
    const limitCount = h("span", {class: "num small"}, video.limit);
    box.replaceChildren(
      h("div", {class: "card-head"},
        h("h2", {}, "Illustrated video"),
        h("span", {class: "muted small"}, "A 720p video of the mix, drawn shot by shot with an Image model."),
        h("span", {class: "spacer"}),
        video.latest ? h("button", {type: "button", class: "btn btn-small", onclick: () => { setMediaMode("video"); window.scrollTo({top: 0, behavior: "smooth"}); }},
          "Watch the latest video") : null),
      !result ? h("p", {class: "video-empty muted"}, "Speak the project first: the video is drawn from its mix.") : h("div", {class: "video-body"},
        h("div", {class: "video-settings"},
          videoField("Image model", h("select", {onchange: (event) => save({video_engine: event.target.value})},
            boot.engines.map((engine) => h("option", {value: engine.id, selected: engine.id === video.engine}, engine.label))),
            "Draws with the Image tab's settings for that model."),
          videoField("Style preset", h("select", {onchange: (event) => {
            const style = boot.styles[event.target.value];
            if (style) save({video_style: style});
          }}, presets.map((name) => h("option", {selected: name === video.preset, disabled: name === boot.custom_style && video.preset !== name}, name)))),
          videoField("Art direction", h("textarea", {rows: 3, value: video.style, oninput: (event) => saveVideo({video_style: event.target.value}, {later: true})}),
            "Add the cast, setting or era. Editing it makes the preset Custom."),
          videoField("Minimum shots", h("span", {class: "range-row"},
            h("input", {type: "range", min: boot.limits[0], max: boot.limits[1], step: 1, value: video.limit,
                        oninput: (event) => { limitCount.textContent = event.target.value; },
                        onchange: (event) => save({video_limit: Number(event.target.value)})}), limitCount),
            "A floor: the planner adds images wherever the story needs them, up to one per 2 seconds."),
          h("label", {class: "check"}, h("input", {type: "checkbox", checked: video.continuity,
            onchange: (event) => save({video_continuity: event.target.checked})}), "Keep characters consistent"),
          h("div", {class: "row-buttons"},
            h("button", {type: "button", class: "btn", disabled: planning || drawing, onclick: planShots},
              planning ? "Planning…" : rows.length ? "Plan again" : "Plan shots"),
            drawing
              ? h("button", {type: "button", class: "btn", onclick: interruptVideo}, "Interrupt")
              : h("button", {type: "button", class: "btn btn-primary", "data-gpu": true, disabled: !rows.length || planning || !state.boot.gpu,
                             title: rows.length ? "" : "Plan shots first", onclick: renderVideoJob}, "Generate illustrated video")),
          note,
          h("details", {class: "more"},
            h("summary", {}, "Use another chat model"),
            h("div", {class: "more-body"},
              h("p", {class: "muted small"}, "Copy the full planner prompt into any chat model, then paste its JSON reply. It is checked like the built-in planner's."),
              h("button", {type: "button", class: "btn btn-small", onclick: copyPlannerPrompt}, "Copy full planner prompt"),
              h("label", {class: "field"}, h("span", {class: "label"}, "Pasted planner reply"),
                h("textarea", {rows: 4, id: "plan-reply", placeholder: '{"art_style": "...", "shots": [...]}'})),
              h("button", {type: "button", class: "btn btn-small", onclick: applyPlannerReply}, "Apply pasted plan")))),
        h("div", {class: "video-plan"},
          h("span", {class: "label"}, rows.length ? `Shot plan · ${rows.length} shots` : "Shot plan"),
          rows.length ? h("div", {class: "table-wrap plan-wrap"}, h("table", {class: "plan-table"},
            h("thead", {}, h("tr", {}, ["Time", "", "Story beat", "Image prompt"].map((label) => h("th", {scope: "col"}, label)))),
            h("tbody", {}, rows.map((row) => h("tr", {},
              h("td", {class: "num"}, h("button", {type: "button", class: "link", onclick: () => seek(row.start, false)},
                `${clock(row.start)}–${clock(row.end)}`)),
              h("td", {}, h("span", {class: `kind ${row.transition === "Cut" ? "kind-shot" : "kind-loop"}`}, row.transition)),
              h("td", {}, row.beat),
              h("td", {class: "muted"}, row.prompt))))))
            : h("p", {class: "muted small"}, video.plan.stale
              ? "The saved plan was made for a different Art direction or Minimum shots. Plan again, or set them back."
              : "Plan shots to see each image's time, story beat and prompt here before drawing."))));
  }

  // Video settings save on their own; the Art direction a moment after typing stops.
  let videoTimer = null;
  let videoEdits = {};
  function saveVideo(body, {later = false} = {}) {
    videoEdits = {...videoEdits, ...body};
    clearTimeout(videoTimer);
    videoTimer = setTimeout(flushVideo, later ? 700 : 0);
  }

  async function flushVideo() {
    clearTimeout(videoTimer);
    const body = videoEdits;
    videoEdits = {};
    if (!Object.keys(body).length) return;
    try {
      const project = await api(named("/project/video"), {method: "PATCH", body});
      state.project.video = project.video;
      renderVideo();
    } catch (error) {
      toast(error.message);
    }
  }

  async function videoCall(path, body) {
    await flushVideo();
    try {
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
  const renderVideoJob = () => videoCall("/project/video/render");
  const interruptVideo = () => videoCall("/project/video/interrupt");

  async function copyPlannerPrompt() {
    await flushVideo();
    try {
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

  let videoPollTimer = null;
  function pollVideo() {
    clearTimeout(videoPollTimer);
    videoPollTimer = setTimeout(async () => {
      const name = state.project.name;
      try {
        const project = await api(`/project?name=${encodeURIComponent(name)}`);
        if (state.project.name !== name) return;
        const before = state.videoRun;
        const hadVideo = state.project.video?.latest?.url;
        state.videoRun = project.video_run;
        state.project.video = project.video;
        if (project.video_run?.active) { renderVideo(); pollVideo(); return; }
        if (before?.active) {
          if (project.video_run?.error) toast(`Stopped: ${project.video_run.error}`);
          else toast(project.video_run?.status || "Done.");
          if (before.kind === "render" && !project.video_run?.error && project.video?.latest?.url !== hadVideo) {
            loadVideoSource();
            setMediaMode("video");
            window.scrollTo({top: 0, behavior: "smooth"});
          }
        }
        renderVideo();
        renderPlayer();
      } catch (error) {
        toast(error.message);
        pollVideo();
      }
    }, POLL_MS);
  }

  /** Point the video player at the mix's newest video, if it changed. */
  function loadVideoSource() {
    const latest = state.project.video?.latest;
    if (!latest) {
      mixVideo.removeAttribute("src");
      if (state.mediaMode === "video") { state.mediaMode = "audio"; mix = mixAudio; }
      return;
    }
    if (mixVideo.dataset.url !== latest.url) {
      mixVideo.src = latest.url;
      mixVideo.dataset.url = latest.url;
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
  const libraryState = {tab: "speakers", group: "Global", data: null, picked: null, takes: null};

  async function openLibrary() {
    libraryState.group = state.project.cast_settings.speaker_group || "Global";
    libraryState.picked = null;
    libraryState.takes = null;
    $("library-dialog").showModal();
    await refreshLibrary();
  }

  async function refreshLibrary() {
    try {
      libraryState.data = await api(`/library/manage?group=${encodeURIComponent(libraryState.group)}`);
    } catch (error) {
      libraryState.data = {groups: ["Global"], speakers: [], sounds: [], error: error.message};
    }
    if (libraryState.picked && !libraryState.data.speakers.some((speaker) => speaker.key === libraryState.picked)) {
      libraryState.picked = null;
    }
    renderLibrary();
  }

  /** A library change also changed the project when it used that voice: show both. */
  async function libraryCall(path, body) {
    try {
      const data = await api(path, {method: "POST", body: {...body, project: state.project.name}});
      if (data.project) castAnswered(data.project, {editor: true});
      if (data.key !== undefined) libraryState.picked = data.key;
      if (data.message) toast(data.message);
      await loadLibrary(state.project.cast_settings.speaker_group);
      await refreshLibrary();
      if (state.screen === "cast") renderSlotEditor(true);
      return data;
    } catch (error) {
      toast(error.message);
      return null;
    }
  }

  function renderLibrary() {
    const data = libraryState.data;
    $("library-tabs").replaceChildren(...[["speakers", "Saved speakers"], ["sounds", "Saved sounds"]].map(([id, label]) =>
      h("button", {type: "button", "aria-pressed": String(libraryState.tab === id),
                   onclick: () => { libraryState.tab = id; renderLibrary(); }}, label)));
    const body = $("library-body");
    if (!data) { body.replaceChildren(h("p", {class: "muted"}, "Loading…")); return; }
    if (data.error) { body.replaceChildren(h("p", {class: "slot-note error"}, data.error)); return; }
    body.replaceChildren(...(libraryState.tab === "speakers" ? speakerLibrary(data) : soundLibrary(data)));
  }

  function speakerLibrary(data) {
    const picked = data.speakers.find((speaker) => speaker.key === libraryState.picked);
    const list = h("div", {class: "library-list"},
      h("label", {class: "field"}, h("span", {class: "label"}, "Group"),
        h("select", {onchange: (event) => { libraryState.group = event.target.value; libraryState.picked = null; refreshLibrary(); }},
          data.groups.map((group) => h("option", {selected: group === libraryState.group}, group)))),
      data.speakers.length ? h("ul", {}, data.speakers.map((speaker) => h("li", {},
        h("button", {type: "button", class: "library-item", "aria-pressed": String(speaker.key === libraryState.picked),
                     onclick: () => { libraryState.picked = speaker.key; libraryState.takes = null; renderLibrary(); }},
          h("strong", {}, speaker.name), h("span", {}, speaker.description || "No description")))))
        : h("p", {class: "muted small"}, `No saved speakers in ${libraryState.group}.`),
      libraryState.group !== "Global" ? h("button", {type: "button", class: "btn btn-small danger", onclick: () => {
        if (confirm(`Delete the group “${libraryState.group}” and all ${data.speakers.length} of its saved speakers? This can't be undone.`)) {
          const group = libraryState.group;
          libraryState.group = "Global";
          libraryCall("/library/group/delete", {group});
        }
      }}, "Delete this group") : null);
    if (!picked) {
      return [list, h("div", {class: "library-detail"}, h("p", {class: "muted"},
        "Pick a saved speaker to hear it, edit it, copy it to another group or try new voices for it."))];
    }
    const name = h("input", {value: picked.name});
    const description = h("textarea", {rows: 3, value: picked.description});
    const transcript = h("textarea", {rows: 3, value: picked.transcript});
    const destination = h("select", {}, data.groups.filter((group) => group !== picked.group).map((group) => h("option", {}, group)));
    const count = h("select", {"aria-label": "How many voices"}, [1, 2, 3, 4].map((n) => h("option", {value: n}, `${n} voice${n > 1 ? "s" : ""}`)));
    const takes = libraryState.takes;
    const working = takes?.active;
    return [list, h("div", {class: "library-detail"},
      h("audio", {controls: true, preload: "none", src: `${API}/library/clip?kind=voice&key=${encodeURIComponent(picked.key)}&t=${Date.now()}`}),
      field("Name", name), field("Voice description", description), field("Exact clip transcript", transcript),
      h("div", {class: "row-buttons"},
        h("button", {type: "button", class: "btn btn-primary btn-small", onclick: () => libraryCall("/library/speaker/update", {
          key: picked.key, name: name.value, description: description.value, transcript: transcript.value})}, "Save changes"),
        h("button", {type: "button", class: "btn btn-small danger", onclick: () => {
          if (confirm(`Delete the saved speaker “${picked.name}” from ${picked.group}? Speakers using it lose their voice. This can't be undone.`)) {
            libraryCall("/library/speaker/delete", {key: picked.key});
          }
        }}, "Delete")),
      destination.children.length ? h("div", {class: "row-buttons"}, destination,
        h("button", {type: "button", class: "btn btn-small", onclick: () => libraryCall("/library/speaker/copy", {key: picked.key, group: destination.value})},
          "Copy to group")) : null,
      h("div", {class: "library-section"},
        h("h2", {}, "Try new voices"),
        h("p", {class: "muted small"}, "Designs new voices from this description and transcript with the Cast's voice generator. "
          + "The saved clip stays until you keep one."),
        h("div", {class: "row-buttons"}, count,
          working ? h("button", {type: "button", class: "btn btn-small", onclick: () => api("/library/takes/interrupt", {method: "POST", body: {run: takes.id}}).then((reply) => toast(reply.message), (error) => toast(error.message))}, "Interrupt")
            : h("button", {type: "button", class: "btn btn-small", "data-gpu": true, onclick: () => tryVoices(picked, Number(count.value), description.value, transcript.value)},
                "Generate new voices")),
        takes && takes.key === picked.key ? h("p", {class: `slot-note${takes.error ? " error" : ""}`}, takes.error || takes.status) : null,
        takes && takes.key === picked.key ? h("div", {class: "takes"}, takes.takes.map((url, index) => url ? h("div", {class: "take-row"},
          h("span", {class: "small"}, `Voice ${index + 1}`), h("audio", {controls: true, preload: "none", src: url}),
          h("button", {type: "button", class: "btn btn-small", disabled: working,
                       onclick: () => libraryCall("/library/speaker/use-take", {run: takes.id, index}).then(() => { libraryState.takes = null; renderLibrary(); })},
            "Keep this voice")) : null)) : null))];
  }

  function soundLibrary(data) {
    if (!data.sounds.length) {
      return [h("p", {class: "muted"}, "No saved sounds yet. Save a sound's clip on Cast & sounds to reuse it.")];
    }
    return [h("div", {class: "library-detail"}, data.sounds.map((sound) => h("div", {class: "sound-row"},
      h("div", {}, h("strong", {}, sound.name), h("p", {class: "muted small"}, `${sound.duration} s · ${sound.description || "No description"}`)),
      h("audio", {controls: true, preload: "none", src: `${API}/library/clip?kind=sound&key=${encodeURIComponent(sound.name)}`}),
      h("button", {type: "button", class: "btn btn-small danger", onclick: () => {
        if (confirm(`Delete the saved sound “${sound.name}”? This can't be undone.`)) libraryCall("/library/sound/delete", {name: sound.name});
      }}, "Delete"))))];
  }

  async function tryVoices(picked, count, description, transcript) {
    try {
      const data = await api("/library/speaker/takes", {method: "POST", body: {
        key: picked.key, count, description, transcript, project: state.project.name}});
      libraryState.takes = data.takes_run;
      renderLibrary();
      pollTakes();
    } catch (error) {
      toast(error.message);
    }
  }

  let takesTimer = null;
  function pollTakes() {
    clearTimeout(takesTimer);
    takesTimer = setTimeout(async () => {
      const run = libraryState.takes;
      if (!run) return;
      try {
        const data = await api(`/library/takes?run=${encodeURIComponent(run.id)}`);
        libraryState.takes = data.takes_run;
        if ($("library-dialog").open) renderLibrary();
        if (data.takes_run.active) pollTakes();
      } catch (error) {
        toast(error.message);
      }
    }, POLL_MS);
  }

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
    await flushScript();
    await flushSlots();
    try {
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
  async function loadProject(name, {keepPick = false} = {}) {
    await flushScript();
    await flushSlots();
    const project = await api(`/project?name=${encodeURIComponent(name)}`);
    const switched = state.project?.name !== project.name;
    const changed = switched || state.project?.result?.version !== project.result?.version;
    state.project = project;
    state.run = project.run;
    state.castRun = project.cast_run;
    if (!keepPick || !project.result || state.picked >= project.result.lines.length) state.picked = 0;
    if (switched) {
      state.castPick = null;
      state.castSource = null;
      library = null;
      state.writerResult = null;
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
  const NAME = /^(\s*)([^\s:：<>[\]{}|][^:：<>[\]{}|\n]{0,39}?)(\s*[:：])/;
  // A sound tag is colored with its description: <sound name="rain">Steady rain</sound>.
  const TOKEN = /(<sound\b[^>]*>(?:[^<\n]*<\/sound>)?|<\/sound>)|(<\|[^|]*\|>|<[^>]*>)|(\[[^\]\n]*\])|(\{[^{}\n]*\})/g;
  const escapeHtml = (text) => text.replace(/[&<>"]/g, (char) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"})[char]);

  /** Speakers in the order the script first names them, with the result view's colors. */
  function scriptSpeakers(text) {
    const narrators = new Set(state.boot.narrators);
    const order = new Map();
    for (const line of text.split("\n")) {
      const match = NAME.exec(line);
      if (!match) continue;
      const name = match[2].trim();
      const entry = order.get(name) || {name, count: 0};
      entry.count += 1;
      order.set(name, entry);
    }
    let character = 0;
    const cast = new Set(state.project.cast.map((speaker) => speaker.name.trim().toLowerCase()));
    const numbered = /^speaker\s+\d+$/i;
    return [...order.values()].map((entry) => {
      const narrator = narrators.has(entry.name.toLowerCase());
      const tone = narrator ? "narrator" : String(character++ % 8);
      // A name the cast doesn't have is an error when speaking; "Speaker 2" means slot 2.
      const known = !state.project.cast.length || cast.has(entry.name.toLowerCase()) || numbered.test(entry.name);
      return {...entry, tone, known};
    });
  }

  function highlight(text, speakers) {
    const byName = new Map(speakers.map((speaker) => [speaker.name, speaker]));
    // A trailing newline needs a line of its own in the mirror, or the caret's last line jumps.
    return text.split("\n").map((line) => {
      let lead = "";
      const match = NAME.exec(line);
      if (match) {
        const speaker = byName.get(match[2].trim());
        const style = speaker ? ` style="color: var(--tone-${speaker.tone})"` : "";
        const unknown = speaker && !speaker.known ? " tok-unknown" : "";
        lead = `${escapeHtml(match[1])}<span class="tok-name${unknown}"${style}>${escapeHtml(match[2])}</span>${escapeHtml(match[3])}`;
        line = line.slice(match[0].length);
      }
      let out = "", cursor = 0;
      for (const token of line.matchAll(TOKEN)) {
        out += escapeHtml(line.slice(cursor, token.index));
        const kind = token[1] ? "sound" : token[2] ? "tag" : token[3] ? "cue" : "effect";
        out += `<span class="tok-${kind}">${escapeHtml(token[0])}</span>`;
        cursor = token.index + token[0].length;
      }
      return lead + out + escapeHtml(line.slice(cursor));
    }).join("\n") + "\n";
  }

  function paintEditor() {
    const area = $("script-text");
    const speakers = scriptSpeakers(area.value);
    $("script-mirror").innerHTML = highlight(area.value, speakers);
    area.style.height = "auto";
    area.style.height = `${Math.max(420, area.scrollHeight)}px`;
    $("speakers").replaceChildren(...speakers.map((speaker) => h("span", {
      class: `speaker-chip${speaker.known ? "" : " unknown"}`,
      title: speaker.known ? `${speaker.count} line${speaker.count > 1 ? "s" : ""}` : "Not in the cast: this line can't be spoken",
    }, h("i", {style: {background: toneColor(speaker.tone)}}), speaker.name,
       h("span", {class: "count"}, speaker.known ? speaker.count : "not in cast"))));
  }

  function renderWriter() {
    const writer = state.project.script.writer;
    const choose = (body) => () => saveScript(body, {now: true});
    $("speaker-mode").replaceChildren(...state.boot.speaker_modes.map((mode) => h("button", {
      type: "button", "aria-pressed": String(writer.speaker_mode === mode), onclick: choose({speaker_mode: mode}),
    }, mode)));
    $("workflows").replaceChildren(...state.boot.workflows.map((workflow) => h("button", {
      type: "button", class: "workflow", "aria-pressed": String(writer.writer_workflow === workflow.id),
      onclick: choose({writer_workflow: workflow.id}),
    }, h("strong", {}, workflow.id.replace(/^Guided \((.*)\)$/, (_, kind) => kind[0].toUpperCase() + kind.slice(1))),
       h("span", {}, workflow.note))));
    $("strict").checked = writer.strict_source;
    const batch = state.boot.batch;
    const guided = writer.writer_workflow !== state.boot.workflows[0].id;
    const smart = writer.writer_workflow === state.boot.smart_workflow;
    const count = h("span", {class: "num small"}, writer.max_batch_lines);
    const service = writerService();
    const update = (key) => (event) => storeWriterService({[key]: event.target.value});
    const backend = service.prompt_writer_backend || "Gemini";
    $("writer-more").replaceChildren(...[
      h("label", {class: "field"}, h("span", {class: "label"}, "Writer service"),
        h("select", {onchange: (event) => { storeWriterService({prompt_writer_backend: event.target.value}); renderWriter(); }},
          ["Gemini", "OpenAI-compatible"].map((name) => h("option", {selected: name === backend}, name)))),
      backend === "Gemini" ? [
        h("label", {class: "field"}, h("span", {class: "label"}, "Gemini model"),
          h("select", {onchange: update("gemini_prompt_model")}, state.boot.writer.gemini_models.map((name) =>
            h("option", {selected: name === (service.gemini_prompt_model || state.boot.writer.gemini_model)}, name)))),
        h("label", {class: "field"}, h("span", {class: "label"}, "Gemini API key (optional)"),
          h("input", {type: "password", autocomplete: "off", value: service.gemini_api_key || "", oninput: update("gemini_api_key")})),
      ] : [
        h("label", {class: "field"}, h("span", {class: "label"}, "Model"),
          h("input", {value: service.openai_model || "", placeholder: state.boot.writer.openai_models[0], oninput: update("openai_model")})),
        h("label", {class: "field"}, h("span", {class: "label"}, "API key (optional)"),
          h("input", {type: "password", autocomplete: "off", value: service.openai_api_key || "", oninput: update("openai_api_key")})),
        h("label", {class: "field"}, h("span", {class: "label"}, "Base URL (optional)"),
          h("input", {value: service.openai_base_url || "", oninput: update("openai_base_url")})),
      ],
      h("p", {class: "muted small"}, "Leave a key empty to use the server's. Keys stay in this browser tab only."),
      guided ? h("label", {class: "field"}, h("span", {class: "label"}, "Guided batch lines"),
        h("span", {class: "range-row"},
          h("input", {type: "range", min: batch.min, max: batch.max, step: 5, value: writer.max_batch_lines,
                      oninput: (event) => { count.textContent = event.target.value; },
                      onchange: (event) => saveScript({max_batch_lines: Number(event.target.value)}, {now: true})}),
          count)) : null,
      smart ? h("label", {class: "field"}, h("span", {class: "label"}, "Smart budget profile"),
        h("select", {onchange: (event) => saveScript({smart_budget_profile: event.target.value}, {now: true})},
          state.boot.budget_profiles.map((name) => h("option", {selected: name === writer.smart_budget_profile}, name)))) : null,
    ].flat().filter(Boolean));
    const rewrite = $("rewrite");
    rewrite.disabled = state.writing;
    if (!state.writing) rewrite.textContent = "Rewrite script";
  }

  function renderWriterResult() {
    const box = $("writer-result");
    const result = state.writerResult;
    box.hidden = !result;
    if (!result) return;
    box.classList.toggle("failed", result.failed);
    box.replaceChildren(
      h("div", {class: "writer-result-head"},
        h("strong", {}, result.failed ? "Script unchanged" : result.label),
        h("span", {class: "spacer"}),
        result.previous !== undefined && !result.failed
          ? h("button", {type: "button", class: "btn btn-small", onclick: undoWriter}, "Undo") : null,
        h("button", {type: "button", class: "btn btn-icon", "aria-label": "Dismiss",
                     onclick: () => { state.writerResult = null; renderWriterResult(); }}, "×")),
      h("p", {}, result.status));
  }

  function renderScript() {
    const script = state.project.script;
    const area = $("script-text");
    // Never replace what is being typed; a saved answer from the server is the same text anyway.
    if (document.activeElement !== area || state.saveTimer === null) area.value = script.text;
    if (document.activeElement !== $("title")) $("title").value = script.title;
    $("example-items").replaceChildren(...state.boot.examples.map((example) => h("button", {
      type: "button", onclick: () => loadExample(example),
    }, example.label)));
    paintEditor();
    renderWriter();
    renderWriterResult();
    renderSaveState();
  }

  function renderSaveState() {
    const labels = {pending: "Unsaved changes", saving: "Saving…", saved: "All changes saved", failed: "Not saved"};
    $("save-state").textContent = labels[state.saving] || "";
  }

  // Script edits save on their own, a moment after typing stops.
  let pendingScript = {};
  function saveScript(body, {now = false} = {}) {
    pendingScript = {...pendingScript, ...body};
    state.saving = "pending";
    renderSaveState();
    clearTimeout(state.saveTimer);
    state.saveTimer = setTimeout(() => flushScript(), now ? 0 : 800);
  }

  async function flushScript() {
    if (state.saveTimer === null) return;
    clearTimeout(state.saveTimer);
    state.saveTimer = null;
    const body = pendingScript;
    pendingScript = {};
    if (!Object.keys(body).length || !state.project) return;
    const name = state.project.name;
    state.saving = "saving";
    renderSaveState();
    try {
      const project = await api(`/project/script?name=${encodeURIComponent(name)}`, {method: "PATCH", body});
      if (state.project.name !== name) return;
      state.saving = state.saveTimer === null ? "saved" : "pending";
      const text = $("script-text").value;
      Object.assign(state.project, {...project, run: state.project.run});
      // Keep what was typed while the save was on its way.
      if (state.saveTimer !== null) state.project.script.text = text;
      renderHeader();
      renderSteps();
      if (body.script_mode || body.text !== undefined) queueEstimate();
      if (!("text" in body) && !("title" in body)) renderScript();
      else renderSaveState();
    } catch (error) {
      state.saving = "failed";
      renderSaveState();
      toast(`Not saved: ${error.message}`);
    }
  }

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
    if (!failed) {
      Object.assign(state.project, {...project, run: state.project.run});
      $("script-text").value = state.project.script.text;
      queueEstimate(0);
    }
    renderAll();
  }

  async function rewrite() {
    if (state.writing) return;
    await flushScript();
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
      state.writerResult = {status: error.message, failed: true};
      renderWriterResult();
    } finally {
      clearInterval(timer);
      state.writing = false;
      renderWriter();
    }
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
    await flushScript();
    try {
      const {prompt} = await api(named("/project/writer-prompt"), {method: "POST", body: {}});
      await navigator.clipboard.writeText(prompt);
      toast("Writer prompt copied. Paste it into any chat model.");
    } catch (error) {
      toast(`Could not copy the prompt: ${error.message}`);
    }
  }

  async function applyReply() {
    await flushScript();
    try {
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
    await flushScript();
    try {
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

  function castRunning() { return Boolean(state.castRun?.active) || autoRunning(); }
  /** The run making the cast: a cast job, or Auto while it makes voices and sounds. */
  function castJob() {
    if (state.castRun?.active) return state.castRun;
    if (autoRunning() && state.run.stage === "cast") return {...state.run, target: "prepare"};
    return state.castRun;
  }
  function slotNote(kind, slot) {
    const run = castJob();
    return run?.slots?.[`${kind}:${slot}`] || null;
  }
  function slotWorking(kind, slot) {
    const run = state.castRun;
    if (!run?.active) return false;
    if (run.target === `${kind}:${slot}` || (kind === "voice" && run.target === `transcribe:${slot}`)) return true;
    return run.target === "prepare" && Boolean(slotNote(kind, slot)) && !/^(Voice generated|Sound generated|Skipped)/.test(slotNote(kind, slot));
  }

  /** Each speaker's tone and script lines, by name, from the saved script. */
  function speakerLines() {
    const text = state.project.script.text;
    const tones = new Map(scriptSpeakers(text).map((speaker) => [speaker.name.toLowerCase(), speaker.tone]));
    const lines = new Map();
    for (const line of text.split("\n")) {
      const match = NAME.exec(line);
      if (!match) continue;
      const key = match[2].trim().toLowerCase();
      if (!lines.has(key)) lines.set(key, []);
      lines.get(key).push(line.slice(match[0].length).replace(TOKEN, "").replace(/\s+/g, " ").trim());
    }
    return {tones, lines};
  }

  function missingItems() {
    const project = state.project;
    const voices = project.speakers.filter((speaker) => !speaker.ready && (speaker.save_name || speaker.description));
    const advanced = project.script.script_mode === state.boot.advanced;
    const sounds = advanced ? project.sounds.filter((sound) => !sound.ready) : [];
    return {voices, sounds};
  }

  function pickSlot(kind, slot) {
    flushSlots();
    state.castPick = {kind, slot};
    renderCastLists();
    renderSlotEditor(true);
  }

  function ensurePick() {
    const project = state.project;
    const pick = state.castPick;
    const exists = pick && (pick.kind === "voice" ? project.speakers : project.sounds).some((item) => item.slot === pick.slot);
    if (exists) return;
    const first = project.speakers.find((speaker) => !speaker.ready) ?? project.speakers[0];
    state.castPick = first ? {kind: "voice", slot: first.slot} : null;
  }

  function renderCast() {
    ensurePick();
    const settings = state.project.cast_settings;
    const select = $("speaker-group");
    const groups = library?.groups?.length ? library.groups : [settings.speaker_group || "Global"];
    select.replaceChildren(...[...new Set([...groups, settings.speaker_group])].map((group) =>
      h("option", {value: group, selected: group === settings.speaker_group}, group)));
    renderCastLists();
    renderSlotEditor(false);
    renderGeneratorSettings();
  }

  function renderCastLists() {
    const project = state.project;
    const {tones, lines} = speakerLines();
    const ready = project.speakers.filter((speaker) => speaker.ready).length;
    $("voices-count").textContent = `${ready} of ${project.speakers.length} ready`;
    const pick = state.castPick;
    const cards = project.speakers.map((speaker) => {
      const key = speaker.name.toLowerCase();
      const tone = tones.get(key) ?? String(speaker.slot % 8);
      const count = lines.get(key)?.length ?? 0;
      const working = slotWorking("voice", speaker.slot);
      const source = speaker.source === "saved" ? `Saved voice${speaker.missing ? " (missing)" : ""}`
        : speaker.source === "clip" ? (speaker.transcript ? "Own clip · transcript ready" : "Own clip · needs a transcript")
        : speaker.description ? "Has a description" : "Needs a description";
      return h("button", {
        type: "button", class: `cast-card${working ? " working" : ""}`,
        "aria-pressed": String(pick?.kind === "voice" && pick.slot === speaker.slot),
        onclick: () => pickSlot("voice", speaker.slot),
      },
      h("span", {class: "cast-card-head"}, h("i", {style: {background: toneColor(tone)}}), h("strong", {}, speaker.name),
        h("span", {}, count ? `${count} line${count > 1 ? "s" : ""}` : "no lines")),
      h("span", {class: speaker.ready ? "ready" : "missing"}, working ? "Working…" : speaker.ready ? "Voice ready" : "No voice yet"),
      h("span", {class: "note"}, working ? (slotNote("voice", speaker.slot) || "Waiting for the GPU…") : source));
    });
    const canAdd = project.speakers.length < state.boot.slots.voices;
    $("cast-grid").replaceChildren(...cards, canAdd ? h("button", {type: "button", class: "cast-add", onclick: () => resizeSlots("voice", true)},
      svg("M12 5v14M5 12h14"), "Add speaker") : null);

    const advanced = project.script.script_mode === state.boot.advanced;
    const sounds = project.sounds;
    $("sounds-count").textContent = sounds.length ? `${sounds.filter((sound) => sound.ready).length} of ${sounds.length} ready` : "";
    $("sounds-note").textContent = advanced ? "Missing sounds are also made while speaking."
                                            : "Basic scripts don't use sounds; switch to Advanced to hear them.";
    const table = $("sound-table");
    if (!sounds.length) {
      table.replaceChildren(h("tbody", {}, h("tr", {}, h("td", {class: "sounds-empty"},
        advanced ? "No sounds yet. Sound tags in an Advanced script get a slot here, or add one."
                 : "No sounds. An Advanced script's sound tags get a slot here."))));
    } else {
      table.replaceChildren(
        h("thead", {}, h("tr", {}, ["Sound", "Kind", "Length", "Clip"].map((label) => h("th", {scope: "col"}, label)))),
        h("tbody", {}, sounds.map((sound) => {
          const selected = pick?.kind === "sound" && pick.slot === sound.slot;
          const working = slotWorking("sound", sound.slot);
          const kind = sound.loop === true ? ["kind-loop", "Loop"] : sound.loop === false ? ["kind-shot", "One-shot"] : ["kind-unused", "Not in script"];
          const cell = (...content) => h("td", {}, h("button", {type: "button", class: "row", tabindex: "-1",
                                                                onclick: () => pickSlot("sound", sound.slot)}, ...content));
          return h("tr", {"aria-selected": String(selected)},
            h("td", {}, h("button", {type: "button", class: "row", onclick: () => pickSlot("sound", sound.slot)},
              h("strong", {}, sound.name || `Sound ${sound.slot + 1}`))),
            cell(h("span", {class: `kind ${kind[0]}`}, kind[1])),
            cell(h("span", {class: "num muted"}, `${sound.duration} s`)),
            cell(h("span", {class: working ? "" : sound.ready ? "state-ready" : "state-missing"},
              working ? "Working…" : sound.ready ? (sound.source === "saved" ? "Saved sound" : "Ready") : "Missing")));
        })));
    }
    $("add-sound").disabled = sounds.length >= state.boot.slots.sounds;
  }

  function field(label, input) {
    return h("label", {class: "field"}, h("span", {class: "label"}, label), input);
  }

  function renderSlotEditor(force) {
    const box = $("slot-editor");
    // Leave the editor alone while something in it is being typed in.
    if (!force && box.contains(document.activeElement) && document.activeElement.matches("input, textarea")) return;
    const pick = state.castPick;
    if (!pick) {
      box.replaceChildren(h("h2", {}, "Pick a speaker or sound"), h("p", {class: "muted small"}, "Its voice or clip is made and checked here."));
      return;
    }
    box.replaceChildren(...(pick.kind === "voice" ? voiceEditor(pick.slot) : soundEditor(pick.slot)).filter(Boolean));
  }

  function noteBox(kind, slot) {
    const note = slotNote(kind, slot);
    if (!note) return null;
    return h("p", {class: `slot-note${/failed/i.test(note) ? " error" : ""}`}, note);
  }

  function sourceSwitch(kind, slot, current, labels) {
    return h("div", {class: "seg", role: "group", "aria-label": kind === "voice" ? "Voice source" : "Sound source"},
      labels.map(([id, label]) => h("button", {
        type: "button", "aria-pressed": String(current === id),
        onclick: () => { state.castSource = {...(state.castSource || {}), [`${kind}:${slot}`]: id}; renderSlotEditor(true); },
      }, label)));
  }

  function voiceEditor(slot) {
    const speaker = state.project.speakers.find((item) => item.slot === slot);
    if (!speaker) return [];
    const {tones, lines} = speakerLines();
    const key = speaker.name.toLowerCase();
    const tone = tones.get(key) ?? String(slot % 8);
    const chosen = state.castSource?.[`voice:${slot}`] ?? (speaker.source === "clip" && !speaker.description ? "upload" : speaker.source === "saved" ? "saved" : "describe");
    const working = slotWorking("voice", slot);
    const busy = castRunning() || !state.boot.gpu;
    const qwen = state.project.cast_settings.voice_engine === state.boot.qwen_engine;
    const save = (body) => saveSlot("voice", slot, body);
    const parts = [
      h("div", {class: "slot-head"},
        h("i", {style: {background: toneColor(tone)}}),
        h("input", {value: speaker.save_name || speaker.saved_name || "", placeholder: `Speaker ${slot}`, "aria-label": "Speaker name",
                    disabled: speaker.source === "saved", oninput: (event) => save({save_name: event.target.value})}),
        h("span", {class: `slot-chip ${speaker.ready ? "ready" : "missing"}`}, speaker.ready ? "Voice ready" : "No voice")),
      sourceSwitch("voice", slot, chosen, [["describe", "Describe"], ["upload", "Upload clip"], ["saved", "Saved voice"]]),
    ];
    if (chosen === "describe") {
      parts.push(
        field("Voice description", h("textarea", {rows: 4, value: speaker.description, disabled: speaker.source === "saved",
          placeholder: "Who they are and how they sound: age, tone, pace, accent.", oninput: (event) => save({description: event.target.value})})),
        field("Line it reads", h("input", {value: speaker.preview_text, placeholder: "A sentence or two in their voice",
          oninput: (event) => save({preview_text: event.target.value})})),
        qwen ? field("Language", h("select", {onchange: (event) => save({language: event.target.value})},
          state.boot.languages.map((language) => h("option", {selected: language === speaker.language}, language)))) : null,
        h("div", {class: "row-buttons"},
          h("button", {type: "button", class: "btn btn-primary", "data-gpu": true, disabled: busy, onclick: () => make("voice", slot)},
            working ? "Generating…" : speaker.source === "clip" ? "Generate a new voice" : "Generate voice")));
    } else if (chosen === "upload") {
      parts.push(h("div", {class: "row-buttons"},
        h("button", {type: "button", class: "btn", onclick: () => chooseClip("voice", slot)},
          speaker.source === "clip" ? "Upload a different clip" : "Upload a clip"),
        h("span", {class: "muted small"}, "A few seconds of clean speech.")));
    } else {
      const options = library?.speakers ?? [];
      parts.push(options.length
        ? field("Saved voice", h("select", {onchange: (event) => event.target.value && useSaved("voice", slot, event.target.value)},
            h("option", {value: ""}, "Choose a saved voice…"),
            options.map((option) => h("option", {value: option.key, selected: option.key === speaker.preset},
              option.group === "Global" ? option.name : `${option.name} · ${option.group}`))))
        : h("p", {class: "muted small"}, library?.error || "No saved voices yet. Save a speaker's voice to reuse it in other projects."));
    }
    if (speaker.clip) {
      parts.push(h("audio", {controls: true, preload: "none", src: speaker.clip}));
      if (speaker.source === "clip") {
        parts.push(field("Exact transcript of the clip", h("textarea", {rows: 3, value: speaker.transcript,
          placeholder: "Every word the clip says", oninput: (event) => save({transcript: event.target.value})})));
        parts.push(h("div", {class: "row-buttons"},
          h("button", {type: "button", class: "btn btn-small", "data-gpu": true, disabled: busy, onclick: () => make("transcribe", slot)},
            state.castRun?.active && state.castRun.target === `transcribe:${slot}` ? "Transcribing…" : "Transcribe"),
          h("button", {type: "button", class: "btn btn-small", onclick: () => saveToLibrary("voice", slot)},
            `Save to ${state.project.cast_settings.speaker_group}`),
          h("button", {type: "button", class: "btn btn-small", onclick: () => removeClip("voice", slot)}, "Remove clip")));
      } else {
        parts.push(h("p", {class: "muted small"}, `Uses the saved voice “${speaker.saved_name}”. Its transcript: ${speaker.transcript || "none"}`),
          h("div", {class: "row-buttons"}, h("button", {type: "button", class: "btn btn-small",
            onclick: () => saveSlot("voice", slot, {source: "upload"}, {now: true})}, "Stop using the saved voice")));
      }
    } else if (speaker.missing) {
      parts.push(h("p", {class: "slot-note error"}, "Its saved voice is no longer in the library. Choose another or make a new one."));
    }
    parts.push(noteBox("voice", slot));
    const spoken = lines.get(key) ?? [];
    parts.push(h("div", {class: "slot-lines"},
      h("span", {class: "label"}, spoken.length ? `${speaker.name}'s lines` : `${speaker.name} has no lines in the script`),
      spoken.slice(0, 3).map((line) => h("p", {}, line)),
      spoken.length > 3 ? h("span", {class: "muted small"}, `and ${spoken.length - 3} more`) : null));
    if (slot === state.project.speakers.at(-1).slot && state.project.speakers.length > 1) {
      parts.push(h("button", {type: "button", class: "danger-link", onclick: () => resizeSlots("voice", false)}, "Remove this speaker slot"));
    }
    return parts;
  }

  function soundEditor(slot) {
    const sound = state.project.sounds.find((item) => item.slot === slot);
    if (!sound) return [];
    const chosen = state.castSource?.[`sound:${slot}`] ?? (sound.source === "saved" ? "saved" : sound.source === "clip" && !sound.description ? "upload" : "describe");
    const working = slotWorking("sound", slot);
    const busy = castRunning() || !state.boot.gpu;
    const save = (body) => saveSlot("sound", slot, body);
    const kind = sound.loop === true ? "Looped in the script" : sound.loop === false ? "Plays once in the script" : "Not in the script";
    const parts = [
      h("div", {class: "slot-head"},
        h("i", {style: {background: sound.loop ? "var(--bed)" : "var(--cue)"}}),
        h("input", {value: sound.save_name, placeholder: "Sound name, as the script's tag names it", "aria-label": "Sound name",
                    disabled: sound.source === "saved", oninput: (event) => save({save_name: event.target.value})}),
        h("span", {class: `slot-chip ${sound.ready ? "ready" : "missing"}`}, sound.ready ? "Clip ready" : "No clip")),
      h("p", {class: "muted small"}, kind),
      sourceSwitch("sound", slot, chosen, [["describe", "Describe"], ["upload", "Upload clip"], ["saved", "Saved sound"]]),
    ];
    if (chosen === "describe") {
      parts.push(
        field("What it sounds like", h("textarea", {rows: 3, value: sound.description, disabled: sound.source === "saved",
          placeholder: "Steady rain on canvas awnings, distant traffic", oninput: (event) => save({description: event.target.value})})),
        field("Length in seconds", h("input", {type: "number", min: 1, max: 47, value: sound.duration, disabled: sound.source === "saved",
          oninput: (event) => event.target.value && save({duration: Number(event.target.value)})})),
        h("div", {class: "row-buttons"},
          h("button", {type: "button", class: "btn btn-primary", "data-gpu": true, disabled: busy, onclick: () => make("sound", slot)},
            working ? "Generating…" : sound.ready ? "Generate a new clip" : "Generate sound")));
    } else if (chosen === "upload") {
      parts.push(h("div", {class: "row-buttons"}, h("button", {type: "button", class: "btn", onclick: () => chooseClip("sound", slot)},
        sound.source === "clip" ? "Upload a different clip" : "Upload a clip")));
    } else {
      const options = library?.sounds ?? [];
      parts.push(options.length
        ? field("Saved sound", h("select", {onchange: (event) => event.target.value && useSaved("sound", slot, event.target.value)},
            h("option", {value: ""}, "Choose a saved sound…"),
            options.map((option) => h("option", {value: option.key, selected: option.key === sound.preset}, option.name))))
        : h("p", {class: "muted small"}, library?.error || "No saved sounds yet. Save a clip to reuse it."));
    }
    if (sound.clip) {
      parts.push(h("audio", {controls: true, preload: "none", src: sound.clip}));
      parts.push(h("div", {class: "row-buttons"},
        sound.source === "clip" ? h("button", {type: "button", class: "btn btn-small", onclick: () => saveToLibrary("sound", slot)}, "Save to library") : null,
        sound.source === "clip" ? h("button", {type: "button", class: "btn btn-small", onclick: () => removeClip("sound", slot)}, "Remove clip")
          : h("button", {type: "button", class: "btn btn-small", onclick: () => saveSlot("sound", slot, {source: "upload"}, {now: true})},
              "Stop using the saved sound")));
    }
    parts.push(noteBox("sound", slot));
    return parts;
  }

  function renderGeneratorSettings() {
    const settings = state.project.cast_settings;
    const save = (key, parse = (event) => event.target.value) => (event) => saveCastSettings({[key]: parse(event)});
    $("generator-settings").replaceChildren(
      field("Voice generator", h("select", {onchange: save("voice_engine")},
        state.boot.voice_engines.map((engine) => h("option", {selected: engine === settings.voice_engine}, engine)))),
      h("p", {class: "muted small"}, "Qwen3-TTS VoiceDesign designs a voice from its description; Drama 3 designs it on SGLang-Omni with the speech settings."),
      field("Transcript model", h("select", {onchange: save("asr_model")},
        state.boot.asr_models.map((model) => h("option", {selected: model === settings.asr_model}, model)))),
      h("label", {class: "check"}, h("input", {type: "checkbox", checked: settings.skip_existing,
        onchange: save("skip_existing", (event) => event.target.checked)}), "Make the missing skips speakers with a voice"),
      h("label", {class: "check"}, h("input", {type: "checkbox", checked: settings.skip_existing_sounds,
        onchange: save("skip_existing_sounds", (event) => event.target.checked)}), "Make the missing skips sounds with a clip"));
  }

  function renderCastActions() {
    const {voices, sounds} = missingItems();
    const total = voices.length + sounds.length;
    const running = castRunning();
    const title = $("action-title");
    const detail = $("action-detail");
    detail.classList.remove("error");
    if (running) {
      title.textContent = autoRunning() ? `Auto · ${AUTO_STAGES[state.run.stage] || "Starting"}` : "Making voices and sounds…";
      detail.textContent = (autoRunning() ? state.run.status : state.castRun?.status) || "Waiting for the GPU…";
    } else if (state.castRun?.error) {
      title.textContent = "That didn't work";
      detail.textContent = state.castRun.error;
      detail.classList.add("error");
    } else if (total) {
      title.textContent = `${total} thing${total > 1 ? "s" : ""} to make before speaking`;
      detail.textContent = [...voices.map((speaker) => `${speaker.name}'s voice`), ...sounds.map((sound) => sound.name)].join(" · ");
    } else {
      title.textContent = state.project.speakers.some((speaker) => speaker.ready) ? "Cast ready" : "No cast yet";
      detail.textContent = state.project.speakers.some((speaker) => speaker.ready)
        ? "Every speaker has a voice. Speak the project in Mix & review."
        : "Rewrite the script to fill the cast, or add speakers and give them voices.";
    }
    const make = $("make-missing");
    make.hidden = running || !total;
    make.disabled = !state.boot.gpu;
    make.title = state.boot.gpu ? "" : NO_GPU;
    if (!state.boot.gpu && total && !running) {
      make.hidden = true;
      detail.textContent += " · Make them when you open the project in Studio.";
    }
    make.textContent = `Make the missing ${total}`;
    $("cast-interrupt").hidden = !state.castRun?.active;  // Auto has its own, in the top bar.
    $("to-script").hidden = false;
    $("to-mix").hidden = false;
    $("to-mix").className = `btn${total || running ? "" : " btn-primary"}`;
    $("to-mix").textContent = "Next: mix & review";
  }

  // Slot edits save on their own, a moment after typing stops.
  const slotEdits = new Map();
  let slotTimer = null;
  function saveSlot(kind, slot, body, {now = false} = {}) {
    const key = `${kind}:${slot}`;
    slotEdits.set(key, {...(slotEdits.get(key) || {}), ...body});
    state.saving = "pending";
    renderSaveState();
    clearTimeout(slotTimer);
    slotTimer = setTimeout(flushSlots, now ? 0 : 700);
  }

  async function flushSlots() {
    clearTimeout(slotTimer);
    slotTimer = null;
    const edits = [...slotEdits.entries()];
    slotEdits.clear();
    if (!edits.length || !state.project) return;
    const name = state.project.name;
    state.saving = "saving";
    renderSaveState();
    try {
      let project = null;
      for (const [key, body] of edits) {
        const [kind, slot] = key.split(":");
        project = await api(`/project/slot?name=${encodeURIComponent(name)}&kind=${kind}&slot=${slot}`, {method: "PATCH", body});
      }
      if (state.project.name !== name) return;
      castAnswered(project);
      state.saving = slotEdits.size ? "pending" : "saved";
      renderSaveState();
    } catch (error) {
      state.saving = "failed";
      renderSaveState();
      toast(`Not saved: ${error.message}`);
    }
  }

  /** The project changed on the server: keep the screen, refresh what it shows. */
  function castAnswered(project, {editor = false} = {}) {
    if (!project) return;
    const {message, ...rest} = project;
    Object.assign(state.project, {...rest, run: state.project.run});
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
    await flushSlots();
    try {
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
    await flushSlots();
    try {
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

  let castPollTimer = null;
  function pollCast() {
    clearTimeout(castPollTimer);
    castPollTimer = setTimeout(async () => {
      const name = state.project.name;
      try {
        const project = await api(`/project?name=${encodeURIComponent(name)}`);
        if (state.project.name !== name) return;
        const before = state.castRun;
        castAnswered(project);
        if (state.screen !== "cast") renderSteps();
        if (project.cast_run?.active) { pollCast(); return; }
        if (before?.active) toast(project.cast_run?.error ? `Stopped: ${project.cast_run.error}` : (project.cast_run?.status || "Done."));
        renderSlotEditor(true);
      } catch (error) {
        toast(error.message);
        pollCast();
      }
    }, POLL_MS);
  }

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
      h("p", {}, "Start one here: write or paste a story, and the writer turns it into a script."),
      h("p", {}, h("button", {type: "button", class: "btn btn-primary", onclick: newProject}, "New project")));
  }

  async function start() {
    applyTheme();
    matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", applyTheme);
    // In the studio's Soundstage tab, follow its light/dark switch as it changes.
    try {
      if (window.parent !== window) {
        new MutationObserver(applyTheme).observe(window.parent.document.body, {attributes: true, attributeFilter: ["class"]});
      }
    } catch { /* Framed by another origin. */ }
    $("project").addEventListener("change", (event) => loadProject(event.target.value).catch((error) => toast(error.message)));
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

  start();
})();
