// Mix view and local controls for Soundstage.
export function createMix({state, $, h, API, STATUS_LABELS, FILTERS, ICONS, takeAudio, mixVideo, clock, isRunning, lineTakes, named, pick, rows, saveSettings, saveVideo, seek, setMediaMode, speak, speakerTone, svg, sync, toggleTake, toneColor, videoFileButton, visualSource, getPlayer}) {
  function setSubtitles(on) {
    state.project.video.subtitles = on;
    saveVideo({video_subtitles: on});
    const box = $("subtitles-on");
    if (box) box.checked = on;
    for (const button of document.querySelectorAll("#subtitle-size button")) button.disabled = !on || sizeLimit(button);
    refreshVideoFile();
    sync();
  }

  // The saved video has its subtitles burned in, so changing them asks for a new one.
  const refreshVideoFile = () => $("video-file")?.replaceWith(videoFileButton());

  function sizeLimit(button) {
    const [low, high] = state.boot.video.subtitle_sizes;
    const size = state.project.video.subtitle_size;
    return button.dataset.step === "-1" ? size <= low : size >= high;
  }

  function setSubtitleSize(direction) {
    const [low, high, step] = state.boot.video.subtitle_sizes;
    const video = state.project.video;
    video.subtitle_size = Math.min(high, Math.max(low, video.subtitle_size + direction * step));
    $("video-stage")?.style.setProperty("--caption-scale", String(video.subtitle_size / 100));
    $("subtitle-size-value").textContent = `${video.subtitle_size}%`;
    for (const button of document.querySelectorAll("#subtitle-size button")) button.disabled = sizeLimit(button);
    refreshVideoFile();
    saveVideo({video_subtitle_size: video.subtitle_size}, {later: true});
  }

  function subtitleControls() {
    const video = state.project.video;
    const sizeButton = (direction, label, text) => {
      const button = h("button", {type: "button", "data-step": String(direction), "aria-label": label, title: label,
                                  onclick: () => setSubtitleSize(direction)}, text);
      button.disabled = !video.subtitles || sizeLimit(button);
      return button;
    };
    return h("div", {class: "subtitle-controls"},
      h("label", {class: "check", title: "Show or hide the subtitles (C)"},
        h("input", {type: "checkbox", id: "subtitles-on", checked: video.subtitles,
                    onchange: (event) => setSubtitles(event.target.checked)}),
        "Subtitles"),
      h("div", {class: "seg subtitle-size", id: "subtitle-size", role: "group", "aria-label": "Subtitle size"},
        sizeButton(-1, "Smaller subtitles", "A−"),
        h("span", {class: "num small", id: "subtitle-size-value", "aria-live": "polite"}, `${video.subtitle_size}%`),
        sizeButton(1, "Larger subtitles", "A+")));
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
                                    onclick: () => (getPlayer().paused ? getPlayer().play() : getPlayer().pause())}, svg(ICONS.play, 18, true));
    const head = h("div", {class: "player-head"},
      playButton,
      h("div", {class: "player-title"},
        h("strong", {}, result.title || state.project.title || state.project.name),
        h("span", {class: "muted small num", id: "clock"}, `0:00 / ${clock(result.duration)}`)),
      h("span", {class: "spacer"}),
      visualSource() ? h("div", {class: "seg", role: "group", "aria-label": "Show"},
        ["audio", "video"].map((mode) => h("button", {type: "button", "aria-pressed": String(state.mediaMode === mode),
                                                       onclick: () => setMediaMode(mode)}, mode === "audio" ? "Audio" : "Video"))) : null,
      state.mediaMode === "video" ? subtitleControls() : null,
      state.mediaMode === "video"
        ? videoFileButton()
        : h("a", {class: "btn btn-small", href: `${API}${named("/audio")}`, download: result.file}, "Download"));
    let stage = null;
    if (state.mediaMode === "video") {
      const playing = () => (getPlayer().paused ? getPlayer().play() : getPlayer().pause());
      const stills = visualSource() === "stills";
      // The stills slide along the mix's audio; an older mix with only a rendered video plays that.
      const picture = stills
        ? h("div", {class: "stills", onclick: playing},
            h("img", {class: "still", id: "still-back", alt: ""}), h("img", {class: "still", id: "still-front", alt: ""}))
        : mixVideo;
      mixVideo.hidden = stills;
      if (stills) document.body.append(mixVideo);
      else mixVideo.onclick = playing;
      stage = h("div", {class: "video-stage", id: "video-stage",
                        style: {"--caption-scale": String(state.project.video.subtitle_size / 100)}},
                picture, h("p", {class: "video-caption", id: "video-caption", "aria-live": "off"}));
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
  return {setSubtitles, sizeLimit, setSubtitleSize, subtitleControls, renderPlayer, wordNodes, renderFilters, renderLines, renderInspector, renderSettings};
}
