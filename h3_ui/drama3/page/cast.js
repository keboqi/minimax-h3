// Cast view and local controls for Soundstage.
export function createCast({state, $, h, toneColor, svg, NO_GPU, AUTO_STAGES, autoRunning, flushSlots, saveSlot, reviewSlotProposal, saveCastSettings, resizeSlots, useSaved, make, chooseClip, saveToLibrary, removeClip, scriptSpeakers, NAME, TOKEN, getLibrary}) {
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
    const groups = getLibrary()?.groups?.length ? getLibrary().groups : [settings.speaker_group || "Global"];
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
    const slot = (pick.kind === "voice" ? state.project.speakers : state.project.sounds).find((item) => item.slot === pick.slot);
    if (slot?.proposal) box.append(h("button", {type: "button", class: "btn", onclick: () => reviewSlotProposal(pick.kind, pick.slot)}, "Review saved result"));
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
      const options = getLibrary()?.speakers ?? [];
      parts.push(options.length
        ? field("Saved voice", h("select", {onchange: (event) => event.target.value && useSaved("voice", slot, event.target.value)},
            h("option", {value: ""}, "Choose a saved voice…"),
            options.map((option) => h("option", {value: option.key, selected: option.key === speaker.preset},
              option.group === "Global" ? option.name : `${option.name} · ${option.group}`))))
        : h("p", {class: "muted small"}, getLibrary()?.error || "No saved voices yet. Save a speaker's voice to reuse it in other projects."));
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
      const options = getLibrary()?.sounds ?? [];
      parts.push(options.length
        ? field("Saved sound", h("select", {onchange: (event) => event.target.value && useSaved("sound", slot, event.target.value)},
            h("option", {value: ""}, "Choose a saved sound…"),
            options.map((option) => h("option", {value: option.key, selected: option.key === sound.preset}, option.name))))
        : h("p", {class: "muted small"}, getLibrary()?.error || "No saved sounds yet. Save a clip to reuse it."));
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

  return {castRunning, castJob, slotNote, slotWorking, speakerLines, missingItems, pickSlot, ensurePick, renderCast, renderCastLists, field, renderSlotEditor, noteBox, sourceSwitch, voiceEditor, soundEditor, renderGeneratorSettings, renderCastActions};
}
