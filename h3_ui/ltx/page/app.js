"use strict";
// The LTX 2.5 page. The server owns the draft, the options and their policy; this draws what it answers.
(() => {
  const API = "/ltx/api";
  const {$, h, createClient, createToast, seg, rangeOf, presetText: sharedPresetText, openStudio, decisionScoreChips, decisionScoreRadar} = globalThis.H3PageCommon;
  const {request, post} = createClient(API);
  const {toast, guard} = createToast();

  const app = {boot: null, s: null, runs: {}, workflows: [], missing: 0, take: 0, suggestion: null, previous: null,
               enhanceNote: "", polling: false, live: null};

  const take = (answer) => {
    if (!answer) return;
    if (answer.state) app.s = answer.state;
    if (answer.runs) app.runs = answer.runs;
    if (answer.workflows) { app.workflows = answer.workflows; app.missing = answer.missing; }
    // A writer may finish before the first poll, so its answer is taken wherever it arrives.
    const written = app.runs.enhance;
    if (app.enhancePending && written && !written.active) {
      app.enhancePending = false;
      app.enhanceNote = "";
      if (written.suggestion) app.suggestion = {text: written.suggestion, original: written.original};
    }
    render();
    if (Object.values(app.runs).some((run) => run?.active)) poll();
  };
  const change = guard(async (action, changes = {}, name) => {
    await saveDraft();
    take(await post("/settings", {action, changes, name}));
  });
  const changeWriter = guard(async (changes) => {
    await saveDraft();
    take(await post("/writer", {changes}));
  });
  const saveUi = (patch) => {
    app.s.ui = {...app.s.ui, ...patch};
    post("/ui", {ui: app.s.ui}).catch(() => { /* Remembering the open group is a convenience. */ });
  };

  // Controls ----------------------------------------------------------------
  const fixed = (field, value) => Number(value).toFixed(field.step && field.step < 1 ? (field.step < 0.1 ? 2 : 1) : 0);
  // A slider sends its value when let go; while it moves only its label follows.
  const slider = (field, label, {suffix = "", disabled = false} = {}) => {
    const out = h("span", {class: "mono", text: `${fixed(field, field.value)}${suffix}`});
    const input = h("input", {type: "range", min: field.minimum, max: field.maximum, step: field.step, value: field.value,
      disabled, "aria-label": label, oninput: (event) => { out.textContent = `${fixed(field, event.target.value)}${suffix}`; },
      onchange: (event) => change("edit", {[field.name]: Number(event.target.value)})});
    return [input, out];
  };
  const commitOf = (field) => (value) => field.store === "writer" ? changeWriter({[field.name]: value}) : change("edit", {[field.name]: value});
  const presetText = (field) => sharedPresetText(field, {booleanLabels: false});
  const control = (field) => {
    const commit = commitOf(field);
    const {kind, label, value} = field;
    const disabled = field.disabled || (app.s.locked && ["fps", "width", "height"].includes(field.name));
    if (kind === "bool") return [seg(h("div", {class: "seg", role: "group", "aria-label": label}), [["Off", false], ["On", true]], Boolean(value), commit, {disabled})];
    if (kind === "choice") {
      const long = field.choices.length > 4 || field.choices.reduce((sum, [name]) => sum + name.length, 0) > 56;
      if (!long) return [seg(h("div", {class: "seg", role: "group", "aria-label": label}), field.choices, value, commit, {disabled})];
      const index = field.choices.findIndex(([, choice]) => choice === value);
      return [h("select", {class: "field", "aria-label": label, disabled, onchange: (event) => commit(field.choices[Number(event.target.value)][1])},
        index < 0 ? h("option", {value: "", selected: true, text: "Not set"}) : null,
        field.choices.map(([name], position) => h("option", {value: String(position), selected: position === index, text: name})))];
    }
    if ((kind === "int" || kind === "num") && field.name === "cfg") return slider(field, label);
    if (kind === "int" || kind === "num") {
      return [h("input", {class: "field num mono", type: "text", inputMode: "decimal", value: String(value), disabled, "aria-label": label,
        onchange: async (event) => {
          const input = event.target;
          const sent = input.value;
          const parsed = Number(sent);
          if (sent.trim() === "" || !Number.isFinite(parsed)) {
            toast(`${label} must be a number.`, true);
            input.value = String(value);
            return;
          }
          await commit(parsed);
          if (input.value === sent) input.value = String(app.s.options[field.name]);
        }}), rangeOf(field) ? h("span", {class: "small muted", text: rangeOf(field)}) : null,
        field.name === "seed" && value !== -1 ? h("button", {type: "button", class: "btn compact", text: "Random", onclick: () => commit(-1)}) : null];
    }
    return [h("input", {class: "field text", type: kind === "password" ? "password" : "text", value: value ?? "", placeholder: field.placeholder, autocomplete: "off",
                        disabled, "aria-label": label, onchange: (event) => commit(event.target.value)})];
  };
  const fineRow = (field) => h("div", {class: "fine-row"},
    h("div", {class: "about"},
      h("span", {class: "label"}, field.label,
        field.changed ? h("span", {class: "chip warn", text: `default: ${presetText(field)}`}) : null),
      field.hint ? h("span", {class: "hint", text: field.hint}) : null),
    h("div", {class: "control"}, control(field)));

  // Fine-tune ---------------------------------------------------------------
  const groupBadge = (index) => {
    if (index === app.boot.workflow_group) return app.missing ? [`${app.missing} missing`, "warn"] : null;
    const count = app.s.changed[index];
    return count ? [String(count), "warn"] : null;
  };
  const openGroup = (index) => {
    saveUi({fine: true, group: index});
    render();
    $("fine").scrollIntoView({block: "nearest", behavior: "smooth"});
  };
  const workflowRow = (row) => {
    const busy = app.runs.models?.active;
    const ready = !row.missing.length;
    return h("div", {class: "workflow"},
      h("div", {class: "about"},
        h("span", {class: "label", text: row.label}),
        h("span", {class: "hint", text: `${row.description} ${row.inputs}`}),
        ready ? null : h("span", {class: "hint", text: `Needs ${row.missing.join(", ")}`})),
      h("div", {class: "control"},
        h("span", {class: `chip ${ready ? "good" : "warn"}`, text: ready ? "Ready" : `${row.missing.length} of ${row.total} missing`}),
        h("a", {class: "btn compact", href: row.json, download: "", text: "Workflow JSON"}),
        ready ? null : h("button", {type: "button", class: "btn compact", text: "Download", disabled: busy || !app.boot.gpu,
          onclick: () => prepareModels(row.label)})));
  };
  const renderFine = () => {
    const open = Boolean(app.s.ui.fine);
    const total = Object.values(app.s.changed).reduce((sum, count) => sum + count, 0);
    $("fine-count").hidden = !total;
    $("fine-count").textContent = `${total} changed`;
    $("fine-toggle").setAttribute("aria-expanded", String(open));
    $("fine").hidden = !open;
    if (!open) return;
    const groups = app.boot.groups.map((group, index) => [index, group]);
    const wanted = Number(app.s.ui.group) || 0;
    const [group, [title, detail]] = groups.find(([index]) => index === wanted) || groups[0];
    const models = group === app.boot.workflow_group;
    const run = app.runs.models;
    $("models-actions").hidden = !models;
    $("models-all").disabled = Boolean(run?.active) || !app.missing || !app.boot.gpu;
    $("models-status").hidden = !models || !run || (!run.active && !run.status);
    $("models-status").className = run?.error ? "status error" : "status";
    $("models-status").textContent = run?.error || run?.status || "";
    // An open list or a half-made click in Fine-tune survives redraws that change nothing here.
    const drawn = JSON.stringify([group, app.s.changed, app.s.locked, app.missing, Boolean(run?.active),
      models ? app.workflows : app.s.fields.filter((field) => field.group === group)]);
    if ($("fine").dataset.drawn === drawn) return;
    $("fine").dataset.drawn = drawn;
    $("fine-nav").replaceChildren(...groups.map(([index, [label]]) => {
      const badge = groupBadge(index);
      return h("button", {type: "button", "aria-pressed": String(index === group), onclick: () => { saveUi({group: index}); render(); }},
        label, badge ? h("span", {class: `chip ${badge[1]}`, "aria-hidden": "true", text: badge[0]}) : null);
    }));
    $("fine-title").textContent = title;
    $("fine-detail").replaceChildren(detail, models ? h("span", {}, " ", h("a", {href: "https://huggingface.co/Lightricks/LTX-2.5", target: "_blank", rel: "noopener", text: "LTX 2.5 model license"})) : "");
    if (models) {
      $("fine-rows").replaceChildren(...app.workflows.map(workflowRow),
        h("p", {class: "small muted", text: "After downloading, reload ComfyUI so its model lists rescan the shared folders."}));
      return;
    }
    const rows = app.s.fields.filter((field) => field.group === group).map(fineRow);
    if (group === 1 && app.s.locked) rows.unshift(h("p", {class: "notice", text: "Reference images sets 768×448 at 24 fps."}));
    $("fine-rows").replaceChildren(...rows);
  };
  const prepareModels = guard(async (workflow) => {
    take(await post("/models/prepare", workflow ? {workflow} : {}));
  });

  // Keyframes and reference pictures ----------------------------------------
  const upload = guard(async (slot, files) => {
    const pictures = [...files].filter((file) => file.type.startsWith("image/"));
    if (!pictures.length) return toast("Add image files here.", true);
    const form = new FormData();
    form.append("slot", slot);
    (slot === "references" ? pictures : pictures.slice(0, 1)).forEach((file) => form.append("files", file));
    return take(await post("/media/upload", form, {form: true}));
  });
  const pickFile = (slot) => {
    const input = $("file-input");
    input.value = "";
    input.multiple = slot === "references";
    input.dataset.slot = slot;
    input.click();
  };
  const openLibrary = guard(async (slot, label) => {
    const dialog = $("library");
    const grid = h("div", {class: "library-grid"});
    const load = guard(async (query) => {
      const {items} = await request("GET", `/library?query=${encodeURIComponent(query)}&limit=60`);
      grid.replaceChildren(...items.map((item) => h("button", {type: "button", title: item.caption, onclick: guard(async () => {
        take(await post("/media/library", {slot, source: item.path}));
        toast(`Added to ${label}.`);
        if (slot !== "references") dialog.close();
      })}, h("img", {src: item.thumb, alt: item.caption, loading: "lazy"}))));
      if (!items.length) grid.replaceChildren(h("p", {class: "small muted", text: "No pictures match."}));
    });
    dialog.replaceChildren(
      h("div", {class: "row between"}, h("h2", {text: `Media library · ${label}`}), h("button", {type: "button", class: "btn quiet", text: "Done", onclick: () => dialog.close()})),
      h("input", {class: "field", type: "search", placeholder: "Filename, tags or settings", "aria-label": "Search pictures", oninput: (event) => load(event.target.value)}),
      grid);
    dialog.showModal();
    await load("");
  });
  const remove = guard(async (slot, index) => take(await post("/media/remove", {slot, index})));
  const droppable = (node, slot) => {
    node.addEventListener("dragover", (event) => { event.preventDefault(); node.classList.add("over"); });
    node.addEventListener("dragleave", () => node.classList.remove("over"));
    node.addEventListener("drop", (event) => {
      event.preventDefault();
      node.classList.remove("over");
      if (event.dataTransfer?.files?.length) upload(slot, event.dataTransfer.files);
    });
    return node;
  };
  const addButtons = (slot, label, text) => [
    h("button", {type: "button", class: "add", "aria-label": `Upload ${label}`, onclick: () => pickFile(slot)}, h("span", {"aria-hidden": "true", text: "+"}), text),
    h("button", {type: "button", class: "link", text: "From Media", onclick: () => openLibrary(slot, label)})];
  // A keyframe's slider: its name and value on one line, the slider under them, so narrow cards keep it whole.
  const knob = (name, field, label, suffix = "") => {
    const [input, out] = slider(field, label, {suffix});
    return h("label", {class: "knob"}, h("span", {class: "knob-head"}, h("span", {text: name}), out), input);
  };
  const KEYS = [["first_image", "Start", "image_strength"], ["middle_image", "Middle", "middle_strength"], ["end_image", "End", "end_strength"]];
  const keyframe = ([slot, name, strength]) => {
    const picture = app.s.keyframes[slot];
    const compose = app.s.compose;
    const at = slot === "first_image" ? "0.0 s" : slot === "end_image" ? `${fixed(compose.duration, compose.duration.value)} s` : null;
    const label = `${name} keyframe`;
    const [add, library] = picture ? [] : addButtons(slot, label, slot === "first_image" ? "Start frame (required)" : `${name} frame`);
    // The middle keyframe stays inside the video: its slider ends a tenth of a second before the last frame.
    const position = {...compose.middle_time, maximum: Math.min(compose.middle_time.maximum, Math.max(0.1, compose.duration.value - 0.1))};
    return droppable(h("div", {class: "slot"},
      picture ? h("div", {class: "pic-box"}, h("img", {src: picture.url, alt: label})) : add,
      h("span", {class: "name", text: at ? `${name} · ${at}` : name}),
      picture ? h("span", {class: "detail", text: picture.detail}) : h("span", {class: "detail"}, library),
      slot === "middle_image" ? knob("Position", position, "Middle keyframe position", " s") : null,
      picture ? knob("Strength", compose[strength], `${name} strength`) : null,
      picture ? h("div", {class: "acts"},
        h("button", {type: "button", class: "btn compact", text: "Replace", "aria-label": `Replace ${label}`, onclick: () => pickFile(slot)}),
        h("button", {type: "button", class: "btn compact quiet", text: "Remove", "aria-label": `Remove ${label}`, onclick: () => remove(slot)})) : null), slot);
  };
  const renderKeyframes = () => {
    const drawn = JSON.stringify([app.s.keyframes, app.s.compose.duration.value, app.s.compose.middle_time.value,
      KEYS.map(([, , strength]) => app.s.compose[strength].value)]);
    if ($("key-list").dataset.drawn !== drawn) {
      $("key-list").replaceChildren(...KEYS.map(keyframe));
      $("key-list").dataset.drawn = drawn;
    }
    const {duration, middle_time: middle} = app.s.compose;
    $("middle-mark").style.left = `${Math.min(100, (middle.value / duration.value) * 100)}%`;
    $("middle-mark").classList.toggle("on", Boolean(app.s.keyframes.middle_image));
    $("end-mark").classList.toggle("on", Boolean(app.s.keyframes.end_image));
  };
  const renderReferences = () => {
    const items = app.s.references;
    $("reference-count").textContent = `${items.length} of ${app.s.reference_limit}`;
    const drawn = JSON.stringify(items.map((item) => item.path));
    if ($("reference-list").dataset.drawn === drawn) return;
    $("reference-list").dataset.drawn = drawn;
    const cells = items.map((item, index) => h("div", {class: "slot"},
      h("div", {class: "pic-box"}, h("img", {src: item.url, alt: `Reference ${index + 1}`})),
      h("span", {class: "name", text: `Panel ${index + 1}`}),
      h("button", {type: "button", class: "btn compact quiet", text: "Remove", "aria-label": `Remove reference ${index + 1}`, onclick: () => remove("references", index)})));
    if (items.length < app.s.reference_limit) {
      const [add, library] = addButtons("references", "Reference images", "Add pictures");
      cells.push(droppable(h("div", {class: "slot"}, add, library), "references"));
    }
    $("reference-list").replaceChildren(...cells);
  };

  // Output ------------------------------------------------------------------
  const renderOutput = () => {
    const field = app.s.compose.duration;
    const input = $("duration");
    if (document.activeElement !== input) {
      Object.assign(input, {min: field.minimum, max: field.maximum, step: field.step});
      input.value = field.value;
      $("duration-value").textContent = `${fixed(field, field.value)} s`;
    }
    input.disabled = app.s.locked;
    const ratios = Object.keys(app.boot.sizes);
    const locked = app.s.locked;
    seg($("aspect"), [...ratios, "Custom"], app.s.ratio, (ratio) => {
      if (ratio === "Custom") return openGroup(1);
      return change("size", {}, `${ratio} ${app.s.tier === "Exact" ? "Medium" : app.s.tier}`);
    }, {disabled: locked});
    seg($("tier"), [...app.boot.sizes["16:9"].map(([tier]) => tier), ["Exact size", "Exact"]], app.s.tier, (tier) => {
      if (tier === "Exact") return openGroup(1);
      return change("size", {}, `${app.s.ratio === "Custom" ? "16:9" : app.s.ratio} ${tier}`);
    }, {disabled: locked});
    $("resolution").replaceChildren(h("span", {class: "mono", text: `${app.s.width}×${app.s.height}`}),
      h("span", {class: "muted", text: ` · ${app.s.frames} frames @ ${app.s.options.fps} fps`}));
  };

  // Prompt ------------------------------------------------------------------
  let promptTimer;
  let pendingSave = Promise.resolve();
  const saveDraft = () => {
    clearTimeout(promptTimer);
    const body = {prompt: $("prompt").value};
    // Blur, autosave and Generate can overlap. Write their snapshots in the order taken.
    const save = pendingSave.then(async () => {
      if (app.s && body.prompt === app.saved) return;
      const answer = await post("/prompt", body);
      app.saved = body.prompt;
      if ($("prompt").value === body.prompt) take(answer);
    });
    pendingSave = save.catch(() => {});
    return save;
  };
  const autosave = guard(() => saveDraft());
  const promptEdited = () => {
    app.s.prompt = $("prompt").value;
    clearTimeout(promptTimer);
    promptTimer = setTimeout(autosave, 500);
    renderReadiness();
  };
  const renderPrompt = () => {
    if (document.activeElement !== $("prompt") && $("prompt").value !== app.s.prompt) $("prompt").value = app.s.prompt;
    app.saved ??= app.s.prompt;
    const negative = app.s.compose.negative_prompt;
    if (document.activeElement !== $("negative") && $("negative").value !== negative.value) $("negative").value = negative.value;
    $("negative").placeholder = negative.placeholder || "Artifacts or qualities to avoid";
    $("negative-chip").textContent = negative.value.trim() ? "set" : "optional";
    $("negative-chip").className = negative.value.trim() ? "chip good" : "chip";
    const run = app.runs.enhance;
    $("writer-line").textContent = `${app.s.writer}${app.s.mode === "Image to video" ? " · reads the keyframes" : ""}`;
    $("enhance").disabled = Boolean(run?.active);
    $("enhance").textContent = run?.active ? "Writing…" : "Generate / enhance prompt";
    $("enhance-status").textContent = app.enhanceNote || (app.enhanceStarted ? run?.error || run?.status : "") || (app.suggestion ? "Review the suggestion below."
      : "Writes a prompt from your idea and keyframes, or improves the one above. You review it before it replaces your prompt.");
    $("suggestion-box").hidden = !app.suggestion;
    $("suggestion").textContent = app.suggestion?.text || "";
    $("undo").hidden = app.previous === null;
  };

  // Runs --------------------------------------------------------------------
  const startRun = guard(async (path) => {
    await saveDraft();
    take(await post(path));
  });
  const poll = async () => {
    if (app.polling) return;
    app.polling = true;
    try {
      for (;;) {
        await new Promise((resolve) => setTimeout(resolve, 1000));
        const before = app.runs;
        const answer = await request("GET", "/run");
        const generated = before.generate?.active && !answer.runs.generate?.active;
        const downloaded = before.models?.active && !answer.runs.models?.active;
        take(generated || downloaded ? await request("GET", `/run?state=${generated}&models=${downloaded}`) : answer);
        if (generated) app.take = 0;
        if (!Object.values(app.runs).some((run) => run?.active)) break;
      }
    } catch (error) {
      toast(error.message, true);
    } finally {
      app.polling = false;
    }
  };

  const renderReadiness = () => {
    const hasPrompt = app.s.prompt.trim().length > 0;
    const steps = [["Prompt", hasPrompt ? "done" : "now"]];
    if (app.s.mode === "Image to video") steps.push(["Start frame", app.s.keyframes.first_image ? "done" : "now"]);
    if (app.s.mode === "Reference images") steps.push(["References", app.s.references.length ? "done" : "now"]);
    $("steps").replaceChildren(...steps.map(([label, state]) => h("span", {class: `step ${state}`}, h("span", {class: "dot", "aria-hidden": "true"}), label)));
    const run = app.runs.generate;
    const ready = app.s.readiness.ready && hasPrompt;
    $("readiness").textContent = run?.active ? "Working. You can keep editing the next run."
      : !hasPrompt ? "Write a video prompt to start." : app.s.readiness.message;
    $("run").disabled = !app.boot.gpu || !ready || Boolean(run?.active) || Boolean(app.runs.models?.active);
    $("stop").hidden = !run?.active;
    const status = $("run-status");
    status.hidden = !run || (!run.active && !run.error && !run.status);
    status.className = run?.error ? "status error" : "status";
    status.textContent = run?.error || run?.status || "";
  };

  // The result's prompt-fit score and tags from Gemini Decision; while one is pending the page asks again.
  let reviewTimer = null;
  const renderReview = (current) => {
    const box = $("result-review");
    const review = current?.review;
    box.hidden = !review;
    clearTimeout(reviewTimer);
    if (review?.status === "pending") reviewTimer = setTimeout(guard(async () => take(await request("GET", "/run?state=true"))), 4000);
    if (!review) return;
    if (review.status === "pending") {
      box.replaceChildren(h("span", {class: "muted", text: "Scoring the video against the prompt…"}));
    } else if (review.status === "done") {
      const radar = decisionScoreRadar?.(review, {chipClass: "chip", includeTags: true, includeHeadline: true});
      box.replaceChildren(...(radar ? [radar] : decisionScoreChips(review, {chipClass: "chip", includeTags: true})));
    } else {
      box.replaceChildren(h("span", {class: "muted", title: review.error || "", text: "No prompt-fit score: the review failed."}));
    }
  };

  const renderViewer = () => {
    const run = app.runs.generate;
    const videos = app.s.results.videos;
    app.take = Math.min(app.take, Math.max(0, videos.length - 1));
    // Newest first: Take N is the latest.
    seg($("takes"), videos.length ? videos.map((_, index) => [`Take ${videos.length - index}`, index]) : [["No takes", 0]], app.take,
      (index) => { app.take = index; render(); }, {disabled: !videos.length});
    const current = videos[app.take];
    const video = $("result-video");
    if (current && video.dataset.path !== current.path) {
      video.src = current.url;
      video.dataset.path = current.path;
    }
    video.hidden = !current;
    $("stage-empty").hidden = Boolean(current);
    $("empty-title").textContent = run?.active ? "Making your video" : "Nothing generated yet";
    $("empty-line").textContent = run?.active ? run.status : "Write a prompt on the right, then press Generate with LTX 2.5.";
    const {width, height, options} = app.s;
    $("viewer-line").textContent = current ? `${videos.length} ${videos.length === 1 ? "take" : "takes"}` : `${width}×${height} · ${options.fps} fps · with audio`;
    $("result-name").textContent = current?.name || "";
    renderReview(current);
    $("download").hidden = !current;
    if (current) { $("download").href = `${current.url}&download=true`; $("download").download = current.name; }
    // The recorded settings belong to the latest take.
    const used = app.take === 0 && app.s.results.settings ? app.s.results.settings : "Settings used will appear with the latest take.";
    if ($("settings-used").dataset.html !== used) { $("settings-used").innerHTML = used; $("settings-used").dataset.html = used; }
  };

  const renderSummary = () => {
    const {options, mode} = app.s;
    const seconds = fixed(app.s.compose.duration, options.duration);
    $("summary").textContent = `LTX 2.5 · ${app.s.width}×${app.s.height} · ${seconds} s · ${app.s.frames} frames`;
    const sampler = app.s.fields.find((field) => field.name === "sampler_name");
    const chips = [mode, options.model_choice, `CFG ${Number(options.cfg).toFixed(2)}`,
      sampler ? sampler.value : null, options.seed === -1 ? "Random seed" : `Seed ${options.seed}`, `${options.fps} fps`];
    if (options.negative_prompt.trim()) chips.push("Negative prompt");
    $("summary-chips").replaceChildren(...chips.filter(Boolean).map((text, index) => h("span", {class: index ? "chip" : "chip good", text})));
  };

  const VIDEO_ENGINES = [["MiniMax H3", "h3"], ["LTX 2.5", "ltx"]];
  const switchEngine = guard(async (value) => {
    if (value === "ltx") return;
    await saveDraft();
    const item = (app.boot?.engines || []).find((candidate) => candidate.id === value);
    const label = item?.label || (value === "h3" ? "MiniMax H3" : value);
    const route = value === "h3" ? "/h3/" : `/${value}/`;
    openStudio("Video", label, () => window.location.assign(route));
  });
  const render = () => {
    if (!app.s) return;
    seg($("mode"), app.boot.modes, app.s.mode, (value) => change("mode", {mode: value}));
    const engines = (app.boot.engines || []).length
      ? app.boot.engines.map((item) => [item.label, item.id])
      : VIDEO_ENGINES;
    seg($("models"), engines, "ltx", switchEngine);
    $("keyframes").hidden = app.s.mode !== "Image to video";
    $("references").hidden = app.s.mode !== "Reference images";
    // Typing in a Fine-tune box must not lose the box to a redraw from another answer.
    const typing = $("fine-rows").contains(document.activeElement) && ["INPUT", "TEXTAREA"].includes(document.activeElement.tagName);
    // Each part draws on its own, so one that fails cannot freeze the others.
    let failed = null;
    const parts = [app.s.mode === "Image to video" && renderKeyframes, app.s.mode === "Reference images" && renderReferences,
                   renderPrompt, renderOutput, renderSummary, renderReadiness, renderViewer, !typing && renderFine];
    for (const part of parts) {
      try { if (part) part(); } catch (error) { console.error(error); failed = error; }
    }
    if (failed) toast(`Part of the page could not be drawn: ${failed.message}`, true);
    queueFit();
  };

  // Frame and theme, as on the other pages ----------------------------------
  const fitFrame = () => {
    const frame = window.frameElement;
    if (!frame) return;
    const height = Math.max(860, document.body.scrollHeight + 24);
    if (Math.abs(frame.offsetHeight - height) > 4) frame.style.height = `${height}px`;
  };
  let fitQueued = false;
  const queueFit = () => {
    if (fitQueued) return;
    fitQueued = true;
    requestAnimationFrame(() => { fitQueued = false; fitFrame(); });
  };
  const applyTheme = () => {
    let dark = matchMedia("(prefers-color-scheme: dark)").matches;
    try {
      if (window.parent !== window) dark = window.parent.document.body.classList.contains("dark");
    } catch { /* Framed by another origin. */ }
    document.documentElement.dataset.theme = dark ? "dark" : "light";
  };
  const refresh = guard(async () => take(await request("GET", "/run?state=true&models=true")));

  const bind = () => {
    $("file-input").addEventListener("change", (event) => {
      if (event.target.files?.length) upload(event.target.dataset.slot, event.target.files);
    });
    $("prompt").addEventListener("input", promptEdited);
    $("prompt").addEventListener("blur", autosave);
    $("negative").addEventListener("change", (event) => change("edit", {negative_prompt: event.target.value}));
    $("duration").addEventListener("input", (event) => { $("duration-value").textContent = `${fixed(app.s.compose.duration, event.target.value)} s`; });
    $("duration").addEventListener("change", (event) => change("edit", {duration: Number(event.target.value)}));
    $("empty").addEventListener("click", guard(async () => {
      clearTimeout(promptTimer);
      await pendingSave;
      const answer = await post("/empty");
      app.suggestion = null;
      app.previous = null;
      app.enhancePending = app.enhanceStarted = false;
      app.enhanceNote = "";
      $("prompt").value = "";
      app.saved = "";
      take(answer);
    }));
    $("run").addEventListener("click", () => startRun("/generate"));
    $("stop").addEventListener("click", guard(async () => {
      const answer = await post("/interrupt");
      if (answer.message) toast(answer.message);
      take(answer);
    }));
    $("refresh").addEventListener("click", guard(async () => { await refresh(); toast("Status refreshed."); }));
    $("restore").addEventListener("click", () => change("restore"));
    $("fine-toggle").addEventListener("click", () => { saveUi({fine: !app.s.ui.fine}); render(); });
    $("fine-close").addEventListener("click", () => { saveUi({fine: false}); render(); });
    $("open-writer").addEventListener("click", () => openGroup(3));
    $("open-model").addEventListener("click", () => openGroup(0));
    $("open-size").addEventListener("click", () => openGroup(1));
    $("models-refresh").addEventListener("click", guard(async () => { take(await request("GET", "/run?models=true")); toast("Model files checked."); }));
    $("models-all").addEventListener("click", () => prepareModels(null));
    $("enhance").addEventListener("click", guard(async () => {
      app.suggestion = null;
      app.enhanceNote = "";
      app.enhanceDraft = JSON.stringify([app.s.mode, app.s.keyframes]);
      await saveDraft();
      const answer = await post("/enhance");
      app.enhancePending = app.enhanceStarted = true;
      take(answer);
    }));
    $("accept").addEventListener("click", guard(async () => {
      const proposal = app.suggestion;
      if (!proposal) return undefined;
      app.suggestion = null;
      if ($("prompt").value !== proposal.original || JSON.stringify([app.s.mode, app.s.keyframes]) !== app.enhanceDraft) {
        app.enhanceNote = "The prompt, mode or keyframes changed. Generate the prompt again for the current draft.";
        return render();
      }
      app.previous = proposal.original;
      app.enhanceNote = "Suggested prompt accepted.";
      $("prompt").value = proposal.text;
      app.s.prompt = proposal.text;
      return saveDraft();
    }));
    $("keep").addEventListener("click", () => { app.suggestion = null; app.enhanceNote = "Original prompt kept."; render(); });
    $("undo").addEventListener("click", () => {
      $("prompt").value = app.previous;
      app.s.prompt = app.previous;
      app.previous = null;
      app.enhanceNote = "";
      autosave();
    });
    document.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" || !(event.ctrlKey || event.metaKey) || event.repeat) return;
      event.preventDefault();
      if (!$("run").disabled) $("run").click();
    });
  };

  const start = async () => {
    applyTheme();
    if (window.frameElement) document.documentElement.dataset.framed = "true";
    try {
      new MutationObserver(applyTheme).observe(window.parent.document.body, {attributes: true, attributeFilter: ["class"]});
    } catch { /* Framed by another origin. */ }
    new MutationObserver(queueFit).observe(document.body, {subtree: true, childList: true, attributes: true, attributeFilter: ["hidden"]});
    document.addEventListener("toggle", queueFit, true);
    window.addEventListener("resize", queueFit);
    // A frame in a hidden tab keeps its last size, so the frame element is watched from the page around it.
    const whenShown = (callback) => {
      let frame = null;
      try { frame = window.frameElement; } catch { /* Framed by another origin. */ }
      const isShown = () => (frame ? frame.getClientRects().length > 0 : innerWidth > 0);
      let shown = isShown();
      new (frame ? window.parent.ResizeObserver : ResizeObserver)(() => {
        const visible = isShown();
        if (visible && !shown) callback();
        shown = visible;
      }).observe(frame || document.documentElement);
    };
    // Media may have handed over a picture while this tab was out of sight.
    whenShown(() => { if (app.s) refresh(); });
    try {
      app.boot = await request("GET", "/boot");
      bind();
      $("workshop-note").hidden = app.boot.gpu;
      $("loading").hidden = true;
      $("editor").hidden = false;
      take(app.boot);
    } catch (error) {
      $("loading").textContent = `LTX 2.5 couldn't open: ${error.message}`;
    }
  };
  start();
})();
