"use strict";
// The Image page. The server owns the draft, every model's options and their policy; this draws what it answers.
(() => {
  const API = "/image/api";
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
    node.append(...children.flat().filter((child) => child !== null && child !== undefined && child !== false));
    return node;
  };

  const app = {boot: null, s: null, runs: {}, progress: [], take: 0, suggestion: null, previous: null,
               enhanceNote: "", polling: false};

  const request = async (method, path, body, {form = false} = {}) => {
    const options = {method, credentials: "same-origin"};
    if (form) options.body = body;
    else if (body !== undefined) {
      options.headers = {"Content-Type": "application/json"};
      options.body = JSON.stringify(body);
    }
    const response = await fetch(`${API}${path}`, options);
    let data = null;
    try { data = await response.json(); } catch { /* A proxy's error page has no JSON. */ }
    if (!response.ok) throw new Error(data?.error || data?.detail || `The server answered ${response.status}.`);
    return data;
  };
  const post = (path, body, extra) => request("POST", path, body, extra);

  const visibleBottom = () => {
    const frame = window.frameElement;
    if (!frame) return innerHeight;
    try { return Math.min(innerHeight, window.parent.innerHeight - frame.getBoundingClientRect().top); } catch { return innerHeight; }
  };
  let toastTimer;
  const toast = (message, error = false) => {
    const node = $("toast");
    node.textContent = message;
    node.className = error ? "toast error" : "toast";
    node.style.bottom = `${Math.max(20, innerHeight - visibleBottom() + 20)}px`;
    node.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { node.hidden = true; }, error ? 8000 : 4000);
  };
  // Every action that can fail reports here instead of leaving the page silent.
  const guard = (action) => async (...args) => {
    try { return await action(...args); } catch (error) { toast(error.message, true); return undefined; }
  };

  const take = (answer) => {
    if (!answer) return;
    if (answer.state) app.s = answer.state;
    if (answer.runs) app.runs = answer.runs;
    if (answer.progress) app.progress = answer.progress;
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
  const change = guard(async (action, changes = {}, name) => take(await post("/settings", {action, changes, name})));
  const changeWriter = guard(async (changes) => take(await post("/writer", {changes})));
  const saveUi = (patch) => {
    app.s.ui = {...app.s.ui, ...patch};
    post("/ui", {ui: app.s.ui}).catch(() => { /* Remembering the open group is a convenience. */ });
  };
  const engine = () => app.boot.engines.find((item) => item.id === app.s.engine);

  // Controls ----------------------------------------------------------------
  // The buttons stay in place while their labels do: a redraw that lands between a press and its
  // release (a poll, a prompt saved as the box loses focus) would otherwise swallow the click.
  const seg = (node, options, current, pick, {disabled = false} = {}) => {
    const items = options.map((option) => Array.isArray(option) ? option : [String(option), option]);
    const kept = node.children.length === items.length && items.every(([label], index) => node.children[index].textContent === label);
    if (!kept) node.replaceChildren(...items.map(([label]) => h("button", {type: "button", text: label})));
    items.forEach(([, value], index) => {
      const button = node.children[index];
      button.setAttribute("aria-pressed", String(value === current));
      button.disabled = disabled;
      button.onclick = () => pick(value);
    });
    return node;
  };
  const commitOf = (field) => (value) => field.store === "writer" ? changeWriter({[field.name]: value}) : change("edit", {[field.name]: value});
  const rangeOf = (field) => field.minimum === null || field.minimum === undefined ? "" : `${field.minimum} – ${field.maximum}`;
  const control = (field) => {
    const commit = commitOf(field);
    const {kind, label, value, disabled} = field;
    if (kind === "bool") return [seg(h("div", {class: "seg", role: "group", "aria-label": label}), [["Off", false], ["On", true]], Boolean(value), commit, {disabled})];
    if (kind === "choice") {
      const long = field.choices.length > 4 || field.choices.reduce((sum, [name]) => sum + name.length, 0) > 44;
      if (!long) return [seg(h("div", {class: "seg", role: "group", "aria-label": label}), field.choices, value, commit, {disabled})];
      const index = field.choices.findIndex(([, choice]) => choice === value);
      return [h("select", {class: "field", "aria-label": label, disabled, onchange: (event) => commit(field.choices[Number(event.target.value)][1])},
        index < 0 ? h("option", {value: "", selected: true, text: "Not set"}) : null,
        field.choices.map(([name], position) => h("option", {value: String(position), selected: position === index, text: name})))];
    }
    if (kind === "int" || kind === "num") {
      return [h("input", {class: "field num mono", type: "text", inputMode: "decimal", value: String(value), disabled, "aria-label": label,
        onchange: (event) => {
          const parsed = Number(event.target.value);
          if (event.target.value.trim() === "" || Number.isNaN(parsed)) { toast(`${label} must be a number.`, true); delete $("fine").dataset.drawn; render(); return; }
          commit(parsed);
        }}), rangeOf(field) ? h("span", {class: "small muted", text: rangeOf(field)}) : null];
    }
    if (kind === "area") return [h("textarea", {class: "field area", rows: 3, value: value ?? "", placeholder: field.placeholder, disabled, "aria-label": label, onchange: (event) => commit(event.target.value)})];
    return [h("input", {class: "field text", type: kind === "password" ? "password" : "text", value: value ?? "", placeholder: field.placeholder, autocomplete: "off",
                        disabled, "aria-label": label, onchange: (event) => commit(event.target.value)})];
  };
  const presetText = (field) => {
    const found = field.choices?.find(([, choice]) => choice === field.preset);
    return found ? found[0] : field.preset === true ? "On" : field.preset === false ? "Off" : String(field.preset ?? "blank");
  };
  const fineRow = (field) => h("div", {class: "fine-row"},
    h("div", {class: "about"},
      h("span", {class: "label"}, field.label,
        field.changed ? h("span", {class: "chip warn", text: `preset: ${presetText(field)}`}) : null,
        field.note ? h("span", {class: "chip", text: field.note}) : null),
      field.hint ? h("span", {class: "hint", text: field.hint}) : null),
    h("div", {class: "control"}, control(field)));

  // Fine-tune ---------------------------------------------------------------
  const openStudio = (task, label) => {
    try {
      const page = window.parent.document;
      const pick = (selector, text) => [...page.querySelectorAll(selector)].find((item) => item.textContent.trim() === text)?.querySelector("input")?.click();
      if (window.parent === window || !page.querySelector(".h3-task-picker")) throw new Error();
      pick(".h3-task-picker label", task);
      const choose = (tries) => {
        const found = [...page.querySelectorAll(".h3-engine-bar label")].find((item) => item.textContent.trim() === label);
        if (found) found.querySelector("input")?.click(); else if (tries > 0) setTimeout(() => choose(tries - 1), 100);
      };
      setTimeout(() => choose(50), 50);
    } catch { toast(`Open ${label} from the ${task} task.`); }
  };
  const groupsShown = () => app.boot.groups.map((group, index) => [index, group])
    .filter(([index]) => app.s.fields.some((field) => field.group === index));
  const openGroup = (index) => {
    saveUi({fine: true, group: index});
    render();
    $("fine").scrollIntoView({block: "nearest", behavior: "smooth"});
  };
  const renderFine = () => {
    const open = Boolean(app.s.ui.fine);
    const total = Object.values(app.s.changed).reduce((sum, count) => sum + count, 0);
    $("fine-count").hidden = !total;
    $("fine-count").textContent = `${total} changed`;
    $("fine-toggle").setAttribute("aria-expanded", String(open));
    $("fine").hidden = !open;
    if (!open) return;
    const shown = groupsShown();
    const wanted = Number(app.s.ui.group) || 0;
    const [group, [title, detail]] = shown.find(([index]) => index === wanted) || shown[0];
    $("fine-line").textContent = `${engine().label} settings, grouped by what they change. A preset sets them; they stay editable.`;
    // An open list or a half-made click in Fine-tune survives redraws that change nothing here.
    const drawn = JSON.stringify([app.s.engine, group, app.s.changed, app.s.fields.filter((field) => field.group === group)]);
    if ($("fine").dataset.drawn === drawn) return;
    $("fine").dataset.drawn = drawn;
    $("fine-nav").replaceChildren(...shown.map(([index, [label]]) => h("button", {
      type: "button", "aria-pressed": String(index === group), onclick: () => { saveUi({group: index}); render(); },
    }, label, app.s.changed[index] ? h("span", {class: "chip warn", "aria-hidden": "true", text: String(app.s.changed[index])}) : null)));
    $("fine-title").textContent = title;
    $("fine-detail").textContent = detail;
    const rows = app.s.fields.filter((field) => field.group === group).map(fineRow);
    if (app.s.engine === "h3" && group === 1) {
      rows.push(h("div", {class: "fine-row"},
        h("div", {class: "about"}, h("span", {class: "label", text: "Model and performance"}),
          h("span", {class: "hint", text: "P-H3-Image runs 8 steps, Euler and CFG 1 on the model profile and performance settings of the MiniMax H3 page."})),
        h("div", {class: "control"}, h("button", {type: "button", class: "btn", text: "Open MiniMax H3", onclick: () => openStudio("Video", "MiniMax H3")}))));
    }
    $("fine-rows").replaceChildren(...rows);
  };

  // Images to edit ----------------------------------------------------------
  const upload = guard(async (files) => {
    const pictures = [...files].filter((file) => file.type.startsWith("image/"));
    if (!pictures.length) return toast("Add image files here.", true);
    const form = new FormData();
    pictures.forEach((file) => form.append("files", file));
    return take(await post("/references/upload", form, {form: true}));
  });
  const openLibrary = guard(async () => {
    const dialog = $("library");
    const grid = h("div", {class: "library-grid"});
    const load = guard(async (query) => {
      const {items} = await request("GET", `/library?query=${encodeURIComponent(query)}&limit=60`);
      grid.replaceChildren(...items.map((item) => h("button", {type: "button", title: item.caption, onclick: guard(async () => {
        take(await post("/references/library", {source: item.path}));
        toast("Added to the images to edit.");
      })}, h("img", {src: item.thumb, alt: item.caption, loading: "lazy"}))));
      if (!items.length) grid.replaceChildren(h("p", {class: "small muted", text: "No pictures match."}));
    });
    dialog.replaceChildren(
      h("div", {class: "row between"}, h("h2", {text: "Media library"}), h("button", {type: "button", class: "btn quiet", text: "Done", onclick: () => dialog.close()})),
      h("input", {class: "field", type: "search", placeholder: "Filename, tags or settings", "aria-label": "Search pictures", oninput: (event) => load(event.target.value)}),
      grid);
    dialog.showModal();
    await load("");
  });
  const insertTag = (tag) => {
    // FLUX 3 names its references in Layout boxes; the other models name them in the prompt.
    const layout = app.s.engine === "flux3";
    if (layout) $("layout-box").open = true;
    const area = layout ? $("layout") : $("prompt");
    const start = area.selectionStart ?? area.value.length, end = area.selectionEnd ?? area.value.length;
    const before = area.value.slice(0, start), after = area.value.slice(end);
    area.focus();
    area.setRangeText((before && !/\s$/.test(before) ? " " : "") + tag + (/^\s/.test(after) ? "" : " "), start, end, "end");
    if (layout) saveLayout(); else promptEdited();
  };
  const move = guard(async (from, to) => take(await post("/references/move", {from, to})));
  const renderReferences = () => {
    const items = app.s.references;
    $("reference-count").textContent = items.length ? `${items.length} added` : "";
    $("clear-references").hidden = !items.length;
    const drawn = JSON.stringify([items.map((item) => [item.path, item.tag]), Boolean(app.s.separate?.value), app.s.engine]);
    if ($("reference-list").dataset.drawn !== drawn) $("reference-list").replaceChildren(...items.map((item, index) => h("div", {class: "pic"},
      h("img", {class: "thumb", src: item.url, alt: `Image ${index + 1}`}),
      h("div", {class: "body"},
        h("div", {class: "row tightrow"}, h("strong", {class: "small", text: `Image ${index + 1}`}),
          index === 0 && !app.s.separate?.value ? h("span", {class: "chip good", text: "edit target"}) : null),
        h("span", {class: "small muted detail", text: item.detail}),
        h("div", {class: "row tightrow"},
          item.tag ? h("button", {type: "button", class: "btn compact tag", text: `+ ${item.tag}`, title: app.s.engine === "flux3" ? "Add this name to Layout boxes" : "Add this name to the prompt", onclick: () => insertTag(item.tag)}) : null,
          h("button", {type: "button", class: "btn compact", text: "Up", "aria-label": `Move image ${index + 1} up`, disabled: index === 0, onclick: () => move(index, index - 1)}),
          h("button", {type: "button", class: "btn compact", text: "Down", "aria-label": `Move image ${index + 1} down`, disabled: index === items.length - 1, onclick: () => move(index, index + 1)}),
          h("button", {type: "button", class: "btn compact quiet", text: "Remove", "aria-label": `Remove image ${index + 1}`, onclick: guard(async () => take(await post("/references/remove", {index})))}))))));
    $("reference-list").dataset.drawn = drawn;
    const separate = app.s.separate;
    $("separate").replaceChildren(separate ? h("label", {class: "check"},
      h("input", {type: "checkbox", checked: Boolean(separate.value), onchange: (event) => change("edit", {[separate.name]: event.target.checked})}),
      h("span", {}, h("strong", {text: separate.label}), h("span", {class: "hint", text: separate.hint}))) : "");
    $("guidance").textContent = app.s.guidance;
  };

  // Output ------------------------------------------------------------------
  // What a model without presets samples with, where the others show their preset.
  const SAMPLING = {
    h3: "P-H3-Image runs 8 steps, Euler and CFG 1 on your MiniMax H3 settings.",
    flux3: "Distilled models run 16 steps at CFG 1; Base models run 50 steps at CFG 4.",
    fibo2: "Dev runs 30 steps at CFG 5. Edit Turbo edits in 4 steps and needs Edit.",
  };
  const sizeChoice = () => {
    const tiers = app.boot.sizes[app.s.engine] || [];
    for (const [index, tier] of tiers.entries()) {
      const found = tier.sizes.find((size) => size.width === app.s.width && size.height === app.s.height);
      if (found) return {ratio: found.ratio, tier: index};
    }
    return {ratio: "Custom", tier: -1};
  };
  const pickSize = (ratio, tier) => {
    const tiers = app.boot.sizes[app.s.engine] || [];
    if (ratio === "Custom" || tier < 0) return openGroup(1);
    const size = tiers[tier].sizes.find((item) => item.ratio === ratio);
    if (!size) return toast(`${ratio} has no ${tiers[tier].label} size.`, true);
    return change("resolution", {}, size.name);
  };
  const renderOutput = () => {
    const {preset, batch, edit_size: editSize} = app.s;
    $("preset-box").hidden = !preset;
    $("no-preset").hidden = Boolean(preset);
    if (preset) {
      seg($("preset"), preset.choices, preset.value, (value) => change("edit", {[preset.name]: value}));
      $("preset-hint").textContent = preset.hint;
    } else {
      $("no-preset-line").textContent = SAMPLING[app.s.engine] || "";
    }
    $("batch-row").hidden = !batch;
    if (batch) seg($("batch"), Array.from({length: batch.maximum - batch.minimum + 1}, (_, index) => batch.minimum + index), batch.value, (value) => change("edit", {[batch.name]: value}));
    $("edit-size-row").hidden = !editSize;
    if (editSize) {
      seg($("edit-size"), editSize.choices, editSize.value, (value) => change("edit", {[editSize.name]: value}));
      $("edit-size").title = editSize.hint;
    }
    const tiers = app.boot.sizes[app.s.engine] || [];
    const chosen = sizeChoice();
    const ratios = [...new Set(tiers.flatMap((tier) => tier.sizes.map((size) => size.ratio)))];
    seg($("aspect"), [...ratios, "Custom"], chosen.ratio, (value) => pickSize(value, Math.max(0, chosen.tier)));
    seg($("tier"), [...tiers.map((tier, index) => [tier.label, index]), ["Exact dimensions", -1]], chosen.tier,
      (value) => pickSize(chosen.ratio === "Custom" ? ratios[0] : chosen.ratio, value));
    const follows = editSize && editSize.value !== editSize.choices[editSize.choices.length - 1][1];
    $("resolution").replaceChildren(h("span", {class: "mono", text: `${app.s.width} × ${app.s.height}`}),
      h("span", {class: "muted", text: ` · ${(app.s.width * app.s.height / 1e6).toFixed(1)} MP${follows ? " · this edit follows Edit size; pick a size to override it" : ""}`}));
  };

  // Prompt ------------------------------------------------------------------
  let promptTimer;
  const savePrompt = guard(async (accepted = false) => {
    clearTimeout(promptTimer);
    const sent = $("prompt").value;
    const answer = await post("/prompt", {prompt: sent, accepted: accepted === true});
    // Typing may have gone on while the save was in flight; the newer text stays.
    if ($("prompt").value === sent) take(answer);
  });
  const promptEdited = () => {
    app.s.prompt = $("prompt").value;
    clearTimeout(promptTimer);
    promptTimer = setTimeout(savePrompt, 500);
    renderReadiness();
  };
  const saveLayout = () => { if (app.s.layout && $("layout").value !== app.s.layout.value) change("edit", {[app.s.layout.name]: $("layout").value}); };
  const renderPrompt = () => {
    const editing = app.s.editing;
    $("prompt-label").textContent = editing ? "Edit instruction" : "Image prompt";
    $("prompt").placeholder = editing ? "Describe what to change and what to preserve." : "Describe the image you want to create.";
    if (document.activeElement !== $("prompt") && $("prompt").value !== app.s.prompt) $("prompt").value = app.s.prompt;
    $("generate-note").hidden = editing || !app.s.guidance;
    $("generate-note").textContent = app.s.guidance;
    const run = app.runs.enhance;
    $("writer-line").textContent = app.s.can_enhance ? app.s.writer : "Not available for this model";
    $("enhance").disabled = !app.s.can_enhance || Boolean(run?.active);
    $("enhance").textContent = run?.active ? "Writing…" : "Suggest a prompt";
    $("open-writer").hidden = !app.s.can_enhance;
    $("enhance-status").textContent = !app.s.can_enhance ? "" : app.enhanceNote || (app.enhanceStarted ? run?.error || run?.status : "") || (app.suggestion ? "Review the suggestion below."
      : "Drafts a prompt from your idea and images, or improves the one above. You review the suggestion before it replaces your prompt.");
    $("suggestion-box").hidden = !app.suggestion;
    $("suggestion").textContent = app.suggestion?.text || "";
    $("undo").hidden = app.previous === null;
    const layout = app.s.layout;
    $("layout-box").hidden = !layout;
    if (layout) {
      $("layout-hint").textContent = layout.hint;
      $("layout").placeholder = layout.placeholder;
      if (document.activeElement !== $("layout") && $("layout").value !== layout.value) $("layout").value = layout.value;
    }
  };

  // Runs --------------------------------------------------------------------
  const startRun = guard(async (path) => {
    await savePrompt();
    saveLayout();
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
        const settled = before.generate?.active && !answer.runs.generate?.active;
        take(settled ? await request("GET", "/run?state=true") : answer);
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
    if (app.s.editing) steps.push(["Images", app.s.references.length ? "done" : "now"]);
    $("steps").replaceChildren(...steps.map(([label, state]) => h("span", {class: `step ${state}`}, h("span", {class: "dot", "aria-hidden": "true"}), label)));
    const run = app.runs.generate;
    const ready = app.s.readiness.ready && hasPrompt;
    $("readiness").textContent = run?.active ? "Working. You can keep editing the next run."
      : !hasPrompt ? "Write an image prompt or edit instruction." : app.s.readiness.message;
    $("run-label").textContent = app.s.editing ? "Edit image" : "Generate image";
    $("run").disabled = !app.boot.gpu || !ready || Boolean(run?.active);
    $("stop").hidden = !run?.active;
    const status = $("run-status");
    status.hidden = !run || (!run.active && !run.error && !run.status);
    status.className = run?.error ? "status error" : "status";
    status.textContent = run?.error || run?.status || "";
  };

  const renderViewer = () => {
    const run = app.runs.generate;
    const images = run?.active ? app.progress : app.s.results.images;
    app.take = Math.min(app.take, Math.max(0, images.length - 1));
    seg($("takes"), Array.from({length: Math.max(1, images.length)}, (_, index) => [`Image ${index + 1}`, index]), app.take, (index) => { app.take = index; render(); });
    const current = images[app.take];
    const picture = $("result-image");
    if (current && picture.dataset.path !== current.path) { picture.src = current.url; picture.dataset.path = current.path; $("result-link").href = current.url; }
    $("result-link").hidden = !current;
    $("stage-empty").hidden = Boolean(current);
    $("empty-title").textContent = run?.active ? "Making your image" : "Nothing generated yet";
    $("empty-line").textContent = run?.active ? run.status : "Write a prompt on the right, then press Generate image.";
    $("viewer-line").textContent = current ? `${images.length} ${images.length === 1 ? "image" : "images"}` : "No result yet";
    $("result-name").textContent = current?.name || "";
    $("download").hidden = !current;
    if (current) { $("download").href = `${current.url}&download=true`; $("download").download = current.name; }
    $("use-result").hidden = !current || run?.active || app.s.generate_only;
    const used = app.s.results.settings || "Settings used will appear with the generated result.";
    if ($("settings-used").dataset.html !== used) { $("settings-used").innerHTML = used; $("settings-used").dataset.html = used; }
  };

  const renderSummary = () => {
    const {preset, batch, edit_size: editSize, editing} = app.s;
    const follows = editSize && editSize.value !== editSize.choices[editSize.choices.length - 1][1];
    const count = batch ? batch.value : 1;
    $("summary").textContent = [engine().label, preset?.value, follows ? editSize.value : `${app.s.width} × ${app.s.height}`,
      `${count} ${count === 1 ? "image" : "images"}`].filter(Boolean).join(" · ");
    const chips = [editing ? `Edit · ${app.s.references.length} ${app.s.references.length === 1 ? "image" : "images"}` : "Generate",
      app.s.seed === -1 ? "Random seed" : `Seed ${app.s.seed}`];
    if (app.s.separate?.value) chips.push("Each image separately");
    if (app.s.layout?.value?.trim()) chips.push("Layout boxes");
    $("summary-chips").replaceChildren(...chips.map((text) => h("span", {class: "chip", text})));
  };

  const render = () => {
    if (!app.s) return;
    seg($("mode"), app.boot.modes, app.s.mode, (value) => change("mode", {mode: value}), {disabled: app.s.generate_only});
    $("mode-help").textContent = app.s.generate_only ? "MOSS-Image generates from text only."
      : app.s.editing ? "Change the images you add. The first one is the edit target." : "Create an image from the prompt alone. Adding an image switches to Edit.";
    seg($("models"), app.boot.engines.map((item) => [item.label, item.id]), app.s.engine, (value) => change("engine", {engine: value}));
    $("about").textContent = engine().description;
    $("references").hidden = !app.s.editing;
    // Typing in a Fine-tune box must not lose the box to a redraw from another answer.
    const typing = $("fine-rows").contains(document.activeElement) && ["INPUT", "TEXTAREA"].includes(document.activeElement.tagName);
    // Each part draws on its own, so one that fails cannot freeze the others.
    let failed = null;
    for (const part of [app.s.editing && renderReferences, renderPrompt, renderOutput, renderSummary, renderReadiness, renderViewer, !typing && renderFine]) {
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
  const refresh = guard(async () => take(await request("GET", "/run?state=true")));

  const bind = () => {
    $("file-input").addEventListener("change", (event) => { if (event.target.files?.length) upload(event.target.files); });
    $("add-upload").addEventListener("click", () => { $("file-input").value = ""; $("file-input").click(); });
    $("add-library").addEventListener("click", openLibrary);
    $("clear-references").addEventListener("click", guard(async () => take(await post("/references/clear"))));
    const drop = $("references");
    drop.addEventListener("dragover", (event) => { event.preventDefault(); $("add-upload").classList.add("over"); });
    drop.addEventListener("dragleave", () => $("add-upload").classList.remove("over"));
    drop.addEventListener("drop", (event) => {
      event.preventDefault();
      $("add-upload").classList.remove("over");
      if (event.dataTransfer?.files?.length) upload(event.dataTransfer.files);
    });
    $("prompt").addEventListener("input", promptEdited);
    $("prompt").addEventListener("blur", () => savePrompt());
    $("layout").addEventListener("change", saveLayout);
    $("run").addEventListener("click", () => { app.progress = []; app.take = 0; startRun("/generate"); });
    $("stop").addEventListener("click", guard(async () => {
      const answer = await post("/interrupt");
      if (answer.message) toast(answer.message);
      take(answer);
    }));
    $("refresh").addEventListener("click", guard(async () => { take(await request("GET", "/run?state=true")); toast("Status refreshed."); }));
    $("restore").addEventListener("click", () => change("restore"));
    $("fine-toggle").addEventListener("click", () => { saveUi({fine: !app.s.ui.fine}); render(); });
    $("fine-close").addEventListener("click", () => { saveUi({fine: false}); render(); });
    $("open-writer").addEventListener("click", () => openGroup(4));
    $("open-model").addEventListener("click", () => openGroup(0));
    $("open-model-plain").addEventListener("click", () => openGroup(app.s.engine === "h3" ? 1 : 0));
    $("open-size").addEventListener("click", () => openGroup(1));
    $("use-result").addEventListener("click", guard(async () => {
      const current = app.s.results.images[app.take];
      if (current) take(await post("/references/use-result", {path: current.path}));
    }));
    $("enhance").addEventListener("click", guard(async () => {
      app.suggestion = null;
      app.enhanceNote = "";
      app.enhanceDraft = JSON.stringify([app.s.engine, app.s.mode, app.s.references.map((item) => item.path)]);
      await savePrompt();
      const answer = await post("/enhance");
      app.enhancePending = app.enhanceStarted = true;
      take(answer);
    }));
    $("accept").addEventListener("click", guard(async () => {
      const proposal = app.suggestion;
      if (!proposal) return undefined;
      app.suggestion = null;
      const draft = JSON.stringify([app.s.engine, app.s.mode, app.s.references.map((item) => item.path)]);
      if ($("prompt").value !== proposal.original || draft !== app.enhanceDraft) {
        app.enhanceNote = "The prompt, model or images changed. Suggest again for the current draft.";
        return render();
      }
      app.previous = proposal.original;
      app.enhanceNote = "Suggested prompt accepted.";
      $("prompt").value = proposal.text;
      app.s.prompt = proposal.text;
      return savePrompt(true);
    }));
    $("keep").addEventListener("click", () => { app.suggestion = null; app.enhanceNote = "Original prompt kept."; render(); });
    $("undo").addEventListener("click", () => {
      $("prompt").value = app.previous;
      app.s.prompt = app.previous;
      app.previous = null;
      app.enhanceNote = "";
      savePrompt();
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
    // Another page may have handed over a picture while this tab was out of sight.
    let shown = innerWidth > 0;
    new ResizeObserver(() => {
      const visible = innerWidth > 0;
      if (visible && !shown && app.s) refresh();
      shown = visible;
    }).observe(document.documentElement);
    try {
      app.boot = await request("GET", "/boot");
      bind();
      $("workshop-note").hidden = app.boot.gpu;
      $("loading").hidden = true;
      $("editor").hidden = false;
      take(app.boot);
    } catch (error) {
      $("loading").textContent = `Image couldn't open: ${error.message}`;
    }
  };
  start();
})();
