"use strict";
// The Song page. The server owns each model's draft, options and takes; this draws what it answers.
(() => {
  const API = "/song/api";
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

  // What each model's brief and draft say, as their Gradio views said it.
  const COPY = {
    music: {
      description: "Song idea / music description",
      hint: "Start with an idea, or paste a finished music description to generate directly.",
      placeholder: "A warm indie-pop song about taking the last train home, with English lyrics, a gentle lead vocal, bright guitars, and a hopeful chorus.",
      draft: "The song check updates as you type. Write keeps your words unless you choose Polish.",
      write: "Write drafts lyrics and a production description for you to review. Write & generate applies it without review and starts a take; Undo still works.",
      change: "Say what you hear. The writer changes only that, and you review the new draft before the next take.",
      run: "Generate with Music 3",
    },
    yue2: {
      description: "Song idea / style tags",
      hint: "Start with an idea, or paste a finished style prompt to generate directly.",
      placeholder: "Indie pop with a warm female vocal, clean guitar, restrained drums, and a hopeful, building chorus.",
      draft: "YuE2 sings these lyrics. Write keeps your words unless you choose Polish.",
      write: "Write drafts lyrics and style tags for you to review. Write & generate applies them without review and starts a take; Undo still works.",
      change: "Say what you hear. The writer changes only that, and you review the new draft before the next take. To keep a melody, reuse its score first.",
      run: "Generate with YuE2",
    },
  };
  const POLICY_HINT = {
    "Keep my lyrics": "Keep sends your words exactly, though a take can still mis-sing a word.",
    "Polish my lyrics": "Polish fixes structure, long lines and rhymes but keeps your meaning; you review every change.",
  };

  const app = {boot: null, s: null, runs: {}, settled: {}, gpu: false, tab: "lyrics", polling: false, writeSeen: null};

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
  const guard = (action) => async (...args) => {
    try { return await action(...args); } catch (error) { toast(error.message, true); return undefined; }
  };

  const take = (answer) => {
    if (!answer) return;
    if (answer.state) app.s = answer.state;
    if (answer.runs) app.runs = answer.runs;
    if (answer.state) {
      for (const [kind, run] of Object.entries(app.runs)) if (run && !run.active) app.settled[kind] = run.id;
    }
    if (answer.gpu !== undefined) app.gpu = answer.gpu;
    render();
    if (Object.values(app.runs).some((run) => run?.active)) poll();
  };
  const change = guard(async (action, changes = {}) => take(await post("/settings", {action, changes})));
  const changeWriter = guard(async (changes) => take(await post("/writer", {changes})));
  const saveUi = (patch) => {
    app.s.ui = {...app.s.ui, ...patch};
    post("/ui", {ui: patch}).catch(() => { /* Remembering the view is a convenience. */ });
  };
  const engine = () => app.boot.engines.find((item) => item.id === app.s.engine);
  const copy = () => COPY[app.s.engine];
  const clock = (seconds) => {
    const total = Math.round(seconds);
    return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
  };

  // Controls ----------------------------------------------------------------
  // The buttons stay in place while their labels do, so a redraw between press and release keeps the click.
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
  const numberInput = (field, commit, props = {}) => h("input", {class: "field num mono", type: "text", inputMode: "decimal", value: String(field.value),
    "aria-label": field.label, ...props,
    onchange: async (event) => {
      const input = event.target;
      const sent = input.value;
      const parsed = Number(sent);
      if (sent.trim() === "" || !Number.isFinite(parsed)) {
        toast(`${field.label} must be a number.`, true);
        input.value = String(field.value);
        return;
      }
      await commit(parsed);
      if (input.value === sent) input.value = String(app.s.options[field.name] ?? field.value);
    }});
  const control = (field) => {
    const commit = commitOf(field);
    const {kind, label, value} = field;
    if (kind === "bool") return [seg(h("div", {class: "seg", role: "group", "aria-label": label}), [["Off", false], ["On", true]], Boolean(value), commit)];
    if (kind === "choice") {
      const long = field.choices.length > 4 || field.choices.reduce((sum, [name]) => sum + name.length, 0) > 44;
      if (!long) return [seg(h("div", {class: "seg", role: "group", "aria-label": label}), field.choices, value, commit)];
      const index = field.choices.findIndex(([, choice]) => choice === value);
      return [h("select", {class: "field select", "aria-label": label, onchange: (event) => commit(field.choices[Number(event.target.value)][1])},
        index < 0 ? h("option", {value: "", selected: true, text: "Not set"}) : null,
        field.choices.map(([name], position) => h("option", {value: String(position), selected: position === index, text: name})))];
    }
    if (kind === "int" || kind === "num") return [numberInput(field, commit), rangeOf(field) ? h("span", {class: "small muted", text: rangeOf(field)}) : null];
    return [h("input", {class: "field text", type: kind === "password" ? "password" : "text", value: value ?? "", placeholder: field.placeholder, autocomplete: "off",
                        "aria-label": label, onchange: (event) => commit(event.target.value)})];
  };
  const presetText = (field) => {
    const found = field.choices?.find(([, choice]) => choice === field.preset);
    return found ? found[0] : field.preset === true ? "On" : field.preset === false ? "Off" : String(field.preset ?? "blank");
  };
  const fineRow = (field) => h("div", {class: "fine-row"},
    h("div", {class: "about"},
      h("span", {class: "label"}, field.label, field.changed ? h("span", {class: "chip warn", text: `default: ${presetText(field)}`}) : null),
      field.hint ? h("span", {class: "hint", text: field.hint}) : null),
    h("div", {class: "control"}, control(field)));

  // Draft text --------------------------------------------------------------
  const TEXT = {description: "description", lyrics: "lyrics", abc: "abc", notes: "notes"};
  let draftTimer;
  let pendingSave = Promise.resolve();
  // The text each box last held from the server, or last sent to it. A box that still holds it was
  // not edited, even when the server has rewritten that text since (a focused box keeps its text).
  const synced = {};
  const fill = (name, value) => {
    $(name).value = value;
    synced[name] = value;
  };
  const saveDraft = () => {
    clearTimeout(draftTimer);
    // Only the boxes edited since they were filled, so a blur cannot write stale text over a draft
    // the server rewrote meanwhile (Write & generate, a reused take).
    const changes = Object.fromEntries(Object.keys(TEXT).filter((name) => $(name).value !== synced[name])
      .map((name) => [name, $(name).value]));
    if (!Object.keys(changes).length) return pendingSave;
    const body = {engine: app.s.engine, changes};
    // Blur, autosave and Generate can overlap. Write their snapshots in the order taken.
    const save = pendingSave.then(async () => {
      const answer = await post("/draft", body);
      if (app.s.engine === body.engine) Object.assign(synced, changes);
      const unchanged = Object.entries(changes).every(([name, value]) => $(name).value === value);
      if (unchanged && app.s.engine === body.engine) take(answer);
    });
    pendingSave = save.catch(() => {});
    return save;
  };
  const autosave = guard(() => saveDraft());
  const edited = () => {
    clearTimeout(draftTimer);
    draftTimer = setTimeout(autosave, 500);
  };
  // A box being typed in, or holding edits not yet saved, keeps its text; the rest show the server's.
  const setText = (name, value) => {
    const box = $(name);
    if (synced[name] !== undefined && (document.activeElement === box || box.value !== synced[name])) return;
    if (box.value !== value) box.value = value;
    synced[name] = value;
  };
  const appendLyrics = (text) => {
    const area = $("lyrics");
    const current = area.value.replace(/\s+$/, "");
    area.value = (current ? `${current}\n\n` : "") + text;
    edited();
  };

  // Visual inspiration --------------------------------------------------------
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
        toast("Added to visual inspiration.");
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
  const renderReferences = () => {
    const shown = app.s.engine === "music";
    $("references").hidden = !shown;
    if (!shown) return;
    const items = app.s.draft.references;
    const drawn = JSON.stringify(items.map((item) => item.path));
    if ($("reference-list").dataset.drawn !== drawn) {
      $("reference-list").replaceChildren(...items.map((item, index) => h("div", {class: "pic"},
        h("img", {class: "thumb", src: item.url, alt: `Reference image ${index + 1}`}),
        h("div", {class: "body"},
          h("strong", {class: "small", text: `Reference image ${index + 1}`}),
          h("span", {class: "small muted detail", text: item.detail}),
          h("div", {class: "row tightrow"},
            h("button", {type: "button", class: "btn compact quiet", text: "Remove", "aria-label": `Remove reference image ${index + 1}`,
                         onclick: guard(async () => take(await post("/references/remove", {index})))}))))));
      $("reference-list").dataset.drawn = drawn;
    }
    $("reference-add").hidden = items.length >= 3;
  };

  // Brief --------------------------------------------------------------------
  const renderBrief = () => {
    const {brief, draft} = app.s;
    $("description-label").textContent = copy().description;
    $("description-hint").textContent = copy().hint;
    $("description").placeholder = copy().placeholder;
    setText("description", draft.description);
    const starters = app.boot.starters[app.s.engine] || [];
    const drawn = app.s.engine;
    if ($("starters").dataset.drawn !== drawn) {
      $("starters").replaceChildren(...starters.map((idea) => h("button", {type: "button", class: "btn compact", text: idea, onclick: () => {
        $("description").value = idea;
        autosave();
      }})));
      $("starters").dataset.drawn = drawn;
    }
    const pick = (field) => (value) => change("edit", {[field.name]: value});
    seg($("vocals"), brief.vocals.choices, brief.vocals.value, pick(brief.vocals));
    seg($("use"), brief.use.choices, brief.use.value, pick(brief.use));
    const language = $("language");
    if (language.dataset.drawn !== JSON.stringify(brief.language.choices)) {
      language.replaceChildren(...brief.language.choices.map(([name, value]) => h("option", {value, text: name})));
      language.dataset.drawn = JSON.stringify(brief.language.choices);
    }
    language.value = brief.language.value;
    language.onchange = () => change("edit", {[brief.language.name]: language.value});
    const duration = brief.duration;
    if (document.activeElement !== $("duration")) $("duration").value = String(duration.value);
    $("duration-range").textContent = `${duration.minimum}–${duration.maximum}`;
    seg($("duration-presets"), (app.boot.durations[app.s.engine] || []).map(([seconds, label]) => [label, seconds]), duration.value, pick(duration));
    $("write-hint").textContent = copy().write;
    const writing = Boolean(app.runs.write?.active);
    $("write").disabled = $("write-generate").disabled = $("revise").disabled = writing;
    $("write").textContent = writing ? "Writing…" : "Write the song";
    $("writer-chip").textContent = app.s.writer.backend;
    if (!$("writer-rows").contains(document.activeElement)) {
      const drawnRows = JSON.stringify(app.s.writer_rows);
      if ($("writer-rows").dataset.drawn !== drawnRows) {
        $("writer-rows").replaceChildren(...app.s.writer_rows.map(fineRow));
        $("writer-rows").dataset.drawn = drawnRows;
      }
    }
  };

  // Draft --------------------------------------------------------------------
  const renderDraft = () => {
    const {draft, brief} = app.s;
    const yue = app.s.engine === "yue2";
    if (!yue) app.tab = "lyrics";
    $("tab-score").hidden = !yue;
    $("tab-lyrics").setAttribute("aria-pressed", String(app.tab === "lyrics"));
    $("tab-score").setAttribute("aria-pressed", String(app.tab === "score"));
    $("lyrics-pane").hidden = app.tab !== "lyrics";
    $("score-pane").hidden = app.tab !== "score";
    $("draft-hint").textContent = copy().draft;
    setText("lyrics", draft.lyrics);
    setText("abc", draft.abc);
    setText("notes", draft.notes);
    if ($("song-check").dataset.html !== app.s.check) {
      $("song-check").innerHTML = app.s.check;
      $("song-check").dataset.html = app.s.check;
    }
    const policy = brief.lyrics_policy;
    seg($("policy"), policy.choices, policy.value, (value) => change("edit", {[policy.name]: value}));
    $("policy-hint").textContent = POLICY_HINT[policy.value] || policy.hint;
    if (yue && brief.mode) {
      seg($("composition"), brief.mode.choices, brief.mode.value, (value) => change("edit", {[brief.mode.name]: value}));
      $("composition-hint").textContent = brief.mode.hint;
      $("abc").disabled = brief.mode.value === "off";
      setText("last-score", app.s.takes.last_score);
    }
    // The writer's answer.
    const run = app.runs.write;
    $("writer-status").textContent = run ? (run.error ? `Write stopped: ${run.error}` : run.status) : "Press Write the song, or type your lyrics below.";
    const suggestion = run?.suggestion;
    $("suggestion-box").hidden = !suggestion;
    if (suggestion) {
      $("suggestion").replaceChildren(
        h("strong", {text: copy().description}), h("pre", {text: suggestion.description}),
        h("strong", {text: "Lyrics and song structure"}), h("pre", {text: suggestion.lyrics || "(none)"}));
    }
    $("undo").hidden = !app.s.undo || Boolean(suggestion);
    // On a phone the draft sits below the takes; bring a finished suggestion into view once.
    if (suggestion && app.writeSeen !== run.id) {
      app.writeSeen = run.id;
      $("draft").scrollIntoView({block: "nearest", behavior: "smooth"});
    }
    const sections = $("sections");
    if (sections.children.length === 1) {
      sections.append(...app.boot.sections.map((tag) => h("button", {type: "button", class: "pill mono", text: `+ ${tag}`, onclick: () => appendLyrics(`[${tag}]\n`)})),
        h("button", {type: "button", class: "pill", text: "Repeat chorus", title: "Repeats the last chorus block", onclick: () => {
          const text = $("lyrics").value.replace(/\s+$/, "");
          appendLyrics(((text.match(/\[chorus[^\]]*\][^[]*/gi) || []).pop() || "[Chorus]\n").trim() + "\n");
        }}));
    }
    $("change-hint").textContent = copy().change;
    const fixes = $("quick-fixes");
    if (!fixes.children.length) {
      fixes.append(...app.boot.quick_fixes.map((label) => h("button", {type: "button", class: "pill", text: label, onclick: () => {
        const notes = $("notes").value.trim();
        $("notes").value = notes ? `${notes}, ${label.toLowerCase()}` : label;
        edited();
      }})));
    }
    $("revise-box").hidden = !app.s.takes.items.length && !draft.notes;
  };

  // Takes --------------------------------------------------------------------
  const setAudio = (node, item) => {
    const url = item?.url || "";
    if ((node.dataset.url || "") !== url) {
      node.dataset.url = url;
      if (url) node.src = url; else node.removeAttribute("src");
    }
  };
  const selectTakes = guard(async (patch) => take(await post("/takes/select", patch)));
  // A take's prompt-fit score and tags from Gemini Decision; while one is pending the page asks again.
  let reviewTimer = null;
  const reviewText = (review) => review?.status === "done"
    ? `Fit ${review.fit} · ${review.label}${review.tags.length ? ` · ${review.tags.join(", ")}` : ""}`
    : review?.status === "pending" ? "Scoring…" : "";
  const renderTakes = () => {
    const {items, a, b, summary, details, settings} = app.s.takes;
    clearTimeout(reviewTimer);
    if (items.some((item) => item.review?.status === "pending") && !app.runs.generate?.active) {
      reviewTimer = setTimeout(guard(async () => take(await request("GET", "/run?state=true"))), 4000);
    }
    const compare = Boolean(app.s.ui.compare) && items.length > 1;
    $("takes-summary").textContent = summary;
    seg($("view-mode"), [["Listen", false], ["Compare A / B", true]], compare, (value) => {
      if (value && (b === null || b === undefined) && items.length > 1) {
        app.s.ui = {...app.s.ui, compare: true};
        selectTakes({compare: true, b: a === items.length - 1 ? items.length - 2 : items.length - 1});
      } else {
        saveUi({compare: value});
        render();
      }
    });
    $("view-mode").hidden = items.length < 2;
    const run = app.runs.generate;
    $("stage-empty").hidden = items.length > 0;
    $("stage-empty-line").textContent = run?.active ? run.status : "Describe your song on the right and press Write the song, or paste lyrics into the draft.";
    $("players").hidden = !items.length;
    const takeA = a === null || a === undefined ? null : items[a];
    const takeB = compare && b !== null && b !== undefined ? items[b] : null;
    setAudio($("audio-a"), takeA);
    setAudio($("audio-b"), takeB);
    $("label-a").textContent = takeA?.label || "";
    $("label-b").textContent = takeB?.label || "";
    $("player-b").hidden = !takeB;
    $("player-a").querySelector(".side").hidden = !compare;
    $("compare-hint").hidden = !compare;
    const drawn = JSON.stringify([items, a, compare ? b : null, compare]);
    if ($("take-list").dataset.drawn !== drawn) {
      $("take-list").replaceChildren(...[...items].reverse().map((item) => {
        const isA = item.index === a, isB = compare && item.index === b;
        const card = h("button", {type: "button", class: "take", "aria-pressed": String(isA || isB),
          title: compare ? "Click to set as A; Shift-click to set as B" : "Listen to this take",
          onclick: (event) => selectTakes(event.shiftKey ? {b: item.index} : {a: item.index})},
          h("span", {class: `badge mono${isB ? " b" : ""}`, text: isA && compare ? "A" : isB ? "B" : String(item.number)}),
          h("span", {class: "text"},
            h("strong", {}, `Take ${item.number} `, h("span", {class: "muted mono", text: [item.time, item.seed !== null && item.seed !== undefined ? `seed ${item.seed}` : ""].filter(Boolean).join(" · ")})),
            h("span", {class: "small muted", text: item.note || " "}),
            reviewText(item.review) ? h("span", {class: "small review", title: "How well the take fits its brief, scored by Gemini Decision; the tags are added in Media",
                                                 text: reviewText(item.review)}) : null));
        if (!compare) return card;
        return h("div", {class: "take-cell"}, card,
          h("button", {type: "button", class: "set-b", text: "B", title: `Set take ${item.number} as B`, "aria-label": `Set take ${item.number} as B`,
                       disabled: isA, onclick: () => selectTakes({b: item.index})}));
      }));
      $("take-list").dataset.drawn = drawn;
    }
    $("take-list").hidden = !items.length;
    if ($("take-details").dataset.html !== details) { $("take-details").innerHTML = details; $("take-details").dataset.html = details; }
    $("take-details").hidden = !items.length;
    $("take-actions").hidden = !takeA;
    $("settings-used-box").hidden = !takeA;
    if (takeA) {
      const who = compare ? "take A's" : "this take's";
      $("take-actions-line").textContent = compare ? `Actions use take A (Take ${takeA.number}).` : `Take ${takeA.number} · ${engine().label}`;
      $("reuse-draft").textContent = `Back to ${who} draft`;
      $("use-seed").textContent = `Use ${who} seed`;
      $("use-seed").disabled = takeA.seed === null || takeA.seed === undefined;
      $("reuse-score").textContent = `Reuse ${who} score`;
      $("reuse-score").hidden = app.s.engine !== "yue2";
      $("reuse-score").disabled = !takeA.has_score;
      $("download").href = `${takeA.url}&download=true`;
      $("download").download = takeA.name;
      const used = settings || "Settings used will appear with the generated result.";
      if ($("settings-used").dataset.html !== used) { $("settings-used").innerHTML = used; $("settings-used").dataset.html = used; }
    }
  };

  // Readiness and the next take ---------------------------------------------
  const renderReadiness = () => {
    const {draft, readiness} = app.s;
    const run = app.runs.generate;
    const steps = [["Brief", draft.description.trim() ? "done" : "now"],
                   ["Lyrics", draft.lyrics.trim() ? "done" : app.s.engine === "yue2" ? "now" : ""],
                   ["Take", run?.active ? "now" : app.s.takes.items.length ? "done" : ""]];
    $("steps").replaceChildren(...steps.map(([label, state]) => h("span", {class: `step ${state}`}, h("span", {class: "dot", "aria-hidden": "true"}), label)));
    const singer = app.boot.engines.find((item) => item.id === run?.engine)?.label || "";
    $("readiness").textContent = run?.active ? `A ${singer} take is being made. You can keep editing the next one.` : readiness.message;
    $("readiness").setAttribute("role", readiness.ready ? "status" : "alert");
    $("run-label").textContent = run?.active ? "Generating…" : copy().run;
    $("run").disabled = !app.gpu || !readiness.ready || Boolean(run?.active);
    $("stop").disabled = !run?.active;
    const status = $("run-status");
    status.hidden = !run || (!run.active && !run.error && !run.status);
    status.className = run?.error ? "status error" : "status";
    status.textContent = run?.error || run?.status || "";
  };
  const renderSummary = () => {
    const {brief, seed} = app.s;
    $("summary").textContent = app.s.readiness.ready ? "Plays the current draft with these settings." : "Starts after the draft is ready.";
    const label = (field) => field.choices.find(([, value]) => value === field.value)?.[0] ?? String(field.value);
    const chips = [engine().label, label(brief.vocals), label(brief.language), label(brief.use), `up to ${clock(brief.duration.value)}`,
      seed === -1 ? "New seed each take" : `Seed ${seed}`];
    if (brief.mode) chips.push(label(brief.mode));
    $("summary-chips").replaceChildren(...chips.map((text, index) => h("span", {class: index ? "chip" : "chip good", text})));
  };
  const renderFine = () => {
    const open = Boolean(app.s.ui.fine);
    const total = Object.values(app.s.changed).reduce((sum, count) => sum + count, 0);
    $("fine-count").hidden = !total;
    $("fine-count").textContent = `${total} changed`;
    $("fine-toggle").setAttribute("aria-expanded", String(open));
    $("fine").hidden = !open;
    if (!open) return;
    const shown = app.boot.groups.map((group, index) => [index, group]).filter(([index]) => app.s.fields.some((field) => field.group === index));
    const wanted = Number(app.s.ui.group) || 0;
    const [group, [title, detail]] = shown.find(([index]) => index === wanted) || shown[0];
    const drawn = JSON.stringify([app.s.engine, group, app.s.changed, app.s.fields.filter((field) => field.group === group)]);
    if ($("fine").dataset.drawn === drawn) return;
    $("fine").dataset.drawn = drawn;
    $("fine-nav").replaceChildren(...shown.map(([index, [label]]) => h("button", {
      type: "button", "aria-pressed": String(index === group), onclick: () => { saveUi({group: index}); render(); },
    }, label, app.s.changed[index] ? h("span", {class: "chip warn", "aria-hidden": "true", text: String(app.s.changed[index])}) : null)));
    $("fine-title").textContent = title;
    $("fine-detail").textContent = detail;
    $("fine-rows").replaceChildren(...app.s.fields.filter((field) => field.group === group).map(fineRow));
  };

  const render = () => {
    if (!app.s) return;
    seg($("models"), app.boot.engines.map((item) => [item.label, item.id]), app.s.engine, guard(async (value) => {
      await saveDraft();
      await change("engine", {engine: value});
    }));
    const about = [engine().description];
    if (app.s.engine === "yue2") {
      about.push(" Model license: CC-BY-NC-4.0 · ",
        h("a", {href: "https://huggingface.co/Comfy-Org/YuE2", target: "_blank", rel: "noopener", text: "model details"}));
    }
    $("about").replaceChildren(...about);
    $("workshop-note").hidden = app.gpu;
    const typing = $("fine-rows").contains(document.activeElement) && ["INPUT", "TEXTAREA"].includes(document.activeElement.tagName);
    let failed = null;
    for (const part of [renderBrief, renderReferences, renderDraft, renderTakes, renderReadiness, renderSummary, !typing && renderFine]) {
      try { if (part) part(); } catch (error) { console.error(error); failed = error; }
    }
    if (failed) toast(`Part of the page could not be drawn: ${failed.message}`, true);
    queueFit();
  };

  // Runs --------------------------------------------------------------------
  const hasUnsettledRun = (runs) => Object.entries(runs || {}).some(([kind, run]) => run && !run.active && app.settled[kind] !== run.id);
  const startRun = guard(async (path) => {
    await saveDraft();
    const answer = await post(path);
    take(hasUnsettledRun(answer?.runs) ? await request("GET", "/run?state=true") : answer);
  });
  const poll = async () => {
    if (app.polling) return;
    app.polling = true;
    try {
      for (;;) {
        await new Promise((resolve) => setTimeout(resolve, 1000));
        const answer = await request("GET", "/run");
        take(hasUnsettledRun(answer.runs) ? await request("GET", "/run?state=true") : answer);
        if (!Object.values(app.runs).some((run) => run?.active)) break;
      }
    } catch (error) {
      toast(error.message, true);
    } finally {
      app.polling = false;
    }
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
  // After an action that rewrote the draft, the boxes take the server's text even while focused.
  const takeRewrite = (answer) => {
    if (answer?.state) for (const name of Object.keys(TEXT)) fill(name, answer.state.draft[name]);
    take(answer);
  };
  const takeAction = (path) => guard(async () => {
    const a = app.s.takes.a;
    if (a === null || a === undefined) return;
    await saveDraft();
    takeRewrite(await post(path, {index: a}));
    toast(path.endsWith("use-seed") ? "The next take uses this seed." : path.endsWith("reuse-score") ? "The next take sings this score." : "The draft is back to this take's.");
  });

  const bind = () => {
    for (const name of Object.keys(TEXT)) {
      $(name).addEventListener("input", edited);
      // After the save, a box that was focused while the server rewrote its text catches up.
      $(name).addEventListener("blur", async () => { await autosave(); render(); });
    }
    $("duration").addEventListener("change", guard(async (event) => {
      const value = Number(event.target.value);
      if (!Number.isFinite(value) || event.target.value.trim() === "") throw new Error("Maximum seconds must be a number.");
      await change("edit", {[app.s.brief.duration.name]: value});
      event.target.value = String(app.s.brief.duration.value);
    }));
    $("tab-lyrics").addEventListener("click", () => { app.tab = "lyrics"; render(); });
    $("tab-score").addEventListener("click", () => { app.tab = "score"; render(); });
    $("file-input").addEventListener("change", (event) => { if (event.target.files?.length) upload(event.target.files); });
    $("add-upload").addEventListener("click", () => { $("file-input").value = ""; $("file-input").click(); });
    $("add-library").addEventListener("click", openLibrary);
    const drop = $("references");
    drop.addEventListener("dragover", (event) => { event.preventDefault(); $("add-upload").classList.add("over"); });
    drop.addEventListener("dragleave", () => $("add-upload").classList.remove("over"));
    drop.addEventListener("drop", (event) => {
      event.preventDefault();
      $("add-upload").classList.remove("over");
      if (event.dataTransfer?.files?.length) upload(event.dataTransfer.files);
    });
    $("empty").addEventListener("click", guard(async () => {
      clearTimeout(draftTimer);
      await pendingSave;
      for (const name of Object.keys(TEXT)) fill(name, "");
      take(await post("/empty"));
    }));
    $("run").addEventListener("click", () => startRun("/generate"));
    $("stop").addEventListener("click", guard(async () => {
      const answer = await post("/interrupt");
      if (answer.message) toast(answer.message);
      take(answer);
    }));
    $("write").addEventListener("click", () => startRun("/write"));
    $("revise").addEventListener("click", () => startRun("/write"));
    $("write-generate").addEventListener("click", () => startRun("/write-generate"));
    $("accept").addEventListener("click", guard(async () => {
      await saveDraft();
      try {
        takeRewrite(await post("/accept"));
      } catch (error) {
        // A refused Accept may still have closed the suggestion (the draft changed meanwhile).
        await refresh();
        throw error;
      }
    }));
    $("keep").addEventListener("click", guard(async () => take(await post("/keep"))));
    $("undo").addEventListener("click", guard(async () => {
      await saveDraft();
      takeRewrite(await post("/undo"));
    }));
    $("clear-notes").addEventListener("click", () => { $("notes").value = ""; autosave(); });
    $("restore").addEventListener("click", () => change("restore"));
    $("fine-toggle").addEventListener("click", () => { saveUi({fine: !app.s.ui.fine}); render(); });
    $("fine-close").addEventListener("click", () => { saveUi({fine: false}); render(); });
    $("reuse-draft").addEventListener("click", takeAction("/takes/reuse-draft"));
    $("use-seed").addEventListener("click", takeAction("/takes/use-seed"));
    $("reuse-score").addEventListener("click", takeAction("/takes/reuse-score"));
    $("reuse-latest-score").addEventListener("click", guard(async () => {
      await saveDraft();
      takeRewrite(await post("/takes/reuse-score", {index: "latest"}));
      toast("The next take sings the latest take's score.");
    }));
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
    // A frame in a hidden tab keeps its last size, so its own size can't tell it was shown again:
    // the frame element is watched from the page around it instead.
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
    // Another page may have handed over a picture while this tab was out of sight.
    whenShown(() => { if (app.s) refresh(); });
    try {
      app.boot = await request("GET", "/boot");
      bind();
      $("loading").hidden = true;
      $("editor").hidden = false;
      take(app.boot);
    } catch (error) {
      $("loading").textContent = `Song couldn't open: ${error.message}`;
    }
  };
  start();
})();
