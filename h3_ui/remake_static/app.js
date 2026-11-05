// The Remake page. The browser holds the editor; a long selection's windows live in their project,
// and scrubbing asks the server for nothing unless a source can't play here.
(() => {
  const API = "/remake/api";
  const POLL_MS = 1500;
  const PULSE_MS = 2500;
  const CHECK_DELAY_MS = 200;
  const PREFS = "h3-remake-prefs";
  const BRIDGE = {edit: (model) => `remake-edit-bridge-${model}`, settings: "remake-settings-bridge"};
  const QUEUED = ["pending", "changed", "failed"];
  const $ = (id) => document.getElementById(id);

  const h = (tag, attributes = {}, ...children) => {
    const node = document.createElement(tag);
    for (const [name, value] of Object.entries(attributes)) {
      if (value === undefined || value === null || value === false) continue;
      if (name.startsWith("on")) node.addEventListener(name.slice(2), value);
      else if (name in node) node[name] = value;
      else node.setAttribute(name, value === true ? "" : value);
    }
    node.append(...children.flat(Infinity).filter((child) => child !== null && child !== undefined && child !== false));
    return node;
  };
  const button = (label, onclick, className = "btn", extra = {}) =>
    h("button", {type: "button", className, onclick, ...extra}, label);
  const svg = (markup) => {
    const holder = document.createElement("span");
    holder.innerHTML = markup;
    return holder.firstChild;
  };
  const CHECK = '<svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M2 6.5l2.8 2.8L10 3.5"/></svg>';
  const CROSS = '<svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="M2.5 2.5l7 7M9.5 2.5l-7 7"/></svg>';
  const CHEVRON = '<svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="M2 4l4 4 4-4"/></svg>';

  const state = {
    boot: null, source: null, start: 0, end: 0, lastStart: 0, shortEdge: 480, maxWindow: 226, mix: 0,
    long: false, text: "", project: null, index: 0, frame: 0, mode: "Swap", pinned: true, smart: true, auto: true,
    slots: [null, null, null, null], useCarried: true, suggestion: "", previous: null, view: "source", wipe: 50,
    alignSlot: null, alignView: "Overlay", check: null, run: null, runSeen: -1, output: null, playing: false,
    menu: false, renaming: null, confirming: null, target: null, settings: null, quick: null,
  };
  const prefs = (() => { try { return JSON.parse(localStorage.getItem(PREFS)) || {}; } catch { return {}; } })();
  const savePrefs = () => {
    const kept = {mode: state.mode, outputEdge: state.shortEdge, maxWindow: state.maxWindow, smart: state.smart,
                  editModel: $("edit-model").value,
                  auto: state.auto, pinned: state.pinned, backend: $("writer-backend").value, model: $("writer-model").value};
    try { localStorage.setItem(PREFS, JSON.stringify(kept)); } catch { /* Private windows refuse it. */ }
  };

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
  const post = (path, body) => request("POST", path, body);
  const fileUrl = (path) => `${API}/file?path=${encodeURIComponent(path)}`;

  let toastTimer;
  const toast = (message, error = false) => {
    const node = $("toast");
    node.textContent = message;
    node.className = error ? "toast error" : "toast";
    node.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { node.hidden = true; }, error ? 8000 : 4000);
  };
  // Every action reports its own failure; the page stays usable.
  const safely = (work) => async (...args) => {
    try { return await work(...args); } catch (error) { toast(error.message, true); return undefined; }
  };

  // Timing -----------------------------------------------------------------
  const fps = () => state.boot.fps;
  const clock = (frames) => {
    const seconds = frames / fps();
    return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${(seconds % 60).toFixed(1).padStart(4, "0")}`;
  };
  const selectionFrames = () => Math.max(1, Math.floor((state.end - state.start) * fps() + 1e-6));
  const total = () => state.project?.frames ?? Math.min(selectionFrames(), state.maxWindow);
  const single = () => ({index: 0, own: [0, total()], render: [0, total()], hold: 0, kind: "start",
                         state: state.output ? "done" : "pending", state_name: state.output ? "Done" : "Not generated",
                         take: state.output, carried: null, prompt_auto: false, prompt: ""});
  const windows = () => state.project?.windows ?? [single()];
  const openWindow = () => windows()[state.index] ?? windows()[0];
  const windowAt = (frame) => {
    const found = windows().findIndex((window) => frame >= window.own[0] && frame < window.own[1]);
    return found < 0 ? windows().length - 1 : found;
  };
  // Pins, and the frames extraction reads, count from the start of what this window renders.
  const base = () => videoStart() + openWindow().render[0] / fps();
  const running = () => Boolean(state.run?.active);
  const playheadPin = () => state.frame - openWindow().render[0];
  const videoStart = () => state.project?.start ?? state.start;

  const selection = () => ({sha: state.source?.sha, start: state.start, end: state.end, short_edge: state.shortEdge,
                            mix: state.mix, max_window: state.maxWindow});
  const writer = () => {
    const backend = $("writer-backend").value;
    const openai = backend !== "Gemini";
    return {backend, model: openai ? state.boot.writer.gemini_model : $("writer-model").value,
            openai_model: openai ? $("writer-model").value : state.boot.writer.openai_model,
            gemini_key: openai ? "" : $("writer-key").value, openai_key: openai ? $("writer-key").value : "",
            openai_url: $("writer-url").value};
  };
  const originals = () => Object.fromEntries(state.slots.flatMap((slot, index) =>
    (slot?.original && slot.untouched ? [[index + 1, slot.original]] : [])));
  const editor = () => ({
    video: state.source?.path, selection: selection(), prompt: $("prompt").value, request: $("request").value,
    pinned: state.pinned, smart: state.smart, auto_prompts: state.auto, window: state.index, mode: state.mode,
    slots: state.slots, originals: originals(), use_carried: state.useCarried, writer: writer(),
  });

  // Saving and checking ----------------------------------------------------
  let checkTimer;
  const queueCheck = () => {
    clearTimeout(checkTimer);
    checkTimer = setTimeout(safely(async () => {
      if (!state.source) return;
      state.check = await post("/check", editor());
      renderAction();
      renderNextRun();
    }), CHECK_DELAY_MS);
  };
  // Saves run one at a time: moving quickly across windows must keep them in order.
  let saving = Promise.resolve();
  const saveWindow = () => {
    if (!state.project) return Promise.resolve();
    const body = editor();
    saving = saving.catch(() => {}).then(() => post("/window/save", body));
    return saving;
  };
  const changed = () => { queueCheck(); savePrefs(); };

  const loadEditor = (window) => {
    const pictures = window.pictures ?? [];
    state.slots = [0, 1, 2, 3].map((index) => (pictures[index]
      ? {path: pictures[index], pin: window.pins?.[index] ?? null} : null));
    $("prompt").value = window.prompt ?? "";
    state.useCarried = window.use_carried ?? true;
    state.suggestion = "";
    state.previous = null;
    state.alignSlot = null;
    if (state.view === "align") state.view = "source";
  };

  const goToWindow = safely(async (index) => {
    if (index === state.index || !state.project) return;
    const project = await saveWindow();
    state.project = project ?? state.project;
    state.index = index;
    loadEditor(openWindow());
    renderAll();
    queueCheck();
  });

  const setFrame = (frame, {follow = true} = {}) => {
    state.frame = Math.max(0, Math.min(total() - 1, Math.round(frame)));
    const index = windowAt(state.frame);
    if (follow && index !== state.index) goToWindow(index);
    renderPlayhead();
    seekVideos();
  };

  // Source and selection ---------------------------------------------------
  const showEditor = (open) => {
    $("editor").hidden = !open;
    $("start").hidden = open;
  };
  const applySelection = (data) => {
    state.text = data.text;
    state.long = data.long;
    const previous = state.project?.key;
    state.project = data.project;
    if (data.project && data.project.key !== previous) {
      state.index = 0;
      loadEditor(openWindow());
      state.mode = data.project.mode;
    } else if (!data.project && previous) {
      // A later window's frames belong to its old plan; window 1's become the new selection's.
      if (state.index) loadEditor({});
      state.index = 0;
    }
    state.frame = Math.min(state.frame, total() - 1);
  };
  const refreshSelection = safely(async () => {
    if (!state.source) return;
    applySelection(await post("/selection", {selection: selection()}));
    renderAll();
    queueCheck();
  });

  const showSource = (source) => {
    state.source = source;
    const video = $("source-video");
    delete video.dataset.failed;
    video.src = fileUrl(source.path);
    $("run-status").hidden = !running();
    showEditor(true);
  };

  const useSource = safely(async (source) => {
    state.start = state.lastStart = 0;
    state.end = source.end;
    state.mix = 0;
    state.frame = 0;
    state.index = 0;
    state.project = null;
    state.output = null;
    state.view = "source";
    loadEditor({});
    showSource(source);
    await refreshSelection();
  });

  const upload = safely(async (file) => {
    if (!file) return;
    $("upload-status").textContent = `Uploading ${file.name}…`;
    const form = new FormData();
    form.append("file", file);
    try {
      await useSource(await request("POST", "/source", form, {form: true}));
    } finally {
      $("upload-status").textContent = "";
    }
  });

  const openProject = safely(async (key) => {
    const data = await post(`/projects/${key}/load`);
    const chosen = data.selection;
    Object.assign(state, {start: chosen.start, lastStart: chosen.start, end: chosen.end, shortEdge: chosen.short_edge,
                          mix: chosen.mix, maxWindow: chosen.max_window, mode: data.mode, smart: data.smart,
                          pinned: data.pinned, auto: data.auto_prompts, long: true, project: data, index: 0,
                          frame: 0, output: data.output, view: "source", menu: false});
    $("request").value = data.request;
    loadEditor(openWindow());
    showSource(data.source_info);
    await refreshSelection();
    if (data.note) toast(data.note);
  });

  const closeProject = () => {
    state.source = null;
    state.project = null;
    state.menu = false;
    showEditor(false);
    renderStart();
  };

  const refreshProjects = safely(async () => {
    state.boot.projects = (await request("GET", "/projects")).projects;
    renderMenu();
    renderStart();
  });

  // Pictures ---------------------------------------------------------------
  const emptySlot = () => state.slots.findIndex((slot) => !slot);

  const measure = safely(async (index, view = "Overlay") => {
    const slot = state.slots[index];
    if (!slot || slot.untouched || slot.pin === null || slot.pin === undefined) return;
    slot.align = await post("/align/check", {video: state.source.path, base: base(), picture: slot.path,
                                             pin: slot.pin, view});
    renderPictures();
    if (state.view === "align") renderViewer();
  });

  const replacePicture = (index, path, {before = null, untouched = false} = {}) => {
    const slot = state.slots[index];
    // A picture for an empty slot belongs at the playhead, like an extracted frame.
    state.slots[index] = {...slot, path, pin: slot?.pin ?? playheadPin(), untouched, sent: false, align: null, before};
    afterPictures();
    measure(index);
  };
  const addPicture = (path, pin = playheadPin()) => {
    const index = state.target ?? emptySlot();
    state.target = null;
    if (index < 0) throw new Error("All four slots are full. Remove a picture first.");
    state.slots[index] = {path, pin: state.slots[index]?.pin ?? pin};
    afterPictures();
    measure(index);
  };
  const afterPictures = () => {
    renderPictures();
    renderTimeline();
    changed();
    saveWindow()?.then((project) => {
      if (project) {
        state.project = project;
        renderTimeline();
        renderAction();
      }
    }).catch(() => {});
  };

  const extractFrame = safely(async () => {
    const index = emptySlot();
    if (index < 0) throw new Error("All four slots are full. Remove a picture first.");
    const data = await post("/extract", {video: state.source.path, base: base(), frame: playheadPin()});
    state.slots[index] = {path: data.path, pin: data.pin, original: data.original, untouched: true};
    afterPictures();
  });

  const sleep = (milliseconds) => new Promise((resolve) => { setTimeout(resolve, milliseconds); });

  // Repaints the picture where it is. It finishes on its own if you move on, and only lands if its slot still holds it.
  const quickEdit = safely(async (index) => {
    const slot = state.slots[index];
    const prompt = (slot.quick ?? "").trim();
    if (!prompt) throw new Error("Write what to change in the frame.");
    const opened = state.index;
    state.quick = {index, status: "Waiting for the GPU…"};
    renderPictures();
    try {
      let {run} = await post("/quick-edit", {picture: slot.path, prompt});
      while (run.active) {
        await sleep(POLL_MS);
        ({run} = await request("GET", "/quick-edit/run"));
        state.quick.status = run.status;
        const line = $("quick-status");
        if (line) line.textContent = run.status;
      }
      if (run.error) throw new Error(run.error);
      if (!run.result_url) return;
      if (state.index !== opened || state.slots[index]?.path !== slot.path) {
        toast("The edit finished after the picture changed, so it's in Media instead.");
        return;
      }
      replacePicture(index, run.result_url, {before: {path: slot.path, untouched: Boolean(slot.untouched)}});
    } finally {
      state.quick = null;
      renderPictures();
    }
  });

  const sendToImage = safely(async (index) => {
    const slot = state.slots[index];
    const model = $("edit-model").value;
    await post("/send-to-image", {path: slot.path, model});
    const bridge = window.parent === window ? null : window.parent.document.getElementById(BRIDGE.edit(model));
    if (!bridge) throw new Error("Open Remake in the app to edit in Image.");
    slot.sent = true;
    renderPictures();
    bridge.click();
  });

  const openLibrary = safely(async (slot = null) => {
    state.target = slot;
    const dialog = $("library");
    const grid = h("div", {className: "library-grid"});
    const search = async (query = "") => {
      const {items} = await request("GET", `/library?limit=60&query=${encodeURIComponent(query)}`);
      grid.replaceChildren(...items.map((item) => h("button", {
        type: "button", title: item.caption, "aria-label": item.caption || "Image",
        onclick: safely(async () => {
          const {path} = await post("/picture/library", {source: item.path});
          dialog.close();
          if (slot !== null && state.slots[slot]) replacePicture(slot, path);
          else addPicture(path);
        }),
      }, h("img", {src: item.thumb, alt: ""}))));
    };
    dialog.replaceChildren(
      h("div", {className: "row between"}, h("h2", {}, "Choose from Media"), button("Close", () => dialog.close(), "btn quiet")),
      h("input", {className: "field", placeholder: "Search images", "aria-label": "Search images",
                  onchange: safely((event) => search(event.target.value))}),
      grid,
      h("div", {}, button("Upload an image", () => { dialog.close(); $("picture-input").click(); })),
    );
    await search();
    dialog.showModal();
  });

  // Running ----------------------------------------------------------------
  const queue = () => (state.project ? state.project.windows.filter((window) => QUEUED.includes(window.state)) : []);
  const allTaken = () => Boolean(state.project) && state.project.windows.every((window) => window.take);

  const startRun = safely(async () => {
    const {run} = await post("/generate", editor());
    watchRun(run);
  });

  const combine = safely(async () => {
    $("run-status").hidden = false;
    $("run-status").textContent = "Joining the windows…";
    const {output, status} = await post("/combine", {selection: selection()});
    $("run-status").textContent = status;
    state.project = (await request("GET", `/projects/${state.project.key}`));
    state.output = output;
    state.view = "result";
    renderAll();
  });

  const primary = () => {
    if (state.project && !queue().length) return combine();
    return startRun();
  };

  const regenerate = safely(async () => {
    state.project = await post("/window/mark", editor());
    renderTimeline();
    await startRun();
  });

  const syncProject = safely(async () => {
    if (!state.project) return;
    const project = await request("GET", `/projects/${state.project.key}`);
    state.project = project;
    const open = openWindow();
    // A prompt the run wrote reaches the editor; one you're typing stays yours.
    if (open.prompt_auto && document.activeElement !== $("prompt")) $("prompt").value = open.prompt;
    if (project.output) state.output = project.output;
  });

  let pollTimer;
  const watchRun = (run) => {
    const wasActive = running();
    state.run = run;
    const status = $("run-status");
    status.hidden = !run;
    if (run) {
      status.textContent = run.error || run.status;
      status.className = run.error ? "status error" : "status";
    }
    renderAction();
    if (run && run.revision !== state.runSeen) {
      state.runSeen = run.revision;
      syncProject().then(() => {
        if (run.result_url && !state.project) state.output = run.result_url;
        if (wasActive && !run.active && !run.error && state.output) state.view = "result";
        renderAll();
        queueCheck();
        refreshProjects();
      });
    }
    clearTimeout(pollTimer);
    if (run?.active) {
      pollTimer = setTimeout(safely(async () => watchRun((await request("GET", "/run")).run)), POLL_MS);
    }
  };

  const pulse = async () => {
    try {
      const frame = window.frameElement;
      if (!document.hidden && (!frame || frame.offsetParent !== null)) {
        const data = await request("GET", `/pulse?collect=${Boolean(state.source)}`);
        for (const {slot, path} of data.inbox) replacePicture(slot - 1, path);
        if (data.settings !== state.settings) {
          state.settings = data.settings;
          queueCheck();
        }
      }
    } catch { /* The next pulse tries again. */ }
    setTimeout(pulse, PULSE_MS);
  };

  // Rendering --------------------------------------------------------------
  const pressed = (list, current, pick) => list.map(([value, label]) =>
    h("button", {type: "button", "aria-pressed": String(value === current), onclick: () => pick(value)}, label));

  const bold = (text) => String(text).split("**").map((part, index) => (index % 2 ? h("strong", {}, part) : part));

  const projectRow = (row, onclick) => h("button", {type: "button", className: "project-row", onclick},
    row.thumb ? h("img", {src: fileUrl(row.thumb), alt: ""}) : h("img", {alt: ""}),
    h("span", {}, h("strong", {}, row.name), h("span", {className: "small muted"}, row.meta)));

  const renderStart = () => {
    $("no-projects").hidden = state.boot.projects.length > 0;
    $("start-projects").replaceChildren(...state.boot.projects.map((row) => projectRow(row, () => openProject(row.key))));
  };

  const renderMenu = () => {
    const menu = $("menu");
    menu.hidden = !state.menu;
    $("project-button").setAttribute("aria-expanded", String(state.menu));
    if (!state.menu) return;
    const rows = state.boot.projects.map((row) => {
      const open = row.key === state.project?.key;
      const renaming = state.renaming === row.key;
      const confirming = state.confirming === row.key;
      const name = renaming
        ? h("input", {className: "field", value: row.name, "aria-label": "Project name",
                      onchange: safely(async (event) => {
                        const data = await request("PATCH", `/projects/${row.key}`, {name: event.target.value});
                        state.boot.projects = data.projects;
                        state.renaming = null;
                        if (open) state.project.name = event.target.value;
                        renderAction();
                        renderMenu();
                      })})
        : button(row.name, () => openProject(row.key), "btn quiet");
      return h("div", {className: open ? "menu-row on" : "menu-row"},
        h("div", {className: "grow"}, name, h("div", {className: "small muted"}, row.meta)),
        confirming
          ? h("div", {className: "row"},
            h("span", {className: "small muted grow"}, "Deletes its renders and copy of the source. Combined videos stay in Media."),
            button("Delete project", safely(async () => {
              const data = await request("DELETE", `/projects/${row.key}`);
              state.boot.projects = data.projects;
              state.confirming = null;
              if (open) closeProject();
              renderMenu();
            })),
            button("Keep it", () => { state.confirming = null; renderMenu(); }, "btn quiet"))
          : h("div", {className: "row"},
            button(renaming ? "Done" : "Rename", () => { state.renaming = renaming ? null : row.key; renderMenu(); }, "btn quiet"),
            button("Delete", () => { state.confirming = row.key; state.renaming = null; renderMenu(); }, "btn quiet")));
    });
    menu.replaceChildren(
      h("div", {className: "row between"}, h("span", {className: "small muted strong"}, "Projects"),
        button("New from a video", closeProject)),
      ...rows);
  };

  const renderAction = () => {
    const name = state.project?.name ?? state.source?.name ?? "";
    const line = state.project
      ? `${(state.project.frames / fps()).toFixed(1)} s · ${state.project.rendered} of ${windows().length} windows rendered`
      : `${state.source?.info ?? ""}`;
    $("project-button").replaceChildren(h("strong", {}, name, svg(CHEVRON)), h("span", {className: "small muted"}, line));
    $("mode").replaceChildren(...pressed([["Swap", "Swap"], ["Edit", "Edit"]], state.mode, (mode) => {
      state.mode = mode;
      renderAction();
      changed();
    }));
    $("mode").children[0].title = "Replaces a character or subject";
    $("mode").children[1].title = "Changes objects, background, weather or light";

    const check = state.check;
    const step = check?.step;
    const order = ["source", "frames", "prompt"];
    const reached = step ? order.indexOf(step) : order.length;
    $("steps").replaceChildren(...["Source", "Frames", "Prompt"].map((label, index) => {
      const done = index < reached;
      return h("span", {className: done ? "step done" : index === reached ? "step now" : "step"},
        done ? svg(CHECK) : h("span", {className: "dot"}), label);
    }));
    $("readiness").textContent = check?.message ?? "";

    const action = $("primary");
    const waiting = queue().length;
    let label = "Generate remake";
    let blocked = !check?.ready;
    if (state.project) {
      if (waiting) label = waiting === 1 ? "Render 1 window" : `Render ${waiting} windows`;
      else if (allTaken() && !state.project.output) { label = "Combine final video"; blocked = false; }
      else { label = "All windows rendered"; blocked = true; }
    } else if (state.output) label = "Generate again";
    action.textContent = running() ? "Generating…" : label;
    action.disabled = blocked || running();
    $("stop").hidden = !running();
  };

  const renderNextRun = () => {
    const details = state.check?.next_run;
    if (!details) return;
    $("next-output").textContent = `Remake ${state.mode} · ${details.profile} · ${details.output} · ${details.sampling}`;
    $("next-context").textContent = details.context.slice(1).join(" · ");
    $("next-notes").replaceChildren(...details.notes.map((note) => h("p", {className: "small muted"}, note)));
  };

  const resultSource = () => (state.project?.output ?? (state.project ? openWindow().take : state.output));

  const seekVideos = () => {
    const source = $("source-video");
    const time = videoStart() + state.frame / fps();
    const seek = (video, value) => {
      if (Math.abs(video.currentTime - value) < 0.02) return;
      if (video.readyState >= 1) video.currentTime = value;
      else video.addEventListener("loadedmetadata", () => { video.currentTime = value; }, {once: true});
    };
    if (!state.playing) seek(source, time);
    const result = $("result-video");
    if (!result.hidden && !state.playing) {
      const whole = Boolean(state.project?.output) || !state.project;
      seek(result, (whole ? state.frame : state.frame - openWindow().render[0]) / fps());
    }
    const fallback = $("fallback");
    if (!fallback.hidden) {
      fallback.src = `${API}/frame?path=${encodeURIComponent(state.source.path)}&base=${videoStart()}&frame=${state.frame}`;
    }
  };

  const renderViewer = () => {
    const view = state.view;
    const align = view === "align";
    const tabs = [["source", "Source"], ["result", "Result"], ["compare", "Compare"]];
    if (state.alignSlot !== null) tabs.push(["align", "Alignment"]);
    $("tabs").replaceChildren(...pressed(tabs, view, (value) => { state.view = value; renderViewer(); }));
    const take = resultSource();
    const wantsResult = view === "result" || view === "compare";
    const result = $("result-video");
    const url = take ? fileUrl(take) : "";
    if (take && result.dataset.url !== url) {
      result.src = url;
      result.dataset.url = url;
    }
    result.hidden = !(wantsResult && take);
    $("source-video").hidden = view === "result" || align || $("source-video").dataset.failed === "true";
    $("fallback").hidden = view === "result" || align || $("source-video").dataset.failed !== "true";
    $("stage-empty").hidden = !(wantsResult && !take);
    $("empty-line").textContent = state.project
      ? `Window ${state.index + 1} has no render. The source is still in the Source view.`
      : "Your result appears here, beside the source.";
    $("wipe-parts").hidden = !(view === "compare" && take);
    result.style.clipPath = view === "compare" ? `inset(0 0 0 ${state.wipe}%)` : "";
    $("wipe-line").style.left = `${state.wipe}%`;

    const slot = align ? state.slots[state.alignSlot] : null;
    const check = slot?.align;
    $("align-image").hidden = !check;
    if (check) $("align-image").src = fileUrl(check.preview);
    $("align-bar").hidden = !align;
    $("transport").hidden = align;
    if (align) {
      $("align-views").replaceChildren(...pressed([["Overlay", "Overlay"], ["Difference", "Difference"]], state.alignView, (value) => {
        state.alignView = value;
        renderViewer();
        measure(state.alignSlot, value);
      }));
      $("align-note").textContent = check?.note ?? "Measuring…";
      $("align-apply").hidden = check?.state !== "shifted";
    }
    seekVideos();
  };

  const renderPlayhead = () => {
    const last = Math.max(1, total() - 1);
    const percent = `${state.frame / last * 100}%`;
    $("playhead").style.left = percent;
    $("scrub").max = last;
    $("scrub").value = state.frame;
    $("viewer-line").textContent = `${clock(state.frame)} · frame ${playheadPin()} of ${state.project ? `window ${state.index + 1}` : "selection"}`;
    $("timecode").textContent = `Source ${clock(state.frame + videoStart() * fps())}`;
    for (const mark of $("marks").children) mark.classList.toggle("on", Number(mark.dataset.frame) === state.frame);
  };

  const pinMarks = () => {
    const open = openWindow();
    const marks = [];
    state.slots.forEach((slot, index) => {
      if (slot && slot.pin !== null && slot.pin !== undefined) marks.push({slot: index + 1, at: open.render[0] + slot.pin, untouched: slot.untouched, window: state.index});
    });
    windows().forEach((window, index) => {
      if (index === state.index) return;
      (window.pins ?? []).forEach((pin, slot) => {
        if (pin !== null) marks.push({slot: slot + 1, at: window.render[0] + pin, window: index});
      });
    });
    return marks;
  };

  const renderTimeline = () => {
    const last = Math.max(1, total() - 1);
    const frames = total();
    const start = videoStart();
    if (state.source) {
      const seconds = frames / fps();
      $("strip").style.backgroundImage = `url("${API}/strip?path=${encodeURIComponent(state.source.path)}&start=${start}&seconds=${seconds}")`;
      const duration = state.source.duration || 1;
      $("band").style.left = `${state.start / duration * 100}%`;
      $("band").style.width = `${Math.min(100, (state.end - state.start) / duration * 100)}%`;
      $("source-length").textContent = `${duration.toFixed(1)} s source`;
    }
    $("start-time").value = state.start;
    $("end-time").value = state.end;
    $("short-edge").value = state.shortEdge;
    $("max-window").value = state.maxWindow;
    $("smart").checked = state.smart;
    $("auto-prompts").checked = state.auto;
    $("window-options").hidden = !state.long;
    const line = state.project ? state.project.summary : state.text;
    $("summary").replaceChildren(...bold(`${line}${state.project ? " · hatched frames are held from the window before" : ""}`));

    $("cuts").replaceChildren(...windows().slice(1).filter((window) => window.kind === "new shot")
      .map((window) => h("div", {className: "cut", style: `left:${window.own[0] / last * 100}%`})));
    $("wins").replaceChildren(...windows().map((window, index) => {
      const span = window.own[1] - window.own[0];
      const label = state.project ? String(index + 1) : state.long ? "Not split yet" : "One render";
      const node = h("button", {type: "button", className: `win ${window.state}`, "aria-pressed": String(index === state.index),
                                "aria-label": `${state.project ? `Window ${index + 1}, ${window.kind}, ` : ""}${window.state_name}`,
                                onclick: () => setFrame(window.own[0])},
        window.hold ? h("span", {className: "hold", style: `width:${window.hold / span * 100}%`}) : null,
        h("span", {className: "top"}, h("strong", {}, label), state.project ? h("span", {className: "kind"}, window.kind) : null),
        h("span", {className: "state"}, window.state_name));
      node.style.left = `calc(${window.own[0] / frames * 100}% + 2px)`;
      node.style.width = `calc(${span / frames * 100}% - 4px)`;
      return node;
    }));
    $("marks").replaceChildren(...pinMarks().map((mark) => {
      const node = button(String(mark.slot), () => setFrame(mark.at), `pinmark${mark.untouched ? " unedited" : ""}`,
        {title: `Picture ${mark.slot} at frame ${mark.at}`, "aria-label": `Picture ${mark.slot}`});
      node.style.left = `${Math.min(100, mark.at / last * 100)}%`;
      node.dataset.frame = mark.at;
      return node;
    }));
    $("final").hidden = !state.long;
    $("split").hidden = Boolean(state.project);
    $("combine").hidden = !state.project;
    $("combine").disabled = !allTaken() || Boolean(state.project?.output) || running();
    $("final-line").textContent = !state.project ? "Longer than one window. A run splits it, or split now to edit each window first."
      : state.project.output ? "Joined with the source's soundtrack and saved to Media."
      : allTaken() ? "Every window has a take. A run joins them itself, or combine now."
      : `Joins when every window has a take: ${state.project.windows.filter((window) => window.take).length} of ${windows().length} so far.`;
    renderPlayhead();
  };

  const pictureRow = (slot, index) => {
    const pinField = state.pinned ? h("label", {className: "inline"}, h("span", {className: "small muted"}, "Frame"),
      h("input", {className: "field tight mono", type: "number", min: 0, value: slot.pin ?? "",
                  "aria-label": `Picture ${index + 1} frame`,
                  onchange: (event) => {
                    slot.pin = event.target.value === "" ? null : Number(event.target.value);
                    slot.align = null;
                    afterPictures();
                    measure(index);
                  }})) : null;
    const status = slot.untouched ? ["Not edited", "chip warn"] : slot.align?.state === "shifted" ? ["Shifted", "chip warn"]
      : slot.align?.state === "lined_up" ? ["Lined up", "chip good"] : ["Edited", "chip"];
    const editing = state.quick?.index === index;
    const quickRow = h("div", {className: "quick"},
      h("input", {className: "field tight", placeholder: "Describe the change", value: slot.quick ?? "",
                  disabled: Boolean(state.quick), "aria-label": `Quick edit for picture ${index + 1}`,
                  oninput: (event) => { slot.quick = event.target.value; },
                  onkeydown: (event) => { if (event.key === "Enter") quickEdit(index); }}),
      editing
        ? button("Interrupt", safely(async () => toast((await post("/quick-edit/interrupt")).message)))
        : button("Quick edit", () => quickEdit(index), slot.untouched ? "btn primary" : "btn",
          {disabled: Boolean(state.quick)}));
    return h("div", {className: `pic${state.view === "align" && state.alignSlot === index ? " on" : ""}`},
      h("img", {src: fileUrl(slot.path), alt: `Picture ${index + 1}`}),
      h("div", {className: "body"},
        h("div", {className: "row tightrow"},
          h("strong", {className: "small"}, `Picture ${index + 1}`), h("span", {className: status[1]}, status[0]),
          h("span", {className: "grow"}),
          button(svg(CROSS), () => {
            state.slots[index] = null;
            if (state.alignSlot === index) { state.alignSlot = null; state.view = "source"; renderViewer(); }
            afterPictures();
          }, "btn quiet mini", {"aria-label": `Remove picture ${index + 1}`})),
        quickRow,
        editing ? h("p", {className: "small muted", id: "quick-status", role: "status"}, state.quick.status) : null,
        h("div", {className: "row tightrow"}, pinField,
          slot.before ? button("Undo edit", () => replacePicture(index, slot.before.path, {untouched: slot.before.untouched}))
            : null,
          // Still offered once sent: the frame can go to another model.
          button("Edit in Image", () => sendToImage(index)),
          slot.sent ? button("Choose the edited image", () => openLibrary(index), "btn primary") : null,
          slot.untouched ? (slot.sent ? null : button("Replace", () => openLibrary(index)))
            : button("Check alignment", () => {
              state.alignSlot = index;
              state.view = "align";
              setFrame(openWindow().render[0] + (slot.pin ?? 0), {follow: false});
              renderViewer();
              renderPictures();
              measure(index, state.alignView);
            }))));
  };

  const carriedRow = () => {
    const open = openWindow();
    const previous = windows()[state.index - 1];
    const ready = Boolean(open.carried);
    return h("div", {className: "pic"},
      ready ? h("img", {src: fileUrl(open.carried), alt: "Last frame of the render before this window"}) : h("div", {className: "wait"}),
      h("div", {className: "body"},
        h("strong", {className: "small"}, `Carried from window ${state.index}`),
        h("p", {className: "small muted"}, !previous?.take ? `Arrives when window ${state.index} has rendered.`
          : open.hold ? `Its last frame passes on the look. The first ${open.hold} frames here are held from that render, so the two meet in a plain cut.`
          : "Its last frame passes on the look across the cut."),
        h("div", {className: "row tightrow"},
          h("label", {className: "inline check"}, h("input", {type: "checkbox", checked: state.useCarried, onchange: (event) => {
            state.useCarried = event.target.checked;
            changed();
            saveWindow();
          }}), "Use the carried frame"),
          button("Copy to a slot", safely(async () => {
            const data = await post("/window/carried", {selection: selection(), window: state.index});
            state.useCarried = false;
            addPicture(data.path, data.pin);
          }), "btn", {disabled: !ready || emptySlot() < 0}))));
  };

  const renderPictures = () => {
    const rows = [];
    if (state.project && state.index > 0 && state.smart) rows.push(carriedRow());
    state.slots.forEach((slot, index) => { if (slot) rows.push(pictureRow(slot, index)); });
    const empty = emptySlot();
    if (empty >= 0) {
      rows.push(h("div", {className: "empty-slot"}, h("span", {}, `Picture ${empty + 1} is empty`),
        h("span", {className: "row tightrow"}, button("From Media", () => openLibrary()))));
    }
    $("pictures").replaceChildren(...rows);
    const count = state.slots.filter(Boolean).length;
    $("pic-count").textContent = `${count} of 4`;
    $("pinned").checked = state.pinned;
    $("extract").disabled = empty < 0;
    $("extract-hint").textContent = empty < 0 ? "All four slots are full" : `Fills Picture ${empty + 1}, pinned here`;
  };

  const renderPromptMeta = () => {
    const text = $("prompt").value;
    const open = openWindow();
    const count = state.boot.sections.filter((name) => new RegExp(`^\\s*${name}\\s*:`, "m").test(text)).length;
    const written = open.prompt_auto && text === open.prompt;
    $("origin").textContent = !text.trim() ? "Not written yet" : written ? "Written, replaced on render" : "Typed by you, stays";
    $("origin").className = text.trim() && !written ? "chip good" : "chip";
    $("sections").textContent = `${count} of 6 sections`;
    $("sections").className = count === 6 ? "chip good" : "chip";
    $("prompt-label").textContent = state.project ? `Remake prompt for window ${state.index + 1}` : "Remake prompt";
    $("request-label").textContent = state.project ? "Edit request, for the whole project" : "Edit request";
    $("suggestion-box").hidden = !state.suggestion;
    $("suggestion").textContent = state.suggestion;
    $("undo").hidden = state.previous === null;
    const openai = $("writer-backend").value !== "Gemini";
    $("writer-line").textContent = openai ? "OpenAI-compatible services only see the frames." : "Gemini reads the clip and the frames.";
    $("writer-url-row").hidden = !openai;
  };

  const renderInspector = () => {
    const open = openWindow();
    $("scope-title").textContent = state.project ? `Window ${state.index + 1}` : "Selection";
    $("scope-time").textContent = `${clock(open.own[0])}–${clock(open.own[1])}${state.project ? ` · ${open.kind} · ${open.state_name}` : ""}`;
    $("regenerate").hidden = !state.project;
    $("regenerate").disabled = running() || !state.check?.ready;
    let notice = "";
    if (open.state === "outdated") {
      notice = open.kind === "continues"
        ? `Window ${state.index} changed. This take still joins, but regenerate it to remove the jump at that join.`
        : `Window ${state.index} changed. This take still joins, and you decide whether to regenerate it.`;
    } else if (open.state === "failed") notice = "This window failed. Regenerate it to try again.";
    $("notice").hidden = !notice;
    $("notice").textContent = notice;
    renderPictures();
    renderPromptMeta();
  };

  const renderAll = () => {
    if (!state.source) return;
    renderAction();
    renderTimeline();
    renderInspector();
    renderViewer();
    renderMenu();
    renderNextRun();
  };

  // Playback ---------------------------------------------------------------
  const activeVideo = () => (state.view === "result" && !$("result-video").hidden ? $("result-video") : $("source-video"));
  const setPlaying = (playing) => {
    state.playing = playing;
    const videos = [$("source-video"), $("result-video")].filter((video) => !video.hidden && video.src);
    for (const video of videos) {
      video.muted = video !== activeVideo();
      if (playing) video.play().catch(() => {});
      else video.pause();
    }
    $("play").setAttribute("aria-label", playing ? "Pause" : "Play");
    if (!playing) setFrame(state.frame);
  };
  const followPlayback = (event) => {
    if (!state.playing || event.target !== activeVideo()) return;
    const whole = event.target.id === "source-video" || Boolean(state.project?.output) || !state.project;
    const zero = event.target.id === "source-video" ? videoStart() : 0;
    const offset = whole ? 0 : openWindow().render[0];
    const frame = (event.target.currentTime - zero) * fps() + offset;
    if (frame >= total() - 1) { state.frame = total() - 1; setPlaying(false); return; }
    state.frame = Math.round(frame);
    renderPlayhead();
  };

  // Wiring -----------------------------------------------------------------
  const bind = () => {
    $("choose-video").addEventListener("click", () => $("video-input").click());
    $("video-input").addEventListener("change", (event) => { upload(event.target.files[0]); event.target.value = ""; });
    $("picture-input").addEventListener("change", safely(async (event) => {
      const form = new FormData();
      form.append("file", event.target.files[0]);
      event.target.value = "";
      const {path} = await request("POST", "/picture", form, {form: true});
      if (state.target !== null && state.slots[state.target]) {
        const index = state.target;
        state.target = null;
        replacePicture(index, path);
      } else addPicture(path);
    }));
    const drop = $("drop");
    drop.addEventListener("dragover", (event) => { event.preventDefault(); drop.classList.add("over"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("over"));
    drop.addEventListener("drop", (event) => {
      event.preventDefault();
      drop.classList.remove("over");
      upload(event.dataTransfer.files[0]);
    });

    $("project-button").addEventListener("click", () => { state.menu = !state.menu; renderMenu(); });
    $("short-edge").addEventListener("change", (event) => { state.shortEdge = Number(event.target.value); savePrefs(); refreshSelection(); });
    $("max-window").addEventListener("change", (event) => { state.maxWindow = Number(event.target.value); savePrefs(); refreshSelection(); });
    $("start-time").addEventListener("change", (event) => {
      const start = Math.max(0, Number(event.target.value) || 0);
      // Pins count from Start, so each stays on its source frame.
      const moved = Math.round((start - state.lastStart) * fps());
      for (const slot of state.slots) {
        if (slot && slot.pin !== null && slot.pin !== undefined) slot.pin = slot.pin >= moved ? slot.pin - moved : null;
      }
      state.start = state.lastStart = start;
      refreshSelection();
    });
    $("end-time").addEventListener("change", (event) => { state.end = Number(event.target.value) || 0; refreshSelection(); });
    $("smart").addEventListener("change", (event) => { state.smart = event.target.checked; renderPictures(); changed(); });
    $("auto-prompts").addEventListener("change", (event) => { state.auto = event.target.checked; changed(); });
    $("pinned").addEventListener("change", (event) => { state.pinned = event.target.checked; renderPictures(); changed(); });
    $("scrub").addEventListener("input", (event) => setFrame(Number(event.target.value)));
    $("wipe").addEventListener("input", (event) => { state.wipe = Number(event.target.value); renderViewer(); });
    $("back").addEventListener("click", () => setFrame(state.frame - 1));
    $("next").addEventListener("click", () => setFrame(state.frame + 1));
    $("play").addEventListener("click", () => setPlaying(!state.playing));
    $("extract").addEventListener("click", extractFrame);
    $("primary").addEventListener("click", primary);
    $("regenerate").addEventListener("click", regenerate);
    $("combine").addEventListener("click", combine);
    $("split").addEventListener("click", safely(async () => {
      state.project = await post("/split", {selection: selection(), video: state.source.path});
      state.index = 0;
      renderAll();
      queueCheck();
      refreshProjects();
    }));
    $("stop").addEventListener("click", safely(async () => {
      const data = await post("/interrupt");
      toast(data.message);
      watchRun(data.run);
    }));
    $("align-apply").addEventListener("click", safely(async () => {
      const index = state.alignSlot;
      const slot = state.slots[index];
      const {path} = await post("/align/apply", {video: state.source.path, base: base(), picture: slot.path, pin: slot.pin});
      replacePicture(index, path);
    }));
    $("source-video").addEventListener("timeupdate", followPlayback);
    $("result-video").addEventListener("timeupdate", followPlayback);
    $("source-video").addEventListener("error", () => {
      $("source-video").dataset.failed = "true";
      renderViewer();
    });
    $("request").addEventListener("input", queueCheck);
    $("prompt").addEventListener("input", () => { renderPromptMeta(); queueCheck(); });
    $("prompt").addEventListener("change", () => { saveWindow(); });
    $("writer-backend").addEventListener("change", () => { fillWriterModels(); renderPromptMeta(); savePrefs(); });
    $("writer-model").addEventListener("change", savePrefs);
    $("edit-model").addEventListener("change", savePrefs);
    $("rewrite").addEventListener("click", safely(async (event) => {
      const control = event.currentTarget;
      control.disabled = true;
      control.textContent = "Writing…";
      try {
        const {suggestion, status} = await post("/rewrite", editor());
        state.suggestion = suggestion;
        renderPromptMeta();
        if (!suggestion) toast(status, true);
      } finally {
        control.disabled = false;
        control.textContent = "Rewrite / enhance";
      }
    }));
    $("accept").addEventListener("click", () => {
      state.previous = $("prompt").value;
      $("prompt").value = state.suggestion;
      state.suggestion = "";
      renderPromptMeta();
      queueCheck();
      saveWindow();
    });
    $("keep").addEventListener("click", () => { state.suggestion = ""; renderPromptMeta(); });
    $("undo").addEventListener("click", () => {
      $("prompt").value = state.previous;
      state.previous = null;
      renderPromptMeta();
      queueCheck();
      saveWindow();
    });
    $("open-settings").addEventListener("click", () => {
      window.parent.document.getElementById(BRIDGE.settings)?.click();
    });
  };

  const fillWriterModels = () => {
    const openai = $("writer-backend").value !== "Gemini";
    const {gemini_models: gemini, gemini_model: firstGemini, openai_models: others, openai_model: firstOther} = state.boot.writer;
    const models = openai ? others : gemini;
    $("writer-model").replaceChildren(...models.map((name) => h("option", {value: name}, name)));
    $("writer-model").value = models.includes(prefs.model) ? prefs.model : openai ? firstOther : firstGemini;
  };

  // In the tab, the frame grows to fit, so the app's page is the only thing that scrolls.
  const fitFrame = () => {
    const frame = window.frameElement;
    if (!frame) return;
    const height = document.documentElement.scrollHeight;
    if (height > frame.offsetHeight) frame.style.height = `${height}px`;
  };
  let fitQueued = false;
  const queueFit = () => {
    if (fitQueued) return;
    fitQueued = true;
    requestAnimationFrame(() => { fitQueued = false; fitFrame(); });
  };

  // Follows the surrounding app's theme when framed; a page of its own follows the system.
  const applyTheme = () => {
    let dark = matchMedia("(prefers-color-scheme: dark)").matches;
    try {
      if (window.parent !== window) dark = window.parent.document.body.classList.contains("dark");
    } catch { /* Framed by another origin. */ }
    document.documentElement.dataset.theme = dark ? "dark" : "light";
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
    try {
      state.boot = await request("GET", "/boot");
      state.shortEdge = prefs.outputEdge ?? state.boot.short_edge;
      state.maxWindow = prefs.maxWindow ?? state.boot.max_window;
      Object.assign(state, {mode: prefs.mode ?? "Swap", smart: prefs.smart ?? true, auto: prefs.auto ?? true, pinned: prefs.pinned ?? true});
      $("short-edge").replaceChildren(h("option", {value: 0}, "Auto, match source"),
        ...state.boot.short_edges.map((edge) => h("option", {value: edge}, String(edge))));
      $("edit-model").replaceChildren(...state.boot.edit_models.map((model) => h("option", {value: model.id}, model.label)));
      if (state.boot.edit_models.some((model) => model.id === prefs.editModel)) $("edit-model").value = prefs.editModel;
      $("max-window").replaceChildren(...state.boot.lengths.map((length) =>
        h("option", {value: length}, `${(length / fps()).toFixed(1)} s (${length} frames)`)));
      if (prefs.backend) $("writer-backend").value = prefs.backend;
      fillWriterModels();
      bind();
      renderStart();
      showEditor(false);
      if (state.boot.run) watchRun(state.boot.run);
      pulse();
    } catch (error) {
      toast(error.message, true);
    }
    $("loading").hidden = true;
  };
  start();
})();
