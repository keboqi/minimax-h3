// The Remake page. The browser holds the editor; a long selection's windows live in their project,
// and scrubbing asks the server for nothing unless a source can't play here.
(() => {
  const API = "/remake/api";
  const POLL_MS = 1500;
  const PULSE_MS = 2500;
  const CHECK_DELAY_MS = 200;
  const DRAG_PX = 4;
  const MIN_SELECTION_S = 0.5;
  const PREFS = "h3-remake-prefs";
  const SESSION = {project: "h3-remake-project", window: "h3-remake-window"};
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
  const PLAY_SVG = '<svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor" aria-hidden="true"><path d="M3 1.5v11l9-5.5z"/></svg>';
  const PAUSE_SVG = '<svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor" aria-hidden="true"><path d="M3 2h3v10H3zM8 2h3v10H8z"/></svg>';
  const VOLUME_SVG = '<svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><path d="M2 5.2h2.6L8 2.5v9L4.6 8.8H2z" fill="currentColor"/><path d="M10 4.5a3.5 3.5 0 0 1 0 5M11.6 2.8a6 6 0 0 1 0 8.4" stroke-linecap="round"/></svg>';
  const MUTE_SVG = '<svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><path d="M2 5.2h2.6L8 2.5v9L4.6 8.8H2z" fill="currentColor"/><path d="M9.8 5l3 4M12.8 5l-3 4" stroke-linecap="round"/></svg>';

  const stored = {
    get(key) { try { return sessionStorage.getItem(key); } catch { return null; } },
    set(key, value) { try { sessionStorage.setItem(key, value); } catch { /* Private windows refuse it. */ } },
    remove(key) { try { sessionStorage.removeItem(key); } catch { /* Private windows refuse it. */ } },
  };

  const state = {
    boot: null, source: null, start: 0, end: 0, lastStart: 0, shortEdge: 480, maxWindow: 226, mix: 0,
    long: false, text: "", project: null, index: 0, frame: 0, mode: "Swap", pinned: true, smart: true, auto: true,
    slots: [null, null, null, null], useCarried: true, suggestion: "", previous: null, view: "source", wipe: 50,
    alignSlot: null, alignView: "Overlay", check: null, run: null, runSeen: -1, output: null, playing: false,
    muted: false, saveLabel: "All changes saved",
    menu: false, renaming: null, confirming: null, target: null, settings: null, quick: null,
  };
  const prefs = (() => { try { return JSON.parse(localStorage.getItem(PREFS)) || {}; } catch { return {}; } })();
  const savePrefs = () => {
    const kept = {mode: state.mode, outputEdge: state.shortEdge, maxWindow: state.maxWindow, smart: state.smart,
                  editWith: $("edit-model").value, muted: state.muted,
                  auto: state.auto, pinned: state.pinned, backend: $("writer-backend").value, model: $("writer-model").value};
    try { localStorage.setItem(PREFS, JSON.stringify(kept)); } catch { /* Private windows refuse it. */ }
  };

  const request = async (method, path, body, {form = false, keepalive = false} = {}) => {
    const options = {method, credentials: "same-origin", keepalive};
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
  const fileUrl = (path) => `${API}/file?path=${encodeURIComponent(path)}`;

  // In the tab the frame is taller than the window, so what's visible is only part of it.
  const visibleArea = () => {
    const frame = window.frameElement;
    if (!frame) return {top: 0, bottom: innerHeight};
    try {
      const box = frame.getBoundingClientRect();
      return {top: Math.max(0, -box.top), bottom: Math.min(innerHeight, window.parent.innerHeight - box.top)};
    } catch {
      return {top: 0, bottom: innerHeight};
    }
  };

  let toastTimer;
  const toast = (message, error = false) => {
    const node = $("toast");
    node.textContent = message;
    node.className = error ? "toast error" : "toast";
    node.style.bottom = `${Math.max(20, innerHeight - visibleArea().bottom + 20)}px`;
    node.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { node.hidden = true; }, error ? 8000 : 4000);
  };
  // Every action reports its own failure; the page stays usable.
  const safely = (work) => async (...args) => {
    try { return await work(...args); } catch (error) { toast(error.message, true); return undefined; }
  };

  // Timing -----------------------------------------------------------------
  const gpu = () => state.boot?.gpu !== false;
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
  const setSaveLabel = (label) => {
    state.saveLabel = label;
    const node = $("save-state");
    if (node) {
      node.textContent = label;
      node.hidden = !state.project;
    }
  };

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
  const saveWindow = (create = false, {keepalive = false} = {}) => {
    if (!state.project && !create) return Promise.resolve();
    setSaveLabel("Saving…");
    // Autosave belongs to the open project, even while a selection change is pending.
    const body = {...editor(), project: create ? null : state.project.key};
    saving = saving.catch(() => {}).then(async () => {
      try {
        const result = await post("/save", body, {keepalive});
        setSaveLabel("All changes saved");
        return result;
      } catch (error) {
        setSaveLabel("Couldn't save");
        throw error;
      }
    });
    return saving;
  };
  // A save can finish after the selection has moved on to another project.
  const adoptSaved = (project) => {
    if (!project || project.key !== state.project?.key) return false;
    state.project = project;
    return true;
  };
  // Reports a failure instead of raising it. Leaving never depended on saving, so a
  // save that keeps failing must not trap anyone in a project.
  const autosave = async () => {
    try {
      await saveWindow();
    } catch (error) {
      toast(`This window's latest edits weren't saved: ${error.message}`, true);
    }
  };
  const flushNow = () => {
    if (!state.project || !state.source) return;
    const body = {...editor(), project: state.project.key};
    post("/save", body, {keepalive: true}).catch(() => {});
  };
  const changed = () => { queueCheck(); savePrefs(); };

  const loadEditor = (window) => {
    const pictures = window.pictures ?? [];
    state.slots = [0, 1, 2, 3].map((index) => (pictures[index]
      ? {path: pictures[index], pin: window.pins?.[index] ?? null,
         original: window.originals?.[index], untouched: Boolean(window.originals?.[index])} : null));
    $("prompt").value = window.prompt ?? "";
    state.useCarried = window.use_carried ?? true;
    state.suggestion = "";
    state.previous = null;
    state.alignSlot = null;
    if (state.view === "align") state.view = "source";
  };

  const goToWindow = safely(async (index) => {
    if (index === state.index || !state.project) return;
    const key = state.project.key;
    adoptSaved(await saveWindow());
    if (state.project?.key !== key) return;
    state.index = index;
    stored.set(SESSION.window, String(index));
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
    queueFit();
  };
  const applySelection = (data) => {
    state.text = data.text;
    state.long = data.long;
    const previous = state.project?.key;
    state.project = data.project;
    if (data.project) stored.set(SESSION.project, data.project.key);
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

  const applyStageAspect = (width, height) => {
    if (width > 0 && height > 0) {
      $("stage").style.aspectRatio = `${width} / ${height}`;
    } else {
      $("stage").style.aspectRatio = "";
    }
  };

  const showSource = (source) => {
    state.source = source;
    applyStageAspect(source.width, source.height);
    $("audio-note").textContent = source.has_audio === false ? "No audio" : source.audio_note ? "Audio kept" : "";
    $("audio-note").title = source.audio_note ?? "";
    const video = $("source-video");
    delete video.dataset.failed;
    video.src = fileUrl(source.path);
    $("run-status").hidden = !running();
    showEditor(true);
  };

  const useSource = safely(async (source) => {
    stored.remove(SESSION.project);
    stored.remove(SESSION.window);
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

  const openProject = safely(async (key, {initialWindow = 0} = {}) => {
    await autosave();
    const data = await post(`/projects/${key}/load`);
    const chosen = data.selection;
    const targetWindow = Math.max(0, Math.min((data.windows?.length || 1) - 1, Number(initialWindow) || 0));
    Object.assign(state, {start: chosen.start, lastStart: chosen.start, end: chosen.end, shortEdge: chosen.short_edge,
                          mix: chosen.mix, maxWindow: chosen.max_window, mode: data.mode, smart: data.smart,
                          pinned: data.pinned, auto: data.auto_prompts, long: true, project: data, index: targetWindow,
                          frame: data.windows?.[targetWindow]?.own?.[0] ?? 0, output: data.output, view: "source",
                          menu: false, saveLabel: "All changes saved"});
    stored.set(SESSION.project, data.key);
    stored.set(SESSION.window, String(targetWindow));
    $("request").value = data.request;
    loadEditor(openWindow());
    showSource(data.source_info);
    await refreshSelection();
    if (targetWindow > 0 && state.project && targetWindow < state.project.windows.length) {
      state.index = targetWindow;
      state.frame = openWindow().own[0];
      loadEditor(openWindow());
      renderAll();
      queueCheck();
    }
    if (data.note) toast(data.note);
  });

  const closeProject = safely(async ({save = true} = {}) => {
    if (save) await autosave();
    stored.remove(SESSION.project);
    stored.remove(SESSION.window);
    state.source = null;
    state.project = null;
    state.menu = false;
    showEditor(false);
    await refreshProjects();
  });

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
    // Undoing passes ``before: false``: the picture it replaces isn't kept to undo again.
    const previous = before === false ? null
      : before ?? (slot?.path ? {path: slot.path, untouched: Boolean(slot.untouched)} : null);
    // A picture for an empty slot belongs at the playhead, like an extracted frame.
    state.slots[index] = {...slot, path, pin: slot?.pin ?? playheadPin(), untouched, sent: false, align: null, before: previous};
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
      if (adoptSaved(project)) {
        renderTimeline();
        renderAction();
      }
    }).catch(() => {});
  };

  const uploadPictureFile = safely(async (file, targetSlot = null) => {
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    const {path} = await request("POST", "/picture", form, {form: true});
    const slot = targetSlot ?? state.target;
    state.target = null;
    if (slot !== null && slot !== undefined && state.slots[slot]) {
      replacePicture(slot, path);
    } else {
      state.target = slot;
      addPicture(path);
    }
  });

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

  const useLatestFromLibrary = safely(async (slot) => {
    const {items} = await request("GET", "/library?limit=1");
    if (!items.length) throw new Error("No images in Media yet.");
    const {path} = await post("/picture/library", {source: items[0].path});
    if (slot !== null && slot !== undefined && state.slots[slot]) replacePicture(slot, path);
    else { state.target = slot; addPicture(path); }
  });

  const openLibrary = safely(async (slot = null) => {
    state.target = slot;
    const dialog = $("library");
    const grid = h("div", {className: "library-grid"});
    const search = async (query = "") => {
      const {items} = await request("GET", `/library?limit=60&query=${encodeURIComponent(query)}`);
      grid.replaceChildren(...items.map((item, idx) => h("button", {
        type: "button", className: idx === 0 && !query ? "latest" : "",
        title: idx === 0 && !query ? `${item.caption || "Latest image"} (Newest)` : item.caption,
        "aria-label": item.caption || "Image",
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
    const area = visibleArea();
    dialog.style.marginTop = `${Math.max(16, area.top + 24)}px`;
    dialog.showModal();
  });

  // Running ----------------------------------------------------------------
  const queue = () => (state.project ? state.project.windows.filter((window) => QUEUED.includes(window.state)) : []);
  const allTaken = () => Boolean(state.project) && state.project.windows.every((window) => window.take);

  let shownRendering = -1;
  const activeRenderingWindow = () => {
    if (!running()) return -1;
    if (state.run.project && state.run.project !== state.project?.key) return -1;
    const match = /^Window\s+(\d+)\s+of\s+\d+/i.exec(state.run?.status || "");
    return match ? Number(match[1]) - 1 : -1;
  };

  const startRun = safely(async () => {
    const {run} = await post("/generate", {...editor(), project: state.project?.key});
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
    } else if (state.project && activeRenderingWindow() !== shownRendering) {
      renderTimeline();
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
              await saving.catch(() => {});
              const data = await request("DELETE", `/projects/${row.key}`);
              state.boot.projects = data.projects;
              state.confirming = null;
              if (open) await closeProject({save: false});
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
    setSaveLabel(state.saveLabel);

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
    action.disabled = !gpu() || blocked || running();
    if (!gpu()) action.textContent = "Render in Studio";
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

  const renderMute = () => {
    const muteBtn = $("mute");
    if (!muteBtn) return;
    muteBtn.replaceChildren(svg(state.muted ? MUTE_SVG : VOLUME_SVG));
    muteBtn.setAttribute("aria-label", state.muted ? "Unmute" : "Mute");
    muteBtn.setAttribute("aria-pressed", String(state.muted));
    for (const video of [$("source-video"), $("result-video")]) {
      video.muted = state.muted || video !== activeVideo();
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
    renderMute();
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
      if (slot && slot.pin !== null && slot.pin !== undefined) {
        marks.push({slot: index + 1, slotIndex: index, at: open.render[0] + slot.pin, untouched: slot.untouched, window: state.index});
      }
    });
    windows().forEach((window, index) => {
      if (index === state.index) return;
      (window.pins ?? []).forEach((pin, slot) => {
        if (pin !== null) marks.push({slot: slot + 1, slotIndex: slot, at: window.render[0] + pin, window: index});
      });
    });
    return marks;
  };

  const showSelection = () => {
    const duration = state.source?.duration || 1;
    $("band").style.left = `${state.start / duration * 100}%`;
    $("band").style.width = `${Math.min(100, (state.end - state.start) / duration * 100)}%`;
    $("start-time").value = state.start;
    $("end-time").value = state.end;
  };

  const renderTimeline = () => {
    const last = Math.max(1, total() - 1);
    const frames = total();
    const start = videoStart();
    if (state.source) {
      const seconds = frames / fps();
      $("strip").style.backgroundImage = `url("${API}/strip?path=${encodeURIComponent(state.source.path)}&start=${start}&seconds=${seconds}")`;
      $("source-length").textContent = `${(state.source.duration || 1).toFixed(1)} s source`;
    }
    showSelection();
    $("short-edge").value = state.shortEdge;
    $("max-window").value = state.maxWindow;
    $("smart").checked = state.smart;
    $("auto-prompts").checked = state.auto;
    $("window-options").hidden = !state.long;
    const line = state.project ? state.project.summary : state.text;
    $("summary").replaceChildren(...bold(`${line}${state.project ? " · hatched frames are held from the window before" : ""}`));

    $("cuts").replaceChildren(...windows().slice(1).filter((window) => window.kind === "new shot")
      .map((window) => h("div", {className: "cut", title: `Scene cut at frame ${window.own[0]} (${clock(window.own[0])})`,
                                 style: `left:${window.own[0] / last * 100}%`})));
    const activeWin = activeRenderingWindow();
    shownRendering = activeWin;
    $("wins").replaceChildren(...windows().map((window, index) => {
      const span = window.own[1] - window.own[0];
      const label = state.project ? String(index + 1) : state.long ? "Not split yet" : "One render";
      const isRendering = index === activeWin;
      const holdTitle = window.hold
        ? `First ${window.hold} frames held from window ${index} (${clock(window.render[0])}–${clock(window.own[0])}) for a seamless join.`
        : "";
      const node = h("button", {type: "button", className: `win ${window.state}${isRendering ? " rendering" : ""}`,
                                "aria-pressed": String(index === state.index),
                                title: holdTitle || `${state.project ? `Window ${index + 1} (${clock(window.own[0])}–${clock(window.own[1])})` : "Selection"}`,
                                "aria-label": `${state.project ? `Window ${index + 1}, ${window.kind}, ` : ""}${window.state_name}`,
                                onclick: () => setFrame(window.own[0])},
        window.hold ? h("span", {className: "hold", style: `width:${window.hold / span * 100}%`}) : null,
        h("span", {className: "top"}, h("strong", {}, label), state.project ? h("span", {className: "kind"}, window.kind) : null),
        h("span", {className: "state"}, isRendering ? "Rendering…" : window.state_name),
        isRendering ? h("span", {className: "bar"}, h("i")) : null);
      node.style.left = `calc(${window.own[0] / frames * 100}% + 2px)`;
      node.style.width = `calc(${span / frames * 100}% - 4px)`;
      return node;
    }));
    $("marks").replaceChildren(...pinMarks().map((mark) => {
      const isCurrentWindow = mark.window === state.index;
      const node = button(String(mark.slot), () => setFrame(mark.at), `pinmark${mark.untouched ? " unedited" : ""}`,
        {title: `Picture ${mark.slot} at frame ${mark.at} (${clock(mark.at)})${isCurrentWindow ? " · drag to move pin" : ""}`,
         "aria-label": `Picture ${mark.slot}`});
      node.style.left = `${Math.min(100, mark.at / last * 100)}%`;
      node.dataset.frame = mark.at;
      if (isCurrentWindow) {
        node.addEventListener("pointerdown", (event) => {
          if (event.button !== 0) return;
          event.stopPropagation();
          const track = $("marks");
          const rect = track.getBoundingClientRect();
          const open = openWindow();
          const windowSpan = open.render[1] - open.render[0];
          const startX = event.clientX;
          const grab = startX - rect.left - mark.at / last * rect.width;
          let moved = false;
          const onMove = (moveEvent) => {
            // A shaky click is still a click, and seeks.
            if (!moved && Math.abs(moveEvent.clientX - startX) < DRAG_PX) return;
            moved = true;
            const ratio = Math.max(0, Math.min(1, (moveEvent.clientX - grab - rect.left) / Math.max(1, rect.width)));
            const globalFrame = Math.round(ratio * last);
            const newPin = Math.max(0, Math.min(windowSpan - 1, globalFrame - open.render[0]));
            const slot = state.slots[mark.slotIndex];
            if (slot && slot.pin !== newPin) {
              slot.pin = newPin;
              slot.align = null;
              const newAt = open.render[0] + newPin;
              node.style.left = `${Math.min(100, newAt / last * 100)}%`;
              node.dataset.frame = newAt;
              setFrame(newAt, {follow: false});
            }
          };
          const onUp = () => {
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", onUp);
            if (moved) {
              afterPictures();
              measure(mark.slotIndex);
            }
          };
          window.addEventListener("pointermove", onMove);
          window.addEventListener("pointerup", onUp);
        });
      }
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

  const bindSlotDrop = (node, slotIndex) => {
    node.addEventListener("dragover", (event) => {
      event.preventDefault();
      node.classList.add("over");
    });
    node.addEventListener("dragleave", () => node.classList.remove("over"));
    node.addEventListener("drop", (event) => {
      event.preventDefault();
      node.classList.remove("over");
      const file = event.dataTransfer?.files?.[0];
      if (file) uploadPictureFile(file, slotIndex);
    });
  };

  const pictureRow = (slot, index) => {
    const open = openWindow();
    const pinTime = slot.pin !== null && slot.pin !== undefined ? clock(open.render[0] + Number(slot.pin)) : "";
    const pinField = state.pinned ? h("span", {className: "inline tightrow"},
      h("label", {className: "inline"}, h("span", {className: "small muted"}, "Frame"),
        h("input", {className: "field tight mono", type: "number", min: 0, value: slot.pin ?? "",
                    "aria-label": `Picture ${index + 1} frame`,
                    onchange: (event) => {
                      slot.pin = event.target.value === "" ? null : Number(event.target.value);
                      slot.align = null;
                      afterPictures();
                      measure(index);
                    }})),
      pinTime ? h("span", {className: "small muted mono"}, pinTime) : null,
      button("Pin here", () => {
        const windowSpan = open.render[1] - open.render[0];
        slot.pin = Math.max(0, Math.min(windowSpan - 1, playheadPin()));
        slot.align = null;
        afterPictures();
        measure(index);
      }, "btn compact", {title: `Set Picture ${index + 1}'s frame pin to the playhead (frame ${playheadPin()})`})) : null;
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

    const img = h("img", {src: fileUrl(slot.path), alt: `Picture ${index + 1}`});
    const badge = slot.before ? h("span", {className: "peek-badge"}, "Hold: before") : null;
    const thumbWrap = h("div", {
      className: "thumb-wrap",
      title: slot.before
        ? "Click to seek playhead to this frame · Hold to peek at original before edit"
        : "Click to seek playhead to this frame · Drop an image here to replace",
      onclick: () => {
        if (slot.pin !== null && slot.pin !== undefined) setFrame(open.render[0] + Number(slot.pin), {follow: false});
      },
      onpointerdown: () => {
        if (slot.before) {
          img.src = fileUrl(slot.before.path);
          if (badge) badge.textContent = "Original";
        }
      },
      onpointerup: () => {
        if (slot.before) {
          img.src = fileUrl(slot.path);
          if (badge) badge.textContent = "Hold: before";
        }
      },
      onpointerleave: () => {
        if (slot.before) {
          img.src = fileUrl(slot.path);
          if (badge) badge.textContent = "Hold: before";
        }
      },
    }, img, badge);

    const card = h("div", {className: `pic${state.view === "align" && state.alignSlot === index ? " on" : ""}`},
      thumbWrap,
      h("div", {className: "body"},
        h("div", {className: "row tightrow"},
          h("strong", {className: "small"}, `Picture ${index + 1}`), h("span", {className: status[1]}, status[0]),
          h("span", {className: "grow"}),
          button(svg(CROSS), () => {
            state.slots[index] = null;
            if (state.alignSlot === index) { state.alignSlot = null; state.view = "source"; renderViewer(); }
            afterPictures();
          }, "btn quiet mini", {"aria-label": `Remove picture ${index + 1}`})),
        gpu() ? quickRow : null,
        editing ? h("p", {className: "small muted", id: "quick-status", role: "status"}, state.quick.status) : null,
        h("div", {className: "row tightrow"}, pinField,
          slot.before ? button("Undo edit", () => replacePicture(index, slot.before.path, {before: false, untouched: slot.before.untouched}))
            : null,
          // Still offered once sent: the frame can go to another model.
          gpu() ? button("Edit in Image", () => sendToImage(index)) : null,
          h("a", {className: "btn", href: fileUrl(slot.path), download: `frame-${index + 1}.png`}, "Download frame"),
          button("Upload edited frame", () => { state.target = index; $("picture-input").click(); }),
          slot.sent ? button("Use latest from Media", () => useLatestFromLibrary(index), "btn primary") : null,
          slot.sent ? button("Choose the edited image", () => openLibrary(index), "btn") : null,
          slot.untouched ? (slot.sent ? null : button("Replace", () => openLibrary(index)))
            : button("Check alignment", () => {
              state.alignSlot = index;
              state.view = "align";
              setFrame(openWindow().render[0] + (slot.pin ?? 0), {follow: false});
              renderViewer();
              renderPictures();
              measure(index, state.alignView);
            }))));
    bindSlotDrop(card, index);
    return card;
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
            autosave();
          }}), "Use the carried frame"),
          button("Copy to a slot", safely(async () => {
            const data = await post("/window/carried", {selection: selection(), window: state.index});
            state.useCarried = false;
            addPicture(data.path, data.pin);
          }), "btn", {disabled: !ready || emptySlot() < 0}))));
  };

  const renderPictures = () => {
    const active = document.activeElement;
    const focusLabel = active && $("pictures").contains(active) ? active.getAttribute("aria-label") : null;
    const selectionStart = focusLabel && typeof active.selectionStart === "number" ? active.selectionStart : null;
    const selectionEnd = focusLabel && typeof active.selectionEnd === "number" ? active.selectionEnd : null;

    const rows = [];
    if (state.project && state.index > 0 && state.smart) rows.push(carriedRow());
    state.slots.forEach((slot, index) => { if (slot) rows.push(pictureRow(slot, index)); });
    const empty = emptySlot();
    if (empty >= 0) {
      const emptyNode = h("div", {className: "empty-slot"},
        h("span", {}, `Picture ${empty + 1} is empty · drop or paste an image`),
        h("span", {className: "row tightrow"},
          button("Upload", () => { state.target = empty; $("picture-input").click(); }),
          button("From Media", () => openLibrary())));
      bindSlotDrop(emptyNode, empty);
      rows.push(emptyNode);
    }
    $("pictures").replaceChildren(...rows);
    if (focusLabel) {
      const restored = [...$("pictures").querySelectorAll("[aria-label]")].find((el) => el.getAttribute("aria-label") === focusLabel);
      if (restored && !restored.disabled) {
        restored.focus();
        if (selectionStart !== null && typeof restored.setSelectionRange === "function") {
          try { restored.setSelectionRange(selectionStart, selectionEnd); } catch { /* number inputs don't support setSelectionRange */ }
        }
      }
    }
    const count = state.slots.filter(Boolean).length;
    $("pic-count").textContent = `${count} of 4`;
    $("pinned").checked = state.pinned;
    $("extract").disabled = empty < 0;
    $("extract-hint").textContent = empty < 0 ? "All four slots are full" : `Fills Picture ${empty + 1}, pinned here`;
  };

  const renderPromptMeta = () => {
    const text = $("prompt").value;
    const open = openWindow();
    const present = state.boot.sections.filter((name) => new RegExp(`^\\s*${name}\\s*:`, "m").test(text));
    const missing = state.boot.sections.filter((name) => !present.includes(name));
    const count = present.length;
    const written = open.prompt_auto && text === open.prompt;
    $("origin").textContent = !text.trim() ? "Not written yet" : written ? "Written, replaced on render" : "Typed by you, stays";
    $("origin").className = text.trim() && !written ? "chip good" : "chip";
    $("sections").textContent = `${count} of 6 sections`;
    $("sections").className = count === 6 ? "chip good" : "chip";
    $("sections").title = missing.length ? `Missing sections: ${missing.join(", ")}` : "All 6 Viggle Remake prompt sections present";
    $("insert-template").hidden = count === 6;
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
    $("regenerate").hidden = !gpu() || !state.project;
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
      video.muted = state.muted || video !== activeVideo();
      if (playing) video.play().catch(() => {});
      else video.pause();
    }
    $("play").setAttribute("aria-label", playing ? "Pause" : "Play");
    $("play").replaceChildren(svg(playing ? PAUSE_SVG : PLAY_SVG));
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

  // Pins and the playhead count from Start, so each stays on its source frame.
  const followStart = () => {
    const moved = Math.round((state.start - state.lastStart) * fps());
    for (const slot of state.slots) {
      if (slot && slot.pin !== null && slot.pin !== undefined) slot.pin = slot.pin >= moved ? slot.pin - moved : null;
    }
    state.frame = Math.max(0, state.frame - moved);
    state.lastStart = state.start;
  };

  const selectionLocked = () => {
    if (running()) toast("Wait for the render to finish before changing the selection.", true);
    return running();
  };
  // A new selection leaves the open project's windows, so a single key or click asks first.
  const leavesProject = () => Boolean(state.project)
    && !confirm("Change the selection? This leaves the open project, which stays saved in Projects.");
  const selectionRefused = (ask) => selectionLocked() || (ask && leavesProject());

  const applyStartChange = (newStart, {ask = false} = {}) => {
    if (selectionRefused(ask)) { showSelection(); return; }
    const start = Math.max(0, Math.round(Number(newStart) * 100) / 100 || 0);
    state.start = Math.min(start, Math.max(0, state.end - MIN_SELECTION_S));
    followStart();
    refreshSelection();
  };

  const applyEndChange = (newEnd, {ask = false} = {}) => {
    if (selectionRefused(ask)) { showSelection(); return; }
    const duration = state.source?.duration || 9999;
    state.end = Math.min(duration, Math.max(state.start + MIN_SELECTION_S, Math.round((Number(newEnd) || 0) * 100) / 100));
    refreshSelection();
  };

  // Wiring -----------------------------------------------------------------
  const bind = () => {
    $("choose-video").addEventListener("click", () => $("video-input").click());
    $("video-input").addEventListener("change", (event) => { upload(event.target.files[0]); event.target.value = ""; });
    $("picture-input").addEventListener("change", (event) => {
      const file = event.target.files[0];
      event.target.value = "";
      uploadPictureFile(file);
    });
    const drop = $("drop");
    drop.addEventListener("dragover", (event) => { event.preventDefault(); drop.classList.add("over"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("over"));
    drop.addEventListener("drop", (event) => {
      event.preventDefault();
      drop.classList.remove("over");
      upload(event.dataTransfer.files[0]);
    });

    document.addEventListener("paste", (event) => {
      if (!state.source) return;
      const active = document.activeElement;
      if (active && (active.tagName === "INPUT" || active.tagName === "TEXTAREA")) return;
      const items = [...(event.clipboardData?.items || [])];
      const imageItem = items.find((item) => item.type.startsWith("image/"));
      if (imageItem) {
        event.preventDefault();
        uploadPictureFile(imageItem.getAsFile());
      }
    });

    document.addEventListener("click", (event) => {
      if (state.menu && event.target.isConnected && !event.target.closest(".project-menu")) {
        state.menu = false;
        renderMenu();
      }
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && state.menu) {
        state.menu = false;
        renderMenu();
        return;
      }
      if (!state.source || $("editor").hidden || $("library").open) return;
      if (event.ctrlKey || event.metaKey || event.altKey) return;
      const tag = document.activeElement?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (event.repeat && !event.key.startsWith("Arrow")) return;
      // Space belongs to a focused control; playback has the rest of the page.
      if (event.key === " " && event.target.closest("button, a, summary")) return;
      if (event.key === " ") {
        event.preventDefault();
        setPlaying(!state.playing);
      } else if (event.key === "ArrowLeft") {
        event.preventDefault();
        if (state.playing) setPlaying(false);
        setFrame(state.frame - (event.shiftKey ? 10 : 1));
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        if (state.playing) setPlaying(false);
        setFrame(state.frame + (event.shiftKey ? 10 : 1));
      } else if (event.key === "e" || event.key === "E") {
        if (!$("extract").disabled) {
          event.preventDefault();
          extractFrame();
        }
      } else if (event.key === "m" || event.key === "M") {
        event.preventDefault();
        state.muted = !state.muted;
        renderMute();
        savePrefs();
      } else if (event.key === "[") {
        event.preventDefault();
        applyStartChange(videoStart() + state.frame / fps(), {ask: true});
      } else if (event.key === "]") {
        event.preventDefault();
        applyEndChange(videoStart() + (state.frame + 1) / fps(), {ask: true});
      } else if (event.key === "1") {
        state.view = "source";
        renderViewer();
      } else if (event.key === "2") {
        state.view = "result";
        renderViewer();
      } else if (event.key === "3") {
        state.view = "compare";
        renderViewer();
      }
    });

    $("project-button").addEventListener("click", () => { state.menu = !state.menu; renderMenu(); });
    $("short-edge").addEventListener("change", (event) => { state.shortEdge = Number(event.target.value); savePrefs(); refreshSelection(); });
    $("max-window").addEventListener("change", (event) => { state.maxWindow = Number(event.target.value); savePrefs(); refreshSelection(); });
    $("start-time").addEventListener("change", (event) => applyStartChange(event.target.value));
    $("end-time").addEventListener("change", (event) => applyEndChange(event.target.value));
    $("set-start").addEventListener("click", () => applyStartChange(videoStart() + state.frame / fps(), {ask: true}));
    $("set-end").addEventListener("click", () => applyEndChange(videoStart() + (state.frame + 1) / fps(), {ask: true}));

    $("overview").addEventListener("pointerdown", (event) => {
      if (!state.source || event.button !== 0 || selectionLocked()) return;
      const bar = $("overview");
      const rect = bar.getBoundingClientRect();
      const duration = state.source.duration || 1;
      const edge = event.target.dataset?.edge;
      const initialStart = state.start;
      const initialEnd = state.end;
      const span = Math.max(0.5, initialEnd - initialStart);
      const startX = event.clientX;
      let dragging = false;

      if (!edge && event.target !== $("band")) {
        const clickTime = Math.max(0, Math.min(duration - span, (event.clientX - rect.left) / Math.max(1, rect.width) * duration - span / 2));
        state.start = Math.round(clickTime * 10) / 10;
        state.end = Math.min(duration, Math.round((state.start + span) * 10) / 10);
        showSelection();
      }

      const onMove = (moveEvent) => {
        if (!dragging && Math.abs(moveEvent.clientX - startX) < DRAG_PX) return;
        dragging = true;
        const deltaSec = (moveEvent.clientX - startX) / Math.max(1, rect.width) * duration;
        if (edge === "start") {
          state.start = Math.max(0, Math.min(state.end - 0.5, Math.round((initialStart + deltaSec) * 10) / 10));
        } else if (edge === "end") {
          state.end = Math.min(duration, Math.max(state.start + 0.5, Math.round((initialEnd + deltaSec) * 10) / 10));
        } else {
          const nextStart = Math.max(0, Math.min(duration - span, Math.round((initialStart + deltaSec) * 10) / 10));
          state.start = nextStart;
          state.end = Math.min(duration, Math.round((nextStart + span) * 10) / 10);
        }
        showSelection();
      };

      const onUp = () => {
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
        if (state.start === initialStart && state.end === initialEnd) return;
        if (leavesProject()) {
          state.start = initialStart;
          state.end = initialEnd;
          showSelection();
          return;
        }
        followStart();
        refreshSelection();
      };

      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
    });

    $("smart").addEventListener("change", (event) => { state.smart = event.target.checked; renderPictures(); changed(); });
    $("auto-prompts").addEventListener("change", (event) => { state.auto = event.target.checked; changed(); });
    $("pinned").addEventListener("change", (event) => { state.pinned = event.target.checked; renderPictures(); changed(); });
    $("scrub").addEventListener("input", (event) => setFrame(Number(event.target.value)));
    $("wipe").addEventListener("input", (event) => { state.wipe = Number(event.target.value); renderViewer(); });
    $("back").addEventListener("click", () => setFrame(state.frame - 1));
    $("next").addEventListener("click", () => setFrame(state.frame + 1));
    $("play").addEventListener("click", () => setPlaying(!state.playing));
    $("mute").addEventListener("click", () => {
      state.muted = !state.muted;
      renderMute();
      savePrefs();
    });
    $("extract").addEventListener("click", extractFrame);
    $("save-project").addEventListener("click", safely(async () => {
      const control = $("save-project");
      control.disabled = true;
      try {
        state.project = await saveWindow(true);
        stored.set(SESSION.project, state.project.key);
        await refreshProjects();
        renderAll();
        queueCheck();
        toast(gpu() ? "Project saved." : "Project saved. Reopen it in Studio to render.");
      } finally { control.disabled = false; }
    }));
    $("primary").addEventListener("click", primary);
    $("regenerate").addEventListener("click", regenerate);
    $("combine").addEventListener("click", combine);
    $("split").addEventListener("click", safely(async () => {
      state.project = await post("/split", {selection: selection(), video: state.source.path});
      stored.set(SESSION.project, state.project.key);
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
    $("prompt").addEventListener("change", autosave);
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
      autosave();
    });
    $("keep").addEventListener("click", () => { state.suggestion = ""; renderPromptMeta(); });
    $("undo").addEventListener("click", () => {
      $("prompt").value = state.previous;
      state.previous = null;
      renderPromptMeta();
      queueCheck();
      autosave();
    });
    $("insert-template").addEventListener("click", () => {
      const current = $("prompt").value;
      const missing = state.boot.sections.filter((name) => !new RegExp(`^\\s*${name}\\s*:`, "m").test(current));
      if (!missing.length) return;
      state.previous = current;
      const block = missing.map((name) => `${name}:\n`).join("\n");
      $("prompt").value = current.trim() ? `${current.trimEnd()}\n\n${block}` : block;
      renderPromptMeta();
      queueCheck();
      autosave();
      $("prompt").focus();
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

  // In the tab, the frame grows or shrinks to fit, so the app's page is the only thing that scrolls.
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
    window.addEventListener("beforeunload", flushNow);
    new MutationObserver(queueFit).observe(document.body, {subtree: true, childList: true, attributes: true, attributeFilter: ["hidden"]});
    document.addEventListener("toggle", queueFit, true);
    window.addEventListener("resize", queueFit);
    try {
      state.boot = await request("GET", "/boot");
      $("workshop-note").hidden = gpu();
      $("edit-model").closest("label").hidden = !gpu();
      $("open-settings").hidden = !gpu();
      if (!gpu()) $("no-projects").textContent = "Save any selection as a project, then reopen it in Studio to render.";
      state.shortEdge = prefs.outputEdge ?? state.boot.short_edge;
      state.maxWindow = prefs.maxWindow ?? state.boot.max_window;
      Object.assign(state, {mode: prefs.mode ?? "Swap", smart: prefs.smart ?? true, auto: prefs.auto ?? true,
                            pinned: prefs.pinned ?? true, muted: Boolean(prefs.muted)});
      $("short-edge").replaceChildren(h("option", {value: 0}, "Auto, match source"),
        ...state.boot.short_edges.map((edge) => h("option", {value: edge}, String(edge))));
      $("edit-model").replaceChildren(...state.boot.edit_models.map((model) => h("option", {value: model.id}, model.label)));
      // Not `editModel`: that was saved for everyone, so it would hold them on the old default.
      if (state.boot.edit_models.some((model) => model.id === prefs.editWith)) $("edit-model").value = prefs.editWith;
      $("max-window").replaceChildren(...state.boot.lengths.map((length) =>
        h("option", {value: length}, `${(length / fps()).toFixed(1)} s (${length} frames)`)));
      if (prefs.backend) $("writer-backend").value = prefs.backend;
      fillWriterModels();
      renderMute();
      bind();
      renderStart();
      const savedKey = stored.get(SESSION.project);
      if (savedKey && state.boot.projects.some((row) => row.key === savedKey)) {
        await openProject(savedKey, {initialWindow: Number(stored.get(SESSION.window) || 0)});
      }
      if (!state.source) {
        stored.remove(SESSION.project);
        stored.remove(SESSION.window);
        showEditor(false);
      }
      if (state.boot.run) watchRun(state.boot.run);
      pulse();
    } catch (error) {
      toast(error.message, true);
    }
    $("loading").hidden = true;
  };
  start();
})();
