import {bindFullscreen} from "./fullscreen.js";
import {
  API, POLL_MS, PULSE_MS, CHECK_DELAY_MS, DRAG_PX, MIN_SELECTION_S, PREFS, SESSION, BRIDGE, QUEUED,
  CHECK, CROSS, CHEVRON, PLAY_SVG, PAUSE_SVG, VOLUME_SVG, MUTE_SVG,
  svg, stored, loadPrefs, formatClock, fileUrl,
} from "./helpers.js";

// The Remake page. The browser holds the editor; a long selection's windows live in their project,
// and scrubbing asks the server for nothing unless a source can't play here.
(() => {
  const {$, h, visibleArea, createToast, createClient} = globalThis.H3PageCommon;
  const {request, post} = createClient(API);
  const {toast, guard: safely} = createToast();
  const button = (label, onclick, className = "btn", extra = {}) =>
    h("button", {type: "button", className, onclick, ...extra}, label);

  const state = {
    boot: null, source: null, start: 0, end: 0, lastStart: 0, shortEdge: 480, maxWindow: 226, mix: 0,
    long: false, text: "", project: null, index: 0, frame: 0, mode: "Swap", pinned: true, smart: true, auto: true,
    keepRest: false, outputSpliced: false, renderFrames: null,
    slots: [null, null, null, null], useCarried: true, suggestion: "", previous: null, view: "source", wipe: 50,
    alignSlot: null, alignView: "Overlay", check: null, run: null, runSeen: -1, output: null, playing: false,
    muted: false, saveLabel: "All changes saved",
    menu: false, renaming: null, confirming: null, confirmingClear: null, target: null, settings: null, quick: null,
  };
  const prefs = loadPrefs();
  const savePrefs = () => {
    const kept = {mode: state.mode, outputEdge: state.shortEdge, maxWindow: state.maxWindow, smart: state.smart,
                  editWith: $("edit-model").value, muted: state.muted,
                  auto: state.auto, pinned: state.pinned, backend: $("writer-backend").value, model: $("writer-model").value};
    try { localStorage.setItem(PREFS, JSON.stringify(kept)); } catch { /* Private windows refuse it. */ }
  };

  // Timing -----------------------------------------------------------------
  const gpu = () => state.boot?.gpu !== false;
  const fps = () => state.boot.fps;
  const clock = (frames) => formatClock(frames, fps());
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
            gemini_key: "", openai_key: openai ? $("writer-key").value : "",
            openai_url: $("writer-url").value};
  };
  const originals = () => Object.fromEntries(state.slots.flatMap((slot, index) =>
    (slot?.original && slot.untouched ? [[index + 1, slot.original]] : [])));
  const editFields = () => ({
    video: state.source?.path, selection: selection(), prompt: $("prompt").value, request: $("request").value,
    pinned: state.pinned, smart: state.smart, auto_prompts: state.auto, keep_rest: state.keepRest,
    window: state.index, mode: state.mode, originals: originals(), use_carried: state.useCarried,
  });

  const reuseBreakdown = () => {
    if (!state.source || !state.keepRest) return null;
    const duration = Number(state.source.duration || 0);
    const startFrame = Math.max(0, Math.round(videoStart() * fps()));
    const remadeFrames = state.project?.frames ?? state.renderFrames;
    if (!remadeFrames) return null;
    const remadeStart = startFrame / fps();
    const remadeEnd = (startFrame + remadeFrames) / fps();
    const hasBefore = startFrame > 0;
    const hasAfter = startFrame + remadeFrames < Math.round(duration * fps());
    return {
      duration, startFrame, remadeFrames, remadeStart, remadeEnd, hasBefore, hasAfter,
      reusesAny: hasBefore || hasAfter,
    };
  };
  const editor = () => ({...editFields(), slots: state.slots, writer: writer()});

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
  let checkRevision = 0;
  const queueCheck = () => {
    const revision = ++checkRevision;
    clearTimeout(checkTimer);
    checkTimer = setTimeout(safely(async () => {
      if (!state.source) return;
      const check = await post("/check", editor());
      if (revision !== checkRevision || !state.source) return;
      state.check = check;
      renderAction();
      renderNextRun();
    }), CHECK_DELAY_MS);
  };
  // Saves run one at a time: moving quickly across windows must keep them in order.
  let saving = Promise.resolve();
  let savedEditor;
  let pendingSaves = 0;
  const saveSession = crypto.getRandomValues(new Uint32Array(4)).join("-");
  let saveRevision = 0;
  const saveBody = (project) => ({
    ...editFields(), slots: state.slots.map((slot) => slot && {path: slot.path, pin: slot.pin}), project,
  });
  const saveWindow = (create = false) => {
    if (!state.project && !create) return Promise.resolve();
    // Autosave belongs to the open project, even while a selection change is pending.
    const body = saveBody(create ? null : state.project.key);
    const snapshot = JSON.stringify(body);
    const saveOrder = [saveSession, ++saveRevision];
    pendingSaves++;
    saving = saving.catch(() => {}).then(async () => {
      if (!create && snapshot === savedEditor) return;
      setSaveLabel("Saving…");
      try {
        const result = await post("/save", {...body, save_order: saveOrder});
        savedEditor = JSON.stringify({...body, project: result.key});
        setSaveLabel("All changes saved");
        return result;
      } catch (error) {
        setSaveLabel("Couldn't save");
        throw error;
      }
    }).finally(() => { pendingSaves--; });
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
      if (adoptSaved(await saveWindow())) {
        renderTimeline();
        renderAction();
      }
    } catch (error) {
      toast(`This window's latest edits weren't saved: ${error.message}`, true);
    }
  };
  const flushNow = () => {
    if (!state.project || !state.source) return;
    const body = saveBody(state.project.key);
    // A pending write can still replace the last acknowledged snapshot during unload.
    if (!pendingSaves && JSON.stringify(body) === savedEditor) return;
    post("/save", {...body, save_order: [saveSession, ++saveRevision]}, {keepalive: true}).catch(() => {});
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
    state.renderFrames = data.frames;
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
  let selectionRevision = 0;
  const refreshSelection = safely(async () => {
    if (!state.source) return;
    const revision = ++selectionRevision;
    const data = await post("/selection", {selection: selection()});
    if (revision !== selectionRevision) return;
    applySelection(data);
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
    const result = $("result-video");
    if (result.dataset.url) {
      delete result.dataset.url;
      result.removeAttribute("src");
      result.load();
    }
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
    state.keepRest = false;
    state.outputSpliced = false;
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
                          pinned: data.pinned, auto: data.auto_prompts, keepRest: Boolean(data.keep_rest),
                          long: data.long, project: data, index: targetWindow,
                          frame: data.windows?.[targetWindow]?.own?.[0] ?? 0, output: data.output,
                          outputSpliced: Boolean(data.output_spliced), view: "source",
                          menu: false, check: null, saveLabel: "All changes saved"});
    stored.set(SESSION.project, data.key);
    stored.set(SESSION.window, String(targetWindow));
    $("request").value = data.request;
    loadEditor(openWindow());
    showSource(data.source_info);
    savedEditor = JSON.stringify(saveBody(data.key));
    renderAll();
    queueCheck();
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
    autosave();
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
    state.quick = {index, status: "Starting image edit…"};
    renderPictures();
    try {
      let {run} = await post("/quick-edit", {picture: slot.path, prompt, engine: gpu() && $("edit-model").value !== "gpt_image" ? "qwen" : "gpt_image"});
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
    const sent = await post("/send-to-image", {path: slot.path, model});
    const bridge = window.parent === window ? null : window.parent.document.getElementById(BRIDGE.edit(model));
    if (!sent.url && !bridge) throw new Error("Open Remake in the app to edit in Image.");
    slot.sent = true;
    renderPictures();
    if (sent.url) window.open(sent.url, "_blank");
    else bridge.click();
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
    const {output, status, spliced} = await post("/combine", {selection: selection(), keep_rest: state.keepRest});
    $("run-status").textContent = status;
    state.project = (await request("GET", `/projects/${state.project.key}`));
    state.output = output;
    state.outputSpliced = Boolean(spliced ?? state.project?.output_spliced);
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
    if (project.output) {
      state.output = project.output;
      state.outputSpliced = Boolean(project.output_spliced);
    }
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
    renderRunBar();
    if (run && run.revision !== state.runSeen) {
      state.runSeen = run.revision;
      syncProject().then(() => {
        if (run.result_url && !state.project) {
          state.output = run.result_url;
          state.outputSpliced = Boolean(run.spliced);
        }
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

  let pulseInFlight = false;
  const pulseNow = async () => {
    if (pulseInFlight) return;
    pulseInFlight = true;
    try {
      const frame = window.frameElement;
      if (!document.hidden && (!frame || frame.offsetParent !== null)) {
        const data = await request("GET", `/pulse?collect=${Boolean(state.source)}&accept_source=${!running()}`);
        if (data.incoming_source) {
          await autosave();
          await useSource(data.incoming_source);
          toast(`Loaded "${data.incoming_source.name}" from Storyboard.`);
          return;
        }
        for (const {slot, path} of data.inbox) replacePicture(slot - 1, path);
        if (data.settings !== state.settings) {
          state.settings = data.settings;
          queueCheck();
        }
      }
    } catch { /* The next pulse tries again. */ }
    finally { pulseInFlight = false; }
  };
  const pulse = async () => {
    await pulseNow();
    setTimeout(pulse, PULSE_MS);
  };

  // Rendering --------------------------------------------------------------
  const pressed = (list, current, pick) => list.map(([value, label]) =>
    h("button", {type: "button", "aria-pressed": String(value === current), onclick: () => pick(value)}, label));

  const bold = (text) => String(text).split("**").map((part, index) => (index % 2 ? h("strong", {}, part) : part));

  const projectRow = (row, onclick) => h("button", {type: "button", className: "project-row", onclick},
    h("span", {className: "thumb"}, row.thumb ? h("img", {src: fileUrl(row.thumb), alt: ""}) : null),
    h("span", {className: "about"},
      h("strong", {}, row.name),
      row.states?.length ? h("span", {className: "states", "aria-hidden": "true"},
        row.states.map((value) => h("i", {className: value}))) : null,
      h("span", {className: "small muted"}, row.meta)));

  const MODES = [["Swap", "Replace a character or subject. Their moves and lip sync stay."],
                 ["Edit", "Change objects, background, weather or light around the action."]];
  const renderStart = () => {
    $("no-projects").hidden = state.boot.projects.length > 0;
    $("start-projects").replaceChildren(...state.boot.projects.map((row) => projectRow(row, () => openProject(row.key))));
    $("start-mode").replaceChildren(...MODES.map(([mode, line]) => h("button", {
      type: "button", className: "mode-card", "aria-pressed": String(state.mode === mode),
      onclick: () => { state.mode = mode; savePrefs(); renderStart(); },
    }, h("strong", {}, mode), h("span", {className: "small muted"}, line))));
  };

  const renderMenu = () => {
    const menu = $("menu");
    menu.hidden = !state.menu;
    $("project-button").setAttribute("aria-expanded", String(state.menu));
    if (!state.menu) return;
    const rows = state.boot.projects.map((row) => {
      const open = row.key === state.project?.key;
      const renaming = state.renaming === row.key;
      const confirmingClear = state.confirmingClear === row.key;
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
          : confirmingClear
            ? h("div", {className: "row"},
              h("span", {className: "small muted grow"}, "Removes unused history takes. Active window takes and Media outputs stay."),
              button("Clear history takes", safely(async () => {
                await saving.catch(() => {});
                const data = await post(`/projects/${row.key}/clear-history-takes`, {});
                state.boot.projects = data.projects;
                state.confirmingClear = null;
                renderMenu();
                toast(data.message || "Cleared history takes.");
              })),
              button("Keep them", () => { state.confirmingClear = null; renderMenu(); }, "btn quiet"))
            : h("div", {className: "row"},
              button(renaming ? "Done" : "Rename", () => { state.renaming = renaming ? null : row.key; state.confirmingClear = null; renderMenu(); }, "btn quiet"),
              button("Clear takes", () => { state.confirmingClear = row.key; state.confirming = null; state.renaming = null; renderMenu(); }, "btn quiet"),
              button("Delete", () => { state.confirming = row.key; state.confirmingClear = null; state.renaming = null; renderMenu(); }, "btn quiet")));
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
      autosave();
    }));
    MODES.forEach(([, line], index) => { $("mode").children[index].title = line; });

    const check = state.check;
    const step = check?.step;
    const order = ["source", "frames", "prompt"];
    const reached = step ? order.indexOf(step) : order.length;
    const filled = state.slots.filter(Boolean);
    const edited = filled.filter((slot) => !slot.untouched).length;
    const sections = presentSections($("prompt").value).length;
    const names = ["Source", "Frames", "Prompt"];
    const labels = [names[0],
                    filled.length ? `${names[1]} · ${edited} of ${filled.length} edited` : names[1],
                    `${names[2]} · ${sections} of 6`];
    $("steps").replaceChildren(...order.map((key, index) => {
      const done = index < reached;
      const now = index === reached;
      return h("button", {type: "button", className: done ? "step done" : now ? "step now" : "step",
                          title: now && check?.message ? `${check.message} Click to go there.` : `Go to ${names[index]}`,
                          "aria-label": `${labels[index]} step, ${done ? "done" : now ? "needed next" : "to do"}`,
                          onclick: () => jumpTo(key)},
        done ? svg(CHECK) : h("span", {className: "dot"}), labels[index]);
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
    $("primary-detail").textContent = primaryDetail(waiting);
  };

  // What the primary button will render, in seconds of video.
  const primaryDetail = (waiting) => {
    if (running() || !state.source || !gpu()) return "";
    const seconds = (frames) => `${(frames / fps()).toFixed(1)} s`;
    const reuse = reuseBreakdown();
    if (state.project) {
      if (!waiting) return "";
      const frames = queue().reduce((sum, window) => sum + window.own[1] - window.own[0], 0);
      return `${seconds(frames)} of video`;
    }
    if (reuse?.reusesAny) {
      return `${seconds(reuse.remadeFrames)} remade → ${reuse.duration.toFixed(1)} s full video`;
    }
    if (!state.renderFrames) return "";
    if (state.long) return `${seconds(state.renderFrames)} · a run splits it into windows`;
    return `${seconds(state.renderFrames)} · one render`;
  };

  // Where each section's header and body sit in the prompt, keyed by name. Each body runs to the next header.
  const sectionSpans = (text) => {
    const found = state.boot.sections.map((name) => {
      const match = new RegExp(`^[ \\t]*${name}[ \\t]*:[ \\t]*`, "m").exec(text);
      return match ? {name, header: match.index, body: match.index + match[0].length} : null;
    }).filter(Boolean).sort((a, b) => a.header - b.header);
    return Object.fromEntries(found.map((span, index) => [span.name, {...span, end: found[index + 1]?.header ?? text.length}]));
  };
  const presentSections = (text) => {
    const spans = sectionSpans(text);
    return state.boot.sections.filter((name) => spans[name]);
  };

  // The readiness steps take you to what's missing.
  const jumpTo = (key) => {
    const go = (node, focus = node) => {
      if (!node) return;
      node.scrollIntoView({block: "center", behavior: "smooth"});
      focus?.focus({preventScroll: true});
    };
    if (key === "source") go($("stage"), null);
    else if (key === "frames") {
      const unedited = state.slots.findIndex((slot) => slot?.untouched);
      if (!state.slots.some(Boolean)) go($("extract"));
      else if (unedited >= 0) go(document.querySelector(`[aria-label="Quick edit for picture ${unedited + 1}"]`) ?? $("pictures"));
      else go($("pictures"), null);
    } else if (!$("request").value.trim() && !$("prompt").value.trim()) go($("request"));
    else go($("prompt"));
  };

  const renderNextRun = () => {
    const details = state.check?.next_run;
    if (!details) return;
    $("next-output").replaceChildren(...[`Remake ${state.mode}`, details.profile, details.output, details.sampling]
      .map((part) => h("span", {className: "chip"}, part)));
    $("next-context").textContent = details.context.slice(1).join(" · ");
    $("next-notes").replaceChildren(...details.notes.map((note) => h("p", {className: "small muted"}, note)));
  };

  const resultSource = () => (state.project?.output ?? (state.project ? openWindow().take : state.output));

  const seekVideos = () => {
    if (state.view === "full") return;
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
      const resultTime = whole && state.outputSpliced
        ? videoStart() + state.frame / fps()
        : (whole ? state.frame : state.frame - openWindow().render[0]) / fps();
      seek(result, resultTime);
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
    const take = resultSource();
    const canFull = Boolean(state.outputSpliced && take);
    if (state.view === "full" && !canFull) state.view = "result";
    const view = state.view;
    const align = view === "align";
    const tabs = [["source", "Source"], ["result", "Result"], ["compare", "Compare"]];
    if (canFull) tabs.push(["full", "Full result"]);
    if (state.alignSlot !== null) tabs.push(["align", "Alignment"]);
    $("tabs").replaceChildren(...pressed(tabs, view, (value) => { state.view = value; renderViewer(); }));
    // Keys 1–3 switch the first three views.
    [...$("tabs").children].slice(0, 3).forEach((tab, index) => { tab.title = `${tabs[index][1]} (${index + 1})`; });
    const wantsResult = view === "result" || view === "compare" || view === "full";
    const result = $("result-video");
    const url = take ? fileUrl(take) : "";
    if (wantsResult && take && result.dataset.url !== url) {
      result.src = url;
      result.dataset.url = url;
    }
    result.hidden = !(wantsResult && take);
    result.controls = view === "full";
    $("source-video").hidden = view === "result" || view === "full" || align || $("source-video").dataset.failed === "true";
    $("fallback").hidden = view === "result" || view === "full" || align || $("source-video").dataset.failed !== "true";
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
    $("transport").hidden = align || view === "full";
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
    $("keep-rest").checked = state.keepRest;
    $("overview").classList.toggle("reusing", state.keepRest);
    $("select-full").disabled = !state.source || (state.start <= 0.001 && state.end >= (state.source.duration || 0) - 0.05);
    $("window-options").hidden = !state.long;
    const line = state.project ? state.project.summary : state.text;
    $("summary").replaceChildren(...bold(`${line}${state.project ? " · hatched frames are held from the window before" : ""}`));
    const reuse = reuseBreakdown();
    const reuseLine = $("reuse-summary");
    if (reuse) {
      const parts = [];
      if (reuse.hasBefore) parts.push(`keep 0.00–${reuse.remadeStart.toFixed(2)} s`);
      parts.push(`remake ${reuse.remadeStart.toFixed(2)}–${reuse.remadeEnd.toFixed(2)} s (${reuse.remadeFrames} frames)`);
      if (reuse.hasAfter) parts.push(`keep ${reuse.remadeEnd.toFixed(2)}–${reuse.duration.toFixed(2)} s`);
      reuseLine.textContent = reuse.reusesAny
        ? `Output splices into the full source: ${parts.join(" · ")}.`
        : "Selected window already covers the full source video.";
      reuseLine.hidden = false;
    } else {
      reuseLine.hidden = true;
      reuseLine.textContent = "";
    }

    renderRuler(frames, last, start);
    $("legend").hidden = !state.project;
    $("joins").hidden = !state.project;
    $("joins").replaceChildren(...(state.project ? windows().slice(1) : []).map((window, offset) => {
      const continues = window.kind === "continues";
      const node = h("span", {
        className: continues ? "join warn" : "join",
        title: continues
          ? `Window ${offset + 2} continues the shot: its first ${window.hold} frames are held from window ${offset + 1}, so they meet in a plain cut.`
          : `Window ${offset + 2} starts on the source's cut at ${clock(window.own[0])}.`,
      }, continues ? `Shot continues · ${window.hold} held` : "On a cut");
      node.style.left = `${window.own[0] / last * 100}%`;
      return node;
    }));
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
    $("final").hidden = !(state.long || (state.project && allTaken()));
    $("split").hidden = Boolean(state.project);
    $("combine").hidden = !state.project;
    $("combine").disabled = !allTaken() || Boolean(state.project?.output) || running();
    $("final-line").textContent = !state.project ? "Longer than one window. A run splits it, or split now to edit each window first."
      : state.project.output ? (state.project.output_spliced ? "Spliced into the full source with its soundtrack and saved to Media." : "Joined with the source's soundtrack and saved to Media.")
      : allTaken() ? "Every window has a take. A run joins them itself, or combine now."
      : `Joins when every window has a take: ${state.project.windows.filter((window) => window.take).length} of ${windows().length} so far.`;
    renderRunBar();
    renderPlayhead();
  };

  // Seconds of the source above the strip, about a dozen ticks whatever the length.
  const renderRuler = (frames, last, start) => {
    const seconds = frames / fps();
    const step = [0.5, 1, 2, 5, 10, 15, 30, 60, 120].find((value) => seconds / value <= 12) ?? 300;
    const ticks = [];
    for (let at = Math.ceil(start / step) * step; at <= start + seconds + 1e-6; at += step) {
      const node = h("span", {}, `${at.toFixed(step < 1 ? 1 : 0)}s`);
      const ratio = Math.min(1, (at - start) * fps() / last);
      node.style.left = `${ratio * 100}%`;
      if (ratio < 0.02) node.classList.add("first");
      else if (ratio > 0.98) node.classList.add("last");
      ticks.push(node);
    }
    $("ruler").replaceChildren(...ticks);
  };

  // While a project renders: one segment per window, sized by its length.
  const renderRunBar = () => {
    const bar = $("run-bar");
    bar.hidden = !(running() && state.project);
    if (bar.hidden) return;
    const active = activeRenderingWindow();
    bar.replaceChildren(...windows().map((window, index) => {
      const kind = index === active ? "rendering" : QUEUED.includes(window.state) ? "queued" : "done";
      const node = h("span", {className: kind, title: `Window ${index + 1} · ${index === active ? "Rendering" : window.state_name}`},
        kind === "rendering" ? h("i") : null);
      node.style.flexGrow = String(window.own[1] - window.own[0]);
      return node;
    }));
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
        quickRow,
        editing ? h("p", {className: "small muted", id: "quick-status", role: "status"}, state.quick.status) : null,
        h("div", {className: "row tightrow"}, pinField,
          slot.before ? button("Undo edit", () => replacePicture(index, slot.before.path, {before: false, untouched: slot.before.untouched}))
            : null,
          // Still offered once sent: the frame can go to another model.
          button("Edit in Image", () => sendToImage(index)),
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
    const present = presentSections(text);
    const missing = state.boot.sections.filter((name) => !present.includes(name));
    renderOutline(text);
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
    $("writer-gemini-note").hidden = openai;
    $("writer-key-row").hidden = !openai;
    $("writer-url-row").hidden = !openai;
  };

  const focusPrompt = (from, to) => {
    const area = $("prompt");
    area.focus({preventScroll: true});
    area.setSelectionRange(from, to);
    const lineHeight = parseFloat(getComputedStyle(area).lineHeight) || 18;
    area.scrollTop = Math.max(0, (area.value.slice(0, from).split("\n").length - 2) * lineHeight);
    area.scrollIntoView({block: "nearest", behavior: "smooth"});
  };

  // Adds a missing section where it belongs, before the next one that's there.
  const addSection = (name) => {
    const text = $("prompt").value;
    const spans = sectionSpans(text);
    const after = state.boot.sections.slice(state.boot.sections.indexOf(name) + 1).find((other) => spans[other]);
    const block = `${name}:\n`;
    let updated;
    let cursor;
    if (after) {
      const at = spans[after].header;
      updated = `${text.slice(0, at)}${block}\n${text.slice(at)}`;
      cursor = at + block.length;
    } else {
      const head = text.trim() ? `${text.trimEnd()}\n\n` : "";
      updated = `${head}${block}`;
      cursor = updated.length;
    }
    $("prompt").value = updated;
    renderPromptMeta();
    queueCheck();
    autosave();
    focusPrompt(cursor, cursor);
  };

  const renderOutline = (text) => {
    const spans = sectionSpans(text);
    $("outline").replaceChildren(...state.boot.sections.map((name) => {
      const span = spans[name];
      const body = span ? text.slice(span.body, span.end).trim().replace(/\s+/g, " ") : "";
      const status = !span ? "missing" : body ? "written" : "empty";
      return h("li", {className: status},
        h("button", {type: "button",
                     title: span ? `Select ${name} in the prompt` : `Add ${name} to the prompt`,
                     onclick: () => (span ? focusPrompt(span.body, span.end) : addSection(name))},
          h("span", {className: "dot", "aria-hidden": "true"}),
          h("span", {className: "mono name"}, name),
          h("span", {className: "excerpt"}, span ? body || "Empty" : "Missing · click to add, or Rewrite fills it")));
    }));
  };

  const renderInspector = () => {
    const open = openWindow();
    const count = windows().length;
    for (const [id, step] of [["prev-window", -1], ["next-window", 1]]) {
      $(id).hidden = !state.project;
      $(id).disabled = state.index + step < 0 || state.index + step >= count;
    }
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
  const activeVideo = () => (["result", "full"].includes(state.view) && !$("result-video").hidden ? $("result-video") : $("source-video"));
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
    if (!state.playing || state.view === "full" || event.target !== activeVideo()) return;
    const whole = event.target.id === "source-video" || Boolean(state.project?.output) || !state.project;
    const zero = event.target.id === "source-video" || (whole && state.outputSpliced) ? videoStart() : 0;
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

    $("smart").addEventListener("change", (event) => { state.smart = event.target.checked; renderPictures(); changed(); autosave(); });
    $("auto-prompts").addEventListener("change", (event) => { state.auto = event.target.checked; changed(); autosave(); });
    $("keep-rest").addEventListener("change", (event) => {
      state.keepRest = event.target.checked;
      renderTimeline();
      renderAction();
      changed();
      autosave();
    });
    $("select-full").addEventListener("click", () => {
      if (!state.source || selectionRefused(true)) return;
      state.start = 0;
      state.end = Math.round((state.source.duration || 0) * 100) / 100;
      followStart();
      refreshSelection();
    });
    $("pinned").addEventListener("change", (event) => { state.pinned = event.target.checked; renderPictures(); changed(); autosave(); });
    $("scrub").addEventListener("input", (event) => setFrame(Number(event.target.value)));
    $("wipe").addEventListener("input", (event) => { state.wipe = Number(event.target.value); renderViewer(); });
    $("back").addEventListener("click", () => setFrame(state.frame - 1));
    $("next").addEventListener("click", () => setFrame(state.frame + 1));
    $("play").addEventListener("click", () => setPlaying(!state.playing));
    bindFullscreen($("viewer"), $("viewer-fullscreen"), (message) => toast(message, true));
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
    // Moving the playhead into a window opens it, as clicking the window does.
    const stepWindow = (delta) => windows()[state.index + delta] && setFrame(windows()[state.index + delta].own[0]);
    $("prev-window").addEventListener("click", () => stepWindow(-1));
    $("next-window").addEventListener("click", () => stepWindow(1));
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
    $("request").addEventListener("change", autosave);
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
      const present = presentSections(current);
      const missing = state.boot.sections.filter((name) => !present.includes(name));
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
      const frame = window.frameElement;
      if (frame) {
        let shown = frame.getClientRects().length > 0;
        new window.parent.ResizeObserver(() => {
          const visible = frame.getClientRects().length > 0;
          if (visible && !shown) {
            queueFit();
            if (state.boot) pulseNow();
          }
          shown = visible;
        }).observe(frame);
      }
    } catch { /* Framed by another origin. */ }
    try {
      state.boot = await request("GET", "/boot");
      $("workshop-note").hidden = gpu();
      $("edit-model").closest("label").hidden = !state.boot.edit_models.length;
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
      if (state.boot.incoming_source) {
        await useSource(state.boot.incoming_source);
        toast(`Loaded "${state.boot.incoming_source.name}" from Storyboard.`);
      } else {
        const savedKey = stored.get(SESSION.project);
        if (savedKey && state.boot.projects.some((row) => row.key === savedKey)) {
          await openProject(savedKey, {initialWindow: Number(stored.get(SESSION.window) || 0)});
        }
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
