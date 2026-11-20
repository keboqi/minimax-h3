// The Storyboard page. The browser holds the whole storyboard: moving between scenes
// asks the server for nothing, and edits save one field in the background.
(() => {
  const API = "/storyboard/api";
  const SAVE_DELAY_MS = 400;
  const SAVE_RETRY_MS = 3000;
  const POLL_MS = 1500;
  const {$, h, visibleArea, createToast} = globalThis.H3PageCommon;
  const {toast} = createToast();
  const chip = (label, pressed, onclick) =>
    h("button", {type: "button", className: "chip", "aria-pressed": String(pressed), onclick}, label);
  const button = (label, onclick, className = "btn", extra = {}) =>
    h("button", {type: "button", className, onclick, ...extra}, label);
  const icon = (path) => {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("width", 18); svg.setAttribute("height", 18); svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("fill", "none"); svg.setAttribute("stroke", "currentColor"); svg.setAttribute("stroke-width", 2);
    svg.setAttribute("aria-hidden", "true");
    const line = document.createElementNS("http://www.w3.org/2000/svg", "path");
    line.setAttribute("d", path); line.setAttribute("stroke-linecap", "round"); line.setAttribute("stroke-linejoin", "round");
    svg.append(line);
    return svg;
  };
  const stored = {
    get(key) { try { return sessionStorage.getItem(key); } catch { return null; } },
    set(key, value) { try { sessionStorage.setItem(key, value); } catch { /* Private windows refuse it. */ } },
  };

  const state = {
    boot: null, board: null, sceneId: null, previewTake: null, panel: null, inspectorTab: "prompt", question: null,
    confirmProfile: false, previousPrompts: new Map(), dragSceneId: null, deletingTake: null,
    includeOutdated: true, showSubtitles: true, burnSubtitles: false, subtitlesOpen: true,
    showTranslation: true, exportTranslation: stored.get("h3-storyboard-translate-subtitles") === "true",
    saveLabel: "All changes saved",
    run: null, seenRevision: null, pollTimer: null, previewKey: null,
    auto: null, autoBoards: null, autoSelected: new Set(), autoTimer: null,
  };
  const scenes = () => state.board.scenes;
  const scene = () => scenes().find((item) => item.id === state.sceneId) || scenes()[0];
  const boardUrl = (path = "") => `${API}/boards/${state.board.id}${path}`;
  const sceneUrl = (path = "") => boardUrl(`/scenes/${state.sceneId}${path}`);

  const request = async (method, url, body, {form = false, keepalive = false} = {}) => {
    const options = {method, credentials: "same-origin", keepalive};
    if (form) options.body = body;
    else if (body !== undefined) {
      options.headers = {"Content-Type": "application/json"};
      options.body = JSON.stringify(body);
    }
    const response = await fetch(url, options);
    let data = null;
    try { data = await response.json(); } catch { /* A proxy's error page has no JSON. */ }
    if (!response.ok) {
      throw Object.assign(new Error(data?.error || data?.detail || `The server answered ${response.status}.`),
        {refused: response.status < 500});
    }
    return data;
  };

  // Any action on the server: a failure reports itself and leaves the page as it is.
  const act = async (work, done) => {
    try {
      const result = await work();
      if (done) toast(done);
      return result;
    } catch (error) {
      toast(error.message, true);
      return undefined;
    }
  };

  // Saving ----------------------------------------------------------------
  const dirty = new Map();  // Waiting for their turn, by URL.
  const sending = new Map();  // The one request under way.
  let saveTimer = null;
  let saves = Promise.resolve();
  let savesFinished = 0;

  const setSaveLabel = (label) => {
    state.saveLabel = label;
    const node = $("save-state");
    if (node) node.textContent = label;
  };
  const edit = (field, value) => {
    const item = scene();
    item[field] = value;
    queueSave("PATCH", boardUrl(`/scenes/${item.id}`), field, value, item.id);
  };
  const queueSave = (method, url, field, value, sceneId = null) => {
    const entry = dirty.get(url) || {method, boardId: state.board.id, sceneId, fields: {}};
    entry.fields[field] = value;
    dirty.set(url, entry);
    setSaveLabel("Saving…");
    clearTimeout(saveTimer);
    saveTimer = setTimeout(flush, SAVE_DELAY_MS);
  };
  const saveEdit = async (url) => {
    const entry = dirty.get(url);
    if (!entry) return;
    const {method, boardId, fields} = entry;
    dirty.delete(url);
    sending.set(url, entry);
    try {
      const data = await request(method, url, fields, {keepalive: true});
      sending.delete(url);
      savesFinished += 1;
      if (state.board?.id === boardId) applyBoard(data, {keepInspector: true});
      if (!dirty.size) setSaveLabel("All changes saved");
    } catch (error) {
      sending.delete(url);
      if (error.refused) {
        setSaveLabel("Couldn't save");
        toast(error.message, true);
        if (state.board?.id === boardId) refreshBoard();
        return;
      }
      // A lost connection loses nothing: the edit waits for the next try, under anything newer.
      const newer = dirty.get(url);
      dirty.set(url, {...entry, fields: {...fields, ...newer?.fields}});
      setSaveLabel("Connection lost, retrying…");
      clearTimeout(saveTimer);
      saveTimer = setTimeout(flush, SAVE_RETRY_MS);
    }
  };
  const flush = () => {
    clearTimeout(saveTimer);
    for (const url of dirty.keys()) saves = saves.then(() => saveEdit(url));
    return saves;
  };
  // The page is closing, so what's waiting can't take its turn behind the request under way.
  const flushNow = () => {
    clearTimeout(saveTimer);
    for (const [url, {method, fields}] of dirty) {
      // Keepalive lets the request finish after the page has gone.
      request(method, url, fields, {keepalive: true}).catch(() => {});
    }
    dirty.clear();
  };
  const refreshBoard = async () => {
    const before = savesFinished;
    const data = await act(() => request("GET", boardUrl()));
    if (!data) return;
    // A save that finished meanwhile may be missing from this reply.
    if (before !== savesFinished) await refreshBoard();
    else applyBoard(data, {keepInspector: true});
  };
  // A server reply is older than anything typed since, so unsaved fields stay on top.
  const applyBoard = (data, {keepInspector = false} = {}) => {
    // An upload or writer reply may arrive after the user opens another storyboard.
    if (data.id !== state.board?.id) return;
    for (const edits of [sending, dirty]) {
      for (const {boardId, sceneId, fields} of edits.values()) {
        if (boardId !== data.id) continue;
        const target = sceneId ? data.scenes.find((item) => item.id === sceneId) : data;
        if (target) Object.assign(target, fields);
      }
    }
    state.board = data;
    const row = state.boot.boards.find((item) => item.id === data.id);
    if (row && data.generated !== undefined && row.generated !== data.generated) row.generated = data.generated;
    if (!data.scenes.some((item) => item.id === state.sceneId)) {
      state.sceneId = data.scenes[0].id;
      keepInspector = false;
    }
    if (data.run !== undefined) followRun(data.run);
    render({keepInspector});
  };

  // Rendering -------------------------------------------------------------
  const render = ({keepInspector = false} = {}) => {
    renderTopbar();
    renderStage();
    renderStrip();
    renderStripActions();
    if (!keepInspector) renderInspector();
    else renderInspectorDerived();
  };

  const closePanel = () => {
    if (!state.panel) return;
    state.panel = null;
    state.confirmProfile = false;
    renderTopbar();
    renderInspector();
  };

  const renderTopbar = () => {
    const panelButton = (name, label) => chip(label, state.panel === name, () => {
      state.panel = state.panel === name ? null : name;
      state.confirmProfile = false;
      renderTopbar();
      renderInspector();
      if (state.panel === "auto") loadAuto();
    });
    const directorButton = chip("Director", state.panel === null, () => {
      state.panel = null;
      state.confirmProfile = false;
      renderTopbar();
      renderInspector();
    });
    const select = h("select", {"aria-label": "Storyboard", onchange: (event) => openBoard(event.target.value)},
      state.boot.boards.map((row) => h("option", {value: row.id, selected: row.id === state.board.id},
        row.name + (row.generated ? " · Generated" : ""))));
    const brand = h("div", {className: "topbar-brand"},
      h("span", {className: "topbar-brand-icon", "aria-hidden": "true"},
        icon("M4 6h16v12H4z M10 9.5l5 2.5-5 2.5z")),
      h("strong", {}, "Storyboard"));
    const subtitleToggle = chip("Subtitles", state.subtitlesOpen, () => {
      state.subtitlesOpen = !state.subtitlesOpen;
      const home = $("subtitle-toolbar-home");
      if (home) home.hidden = !state.subtitlesOpen;
      renderTopbar();
    });
    const home = $("subtitle-toolbar-home");
    if (home) home.hidden = !state.subtitlesOpen;
    $("topbar").replaceChildren(...[
      brand,
      h("label", {className: "board-picker"}, "Storyboard", select),
      button("New storyboard", showStart, "btn sm"),
      h("div", {className: "stat-pill mono small"},
        `${scenes().length} scenes · ${state.board.total} · ${state.board.done} done`),
      h("div", {className: "small muted", id: "save-state", role: "status"}, state.saveLabel),
      h("div", {className: "grow"}),
      state.auto?.active ? h("div", {className: "small muted", id: "auto-state", role: "status"}, autoSummary()) : null,
      h("nav", {className: "chips topbar-nav", role: "group", "aria-label": "Panels"},
        directorButton,
        state.boot.gpu ? panelButton("auto", "Auto render") : null,
        panelButton("cast", "Cast"), panelButton("versions", "Versions"), panelButton("settings", "Settings")),
      subtitleToggle,
      button("Play full video", playFullVideo, "btn primary sm"),
    ].filter(Boolean));
  };

  const shownTake = () => {
    const item = scene();
    return item.takes.find((take) => take.id === (state.previewTake ?? item.in_use)) || null;
  };

  const renderStage = () => {
    const item = scene();
    const index = scenes().indexOf(item);
    $("stage-head").replaceChildren(
      h("div", {className: "nav-group"},
        button(icon("M15 6l-6 6 6 6"), () => step(-1), "btn icon", {"aria-label": "Previous scene", disabled: index === 0}),
        button(icon("M9 6l6 6-6 6"), () => step(1), "btn icon",
          {"aria-label": "Next scene", disabled: index === scenes().length - 1})),
      h("span", {className: "scene-code mono"}, String(item.number).padStart(2, "0")),
      h("h1", {id: "scene-heading"}, `Scene ${item.number} · ${item.title || "Untitled scene"}`),
      stateBadge(item, false, "scene-state"),
      h("span", {className: "small muted"}, `Runs as ${item.runs_as}`),
      h("div", {className: "stage-actions", role: "group", "aria-label": "Scene actions"},
        button("Insert scene after", () => addScene(item.id), "btn sm"),
        button("Insert video after", () => addScene(item.id, "media"), "btn sm",
          {title: "Add a scene that plays a video from Media; the scene after it can continue it"}),
        button("Duplicate scene", duplicateScene, "btn sm"),
        h("span", {className: "divider", "aria-hidden": "true"}),
        button("Move earlier", () => moveScene(-1), "btn sm", {disabled: item.number === 1}),
        button("Move later", () => moveScene(1), "btn sm", {disabled: item.number === scenes().length}),
        h("div", {className: "grow"}),
        button("Delete scene", deleteScene, "btn sm danger", {disabled: scenes().length === 1})),
    );
    renderPreview();
    const notes = [];
    if (item.state === "outdated") {
      notes.push(h("div", {className: "warning", role: "status"}, item.kind === "media"
        ? "This clip is out of date: its video, In, Out, Fit or the storyboard's size changed since it was prepared. Prepare it again to update it."
        : "This take is out of date: the scene, or the scene it continues from, changed since it was made. Generate it again to update it."));
    }
    if (item.takes.length) notes.push(h("div", {className: "small muted"}, state.board.take_note));
    if (state.board.queue.length && !activeGenerate()) {
      notes.push(h("div", {className: "small muted"},
        `${state.board.queue.length} scene(s) remain in the generation queue. Press Generate all remaining scenes to continue.`));
    }
    $("stage-notes").replaceChildren(...notes);
    renderTakes();
  };

  const subtitleTrack = (src) => src && h("track", {src, kind: "subtitles", srclang: "und", label: "Subtitles"});

  const setSubtitleTrack = (video, src, show) => {
    let track = video.querySelector("track");
    if (src !== (track?.getAttribute("src") ?? null)) {
      track?.remove();
      track = subtitleTrack(src);
      if (track) video.append(track);
    }
    if (track) track.track.mode = show ? "showing" : "disabled";
  };

  const pendingTranslations = new Map();
  const savedTranslation = (take, language) => take?.subtitle_translations?.[language];
  const translationKey = (boardId, take, language) => JSON.stringify([boardId, take.id, take.subtitles, language]);
  const requestTakeTranslation = (boardId, take, options, force = false) => {
    const {language} = options;
    const cached = savedTranslation(take, language);
    if (cached && !force) return Promise.resolve(cached);
    const key = translationKey(boardId, take, language);
    if (pendingTranslations.has(key)) return pendingTranslations.get(key);
    const url = `${API}/boards/${boardId}/subtitle-translations`;
    const pending = request("POST", url, {...options, force, take_ids: [take.id]}).then(({tracks}) => {
      const track = tracks[take.id];
      if (!track) throw new Error("This take has no subtitles to translate.");
      (take.subtitle_translations ??= {})[language] = track;
      if (state.board?.id === boardId) {
        const latest = scenes().flatMap((item) => item.takes)
          .find((item) => item.id === take.id && item.subtitles === take.subtitles);
        if (latest) (latest.subtitle_translations ??= {})[language] = track;
      }
      return track;
    }).finally(() => {
      pendingTranslations.delete(key);
      if (state.board?.id === boardId && !$("editor").hidden) renderPreview();
    });
    pendingTranslations.set(key, pending);
    return pending;
  };

  const releaseVideo = (video) => {
    video.pause();
    video.removeAttribute("src");
    video.load();
  };

  const bulkTranslations = new Map();
  let refreshFullSubtitles = null;
  const subtitleTakes = () => scenes().map((item) => item.takes.find((take) =>
    take.id === item.in_use && !take.missing && take.subtitles)).filter(Boolean);
  const bulkTranslationKey = (boardId, language) => JSON.stringify([boardId, language]);

  const renderSubtitleControls = (take) => {
    const language = subtitleLanguage();
    if ($("subtitle-language").value !== language) $("subtitle-language").value = language;
    const boardId = state.board.id;
    const available = take?.subtitles && !take.missing;
    const saved = available && savedTranslation(take, language);
    const pending = available && pendingTranslations.has(translationKey(boardId, take, language));
    const captioned = subtitleTakes();
    const ready = captioned.filter((item) => savedTranslation(item, language)).length;
    const missing = captioned.length - ready;
    const bulk = bulkTranslations.get(bulkTranslationKey(boardId, language));

    const subtitles = $("show-subtitles");
    subtitles.checked = state.showSubtitles;
    subtitles.disabled = !available && !captioned.length;
    const show = $("show-translation");
    show.disabled = !state.showSubtitles || (!saved && !ready);
    show.checked = Boolean(saved || ready) && state.showTranslation;

    const sceneAction = $("translate-scene");
    sceneAction.disabled = !available || Boolean(pending);
    sceneAction.textContent = pending ? "Translating this take…" : saved ? "Regenerate this take" : "Translate this take";
    const boardAction = $("translate-board");
    boardAction.disabled = !captioned.length || Boolean(bulk?.running);
    boardAction.textContent = missing ? `Translate missing takes (${missing})`
      : captioned.length ? "Regenerate all translations" : "Translate missing takes";

    const exportTranslation = $("export-translation");
    exportTranslation.disabled = !captioned.length || Boolean(state.run?.active);
    exportTranslation.checked = state.exportTranslation;
    const burn = $("burn-subtitles");
    burn.disabled = exportTranslation.disabled;
    burn.checked = state.burnSubtitles;

    const status = $("subtitle-status");
    const number = scene().number;
    const selected = !available ? "No subtitles" : pending ? "Subtitles available, translating…"
      : saved ? "Subtitles available, translation ready" : "Subtitles available, translation missing";
    const text = [
      `Scene ${number} · ${selected}`,
      captioned.length ? `${ready} of ${captioned.length} current takes translated` : "",
      bulk?.running && `Processing translations (${bulk.done}/${bulk.total} finished)`,
      bulk?.error,
    ].filter(Boolean).join("  ·  ");
    if (status.textContent !== text) status.textContent = text;
  };

  const updatePreviewSubtitles = (video, take) => {
    renderSubtitleControls(take);
    const src = take?.missing ? null : take?.subtitles || null;
    const translated = state.showTranslation && src && savedTranslation(take, subtitleLanguage());
    if (video) setSubtitleTrack(video, translated || src, state.showSubtitles);
    refreshFullSubtitles?.();
  };

  const syncSubtitleVisibility = (visible) => {
    if (state.showSubtitles === visible) return;
    state.showSubtitles = visible;
    renderPreview();
  };

  const translateScene = async () => {
    const take = shownTake();
    if (!take?.subtitles || take.missing) return;
    const options = translationOptions(), boardId = state.board.id;
    const force = Boolean(savedTranslation(take, options.language));
    const pending = requestTakeTranslation(boardId, take, options, force);
    renderPreview();
    const track = await act(() => pending);
    if (track && !force && state.board?.id === boardId && shownTake()?.id === take.id
        && subtitleLanguage() === options.language) {
      state.showTranslation = true;
      renderPreview();
    }
  };

  const translateBoard = async () => {
    const options = translationOptions(), boardId = state.board.id;
    const key = bulkTranslationKey(boardId, options.language);
    if (bulkTranslations.get(key)?.running) return;
    const captioned = subtitleTakes();
    const missing = captioned.filter((take) => !savedTranslation(take, options.language));
    const force = !missing.length;
    const queue = force ? captioned : missing;
    if (!queue.length) return;

    const selected = queue.findIndex((take) => take.id === shownTake()?.id);
    if (selected > 0) queue.unshift(queue.splice(selected, 1)[0]);
    const progress = {running: true, done: 0, total: queue.length, error: ""};
    bulkTranslations.set(key, progress);
    state.showTranslation = true;
    const refresh = () => {
      if (state.board?.id === boardId && !$("editor").hidden) renderPreview();
    };
    refresh();
    const translateNext = async () => {
      while (queue.length) {
        const take = queue.shift();
        const pending = requestTakeTranslation(boardId, take, options, force);
        refresh();
        try {
          await pending;
        } catch (error) {
          progress.error ||= error.message;
        }
        progress.done += 1;
        refresh();
      }
    };
    // Reserve browser connections for individual take requests while the batch runs.
    await Promise.all(Array.from({length: Math.min(4, queue.length)}, translateNext));
    progress.running = false;
    if (!progress.error) bulkTranslations.delete(key);
    refresh();
  };

  // Rebuilt only when the take changes, so playback survives saves and polls.
  const renderPreview = () => {
    const item = scene();
    const take = shownTake();
    const key = take ? take.id : `empty:${item.id}`;
    const node = $("preview");
    if (key === state.previewKey) {
      updatePreviewSubtitles(node.querySelector("video"), take);
      return;
    }
    state.previewKey = key;
    node.querySelectorAll("video").forEach(releaseVideo);
    if (take && !take.missing) {
      const video = h("video", {src: take.video, poster: take.poster, preload: "none", controls: true,
        "aria-label": `Take ${take.number} of scene ${item.number}`,
        onplay: () => $("full-video").querySelector("video:not(.standby)")?.pause()});
      video.textTracks.addEventListener("change", () => {
        if (!node.contains(video)) return;
        const track = video.querySelector("track");
        if (track) syncSubtitleVisibility(track.track.mode === "showing");
      });
      node.replaceChildren(video);
      updatePreviewSubtitles(video, take);
    } else {
      updatePreviewSubtitles(null, null);
      node.replaceChildren(h("div", {className: "empty"},
        h("strong", {}, take ? "This take is missing from disk" : item.kind === "media" ? "Not prepared yet" : "No take yet"),
        h("span", {className: "muted"}, item.kind === "media"
          ? "Choose a video, set In, Out and Fit, then press Prepare clip."
          : state.boot.gpu ? "Finish the scene, then generate it." : "Prepare it here, then generate it on the GPU endpoint.")));
    }
  };

  const activeGenerate = () => (state.run?.active && state.run.kind === "generate" ? state.run : null);
  const generateSceneLabel = (item) => item.takes.length ? "Generate another take" : "Generate this scene";
  const generateSceneDisabled = (item) => Boolean(state.run?.active) || item.issues.length > 0;
  const generateSceneButton = (item, id) => state.boot.gpu && button(generateSceneLabel(item),
    () => startGeneration({action: "scene", scene_id: item.id}), "btn primary",
    {disabled: generateSceneDisabled(item), id});

  const syncInspectorGenerate = () => {
    const generate = $("inspector-generate-scene");
    if (!generate) return;
    const item = scene();
    generate.disabled = generateSceneDisabled(item);
    const label = generateSceneLabel(item);
    if (generate.textContent !== label) generate.textContent = label;
  };

  const renderTakes = () => {
    const item = scene();
    if (item.kind === "media") {
      const busy = Boolean(state.run?.active);
      $("take-row").replaceChildren(
        h("span", {className: "small muted"}, item.state === "done"
          ? "Prepared: fitted to the storyboard, as it will play in the full video."
          : item.takes.length ? "In, Out, Fit or the storyboard size changed since it was prepared." : ""),
        h("div", {className: "grow"}),
        button(item.takes.length ? "Prepare again" : "Prepare clip", startPrepare, "btn primary",
          {disabled: item.issues.length > 0 || busy || (item.state === "done"), id: "prepare-scene",
            title: busy ? "Wait for the current run to finish." : "Fit the clip to the storyboard and cut its ending for the next scene"}));
      $("take-confirm").replaceChildren();
      $("takes-panel").hidden = false;
      return;
    }
    const shown = shownTake();
    const cards = item.takes.map((take) => {
      const selected = shown?.id === take.id;
      const inUse = take.id === item.in_use;
      const thumbSrc = take.poster || item.thumb;
      return h("button", {
        type: "button",
        className: "take-card",
        "aria-pressed": String(selected),
        onclick: () => { state.previewTake = take.id; state.question = null; state.deletingTake = null; renderPreview(); renderTakes(); },
      },
      h("span", {className: "take-card-thumb"},
        thumbSrc ? h("img", {src: thumbSrc, alt: "", loading: "lazy"}) : null,
        inUse ? h("span", {className: "take-in-use"}, "In use") : null),
      h("span", {className: "take-card-meta"},
        h("strong", {}, `Take ${take.number}`),
        h("span", {className: "mono small muted"}, take.seed !== undefined ? `#${take.seed}` : "")));
    });
    const chips = cards;
    const row = [];
    if (shown?.subtitle_error) row.push(h("span", {className: "small muted"}, shown.subtitle_error));
    const choices = [];
    if (shown && shown.id !== item.in_use) {
      choices.push(button("Use this take", useTake, "btn quiet"));
      choices.push(button("Delete take", () => { state.deletingTake = shown.id; renderTakes(); }, "btn sm danger",
        {disabled: Boolean(state.run?.active),
          title: state.run?.active ? "Wait for the current run to finish." : "Delete this take for good"}));
    }
    clearTimeout(state.reviewTimer);
    const review = shown?.review;
    if (review?.status === "pending") {
      row.push(h("span", {className: "small muted"}, "Scoring this take…"));
      // A take's score arrives after its run; look again until it has.
      if (!activeGenerate()) state.reviewTimer = setTimeout(() => refreshBoard().catch(() => {}), 4000);
    } else if (review?.status === "done") {
      row.push(h("span", {className: `review-chip ${review.fit >= 70 ? "good" : review.fit < 45 ? "warn" : ""}`,
        title: `How well this take fits the scene prompt, scored by Gemini Decision.${review.tags.length ? ` Tags: ${review.tags.join(", ")}` : ""}`},
        `Prompt fit ${review.fit} · ${review.label}`));
    }
    if (shown) {
      row.push(h("span", {className: "take-meta mono"},
        `Seed ${shown.seed}`,
        button("Use seed", () => useSeed(shown.seed), "btn sm",
          {title: "Set this scene's seed to this take's seed"})));
    }
    const generate = generateSceneButton(item, "generate-scene");
    $("take-row").replaceChildren(...[
      chips.length ? h("div", {className: "take-chips"}, chips) : null,
      row.length || choices.length ? h("div", {className: "take-detail"}, row, h("div", {className: "grow"}), choices) : null,
      generate ? h("div", {className: "take-foot"}, generate) : null].filter(Boolean));
    const doomed = shown?.id === state.deletingTake && shown.id !== item.in_use ? shown : null;
    $("take-confirm").replaceChildren(...(state.question ? [h("div", {className: "confirm", role: "alertdialog", "aria-label": "Switch takes"},
      h("p", {}, state.question),
      state.boot.gpu && button("Switch and regenerate them", () => confirmTake(true), "btn primary"),
      button("Switch, regenerate later", () => confirmTake(false)),
      button("Cancel", () => { state.question = null; renderTakes(); }),
    )] : doomed ? [h("div", {className: "confirm", role: "alertdialog", "aria-label": "Delete take"},
      h("p", {}, deleteQuestion(doomed)),
      button("Delete permanently", () => deleteTake(doomed), "btn danger"),
      button("Cancel", () => { state.deletingTake = null; renderTakes(); }),
    )] : []));
    $("takes-panel").hidden = !$("take-row").children.length && !$("take-confirm").children.length;
  };

  const deleteQuestion = (take) => {
    const later = take.continued_by;
    const scenesNote = later.length
      ? ` Scene ${later.join(" and ")} ${later.length === 1 ? "was" : "were"} made from it, and will stay out of date until regenerated.`
      : "";
    return `Delete take ${take.number} for good? Its video can't be recovered.${scenesNote}`;
  };

  const useSeed = (seed) => {
    edit("seed", seed);
    const input = $("scene-seed");
    if (input) input.value = String(seed);
    toast(`Scene seed set to ${seed}.`);
  };

  const deleteTake = async (take) => {
    state.deletingTake = null;
    const data = await act(
      () => request("DELETE", sceneUrl(`/takes/${take.id}`)),
      `Deleted take ${take.number}.`,
    );
    if (!data) return;
    state.previewTake = null;
    state.question = null;
    applyBoard(data, {keepInspector: true});
  };

  const useTake = async () => {
    const data = await act(() => request("POST", sceneUrl("/use-take"), {take_id: shownTake().id}));
    if (!data) return;
    state.question = data.question ? `Use this take for scene ${scene().number}? ${data.question}` : null;
    if (!data.question) state.previewTake = null;
    applyBoard(data, {keepInspector: true});
  };
  const confirmTake = async (regenerate) => {
    const take = shownTake();
    const sceneId = state.sceneId;
    state.question = null;
    if (regenerate) {
      await startGeneration({action: "switch", scene_id: sceneId, take_id: take.id});
      state.previewTake = null;
      return;
    }
    const data = await act(() => request("POST", sceneUrl("/use-take"), {take_id: take.id, confirmed: true}),
      "Take in use changed. The scenes that continue from it are out of date.");
    state.previewTake = null;
    if (data) applyBoard(data, {keepInspector: true});
  };

  const stateClass = (item) => (item.state_label === "Needs input" ? "state-needs" : `state-${item.state}`);
  const statePrefix = (item, isRunning) => {
    if (isRunning) return "● ";
    if (item.state === "done") return "✓ ";
    if (item.state === "outdated") return "↻ ";
    if (item.state_label === "Needs input") return "! ";
    return "○ ";
  };
  const stateBadge = (item, isRunning, extraClass = "") => h("span", {className: `${stateClass(item)} ${extraClass}`.trim()},
    `${statePrefix(item, isRunning)}${isRunning ? "Generating" : item.state_label}`);

  const toggleSceneLink = async (item) => {
    const url = `${API}/boards/${state.board.id}/scenes/${item.id}`;
    await flush();
    await saves;
    const data = await act(() => request("PATCH", url, {link: item.flow ? "cut" : "flow"}));
    if (data) applyBoard(data, {keepInspector: item.id !== state.sceneId});
  };

  const dropSceneOn = async (targetSceneId) => {
    const fromId = state.dragSceneId;
    state.dragSceneId = null;
    if (!fromId || fromId === targetSceneId) return;
    const list = scenes();
    const fromIndex = list.findIndex((item) => item.id === fromId);
    const toIndex = list.findIndex((item) => item.id === targetSceneId);
    if (fromIndex < 0 || toIndex < 0 || fromIndex === toIndex) return;
    await flush();
    await saves;
    const data = await act(() => request("POST", `${API}/boards/${state.board.id}/scenes/${fromId}/move`,
      {offset: toIndex - fromIndex}));
    if (data) applyBoard(data, {keepInspector: true});
  };

  const FLOW_ICON = "M5 12h14M13 6l6 6-6 6";
  const CUT_ICON = "M8.1 8.1L20 20M8.1 15.9L20 4M9 6a3 3 0 1 1-6 0a3 3 0 1 1 6 0M9 18a3 3 0 1 1-6 0a3 3 0 1 1 6 0";
  const VIDEO_ICON = "M5 5h14a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2zM10 9l5 3-5 3z";

  const renderStrip = () => {
    const stripHead = $("strip-head");
    if (stripHead) {
      stripHead.replaceChildren(
        h("div", {className: "filmstrip-head-left"},
          h("span", {className: "filmstrip-label"}, `Timeline · ${scenes().length} Scenes`),
          h("span", {className: "small muted"}, "Click connector pill to toggle Continue vs Cut")),
        h("div", {className: "chips"},
          button([icon("M12 5v14M5 12h14"), "Add scene"], () => addScene(), "btn sm"),
          button([icon(VIDEO_ICON), "Add video from Media"], () => addScene(null, "media"), "btn sm",
            {title: "A scene that plays a video from Media; the next scene can continue it"})));
    }
    const running = activeGenerate()?.scene_id;
    const items = [];
    scenes().forEach((item, index) => {
      if (index > 0 && item.kind === "media") {
        items.push(h("span", {className: "link-toggle cut", title: `Scene ${item.number} plays a video, so it always starts fresh.`,
          "aria-label": `Transition to scene ${item.number}: fresh start, a video from Media.`},
          h("span", {className: "link-pill-text"}, "CLIP")));
      } else if (index > 0) {
        items.push(h("button", {
          type: "button",
          className: `link-toggle ${item.flow ? "flow" : "cut"}`,
          title: item.flow
            ? `Scene ${item.number} continues scene ${item.number - 1}. Click to start fresh (cut).`
            : `Scene ${item.number} starts fresh (cut). Click to continue scene ${item.number - 1}.`,
          "aria-label": item.flow
            ? `Transition to scene ${item.number}: continues scene ${item.number - 1}. Click to switch to fresh start.`
            : `Transition to scene ${item.number}: fresh start. Click to switch to continue scene ${item.number - 1}.`,
          onclick: () => toggleSceneLink(item),
        }, h("span", {className: "link-pill-text"}, item.flow ? "CONT" : "CUT")));
      }
      const thumb = item.thumb
        ? h("img", {className: "thumb", alt: "", decoding: "async",
          loading: item.id === state.sceneId ? "eager" : "lazy", src: item.thumb})
        : h("span", {className: "thumb"});
      thumb.draggable = false;
      const isRunning = item.id === running;
      const durationText = item.flow
        ? `${item.seconds}s (+${Number(item.added_seconds.toFixed(1))}s)`
        : `${item.seconds}s`;
      const thumbWrap = h("span", {className: "thumb-wrap"},
        thumb,
        h("span", {className: "thumb-num mono"}, String(item.number).padStart(2, "0")),
        h("span", {className: "thumb-dur mono"}, durationText));
      const card = h("button", {
        type: "button", className: "frame", draggable: true,
        "aria-current": String(item.id === state.sceneId), "data-scene": item.id,
        "aria-label": `Scene ${item.number}, ${item.title}, ${item.seconds} seconds, ${item.state_label}. ${item.summary}`,
        onclick: () => select(item.id),
        ondragstart: (event) => {
          state.dragSceneId = item.id;
          if (event.dataTransfer) event.dataTransfer.effectAllowed = "move";
        },
        ondragover: (event) => {
          if (!state.dragSceneId || state.dragSceneId === item.id) return;
          event.preventDefault();
          card.classList.add("drag-over");
        },
        ondragleave: () => card.classList.remove("drag-over"),
        ondrop: (event) => {
          event.preventDefault();
          card.classList.remove("drag-over");
          dropSceneOn(item.id);
        },
        ondragend: () => {
          state.dragSceneId = null;
          document.querySelectorAll("#strip .frame.drag-over").forEach((node) => node.classList.remove("drag-over"));
        },
      }, thumbWrap,
      isRunning ? h("span", {className: "bar"}, h("i")) : null,
      h("span", {className: "title"}, item.title || "Untitled scene"),
      h("span", {className: "meta"},
        stateBadge(item, isRunning),
        h("span", {className: "small muted"}, item.flow ? "Continue" : "Cut")));
      items.push(card);
    });
    items.push(button([icon("M12 5v14M5 12h14"), "Add scene"], () => addScene(), "add-frame"));
    items.push(button([icon(VIDEO_ICON), "Add video from Media"], () => addScene(null, "media"), "add-frame",
      {title: "A scene that plays a video from Media; the next scene can continue it"}));
    $("strip").replaceChildren(...items);
  };

  const renderStripActions = () => {
    const run = state.run;
    const busy = run?.active;
    const generating = activeGenerate();
    const pending = state.board.pending;
    const actions = [];
    if (state.boot.gpu) {
      actions.push(generating
        ? button("Interrupt", interrupt, "btn danger")
        : button(pending ? `Generate all remaining scenes (${pending})` : "All scenes are done",
          () => startGeneration({action: "remaining"}), "btn primary", {disabled: !pending || busy}));
      actions.push(busy && run.kind === "subtitles"
        ? button("Interrupt subtitles", interrupt, "btn danger")
        : button("Generate word-aligned subtitles", startSubtitles, "btn", {
          disabled: busy || !scenes().some((scene) => scene.kind !== "media"
            && scene.takes.some((take) => take.id === scene.in_use && !take.missing)),
          title: "Transcribe and time the audio in all generated takes currently in use",
        }));
    }
    actions.push(
      button("Export to Media", exportToMedia, "btn", {disabled: busy}),
      h("label", {className: "check-label"},
        h("input", {type: "checkbox", checked: state.includeOutdated,
          onchange: (event) => { state.includeOutdated = event.target.checked; }}),
        "Include out-of-date scenes"),
    );
    if (busy) {
      actions.push(h("div", {className: "run-status", role: "status", id: "run-status"}, run.status));
    }
    $("strip-actions").replaceChildren(...actions);
    renderSubtitleControls(shownTake());
  };

  const restoreSubtitleToolbar = () => {
    const home = $("subtitle-toolbar-home");
    const toolbar = $("subtitle-toolbar");
    if (!home.contains(toolbar)) home.append(toolbar);
  };

  const closeFullVideo = () => {
    const root = $("full-video");
    const videos = root.querySelectorAll("video");
    restoreSubtitleToolbar();
    refreshFullSubtitles = null;
    root.replaceChildren();
    videos.forEach(releaseVideo);
  };

  const playFullVideo = async () => {
    if (!state.includeOutdated && (dirty.size || sending.size)) {
      const boardId = state.board.id;
      await flush();
      if (state.board.id !== boardId || $("editor").hidden) return;
    }
    const playlist = [];
    for (const item of scenes()) {
      const take = item.takes.find((take) => take.id === item.in_use);
      if (!take) return toast(`Scene ${item.number} has no take yet.`, true);
      if (item.state === "outdated" && !state.includeOutdated) {
        return toast(`Scene ${item.number} is outdated. Regenerate it or include out-of-date scenes.`, true);
      }
      if (take.missing) return toast(`Scene ${item.number}'s take is missing from disk.`, true);
      playlist.push(take);
    }
    closeFullVideo();
    // Preload the next take and seek past its overlap before the handoff.
    let active = h("video", {preload: "auto", playsInline: true});
    let standby = h("video", {preload: "auto", playsInline: true, className: "standby"});
    const videos = [active, standby];
    const takesByVideo = new Map();
    let index = 0;
    const position = h("span", {className: "small muted", role: "status"});
    const updateSubtitles = (video) => {
      const take = takesByVideo.get(video);
      const current = take && scenes().flatMap((item) => item.takes).find((item) => item.id === take.id);
      const source = current || take;
      const translated = state.showTranslation && savedTranslation(source, subtitleLanguage());
      setSubtitleTrack(video, translated || source?.subtitles || null, state.showSubtitles);
    };
    const loadClip = (video, take) => {
      takesByVideo.set(video, take);
      if (!take) {
        updateSubtitles(video);
        video.onloadedmetadata = null;
        video.onseeking = null;
        releaseVideo(video);
        return;
      }
      video.onloadedmetadata = () => {
        if (take.preview_start) video.currentTime = take.preview_start;
      };
      video.onseeking = () => {
        if (video.currentTime < take.preview_start - 0.01) video.currentTime = take.preview_start;
      };
      video.src = take.video;
      updateSubtitles(video);
    };
    const playScene = () => {
      const current = active;
      current.classList.remove("standby");
      current.controls = true;
      current.setAttribute("aria-label", `Storyboard scene ${index + 1}`);
      standby.classList.add("standby");
      standby.controls = false;
      position.textContent = `Scene ${index + 1} of ${playlist.length}`;
      current.play().catch(() => {
        if (current === active && current.isConnected && current.error) {
          position.textContent = `Scene ${index + 1} could not be played.`;
        }
      });
      loadClip(standby, playlist[index + 1]);
    };
    for (const video of videos) {
      if (document.fullscreenEnabled) video.controlsList?.add("nofullscreen");
      video.disablePictureInPicture = true;
      video.onplay = () => {
        if (video.isConnected) $("preview").querySelector("video")?.pause();
      };
      video.onerror = () => {
        if (video === active && video.isConnected) position.textContent = `Scene ${index + 1} could not be played.`;
      };
      video.onended = () => {
        if (video !== active || !video.isConnected || index + 1 === playlist.length) return;
        index += 1;
        [active, standby] = [standby, active];
        active.muted = video.muted;
        active.volume = video.volume;
        active.playbackRate = video.playbackRate;
        playScene();
      };
      video.textTracks.addEventListener("change", () => {
        if (video !== active || !video.isConnected) return;
        const track = video.querySelector("track");
        if (!track) return;
        syncSubtitleVisibility(track.track.mode === "showing");
      });
    }
    const fullscreen = button("Fullscreen", () => {
      const change = document.fullscreenElement === panel
        ? document.exitFullscreen() : panel.requestFullscreen();
      change.catch(() => toast("Couldn't change fullscreen mode.", true));
    }, "btn sm");
    const panel = h("div", {className: "full-video-panel", onfullscreenchange: () => {
      const expanded = document.fullscreenElement === panel;
      fullscreen.textContent = expanded ? "Exit fullscreen" : "Fullscreen";
      if (expanded) panel.prepend($("subtitle-toolbar"));
      else if (panel.isConnected) restoreSubtitleToolbar();
    }},
      h("div", {className: "full-video-head"},
        h("strong", {className: "small"}, "Full storyboard preview"), position, h("div", {className: "grow"}),
        document.fullscreenEnabled && fullscreen,
        button("Close", closeFullVideo, "btn sm")),
      h("div", {className: "full-video-viewport"}, videos));
    $("full-video").replaceChildren(panel);
    refreshFullSubtitles = () => videos.forEach(updateSubtitles);
    loadClip(active, playlist[0]);
    playScene();
  };

  const showJoinedVideo = (run) => {
    closeFullVideo();
    $("full-video").replaceChildren(h("div", {className: "full-video-panel"},
      h("div", {className: "full-video-head"},
        h("strong", {className: "small"}, "Full storyboard video"),
        h("span", {className: "small muted"}, "Continuing scenes drop their repeated opening; fresh starts cut."),
        h("div", {className: "grow"}),
        run.subtitles_url && h("a", {className: "btn sm", href: run.subtitles_url,
          download: `${state.board.name || "storyboard"}.vtt`}, "Download subtitles"),
        h("a", {className: "btn sm", href: run.result_url, download: `${state.board.name || "storyboard"}.mp4`}, "Download"),
        button("Close", closeFullVideo, "btn sm")),
      h("video", {src: run.result_url, controls: true, preload: "metadata", "aria-label": "Full video"},
        subtitleTrack(run.subtitles_url))));
  };

  // Moving between scenes -------------------------------------------------
  const select = (sceneId) => {
    if (sceneId === state.sceneId && !state.panel) return;
    flush();
    state.sceneId = sceneId;
    state.previewTake = null;
    state.question = null;
    state.deletingTake = null;
    state.panel = null;
    state.confirmProfile = false;
    stored.set("h3-storyboard-scene", sceneId);
    document.querySelectorAll("#strip .frame").forEach((frame) => {
      frame.setAttribute("aria-current", String(frame.dataset.scene === sceneId));
      if (frame.dataset.scene === sceneId) frame.scrollIntoView({block: "nearest", inline: "nearest"});
    });
    renderTopbar();
    renderStage();
    renderInspector();
  };
  const step = (offset) => {
    const next = scenes()[scenes().indexOf(scene()) + offset];
    if (next) select(next.id);
  };
  document.addEventListener("keydown", (event) => {
    if ($("editor").hidden || event.altKey || event.ctrlKey || event.metaKey) return;
    if (event.key === "Escape" && state.panel && !$("library").open) {
      closePanel();
      event.preventDefault();
      return;
    }
    if (event.target.closest("input, textarea, select, video, audio, dialog")) return;
    if (event.key === "ArrowLeft") { step(-1); event.preventDefault(); }
    if (event.key === "ArrowRight") { step(1); event.preventDefault(); }
  });

  // The inspector ---------------------------------------------------------
  const renderInspector = () => {
    const panels = {cast: castPanel, versions: versionsPanel, settings: settingsPanel, auto: autoPanel};
    const isPanel = Boolean(panels[state.panel]);
    const workarea = document.querySelector(".workarea");
    if (workarea) workarea.classList.toggle("panel-open", isPanel);
    const nodes = (panels[state.panel] || scenePanel)();
    $("inspector").classList.toggle("scene-mode", !isPanel);
    $("inspector").replaceChildren(...nodes.flat().filter(Boolean));
  };

  // What depends on the server but sits beside fields being typed in.
  const renderInspectorDerived = () => {
    if (state.panel) return;  // Its inputs may hold what's being typed.
    const item = scene();
    const set = (id, text) => { const node = $(id); if (node) node.textContent = text; };
    set("prompt-hint", item.hint);
    set("added-seconds", item.added);
    set("seconds-label", `Length: ${item.seconds} s`);
    const check = $("checklist");
    if (check) check.replaceWith(checklist(item));
    syncInspectorGenerate();
    const previews = $("tag-hint");
    if (previews) previews.replaceWith(tagHintNode(item));
    const fit = $("fit-preview");
    if (fit && item.video) fit.replaceWith(fitPreview(item));
  };

  const checklist = (item) => (item.issues.length
    ? h("div", {className: "needs", id: "checklist"}, h("ul", {}, item.issues.map((issue) => h("li", {}, issue))))
    : h("div", {className: "ready", id: "checklist"}, item.kind === "media" ? "Ready to prepare." : "Ready to generate."));

  const insertIntoPrompt = (snippet) => {
    const fromMediaTab = state.inspectorTab === "media";
    if (fromMediaTab) {
      state.inspectorTab = "prompt";
      renderInspector();
    }
    const textarea = document.querySelector("#inspector textarea");
    if (!textarea) return;
    const start = fromMediaTab ? textarea.value.length : (textarea.selectionStart ?? textarea.value.length);
    const end = fromMediaTab ? start : (textarea.selectionEnd ?? textarea.value.length);
    const before = textarea.value.slice(0, start);
    const after = textarea.value.slice(end);
    const padBefore = before.length && !/\s$/.test(before) ? " " : "";
    const padAfter = after.length && !/^\s|^[.,;!?]/.test(after) ? " " : "";
    const next = `${before}${padBefore}${snippet}${padAfter}${after}`;
    textarea.value = next;
    edit("prompt", next);
    textarea.focus();
    const cursor = (before + padBefore + snippet).length;
    textarea.setSelectionRange(cursor, cursor);
  };

  const tagHintNode = (item) => {
    if (!item.tags.length) {
      return h("p", {className: "small muted", id: "tag-hint"}, "Tick cast or add an image, video or audio.");
    }
    const chips = item.tags.map((entry) => {
      const match = entry.match(/<Picture \d+>/);
      const tag = match ? match[0] : entry;
      return button(`+ ${entry}`, () => insertIntoPrompt(tag), "tag-chip",
        {title: `Insert ${tag} into prompt`});
    });
    return h("div", {className: "tag-bar", id: "tag-hint"},
      h("span", {className: "small muted"}, `In the prompt (click to insert): ${item.tags.join(", ")}`),
      h("div", {className: "chips"}, chips));
  };

  const textField = (label, field, attributes = {}, tag = "input") => h("label", {className: "field"}, label,
    h(tag, {value: scene()[field] ?? "", oninput: (event) => {
      edit(field, event.target.value);
      if (field === "title") {
        $("scene-heading").textContent = `Scene ${scene().number} · ${event.target.value || "Untitled scene"}`;
        const title = document.querySelector(`#strip [data-scene="${scene().id}"] .title`);
        if (title) title.textContent = event.target.value || "Untitled scene";
      }
    }, ...attributes}));

  const choose = async (fields) => {
    const url = sceneUrl();  // Taken first: the selection can move during the wait.
    await flush();
    await saves;
    const data = await act(() => request("PATCH", url, fields));
    if (data) applyBoard(data);
  };

  const scenePanel = () => {
    const item = scene();
    if (item.kind === "media") return mediaScenePanel(item);
    const [low, high] = state.boot.seconds;
    const previousPrompt = state.previousPrompts.get(item.id);
    const canUndoPrompt = previousPrompt !== undefined && previousPrompt !== item.prompt;
    const tab = state.inspectorTab || "prompt";
    const tabBtn = (id, label) => h("button", {
      type: "button",
      role: "tab",
      className: "inspector-tab",
      "aria-selected": String(tab === id),
      onclick: () => { state.inspectorTab = id; renderInspector(); },
    }, label);

    const headerBar = h("div", {className: "inspector-head"},
      h("div", {className: "inspector-tabs", role: "tablist", "aria-label": "Inspector sections"},
        tabBtn("prompt", "Prompt & Flow"),
        tabBtn("media", item.mode === "Text to video" ? "Media & Cast" : `Media & Cast (${item.pictures.length})`),
        tabBtn("timing", "Timing & Seed")),
      checklist(item));

    const titleField = textField("Scene title", "title", {maxLength: 120});

    const promptTab = [
      h("div", {className: "column"}, [
        titleField,
        h("div", {className: item.number > 1 ? "split" : ""}, [
          item.number > 1 && h("fieldset", {}, h("legend", {}, "How does this scene start?"),
            h("div", {className: "segmented two"},
              chip(`Continue scene ${item.number - 1}`, item.link === "flow", () => choose({link: "flow"})),
              chip("Start fresh", item.link === "cut", () => choose({link: "cut"})))),
          h("fieldset", {}, h("legend", {}, "Conditioning"),
            h("div", {className: "segmented three"},
              chip("Text", item.mode === "Text to video", () => choose({mode: "Text to video"})),
              chip(item.flow ? "Last frame" : "Frames", item.mode === "First / last frame", () => choose({mode: "First / last frame"})),
              chip(item.flow ? "References (recommended)" : "References", item.mode === "Reference media", () => choose({mode: "Reference media"})))),
        ].filter(Boolean)),
        item.mode === "Reference media" ? tagHintNode(item) : null,
        textField("Prompt", "prompt", {rows: 9, placeholder: "Describe what happens, the camera and the sound."}, "textarea"),
        h("div", {className: "row"},
          h("span", {className: "small muted", id: "prompt-hint"}, item.hint || item.context),
          h("div", {className: "chips"},
            canUndoPrompt && button("Undo rewrite", () => {
              state.previousPrompts.delete(item.id);
              edit("prompt", previousPrompt);
              renderInspector();
            }, "btn sm"),
            state.boot.enhance && button("Improve with the prompt writer", enhance, "btn sm"))),
      ]),
    ];

    const mediaTab = [
      h("div", {className: "column"}, [
        item.mode === "Text to video"
          ? h("p", {className: "small muted"}, "Switch Conditioning to Frames or References under Prompt & Flow to attach images, cast, or audio.")
          : null,
        item.mode === "First / last frame" && h("div", {className: "split"},
          item.flow ? h("p", {className: "small muted"},
            `No first frame. This scene opens on scene ${item.number - 1}'s last frames. To open on a picture of your own, choose Start fresh.`)
            : mediaSlot("First frame", "first_image", {optional: true}),
          mediaSlot(item.flow ? "Last frame (required)" : "Last frame", "last_image", {optional: !item.flow})),
        item.mode === "Reference media" && [
          state.board.cast.length ? h("fieldset", {}, h("legend", {}, "Cast in this scene"),
            h("div", {className: "chips"}, state.board.cast.map((member) => {
              const on = item.cast.includes(member.name);
              return chip(member.name, on, () => choose({cast: on ? item.cast.filter((name) => name !== member.name) : [...item.cast, member.name]}));
            }))) : h("p", {className: "small muted"}, "No cast yet: add people under Cast."),
          castPictures(item),
          h("div", {className: "split"},
            mediaSlot("Extra images", "ref_images", {multiple: true}),
            mediaSlot("Reference videos (up to 3)", "ref_videos", {multiple: true})),
        ],
        item.mode !== "Text to video" && mediaSlot(
          item.mode === "First / last frame" ? "Voices (up to 3): <Audio 1> to <Audio 3>" : "Reference audio (up to 3)",
          "ref_audios", {multiple: true}),
      ]),
    ];

    const timingTab = [
      h("div", {className: "column"}, [
        h("label", {className: "field"}, h("span", {id: "seconds-label"}, `Length: ${item.seconds} s`),
          h("input", {type: "range", min: low, max: high, step: 1, value: item.seconds, oninput: (event) => {
            edit("seconds", Number(event.target.value));
            $("seconds-label").textContent = `Length: ${event.target.value} s`;
          }}),
          h("span", {className: "small muted", id: "added-seconds", style: "font-weight:400"}, item.added)),
        h("div", {className: "split"},
          h("label", {className: "field"}, "Frames shared with the previous scene",
            h("select", {disabled: !item.flow, onchange: (event) => edit("overlap", Number(event.target.value))},
              state.boot.overlaps.map((frames) => h("option", {value: frames, selected: frames === item.overlap}, `${frames} frames`)))),
          h("label", {className: "field"}, "Takes per click",
            h("select", {onchange: (event) => edit("takes_per_run", Number(event.target.value))},
              Array.from({length: state.boot.max_takes}, (_, i) => h("option", {value: i + 1, selected: i + 1 === item.takes_per_run}, String(i + 1)))))),
        h("label", {className: "field"}, "Seed (-1 is random)",
          h("input", {type: "number", step: 1, id: "scene-seed", value: item.seed,
            onchange: (event) => {
              const seed = Number.parseInt(event.target.value, 10);
              edit("seed", Number.isNaN(seed) ? -1 : seed);
            }})),
      ]),
    ];

    const footer = h("div", {className: "inspector-footer"},
      h("div", {className: "small muted"},
        h("strong", {style: "color:var(--text)"}, `${item.seconds}s`),
        ` · ${item.added || item.runs_as}`),
      generateSceneButton(item, "inspector-generate-scene"));

    const body = tab === "media" ? mediaTab : (tab === "timing" ? timingTab : promptTab);
    return [headerBar, ...body, footer];
  };

  // The ticked cast, first in the prompt's picture order; the scene's own images follow below.
  const castPictures = (item) => {
    const ticked = item.pictures.slice(0, item.pictures.length - item.ref_images.length);
    return ticked.length ? h("div", {className: "tiles"}, ticked.map((picture) => {
      const match = picture.tag.match(/<Picture \d+>/);
      const tag = match ? match[0] : picture.tag;
      return h("div", {className: "tile"},
        h("img", {src: picture.thumb, alt: picture.tag, loading: "lazy",
          title: `Click to insert ${tag} into prompt`, onclick: () => insertIntoPrompt(tag)}),
        button(picture.tag, () => insertIntoPrompt(tag), "tag", {title: `Insert ${tag} into prompt`}));
    })) : null;
  };

  // A scene from Media ----------------------------------------------------
  const FITS = [
    ["resize", "Resize", "The whole frame, stretched or squeezed to the storyboard's shape."],
    ["crop", "Crop", "Keeps proportions; the edges that don't fit are cut off, centred."],
  ];
  const seconds = (value) => `${Number(value).toFixed(2)} s`;

  // The first and last frame drawn as the chosen Fit will make them: switching Fit only redraws.
  const fitPreview = (item) => {
    const video = item.video;
    const [width, height] = video.canvas;
    const clip = video.clip;
    if (!clip) return h("div", {id: "fit-preview"});
    const same = clip.width && clip.height && Math.abs(clip.width / clip.height - width / height) < 0.01;
    const frame = (label, url) => h("figure", {className: "fit-frame"},
      h("div", {className: "fit-box", style: `aspect-ratio:${width} / ${height}`},
        h("img", {src: url, alt: `${label} frame, ${video.fit === "crop" ? "cropped" : "resized"} to ${width}×${height}`,
          className: `fit-${video.fit}`})),
      h("figcaption", {className: "small muted"}, label));
    return h("div", {id: "fit-preview", className: "fit-preview"},
      h("div", {className: "fit-frames"},
        frame(`First · ${seconds(video.in)}`, video.stills.first),
        frame(`Last · ${seconds(video.out)}`, video.stills.last)),
      h("p", {className: "small muted"},
        `Storyboard ${width}×${height} · video ${clip.width}×${clip.height}. `,
        same ? "Same shape, so Fit changes nothing."
          : video.fit === "crop" ? "Crop cuts off what's outside the box." : "Resize changes the proportions to fill the box."),
      !same && video.can_reshape && button("Match the storyboard to this video's shape", matchShape, "btn sm",
        {title: "Reshape the storyboard, keeping about its pixel count, so the video fits without stretching or cropping"}));
  };
  const matchShape = async () => {
    const url = sceneUrl("/match-shape");
    await flush();
    await saves;
    const data = await act(() => request("POST", url), "The storyboard now has this video's shape.");
    if (data) applyBoard(data);
  };

  const setFit = (fit) => {
    const item = scene();
    item.video.fit = fit;
    edit("fit", fit);
    document.querySelectorAll("#fit-chips .chip").forEach((node) => node.setAttribute("aria-pressed", String(node.dataset.fit === fit)));
    $("fit-preview").replaceWith(fitPreview(item));
  };

  const rangeField = (label, field, key) => {
    const item = scene();
    const clip = item.video.clip;
    const input = h("input", {type: "number", min: 0, max: clip.seconds, step: 0.04, value: Number(item.video[key]).toFixed(2),
      onchange: (event) => {
        const value = Math.max(0, Math.min(clip.seconds, Number(event.target.value) || 0));
        item.video[key] = value;
        edit(field, field === "video_out" && value >= clip.seconds - 0.001 ? null : value);
      }});
    return h("label", {className: "field"}, label,
      h("div", {className: "row"}, input,
        button("Set from player", () => {
          const player = $("clip-player");
          if (!player) return;
          input.value = player.currentTime.toFixed(2);
          input.dispatchEvent(new Event("change"));
        }, "btn sm", {title: `Use the player's position as ${label}`})));
  };

  const mediaScenePanel = (item) => {
    const video = item.video;
    const clip = video.clip;
    const fromLibrary = state.boot.library && button("Media library", () => openLibrary((path) => libraryVideo(path), "video"), "btn sm");
    const upload = button(clip ? "Replace" : "Upload", async () => {
      const [file] = await pickFiles("video/*", false);
      if (file) uploadVideo(file);
    }, "btn sm");
    const inputs = [
      h("p", {className: "small muted"}, item.context),
      textField("Scene title", "title", {maxLength: 120}),
      h("div", {className: "slot"}, h("div", {className: "slot-head"}, h("span", {}, "Video")),
        h("div", {className: "drop",
          ondragover: (event) => { event.preventDefault(); event.currentTarget.classList.add("over"); },
          ondragleave: (event) => event.currentTarget.classList.remove("over"),
          ondrop: (event) => {
            event.preventDefault();
            event.currentTarget.classList.remove("over");
            const [file] = event.dataTransfer.files;
            if (file) uploadVideo(file);
          }},
        clip ? h("video", {id: "clip-player", src: clip.url, poster: clip.poster, preload: "metadata", controls: true,
          className: "clip-player", "aria-label": clip.name}) : null,
        clip ? h("p", {className: "small muted"},
          `${clip.name} · ${clip.width}×${clip.height} · ${seconds(clip.seconds)}${clip.audio ? "" : " · no sound (plays silent)"}`) : null,
        h("div", {className: "slot-actions"}, upload, fromLibrary))),
      clip && h("div", {className: "split"}, rangeField("In (seconds)", "video_in", "in"), rangeField("Out (seconds)", "video_out", "out")),
      clip && h("p", {className: "small muted", id: "added-seconds"}, item.added),
    ];
    const fitting = clip ? [
      h("fieldset", {}, h("legend", {}, "Fit to the storyboard"),
        h("div", {className: "chip-grid two", id: "fit-chips"}, FITS.map(([value, label, hint]) => {
          const node = chip(label, video.fit === value, () => setFit(value));
          node.dataset.fit = value;
          node.title = hint;
          return node;
        }))),
      fitPreview(item),
      video.continued && h("p", {className: "small muted"},
        `Scene ${item.number + 1} continues from the last ${video.tail_frames} frames (${seconds(video.min_seconds)}), as Fit shows them.`),
    ] : [h("p", {className: "small muted"}, "Choose a video to set its range and how it fits the storyboard.")];
    return [h("div", {className: "scene-columns"},
      h("div", {className: "column"}, inputs),
      h("div", {className: "column"}, fitting, checklist(item)))];
  };

  const uploadVideo = async (file) => {
    const sceneId = state.sceneId;
    const form = new FormData();
    form.append("file", file);
    toast("Uploading…");
    const data = await act(() => request("POST", boardUrl(`/scenes/${sceneId}/video`), form, {form: true}), "Video added.");
    if (data) applyBoard(data, {keepInspector: sceneId !== state.sceneId});
  };
  const libraryVideo = async (source) => {
    const url = sceneUrl("/video");
    await flush();
    await saves;
    const data = await act(() => request("PUT", url, {source}), "Video added.");
    if (data) applyBoard(data);
  };
  const startPrepare = async () => {
    const url = sceneUrl("/prepare");
    await flush();
    await saves;
    const data = await act(() => request("POST", url));
    if (data) followRun(data.run, {started: true});
    renderTakes();
    renderStripActions();
  };

  const addScene = async (afterId = null, kind = null) => {
    await flush();
    const body = {...(afterId ? {after: afterId} : {}), ...(kind ? {kind} : {})};
    const data = await act(() => request("POST", boardUrl("/scenes"), body));
    if (!data) return;
    state.sceneId = data.scene_id;
    state.previewTake = null;
    applyBoard(data);
  };
  const duplicateScene = async () => {
    const url = sceneUrl("/duplicate");
    await flush();
    const data = await act(() => request("POST", url), "Scene duplicated.");
    if (!data) return;
    state.sceneId = data.scene_id;
    applyBoard(data);
  };
  const moveScene = async (offset) => {
    const url = sceneUrl("/move");
    await flush();
    const data = await act(() => request("POST", url, {offset}));
    if (data) applyBoard(data);
  };
  const deleteScene = async () => {
    if (!confirm(`Delete scene ${scene().number}?`)) return;
    dirty.delete(sceneUrl());
    const data = await act(() => request("DELETE", sceneUrl()), "Scene deleted.");
    if (data) applyBoard(data);
  };
  const enhance = async (event) => {
    const control = event.currentTarget;
    control.disabled = true;
    const currentScene = scene();
    const sceneId = currentScene.id;
    const previousPrompt = currentScene.prompt;
    const url = sceneUrl("/enhance");
    await flush();
    await saves;
    const data = await act(() => request("POST", url, {writer: writerSettings()}));
    control.disabled = false;
    if (!data) return;
    state.previousPrompts.set(sceneId, previousPrompt);
    toast("Prompt improved. Edit it further or click Undo rewrite.");
    applyBoard(data);
  };

  // Media -----------------------------------------------------------------
  const ACCEPT = {first_image: "image/*", last_image: "image/*", ref_images: "image/*", ref_videos: "video/*", ref_audios: "audio/*"};
  const pickFiles = (accept, multiple) => new Promise((resolve) => {
    const input = $("file-input");
    input.accept = accept;
    input.multiple = multiple;
    input.value = "";
    input.onchange = () => resolve([...input.files]);
    input.click();
  });
  const uploadMedia = async (field, files) => {
    if (!files.length) return;
    const sceneId = state.sceneId;
    const form = new FormData();
    files.forEach((file) => form.append("files", file));
    toast("Uploading…");
    const data = await act(() => request("POST", boardUrl(`/scenes/${sceneId}/media/${field}`), form, {form: true}), "Added.");
    if (data) applyBoard(data, {keepInspector: sceneId !== state.sceneId});
  };
  const removeMedia = async (field, path) => {
    const data = await act(() => request("DELETE", `${sceneUrl(`/media/${field}`)}?path=${encodeURIComponent(path)}`));
    if (data) applyBoard(data);
  };
  const removeButton = (field, path) => h("button", {type: "button", className: "remove", "aria-label": "Remove", onclick: () => removeMedia(field, path)}, "×");

  const mediaSlot = (label, field, {multiple = false, optional = false} = {}) => {
    const item = scene();
    const value = item[field];
    const list = Array.isArray(value) ? value : value ? [value] : [];
    const pictures = ACCEPT[field] === "image/*";
    const castOffset = Math.max(0, item.pictures.length - item.ref_images.length);
    const tiles = list.map((media, index) => {
      if (field === "ref_audios") {
        const audioTag = `<Audio ${index + 1}>`;
        return h("div", {className: "audio-line"},
          button(audioTag, () => insertIntoPrompt(audioTag), "tag-chip mono",
            {title: `Insert ${audioTag} into prompt`}),
          h("audio", {src: media.url, controls: true, preload: "metadata", "aria-label": media.name}),
          button("Remove", () => removeMedia(field, media.path), "btn sm"));
      }
      if (field === "ref_videos") {
        return h("div", {className: "tile video-tile"},
          h("video", {src: media.url, poster: media.poster, preload: "none", controls: true, muted: true, "aria-label": media.name}),
          removeButton(field, media.path),
          h("span", {className: "tag"}, media.name));
      }
      const picTag = field === "ref_images" ? `<Picture ${castOffset + index + 1}>` : null;
      return h("div", {className: "tile"},
        h("img", {
          src: media.thumb || media.poster, alt: media.name, loading: "lazy",
          ...(picTag ? {title: `Click to insert ${picTag} into prompt`, onclick: () => insertIntoPrompt(picTag)} : {}),
        }),
        removeButton(field, media.path),
        picTag ? button(picTag, () => insertIntoPrompt(picTag), "tag", {title: `Insert ${picTag} into prompt`}) : null);
    });
    const drop = h("div", {className: "drop",
      ondragover: (event) => { event.preventDefault(); drop.classList.add("over"); },
      ondragleave: () => drop.classList.remove("over"),
      ondrop: (event) => {
        event.preventDefault();
        drop.classList.remove("over");
        uploadMedia(field, [...event.dataTransfer.files]);
      },
    },
    tiles.length ? h("div", {className: field === "ref_audios" ? "audio-list" : "tiles"}, tiles) : null,
    h("div", {className: "slot-actions"},
      button(list.length && !multiple ? "Replace" : "Upload", async () => uploadMedia(field, await pickFiles(ACCEPT[field], multiple)), "btn sm"),
      pictures && state.boot.library && button("Media library", () => openLibrary((path) => libraryMedia(sceneUrl(`/media/${field}`), path)), "btn sm")));
    return h("div", {className: "slot"}, h("div", {className: "slot-head"}, h("span", {}, label),
      optional ? h("span", {className: "muted"}, "Optional") : null), drop);
  };
  const libraryMedia = async (url, source) => {
    const data = await act(() => request("PUT", url, {source}), "Added.");
    if (data) applyBoard(data);
  };

  const openLibrary = (onPick, kind = "image", attachLabel = "") => {
    const dialog = $("library");
    let limit = 24;
    let selectedPath = null;
    const noun = kind === "video" ? "videos" : "images";
    const grid = h("div", {className: "library-grid"});
    const status = h("p", {className: "small muted", role: "status"}, "Loading…");
    const pick = (path) => { dialog.close(); onPick(path); };
    const attach = attachLabel ? button(attachLabel, () => pick(selectedPath), "btn primary sm", {disabled: true}) : null;
    const more = button(`Show more ${noun}`, () => { limit += 24; load(); }, "btn sm", {hidden: true});
    const query = h("input", {type: "search", placeholder: "Filename, tags or settings",
      onkeydown: (event) => { if (event.key === "Enter") { event.preventDefault(); load(); } }});
    const load = async () => {
      const data = await act(() => request("GET", `${API}/library?kind=${kind}&limit=${limit}&query=${encodeURIComponent(query.value)}`));
      if (!data) return;
      const hint = attach ? " Select a thumbnail, then confirm the attachment."
        : onPick ? " Select a thumbnail to use it." : "";
      status.textContent = `${data.total} matching ${noun}.${hint}`;
      more.hidden = !data.more;
      if (attach) {
        selectedPath = null;
        attach.disabled = true;
      }
      grid.replaceChildren(...data.items.map((row) => {
        const thumbnail = h("img", {src: row.thumb, alt: onPick ? "" : row.caption, loading: "lazy"});
        if (!onPick) return thumbnail;
        return button(thumbnail, (event) => {
          if (!attach) return pick(row.path);
          grid.querySelector('[aria-pressed="true"]')?.setAttribute("aria-pressed", "false");
          event.currentTarget.setAttribute("aria-pressed", "true");
          selectedPath = row.path;
          attach.disabled = false;
        }, "", {"aria-label": row.caption, "aria-pressed": attach ? "false" : null});
      }));
    };
    const area = visibleArea();
    dialog.style.marginTop = `${area.top + 16}px`;
    dialog.style.maxHeight = `${area.bottom - area.top - 32}px`;
    dialog.replaceChildren(
      h("div", {className: "library-head"},
        h("label", {className: "field grow"}, `Search library ${noun}`, query),
        button("Search", load), button("Close", () => dialog.close())),
      attach ? h("div", {className: "row"}, status, attach) : status, grid, more);
    dialog.showModal();
    load();
  };

  const browseGlobalMedia = () => {
    const item = scene();
    if (item.kind === "media" || item.mode !== "Reference media") return openLibrary(null);
    const url = sceneUrl("/media/ref_images");
    openLibrary((path) => libraryMedia(url, path), "image", `Add image to Scene ${item.number}`);
  };

  // Panels ----------------------------------------------------------------
  const panelHead = (title) => h("div", {className: "panel-head"},
    h("h2", {}, title),
    button(`← Back to scene ${scene().number}`, closePanel, "btn sm"));

  const castPanel = () => {
    const name = h("input", {maxLength: 80, placeholder: "Mei"});
    const add = async (fill) => {
      if (!name.value.trim()) { toast("Give the person a name first.", true); return; }
      const form = new FormData();
      form.append("name", name.value);
      fill(form);
      const data = await act(() => request("POST", boardUrl("/cast"), form, {form: true}), "Added to the cast.");
      if (data) applyBoard(data);
    };

    const castRows = state.board.cast.length ? state.board.cast.map((member) => {
      const usedIn = scenes().filter((s) => (s.cast || []).includes(member.name)).map((s) => `Scene ${s.number}`);
      return h("div", {className: "list-row"},
        h("img", {src: member.thumb, alt: "", loading: "lazy"}),
        h("div", {className: "grow"},
          h("strong", {}, member.name),
          h("div", {className: "small muted"}, usedIn.length ? `Used in ${usedIn.join(", ")}` : "Not yet assigned to a scene")),
        button("Remove", async () => {
          const data = await act(() => request("DELETE", boardUrl(`/cast/${encodeURIComponent(member.name)}`)), "Removed from the cast.");
          if (data) applyBoard(data);
        }, "btn sm danger"));
    }) : [h("p", {className: "muted"}, "No cast yet.")];

    const castColumn = h("div", {className: "column"}, [
      h("h3", {}, "Cast Roster"),
      h("p", {className: "small muted"}, "Add a picture of each person. Scenes using Reference media can then tick them, so they look the same in every scene."),
      ...castRows,
      h("label", {className: "field"}, "Name", name),
      h("div", {className: "chips"},
        button("Upload picture", async () => {
          const [file] = await pickFiles("image/*", false);
          if (file) add((form) => form.append("file", file));
        }, "btn quiet"),
        state.boot.library && button("Media library", () => openLibrary((path) => add((form) => form.append("source", path))), "btn")),
    ]);

    const collectProjectMedia = () => {
      const byKey = new Map();
      const addAsset = (kind, media, sceneNum, roleLabel) => {
        if (!media) return;
        const key = `${kind}:${media.path || media.url || media.name}`;
        if (!byKey.has(key)) {
          byKey.set(key, {
            kind,
            name: media.name || "Untitled asset",
            thumb: media.thumb || media.poster || "",
            url: media.url || "",
            path: media.path || "",
            usages: [],
          });
        }
        const entry = byKey.get(key);
        const tag = `Scene ${sceneNum} (${roleLabel})`;
        if (!entry.usages.includes(tag)) entry.usages.push(tag);
      };
      scenes().forEach((s) => {
        if (s.first_image) addAsset("image", s.first_image, s.number, "First frame");
        if (s.last_image) addAsset("image", s.last_image, s.number, "Last frame");
        (s.ref_images || []).forEach((m) => addAsset("image", m, s.number, "Ref image"));
        (s.ref_videos || []).forEach((m) => addAsset("video", m, s.number, "Ref video"));
        if (s.video?.clip) addAsset("video", s.video.clip, s.number, "Media clip");
        (s.ref_audios || []).forEach((m) => addAsset("audio", m, s.number, "Audio"));
      });
      return [...byKey.values()];
    };

    const projectMedia = collectProjectMedia();
    const libGrid = h("div", {className: "project-media-grid"});
    const libStatus = h("p", {className: "small muted"});
    const libQuery = h("input", {type: "search", placeholder: "Filter project media by name or scene…"});
    let libKind = "all";

    const renderProjectMedia = () => {
      const q = libQuery.value.trim().toLowerCase();
      const filtered = projectMedia.filter((item) => {
        if (libKind !== "all" && item.kind !== libKind) return false;
        if (!q) return true;
        return item.name.toLowerCase().includes(q) || item.usages.some((u) => u.toLowerCase().includes(q));
      });
      libStatus.textContent = filtered.length
        ? `${filtered.length} asset(s) in "${state.board.name}"`
        : "No matching assets in this storyboard.";
      libGrid.replaceChildren(...filtered.map((item) => {
        const preview = item.kind === "audio"
          ? h("div", {className: "project-media-thumb mono small"}, "AUDIO")
          : item.thumb
            ? h("img", {src: item.thumb, alt: item.name, loading: "lazy"})
            : h("div", {className: "project-media-thumb mono small"}, item.kind.toUpperCase());
        return h("div", {className: "project-media-card"},
          preview,
          h("div", {className: "project-media-info"},
            h("strong", {title: item.name}, item.name),
            h("span", {className: "small muted"}, item.usages.join(" · "))),
          item.kind === "audio" && item.url
            ? h("audio", {src: item.url, controls: true, preload: "none", style: "width:100%;height:32px"})
            : null);
      }).filter(Boolean));
    };

    libQuery.oninput = renderProjectMedia;
    const filterChip = (kind, label) => chip(label, libKind === kind, () => {
      libKind = kind;
      filterBar.querySelectorAll(".chip").forEach((c, idx) => {
        const keys = ["all", "image", "video", "audio"];
        c.setAttribute("aria-pressed", String(keys[idx] === libKind));
      });
      renderProjectMedia();
    });
    const counts = {
      all: projectMedia.length,
      image: projectMedia.filter((x) => x.kind === "image").length,
      video: projectMedia.filter((x) => x.kind === "video").length,
      audio: projectMedia.filter((x) => x.kind === "audio").length,
    };
    const filterBar = h("div", {className: "chips"},
      filterChip("all", `All (${counts.all})`),
      filterChip("image", `Images (${counts.image})`),
      filterChip("video", `Video Clips (${counts.video})`),
      filterChip("audio", `Audio (${counts.audio})`));
    renderProjectMedia();

    const mediaColumn = h("div", {className: "column"}, [
      h("div", {className: "row"},
        h("h3", {}, "Project Media Library"),
        state.boot.library && button("Browse global Media", browseGlobalMedia, "btn sm")),
      filterBar,
      h("div", {className: "field"}, libQuery),
      libStatus,
      libGrid,
    ]);

    return [
      panelHead("Cast & Media Library"),
      h("div", {className: "scene-columns"}, castColumn, mediaColumn),
    ];
  };

  const versionsPanel = () => {
    const label = h("input", {maxLength: 80});
    return [
      panelHead("Versions"),
      h("p", {className: "small muted"}, "Save the scene list to come back to, or branch it into a new storyboard."),
      h("label", {className: "field"}, "Version name", label),
      button("Save version", async () => {
        await flush();
        const data = await act(() => request("POST", boardUrl("/versions"), {label: label.value}), "Version saved.");
        if (data) applyBoard(data);
      }, "btn quiet"),
      state.board.versions.map((version) => h("div", {className: "list-row"},
        h("div", {className: "grow"}, version.label),
        button("Branch", async () => {
          const data = await act(() => request("POST", boardUrl(`/versions/${version.id}/branch`)), "Branched into a new storyboard.");
          if (data) adoptBoard(data);
        }, "btn sm"))),
    ];
  };

  const AUTO_LABELS = {waiting: "Waiting", rendering: "Rendering", exporting: "Exporting", exported: "Exported",
    incomplete: "Not exported", failed: "Failed", stopped: "Stopped"};
  const autoSummary = () => {
    const auto = state.auto;
    const current = auto.items.find((item) => item.id === auto.current);
    return `Auto render ${Math.min(auto.finished + 1, auto.items.length)}/${auto.items.length}`
      + (current ? ` · ${current.name}` : "") + (auto.stopping ? " · stopping" : "");
  };
  const autoBoardNote = (board) => {
    const notes = [board.pending
      ? `${board.pending} of ${board.scenes} scene${board.scenes === 1 ? "" : "s"} to render, then export`
      : "Every scene is done: exports only"];
    if (board.needs_input.length) {
      notes.push(`scene ${board.needs_input.join(", ")} need${board.needs_input.length === 1 ? "s" : ""} input, so it won't be exported`);
    }
    return notes.join(" · ");
  };
  const autoPanel = () => {
    const auto = state.auto;
    const nodes = [
      panelHead("Auto render"),
      h("p", {className: "small muted"}, "Renders each selected storyboard's remaining scenes in turn. Once every scene of a storyboard is done, its full video is exported to Media and the next storyboard starts. It runs on the server, so this page can be closed; Jobs lists each storyboard's run."),
    ];
    if (auto) {
      nodes.push(h("h3", {className: "small"}, auto.active ? "Rendering" : "Last auto render"),
        auto.items.map((item) => {
          const live = item.id === auto.current && auto.status;
          return h("div", {className: "list-row"},
            h("div", {className: "grow"},
              h("div", {}, item.name),
              (live || item.detail) ? h("div", {className: "small muted", id: item.id === auto.current ? "auto-status" : null},
                live ? auto.status : item.detail) : null),
            h("span", {className: `auto-badge ${item.state}`}, AUTO_LABELS[item.state] || item.state),
            button("Open", () => openBoard(item.id), "btn sm", {disabled: item.id === state.board.id}));
        }));
      if (auto.error) nodes.push(h("div", {className: "warning", role: "status"}, auto.error));
      if (auto.active) {
        nodes.push(h("div", {className: "chips"},
          button(auto.stopping ? "Stopping…" : "Stop auto render", stopAuto, "btn danger", {disabled: auto.stopping})));
        return nodes;
      }
    }
    const boards = state.autoBoards;
    if (!boards) return [...nodes, h("p", {className: "muted"}, "Loading storyboards…")];
    const known = new Set(boards.map((board) => board.id));
    for (const id of [...state.autoSelected]) if (!known.has(id)) state.autoSelected.delete(id);
    const chosen = boards.filter((board) => state.autoSelected.has(board.id));
    const toggle = (id, on) => {
      if (on) state.autoSelected.add(id); else state.autoSelected.delete(id);
      renderInspector();
    };
    nodes.push(
      h("div", {className: "row"},
        h("h3", {className: "small"}, "Choose storyboards"),
        h("div", {className: "chips"},
          button("Select all with scenes left", () => {
            boards.filter((board) => board.runnable).forEach((board) => state.autoSelected.add(board.id));
            renderInspector();
          }, "btn sm"),
          button("Clear", () => { state.autoSelected.clear(); renderInspector(); }, "btn sm", {disabled: !chosen.length}))),
      boards.map((board) => h("label", {className: "list-row auto-pick"},
        h("input", {type: "checkbox", checked: state.autoSelected.has(board.id),
          onchange: (event) => toggle(board.id, event.target.checked)}),
        h("div", {className: "grow"},
          h("div", {}, board.name),
          h("div", {className: board.needs_input.length ? "small needs" : "small muted"}, autoBoardNote(board))))),
      h("p", {className: "small muted"}, "Storyboards render in the order listed. Scenes that still need input are skipped, and their storyboard isn't exported."),
      h("div", {className: "chips"},
        button(chosen.length ? `Render and export ${chosen.length} storyboard${chosen.length === 1 ? "" : "s"}` : "Select storyboards to render",
          startAuto, "btn primary", {disabled: !chosen.length})),
    );
    return nodes;
  };

  const WRITER_KEY = "h3-storyboard-writer";
  const writerSettings = () => {
    try { return JSON.parse(stored.get(WRITER_KEY)) || {}; } catch { return {}; }
  };
  const TRANSLATION_MODEL_KEY = "h3-storyboard-subtitle-model";
  const subtitleLanguage = () => {
    const language = state.board.subtitle_language;
    const supported = state.boot.subtitle_languages;
    return supported.includes(language) ? language : supported[0];
  };
  const translationOptions = () => {
    const model = stored.get(TRANSLATION_MODEL_KEY);
    return {
      language: subtitleLanguage(),
      model: state.boot.subtitle_models.includes(model) ? model : state.boot.subtitle_models[0],
      gemini_api_key: writerSettings().gemini_api_key || "",
    };
  };
  const bindSubtitleControls = () => {
    const language = $("subtitle-language");
    language.replaceChildren(...state.boot.subtitle_languages.map((name) => h("option", {value: name}, name)));
    language.onchange = () => {
      state.board.subtitle_language = language.value;
      queueSave("PUT", boardUrl("/subtitle-settings"), "subtitle_language", language.value);
      renderPreview();
    };
    const model = $("subtitle-model");
    model.replaceChildren(...state.boot.subtitle_models.map((name) => h("option", {value: name}, name)));
    model.value = state.boot.subtitle_models.includes(stored.get(TRANSLATION_MODEL_KEY))
      ? stored.get(TRANSLATION_MODEL_KEY) : state.boot.subtitle_models[0];
    model.onchange = () => stored.set(TRANSLATION_MODEL_KEY, model.value);
    const key = $("subtitle-api-key");
    key.closest("details").ontoggle = () => { key.value = writerSettings().gemini_api_key || ""; };
    key.onchange = () => stored.set(WRITER_KEY, JSON.stringify({...writerSettings(), gemini_api_key: key.value}));

    $("translate-scene").onclick = translateScene;
    $("translate-board").onclick = translateBoard;
    $("show-subtitles").onchange = (event) => syncSubtitleVisibility(event.target.checked);
    $("show-translation").onchange = (event) => {
      state.showTranslation = event.target.checked;
      renderPreview();
    };
    $("export-translation").onchange = (event) => {
      state.exportTranslation = event.target.checked;
      stored.set("h3-storyboard-translate-subtitles", String(state.exportTranslation));
    };
    $("burn-subtitles").onchange = (event) => { state.burnSubtitles = event.target.checked; };
  };
  const writerForm = () => {
    const values = writerSettings();
    const update = (name, value) => { stored.set(WRITER_KEY, JSON.stringify({...writerSettings(), [name]: value})); };
    const writer = state.boot.writer;
    const input = (label, name, attributes = {}) => h("label", {className: "field"}, label,
      h("input", {value: values[name] ?? "", oninput: (event) => update(name, event.target.value), ...attributes}));
    const backend = values.prompt_writer_backend || "Gemini";
    const model = values.gemini_prompt_model || writer.gemini_default;
    const geminiGroup = h("div", {className: "details-body", style: "padding-top:0", hidden: backend !== "Gemini"},
      h("label", {className: "field"}, "Gemini model",
        h("select", {onchange: (event) => update("gemini_prompt_model", event.target.value)},
          writer.gemini_models.map((name) => h("option", {selected: name === model}, name)))),
      input("Gemini API key (optional)", "gemini_api_key", {type: "password", autocomplete: "off"}));
    const openaiGroup = h("div", {className: "details-body", style: "padding-top:0", hidden: backend === "Gemini"},
      input("OpenAI-compatible model", "openai_model", {placeholder: writer.openai_models[0]}),
      input("OpenAI-compatible API key (optional)", "openai_api_key", {type: "password", autocomplete: "off"}),
      input("Base URL (optional)", "openai_base_url"));
    return h("details", {},
      h("summary", {}, "Prompt writer"),
      h("div", {className: "details-body"},
        h("p", {className: "small muted"}, "Rewrite / extend uses Gemini or an OpenAI-compatible service. Leave a key empty to use the server's key."),
        h("label", {className: "field"}, "Writer service",
          h("select", {onchange: (event) => {
            const next = event.target.value;
            update("prompt_writer_backend", next);
            geminiGroup.hidden = next !== "Gemini";
            openaiGroup.hidden = next === "Gemini";
          }},
            ["Gemini", "OpenAI-compatible"].map((name) => h("option", {selected: name === backend}, name)))),
        geminiGroup,
        openaiGroup));
  };

  const applyProfileNow = async () => {
    state.confirmProfile = false;
    const data = await act(() => request("POST", boardUrl("/profile")),
      "Applied your current MiniMax H3 settings. Scenes made with the old ones are out of date.");
    if (data) applyBoard(data);
  };

  const settingsPanel = () => {
    const name = h("input", {value: state.board.name, maxLength: 120});
    return [
      panelHead("Storyboard settings"),
      state.board.settings && h("p", {className: "small muted"}, `Uses: ${state.board.settings}`),
      h("label", {className: "check-label"},
        h("input", {type: "checkbox", checked: state.board.auto_subtitles, onchange: async (event) => {
          const input = event.target;
          input.disabled = true;
          const data = await act(() => request("PUT", boardUrl("/subtitle-settings"), {auto_subtitles: input.checked}));
          if (data) applyBoard(data);
          else { input.checked = !input.checked; input.disabled = false; }
        }}), "Automatically generate word-aligned subtitles after each take"),
      h("p", {className: "small muted"}, "Audio transcription runs on the GPU endpoint. Dialogue tagged in the prompt gets approximate subtitles until audio is analyzed."),
      state.boot.gpu && [
        button("Use my current MiniMax H3 settings", () => {
          if (state.board.done > 0 && !state.confirmProfile) {
            state.confirmProfile = true;
            renderInspector();
            return;
          }
          applyProfileNow();
        }),
        state.confirmProfile ? h("div", {className: "confirm", role: "alertdialog", "aria-label": "Apply MiniMax H3 settings"},
          h("p", {}, `Applying your current MiniMax H3 settings will mark ${state.board.done} generated scene(s) out of date.`),
          button("Apply settings", applyProfileNow, "btn primary"),
          button("Cancel", () => { state.confirmProfile = false; renderInspector(); }))
          : h("p", {className: "small muted"}, "Scenes made with the old settings will need generating again."),
      ],
      h("label", {className: "field"}, "Storyboard name", name),
      button("Rename", async () => {
        const data = await act(() => request("PATCH", boardUrl(), {name: name.value}), "Renamed.");
        if (!data) return;
        state.boot.boards.find((row) => row.id === data.id).name = data.name;
        applyBoard(data);
      }),
      button("Duplicate storyboard", async () => {
        await flush();
        const data = await act(() => request("POST", boardUrl("/duplicate")), "Duplicated.");
        if (data) adoptBoard(data);
      }),
      button("Delete storyboard", async () => {
        if (!confirm(`Delete "${state.board.name}" and all its takes?`)) return;
        dirty.clear();
        const data = await act(() => request("DELETE", boardUrl()), "Storyboard deleted.");
        if (!data) return;
        state.boot.boards = state.boot.boards.filter((row) => row.id !== state.board.id);
        if (state.boot.boards.length) openBoard(state.boot.boards[0].id);
        else showStart();
      }, "btn danger"),
      writerForm(),
    ];
  };

  // Long work -------------------------------------------------------------
  const startGeneration = async (body) => {
    await flush();
    await saves;
    const data = await act(() => request("POST", boardUrl("/generate"), body));
    if (data) followRun(data.run, {started: true});
    renderStage();
    renderStrip();
    renderStripActions();
  };
  const exportToMedia = async () => {
    await flush();
    await saves;
    const hasSubtitles = subtitleTakes().length > 0;
    const data = await act(() => request("POST", boardUrl("/join"), {
      include_outdated: state.includeOutdated,
      export: true,
      burn_subtitles: state.burnSubtitles && hasSubtitles,
      ...(state.exportTranslation && hasSubtitles ? {translation: translationOptions()} : {}),
    }));
    if (data) followRun(data.run, {started: true});
    renderStripActions();
  };
  const startSubtitles = async () => {
    const data = await act(() => request("POST", boardUrl("/subtitles")));
    if (data) {
      followRun(data.run, {started: true});
      if (!data.run.active) await refreshBoard();
    }
    renderStage();
    renderStripActions();
  };
  const interrupt = async () => {
    const data = await act(() => request("POST", boardUrl("/interrupt")));
    if (data) toast(data.message);
  };

  // Auto render ------------------------------------------------------------
  const AUTO_POLL_MS = 3000;
  const autoKey = (auto) => auto
    ? JSON.stringify([auto.id, auto.active, auto.stopping, auto.current, auto.items.map((item) => [item.state, item.detail])])
    : "";
  const loadAuto = async () => {
    let data;
    try {
      data = await request("GET", `${API}/auto`);
    } catch (error) {
      toast(error.message, true);
      return;
    }
    state.autoBoards = data.boards;
    followAuto(data.auto, {render: true});
  };
  const followAuto = (auto, {render = false} = {}) => {
    const before = state.auto;
    const changed = autoKey(auto) !== autoKey(before);
    state.auto = auto;
    if (auto?.active && !state.autoTimer) state.autoTimer = setTimeout(pollAuto, AUTO_POLL_MS);
    if (!state.board) return;
    const sameQueue = before && auto && before.id === auto.id;
    if (sameQueue && before.active && !auto.active) announceAuto(auto);
    // The page moves on with the queue while it shows the storyboard the queue just left,
    // but not away from a new storyboard being written.
    if (sameQueue && auto.active && auto.current && before.current !== auto.current
        && state.board.id === before.current && !$("editor").hidden) {
      openBoard(auto.current);
    } else if (changed && auto && (auto.current === state.board.id || before?.current === state.board.id)
               && !state.run?.active) {
      // A run the queue started here since this page last looked.
      refreshBoard();
    }
    if (render || changed || Boolean(before?.active) !== Boolean(auto?.active)) {
      renderTopbar();
      if (state.panel === "auto") renderInspector();
    } else {
      const summary = $("auto-state");
      if (summary && auto?.active) summary.textContent = autoSummary();
      const status = $("auto-status");
      if (status && auto?.status) status.textContent = auto.status;
    }
  };
  const pollAuto = async () => {
    state.autoTimer = null;
    let auto;
    try {
      ({auto} = await request("GET", `${API}/auto?boards=false`));
    } catch {
      state.autoTimer = setTimeout(pollAuto, AUTO_POLL_MS * 2);
      return;
    }
    const wasActive = state.auto?.active;
    followAuto(auto);
    // Finished: the list's counts changed.
    if (wasActive && !auto?.active && state.panel === "auto") loadAuto();
  };
  const announceAuto = (auto) => {
    const count = (name) => auto.items.filter((item) => item.state === name).length;
    const parts = [`${count("exported")} exported`];
    if (count("incomplete")) parts.push(`${count("incomplete")} not exported (scenes need input)`);
    if (count("failed")) parts.push(`${count("failed")} failed`);
    if (count("stopped")) parts.push(`${count("stopped")} stopped`);
    toast(`Auto render finished: ${parts.join(", ")}.`, Boolean(count("failed") || auto.error));
  };
  const startAuto = async () => {
    await flush();
    await saves;
    const ids = state.autoBoards.filter((board) => state.autoSelected.has(board.id)).map((board) => board.id);
    const data = await act(() => request("POST", `${API}/auto`, {ids}), "Auto render started.");
    if (!data) return;
    state.autoSelected.clear();
    followAuto(data.auto, {render: true});
    const first = data.auto.current || data.auto.items[0]?.id;
    if (first && first !== state.board.id) openBoard(first);
  };
  const stopAuto = async () => {
    const data = await act(() => request("POST", `${API}/auto/stop`));
    if (!data) return;
    toast(data.message);
    followAuto(data.auto, {render: true});
  };

  // Runs already reported in this tab, so a reload or a board switch doesn't repeat them.
  const ANNOUNCED_KEY = "h3-storyboard-announced";
  const announced = new Set((() => {
    try { return JSON.parse(stored.get(ANNOUNCED_KEY)) || []; } catch { return []; }
  })());
  const followRun = (run, {started = false} = {}) => {
    const changed = run?.id !== state.run?.id;
    state.run = run;
    syncInspectorGenerate();
    if (changed) state.seenRevision = started || run?.active ? 0 : run?.revision ?? null;
    if (run?.active && !state.pollTimer) state.pollTimer = setTimeout(poll, POLL_MS);
    // A run that ended while this page wasn't watching still reports, once.
    if (run && !run.active && !announced.has(run.id)) announce(run);
  };
  const announce = (run) => {
    announced.add(run.id);
    stored.set(ANNOUNCED_KEY, JSON.stringify([...announced].slice(-50)));
    if (run.error) toast(run.error, true);
    else if (run.message || run.status) toast(run.message || run.status);
    if (run.result_url && !$("full-video").querySelector(".full-video-viewport")) showJoinedVideo(run);
  };
  const poll = async () => {
    state.pollTimer = null;
    if (!state.board) return;
    const boardId = state.board.id;
    let run = null;
    try {
      ({run} = await request("GET", boardUrl("/run")));
    } catch {
      // The next poll tries again; a run doesn't depend on this page.
      state.pollTimer = setTimeout(poll, POLL_MS * 2);
      return;
    }
    if (state.board.id !== boardId) return;
    const revised = run && run.revision !== state.seenRevision;
    const sceneChanged = run?.scene_id !== state.run?.scene_id;
    followRun(run);
    if (revised) {
      state.seenRevision = run.revision;
      await refreshBoard();
    } else {
      // Only what changed: a rebuilt button loses a click landing on it.
      const status = $("run-status");
      if (status) status.textContent = run.status;
      if (sceneChanged) renderStrip();
    }
  };

  // Starting --------------------------------------------------------------
  const parseScenes = (text) => {
    const pattern = /^[ \t]*---(?:[ \t]*(\d+)[ \t]*---)?[ \t]*\r?$/gm;
    const found = [];
    let seconds = null;
    let start = 0;
    for (const match of text.matchAll(pattern)) {
      const prompt = text.slice(start, match.index).trim();
      if (prompt) { found.push([prompt, seconds]); seconds = null; }
      if (match[1] !== undefined) seconds = Number(match[1]);
      start = match.index + match[0].length;
    }
    const prompt = text.slice(start).trim();
    if (prompt) found.push([prompt, seconds]);
    return found;
  };

  const showStart = () => {
    flush();
    closeFullVideo();
    $("editor").hidden = true;
    const root = $("start");
    root.hidden = false;
    let connect = "flow";
    let previousStory = null;
    const name = h("input", {placeholder: "Rooftop chase", maxLength: 120});
    const story = h("textarea", {rows: 14, placeholder: "---8---\n\nWide shot over neon rooftops at dusk. Mei stands at the edge as sirens close in.\n\n---6---\n\nShe sprints and leaps across the gap to the next roof.\n\n---5---\n\nClose-up: she lands hard and catches her breath.", oninput: () => preview()});
    const usual = h("input", {type: "range", min: state.boot.seconds[0], max: state.boot.seconds[1], step: 1, value: 10,
      oninput: () => { usualLabel.textContent = `Usual scene length: ${usual.value} s`; preview(); }});
    const usualLabel = h("span", {}, "Usual scene length: 10 s");
    const count = h("input", {type: "number", min: 2, max: 24, value: 5});
    const previewRoot = h("div", {style: "display:contents"});
    const connectChips = h("div", {className: "chip-grid two"});
    const status = h("p", {className: "small muted", role: "status"});
    const undoBtn = button("Undo rewrite", () => {
      if (previousStory === null) return;
      story.value = previousStory;
      previousStory = null;
      undoBtn.hidden = true;
      status.textContent = "Restored your original story.";
      preview();
    }, "btn sm", {hidden: true});
    const renderConnect = () => connectChips.replaceChildren(
      chip("Continue one into the next", connect === "flow", () => { connect = "flow"; renderConnect(); preview(); }),
      chip("Every scene starts fresh", connect === "cut", () => { connect = "cut"; renderConnect(); preview(); }));
    const preview = () => {
      const found = parseScenes(story.value);
      const total = found.reduce((sum, [, seconds]) => sum + (seconds ?? Number(usual.value)), 0);
      previewRoot.replaceChildren(
        h("div", {className: "row"}, h("h2", {}, `${found.length} ${found.length === 1 ? "scene" : "scenes"}`),
          h("span", {className: "mono small muted"}, `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`)),
        ...found.flatMap(([text, seconds], index) => [
          index > 0 ? h("span", {className: `start-conn mono ${connect === "cut" ? "cut" : ""}`},
            connect === "flow" ? "CONTINUE" : "CUT") : null,
          h("article", {className: "card"},
            h("div", {className: "card-head"},
              h("span", {}, `Scene ${index + 1} · ${index === 0 ? "Opening" : connect === "flow" ? "Continues" : "Fresh start"}`),
              h("span", {className: "mono muted"}, `${seconds ?? usual.value}s`)),
            h("p", {}, text)),
        ].filter(Boolean)));
    };
    const rewrite = async (event) => {
      const control = event.currentTarget;
      const before = story.value;
      control.disabled = true;
      status.textContent = "Rewriting…";
      const data = await act(() => request("POST", `${API}/rewrite`, {story: story.value, connect, seconds: Number(usual.value), count: Number(count.value), writer: writerSettings()}));
      control.disabled = false;
      status.textContent = "";
      if (!data) return;
      previousStory = before;
      undoBtn.hidden = false;
      story.value = data.story;
      status.textContent = `Wrote ${count.value} scenes; each ---N--- line gives the next scene's length. Review or edit them, then choose Create storyboard.`;
      preview();
    };
    const create = async () => {
      const data = await act(() => request("POST", `${API}/boards`, {name: name.value, story: story.value, connect, seconds: Number(usual.value)}),
        "Storyboard created. Check each scene, then generate it.");
      if (data) adoptBoard(data);
    };
    root.replaceChildren(
      h("header", {className: "topbar"},
        h("h1", {style: "font-size:22px"}, "New storyboard"), h("div", {className: "grow"}),
        state.boot.boards.length ? button("Cancel", () => openBoard(state.board?.id || state.boot.boards[0].id)) : null,
        button("Create storyboard", create, "btn primary")),
      h("div", {className: "start-grid"},
        h("main", {className: "start-form"},
          h("label", {className: "field"}, "Name", name),
          h("label", {className: "field grow"}, "Story", story),
          h("p", {className: "small muted"}, "Enter a full story or short idea, then use Rewrite / extend into scenes. Or separate your own scenes with --- on its own line; ---6--- also makes the scene after it 6 seconds long."),
          h("div", {className: "start-writer-bar"},
            h("label", {className: "field grow"}, usualLabel, usual),
            h("label", {className: "field", style: "width:120px"}, "Scenes to write", count),
            button("Rewrite / extend into scenes", rewrite, "btn quiet"),
            undoBtn),
          status, writerForm()),
        h("aside", {className: "start-preview", "aria-label": "Scene preview"},
          h("fieldset", {}, h("legend", {}, "How should the scenes connect?"), connectChips),
          previewRoot,
          h("p", {className: "small muted"}, state.boot.gpu
            ? "Videos use your current MiniMax H3 settings, kept for this storyboard so every scene matches. Prompts and pictures are saved on this server."
            : "Videos use MiniMax H3's default settings, kept for this storyboard so every scene matches. To use your own, open the storyboard on the GPU endpoint and choose Storyboard settings › Use my current MiniMax H3 settings."))));
    renderConnect();
    preview();
    name.focus();
  };

  // Boards ----------------------------------------------------------------
  const adoptBoard = (data) => {
    state.boot.boards = [{id: data.id, name: data.name, generated: Boolean(data.generated)},
      ...state.boot.boards.filter((row) => row.id !== data.id)];
    showBoard(data);
  };
  const showBoard = (data) => {
    closeFullVideo();
    clearTimeout(state.pollTimer);
    state.pollTimer = null;
    state.run = null;
    state.board = data;
    state.previewKey = null;
    state.previewTake = null;
    state.question = null;
    state.deletingTake = null;
    // Auto render follows the queue from board to board, so its panel stays open.
    state.panel = state.panel === "auto" ? "auto" : null;
    state.confirmProfile = false;
    stored.set("h3-storyboard-board", data.id);
    const saved = stored.get("h3-storyboard-scene");
    state.sceneId = data.scenes.some((item) => item.id === saved) ? saved : data.scenes[0].id;
    followRun(data.run ?? null);
    $("start").hidden = true;
    $("editor").hidden = false;
    render();
  };
  const openBoard = async (storyboardId) => {
    await flush();
    await saves;
    const data = await act(() => request("GET", `${API}/boards/${storyboardId}`));
    if (data) showBoard(data);
  };

  // In the tab, the frame sizes to fit the active view so only the parent window scrolls.
  // It never ends above the window's bottom edge: that would move the page under the reader,
  // so a shorter view gives the space back as they scroll up.
  const fitFrame = () => {
    const frame = window.frameElement;
    if (!frame || frame.offsetParent === null) return;
    const active = $("editor").hidden ? $("start") : $("editor");
    let height = Math.max(860, Math.ceil(active.getBoundingClientRect().height));
    let scrollY = null;
    try {
      scrollY = window.parent.scrollY;
      const toWindowBottom = Math.ceil(window.parent.innerHeight - frame.getBoundingClientRect().top);
      height = Math.max(height, Math.min(frame.offsetHeight, toWindowBottom));
    } catch { /* Framed by another origin. */ }
    if (Math.abs(height - frame.offsetHeight) > 2) {
      frame.style.height = `${height}px`;
      if (scrollY !== null) {
        try { window.parent.scrollTo({top: scrollY, behavior: "instant"}); } catch { /* Framed by another origin. */ }
      }
    }
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
    // Anything that changes a view's height reaches the frame, or the frame would clip it.
    const views = new ResizeObserver(queueFit);
    views.observe($("editor"));
    views.observe($("start"));
    window.addEventListener("resize", queueFit);
    try {
      window.parent.addEventListener("scroll", queueFit, {passive: true});
    } catch { /* Framed by another origin. */ }
    try {
      state.boot = await request("GET", `${API}/boot`);
      bindSubtitleControls();
      $("gpu-note").hidden = state.boot.gpu;
      state.auto = state.boot.auto ?? null;
      if (state.auto?.active) state.autoTimer = setTimeout(pollAuto, AUTO_POLL_MS);
      const boards = state.boot.boards;
      if (!boards.length) {
        showStart();
      } else {
        const saved = stored.get("h3-storyboard-board");
        await openBoard(boards.some((row) => row.id === saved) ? saved : boards[0].id);
      }
    } catch (error) {
      toast(error.message, true);
    }
    $("loading").hidden = true;
  };
  start();
})();
