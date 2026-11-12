"use strict";
// The Media page. The server owns the library, annotations, deletion checks and long work; this draws what it answers.
(() => {
  const API = "/media/api";
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
  const STAR = '<svg width="12" height="12" viewBox="0 0 24 24" fill="#ffd27a" role="img" aria-label="Favorite"><path d="M12 3.5l2.6 5.3 5.9.9-4.3 4.1 1 5.8L12 16.9l-5.2 2.7 1-5.8-4.3-4.1 5.9-.9z"></path></svg>';
  const NOUN = {Video: "video", Image: "image", Audio: "audio"};
  const PAIR_EMPTY = "Choose an image or video in the library to start.";

  const app = {
    boot: null, mode: "Video", query: "", favorite: false, engine: "", sort: "newest", limit: 48,
    items: [], selected: null, item: null, tab: "details", stage: "audio",
    pair: {kind: null, a: null, b: null}, pairMessage: null, runs: {}, seen: {}, polling: false, results: null,
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
  const post = (path, body = {}) => request("POST", path, body);
  const query = (params) => new URLSearchParams(params).toString();

  let toastTimer = null;
  const toast = (message, error = false) => {
    const node = $("toast");
    node.textContent = message;
    node.classList.toggle("error", error);
    node.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { node.hidden = true; }, error ? 6000 : 3500);
  };
  const guard = (action) => async (...args) => {
    try { return await action(...args); } catch (error) { toast(error.message, true); }
  };

  // Browsing ------------------------------------------------------------------
  const renderKinds = (counts = {}) => {
    $("kinds").replaceChildren(...app.boot.kinds.map((kind) => h("button", {
      type: "button", "aria-pressed": String(kind === app.mode), onclick: () => switchMode(kind),
    }, kind, " ", h("span", {class: "count", text: String(counts[kind] ?? "")}))));
  };
  const renderEngines = (engines, engine) => {
    const select = $("engine");
    select.replaceChildren(h("option", {value: "", text: "All engines"}),
      ...engines.map((name) => h("option", {value: name, text: name})));
    select.value = engine;
    app.engine = engine;
  };
  const thumbCard = (item) => {
    const mark = app.pair.a?.path === item.path ? "A" : app.pair.b?.path === item.path ? "B" : "";
    const thumb = h("div", {class: "thumb"}, h("img", {src: item.thumb, alt: "", loading: "lazy"}));
    if (item.favorite) thumb.append(h("span", {class: "star", innerHTML: STAR}));
    if (mark) thumb.append(h("span", {class: "mark", text: mark}));
    return h("button", {
      type: "button", class: "card", "aria-pressed": String(item.path === app.selected), title: item.caption,
      onclick: () => select(item.path, {scroll: true}),
    }, thumb, h("span", {class: "label"},
      h("span", {class: "name", text: item.name}), h("span", {class: "meta", text: item.caption})));
  };
  const renderGrid = () => {
    $("grid").replaceChildren(...app.items.map(thumbCard));
    $("empty").hidden = app.items.length > 0;
  };
  let listing = 0;
  const loadItems = async ({reset = false} = {}) => {
    if (reset) app.limit = app.boot.page_size;
    const ticket = ++listing;
    const found = await request("GET", `/items?${query({
      mode: app.mode, query: app.query, favorite: app.favorite, engine: app.engine, sort: app.sort, limit: app.limit,
    })}`);
    if (ticket !== listing) return;  // A later search or type switch answered first.
    app.items = found.items;
    $("status").textContent = found.status;
    $("more").disabled = !found.more;
    renderKinds(found.counts);
    renderEngines(found.engines, found.engine);
    renderGrid();
    // A selection the library no longer lists is cleared, as the Gradio view did.
    if (app.selected && !found.items.some((item) => item.path === app.selected)) clearSelection();
  };
  const switchMode = guard(async (mode) => {
    if (mode === app.mode) return;
    app.mode = mode;
    app.engine = "";  // Engines differ per media type.
    clearSelection();
    await loadItems({reset: true});
  });
  let searchTimer = null;
  const liveSearch = () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(guard(async () => {
      app.query = $("query").value.trim();
      await loadItems({reset: true});
    }), 300);
  };

  // The selected item ------------------------------------------------------------
  const clearSelection = () => {
    app.selected = null;
    app.item = null;
    hideConfirm();
    renderSelection();
  };
  let picking = 0;
  const select = guard(async (path, {scroll = false} = {}) => {
    app.selected = path;
    hideConfirm();
    renderGrid();
    const ticket = ++picking;
    const item = await request("GET", `/item?${query({mode: app.mode, path})}`);
    if (ticket !== picking) return;  // A later pick answered first.
    app.item = item;
    app.stage = "audio";
    $("set-as-status").textContent = "Single images replace; references fill the first empty slot or append to the list.";
    $("tags-status").textContent = "";
    renderSelection();
    if (scroll && matchMedia("(max-width: 767px)").matches) {
      const smooth = !matchMedia("(prefers-reduced-motion: reduce)").matches;
      document.querySelector(".inspector")?.scrollIntoView({behavior: smooth ? "smooth" : "auto", block: "start"});
    }
  });
  const pausePreview = () => {
    for (const media of document.querySelectorAll(".inspector video, .inspector audio")) media.pause();
  };
  const renderPreview = () => {
    const item = app.item, views = item.views;
    const dub = Boolean(views.dub), switchable = views.switch && Boolean(views.stage);
    const illustrated = Boolean(views.stage) && !views.switch;
    const showStage = illustrated || (switchable && app.stage === "video");
    $("stage-switch").hidden = !switchable;
    for (const button of $("stage-switch").querySelectorAll("button")) {
      button.setAttribute("aria-pressed", String(button.dataset.stage === app.stage));
    }
    const preview = $("preview");
    preview.replaceChildren();
    if (!dub && !showStage) {
      // Assigned through the property so the Soundstage view finds its player.
      const node = item.mode === "Image" ? h("img", {alt: item.name})
        : item.mode === "Audio" ? h("audio", {controls: true, preload: "metadata"})
          : h("video", {controls: true, preload: "metadata", playsInline: true});
      node.src = item.url;
      preview.append(node);
    }
    const holder = $("views");
    holder.replaceChildren();
    const mount = (html, bind) => {
      if (!html) return;
      const box = h("div", {class: "view", innerHTML: html});
      holder.append(box);
      if (bind && app.results) bind(box);
    };
    if (dub) mount(views.dub, app.results?.bindPlayer);
    if (showStage) mount(views.stage, app.results?.bindStage);
    mount(views.soundstage);
  };
  const renderSelection = () => {
    const item = app.item;
    $("no-selection").hidden = Boolean(item);
    $("selection").hidden = !item;
    if (!item) { pausePreview(); return; }
    pausePreview();
    renderPreview();
    $("item-name").textContent = item.name;
    const listed = app.items.find((entry) => entry.path === item.path);
    $("item-caption").textContent = listed?.caption || "";
    $("favorite").setAttribute("aria-pressed", String(item.favorite));
    $("favorite").disabled = !item.asset_id;
    $("download").href = item.download;
    $("download").setAttribute("download", item.name);
    $("download-label").textContent = item.mode === "Image" ? "Download original image"
      : item.mode === "Audio" ? "Download audio" : "Download video";
    const comparable = item.mode !== "Audio" && Boolean(item.asset_id);
    $("add-a").hidden = $("add-b").hidden = item.mode === "Audio";
    $("add-a").disabled = $("add-b").disabled = !comparable;
    $("set-as").hidden = item.mode !== "Image" || !app.boot.set_as.length;
    // Details
    $("settings-title").textContent = `Settings used · ${item.name}`;
    $("settings").textContent = item.settings;
    // Tags & lineage
    $("tags").value = item.tags;
    $("tags").disabled = $("tags-save").disabled = !item.asset_id;
    $("lineage").replaceChildren(...(item.lineage.length ? item.lineage.map((step) => h("li",
      {"aria-current": step.current ? "true" : null}, h("strong", {text: step.name}), h("span", {text: step.note})))
      : [h("li", {class: "none", text: "Select an output to see where it came from."})]));
    $("identity").textContent = item.identity || "This item isn't in the library index yet.";
    // Enhance
    const enhance = app.boot.enhance.on && item.mode !== "Audio";
    $("tab-enhance").hidden = !enhance;
    if (!enhance && app.tab === "enhance") app.tab = "details";
    if (enhance) fillMethods();
    renderTabs();
    $("empty-library").textContent = `Empty ${NOUN[item.mode]} library…`;
    queueFit();
  };
  const renderTabs = () => {
    for (const button of $("tabs").querySelectorAll("button")) {
      button.setAttribute("aria-selected", String(button.dataset.tab === app.tab));
    }
    for (const name of ["details", "tags", "enhance"]) $(`panel-${name}`).hidden = app.tab !== name;
  };
  const annotate = guard(async (changes, message) => {
    if (!app.item) return;
    app.item = await post("/annotate", {mode: app.item.mode, path: app.item.path, ...changes});
    const listed = app.items.find((entry) => entry.path === app.item.path);
    if (listed) listed.favorite = app.item.favorite;
    renderGrid();
    $("favorite").setAttribute("aria-pressed", String(app.item.favorite));
    $("tags").value = app.item.tags;
    if (message) $("tags-status").textContent = message;
    if (app.favorite && !app.item.favorite) await loadItems();
  });

  // Enhance --------------------------------------------------------------------------
  const fillSelect = (select, values, value) => {
    const current = select.value;
    select.replaceChildren(...values.map((entry) => h("option", {value: entry, text: entry})));
    select.value = values.includes(current) ? current : (value ?? values[0] ?? "");
  };
  const fillMethods = () => {
    const methods = app.boot.enhance.methods[app.item.mode] || [];
    const select = $("method");
    const before = select.value;
    fillSelect(select, methods, methods[0]);
    if (!methods.includes(before)) select.value = methods[0];
    $("enhance-run").textContent = app.item.mode === "Image" ? "Enhance selected image" : "Enhance selected media";
    renderMethod();
  };
  const renderMethod = () => {
    const method = $("method").value, fields = app.boot.enhance.fields[method] || {};
    $("method-help").textContent = app.boot.enhance.method_help;
    $("resolution-box").hidden = !fields.resolution;
    $("seedvr2-box").hidden = !fields.seedvr2;
    $("ltx-box").hidden = !fields.ltx_model;
    $("prompt-box").hidden = !fields.prompt;
    $("prompt-help").textContent = fields.prompt_hint || "";
    $("seed-box").hidden = !fields.seed;
    queueFit();
  };
  const enhance = guard(async () => {
    if (!app.item) return;
    const answer = await post("/enhance", {
      mode: app.item.mode, path: app.item.path, method: $("method").value,
      seed: Number($("seed").value || -1), seedvr2_model: $("seedvr2").value, ltx_model: $("ltx").value,
      prompt: $("prompt").value, force_offload: $("offload").checked, resolution: $("resolution").value,
    });
    takeRuns(answer.runs);
  });

  // Compare ------------------------------------------------------------------------------
  const pairStatus = () => {
    const filled = ["a", "b"].filter((slot) => app.pair[slot]).map((slot) => slot.toUpperCase());
    if (filled.length === 2) return "A and B are ready. Compare them below.";
    if (filled.length) return `${filled[0]} is set. Select another thumbnail and add it to ${filled[0] === "A" ? "B" : "A"}.`;
    return PAIR_EMPTY;
  };
  const renderTray = () => {
    const {a, b} = app.pair;
    $("tray").hidden = !a && !b;
    $("slots").replaceChildren(...[["a", a], ["b", b]].map(([slot, entry]) => h("div", {class: "slot"},
      h("div", {class: "head"}, h("span", {class: "badge", text: slot.toUpperCase()}),
        h("span", {class: "slot-name", text: entry ? entry.name : "Empty"}),
        entry && h("button", {type: "button", class: "btn compact quiet", text: `Clear ${slot.toUpperCase()}`,
          onclick: () => { app.pair[slot] = null; app.pairMessage = null; $("compare-view").replaceChildren(); renderTray(); renderGrid(); }})),
      h("div", {class: "slot-thumb"}, entry && h("img", {src: entry.thumb, alt: ""})))));
    $("compare-status").textContent = app.pairMessage || pairStatus();
    $("compare-run").disabled = !(a && b);
    queueFit();
  };
  const assign = (slot) => {
    const item = app.item;
    if (!item || !["Image", "Video"].includes(item.mode) || !item.asset_id) {
      toast("Select an image or video thumbnail in the library first.", true);
      return;
    }
    const other = slot === "a" ? "b" : "a";
    app.pairMessage = null;
    if (app.pair[other]) {
      if (app.pair.kind !== item.mode) {
        app.pair[other] = null;
        app.pairMessage = `Started a new ${item.mode.toLowerCase()} pair. Add another ${item.mode.toLowerCase()} to ${other.toUpperCase()}.`;
      } else if (app.pair[other].path === item.path) {
        toast("This media is already in the other slot. Choose a different thumbnail.", true);
        return;
      }
    }
    const listed = app.items.find((entry) => entry.path === item.path);
    app.pair.kind = item.mode;
    app.pair[slot] = {path: item.path, name: item.name, url: item.url, thumb: listed?.thumb || item.url};
    $("compare-view").replaceChildren();
    renderTray();
    renderGrid();
  };
  const imageSlider = (a, b) => {
    const over = h("div", {class: "over"}, h("img", {src: a.url, alt: `A · ${a.name}`}));
    const handle = h("span", {class: "handle"});
    const range = h("input", {type: "range", min: "0", max: "100", value: "50", "aria-label": "Comparison image A / B"});
    const place = () => {
      over.style.clipPath = `inset(0 ${100 - Number(range.value)}% 0 0)`;
      handle.style.left = `${range.value}%`;
    };
    range.addEventListener("input", place);
    place();
    return h("div", {class: "slider"}, h("img", {src: b.url, alt: `B · ${b.name}`}), over, handle,
      h("span", {class: "tag", style: "left: 8px", text: "A"}), h("span", {class: "tag", style: "right: 8px", text: "B"}), range);
  };
  const videoPair = (left, right) => {
    const a = h("video", {preload: "metadata", playsInline: true}), b = h("video", {preload: "metadata", playsInline: true});
    a.src = left.url; b.src = right.url;
    const play = h("button", {type: "button", class: "btn", text: "Play both"});
    const seek = h("input", {type: "range", min: "0", max: "0", step: "0.01", value: "0", class: "grow"});
    const audio = h("select", {class: "field"}, h("option", {value: "mute", text: "Muted"}),
      h("option", {value: "a", text: "A only"}), h("option", {value: "b", text: "B only"}));
    const state = h("p", {class: "small muted", role: "status", "aria-live": "polite", text: "Loading media timing…"});
    let syncing = false;
    const end = () => Math.min(a.duration || 0, b.duration || 0);
    const pause = () => { a.pause(); b.pause(); play.textContent = "Play both"; };
    const mute = () => { a.muted = audio.value !== "a"; b.muted = audio.value !== "b"; };
    audio.addEventListener("change", mute); mute();
    play.addEventListener("click", async () => {
      if (!a.paused) { pause(); return; }
      b.currentTime = a.currentTime;
      try { await Promise.all([a.play(), b.play()]); play.textContent = "Pause both"; } catch { pause(); state.textContent = "Playback unavailable. Check browser codec support."; }
    });
    seek.addEventListener("input", () => { pause(); a.currentTime = b.currentTime = Number(seek.value); });
    for (const video of [a, b]) {
      video.addEventListener("loadedmetadata", () => { seek.max = end(); state.textContent = `Shared timeline: ${end().toFixed(2)} seconds. Audio: one source only.`; });
      video.addEventListener("waiting", pause);
      video.addEventListener("ended", pause);
    }
    a.addEventListener("timeupdate", () => {
      if (syncing) return;
      syncing = true;
      if (a.currentTime >= end()) pause();
      if (Math.abs(a.currentTime - b.currentTime) > 0.08) b.currentTime = a.currentTime;
      seek.value = a.currentTime;
      syncing = false;
    });
    return h("div", {class: "stack"},
      h("div", {class: "pairs"}, h("figure", {}, h("figcaption", {text: `A · ${left.name}`}), a),
        h("figure", {}, h("figcaption", {text: `B · ${right.name}`}), b)),
      h("div", {class: "row tight"}, play, h("label", {class: "inline small grow"}, "Comparison time", seek),
        h("label", {class: "inline small"}, "Audio", audio)), state);
  };
  const compare = () => {
    const {a, b, kind} = app.pair;
    if (!a || !b) return;
    $("compare-view").replaceChildren(kind === "Image" ? imageSlider(a, b) : videoPair(a, b));
    queueFit();
  };

  // Deletion -----------------------------------------------------------------------------
  const hideConfirm = () => { $("confirm").hidden = true; $("manage-actions").hidden = false; };
  const askDeletion = (action) => guard(async () => {
    const answer = await post("/delete/prepare", {mode: app.mode, path: app.selected, action});
    $("confirm-text").textContent = answer.message;
    $("confirm").hidden = false;
    $("manage-actions").hidden = true;
    queueFit();
  });
  const confirmDeletion = guard(async () => {
    try {
      const answer = await post("/delete/confirm", {mode: app.mode, path: app.selected});
      toast(answer.message);
      clearSelection();
    } finally {
      hideConfirm();
      await loadItems();
    }
  });

  // Long work --------------------------------------------------------------------------
  const RUN_STATUS = {scan: "scan-status", download: "download-status", enhance: "enhance-status"};
  const anyActive = () => Object.values(app.runs).some((run) => run?.active);
  const takeRuns = (runs) => {
    app.runs = runs || {};
    for (const [kind, run] of Object.entries(app.runs)) {
      if (!run) continue;
      const node = $(RUN_STATUS[kind]);
      node.textContent = run.error ? `${run.status} ${run.error}` : run.status;
      node.hidden = false;
      const seen = app.seen[kind];
      if (seen && seen.id === run.id && seen.active && !run.active) finished(run);
      app.seen[kind] = {id: run.id, active: run.active};
    }
    $("scan").disabled = Boolean(app.runs.scan?.active);
    $("download-run").disabled = Boolean(app.runs.download?.active);
    $("enhance-run").disabled = Boolean(app.runs.enhance?.active);
    // A running scan reports in the library's status line too, as it did in Gradio.
    if (app.runs.scan?.active) $("status").textContent = app.runs.scan.status;
    if (anyActive()) poll();
  };
  const finished = guard(async (run) => {
    if (run.error) toast(run.error, true);
    if (run.result && run.mode) {
      // A new file: show it, as the Gradio view switched to its type and selected it.
      if (app.mode !== run.mode) { app.mode = run.mode; app.engine = ""; }
      await loadItems({reset: true});
      await select(run.result);
      return;
    }
    await loadItems();
  });
  // One loop at a time follows every running job; takeRuns starts it and it stops when all are done.
  const poll = async () => {
    if (app.polling) return;
    app.polling = true;
    try {
      while (anyActive()) {
        await new Promise((resolve) => setTimeout(resolve, 1200));
        takeRuns((await request("GET", "/run")).runs);
      }
    } catch (error) {
      toast(error.message, true);
    } finally {
      app.polling = false;
    }
  };

  // Add media ------------------------------------------------------------------------
  const showAddTab = (name) => {
    for (const button of $("add-tabs").querySelectorAll("button")) {
      button.setAttribute("aria-selected", String(button.dataset.add === name));
    }
    $("add-file").hidden = name !== "file";
    $("add-link").hidden = name !== "link";
    $("add-upkeep").hidden = name !== "upkeep";
  };
  // Files waiting to be added; picking or dropping more adds to the batch.
  const IMPORT_IDLE = "The library switches to the last file's type and selects it.";
  let picked = [];
  const kindOf = (file) => ({video: "Video", image: "Image", audio: "Audio"})[file.type.split("/")[0]] || null;
  const size = (bytes) => bytes >= 1 << 30 ? `${(bytes / (1 << 30)).toFixed(1)} GB`
    : bytes >= 1 << 20 ? `${(bytes / (1 << 20)).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`;
  const renderPicked = () => {
    const list = $("picked");
    list.hidden = !picked.length;
    list.replaceChildren(...picked.map((file, index) => {
      const kind = kindOf(file);
      return h("li", {},
        h("span", {class: kind ? "kind" : "kind bad", text: kind || "Other"}),
        h("span", {class: "file", text: file.name, title: file.name}),
        h("span", {class: "size", text: size(file.size)}),
        h("button", {type: "button", class: "btn quiet icon", "aria-label": `Remove ${file.name}`, text: "✕",
          onclick: () => { picked.splice(index, 1); renderPicked(); }}));
    }));
    const count = picked.length;
    $("import-run").disabled = !count;
    $("import-run").textContent = count > 1 ? `Add ${count} files to library` : "Add to library";
    $("import-clear").hidden = !count;
    const unsupported = picked.filter((file) => !kindOf(file)).length;
    $("import-status").textContent = !count ? IMPORT_IDLE
      : unsupported ? `${unsupported} file${unsupported > 1 ? "s aren't" : " isn't"} video, image or audio and may be refused.`
        : IMPORT_IDLE;
    queueFit();
  };
  const pickFiles = (files) => {
    const key = (file) => `${file.name}:${file.size}:${file.lastModified}`;
    const known = new Set(picked.map(key));
    picked = [...picked, ...[...files].filter((file) => !known.has(key(file)))];
    $("import-file").value = "";
    renderPicked();
  };
  const importFiles = guard(async () => {
    if (!picked.length) return;
    const form = new FormData();
    for (const file of picked) form.append("files", file);
    $("import-run").disabled = true;
    $("import-status").textContent = picked.length > 1 ? `Adding ${picked.length} files…` : `Adding ${picked[0].name}…`;
    try {
      const answer = await request("POST", "/import", form, {form: true});
      toast(answer.status, answer.failed.length > 0);
      // Keep only the files that were refused, so they can be removed or tried again.
      const refused = new Set(answer.failed.map((item) => item.name));
      picked = picked.filter((file) => refused.has(file.name));
      renderPicked();
      $("import-status").textContent = answer.status;
      app.mode = answer.mode;
      app.engine = "";
      await loadItems({reset: true});
      await select(answer.selected);
      if (!picked.length) $("add-dialog").close();
    } finally {
      $("import-run").disabled = !picked.length;
    }
  });
  const cookieSource = () => $("cookie-source").querySelector('[aria-pressed="true"]').dataset.cookie;
  const takeCookies = (answer) => {
    $("cookie-summary").textContent = answer.summary;
    $("cookie-count").hidden = !answer.sites.length;
    $("cookie-count").textContent = `${answer.sites.length} site${answer.sites.length === 1 ? "" : "s"} saved`;
    fillSelect($("cookie-site"), answer.sites, answer.sites[0]);
    $("cookie-delete").disabled = !answer.sites.length;
    if (answer.message) $("cookie-status").textContent = answer.message;
  };
  const saveCookies = guard(async () => {
    if (cookieSource() === "curl") {
      const answer = await post("/cookies/curl", {curl: $("curl").value, domain: $("cookie-domain").value});
      takeCookies(answer);
      if (answer.saved) $("curl").value = "";
    } else {
      const file = $("cookie-file").files?.[0];
      if (!file) throw new Error("Choose a cookies.txt file.");
      const form = new FormData();
      form.append("file", file);
      form.append("domain", $("cookie-domain").value);
      takeCookies(await request("POST", "/cookies/file", form, {form: true}));
    }
  });

  // Frame and theme, as on the other pages ----------------------------------------
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

  const bind = () => {
    $("search-form").addEventListener("submit", (event) => { event.preventDefault(); clearTimeout(searchTimer); app.query = $("query").value.trim(); guard(() => loadItems({reset: true}))(); });
    $("query").addEventListener("input", liveSearch);
    const toggleFavorites = guard(async () => {
      app.favorite = !app.favorite;
      $("favorite-only").setAttribute("aria-pressed", String(app.favorite));
      $("favorite-chip").hidden = !app.favorite;
      await loadItems({reset: true});
    });
    $("favorite-only").addEventListener("click", toggleFavorites);
    $("favorite-chip").addEventListener("click", toggleFavorites);
    $("engine").addEventListener("change", guard(async (event) => { app.engine = event.target.value; await loadItems({reset: true}); }));
    $("sort").addEventListener("change", guard(async (event) => { app.sort = event.target.value; await loadItems({reset: true}); }));
    $("refresh").addEventListener("click", guard(() => loadItems({reset: true})));
    $("more").addEventListener("click", guard(async () => { app.limit += app.boot.page_size; await loadItems(); }));
    // Inspector
    $("stage-switch").addEventListener("click", (event) => {
      const button = event.target.closest("button[data-stage]");
      if (!button || button.dataset.stage === app.stage) return;
      pausePreview();
      app.stage = button.dataset.stage;
      renderPreview();
      queueFit();
    });
    $("favorite").addEventListener("click", () => annotate({favorite: !app.item?.favorite}));
    $("tags-save").addEventListener("click", () => annotate({tags: $("tags").value}, "Asset annotations saved."));
    $("tags").addEventListener("keydown", (event) => { if (event.key === "Enter") { event.preventDefault(); $("tags-save").click(); } });
    $("add-a").addEventListener("click", () => assign("a"));
    $("add-b").addEventListener("click", () => assign("b"));
    $("compare-run").addEventListener("click", compare);
    $("set-as-apply").addEventListener("click", guard(async () => {
      if (!$("set-as-name").value) throw new Error("Choose an image input from Set as.");
      const answer = await post("/set-as", {path: app.item.path, name: $("set-as-name").value});
      $("set-as-status").textContent = answer.message;
    }));
    $("tabs").addEventListener("click", (event) => {
      const button = event.target.closest("button[data-tab]");
      if (!button) return;
      app.tab = button.dataset.tab;
      renderTabs();
      queueFit();
    });
    $("method").addEventListener("change", renderMethod);
    $("enhance-run").addEventListener("click", enhance);
    $("enhance-stop").addEventListener("click", guard(async () => {
      const answer = await post("/enhance/stop");
      toast(answer.message);
      takeRuns(answer.runs);
    }));
    $("delete-selected").addEventListener("click", askDeletion("selected"));
    $("empty-library").addEventListener("click", askDeletion("library"));
    $("confirm-yes").addEventListener("click", confirmDeletion);
    $("confirm-no").addEventListener("click", guard(async () => { hideConfirm(); await post("/delete/cancel"); }));
    // Add media
    $("add-open").addEventListener("click", () => { $("add-dialog").showModal(); });
    for (const id of ["add-close", "add-done"]) $(id).addEventListener("click", () => $("add-dialog").close());
    $("add-tabs").addEventListener("click", (event) => {
      const button = event.target.closest("button[data-add]");
      if (button) showAddTab(button.dataset.add);
    });
    $("import-file").addEventListener("change", (event) => pickFiles(event.target.files || []));
    const drop = $("drop");
    drop.addEventListener("dragover", (event) => { event.preventDefault(); drop.classList.add("over"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("over"));
    drop.addEventListener("drop", (event) => {
      event.preventDefault();
      drop.classList.remove("over");
      if (event.dataTransfer?.files?.length) pickFiles(event.dataTransfer.files);
    });
    $("import-run").addEventListener("click", importFiles);
    $("import-clear").addEventListener("click", () => { picked = []; renderPicked(); });
    $("download-run").addEventListener("click", guard(async () => {
      const answer = await post("/download", {url: $("url").value, quality: $("quality").value});
      takeRuns(answer.runs);
    }));
    $("download-info").addEventListener("click", guard(async () => {
      $("download-status").hidden = false;
      $("download-status").textContent = "Checking the link…";
      $("download-status").textContent = (await post("/download/info", {url: $("url").value})).message;
    }));
    $("download-stop").addEventListener("click", guard(async () => {
      const answer = await post("/download/stop");
      $("download-status").hidden = false;
      $("download-status").textContent = answer.message;
      takeRuns(answer.runs);
    }));
    $("cookies").addEventListener("toggle", guard(async () => { if ($("cookies").open) takeCookies(await request("GET", "/cookies")); }));
    $("cookie-source").addEventListener("click", (event) => {
      const button = event.target.closest("button[data-cookie]");
      if (!button) return;
      for (const other of $("cookie-source").querySelectorAll("button")) other.setAttribute("aria-pressed", String(other === button));
      $("curl-box").hidden = button.dataset.cookie !== "curl";
      $("cookie-file-box").hidden = button.dataset.cookie !== "file";
      $("cookie-domain").placeholder = button.dataset.cookie === "curl" ? "Detected from the command" : "youtube.com";
    });
    $("cookie-save").addEventListener("click", saveCookies);
    $("cookie-delete").addEventListener("click", guard(async () => takeCookies(await post("/cookies/delete", {domain: $("cookie-site").value}))));
    $("scan").addEventListener("click", guard(async () => takeRuns((await post("/scan")).runs)));
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
    let shown = innerWidth > 0;
    new ResizeObserver(() => {
      // Opening the Media tab refreshes the library, as selecting the tab did.
      const visible = innerWidth > 0;
      if (visible && !shown && app.boot) guard(() => loadItems())();
      shown = visible;
    }).observe(document.documentElement);
    try {
      // The Soundstage view follows its player from the moment media gets a source, so this loads first.
      app.results = await import("./results.js");
      app.results.followSubtitles();
    } catch { app.results = null; }
    try {
      app.boot = await request("GET", "/boot");
      const sort = $("sort");
      sort.replaceChildren(...app.boot.sorts.map(([label, value]) => h("option", {value, text: label})));
      sort.value = app.sort;
      fillSelect($("quality"), app.boot.qualities, app.boot.quality);
      $("set-as-name").replaceChildren(h("option", {value: "", text: "Choose an image input…"}),
        ...app.boot.set_as.map((name) => h("option", {value: name, text: name})));
      const enhance = app.boot.enhance;
      fillSelect($("resolution"), enhance.resolutions, enhance.resolution);
      fillSelect($("seedvr2"), enhance.seedvr2, enhance.seedvr2_default);
      $("seedvr2-help").textContent = enhance.seedvr2_help;
      fillSelect($("ltx"), enhance.ltx, enhance.ltx_default);
      $("add-tab-link").hidden = !app.boot.downloads;
      showAddTab("file");
      bind();
      renderKinds();
      renderTray();
      renderSelection();
      $("loading").hidden = true;
      $("app").hidden = false;
      await loadItems({reset: true});
      takeRuns(app.boot.runs);
    } catch (error) {
      $("loading").textContent = `Media couldn't open: ${error.message}`;
    }
  };
  start();
})();
