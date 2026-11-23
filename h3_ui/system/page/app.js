"use strict";
// The System page. Server routes own GPU stats, Drama 3 settings, session Gemini key, project transfer,
// model presence, and media index maintenance; this renders the state and dispatches user actions.
(() => {
  const API = "/system/api";
  const {$, h, createClient, createToast} = globalThis.H3PageCommon;
  const {request, post: rawPost} = createClient(API);
  const post = (path, body = {}) => rawPost(path, body);
  const {toast, guard} = createToast({duration: 4000, errorDuration: 7000});

  const PROJECT_KINDS = [
    ["all", "All"],
    ["soundstage", "Soundstage"],
    ["storyboard", "Storyboard"],
    ["dubbing", "Dubbing"],
    ["remake", "Remake"],
  ];
  const CONFLICT_MODES = [
    ["rename", "Import as copy (rename)"],
    ["skip", "Skip existing"],
    ["overwrite", "Overwrite existing"],
  ];
  const PRESENCE_FILTERS = [
    ["all", "All"],
    ["local", "Present locally"],
    ["ondemand", "Downloads when used"],
  ];
  const ENGINE_FILTERS = [
    ["all", "All"],
    ["video", "Video"],
    ["image", "Image"],
    ["audio", "Audio / Music"],
    ["production", "Production"],
  ];

  const app = {
    booted: false,
    tab: "runtime",
    backend: null,
    drama3: null,
    drama3Error: null,
    geminiKey: null,
    projects: {projects: [], counts: {}},
    localKind: "all",
    localQuery: "",
    editingKey: null,
    renameDraft: "",
    confirmDeleteKey: null,
    remoteServerApp: "",
    remoteLoadedUrl: "",
    remoteProjects: [],
    remoteSelected: new Set(),
    conflict: "rename",
    importResults: null,
    models: {items: [], present_count: 0, missing_count: 0, total_count: 0, families: []},
    presenceFilter: "all",
    familyFilter: "all",
    modelQuery: "",
    engines: [],
    engineFilter: "all",
    runs: {},
    seenFinished: {},
    polling: false,
    unloadMessage: "",
  };

  // Frame height & parent theme sync -------------------------------------------
  const fitFrame = () => {
    const frame = window.frameElement;
    if (!frame) return;
    const height = Math.max(760, document.body.scrollHeight + 24);
    if (Math.abs(frame.offsetHeight - height) > 4) frame.style.height = `${height}px`;
  };
  let fitQueued = false;
  const queueFit = () => {
    if (fitQueued) return;
    fitQueued = true;
    requestAnimationFrame(() => {
      fitQueued = false;
      fitFrame();
    });
  };
  const applyTheme = () => {
    let dark = matchMedia("(prefers-color-scheme: dark)").matches;
    try {
      if (window.parent !== window) dark = window.parent.document.body.classList.contains("dark");
    } catch { /* Framed by another origin. */ }
    document.documentElement.dataset.theme = dark ? "dark" : "light";
  };

  // Rendering helpers ----------------------------------------------------------
  const projectKey = (item) => `${item.kind}:${item.id}`;

  const renderHeroAndTiles = () => {
    const b = app.backend || {};
    const d = app.drama3 || {};
    const m = app.models || {};
    const pCount = app.projects?.projects?.length ?? 0;

    // Header pill & ComfyUI actions
    const pill = $("header-status");
    pill.className = b.mode === "studio" ? "pill" : b.mode === "workshop" ? "pill warn" : "pill fail";
    pill.textContent = b.mode === "studio" && b.gpu ? `${b.status_label} · ${b.gpu}` : (b.status_label || "System");

    $("open-comfy-link").hidden = !b.can_open_comfyui;
    $("open-comfy-disabled").hidden = Boolean(b.can_open_comfyui);

    // Tile 1: Runtime
    const rVal = $("tile-runtime-val");
    rVal.textContent = b.mode === "studio" ? "Connected" : b.mode === "workshop" ? "CPU Workshop" : "Unavailable";
    rVal.className = b.mode === "studio" ? "ink" : b.mode === "workshop" ? "warn" : "fail";
    $("tile-runtime-sub").textContent = b.mode === "studio"
      ? (b.gpu || "GPU worker ready")
      : b.mode === "workshop"
        ? "No GPU backend"
        : "Backend check failed";

    // Tile 2: VRAM
    const vram = b.vram;
    if (b.mode === "studio" && vram && vram.free_gib !== null) {
      $("tile-vram-val").textContent = `${vram.free_gib.toFixed(1)} / ${vram.total_gib.toFixed(1)} GiB`;
      $("tile-vram-track").hidden = false;
      $("tile-vram-bar").style.width = `${Math.min(100, Math.max(0, vram.used_pct || 0))}%`;
      $("tile-vram-sub").textContent = `${vram.used_gib.toFixed(1)} GiB used (${vram.used_pct}%)`;
    } else if (b.mode === "studio" && vram) {
      $("tile-vram-val").textContent = `${vram.total_gib.toFixed(1)} GiB`;
      $("tile-vram-track").hidden = true;
      $("tile-vram-sub").textContent = "Total VRAM reported";
    } else {
      $("tile-vram-val").textContent = b.mode === "workshop" ? "Disabled" : "—";
      $("tile-vram-track").hidden = true;
      $("tile-vram-sub").textContent = b.mode === "workshop" ? "CPU mode active" : "VRAM stats unavailable";
    }

    // Tile 3: Drama 3 serving
    $("tile-drama-val").textContent = `${d.backend_label || "—"} · ${String(d.model || "").replace("drama3_comfy_", "").toUpperCase()}`;
    $("tile-drama-sub").textContent = d.show_comfy_experts
      ? `Sequential · experts ${d.comfy_experts}`
      : (d.model_label || "Parallel batch");

    // Tile 4: Local readiness
    $("tile-ready-val").textContent = `${m.present_count ?? 0} / ${m.total_count ?? 0} local`;
    $("tile-ready-sub").textContent = `${pCount} saved project${pCount === 1 ? "" : "s"} · ${m.missing_count ?? 0} on demand`;

    // Tab counts
    $("tab-projects-count").textContent = String(pCount);
    $("tab-models-count").textContent = `${m.present_count ?? 0}/${m.total_count ?? 0}`;
  };

  const renderRuntimeTab = () => {
    const b = app.backend || {};
    const unloadRun = app.runs.unload;
    const unloading = Boolean(unloadRun?.active);
    $("unload-btn").disabled = !b.can_unload_vram || unloading;
    $("unload-btn").textContent = unloading ? "Unloading models…" : "Unload all models / free VRAM";

    const banner = $("runtime-banner");
    if (b.mode === "studio") {
      const vram = b.vram;
      const nodes = [
        h("div", {class: "row between"},
          h("div", {class: "row tight"},
            h("span", {class: "pill", text: "Connected"}),
            h("strong", {text: b.gpu || "GPU ready"})),
          vram && vram.free_gib !== null
            ? h("span", {class: "mono small", text: `${vram.free_gib.toFixed(1)} / ${vram.total_gib.toFixed(1)} GiB VRAM free`})
            : null),
      ];
      if (vram && vram.free_gib !== null) {
        nodes.push(
          h("div", {class: "bar-track lg"}, h("div", {class: "bar-fill", style: `width: ${Math.min(100, Math.max(0, vram.used_pct || 0))}%`})),
          h("div", {class: "row between small muted"},
            h("span", {text: `${vram.used_gib.toFixed(1)} GiB allocated (${vram.used_pct}%)`}),
            h("span", {text: "ComfyUI + Torch allocator"})));
      }
      banner.replaceChildren(h("div", {class: "callout"}, ...nodes));
    } else if (b.mode === "workshop") {
      banner.replaceChildren(h("div", {class: "callout warn", role: "status"},
        h("strong", {text: "CPU Workshop · No GPU backend attached"}),
        h("p", {class: "small", text: "ComfyUI, SGLang-Omni and GPU workers are disabled. All UI pages, Media library, Models manager (on Modal), prompt writers, project editing and GPT Image API remain available."})));
    } else {
      banner.replaceChildren(h("div", {class: "callout fail", role: "alert"},
        h("strong", {text: "Backend needs attention"}),
        h("p", {class: "mono small", text: b.message || "Backend unavailable."})));
    }

    const uStatus = $("unload-status");
    const uMsg = unloadRun?.active ? unloadRun.status : (unloadRun?.error || app.unloadMessage);
    uStatus.hidden = !uMsg;
    if (uMsg) {
      uStatus.className = unloadRun?.error ? "callout fail small" : "callout soft small";
      uStatus.textContent = uMsg;
    }

    // Capabilities grid
    $("runtime-caps").replaceChildren(...(b.capabilities || []).map((cap) =>
      h("div", {class: "cap-item"},
        h("dt", {text: cap.label}),
        h("dd", {text: cap.value}))));

    // Diagnostics & profiles
    const diagBody = $("diagnostics-body");
    const profiles = b.profiles || [];
    if (profiles.length) {
      diagBody.replaceChildren(...profiles.map((p) =>
        h("div", {class: "profile-row"},
          h("strong", {text: p.name}),
          h("span", {class: "mono muted small", text: p.detail}))));
    } else {
      diagBody.replaceChildren(h("p", {class: "small muted", text: b.raw || "No active model profiles reported."}));
    }

    // Drama 3 serving
    const d = app.drama3 || {};
    $("drama3-backends").replaceChildren(...(d.backends || []).map((opt) => {
      const checked = opt.id === d.backend;
      return h("button", {
        type: "button",
        class: "card-radio",
        role: "radio",
        "aria-checked": String(checked),
        onclick: () => saveDrama3(opt.id, d.model, d.comfy_experts),
      },
      h("div", {class: "card-radio-head"},
        h("strong", {text: opt.id === "sglang" ? "SGLang-Omni" : "ComfyUI"}),
        h("span", {class: checked ? "pill" : "pill muted-pill", text: opt.badge})),
      h("span", {class: "small muted", text: opt.description}));
    }));

    const modelSel = $("drama3-model");
    modelSel.replaceChildren(...(d.models || []).map((m) =>
      h("option", {value: m.id, text: m.label})));
    modelSel.value = d.model || "";

    $("drama3-experts-wrap").hidden = !d.show_comfy_experts;
    if (d.show_comfy_experts) {
      $("drama3-experts").replaceChildren(...(d.experts_choices || []).map((ex) => {
        const checked = ex.id === d.comfy_experts;
        return h("button", {
          type: "button",
          class: "card-radio",
          role: "radio",
          "aria-checked": String(checked),
          onclick: () => saveDrama3(d.backend, d.model, ex.id),
        },
        h("strong", {text: ex.id === "in VRAM" ? "In VRAM" : "In RAM"}),
        h("span", {class: "small muted", text: ex.label}));
      }));
    }

    $("drama3-error").hidden = !app.drama3Error;
    $("drama3-error").textContent = app.drama3Error || "";
    $("drama3-summary-text").textContent = d.summary_plain || "";

    // Session Gemini API key
    const gk = app.geminiKey || {};
    const gPill = $("gemini-key-pill");
    if (gk.session_set) {
      gPill.className = "pill";
      gPill.textContent = `Session key active · ${gk.masked}`;
    } else if (gk.env_set) {
      gPill.className = "pill muted-pill";
      gPill.textContent = "Using server GEMINI_API_KEY";
    } else {
      gPill.className = "pill warn";
      gPill.textContent = "Not configured";
    }
    $("gemini-key-clear").hidden = !gk.session_set;
  };

  const renderLocalProjects = () => {
    const counts = app.projects?.counts || {};
    $("local-kind-filters").replaceChildren(...PROJECT_KINDS.map(([kind, label]) => {
      const count = counts[kind] ?? 0;
      const pressed = app.localKind === kind;
      return h("button", {
        type: "button",
        "aria-pressed": String(pressed),
        onclick: () => { app.localKind = kind; renderLocalProjects(); queueFit(); },
      }, label, " ", h("span", {class: "count mono", text: String(count)}));
    }));

    const q = app.localQuery.trim().toLowerCase();
    const visible = (app.projects?.projects || []).filter((item) => {
      if (app.localKind !== "all" && item.kind !== app.localKind) return false;
      if (q && !String(item.name || "").toLowerCase().includes(q)
          && !String(item.summary || "").toLowerCase().includes(q)) return false;
      return true;
    });

    const listEl = $("local-projects-list");
    $("local-projects-empty").hidden = visible.length > 0;
    listEl.replaceChildren(...visible.map((item) => {
      const key = projectKey(item);
      const renaming = app.editingKey === key;
      const confirming = app.confirmDeleteKey === key;

      const card = h("div", {class: confirming ? "project-item danger" : "project-item"},
        h("div", {class: "project-row"},
          h("div", {class: "stack-4 grow"},
            h("div", {class: "row tight"},
              h("span", {class: "pill", text: item.kind_label || item.kind}),
              h("strong", {text: item.name})),
            item.summary ? h("span", {class: "small muted", text: item.summary}) : null),
          h("div", {class: "row tight"},
            h("button", {
              type: "button",
              class: "btn compact",
              text: "Rename",
              onclick: () => {
                app.editingKey = key;
                app.confirmDeleteKey = null;
                app.renameDraft = item.name || "";
                renderLocalProjects();
                queueFit();
              },
            }),
            h("button", {
              type: "button",
              class: "btn compact danger",
              text: "Remove",
              onclick: () => {
                app.confirmDeleteKey = key;
                app.editingKey = null;
                renderLocalProjects();
                queueFit();
              },
            }))));

      if (renaming) {
        const input = h("input", {
          type: "text",
          class: "field grow",
          value: app.renameDraft,
          "aria-label": "New project name",
          oninput: (event) => { app.renameDraft = event.target.value; },
          onkeydown: (event) => {
            if (event.key === "Enter") { event.preventDefault(); commitRename(item); }
            if (event.key === "Escape") { app.editingKey = null; renderLocalProjects(); queueFit(); }
          },
        });
        card.append(h("div", {class: "row tight line-above"},
          input,
          h("button", {type: "button", class: "btn compact primary", text: "Save name", onclick: () => commitRename(item)}),
          h("button", {type: "button", class: "btn compact", text: "Cancel", onclick: () => { app.editingKey = null; renderLocalProjects(); queueFit(); }})));
      }

      if (confirming) {
        card.append(h("div", {class: "row between line-above", role: "alert"},
          h("span", {class: "small", text: `Delete "${item.name}" permanently? Clips, cast, takes, and results will be removed.`}),
          h("div", {class: "row tight"},
            h("button", {type: "button", class: "btn compact", text: "Cancel", onclick: () => { app.confirmDeleteKey = null; renderLocalProjects(); queueFit(); }}),
            h("button", {type: "button", class: "btn compact solid-danger", text: "Remove permanently", onclick: () => commitDelete(item)}))));
      }
      return card;
    }));
  };

  const renderRemoteImport = () => {
    const hasLoaded = Boolean(app.remoteLoadedUrl);
    $("remote-idle").hidden = hasLoaded;
    $("remote-loaded").hidden = !hasLoaded;
    if (!hasLoaded) return;

    $("remote-server-banner").textContent = `Connected to ${app.remoteServerApp} · ${app.remoteProjects.length} project${app.remoteProjects.length === 1 ? "" : "s"}`;

    const localNames = new Set((app.projects?.projects || []).map((p) => String(p.name || "").toLowerCase()));
    $("remote-projects-list").replaceChildren(...app.remoteProjects.map((rp) => {
      const key = projectKey(rp);
      const checked = app.remoteSelected.has(key);
      const collides = localNames.has(String(rp.name || "").toLowerCase());
      return h("label", {class: checked ? "project-check selected" : "project-check"},
        h("input", {
          type: "checkbox",
          checked,
          onchange: (event) => {
            if (event.target.checked) app.remoteSelected.add(key);
            else app.remoteSelected.delete(key);
            renderRemoteImport();
            queueFit();
          },
        }),
        h("span", {class: "stack-4 grow"},
          h("span", {class: "row tight"},
            h("span", {class: "pill", text: rp.kind_label || rp.kind}),
            h("strong", {text: rp.name}),
            collides ? h("span", {class: "pill warn", text: "Matches local name"}) : null),
          rp.summary ? h("span", {class: "small muted", text: rp.summary}) : null));
    }));

    $("remote-conflict").replaceChildren(...CONFLICT_MODES.map(([mode, label]) =>
      h("button", {
        type: "button",
        role: "radio",
        "aria-checked": String(app.conflict === mode),
        onclick: () => { app.conflict = mode; renderRemoteImport(); queueFit(); },
      }, label)));

    $("overwrite-warning").hidden = app.conflict !== "overwrite";

    const importRun = app.runs.import;
    const importing = Boolean(importRun?.active);
    const count = app.remoteSelected.size;
    const importBtn = $("remote-import-btn");
    importBtn.disabled = importing || count === 0;
    importBtn.textContent = importing
      ? "Importing selected projects…"
      : count === 0
        ? "Select projects to import"
        : `Import ${count} selected project${count === 1 ? "" : "s"}`;

    const prog = $("remote-progress");
    const progText = importRun?.active ? importRun.status : importRun?.error;
    prog.hidden = !progText;
    if (progText) {
      prog.className = importRun?.error ? "callout fail small" : "callout soft small";
      prog.textContent = progText;
    }

    const resBox = $("remote-results");
    const results = app.importResults || importRun?.results;
    resBox.hidden = !results || !results.length;
    if (results && results.length) {
      resBox.replaceChildren(
        h("strong", {text: importRun?.message || `Import finished (${results.length} processed)`}),
        ...results.map((r) => h("span", {text: `· ${r.message}`})));
    }
  };

  const renderModelsAndEngines = () => {
    const m = app.models || {};
    $("models-summary-pill").textContent = `${m.present_count ?? 0} present locally · ${m.missing_count ?? 0} downloads when used`;

    $("models-presence-filters").replaceChildren(...PRESENCE_FILTERS.map(([id, label]) =>
      h("button", {
        type: "button",
        "aria-pressed": String(app.presenceFilter === id),
        onclick: () => { app.presenceFilter = id; renderModelsAndEngines(); queueFit(); },
      }, label)));

    const familySel = $("models-family");
    familySel.replaceChildren(
      h("option", {value: "all", text: "All families"}),
      ...(m.families || []).map((f) => h("option", {value: f, text: f})));
    familySel.value = app.familyFilter;

    const q = app.modelQuery.trim().toLowerCase();
    const visibleModels = (m.items || []).filter((item) => {
      if (app.presenceFilter === "local" && !item.present) return false;
      if (app.presenceFilter === "ondemand" && item.present) return false;
      if (app.familyFilter !== "all" && item.family !== app.familyFilter) return false;
      if (q && !item.key.toLowerCase().includes(q)
          && !item.path.toLowerCase().includes(q)
          && !item.family.toLowerCase().includes(q)) return false;
      return true;
    });

    $("models-shown-count").textContent = `Showing ${visibleModels.length} of ${m.total_count ?? 0}`;
    $("models-tbody").replaceChildren(...visibleModels.map((item) =>
      h("tr", {},
        h("td", {class: "mono", text: item.key}),
        h("td", {class: "muted", text: item.family}),
        h("td", {class: "mono path", text: item.path}),
        h("td", {},
          h("span", {class: item.present ? "pill" : "pill warn"},
            h("span", {class: "dot", "aria-hidden": "true"}),
            item.present ? "Present locally" : "Downloads when used")))));

    const engines = app.engines || [];
    $("engines-heading").textContent = `Engine capabilities (${engines.length})`;
    $("engine-task-filters").replaceChildren(...ENGINE_FILTERS.map(([group, label]) =>
      h("button", {
        type: "button",
        "aria-pressed": String(app.engineFilter === group),
        onclick: () => { app.engineFilter = group; renderModelsAndEngines(); queueFit(); },
      }, group === "all" ? `All ${engines.length}` : label)));

    const visibleEngines = engines.filter((eng) => {
      if (app.engineFilter === "all") return true;
      if (app.engineFilter === "image") return eng.tasks.includes("Image");
      if (app.engineFilter === "video") return eng.tasks.includes("Video");
      return eng.group === app.engineFilter;
    });
    $("engines-grid").replaceChildren(...visibleEngines.map((eng) =>
      h("article", {class: "engine-card"},
        h("div", {class: "engine-head"},
          h("strong", {text: eng.label}),
          h("span", {class: "pill", text: eng.tasks_display})),
        h("p", {class: "small muted", text: eng.description}))));
  };

  const renderMaintenance = () => {
    const scanRun = app.runs.scan;
    const scanning = Boolean(scanRun?.active);
    $("scan-btn").disabled = scanning;
    $("scan-btn").textContent = scanning ? "Scanning historical media…" : "Scan historical media";

    const box = $("scan-status");
    const msg = scanRun ? (scanRun.error || scanRun.status) : "";
    box.hidden = !msg;
    if (msg) {
      box.className = scanRun?.error ? "callout fail small" : "callout soft small";
      box.textContent = msg;
    }
  };

  const renderAll = () => {
    renderHeroAndTiles();
    renderRuntimeTab();
    renderLocalProjects();
    renderRemoteImport();
    renderModelsAndEngines();
    renderMaintenance();
    queueFit();
  };

  const switchTab = (tabId) => {
    app.tab = tabId;
    for (const button of $("section-tabs").querySelectorAll("button[data-tab]")) {
      button.setAttribute("aria-selected", String(button.dataset.tab === tabId));
    }
    for (const id of ["runtime", "projects", "models", "maintenance"]) {
      $(`panel-${id}`).hidden = id !== tabId;
    }
    queueFit();
  };

  // Actions & run polling -------------------------------------------------------
  const saveDrama3 = guard(async (backend, model, comfyExperts) => {
    const data = await post("/drama3", {backend, model, comfy_experts: comfyExperts});
    app.drama3 = data.drama3;
    app.drama3Error = data.error || null;
    renderHeroAndTiles();
    renderRuntimeTab();
    queueFit();
    if (!data.error) toast(`Drama 3 set to ${data.drama3.backend_label} (${data.drama3.model_label}).`);
  });

  const commitRename = guard(async (item) => {
    const data = await post("/projects/rename", {
      kind: item.kind,
      id: item.id,
      name: app.renameDraft,
    });
    app.projects = {projects: data.projects, counts: data.counts};
    app.editingKey = null;
    renderHeroAndTiles();
    renderLocalProjects();
    renderRemoteImport();
    queueFit();
    toast(data.outcome?.message || "Project renamed.");
  });

  const commitDelete = guard(async (item) => {
    const data = await post("/projects/delete", {kind: item.kind, id: item.id});
    app.projects = {projects: data.projects, counts: data.counts};
    app.confirmDeleteKey = null;
    renderHeroAndTiles();
    renderLocalProjects();
    renderRemoteImport();
    queueFit();
    toast(data.outcome?.message || "Project removed.");
  });

  const takeRuns = async (runs) => {
    app.runs = runs || {};
    let needsRefresh = false;

    const unload = app.runs.unload;
    if (unload && !unload.active && app.seenFinished.unload !== unload.id) {
      app.seenFinished.unload = unload.id;
      if (unload.backend) app.backend = unload.backend;
      if (unload.message) app.unloadMessage = unload.message;
      if (unload.error) toast(unload.error, true);
      else if (unload.message) toast(unload.message);
    }

    const imp = app.runs.import;
    if (imp && !imp.active && app.seenFinished.import !== imp.id) {
      app.seenFinished.import = imp.id;
      if (imp.results) app.importResults = imp.results;
      if (imp.error) toast(imp.error, true);
      else {
        toast(imp.message || "Project import finished.");
        needsRefresh = true;
      }
    }

    const scan = app.runs.scan;
    if (scan && !scan.active && app.seenFinished.scan !== scan.id) {
      app.seenFinished.scan = scan.id;
      if (scan.error) toast(scan.error, true);
      else if (scan.message) toast(scan.message);
    }

    if (needsRefresh) {
      app.projects = await request("GET", "/projects");
    }
    renderAll();

    const anyActive = Object.values(app.runs).some((run) => run && run.active);
    if (anyActive && !app.polling) {
      app.polling = true;
      setTimeout(pollRuns, 650);
    }
  };

  const pollRuns = async () => {
    app.polling = false;
    try {
      const data = await request("GET", "/run");
      await takeRuns(data.runs);
    } catch {
      app.polling = true;
      setTimeout(pollRuns, 1500);
    }
  };

  const refreshBackend = guard(async () => {
    const [bData, gData] = await Promise.all([
      request("GET", "/backend"),
      request("GET", "/gemini-key"),
    ]);
    app.backend = bData.backend;
    app.geminiKey = gData.gemini_key;
    renderHeroAndTiles();
    renderRuntimeTab();
    queueFit();
  });

  const resetBrowserSettings = () => {
    const clearStorage = (storage) => {
      try {
        if (!storage) return;
        Object.keys(storage)
          .filter((key) => key.startsWith("minimax-h3:"))
          .forEach((key) => storage.removeItem(key));
      } catch { /* Ignore storage access errors. */ }
    };
    clearStorage(window.localStorage);
    try {
      if (window.parent && window.parent !== window) clearStorage(window.parent.localStorage);
    } catch { /* Framed by another origin. */ }
    try {
      (window.top || window.parent || window).location.reload();
    } catch {
      window.location.reload();
    }
  };

  const bind = () => {
    $("section-tabs").addEventListener("click", (event) => {
      const button = event.target.closest("button[data-tab]");
      if (button) switchTab(button.dataset.tab);
    });

    $("refresh-backend").addEventListener("click", async () => {
      await refreshBackend();
      toast("Backend telemetry refreshed.");
    });

    $("unload-btn").addEventListener("click", guard(async () => {
      app.unloadMessage = "";
      await takeRuns((await post("/unload")).runs);
    }));

    $("drama3-model").addEventListener("change", (event) => {
      const d = app.drama3 || {};
      saveDrama3(d.backend, event.target.value, d.comfy_experts);
    });

    $("gemini-key-form").addEventListener("submit", guard(async (event) => {
      event.preventDefault();
      const input = $("gemini-key-input");
      const raw = input.value.trim();
      const data = await post("/gemini-key", {api_key: raw});
      input.value = "";
      app.geminiKey = data.gemini_key;
      renderRuntimeTab();
      toast(data.message);
    }));

    $("gemini-key-clear").addEventListener("click", guard(async () => {
      const data = await request("DELETE", "/gemini-key");
      $("gemini-key-input").value = "";
      app.geminiKey = data.gemini_key;
      renderRuntimeTab();
      toast(data.message);
    }));

    $("projects-refresh").addEventListener("click", guard(async () => {
      app.projects = await request("GET", "/projects");
      renderHeroAndTiles();
      renderLocalProjects();
      renderRemoteImport();
      toast(`Showing ${app.projects.projects.length} local project(s).`);
    }));

    $("local-search").addEventListener("input", (event) => {
      app.localQuery = event.target.value;
      renderLocalProjects();
      queueFit();
    });

    $("remote-load-form").addEventListener("submit", guard(async (event) => {
      event.preventDefault();
      const url = $("remote-url").value.trim();
      const data = await post("/projects/remote", {
        url,
        modal_key: $("modal-key").value,
        modal_secret: $("modal-secret").value,
      });
      app.remoteLoadedUrl = url;
      app.remoteServerApp = data.app || "remote server";
      app.remoteProjects = data.projects || [];
      app.remoteSelected = new Set();
      app.importResults = null;
      renderRemoteImport();
      queueFit();
      toast(`Loaded ${app.remoteProjects.length} project(s) from ${app.remoteServerApp}.`);
    }));

    $("remote-select-all").addEventListener("click", () => {
      app.remoteSelected = new Set(app.remoteProjects.map(projectKey));
      renderRemoteImport();
    });

    $("remote-clear").addEventListener("click", () => {
      app.remoteSelected = new Set();
      renderRemoteImport();
    });

    $("remote-import-btn").addEventListener("click", guard(async () => {
      const items = app.remoteProjects
        .filter((rp) => app.remoteSelected.has(projectKey(rp)))
        .map((rp) => ({kind: rp.kind, id: rp.id}));
      app.importResults = null;
      await takeRuns((await post("/projects/import", {
        url: app.remoteLoadedUrl,
        items,
        conflict: app.conflict,
        modal_key: $("modal-key").value,
        modal_secret: $("modal-secret").value,
      })).runs);
    }));

    $("models-refresh").addEventListener("click", guard(async () => {
      app.models = await request("GET", "/models?refresh=1");
      renderHeroAndTiles();
      renderModelsAndEngines();
      queueFit();
      toast(`Local model presence refreshed (${app.models.present_count}/${app.models.total_count} local).`);
    }));

    $("models-family").addEventListener("change", (event) => {
      app.familyFilter = event.target.value;
      renderModelsAndEngines();
      queueFit();
    });

    $("models-search").addEventListener("input", (event) => {
      app.modelQuery = event.target.value;
      renderModelsAndEngines();
      queueFit();
    });

    $("scan-btn").addEventListener("click", guard(async () => {
      await takeRuns((await post("/scan")).runs);
    }));

    $("reset-arm").addEventListener("click", () => {
      $("reset-idle").hidden = true;
      $("reset-confirm").hidden = false;
      queueFit();
    });

    $("reset-cancel").addEventListener("click", () => {
      $("reset-confirm").hidden = true;
      $("reset-idle").hidden = false;
      queueFit();
    });

    $("reset-go").addEventListener("click", resetBrowserSettings);
  };

  const start = async () => {
    applyTheme();
    if (window.frameElement) document.documentElement.dataset.framed = "true";
    try {
      new MutationObserver(applyTheme).observe(window.parent.document.body, {
        attributes: true,
        attributeFilter: ["class"],
      });
    } catch { /* Framed by another origin. */ }

    new MutationObserver(queueFit).observe(document.body, {
      subtree: true,
      childList: true,
      attributes: true,
      attributeFilter: ["hidden"],
    });
    document.addEventListener("toggle", queueFit, true);
    window.addEventListener("resize", queueFit);

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
    whenShown(() => {
      if (app.booted) {
        guard(async () => {
          await refreshBackend();
          app.projects = await request("GET", "/projects");
          renderHeroAndTiles();
          renderLocalProjects();
          queueFit();
        })();
      }
    });

    try {
      const boot = await request("GET", "/boot");
      app.backend = boot.backend;
      app.drama3 = boot.drama3;
      app.geminiKey = boot.gemini_key;
      app.projects = boot.projects;
      app.models = boot.models;
      app.engines = boot.engines;
      app.booted = true;
      bind();
      switchTab("runtime");
      await takeRuns(boot.runs);
      $("loading").hidden = true;
      $("app").hidden = false;
      queueFit();
    } catch (error) {
      $("loading").textContent = `System couldn't open: ${error.message}`;
    }
  };

  start();
})();
