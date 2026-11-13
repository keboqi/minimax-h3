// Batch produce: run Auto-produce on several projects in turn, on the server.
// Projects can be prepared in the CPU Workshop; the batch itself starts on the GPU endpoint.
export function createBatch({state, $, h, api, toast, NO_GPU, flushEdits, isRunning, writerService, openProject, refreshProjects}) {
  const POLL_MS = 3000;
  const LABELS = {waiting: "Waiting", producing: "Producing", done: "Done", failed: "Failed", stopped: "Stopped"};
  const STAGES = {script: "Script", cast: "Voices and sounds", speak: "Speaking"};
  // ``followed`` is the batch and project the page last followed, so it follows each once.
  const batchState = {batch: null, projects: null, selected: new Set(), timer: null, followed: null};

  const plural = (count, word) => `${count} ${word}${count === 1 ? "" : "s"}`;

  function summary(batch) {
    const current = batch.items.find((item) => item.name === batch.current);
    return `Batch ${Math.min(batch.finished + 1, batch.items.length)}/${batch.items.length}`
      + (current ? ` · ${current.name}` : "") + (batch.stopping ? " · stopping" : "");
  }

  function projectNote(project) {
    if (!project.lines) return {text: "No script yet: write one on Script first.", warn: true};
    if (project.busy) return {text: "Running something now: wait for it to finish.", warn: true};
    const notes = [project.prepared ? "Prepared script" : "The writer rewrites the script first"];
    const making = [project.voices && plural(project.voices, "voice"), project.sounds && plural(project.sounds, "sound")].filter(Boolean);
    if (making.length) notes.push(`makes ${making.join(" and ")}`);
    else if (project.prepared) notes.push("every voice and sound is ready");
    notes.push(project.generated ? "spoken before, so it speaks again" : "then speaks");
    return {text: notes.join(" · "), warn: false};
  }

  const selectable = (project) => project.lines > 0 && !project.busy;

  function renderChip() {
    const chip = $("batch-state");
    const batch = batchState.batch;
    chip.hidden = !batch?.active;
    if (batch?.active) {
      chip.textContent = summary(batch);
      chip.title = batch.status ? `${STAGES[batch.stage] || "Starting"}: ${batch.status}` : "";
    }
  }

  function queueView(batch) {
    const nodes = [h("h3", {}, batch.active ? "Producing" : "Last batch"),
      h("div", {class: "batch-list"}, batch.items.map((item) => {
        const live = batch.active && item.name === batch.current;
        const note = live && batch.status ? `${STAGES[batch.stage] || "Starting"}: ${batch.status}` : item.detail;
        return h("div", {class: "batch-row"},
          h("div", {class: "grow"},
            h("div", {class: "name"}, item.name),
            note ? h("div", {class: item.state === "failed" ? "batch-note warn" : "batch-note"}, note) : null),
          h("span", {class: `batch-badge ${item.state}`}, LABELS[item.state] || item.state),
          h("button", {type: "button", class: "btn btn-small", disabled: item.name === state.project?.name,
            onclick: () => { $("batch-dialog").close(); openProject(item.name); }}, "Open"));
      }))];
    if (batch.error) nodes.push(h("p", {class: "batch-error", role: "status"}, batch.error));
    if (batch.active) {
      nodes.push(h("div", {class: "batch-tools"},
        h("button", {type: "button", class: "btn", disabled: batch.stopping, onclick: stopBatch},
          batch.stopping ? "Stopping…" : "Stop the batch")));
    }
    return nodes;
  }

  function pickerView() {
    const projects = batchState.projects;
    if (!projects) return [h("p", {class: "muted"}, "Loading projects…")];
    const known = new Set(projects.filter(selectable).map((project) => project.name));
    for (const name of [...batchState.selected]) if (!known.has(name)) batchState.selected.delete(name);
    const chosen = projects.filter((project) => batchState.selected.has(project.name));
    const toggle = (name, on) => {
      if (on) batchState.selected.add(name); else batchState.selected.delete(name);
      render();
    };
    const gpu = state.boot.gpu;
    return [
      h("div", {class: "batch-tools"},
        h("h3", {}, "Choose projects"),
        h("span", {class: "spacer"}),
        h("button", {type: "button", class: "btn btn-small", onclick: () => {
          projects.filter((project) => selectable(project) && !project.generated)
            .forEach((project) => batchState.selected.add(project.name));
          render();
        }}, "Select all not yet spoken"),
        h("button", {type: "button", class: "btn btn-small", disabled: !chosen.length,
          onclick: () => { batchState.selected.clear(); render(); }}, "Clear")),
      projects.length ? h("div", {class: "batch-list"}, projects.map((project) => {
        const note = projectNote(project);
        return h("label", {class: "batch-row"},
          h("input", {type: "checkbox", checked: batchState.selected.has(project.name), disabled: !selectable(project),
            onchange: (event) => toggle(project.name, event.target.checked)}),
          h("div", {class: "grow"},
            h("div", {class: "name"}, project.name),
            h("div", {class: note.warn ? "batch-note warn" : "batch-note"}, note.text)));
      })) : h("p", {class: "muted"}, "No projects yet."),
      h("p", {class: "small muted"}, "Projects are produced in the order listed, by name, so Chapter 2 comes before Chapter 10. A speaker the group already has a voice for uses it, including voices an earlier project in the batch saved. A project the writer hasn't prepared is rewritten with the writer service set on Script in this tab."),
      gpu ? null : h("p", {class: "small batch-note warn"}, "Prepare only here: open Soundstage in Studio to start the batch."),
      h("div", {class: "batch-tools"},
        h("span", {class: "spacer"}),
        h("button", {type: "button", class: "btn btn-primary", disabled: !chosen.length || !gpu,
          title: gpu ? "" : NO_GPU, onclick: startBatch},
          chosen.length ? `Produce ${plural(chosen.length, "project")}` : "Select projects to produce")),
    ];
  }

  function render() {
    renderChip();
    if (!$("batch-dialog").open) return;
    const batch = batchState.batch;
    const nodes = [h("p", {class: "small muted"}, "Runs Auto-produce on each selected project in turn: rewrite the script if it needs it, make every missing voice and sound, then speak. It runs on the server, so this page can be closed; Jobs lists each step.")];
    if (batch) nodes.push(...queueView(batch));
    if (!batch?.active) nodes.push(...pickerView());
    $("batch-body").replaceChildren(...nodes.filter(Boolean));
  }

  async function load() {
    try {
      const data = await api("/batch", {unscoped: true});
      batchState.projects = data.projects;
      follow(data.batch);
    } catch (error) {
      toast(error.message);
    }
    render();
  }

  async function openBatch() {
    batchState.projects = null;
    $("batch-dialog").showModal();
    render();
    try { await flushEdits(); } catch { /* The list still shows; a failed save says so itself. */ }
    await load();
  }

  function announce(batch) {
    const count = (name) => batch.items.filter((item) => item.state === name).length;
    const parts = [`${count("done")} done`];
    if (count("failed")) parts.push(`${count("failed")} failed`);
    if (count("stopped")) parts.push(`${count("stopped")} stopped`);
    toast(`Batch finished: ${parts.join(", ")}.`);
  }

  function follow(batch) {
    const before = batchState.batch;
    batchState.batch = batch;
    if (batch?.active && !batchState.timer) batchState.timer = setTimeout(poll, POLL_MS);
    const same = before && batch && before.id === batch.id;
    if (same && before.active && !batch.active) {
      announce(batch);
      refreshProjects();
      if (state.project?.name === before.current) openProject(before.current);
    }
    const step = batch?.active && batch.current ? `${batch.id}:${batch.current}` : null;
    if (step && step !== batchState.followed && state.project) {
      batchState.followed = step;
      // The page moves on with the queue while it shows the project the queue just left, and
      // picks up the run the queue started on the project it shows.
      if (same && state.project.name === before.current) openProject(batch.current);
      else if (state.project.name === batch.current && !isRunning()) openProject(batch.current);
    }
    render();
  }

  async function poll() {
    batchState.timer = null;
    let batch;
    try {
      ({batch} = await api("/batch?projects=false", {unscoped: true}));
    } catch {
      batchState.timer = setTimeout(poll, POLL_MS * 2);
      return;
    }
    const wasActive = batchState.batch?.active;
    follow(batch);
    if (wasActive && !batch?.active && $("batch-dialog").open) load();
  }

  async function startBatch() {
    const names = batchState.projects.filter((project) => batchState.selected.has(project.name)).map((project) => project.name);
    if (!names.length) return;
    try {
      await flushEdits();
      const data = await api("/batch", {method: "POST", body: {names, writer: writerService()}, unscoped: true});
      batchState.selected.clear();
      toast(`Batch started: ${plural(names.length, "project")}.`);
      follow(data.batch);
    } catch (error) {
      toast(error.message);
    }
  }

  async function stopBatch() {
    try {
      const data = await api("/batch/stop", {method: "POST", unscoped: true});
      toast(data.message);
      follow(data.batch);
    } catch (error) {
      toast(error.message);
    }
  }

  /** Pick up a batch that was already running when the page opened. */
  async function resume() {
    try {
      const {batch} = await api("/batch?projects=false", {unscoped: true});
      batchState.batch = batch;
      if (batch?.active && !batchState.timer) batchState.timer = setTimeout(poll, POLL_MS);
      renderChip();
    } catch { /* The Batch button still opens it. */ }
  }

  function bind() {
    $("batch-open").addEventListener("click", openBatch);
    $("batch-state").addEventListener("click", openBatch);
    $("batch-close").addEventListener("click", () => $("batch-dialog").close());
  }

  return {bind, resume, openBatch, batchState};
}
