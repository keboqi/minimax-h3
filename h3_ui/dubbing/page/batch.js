import {api} from "./api.js";
import {esc, toast} from "./ui.js";

export function bindBatch(app) {
  const $ = (id) => document.getElementById(id);
  const selected = new Set();
  let timer, batch, loading = false, starting = false, pickerKey = "";

  function renderButtons() {
    $("batch-start").disabled = !app.boot.gpu || !!batch?.active || starting || !selected.size;
    $("batch-stop").disabled = !batch?.active || batch.stopping;
  }

  function render(projects) {
    const known = new Set(projects);
    for (const name of selected) if (!known.has(name)) selected.delete(name);
    $("batch-open").textContent = batch?.active ? `Batch ${batch.finished}/${batch.items.length}` : "Batch generate";
    // Leave the picker mounted between polls so keyboard focus stays on its checkbox.
    const key = JSON.stringify([projects, !!batch?.active]);
    if (key !== pickerKey) {
      pickerKey = key;
      $("batch-projects").innerHTML = projects.map((name) => `<label class="check">
        <input type="checkbox" data-project="${esc(name)}" ${selected.has(name) ? "checked" : ""} ${batch?.active ? "disabled" : ""}>${esc(name)}</label>`).join("") || "No saved projects. Save a project first.";
    }
    renderButtons();
    $("batch-progress").innerHTML = batch ? batch.items.map((item) => `<div>
      <div class="row"><strong class="grow">${esc(item.name)}</strong><span>${esc(item.state)}</span>
        <button type="button" class="btn btn-small" data-open-project="${esc(item.name)}">Open</button></div>
      <p class="muted small">${esc(item.id === batch.current && batch.active ? batch.status : item.detail)}</p></div>`).join("")
      + (batch.error ? `<p>${esc(batch.error)}</p>` : "") : "";
  }

  async function refresh() {
    if (loading) return;
    clearTimeout(timer);
    loading = true;
    try {
      const data = await api.get("batch");
      batch = data.batch;
      if (batch?.active) app.watchRuns();
      render(data.projects);
    } catch (error) {
      toast(error.message, {error: true});
    } finally {
      loading = false;
      if (batch?.active || $("batch-dialog").open) timer = setTimeout(refresh, 2000);
    }
  }

  $("batch-open").addEventListener("click", () => { $("batch-dialog").showModal(); refresh(); });
  $("batch-close").addEventListener("click", () => $("batch-dialog").close());
  $("batch-projects").addEventListener("change", (event) => {
    const name = event.target.dataset.project;
    if (!name) return;
    if (event.target.checked) selected.add(name); else selected.delete(name);
    renderButtons();
  });
  $("batch-progress").addEventListener("click", (event) => {
    const button = event.target.closest("[data-open-project]");
    if (!button) return;
    $("batch-dialog").close();
    app.openProject(button.dataset.openProject);
  });
  $("batch-start").addEventListener("click", async () => {
    starting = true;
    renderButtons();
    try {
      await api.post("batch", {names: [...selected], api_key: app.apiKey});
      app.watchRuns();
    } catch (error) {
      toast(error.message, {error: true});
    } finally {
      starting = false;
      await refresh();
      renderButtons();
    }
  });
  $("batch-stop").addEventListener("click", async () => {
    try { toast((await api.post("batch/stop")).message); }
    catch (error) { toast(error.message, {error: true}); }
    await refresh();
  });
  refresh();
}
