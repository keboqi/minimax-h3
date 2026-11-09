import {createSaves} from "./saves.js";
import {StaleResponse} from "./transport.js";

// Reconcile server snapshots with edits still owned by this browser.
export function createProjects({state, api, named, scope, renderSaveState, renderHeader, renderSteps, renderSettings, renderScript, queueEstimate, renderVideo, castAnswered, toast}) {
  // All saves share project ownership and a real completion barrier.
  const saves = createSaves({
    send: (name, key, body) => {
      const [area, kind, slot] = key.split(":");
      const path = area === "slot" ? `/project/slot?kind=${kind}&slot=${slot}` : `/project/${area}`;
      return api(named(path, name), {method: "PATCH", body, unscoped: true});
    },
    changed: (name) => {
      if (state.project?.name !== name) return;
      state.saving = saves.status(name);
      renderSaveState();
    },
    saved: (name, key, project) => {
      if (state.project?.name !== name) return;
      reconcile(project);
      renderHeader(); renderSteps(); renderSettings();
      if (key === "script") { renderScript(); queueEstimate(); }
      if (key === "video") renderVideo();
      if (key.startsWith("slot:")) castAnswered(project);
    },
    failed: (name, error) => { if (state.project?.name === name) toast(`Not saved: ${error.message}`); },
  });

  function reconcile(project) {
    if (!project || project.name !== state.project?.name) return false;
    Object.assign(state.project, {...project, run: state.project.run});
    for (const [key, body] of Object.entries(saves.edits(project.name))) {
      if (key === "script") {
        for (const [field, value] of Object.entries(body)) {
          if (field in state.project.script.writer) state.project.script.writer[field] = value;
          else state.project.script[field] = value;
        }
      } else if (key === "settings") Object.assign(state.project.settings, body);
      else if (key === "video" && state.project.video) {
        const names = {video_engine: "engine", video_style: "style", video_limit: "limit",
          video_continuity: "continuity", video_subtitles: "subtitles", video_subtitle_size: "subtitle_size"};
        for (const [field, value] of Object.entries(body)) state.project.video[names[field]] = value;
      } else if (key.startsWith("slot:")) {
        const [, kind, index] = key.split(":");
        const list = kind === "voice" ? state.project.speakers : state.project.sounds;
        const slot = list.find((item) => item.slot === Number(index));
        if (slot) Object.assign(slot, body);
      }
    }
    return true;
  }

  function saveSettings(values) { saves.queue(state.project.name, "settings", values, 0); }

  async function flushEdits() {
    if (!state.project) return;
    const before = scope(), name = state.project.name;
    await saves.flush(name);
    if (before !== scope()) throw new StaleResponse("The project changed while saving.");
  }

  return {saves, reconcile, saveSettings, flushEdits};
}
