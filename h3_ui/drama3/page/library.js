// Library view and local controls for Soundstage.
export function createLibrary({state, $, h, API, api, castAnswered, loadLibrary, renderSlotEditor, field, toast, poller}) {
  const libraryState = {tab: "speakers", group: "Global", data: null, picked: null, takes: null};

  async function openLibrary() {
    libraryState.group = state.project.cast_settings.speaker_group || "Global";
    libraryState.picked = null;
    libraryState.takes = null;
    $("library-dialog").showModal();
    await refreshLibrary();
  }

  async function refreshLibrary() {
    try {
      libraryState.data = await api(`/library/manage?group=${encodeURIComponent(libraryState.group)}`);
    } catch (error) {
      libraryState.data = {groups: ["Global"], speakers: [], sounds: [], error: error.message};
    }
    if (libraryState.picked && !libraryState.data.speakers.some((speaker) => speaker.key === libraryState.picked)) {
      libraryState.picked = null;
    }
    renderLibrary();
  }

  /** A library change also changed the project when it used that voice: show both. */
  async function libraryCall(path, body) {
    try {
      const data = await api(path, {method: "POST", body: {...body, project: state.project.name}});
      if (data.project) castAnswered(data.project, {editor: true});
      if (data.key !== undefined) libraryState.picked = data.key;
      if (data.message) toast(data.message);
      await loadLibrary(state.project.cast_settings.speaker_group);
      await refreshLibrary();
      if (state.screen === "cast") renderSlotEditor(true);
      return data;
    } catch (error) {
      toast(error.message);
      return null;
    }
  }

  function renderLibrary() {
    const data = libraryState.data;
    $("library-tabs").replaceChildren(...[["speakers", "Saved speakers"], ["sounds", "Saved sounds"]].map(([id, label]) =>
      h("button", {type: "button", "aria-pressed": String(libraryState.tab === id),
                   onclick: () => { libraryState.tab = id; renderLibrary(); }}, label)));
    const body = $("library-body");
    if (!data) { body.replaceChildren(h("p", {class: "muted"}, "Loading…")); return; }
    if (data.error) { body.replaceChildren(h("p", {class: "slot-note error"}, data.error)); return; }
    body.replaceChildren(...(libraryState.tab === "speakers" ? speakerLibrary(data) : soundLibrary(data)));
  }

  function speakerLibrary(data) {
    const picked = data.speakers.find((speaker) => speaker.key === libraryState.picked);
    const list = h("div", {class: "library-list"},
      h("label", {class: "field"}, h("span", {class: "label"}, "Group"),
        h("select", {onchange: (event) => { libraryState.group = event.target.value; libraryState.picked = null; refreshLibrary(); }},
          data.groups.map((group) => h("option", {selected: group === libraryState.group}, group)))),
      data.speakers.length ? h("ul", {}, data.speakers.map((speaker) => h("li", {},
        h("button", {type: "button", class: "library-item", "aria-pressed": String(speaker.key === libraryState.picked),
                     onclick: () => { libraryState.picked = speaker.key; libraryState.takes = null; renderLibrary(); }},
          h("strong", {}, speaker.name), h("span", {}, speaker.description || "No description")))))
        : h("p", {class: "muted small"}, `No saved speakers in ${libraryState.group}.`),
      libraryState.group !== "Global" ? h("button", {type: "button", class: "btn btn-small danger", onclick: () => {
        if (confirm(`Delete the group “${libraryState.group}” and all ${data.speakers.length} of its saved speakers? This can't be undone.`)) {
          const group = libraryState.group;
          libraryState.group = "Global";
          libraryCall("/library/group/delete", {group});
        }
      }}, "Delete this group") : null);
    if (!picked) {
      return [list, h("div", {class: "library-detail"}, h("p", {class: "muted"},
        "Pick a saved speaker to hear it, edit it, copy it to another group or try new voices for it."))];
    }
    const name = h("input", {value: picked.name});
    const description = h("textarea", {rows: 3, value: picked.description});
    const transcript = h("textarea", {rows: 3, value: picked.transcript});
    const destination = h("select", {}, data.groups.filter((group) => group !== picked.group).map((group) => h("option", {}, group)));
    const count = h("select", {"aria-label": "How many voices"}, [1, 2, 3, 4].map((n) => h("option", {value: n}, `${n} voice${n > 1 ? "s" : ""}`)));
    const takes = libraryState.takes;
    const working = takes?.active;
    return [list, h("div", {class: "library-detail"},
      h("audio", {controls: true, preload: "none", src: `${API}/library/clip?kind=voice&key=${encodeURIComponent(picked.key)}&t=${Date.now()}`}),
      field("Name", name), field("Voice description", description), field("Exact clip transcript", transcript),
      h("div", {class: "row-buttons"},
        h("button", {type: "button", class: "btn btn-primary btn-small", onclick: () => libraryCall("/library/speaker/update", {
          key: picked.key, name: name.value, description: description.value, transcript: transcript.value})}, "Save changes"),
        h("button", {type: "button", class: "btn btn-small danger", onclick: () => {
          if (confirm(`Delete the saved speaker “${picked.name}” from ${picked.group}? Speakers using it lose their voice. This can't be undone.`)) {
            libraryCall("/library/speaker/delete", {key: picked.key});
          }
        }}, "Delete")),
      destination.children.length ? h("div", {class: "row-buttons"}, destination,
        h("button", {type: "button", class: "btn btn-small", onclick: () => libraryCall("/library/speaker/copy", {key: picked.key, group: destination.value})},
          "Copy to group")) : null,
      h("div", {class: "library-section"},
        h("h2", {}, "Try new voices"),
        h("p", {class: "muted small"}, "Designs new voices from this description and transcript with the Cast's voice generator. "
          + "The saved clip stays until you keep one."),
        h("div", {class: "row-buttons"}, count,
          working ? h("button", {type: "button", class: "btn btn-small", onclick: () => api("/library/takes/interrupt", {method: "POST", body: {run: takes.id}}).then((reply) => toast(reply.message), (error) => toast(error.message))}, "Interrupt")
            : h("button", {type: "button", class: "btn btn-small", "data-gpu": true, onclick: () => tryVoices(picked, Number(count.value), description.value, transcript.value)},
                "Generate new voices")),
        takes && takes.key === picked.key ? h("p", {class: `slot-note${takes.error ? " error" : ""}`}, takes.error || takes.status) : null,
        takes && takes.key === picked.key ? h("div", {class: "takes"}, takes.takes.map((url, index) => url ? h("div", {class: "take-row"},
          h("span", {class: "small"}, `Voice ${index + 1}`), h("audio", {controls: true, preload: "none", src: url}),
          h("button", {type: "button", class: "btn btn-small", disabled: working,
                       onclick: () => libraryCall("/library/speaker/use-take", {run: takes.id, index}).then(() => { libraryState.takes = null; renderLibrary(); })},
            "Keep this voice")) : null)) : null))];
  }

  function soundLibrary(data) {
    if (!data.sounds.length) {
      return [h("p", {class: "muted"}, "No saved sounds yet. Save a sound's clip on Cast & sounds to reuse it.")];
    }
    return [h("div", {class: "library-detail"}, data.sounds.map((sound) => h("div", {class: "sound-row"},
      h("div", {}, h("strong", {}, sound.name), h("p", {class: "muted small"}, `${sound.duration} s · ${sound.description || "No description"}`)),
      h("audio", {controls: true, preload: "none", src: `${API}/library/clip?kind=sound&key=${encodeURIComponent(sound.name)}`}),
      h("button", {type: "button", class: "btn btn-small danger", onclick: () => {
        if (confirm(`Delete the saved sound “${sound.name}”? This can't be undone.`)) libraryCall("/library/sound/delete", {name: sound.name});
      }}, "Delete"))))];
  }

  async function tryVoices(picked, count, description, transcript) {
    try {
      const data = await api("/library/speaker/takes", {method: "POST", body: {
        key: picked.key, count, description, transcript, project: state.project.name}});
      libraryState.takes = data.takes_run;
      renderLibrary();
      pollTakes();
    } catch (error) {
      toast(error.message);
    }
  }

  const pollTakes = poller(async (again) => {
    const run = libraryState.takes;
    if (!run) return;
    const data = await api(`/library/takes?run=${encodeURIComponent(run.id)}`);
    libraryState.takes = data.takes_run;
    if ($("library-dialog").open) renderLibrary();
    if (data.takes_run.active) again();
  }, {retry: false});

  return {openLibrary, refreshLibrary, libraryCall, renderLibrary, speakerLibrary, soundLibrary, tryVoices, libraryState, pollTakes};
}
