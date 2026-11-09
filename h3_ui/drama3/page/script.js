// Script view and local controls for Soundstage.
export function createScript({state, $, h, toneColor, saveScript, writerService, storeWriterService, reviewWriterProposal, undoWriter, loadExample, renderSaveState}) {
  const NAME = /^(\s*)([^\s:：<>[\]{}|][^:：<>[\]{}|\n]{0,39}?)(\s*[:：])/;
  // A sound tag is colored with its description: <sound name="rain">Steady rain</sound>.
  const TOKEN = /(<sound\b[^>]*>(?:[^<\n]*<\/sound>)?|<\/sound>)|(<\|[^|]*\|>|<[^>]*>)|(\[[^\]\n]*\])|(\{[^{}\n]*\})/g;
  const escapeHtml = (text) => text.replace(/[&<>"]/g, (char) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"})[char]);

  /** Speakers in the order the script first names them, with the result view's colors. */
  function scriptSpeakers(text) {
    const narrators = new Set(state.boot.narrators);
    const order = new Map();
    for (const line of text.split("\n")) {
      const match = NAME.exec(line);
      if (!match) continue;
      const name = match[2].trim();
      const entry = order.get(name) || {name, count: 0};
      entry.count += 1;
      order.set(name, entry);
    }
    let character = 0;
    const cast = new Set(state.project.cast.map((speaker) => speaker.name.trim().toLowerCase()));
    const numbered = /^speaker\s+\d+$/i;
    return [...order.values()].map((entry) => {
      const narrator = narrators.has(entry.name.toLowerCase());
      const tone = narrator ? "narrator" : String(character++ % 8);
      // A name the cast doesn't have is an error when speaking; "Speaker 2" means slot 2.
      const known = !state.project.cast.length || cast.has(entry.name.toLowerCase()) || numbered.test(entry.name);
      return {...entry, tone, known};
    });
  }

  function highlight(text, speakers) {
    const byName = new Map(speakers.map((speaker) => [speaker.name, speaker]));
    // A trailing newline needs a line of its own in the mirror, or the caret's last line jumps.
    return text.split("\n").map((line) => {
      let lead = "";
      const match = NAME.exec(line);
      if (match) {
        const speaker = byName.get(match[2].trim());
        const style = speaker ? ` style="color: var(--tone-${speaker.tone})"` : "";
        const unknown = speaker && !speaker.known ? " tok-unknown" : "";
        lead = `${escapeHtml(match[1])}<span class="tok-name${unknown}"${style}>${escapeHtml(match[2])}</span>${escapeHtml(match[3])}`;
        line = line.slice(match[0].length);
      }
      let out = "", cursor = 0;
      for (const token of line.matchAll(TOKEN)) {
        out += escapeHtml(line.slice(cursor, token.index));
        const kind = token[1] ? "sound" : token[2] ? "tag" : token[3] ? "cue" : "effect";
        out += `<span class="tok-${kind}">${escapeHtml(token[0])}</span>`;
        cursor = token.index + token[0].length;
      }
      return lead + out + escapeHtml(line.slice(cursor));
    }).join("\n") + "\n";
  }

  function paintEditor() {
    const area = $("script-text");
    const speakers = scriptSpeakers(area.value);
    $("script-mirror").innerHTML = highlight(area.value, speakers);
    area.style.height = "auto";
    area.style.height = `${Math.max(420, area.scrollHeight)}px`;
    $("speakers").replaceChildren(...speakers.map((speaker) => h("span", {
      class: `speaker-chip${speaker.known ? "" : " unknown"}`,
      title: speaker.known ? `${speaker.count} line${speaker.count > 1 ? "s" : ""}` : "Not in the cast: this line can't be spoken",
    }, h("i", {style: {background: toneColor(speaker.tone)}}), speaker.name,
       h("span", {class: "count"}, speaker.known ? speaker.count : "not in cast"))));
  }

  function renderWriter() {
    const writer = state.project.script.writer;
    const choose = (body) => () => saveScript(body, {now: true});
    $("speaker-mode").replaceChildren(...state.boot.speaker_modes.map((mode) => h("button", {
      type: "button", "aria-pressed": String(writer.speaker_mode === mode), onclick: choose({speaker_mode: mode}),
    }, mode)));
    $("workflows").replaceChildren(...state.boot.workflows.map((workflow) => h("button", {
      type: "button", class: "workflow", "aria-pressed": String(writer.writer_workflow === workflow.id),
      onclick: choose({writer_workflow: workflow.id}),
    }, h("strong", {}, workflow.id.replace(/^Guided \((.*)\)$/, (_, kind) => kind[0].toUpperCase() + kind.slice(1))),
       h("span", {}, workflow.note))));
    $("strict").checked = writer.strict_source;
    const batch = state.boot.batch;
    const guided = writer.writer_workflow !== state.boot.workflows[0].id;
    const smart = writer.writer_workflow === state.boot.smart_workflow;
    const count = h("span", {class: "num small"}, writer.max_batch_lines);
    const service = writerService();
    const update = (key) => (event) => storeWriterService({[key]: event.target.value});
    const backend = service.prompt_writer_backend || "Gemini";
    $("writer-more").replaceChildren(...[
      h("label", {class: "field"}, h("span", {class: "label"}, "Writer service"),
        h("select", {onchange: (event) => { storeWriterService({prompt_writer_backend: event.target.value}); renderWriter(); }},
          ["Gemini", "OpenAI-compatible"].map((name) => h("option", {selected: name === backend}, name)))),
      backend === "Gemini" ? [
        h("label", {class: "field"}, h("span", {class: "label"}, "Gemini model"),
          h("select", {onchange: update("gemini_prompt_model")}, state.boot.writer.gemini_models.map((name) =>
            h("option", {selected: name === (service.gemini_prompt_model || state.boot.writer.gemini_model)}, name)))),
        h("label", {class: "field"}, h("span", {class: "label"}, "Gemini API key (optional)"),
          h("input", {type: "password", autocomplete: "off", value: service.gemini_api_key || "", oninput: update("gemini_api_key")})),
      ] : [
        h("label", {class: "field"}, h("span", {class: "label"}, "Model"),
          h("input", {value: service.openai_model || "", placeholder: state.boot.writer.openai_models[0], oninput: update("openai_model")})),
        h("label", {class: "field"}, h("span", {class: "label"}, "API key (optional)"),
          h("input", {type: "password", autocomplete: "off", value: service.openai_api_key || "", oninput: update("openai_api_key")})),
        h("label", {class: "field"}, h("span", {class: "label"}, "Base URL (optional)"),
          h("input", {value: service.openai_base_url || "", oninput: update("openai_base_url")})),
      ],
      h("p", {class: "muted small"}, "Leave a key empty to use the server's. Keys stay in this browser tab only."),
      guided ? h("label", {class: "field"}, h("span", {class: "label"}, "Guided batch lines"),
        h("span", {class: "range-row"},
          h("input", {type: "range", min: batch.min, max: batch.max, step: 5, value: writer.max_batch_lines,
                      oninput: (event) => { count.textContent = event.target.value; },
                      onchange: (event) => saveScript({max_batch_lines: Number(event.target.value)}, {now: true})}),
          count)) : null,
      smart ? h("label", {class: "field"}, h("span", {class: "label"}, "Smart budget profile"),
        h("select", {onchange: (event) => saveScript({smart_budget_profile: event.target.value}, {now: true})},
          state.boot.budget_profiles.map((name) => h("option", {selected: name === writer.smart_budget_profile}, name)))) : null,
    ].flat().filter(Boolean));
    const rewrite = $("rewrite");
    rewrite.disabled = state.writing;
    if (!state.writing) rewrite.textContent = "Rewrite script";
  }

  function renderWriterResult() {
    const box = $("writer-result");
    const result = state.writerResult;
    box.hidden = !result && !state.project.script.proposal;
    if (!result) {
      box.replaceChildren(h("button", {type: "button", class: "btn btn-small", onclick: reviewWriterProposal}, "Review saved writer answer"));
      return;
    }
    box.classList.toggle("failed", result.failed);
    box.replaceChildren(
      h("div", {class: "writer-result-head"},
        h("strong", {}, result.failed ? "Script unchanged" : result.label),
        h("span", {class: "spacer"}),
        result.previous !== undefined && !result.failed
          ? h("button", {type: "button", class: "btn btn-small", onclick: undoWriter}, "Undo") : null,
        h("button", {type: "button", class: "btn btn-icon", "aria-label": "Dismiss",
                     onclick: () => { state.writerResult = null; renderWriterResult(); }}, "×")),
      h("p", {}, result.status),
      state.project.script.proposal ? h("button", {type: "button", class: "btn btn-small", onclick: reviewWriterProposal},
        "Review saved answer") : null);
  }

  function renderScript() {
    const script = state.project.script;
    const area = $("script-text");
    // Never replace what is being typed; a saved answer from the server is the same text anyway.
    if (document.activeElement !== area || state.saving === "saved") area.value = script.text;
    if (document.activeElement !== $("title")) $("title").value = script.title;
    $("example-items").replaceChildren(...state.boot.examples.map((example) => h("button", {
      type: "button", onclick: () => loadExample(example),
    }, example.label)));
    paintEditor();
    renderWriter();
    renderWriterResult();
    renderSaveState();
  }

  return {scriptSpeakers, highlight, paintEditor, renderWriter, renderWriterResult, renderScript, NAME, TOKEN};
}
