// The Illustrate step: the shot plan and its characters and places, drawn into stills that slide
// along the mix in the shared player. Save to Media renders them as a video from the player.
export function createVideo({state, $, h, mixAudio, mixVideo, clock, saveVideo, seek, renderPlayer, sync, saveVideoToMedia, planShots, drawImages, drawPreviews, interruptVideo, copyPlannerPrompt, applyPlannerReply, editShot, editSubject, show, toneColor, speakerTone, getPlayer, setPlayer}) {
  const STATUS = {drawn: ["Drawn", "status-drawn"], changed: ["Prompt changed", "status-changed"], todo: ["Not drawn", "status-todo"]};
  const KINDS = [["characters", "Characters"], ["locations", "Places"], ["props", "Props"]];

  function videoRunning(kind) {
    const run = state.videoRun;
    return Boolean(run?.active && (!kind || run.kind === kind));
  }

  /** What Video shows: the drawn stills, an older mix's rendered video, or nothing yet. */
  function visualSource() {
    const video = state.project.video;
    if (video?.stills?.shots?.length) return "stills";
    return video?.latest ? "file" : null;
  }

  /** Audio or Video in the player. The stills slide along the mix's own audio; a rendered video carries it. */
  function setMediaMode(mode) {
    const source = visualSource();
    mode = mode === "video" && source ? "video" : "audio";
    const player = mode === "video" && source === "file" ? mixVideo : mixAudio;
    if (mode === state.mediaMode && player === getPlayer()) return;
    const time = getPlayer().currentTime || 0;
    const wasPlaying = !getPlayer().paused;
    state.mediaMode = mode;
    if (player !== getPlayer()) {
      getPlayer().pause();
      setPlayer(player);
      try { getPlayer().currentTime = time; } catch { /* Not loaded yet; it starts from the top. */ }
      if (wasPlaying) getPlayer().play().catch(() => {});
    }
    renderPlayer();
    sync();
  }

  /** The header's video file: download the saved one, or render it from the stills for Media.
   *  The saved one is current when it was made from these stills with the subtitles shown now. */
  function videoFileButton() {
    const video = state.project.video;
    const saving = videoRunning("save");
    const subtitles = video.subtitles ? video.subtitle_size : 0;
    const current = video.latest?.current && (video.latest.subtitles ?? 0) === subtitles;
    if (video.latest && (current || !video.stills)) {
      return h("a", {class: "btn btn-small", id: "video-file", href: video.latest.url, download: video.latest.file,
                     title: "This video is in Media too."}, "Download video");
    }
    return h("button", {type: "button", class: "btn btn-small", id: "video-file", disabled: saving || videoRunning("draw"),
                        title: "Render a 720p video with the mix from these images, with the subtitles as shown, and add it to Media",
                        onclick: saveVideoToMedia}, saving ? "Saving to Media…" : "Save to Media");
  }

  const rowsOf = () => state.project.video?.plan?.rows || [];
  const needsDrawing = (rows) => rows.filter((row) => row.status !== "drawn").length;
  const plural = (count, word) => `${count} ${word}${count === 1 ? "" : "s"}`;
  const shotAt = (rows, time) => {
    let index = -1;
    for (const row of rows) if (row.start <= time + 0.01) index = row.index;
    return index;
  };

  /** Pick a shot: it opens in the panel, and the player jumps to it so its image shows there. */
  function pickShot(index, {jump = true} = {}) {
    const row = rowsOf()[index];
    if (!row) return;
    state.shotPick = index;
    if (jump) seek(row.start);
    if (state.screen === "illustrate") {
      renderIllustrate();
      // Bring the picked row into the list's view without moving the page.
      const list = $("shot-list");
      const item = list?.querySelector(`.shot-row[data-shot="${index}"]`);
      if (item && (item.offsetTop < list.scrollTop || item.offsetTop + item.offsetHeight > list.scrollTop + list.clientHeight)) {
        list.scrollTo({top: item.offsetTop - list.offsetTop - 8, behavior: "smooth"});
      }
    }
    for (const clip of document.querySelectorAll("#lanes .shot-clip")) {
      clip.setAttribute("aria-pressed", String(Number(clip.dataset.shot) === index));
    }
  }

  // The stills on the mix's clock: the shot under the playhead, dissolving in over the one before it.
  const preloaded = new Set();
  function paintStills(time) {
    const rows = rowsOf();
    const current = shotAt(rows, time);
    for (const clip of document.querySelectorAll("#lanes .shot-clip")) clip.classList.toggle("active", Number(clip.dataset.shot) === current);
    for (const item of document.querySelectorAll("#shot-list .shot-row")) item.classList.toggle("playing", Number(item.dataset.shot) === current);
    const back = $("still-back"), front = $("still-front");
    const shots = state.project.video?.stills?.shots;
    if (!back || !front || !shots?.length) return;
    let index = 0;
    while (index + 1 < shots.length && shots[index + 1].start <= time) index += 1;
    const shot = shots[index];
    const fading = index > 0 && shot.fade > 0 && time < shot.start + shot.fade;
    const show = (image, url) => { if (image.dataset.src !== url) { image.src = url; image.dataset.src = url; } };
    show(back, fading ? shots[index - 1].url : shot.url);
    if (fading) show(front, shot.url);
    front.style.opacity = fading ? String(Math.max(0, (time - shot.start) / shot.fade)) : "0";
    // Fetch the next images before they are due, so a cut never waits on the network.
    for (const next of shots.slice(index + 1, index + 3)) {
      if (!preloaded.has(next.url)) { preloaded.add(next.url); new Image().src = next.url; }
    }
  }

  /** The player's Shots lane: one block per planned shot, showing its image once drawn. */
  function shotLane(place) {
    const rows = rowsOf();
    if (!rows.length || (state.screen !== "illustrate" && state.mediaMode !== "video")) return null;
    const lane = h("div", {class: "lane shot-lane"}, rows.map((row) => h("button", {
      type: "button", class: `shot-clip shot-${row.status}`, "data-shot": row.index,
      "aria-pressed": String(row.index === state.shotPick),
      "aria-label": `Shot ${row.index + 1}, ${clock(row.start)}: ${row.beat} (${STATUS[row.status][0]})`,
      title: `Shot ${row.index + 1} · ${row.beat}`,
      style: {...place(row.start, row.end), ...(row.image ? {backgroundImage: `url("${row.image}")`} : {})},
      onclick: (event) => { event.stopPropagation(); pickShot(row.index); },
    }, h("span", {"aria-hidden": "true"}, row.index + 1))));
    return {name: h("span", {class: "lane-name shot-lane-name"}, "Images"), lane};
  }

  function videoField(label, input, hint) {
    return h("label", {class: "field"}, h("span", {class: "label"}, label), input,
             hint ? h("span", {class: "muted small"}, hint) : null);
  }

  function statusChip(status) {
    const [label, name] = STATUS[status];
    return h("span", {class: `status ${name}`}, label);
  }

  function thumb(row) {
    return h("span", {class: "shot-thumb"},
      row.image ? h("img", {src: row.image, alt: "", loading: "lazy"}) : h("span", {class: "shot-empty"}, "Not drawn"),
      h("span", {class: "shot-number"}, row.index + 1));
  }

  // -- The Mix & review pointer ----------------------------------------------
  function renderPointer() {
    const box = $("video-panel");
    const video = state.project.video;
    box.hidden = !state.boot.video || !video || !state.project.result;
    if (box.hidden) return;
    const rows = rowsOf();
    const needs = needsDrawing(rows);
    box.replaceChildren(
      h("div", {class: "video-pointer-text"},
        h("h2", {}, "Illustrated video"),
        h("span", {class: "muted small"}, !rows.length ? "Plan shots and draw them in the Illustrate step; they play in this player's Video view."
          : needs ? `${plural(rows.length, "shot")} planned, ${needs} still to draw. Video shows the drawn ones here.`
          : `${plural(rows.length, "shot")} drawn. Switch the player to Video to watch, or Save to Media.`)),
      h("button", {type: "button", class: "btn btn-small", onclick: () => show("illustrate")}, "Open Illustrate"));
  }

  // -- The Illustrate step ---------------------------------------------------
  function renderShotList(rows) {
    const video = state.project.video;
    const needs = needsDrawing(rows);
    if (state.shotFilter === "needs" && !needs) state.shotFilter = "all";
    const shown = state.shotFilter === "needs" ? rows.filter((row) => row.status !== "drawn") : rows;
    const head = h("div", {class: "seg", role: "group", "aria-label": "Show shots"},
      [["all", `All ${rows.length}`], ["needs", `Needs drawing ${needs}`]].map(([id, label]) => h("button", {
        type: "button", "aria-pressed": String(state.shotFilter === id), disabled: id === "needs" && !needs,
        onclick: () => { state.shotFilter = id; renderIllustrate(); }}, label)));
    if (!rows.length) {
      return [null, h("div", {class: "illustrate-empty"},
        h("strong", {}, video.plan.stale ? "The saved plan was made for other settings" : "No shot plan yet"),
        h("p", {class: "muted"}, video.plan.stale
          ? "It was planned for a different Art direction or Minimum shots. Plan again, or set them back in How it's drawn."
          : "Plan shots reads the timed mix, its Cast voices and the art direction, then picks one image per story beat."))];
    }
    const list = h("ol", {class: "shot-list", id: "shot-list"}, shown.map((row) => h("li", {},
      h("button", {type: "button", class: "shot-row", "data-shot": row.index, "aria-pressed": String(row.index === state.shotPick),
                   onclick: () => pickShot(row.index)},
        thumb(row),
        h("span", {class: "shot-body"},
          h("span", {class: "shot-meta"},
            h("span", {class: "num shot-time"}, `${clock(row.start)}–${clock(row.end)}`),
            h("span", {class: `kind ${row.transition === "Cut" ? "kind-shot" : "kind-loop"}`}, row.transition),
            statusChip(row.status)),
          h("span", {class: "shot-beat"}, row.beat),
          h("span", {class: "shot-prompt muted"}, row.prompt))))));
    return [head, list];
  }

  function renderLooks() {
    const plan = state.project.video.plan;
    if (!plan.subjects.length) {
      return h("div", {class: "illustrate-empty"}, h("strong", {}, plan.rows.length ? "No recurring characters, props or places" : "No shot plan yet"),
        h("p", {class: "muted"}, plan.rows.length ? "The planner found nothing that recurs across shots." : "Plan shots first: the planner writes how each character, prop and place looks."));
    }
    const busy = videoRunning();
    return h("div", {class: "looks"}, KINDS.map(([kind, title]) => {
      const items = plan.subjects.filter((item) => item.kind === kind);
      if (!items.length) return null;
      return h("section", {class: "looks-group", "aria-label": title},
        h("h3", {}, title),
        h("div", {class: `looks-grid looks-${kind}`}, items.map((item) => {
          const voice = kind === "characters" && item.voice ? h("div", {class: "look-voice"},
            h("span", {class: "label"}, `Voice from Cast & sounds · ${item.speaker}`),
            h("span", {class: "small"}, item.voice),
            item.conflicts?.length ? h("p", {class: "look-conflict", role: "alert"},
              h("strong", {}, "The looks disagree with the voice. "), item.conflicts.join(" "),
              " Edit the looks below, or the voice in Cast & sounds.") : null) : null;
          const art = kind === "characters"
            ? h("span", {class: "look-portrait"}, item.portrait ? h("img", {src: item.portrait, alt: `Portrait of ${item.key}`}) : h("span", {class: "shot-empty"}, "Drawn with its first shot"))
            : null;
          return h("article", {class: `look-card${item.conflicts?.length ? " look-warn" : ""}`},
            art,
            h("div", {class: "look-body"},
              h("div", {class: "look-head"},
                h("strong", {style: kind === "characters" && item.speaker ? {color: toneColor(speakerTone(item.speaker))} : {}}, item.key),
                h("span", {class: "muted small"}, `In ${plural(item.uses, "shot")}`),
                item.portrait && item.portrait_current === false ? h("span", {class: "status status-changed"}, "Looks changed") : null),
              voice,
              h("label", {class: "field"}, h("span", {class: "label"}, "Looks in every shot"),
                h("textarea", {rows: 3, value: item.anchor, disabled: busy,
                               onchange: (event) => editSubject({kind, key: item.key, anchor: event.target.value})}))));
        })));
    }));
  }

  function renderInspector(rows) {
    const box = $("shot-inspector");
    box.hidden = !rows.length;
    if (box.hidden) return;
    if (!rows[state.shotPick]) state.shotPick = 0;
    const row = rows[state.shotPick];
    const busy = videoRunning();
    const subjects = new Map(state.project.video.plan.subjects.map((item) => [item.key, item]));
    const step = (by) => pickShot(Math.min(rows.length - 1, Math.max(0, row.index + by)));
    const draw = {drawn: "Draw this shot again", changed: "Redraw with the new prompt", todo: "Draw this shot"}[row.status];
    box.replaceChildren(
      h("div", {class: "inspector-head shot-inspector-head"},
        h("h2", {}, `Shot ${row.index + 1}`),
        h("span", {class: "num"}, `${clock(row.start)}–${clock(row.end)}`),
        statusChip(row.status),
        h("span", {class: "spacer"}),
        h("button", {type: "button", class: "btn btn-icon", "aria-label": "Previous shot", disabled: row.index === 0, onclick: () => step(-1)}, "‹"),
        h("button", {type: "button", class: "btn btn-icon", "aria-label": "Next shot", disabled: row.index === rows.length - 1, onclick: () => step(1)}, "›")),
      h("p", {class: "muted small"}, row.image ? "The player shows this shot while the playhead is on it." : "Not drawn yet: the player holds the image before it."),
      h("div", {class: "field"}, h("span", {class: "label"}, "Story beat"), h("span", {class: "shot-inspector-beat"}, row.beat)),
      h("label", {class: "field"}, h("span", {class: "label"}, "Image prompt"),
        h("textarea", {rows: 6, value: row.prompt, disabled: busy, onchange: (event) => editShot({index: row.index, prompt: event.target.value})}),
        h("span", {class: "muted small"}, "@name brings in a character, prop or place as Characters & places describes it.")),
      row.subjects.length ? h("div", {class: "field"}, h("span", {class: "label"}, "Shows"),
        h("div", {class: "shot-subjects"}, row.subjects.map((key) => h("button", {
          type: "button", class: "chip chip-quiet", onclick: () => { state.illustrateView = "looks"; renderIllustrate(); },
          title: subjects.get(key)?.anchor || ""}, key)))) : null,
      h("div", {class: "field"}, h("span", {class: "label"}, "Comes in with"),
        h("div", {class: "seg", role: "group", "aria-label": "Transition"}, ["Cut", "Dissolve"].map((name) => h("button", {
          type: "button", "aria-pressed": String(row.transition === name), disabled: busy || (row.index === 0 && name === "Dissolve"),
          onclick: () => row.transition !== name && editShot({index: row.index, transition: name})}, name)))),
      h("button", {type: "button", class: "btn btn-primary", "data-gpu": true, disabled: busy || !state.boot.gpu,
                   onclick: () => drawImages([row.index])}, draw));
  }

  function pickStyle(name) {
    const style = state.boot.video.styles[name];
    // By text, so a project on a preset's earlier wording moves to the current one.
    if (style && style !== state.project.video.style) saveVideo({video_style: style});
  }

  const engineLabel = (video) => state.boot.video.engines.find((engine) => engine.id === video.engine)?.label || "this model";

  /** A preset's preview with this image model, or a placeholder (with Draw preview in the picker). */
  function styleThumb(video, name, {draw = false, busy = false} = {}) {
    const url = video.previews?.[name];
    if (url) return h("span", {class: "style-thumb"}, h("img", {src: url, alt: "", loading: "lazy"}));
    const text = state.boot.video.styles[name] ? "No preview yet" : "Your own art direction";
    return h("span", {class: "style-thumb style-thumb-empty"}, h("span", {}, text),
      draw ? h("button", {type: "button", class: "btn btn-small", "data-gpu": true, disabled: busy || !state.boot.gpu,
                          title: `Draw this preset's sample scene with ${engineLabel(video)}`,
                          onclick: (event) => { event.stopPropagation(); drawPreviews([name]); }}, "Draw preview") : null);
  }

  /** The panel's Style field: the current style with its preview; it opens the picker. */
  function styleField(video, busy) {
    const boot = state.boot.video;
    const name = video.preset;
    return h("div", {class: "field"}, h("span", {class: "label"}, "Style"),
      h("button", {type: "button", class: "style-current", disabled: busy, "aria-haspopup": "dialog",
                   title: "Compare the styles' previews and pick one", onclick: openStyles},
        styleThumb(video, name),
        h("span", {class: "style-current-text"},
          h("span", {class: "style-name"}, name),
          h("span", {class: "muted small"}, boot.style_uses[name] || "Edited Art direction"),
          h("span", {class: "style-change"}, "Change style…"))));
  }

  function openStyles() {
    renderStyles();
    $("style-dialog").showModal();
  }

  /** The style picker: every preset by use, large, with its preview for this image model. Picking one
   *  fills Art direction; presets without a preview can be drawn here, one or all missing at once. */
  function renderStyles() {
    const dialog = $("style-dialog");
    const boot = state.boot.video;
    const video = state.project?.video;
    if (!video) return;
    const busy = videoRunning();
    const drawing = videoRunning("preview");
    const previews = video.previews || {};
    const missing = Object.keys(boot.styles).filter((name) => !previews[name]).length;
    const status = drawing ? state.videoRun.status : busy ? "Wait for the current planning, drawing or saving to finish." :
      `${Object.keys(previews).length} of ${Object.keys(boot.styles).length} previewed with ${engineLabel(video)}. Previews are kept per image model for every project.`;
    const body = dialog.querySelector(".style-body");
    const scroll = body?.scrollTop || 0;
    dialog.replaceChildren(
      h("div", {class: "library-head"},
        h("div", {class: "style-head-text"}, h("h2", {id: "style-title"}, "Choose a style"),
          h("span", {class: "muted small"}, status)),
        h("span", {class: "spacer"}),
        missing ? h("button", {type: "button", class: "btn btn-small", "data-gpu": true, disabled: busy || !state.boot.gpu,
                               title: `Draw each preset without a preview with ${engineLabel(video)}, in one run`,
                               onclick: () => drawPreviews("missing")}, `Draw ${plural(missing, "missing preview")}`) : null,
        drawing ? h("button", {type: "button", class: "btn btn-small", onclick: interruptVideo}, "Interrupt") : null,
        h("button", {type: "button", class: "btn btn-icon", "aria-label": "Close", onclick: () => dialog.close()}, "×")),
      h("div", {class: "style-body"}, boot.style_groups.map((group) => h("section", {class: "style-group"},
        h("h3", {}, group.name),
        h("div", {class: "style-grid"}, group.presets.map((name) => h("div", {class: "style-card", "aria-current": name === video.preset ? "true" : null},
          h("button", {type: "button", class: "style-pick", disabled: busy, "aria-pressed": String(name === video.preset),
                       onclick: () => { pickStyle(name); dialog.close(); }},
            h("span", {class: "style-name"}, name, name === video.preset ? h("span", {class: "chip"}, "Current") : null),
            h("span", {class: "muted small"}, boot.style_uses[name])),
          styleThumb(video, name, {draw: true, busy}))))))));
    dialog.querySelector(".style-body").scrollTop = scroll;
  }

  function renderSettings() {
    const box = $("drawn-settings");
    const boot = state.boot.video;
    const video = state.project.video;
    const busy = videoRunning();
    const limitCount = h("span", {class: "num small"}, video.limit);
    if ($("style-dialog").open) renderStyles();
    box.replaceChildren(
      h("div", {class: "drawn-settings-head"}, h("h2", {}, "How it's drawn"),
        h("span", {class: "muted small"}, "Style and Minimum shots apply the next time you plan.")),
      videoField("Image model", h("select", {onchange: (event) => saveVideo({video_engine: event.target.value})},
        boot.engines.map((engine) => h("option", {value: engine.id, selected: engine.id === video.engine}, engine.label))),
        "Draws with the Image tab's settings for that model."),
      styleField(video, busy),
      videoField("Art direction", h("textarea", {rows: 4, value: video.style, oninput: (event) => saveVideo({video_style: event.target.value}, {later: true})}),
        "The medium and rendering across all shots. Editing makes the preset Custom."),
      videoField("Minimum shots", h("span", {class: "range-row"},
        h("input", {type: "range", min: boot.limits[0], max: boot.limits[1], step: 1, value: video.limit,
                    oninput: (event) => { limitCount.textContent = event.target.value; },
                    onchange: (event) => saveVideo({video_limit: Number(event.target.value)})}), limitCount),
        "A floor: the planner adds images wherever the story needs them, up to one per 2 seconds."),
      h("label", {class: "check"}, h("input", {type: "checkbox", checked: video.continuity,
        onchange: (event) => saveVideo({video_continuity: event.target.checked})}), "Keep characters consistent"),
      rowsOf().length ? h("button", {type: "button", class: "btn btn-small", "data-gpu": true, disabled: busy || !state.boot.gpu,
                                     title: "Draw every shot from scratch, with new portraits", onclick: () => drawImages("all")}, "Draw every shot again") : null,
      h("details", {class: "more"},
        h("summary", {}, "Use another chat model"),
        h("div", {class: "more-body"},
          h("p", {class: "muted small"}, "Copy the full planner prompt into any chat model, then paste its JSON reply. It is checked like the built-in planner's."),
          h("button", {type: "button", class: "btn btn-small", onclick: copyPlannerPrompt}, "Copy full planner prompt"),
          h("label", {class: "field"}, h("span", {class: "label"}, "Pasted planner reply"),
            h("textarea", {rows: 4, id: "plan-reply", placeholder: '{"art_style": "...", "shots": [...]}'})),
          h("button", {type: "button", class: "btn btn-small", onclick: applyPlannerReply}, "Apply pasted plan"))));
  }

  function renderIllustrate() {
    const screen = $("screen-illustrate");
    if (state.screen !== "illustrate" || !state.project.video) return;
    // Leave the step alone while one of its fields is being typed in.
    if (screen.contains(document.activeElement) && document.activeElement.matches("input[type=text], textarea")) return;
    const main = $("illustrate-main");
    if (!state.project.result) {
      main.replaceChildren(h("div", {class: "illustrate-empty"}, h("strong", {}, "Speak the project first"),
        h("p", {class: "muted"}, "The pictures are planned from the timed mix: its lines, sounds and Cast voices.")));
      $("shot-inspector").hidden = true;
      renderSettings();
      return;
    }
    const plan = state.project.video.plan;
    const rows = plan.rows;
    const warnings = plan.subjects.filter((item) => item.conflicts?.length).length;
    const looks = state.illustrateView === "looks";
    const views = h("div", {class: "seg", role: "group", "aria-label": "Illustrate view"},
      h("button", {type: "button", "aria-pressed": String(!looks), onclick: () => { state.illustrateView = "shots"; renderIllustrate(); }},
        rows.length ? `Shot plan · ${rows.length}` : "Shot plan"),
      h("button", {type: "button", "aria-pressed": String(looks), onclick: () => { state.illustrateView = "looks"; renderIllustrate(); }},
        "Characters & places", warnings ? h("span", {class: "count-badge", title: "Looks that disagree with a Cast voice"}, warnings) : null));
    const [filter, body] = looks ? [null, renderLooks()] : renderShotList(rows);
    main.setAttribute("aria-label", looks ? "Characters and places" : "Shot plan");
    main.replaceChildren(h("div", {class: "card-head"}, views, filter, h("span", {class: "spacer"}),
      looks ? h("span", {class: "muted small"}, "Drawn once as references; shots that mention one match it.") : null), body);
    renderInspector(rows);
    renderSettings();
  }

  /** The action bar on the Illustrate step: plan, draw what needs drawing, or interrupt. */
  function renderIllustrateActions(NO_GPU) {
    const video = state.project.video;
    const title = $("action-title"), detail = $("action-detail");
    const planButton = $("plan-shots"), drawButton = $("draw-shots"), stop = $("video-interrupt");
    detail.classList.remove("error");
    planButton.hidden = drawButton.hidden = stop.hidden = true;
    if (!video) { title.textContent = "Illustrated video isn't available here"; detail.textContent = ""; return; }
    if (!state.project.result) {
      title.textContent = "Speak the project first";
      detail.textContent = "Shots are planned from the timed mix.";
      return;
    }
    const rows = rowsOf();
    const needs = needsDrawing(rows);
    const run = state.videoRun;
    const busy = videoRunning();
    planButton.hidden = false;
    planButton.disabled = busy;
    planButton.textContent = videoRunning("plan") ? "Planning…" : rows.length ? "Plan again" : "Plan shots";
    planButton.className = rows.length ? "btn" : "btn btn-primary";
    drawButton.hidden = !rows.length || !needs;
    drawButton.disabled = busy || !state.boot.gpu;
    drawButton.title = state.boot.gpu ? "Draw the shots not drawn from their current prompt; the others keep their images" : NO_GPU;
    drawButton.textContent = `Draw ${plural(needs, "shot")}`;
    stop.hidden = !videoRunning("draw") && !videoRunning("preview");
    if (busy) {
      title.textContent = {plan: "Planning shots…", draw: "Drawing…", preview: "Drawing style previews…", save: "Saving to Media…"}[run.kind] || "Working…";
      detail.textContent = run.status || "";
    } else if (run?.error) {
      title.textContent = {plan: "Planning stopped", save: "Saving stopped", preview: "Style previews stopped"}[run.kind] || "Drawing stopped";
      detail.textContent = run.error;
      detail.classList.add("error");
    } else if (!rows.length) {
      title.textContent = video.plan.stale ? "The saved plan was made for other settings" : "No shot plan yet";
      detail.textContent = "Plan shots uses the writer set on Script.";
    } else if (needs) {
      const counts = video.plan.counts;
      title.textContent = `${plural(needs, "shot")} need${needs === 1 ? "s" : ""} drawing`;
      detail.textContent = [counts.changed ? `${counts.changed} prompt${counts.changed === 1 ? "" : "s"} changed` : null,
                            counts.todo ? `${counts.todo} not drawn` : null].filter(Boolean).join(", ")
        + (counts.drawn ? `. The other ${counts.drawn} keep their images.` : ".");
    } else {
      title.textContent = `Every shot is drawn (${rows.length})`;
      detail.textContent = "Watch them in the player's Video view, or Save to Media to render the video.";
    }
  }

  function illustrateSummary() {
    const video = state.project.video;
    if (!video || !state.project.result) return {text: state.project.result ? "" : "Speak first", state: null};
    const rows = rowsOf();
    if (!rows.length) return {text: "No shot plan yet", state: null};
    const needs = needsDrawing(rows);
    return needs ? {text: `${plural(rows.length, "shot")} · ${needs} to draw`, state: "needs"}
                 : {text: `${plural(rows.length, "shot")} drawn`, state: "done"};
  }

  function renderVideo() {
    if (!state.project) return;
    renderPointer();
    renderIllustrate();
  }

  return {videoRunning, visualSource, setMediaMode, videoFileButton, paintStills, videoField, renderVideo, renderIllustrate,
          renderIllustrateActions, illustrateSummary, shotLane, pickShot};
}
