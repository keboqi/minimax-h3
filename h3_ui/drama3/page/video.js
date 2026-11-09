// Video view and local controls for Soundstage.
export function createVideo({state, $, h, mixAudio, mixVideo, clock, saveVideo, seek, renderPlayer, sync, saveVideoToMedia, planShots, drawImages, interruptVideo, copyPlannerPrompt, applyPlannerReply, getPlayer, setPlayer}) {
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

  /** The header's video file: download the saved one, or render it from the stills for Media. */
  function videoFileButton() {
    const video = state.project.video;
    const saving = videoRunning("save");
    if (video.latest && (video.latest.current || !video.stills)) {
      return h("a", {class: "btn btn-small", href: video.latest.url, download: video.latest.file,
                     title: "This video is in Media too."}, "Download video");
    }
    return h("button", {type: "button", class: "btn btn-small", disabled: saving || videoRunning("draw"),
                        title: "Render a 720p video with the mix from these images and add it to Media",
                        onclick: saveVideoToMedia}, saving ? "Saving to Media…" : "Save to Media");
  }

  // The stills on the mix's clock: the shot under the playhead, dissolving in over the one before it.
  const preloaded = new Set();
  function paintStills(time) {
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

  function videoField(label, input, hint) {
    return h("label", {class: "field"}, h("span", {class: "label"}, label), input,
             hint ? h("span", {class: "muted small"}, hint) : null);
  }

  function renderVideo() {
    const box = $("video-panel");
    const boot = state.boot.video;
    const video = state.project.video;
    box.hidden = !boot || !video;
    if (box.hidden) return;
    // Leave the panel alone while one of its fields is being typed in.
    if (box.contains(document.activeElement) && document.activeElement.matches("input, textarea")) return;
    const result = state.project.result;
    const planning = videoRunning("plan");
    const drawing = videoRunning("draw");
    const shown = visualSource();
    const save = (body) => saveVideo(body);
    const presets = [...Object.keys(boot.styles), boot.custom_style];
    const rows = video.plan.rows;
    const run = state.videoRun;
    const note = run && (run.active || run.error || run.status)
      ? h("p", {class: `slot-note${run.error ? " error" : ""}`}, run.error || run.status) : null;
    const limitCount = h("span", {class: "num small"}, video.limit);
    box.replaceChildren(
      h("div", {class: "card-head"},
        h("h2", {}, "Illustrated video"),
        h("span", {class: "muted small"}, "Images drawn shot by shot with an Image model slide along the mix here. "
          + "Save to Media renders them as a 720p video."),
        h("span", {class: "spacer"}),
        shown ? h("button", {type: "button", class: "btn btn-small", onclick: () => { setMediaMode("video"); window.scrollTo({top: 0, behavior: "smooth"}); }},
          shown === "stills" ? "Watch with the mix" : "Watch the latest video") : null),
      !result ? h("p", {class: "video-empty muted"}, "Speak the project first: the video is drawn from its getPlayer().") : h("div", {class: "video-body"},
        h("div", {class: "video-settings"},
          videoField("Image model", h("select", {onchange: (event) => save({video_engine: event.target.value})},
            boot.engines.map((engine) => h("option", {value: engine.id, selected: engine.id === video.engine}, engine.label))),
            "Draws with the Image tab's settings for that model."),
          videoField("Style preset", h("select", {onchange: (event) => {
            const style = boot.styles[event.target.value];
            if (style) save({video_style: style});
          }}, presets.map((name) => h("option", {selected: name === video.preset, disabled: name === boot.custom_style && video.preset !== name}, name)))),
          videoField("Art direction", h("textarea", {rows: 3, value: video.style, oninput: (event) => saveVideo({video_style: event.target.value}, {later: true})}),
            "Add the cast, setting or era. Editing it makes the preset Custom."),
          videoField("Minimum shots", h("span", {class: "range-row"},
            h("input", {type: "range", min: boot.limits[0], max: boot.limits[1], step: 1, value: video.limit,
                        oninput: (event) => { limitCount.textContent = event.target.value; },
                        onchange: (event) => save({video_limit: Number(event.target.value)})}), limitCount),
            "A floor: the planner adds images wherever the story needs them, up to one per 2 seconds."),
          h("label", {class: "check"}, h("input", {type: "checkbox", checked: video.continuity,
            onchange: (event) => save({video_continuity: event.target.checked})}), "Keep characters consistent"),
          h("div", {class: "row-buttons"},
            h("button", {type: "button", class: "btn", disabled: planning || drawing, onclick: planShots},
              planning ? "Planning…" : rows.length ? "Plan again" : "Plan shots"),
            drawing
              ? h("button", {type: "button", class: "btn", onclick: interruptVideo}, "Interrupt")
              : h("button", {type: "button", class: "btn btn-primary", "data-gpu": true, disabled: !rows.length || planning || !state.boot.gpu,
                             title: rows.length ? "" : "Plan shots first", onclick: drawImages},
                  video.stills ? "Draw the images again" : "Draw the images")),
          note,
          h("details", {class: "more"},
            h("summary", {}, "Use another chat model"),
            h("div", {class: "more-body"},
              h("p", {class: "muted small"}, "Copy the full planner prompt into any chat model, then paste its JSON reply. It is checked like the built-in planner's."),
              h("button", {type: "button", class: "btn btn-small", onclick: copyPlannerPrompt}, "Copy full planner prompt"),
              h("label", {class: "field"}, h("span", {class: "label"}, "Pasted planner reply"),
                h("textarea", {rows: 4, id: "plan-reply", placeholder: '{"art_style": "...", "shots": [...]}'})),
              h("button", {type: "button", class: "btn btn-small", onclick: applyPlannerReply}, "Apply pasted plan")))),
        h("div", {class: "video-plan"},
          h("span", {class: "label"}, rows.length ? `Shot plan · ${rows.length} shots` : "Shot plan"),
          rows.length ? h("div", {class: "table-wrap plan-wrap"}, h("table", {class: "plan-table"},
            h("thead", {}, h("tr", {}, ["Time", "", "Story beat", "Image prompt"].map((label) => h("th", {scope: "col"}, label)))),
            h("tbody", {}, rows.map((row) => h("tr", {},
              h("td", {class: "num"}, h("button", {type: "button", class: "link", onclick: () => seek(row.start, false)},
                `${clock(row.start)}–${clock(row.end)}`)),
              h("td", {}, h("span", {class: `kind ${row.transition === "Cut" ? "kind-shot" : "kind-loop"}`}, row.transition)),
              h("td", {}, row.beat),
              h("td", {class: "muted"}, row.prompt))))))
            : h("p", {class: "muted small"}, video.plan.stale
              ? "The saved plan was made for a different Art direction or Minimum shots. Plan again, or set them back."
              : "Plan shots to see each image's time, story beat and prompt here before drawing."))));
  }

  return {videoRunning, visualSource, setMediaMode, videoFileButton, paintStills, videoField, renderVideo};
}
