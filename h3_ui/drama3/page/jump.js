// Bidirectional jump matching between Script Visual view and the Mix / Illustrate timeline.

export function normalizeSpeech(text) {
  return String(text || "").toLowerCase().replace(/[^\p{L}\p{N}]+/gu, "");
}

function collectScriptItems(root) {
  const lines = [], sounds = [];
  const walk = (node) => {
    if (node.type === "line") lines.push(node);
    else if (node.type === "sound") sounds.push(node);
    else if (node.type === "group") node.children.forEach(walk);
  };
  walk(root);
  return {lines, sounds};
}

function sameSpeaker(a, b) {
  return !a || !b || a === b;
}

function speechOverlap(a, b) {
  return Boolean(a && b && (a === b || a.includes(b) || b.includes(a)));
}

function matchSoundByOccurrence(sourceList, sourceIndex, targetList, name, loop) {
  const sameLoop = (item) => item.name === name && Boolean(item.loop) === Boolean(loop);
  const sameName = (item) => item.name === name;
  const loopOcc = sourceList.slice(0, sourceIndex).filter(sameLoop).length;
  const loopMatches = targetList.flatMap((item, i) => (sameLoop(item) ? [i] : []));
  if (loopMatches[loopOcc] !== undefined) return loopMatches[loopOcc];
  const nameOcc = sourceList.slice(0, sourceIndex).filter(sameName).length;
  const nameMatches = targetList.flatMap((item, i) => (sameName(item) ? [i] : []));
  return nameMatches[nameOcc] ?? nameMatches[0] ?? -1;
}

/** Find the 1-based script line for a timeline clip (speech or sound). */
export function matchScriptLine({kind, index, estimate, result, parse, text}) {
  if (estimate?.actual) {
    const items = kind === "sound" ? estimate.sounds : estimate.lines;
    if (items?.[index]?.line) return items[index].line;
  }
  if (!result) return null;
  const parsed = collectScriptItems(parse(text));
  if (kind === "sound") {
    const sound = result.sounds?.[index];
    if (!sound) return null;
    const match = matchSoundByOccurrence(result.sounds, index, parsed.sounds, sound.name, sound.loop);
    return match >= 0 ? parsed.sounds[match].n : null;
  }
  const line = result.lines?.[index];
  const targetWords = normalizeSpeech(line?.text);
  if (!line || !targetWords) return null;
  const matches = parsed.lines.filter((item) => sameSpeaker(line.speaker, item.speaker)
    && speechOverlap(normalizeSpeech(item.text), targetWords));
  if (!matches.length) return null;
  const occurrence = result.lines.slice(0, index).filter(
    (other) => other.speaker === line.speaker && normalizeSpeech(other.text) === targetWords,
  ).length;
  return (matches[occurrence] ?? matches[0]).n;
}

/** Find the 0-based index in ``result.lines`` or ``result.sounds`` for a script line / sound. */
export function matchResultIndex({kind, line, name = "", loop = false, estimate, result, parse, text}) {
  if (!result) return -1;
  if (estimate?.actual) {
    if (kind === "sound") {
      const sounds = estimate.sounds || [];
      const exact = sounds.findIndex((s) => s.line === line && (!name || s.name === name) && Boolean(s.loop) === Boolean(loop));
      if (exact >= 0) return exact;
      const byName = sounds.findIndex((s) => s.line === line && (!name || s.name === name));
      if (byName >= 0) return byName;
      const byLine = sounds.findIndex((s) => s.line === line);
      if (byLine >= 0) return byLine;
    } else {
      const idx = (estimate.lines || []).findIndex((item) => item.line === line);
      if (idx >= 0) return idx;
    }
  }
  const parsed = collectScriptItems(parse(text));
  if (kind === "sound") {
    const sounds = result.sounds || [];
    const scriptIdx = parsed.sounds.findIndex((s) => s.n === line && (!name || s.name === name));
    const target = scriptIdx >= 0 ? parsed.sounds[scriptIdx] : {name, loop};
    if (!target.name) return -1;
    if (scriptIdx >= 0) {
      return matchSoundByOccurrence(parsed.sounds, scriptIdx, sounds, target.name, target.loop);
    }
    const idx = sounds.findIndex((s) => s.name === target.name && Boolean(s.loop) === Boolean(target.loop));
    return idx >= 0 ? idx : sounds.findIndex((s) => s.name === target.name);
  }
  const lines = result.lines || [];
  const scriptIdx = parsed.lines.findIndex((item) => item.n === line);
  if (scriptIdx < 0) return -1;
  const node = parsed.lines[scriptIdx];
  const targetWords = normalizeSpeech(node.text);
  if (!targetWords) return -1;
  const occurrence = parsed.lines.slice(0, scriptIdx).filter(
    (other) => other.speaker === node.speaker && normalizeSpeech(other.text) === targetWords,
  ).length;
  const matches = lines.flatMap((item, i) => (
    sameSpeaker(node.speaker, item.speaker) && speechOverlap(normalizeSpeech(item.text), targetWords) ? [i] : []
  ));
  return matches[occurrence] ?? matches[0] ?? -1;
}

export function createJump({state, $, show, pick, seek, toast, showTimeline, visualSource, scriptView}) {
  const scriptText = () => $("script-text")?.value ?? state.project?.script?.text ?? "";

  function hasVisuals() {
    return Boolean(state.boot?.video && state.project?.video
      && (visualSource() || state.project.video.plan?.rows?.length));
  }

  function clearTimelinePick() {
    for (const clip of document.querySelectorAll("#lanes .clip.picked")) clip.classList.remove("picked");
  }

  function focusTimelineClip(selector) {
    showTimeline();
    clearTimelinePick();
    const clip = $("lanes")?.querySelector(selector);
    if (!clip) return;
    clip.classList.add("picked");
    clip.focus({preventScroll: true});
    clip.scrollIntoView?.({block: "nearest", inline: "center", behavior: "smooth"});
  }

  function jumpToScript(kind, index) {
    const line = matchScriptLine({
      kind, index, estimate: state.estimate, result: state.project?.result,
      parse: scriptView.parse, text: scriptText(),
    });
    if (!line) {
      toast("Could not match this clip to a line in the current script.");
      return;
    }
    const name = kind === "sound" ? (state.project?.result?.sounds?.[index]?.name || "") : "";
    show("script");
    scriptView.locateVisual(line, {kind, name});
  }

  function jumpToTimeline({line, kind, name = "", loop = false}) {
    const result = state.project?.result;
    if (!result) {
      toast("Speak the project first to view it on the timeline.");
      return;
    }
    const targetScreen = hasVisuals() ? "illustrate" : "mix";
    if (kind === "sound") {
      const soundIndex = matchResultIndex({
        kind: "sound", line, name, loop, estimate: state.estimate, result,
        parse: scriptView.parse, text: scriptText(),
      });
      if (soundIndex < 0) {
        toast("This sound isn't in the current mix yet.");
        return;
      }
      show(targetScreen);
      seek(result.sounds[soundIndex].start);
      focusTimelineClip(`.clip[data-sound="${soundIndex}"]`);
      return;
    }
    const lineIndex = matchResultIndex({
      kind: "line", line, estimate: state.estimate, result,
      parse: scriptView.parse, text: scriptText(),
    });
    if (lineIndex < 0) {
      toast("This line isn't in the current mix yet.");
      return;
    }
    show(targetScreen);
    pick(lineIndex, {seek: true});
    if (state.screen === "mix") {
      const list = $("lines");
      const row = list?.querySelector(`.line[data-line="${lineIndex}"]`);
      if (row) list.scrollTo({top: row.offsetTop - list.offsetTop - (list.clientHeight - row.offsetHeight) / 2, behavior: "smooth"});
    }
    focusTimelineClip(`.clip[data-line="${lineIndex}"]`);
  }

  return {jumpToScript, jumpToTimeline, clearTimelinePick};
}
