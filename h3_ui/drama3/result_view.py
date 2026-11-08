"""A Soundstage result's rich view: its lines and sounds on a timeline, subtitles and the stage.

Drawn from the result's settings snapshot alone, so Speak and Media show the same view,
and the GPU-free Library Console can draw it without the generation stack.
"""

from __future__ import annotations

import html
from pathlib import Path
import re

from h3_app.drama3.results import CHARACTER_POSITIONS, NARRATOR_NAMES
from h3_ui.subtitle_words import word_spans_html

_CUE = re.compile(r"<\|[^|]*\|>|<[^>]*>|\[[^\]]*\]|\{[^{}\n]*\}")
# The script box's name colors: the narrator has its own, characters take eight in turn.
_NAME_TONES = 8


# Gradio's waveform player keeps its <audio> element out of the DOM, so media elements are
# registered as they get a source; each visible subtitle panel follows the player whose file
# (or, for a blob source, whose length) matches its result, so playing and seeking both sync.
# Clicking a line, or a voice or sound on the timeline, seeks the player to it.
DRAMA3_SUBTITLES_JS = """
() => {
  if (window.h3Drama3Subtitles) return;
  window.h3Drama3Subtitles = true;
  const media = new Set();
  const proto = HTMLMediaElement.prototype;
  const source = Object.getOwnPropertyDescriptor(proto, "src");
  Object.defineProperty(proto, "src", {
    ...source,
    set(value) { media.add(this); source.set.call(this, value); },
  });
  // While anything plays, follow it every frame; the interval below covers seeks while paused.
  let following = false;
  const follow = () => {
    if (following || typeof requestAnimationFrame !== "function") return;
    following = true;
    const frame = () => {
      sync();
      following = [...media].some((element) => !element.paused);
      if (following) requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  };
  const play = proto.play;
  proto.play = function (...args) {
    media.add(this);
    const result = play.apply(this, args);
    follow();
    return result;
  };
  const playerFor = (box) => {
    if (box.dataset.preview !== undefined) return null;
    const id = box.dataset.media, duration = Number(box.dataset.duration);
    let found = null;
    for (const element of media) {
      const url = element.currentSrc || element.src || "";
      const matches = (id && url.includes(id))
        || (url.startsWith("blob:") && duration > 0 && Math.abs(element.duration - duration) < 0.6);
      if (matches && (!found || (found.paused && !element.paused))) found = element;
    }
    return found;
  };
  const sync = () => {
    for (const box of document.querySelectorAll(".h3-drama3-subtitles")) {
      if (!box.offsetParent || box.dataset.preview !== undefined) continue;
      const player = playerFor(box);
      const time = player ? player.currentTime : -1;
      const lines = [...box.querySelectorAll("li")];
      // Parallel branches arrive in script order, not playback order.
      const started = lines.filter((line) => Number(line.dataset.start) <= time + 0.05)
        .sort((a, b) => Number(a.dataset.start) - Number(b.dataset.start));
      const playing = started.filter((line) => Number(line.dataset.end) > time);
      // Highlight overlapping speech together; keep the latest start lit during gaps.
      const highlighted = playing.length ? playing : started.slice(-1);
      const active = highlighted.at(-1) ?? null;
      for (const line of lines) {
        const lit = highlighted.includes(line);
        line.classList.toggle("active", lit);
        line.classList.toggle("past", time >= 0 && Number(line.dataset.end) <= time && !lit);
        // Aligned lines light word by word: what has been said, and the word being said.
        for (const word of line.querySelectorAll(".h3-word")) {
          const start = Number(word.dataset.start), end = Number(word.dataset.end);
          word.classList.toggle("spoken", lit && start <= time);
          word.classList.toggle("current", lit && start <= time && time < end + 0.08);
        }
      }
      // Voice segments follow their subtitle lines, sounds light while they play;
      // the playhead follows the player.
      for (const segment of box.querySelectorAll(".h3-drama3-segment")) {
        const { line, start, end } = segment.dataset;
        segment.classList.toggle("active", line === undefined
          ? Number(start) <= time && time < Number(end)
          : highlighted.includes(lines[Number(line)]));
      }
      const head = box.querySelector(".h3-drama3-playhead");
      const duration = Number(box.dataset.duration) || player?.duration || 0;
      if (head?.style) {
        head.hidden = !(time >= 0 && duration > 0);
        head.style.left = `${Math.min(100, Math.max(0, time / duration * 100))}%`;
      }
      if (active && box.h3Active !== active) {
        const list = box.querySelector("ol");
        list.scrollTo({ top: active.offsetTop - (list.clientHeight - active.offsetHeight) / 2, behavior: "smooth" });
      }
      box.h3Active = active;
    }
  };
  setInterval(sync, 200);
  document.addEventListener("click", (event) => {
    const line = event.target.closest?.(".h3-drama3-subtitles :is(li, .h3-drama3-segment)");
    const player = line && playerFor(line.closest(".h3-drama3-subtitles"));
    if (!player) return;
    player.currentTime = Number(line.dataset.start);
    sync();
  });
}
"""


def _stage_tones(speakers, pans=None):
    """Each speaker's color and place in the stereo image, by name, in order of first line.

    Places follow the mix (``speaker_positions``): the narrator is centred, and characters
    spread to both sides only when there are two or more of them. ``pans`` holds where the
    mix placed each speaker's first line, which a scene's ``stage`` can move; results from
    before it was recorded have none.
    """
    ordered = list(dict.fromkeys(speakers))
    characters = [name for name in ordered if " ".join(name.split()).casefold() not in NARRATOR_NAMES]
    spread = len(characters) >= 2
    tones = {name: ("narrator", 0.0) for name in ordered}
    tones.update((name, (str(index % _NAME_TONES),
                         CHARACTER_POSITIONS[index % len(CHARACTER_POSITIONS)] if spread else 0.0))
                 for index, name in enumerate(characters))
    for name, pan in (pans or {}).items():
        if name in tones:
            tones[name] = (tones[name][0], float(pan))
    return tones


def _rows(spans):
    """A row for each (start, end) so clips that overlap in time sit on different rows, and the count."""
    ends, rows = [], [0] * len(spans)
    for index in sorted(range(len(spans)), key=lambda index: spans[index][0]):
        start, end = spans[index]
        # Back-to-back clips share a row; a small overlap from rounding does not open a new one.
        row = next((row for row, last in enumerate(ends) if last <= start + 0.01), len(ends))
        if row == len(ends):
            ends.append(end)
        else:
            ends[row] = end
        rows[index] = row
    return rows, max(len(ends), 1)


def _place(start, end, duration, row):
    return (f'data-start="{start:.3f}" data-end="{end:.3f}" style="left: {min(start / duration, 1) * 100:.2f}%; '
            f'width: {max(end - start, 0) / duration * 100:.2f}%; --row: {row}"')


def _timeline_html(entries, tones, duration, sounds=()):
    """The result's lines and sounds on lanes, placed by time; clicking one seeks.

    Voices are colored by speaker and sounds by kind (looped beds, one-shot cues). Clips that
    play at the same time, like overlapping speech or music under an effect, stack in rows.
    """
    duration = duration or max(end for *_, end in [*entries, *sounds])
    if duration <= 0:
        return ""
    rows, count = _rows([(start, end) for *_, start, end in entries])
    lanes = [("Voices", count, "".join(
        f'<button type="button" class="h3-drama3-segment h3-drama3-tone-{tones[speaker][0]}" '
        f'data-line="{index}" {_place(start, end, duration, row)} '
        f'title="{html.escape(f"{speaker}: {text}")}" aria-label="{html.escape(f"Play from {speaker}: {text}")}">'
        f'</button>'
        for index, ((speaker, text, start, end), row) in enumerate(zip(entries, rows))))]
    if sounds:
        rows, count = _rows([(start, end) for *_, start, end in sounds])
        lanes.append(("Sounds", count, "".join(
            f'<button type="button" class="h3-drama3-segment h3-drama3-sound h3-drama3-tone-{"bed" if loop else "cue"}" '
            f'{_place(start, end, duration, row)} '
            f'title="{html.escape(name + (" (looped)" if loop else " (one-shot)"))}" '
            f'aria-label="{html.escape(f"Play from sound: {name}")}"><span>{html.escape(name)}</span></button>'
            for (name, loop, start, end), row in zip(sounds, rows))))
    names = "".join(f'<span style="--rows: {count}">{label}</span>' for label, count, _ in lanes)
    tracks = "".join(f'<div class="h3-drama3-lane" style="--rows: {count}">{segments}</div>'
                     for _, count, segments in lanes)
    return (f'<div class="h3-drama3-timeline" role="group" aria-label="Voices and sounds on the timeline">'
            f'<div class="h3-drama3-lane-names" aria-hidden="true">{names}</div><div class="h3-drama3-lanes">'
            f'{tracks}<span class="h3-drama3-playhead" aria-hidden="true" hidden></span></div></div>')


def _stage_html(tones):
    """Where each voice stands, left to right, as the mix places it."""
    voices = "".join(
        f'<span role="listitem" class="h3-drama3-stage-voice h3-drama3-tone-{tone}" '
        f'data-row="{row % 2}" style="left: {(position + 1) * 50:.1f}%">{html.escape(name)}</span>'
        for row, (name, (tone, position)) in enumerate(tones.items()))
    return ('<div class="h3-drama3-stage-map"><span class="h3-drama3-stage-label">Stage</span>'
            f'<div class="h3-drama3-stage-field" role="list" aria-label="Where each voice stands, left to right">'
            f'{voices}<span class="h3-drama3-stage-side" aria-hidden="true">L</span>'
            '<span class="h3-drama3-stage-side" aria-hidden="true">C</span>'
            '<span class="h3-drama3-stage-side" aria-hidden="true">R</span></div></div>')


def script_preview_html(script: str, script_mode: str) -> str:
    """Estimate a silent timeline from the same script tree used for speech generation."""
    from h3_app.drama3.serving.planner import expected_seconds
    from h3_app.drama3.serving.tree import (
        ParallelNode, PauseNode, SoundNode, SpeechNode, parse_drama3_tree, strip_drama3_sound_tags,
    )
    from h3_app.prompt_writers.drama3 import DRAMA3_SCRIPT_BASIC

    chunks, sounds = [], []

    def place(node, start, stage):
        if isinstance(node, PauseNode):
            return max(0, start + node.seconds)
        if isinstance(node, SoundNode):
            end = start + node.duration
            sounds.append(dict(name=node.name or node.description, loop=node.loop, start=start, end=end))
            return end
        if isinstance(node, SpeechNode):
            for turn in node.turns:
                end = start + max(0.1, expected_seconds(turn.text))
                chunks.append(dict(name=turn.speaker, text=turn.text, start=start, end=end, pan=stage.get(turn.speaker)))
                start = end
            return start
        if isinstance(node, ParallelNode):
            staging = {**stage, **dict(node.stage)}
            return max((place(child, start, staging) for child in node.children), default=start)
        for child in node.children:
            start = place(child, start, stage)
        return start

    if script_mode == DRAMA3_SCRIPT_BASIC:
        script = strip_drama3_sound_tags(script)
    duration = place(parse_drama3_tree(script), 0, {})
    # A trailing overlap can move the cursor behind the end of an earlier clip.
    duration = max(duration, max((item["end"] for item in chunks + sounds), default=0))
    panel = subtitles_html(dict(chunks=chunks, sounds=sounds, duration=duration), None)
    return ("<p>Script preview · Estimated timing</p>" + panel if panel
            else "<p>Add a spoken line to preview the script.</p>")


def timeline_data(settings: dict):
    """A result's timed lines, their aligned words, each speaker's (tone, pan) and its timed sounds.

    Lines are (speaker, text, start, end) with cues left out; sounds are (name, loop, start, end).
    Results from before line timings were recorded have no lines.
    """
    entries, pans, words = [], {}, []
    for chunk in settings.get("chunks") or []:
        if chunk.get("start") is None:
            continue
        speaker = str(chunk.get("name") or f"Speaker {chunk.get('speaker', 0)}")
        if chunk.get("pan") is not None:
            pans.setdefault(speaker, chunk["pan"])
        # A take can hold several script lines; results from before line timings show it whole.
        for line in chunk.get("lines") or [chunk]:
            text = clean_line(line.get("text", ""))
            if not text or line.get("start") is None:
                continue
            start = float(line["start"])
            entries.append((speaker, text, start, float(line.get("end", start))))
            words.append(line.get("words"))
    tones = _stage_tones((speaker for speaker, *_ in entries), pans)
    sounds = [(str(sound.get("name") or "Sound"), bool(sound.get("loop")), float(sound["start"]),
               float(sound.get("end", sound["start"])))
              for sound in settings.get("sounds") or [] if sound.get("start") is not None]
    return entries, words, tones, sounds


def clean_line(text) -> str:
    """A script line as it is heard: cues and tags left out, spaces collapsed."""
    return " ".join(_CUE.sub(" ", str(text or "")).split())


def subtitles_html(settings: dict, output) -> str:
    """The rich view for a result with these snapshot ``settings``, or "" when it has no line timings.

    The browser highlights the line playing and moves the timeline's playhead, following the
    player whose file carries ``output``'s 32-hex id. Results from before line timings were
    recorded have none.
    """
    entries, words, tones, sounds = timeline_data(settings)
    if not entries:
        return ""
    lines = "".join(
        f'<li data-start="{start:.3f}" data-end="{end:.3f}">'
        f'<span class="h3-subtitle-speaker h3-drama3-tone-{tones[speaker][0]}">{html.escape(speaker)}</span>'
        f'<span class="h3-subtitle-text">{word_spans_html(text, line_words)}'
        + ('<small> · Word timing unavailable; use Generate / repair word alignment.</small>'
           if settings.get("word_timing") and not line_words else '') + '</span></li>'
        for (speaker, text, start, end), line_words in zip(entries, words))
    duration = float(settings.get("duration") or 0)
    # The result's 32-hex id is in every copy's file name, so the browser can find its player.
    media = re.search(r"[0-9a-f]{32}", Path(str(output)).name)
    preview = ' data-preview' if output is None else ''
    timeline = _timeline_html(entries, tones, duration, sounds)
    if preview:
        timeline = f'<div inert>{timeline}</div>'
    return (f'<div class="h3-drama3-subtitles"{preview} data-media="{media.group(0) if media else ""}" '
            f'data-duration="{duration:.3f}">'
            f'{timeline}<ol>{lines}</ol>{_stage_html(tones)}</div>')
