"""The illustrated video beside a Sound Stage result, with the result's subtitles laid over it.

The captions come from the result's subtitle panel on the same page: its lines already carry
their timings and word spans, and the page's subtitle script lights them as this video plays,
because the video's file name holds the mix's 32-hex id. The overlay copies the active lines'
text, so captions light word by word along with the list.
"""

from __future__ import annotations

import glob
from html import escape
import re
from pathlib import Path

from h3_ui.dubbing.player import file_url

ELEM_CLASS = "h3-stage-video-panel"
# The settings-snapshot family of an illustrated video; its settings name the source mix.
ILLUSTRATED_FAMILY = "Sound Stage illustrated video"
# Pauses every result player in a scope before a switch hides one of them.
PAUSE_JS = ("(mode) => {{ document.querySelectorAll('{scope} :is(audio, video)')"
            ".forEach((media) => media.pause()); return mode; }}")


def latest_export(audio):
    """The newest illustrated video made from this mix, if any."""
    audio = Path(audio)
    videos = [path for path in audio.parent.glob(f"{glob.escape(audio.stem)}_illustrated_*.mp4") if path.is_file()]
    return max(videos, key=lambda path: path.stat().st_mtime, default=None)

STAGE_VIDEO_JS = """
// Gradio patches a new value into the existing DOM, so each new video rebinds through its revision.
let bound = null;
const bind = () => {
  const root = element.querySelector(".h3-stage-video");
  if (!root || bound?.root === root && bound.revision === root.dataset.revision) return;
  bound?.controller.abort();
  const controller = new AbortController(), signal = controller.signal;
  bound = { root, revision: root.dataset.revision, controller };
  const video = root.querySelector("video"), target = root.querySelector("[data-caption]");
  // The page's subtitle script tracks media whose source is set from script; a source parsed from
  // HTML is not, so the list would follow this video only once it plays, not on a seek while paused.
  if (video.getAttribute("src")) video.src = video.getAttribute("src");
  const toggle = root.querySelector("[data-captions]");
  const panel = () => {
    const id = root.dataset.media;
    return [...document.querySelectorAll(".h3-drama3-subtitles")]
      .find((box) => box.dataset.preview === undefined && (!id || box.dataset.media === id)) || null;
  };
  let parts = [];
  const render = () => {
    const box = panel();
    const time = video.currentTime;
    const wanted = !toggle || toggle.checked;
    const lines = box && wanted ? [...box.querySelectorAll("li")].filter((line) =>
      Number(line.dataset.start) <= time + 0.05 && time < Number(line.dataset.end)) : [];
    const texts = lines.map((line) => line.querySelector(".h3-subtitle-text")).filter(Boolean);
    if (texts.length === parts.length && texts.every((part, index) => part === parts[index])) {
      // Same lines: copy the word classes the page's subtitle script keeps up to date.
      const copies = target.querySelectorAll(".h3-word");
      texts.flatMap((part) => [...part.querySelectorAll(".h3-word")]).forEach((word, index) => {
        if (copies[index] && copies[index].className !== word.className) copies[index].className = word.className;
      });
      return;
    }
    parts = texts;
    // Notes such as "Word timing unavailable" stay in the list, never on the picture.
    target.replaceChildren(...texts.flatMap((part, index) => [
      ...(index ? [document.createTextNode("\\n")] : []),
      ...[...part.childNodes].filter((node) => node.nodeName !== "SMALL").map((node) => node.cloneNode(true))]));
  };
  let frame = 0;
  const follow = () => {
    cancelAnimationFrame(frame);
    if (video.paused || signal.aborted) return;
    render();
    frame = requestAnimationFrame(follow);
  };
  const on = (node, name, handler) => node?.addEventListener(name, handler, { signal });
  on(video, "play", follow);
  for (const name of ["pause", "seeked", "timeupdate", "loadedmetadata", "ended"]) on(video, name, render);
  on(toggle, "change", render);
  render();
};
new MutationObserver(bind).observe(element, { childList: true, subtree: true,
  attributes: true, attributeFilter: ["data-revision"] });
bind();
"""

_MEDIA_ID = re.compile(r"[0-9a-f]{32}")


def stage_video_html(path, *, url=file_url) -> str:
    """The illustrated video with a caption overlay, or nothing when there is no video."""
    if not path:
        return ""
    path = Path(str(path))
    source = url(path)
    media = _MEDIA_ID.search(path.name)
    revision = escape(f"{path.name}:{source}", quote=True)
    return (
        f'<div class="h3-stage-video" data-revision="{revision}" data-media="{media.group(0) if media else ""}">'
        '<div class="h3-dub-player-screen">'
        f'<video controls playsinline preload="metadata" src="{escape(source, quote=True)}"></video>'
        '<div class="h3-dub-player-caption" aria-hidden="true"><span data-caption></span></div></div>'
        '<div class="h3-stage-video-options">'
        '<label><input type="checkbox" data-captions checked> Subtitles on the video</label>'
        f'<a href="{escape(source, quote=True)}" download="{escape(path.name, quote=True)}">Download video</a>'
        '</div></div>'
    )
