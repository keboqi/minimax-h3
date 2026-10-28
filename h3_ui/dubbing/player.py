"""Dub & Translate's result player: the dub with original or translated subtitles, either voice, music on or off.

The voice track is the clock; the muted picture and the background music follow it,
so switching the voice or the music never interrupts playback. Subtitles come from
the dub's own lines: each line's original timing, and the translated timing that
stays up while its dubbed take is still being spoken. Clicking a line seeks to it.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from urllib.parse import quote

from h3_app.dubbing import pipeline
from h3_app.dubbing.segments import format_time

ELEM_CLASS = "h3-dub-player-panel"

PLAYER_JS = """
// Gradio patches a new value into the existing DOM, so a regenerated result can reuse the old root:
// its listeners are dropped through the AbortController before it is bound again.
const players = new Map();
const fmt = (seconds) => {
  const whole = Math.max(0, Math.floor(seconds || 0));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
};
const bind = (root) => {
  players.get(root)?.controller.abort();
  const controller = new AbortController(), signal = controller.signal;
  root.dataset.bound = "true";
  const voice = root.querySelector("[data-voice]");
  const picture = root.querySelector("[data-picture]"), music = root.querySelector("[data-music]");
  const followers = [picture, music].filter(Boolean);
  const play = root.querySelector("[data-play]"), seek = root.querySelector("[data-seek]");
  const clock = root.querySelector("[data-time]"), state = root.querySelector("[data-state]");
  const subtitles = root.querySelector("[data-subtitles]"), musicToggle = root.querySelector("[data-music-toggle]");
  const captionTranslation = root.querySelector("[data-caption-translation]");
  const captionSource = root.querySelector("[data-caption-source]");
  const list = root.querySelector("[data-cues]"), cues = [...root.querySelectorAll("[data-cues] li")];
  const duration = Number(root.dataset.duration) || 0;
  let dragging = false, lit = null;
  players.set(root, { controller, media: [voice, ...followers] });
  const on = (target, name, handler, options) => target?.addEventListener(name, handler, { ...options, signal });

  const align = (force) => {
    for (const media of followers) {
      const end = Number.isFinite(media.duration) ? media.duration : Infinity;
      const target = Math.min(voice.currentTime, end);
      if (force || Math.abs(media.currentTime - target) > 0.25) {
        try { media.currentTime = target; } catch (_) { /* metadata not loaded yet */ }
      }
      if (!voice.paused && media.paused && target < end - 0.05) media.play().catch(() => {});
    }
  };
  const shown = (cue, mode) => {
    const time = voice.currentTime, start = Number(cue.dataset.start);
    const end = Number(cue.dataset.end), translatedEnd = Number(cue.dataset.shownEnd);
    if (time < start) return false;
    if (mode === "translation") return time < translatedEnd;
    if (mode === "source") return time < end;
    return time < Math.max(end, translatedEnd);
  };
  const render = () => {
    const time = voice.currentTime, mode = subtitles.value;
    if (!dragging) seek.value = String(time);
    clock.textContent = `${fmt(time)} / ${fmt(duration || voice.duration)}`;
    const active = cues.filter((cue) => shown(cue, mode === "off" ? "both" : mode));
    const text = (field, cueMode) => cues.filter((cue) => shown(cue, cueMode))
      .map((cue) => cue.querySelector(field)?.textContent || "").filter(Boolean).join("\\n");
    captionTranslation.textContent = mode === "translation" || mode === "both" ? text(".h3-dub-cue-translation", "translation") : "";
    captionSource.textContent = mode === "source" || mode === "both" ? text(".h3-dub-cue-source", "source") : "";
    for (const cue of cues) {
      cue.classList.toggle("active", active.includes(cue));
      cue.classList.toggle("past", !active.includes(cue)
        && time >= Math.max(Number(cue.dataset.end), Number(cue.dataset.shownEnd)));
    }
    const current = active.at(-1) || null;
    if (current && current !== lit && !voice.paused) {
      // Measured against the list itself: offsetTop is relative to whichever ancestor is positioned.
      const offset = current.getBoundingClientRect().top - list.getBoundingClientRect().top + list.scrollTop;
      list.scrollTo({ top: offset - (list.clientHeight - current.offsetHeight) / 2, behavior: "smooth" });
    }
    lit = current;
  };
  const jump = (time) => {
    voice.currentTime = Math.max(0, time);
    align(true);
    render();
  };
  const label = (playing) => {
    play.textContent = playing ? "Pause" : "Play";
    play.setAttribute("aria-pressed", playing ? "true" : "false");
  };
  // Starting playback brings the current line into view even if it was already lit while paused.
  on(voice, "play", () => { align(true); label(true); lit = null; render(); });
  for (const name of ["pause", "ended"]) {
    on(voice, name, () => { followers.forEach((media) => media.pause()); label(false); render(); });
  }
  on(voice, "seeked", () => { align(true); render(); });
  on(voice, "timeupdate", () => { if (!voice.paused) align(false); render(); });
  on(voice, "loadedmetadata", render);
  on(play, "click", () => {
    if (!voice.paused) { voice.pause(); return; }
    voice.play().then(() => { state.textContent = ""; })
      .catch(() => { state.textContent = "Playback unavailable. Check browser codec support."; });
  });
  on(picture, "click", () => play.click());
  on(seek, "pointerdown", () => { dragging = true; });
  on(seek, "change", () => { dragging = false; });
  on(seek, "input", () => jump(Number(seek.value)));
  on(musicToggle, "change", () => { if (music) music.muted = !musicToggle.checked; });
  on(subtitles, "change", () => {
    root.dataset.subtitles = subtitles.value;
    render();
  });
  for (const choice of root.querySelectorAll("[data-voice-choice]")) {
    on(choice, "change", () => {
      if (!choice.checked) return;
      const time = voice.currentTime, playing = !voice.paused;
      on(voice, "loadedmetadata", () => {
        voice.currentTime = time;
        if (playing) voice.play().catch(() => {});
        align(true);
        render();
      }, { once: true });
      voice.src = choice.value;
    });
  }
  on(list, "click", (event) => {
    const cue = event.target.closest("li");
    if (cue) jump(Number(cue.dataset.start));
  });
  render();
};
const setup = () => {
  // A new result replaces the old player; a detached element would keep playing.
  for (const [root, player] of [...players]) {
    if (!root.isConnected) {
      player.controller.abort();
      player.media.forEach((media) => media.pause());
      players.delete(root);
    }
  }
  for (const root of element.querySelectorAll(".h3-dub-player:not([data-bound])")) bind(root);
};
new MutationObserver(setup).observe(element, { childList: true, subtree: true });
setup();
"""


def file_url(path: Path) -> str:
    """The served URL, versioned by mtime so a regenerated file at the same path is not served from cache."""
    path = Path(path).resolve()
    try:
        version = f"?v={path.stat().st_mtime_ns}"
    except OSError:
        version = ""
    return "/gradio_api/file=" + quote(str(path).replace("\\", "/"), safe="/:=") + version


def _language(name: str, fallback: str) -> str:
    return escape(name or fallback)


def player_html(player: pipeline.Player, *, key: str) -> str:
    """The player for one dub; ``key`` keeps its radio group apart from earlier players on the page."""
    dub, original = file_url(player.dub), file_url(player.original)
    target = _language(player.target_language, "translated")
    source = _language(player.source_language, "original")
    if player.video is not None:
        screen = (f'<video data-picture muted playsinline preload="metadata" '
                  f'src="{escape(file_url(player.video), quote=True)}"></video>')
    else:
        screen = '<div class="h3-dub-player-poster" aria-hidden="true">Audio</div>'
    music = (f'<audio data-music preload="auto" src="{escape(file_url(player.music), quote=True)}"></audio>'
             if player.music is not None else "")
    toggle = ('<label class="h3-dub-player-check"><input type="checkbox" data-music-toggle checked> '
              'Background music</label>' if player.music is not None else
              '<label class="h3-dub-player-check"><input type="checkbox" checked disabled> '
              'Background music <span>(needs vocal separation)</span></label>')
    speakers = len({cue["speaker"] for cue in player.lines}) > 1
    cues = "".join(
        f'<li data-start="{cue["start"]:.3f}" data-end="{cue["end"]:.3f}" data-shown-end="{cue["shown_end"]:.3f}">'
        f'<span class="h3-dub-cue-time">{format_time(cue["start"])[:-4]}</span>'
        + (f'<span class="h3-dub-cue-speaker">{escape(cue["speaker"])}</span>' if speakers else "")
        + f'<span class="h3-dub-cue-text"><span class="h3-dub-cue-translation">{escape(cue["translation"])}</span>'
        f'<span class="h3-dub-cue-source">{escape(cue["source"])}</span></span></li>'
        for cue in player.lines)
    group = escape(f"h3-dub-voice-{key}", quote=True)
    note = ("" if player.music is not None else
            '<p class="h3-dub-player-note">This source was analyzed without vocal separation, so the dub plays '
            "as one mix with its background.</p>")
    return (
        f'<div class="h3-dub-player" data-duration="{player.duration:.3f}" data-subtitles="translation">'
        f'<div class="h3-dub-player-screen">{screen}<div class="h3-dub-player-caption" aria-hidden="true">'
        '<span data-caption-translation></span><span data-caption-source></span></div></div>'
        f'<audio data-voice preload="auto" src="{escape(dub, quote=True)}"></audio>{music}'
        '<div class="h3-dub-player-transport">'
        '<button type="button" data-play aria-pressed="false">Play</button>'
        f'<input type="range" data-seek min="0" max="{player.duration:.3f}" step="0.01" value="0" '
        'aria-label="Position">'
        '<span data-time class="h3-dub-player-time">0:00</span></div>'
        '<div class="h3-dub-player-options">'
        f'<fieldset><legend>Voice</legend>'
        f'<label><input type="radio" name="{group}" data-voice-choice value="{escape(dub, quote=True)}" checked> '
        f'Dub ({target})</label>'
        f'<label><input type="radio" name="{group}" data-voice-choice value="{escape(original, quote=True)}"> '
        f'Original ({source})</label></fieldset>'
        f'{toggle}'
        '<label class="h3-dub-player-select">Subtitles <select data-subtitles>'
        f'<option value="translation" selected>{target}</option><option value="source">{source}</option>'
        '<option value="both">Both</option><option value="off">Off</option></select></label></div>'
        f'{note}<p data-state class="h3-dub-player-note" role="status" aria-live="polite"></p>'
        f'<ol data-cues class="h3-dub-player-cues" aria-label="Lines">{cues}</ol></div>'
    )
