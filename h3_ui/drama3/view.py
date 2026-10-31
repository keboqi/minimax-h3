"""Soundstage as four steps: Script, Cast & sounds, Speak and Review lines.

The steps share Song's stepper: each pairs a working card with an action card that
holds the step's next move. Every step stays mounted, so Auto on Script can press the
Auto button on Speak. Event wiring lives in bindings.py.
"""

from dataclasses import dataclass
import json
import gradio as gr
from h3_app.catalog import DEFAULT_GEMINI_PROMPT_MODEL, GEMINI_PROMPT_MODELS
from h3_ui.drama3.speaker_library import SpeakerLibraryView, build_speaker_library
from h3_app.drama3.voice_design import ENGINE_QWEN, VOICE_DESIGN_ENGINES
from h3_app.prompt_writers.drama3 import (
    DRAMA3_SCRIPT_ADVANCED, DRAMA3_SCRIPT_MODES, DRAMA3_SPEAKER_AUTO, DRAMA3_SPEAKER_MODES, DRAMA3_WORKFLOW_SINGLE,
    DRAMA3_WORKFLOWS,
)
from h3_app.voice_design import LANGUAGES as VOICE_DESIGN_LANGUAGES
from h3_models import DEFAULT_DRAMA3_ASR, DRAMA3_ASR_CHOICES
from ..prompt_writer_controls import RemotePromptWriterControls, build_remote_prompt_writer_controls
from ..step_cards import StepStrip, card_heading, intro, step_strip
from h3_ui.engines.controls import (
    QUIET,
    generation_actions,
)
from h3_ui.drama3.controls import (
    DRAMA3_SAMPLING_PRESETS,
    DRAMA3_SOUND_SLOTS,
    DRAMA3_VISIBLE_SLOTS,
    DRAMA3_VISIBLE_SOUND_SLOTS,
    DRAMA3_VOICE_SLOTS,
    UPLOAD_SOUND,
    UPLOAD_VOICE,
    drama3_slot_title,
    drama3_sound_choices,
    drama3_sound_slot_title,
    drama3_sounds_title,
    drama3_voice_choices,
    drama3_voices_title,
)


@dataclass(frozen=True)
class Drama3VoiceSlot:
    speaker: int
    group: gr.Accordion
    preset: gr.Dropdown
    upload: gr.Group
    audio: gr.Audio
    transcribe: gr.Button
    transcript: gr.Textbox
    save_row: gr.Row
    save_name: gr.Textbox
    save: gr.Button
    delete: gr.Button
    description: gr.Textbox
    preview_text: gr.Textbox
    language: gr.Dropdown
    generate: gr.Button
    stop: gr.Button
    status: gr.Markdown
    has_clip: gr.Checkbox
    saved_audio: gr.Audio


@dataclass(frozen=True)
class Drama3SoundSlot:
    index: int
    group: gr.Accordion
    preset: gr.Dropdown
    upload: gr.Group
    audio: gr.Audio
    save_row: gr.Row
    save_name: gr.Textbox
    save: gr.Button
    delete: gr.Button
    description: gr.Textbox
    duration: gr.Slider
    generate: gr.Button
    stop: gr.Button
    status: gr.Markdown
    has_clip: gr.Checkbox


@dataclass(frozen=True)
class Drama3View:
    text: gr.Textbox
    speaker_mode: gr.Radio
    writer: RemotePromptWriterControls
    enhance: gr.Button
    enhance_status: gr.Textbox
    voices: tuple[Drama3VoiceSlot, ...]
    shown_slots: gr.Number
    add_speaker: gr.Button
    remove_speaker: gr.Button
    voice_status: gr.Markdown
    temperature: gr.Slider
    top_p: gr.Slider
    top_k: gr.Slider
    max_frames: gr.Slider
    parallel: gr.Slider
    retries: gr.Slider
    output: gr.Audio
    run: gr.Button
    stop: gr.Button
    status: gr.Textbox
    asr_model: gr.Dropdown
    voice_engine: gr.Radio
    generate_voices: gr.Button
    stop_voices: gr.Button
    skip_existing: gr.Checkbox
    voices_panel: gr.Accordion
    copy_prompt: gr.Button
    writer_prompt: gr.Textbox
    pasted_reply: gr.Textbox
    apply_reply: gr.Button
    script_mode: gr.Radio | None = None
    writer_workflow: gr.Radio | None = None
    sounds_panel: gr.Accordion | None = None
    sounds: tuple[Drama3SoundSlot, ...] = ()
    shown_sound_slots: gr.Number | None = None
    add_sound: gr.Button | None = None
    remove_sound: gr.Button | None = None
    generate_sounds: gr.Button | None = None
    stop_sounds: gr.Button | None = None
    skip_existing_sounds: gr.Checkbox | None = None
    sound_status: gr.Markdown | None = None
    auto: gr.Button | None = None
    stop_auto: gr.Button | None = None
    auto_status: gr.Markdown | None = None
    auto_preload_status: gr.Markdown | None = None
    auto_sound_slots: gr.Number | None = None
    auto_save_speakers: gr.Checkbox | None = None
    speaker_group: gr.Dropdown | None = None
    speaker_group_size: gr.Number | None = None
    new_speaker_group: gr.Textbox | None = None
    create_speaker_group: gr.Button | None = None
    save_all_speakers: gr.Button | None = None
    speaker_group_status: gr.Markdown | None = None
    speaker_library: SpeakerLibraryView | None = None
    takes: gr.Dataframe | None = None
    take_actions: gr.Row | None = None
    take_line: gr.Dropdown | None = None
    regenerate_line: gr.Button | None = None
    remix: gr.Button | None = None
    title: gr.Textbox | None = None
    word_timing: gr.Checkbox | None = None
    repair_words: gr.Button | None = None
    stop_repair_words: gr.Button | None = None
    alignment_result: gr.State | None = None
    alignment_status: gr.Markdown | None = None
    prepare_cast: gr.Button | None = None
    stop_prepare_cast: gr.Button | None = None
    prepare_status: gr.Markdown | None = None
    take_audio: gr.Audio | None = None
    subtitles: gr.HTML | None = None
    steps: StepStrip | None = None
    load_example: gr.Button | None = None
    load_example_zh: gr.Button | None = None


SCRIPT, CAST, SPEAK, REVIEW = range(4)
STEPS_ID = "h3-drama3-steps"
AUTO_ID = "h3-drama3-auto"
GENERATE_ID = "h3-drama3-generate"
# Each step's action card: Song's stage styling, kept in view while the working card scrolls.
STAGE = ["h3-preview-panel", "h3-song-stage", "h3-drama3-stage"]


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
      if (!box.offsetParent) continue;
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


# The script box stays a plain textarea with transparent text over a <pre> that mirrors it
# with colored spans, so editing, undo and every binding keep working. Gradio sets the value
# without an input event when the writer fills it, so the mirror also polls for changes.
DRAMA3_SCRIPT_HIGHLIGHT_JS = """
() => {
  if (window.h3Drama3Highlight) return;
  window.h3Drama3Highlight = true;
  const escape = (text) => text.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]);
  const span = (kind, text) => text ? `<span class="h3-hl-${kind}">${escape(text)}</span>` : "";
  const NAME = /^([ \\t]*)([^\\s:：<>\\[\\]|#][^:：<>\\[\\]|\\n]{0,63}?)([ \\t]*[:：])/;
  const TOKEN = /(\\[[^\\[\\]\\n]*\\])|(<#\\s*-?\\d+(?:\\.\\d+)?\\s*#>)|(<\\|[^|>\\n]*\\|>)|(<\\/?)([A-Za-z_][\\w-]*)([^<>\\n]*?)(\\/?>)|(\\{[^{}\\n]*\\})/g;
  const ATTR = /([\\w-]+)(\\s*=\\s*)("[^"]*"|'[^']*'|[^\\s>]+)?/g;
  // The narrator has its own color; characters take the name colors in order of first line,
  // so no two share one until a ninth character.
  const NARRATOR = new Set(["narrator", "narration", "旁白", "storyteller", "story teller"]);
  const NAME_COLORS = 8;
  const nameKinds = (lines) => {
    const kinds = new Map();
    for (const line of lines) {
      const key = NAME.exec(line)?.[2].trim().toLowerCase();
      if (key && !kinds.has(key)) {
        // span() prefixes "h3-hl-": a character's name gets "h3-hl-name h3-hl-nameN".
        kinds.set(key, NARRATOR.has(key) ? "narrator" : `name h3-hl-name${[...kinds.values()]
          .filter((kind) => kind !== "narrator").length % NAME_COLORS}`);
      }
    }
    return kinds;
  };
  const attrs = (text) => {
    let out = "", last = 0;
    for (const m of text.matchAll(ATTR)) {
      out += escape(text.slice(last, m.index)) + span("attr", m[1]) + escape(m[2]) + span("value", m[3] || "");
      last = m.index + m[0].length;
    }
    return out + escape(text.slice(last));
  };
  const highlight = (script) => {
    let inSound = false;
    const plain = (text) => inSound ? span("sound", text) : escape(text);
    const lines = script.split("\\n");
    const kinds = nameKinds(lines);
    return lines.map((line) => {
      let out = "", rest = line;
      const name = !inSound && NAME.exec(line);
      if (name) {
        out = escape(name[1]) + span(kinds.get(name[2].trim().toLowerCase()), name[2]) + span("punct", name[3]);
        rest = line.slice(name[0].length);
      }
      let last = 0;
      for (const m of rest.matchAll(TOKEN)) {
        out += plain(rest.slice(last, m.index));
        last = m.index + m[0].length;
        if (m[1]) { out += span("cue", m[1]); continue; }
        if (m[2]) { out += span("pause", m[2]); continue; }
        if (m[3]) { out += span("keyword", m[3]); continue; }
        if (m[8]) { out += span("value", m[8]); continue; }  // a pinyin pronunciation note
        const tag = m[5].toLowerCase();
        const kind = tag === "parallel" ? "keyword" : tag === "sequential" ? "sequence" : tag === "sound" ? "tag" : "effect";
        out += span("punct", m[4]) + span(kind, m[5]) + attrs(m[6]) + span("punct", m[7]);
        if (tag === "sound") inSound = m[4] === "<" && m[7] === ">";
      }
      return out + plain(rest.slice(last));
    }).join("\\n") + "\\n";  // a trailing newline keeps the last empty line's height
  };
  const COPIED = ["fontFamily", "fontSize", "fontWeight", "lineHeight", "letterSpacing", "tabSize", "paddingTop",
    "paddingBottom", "paddingLeft", "borderTopWidth", "borderRightWidth", "borderBottomWidth", "borderLeftWidth",
    "borderRadius"];
  const place = (area) => {
    const pre = area.h3Mirror, style = getComputedStyle(area);
    for (const key of COPIED) pre.style[key] = style[key];
    // The textarea's scrollbar narrows its text, so the mirror pads by the same width.
    const scrollbar = area.offsetWidth - area.clientWidth
      - parseFloat(style.borderLeftWidth) - parseFloat(style.borderRightWidth);
    pre.style.paddingRight = `${parseFloat(style.paddingRight) + scrollbar}px`;
    Object.assign(pre.style, { top: `${area.offsetTop}px`, left: `${area.offsetLeft}px`,
      width: `${area.offsetWidth}px`, height: `${area.offsetHeight}px` });
    pre.scrollTop = area.scrollTop;
    pre.scrollLeft = area.scrollLeft;
  };
  const render = (area) => {
    if (area.h3Rendered !== area.value) {
      area.h3Rendered = area.value;
      area.h3Mirror.innerHTML = highlight(area.value);
    }
    place(area);
  };
  const attach = (area) => {
    area.parentElement.querySelector(":scope > pre.h3-drama3-highlight")?.remove();
    const pre = document.createElement("pre");
    pre.className = "h3-drama3-highlight";
    pre.setAttribute("aria-hidden", "true");
    area.before(pre);
    area.h3Mirror = pre;
    area.classList.add("h3-drama3-highlighted");
    area.addEventListener("input", () => render(area));
    area.addEventListener("scroll", () => place(area));
    new ResizeObserver(() => place(area)).observe(area);
  };
  setInterval(() => {
    for (const area of document.querySelectorAll(".h3-drama3-script textarea")) {
      if (!area.offsetParent) continue;
      if (!area.h3Mirror) attach(area);
      render(area);
    }
  }, 200);
}
"""


# A card grid shows one card per shown speaker (or sound) slot and the picked slot below it
# as its editor; the grid's data-kind says which. It reads each slot's title (kept current by
# bindings.py and the writer) and only hides the other slots, so every slot keeps its
# components and bindings. The first slot is picked at first, a new slot from Add when it
# appears, and the last one when the picked slot is removed.
DRAMA3_CAST_JS = """
() => {
  if (window.h3Drama3Cast) return;
  window.h3Drama3Cast = true;
  const NARRATOR = new Set(["narrator", "narration", "旁白", "storyteller", "story teller"]);
  const STATUS = {
    "saved voice": ["Saved voice", "ready"], "voice ready": ["Voice ready", "ready"],
    "saved sound": ["Saved sound", "ready"], "clip ready": ["Clip ready", "ready"],
  };
  // Speakers take the script box's name colors; sounds share the sound color.
  const KINDS = {
    voices: { missing: "Needs a voice", tone: (name, next) => NARRATOR.has(name.toLowerCase()) ? "narrator"
      : name ? String(next() % 8) : "none" },
    sounds: { missing: "Needs a clip", tone: (name) => name ? "cue" : "none" },
  };
  const escape = (text) => text.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const card = (slot) => {
    const title = slot.querySelector("button.label-wrap > span:first-child")?.textContent ?? "";
    const parts = title.split(" · ").map((part) => part.trim());
    const status = STATUS[parts.at(-1)];
    const name = parts.slice(1, status ? -1 : undefined).join(" · ");
    return { id: slot.id, number: parts[0] || "Slot", name, status };
  };
  const render = (grid) => {
    const kind = KINDS[grid.dataset.kind] ? grid.dataset.kind : "voices";
    const panel = grid.closest(`.h3-drama3-${kind}`);
    if (!panel) return;
    // Gradio leaves a slot Add has not shown out of the page, or marks it .hidden.
    const slots = [...panel.querySelectorAll(".h3-drama3-card-slot")].filter((slot) => !slot.closest(".hidden"));
    const cards = slots.map(card);
    let picked = grid.h3Pick ?? grid.dataset.picked;
    grid.h3Pick = undefined;
    if (cards.length === Number(grid.dataset.count || cards.length) + 1) picked = cards.at(-1).id;
    // At first the first slot; after the picked slot is removed, the last one left.
    if (!cards.some((item) => item.id === picked)) picked = (grid.dataset.picked ? cards.at(-1) : cards[0])?.id;
    grid.dataset.count = cards.length;
    let tones = 0;
    const next = () => tones++;
    const html = cards.map((item) => {
      const tone = KINDS[kind].tone(item.name, next);
      const [label, state] = item.status ?? [KINDS[kind].missing, "missing"];
      return `<button type="button" class="h3-drama3-cast-card h3-drama3-tone-${tone}" data-slot="${item.id}" `
        + `data-state="${state}" aria-pressed="${item.id === picked}">`
        + `<span class="h3-drama3-cast-avatar" aria-hidden="true">${escape((item.name || "#").slice(0, 1).toUpperCase())}</span>`
        + `<span class="h3-drama3-cast-name">${escape(item.name || "Unnamed")}</span>`
        + `<span class="h3-drama3-cast-slot">${escape(item.number)}</span>`
        + `<span class="h3-drama3-cast-state">${label}</span></button>`;
    }).join("");
    if (grid.h3Html !== html) {
      grid.h3Html = html;
      grid.innerHTML = html;
    }
    for (const slot of slots) slot.classList.toggle("h3-drama3-cast-off", slot.id !== picked);
    if ((grid.dataset.picked ?? "") !== (picked ?? "")) {
      grid.dataset.picked = picked ?? "";
      open(picked);
    }
  };
  // Open a picked slot that was folded; the user may fold it again.
  const open = (id) => {
    const header = id && document.getElementById(id)?.querySelector("button.label-wrap");
    if (header && !header.classList.contains("open")) header.click();
  };
  setInterval(() => {
    for (const grid of document.querySelectorAll(".h3-drama3-cast")) if (grid.offsetParent) render(grid);
  }, 200);
  document.addEventListener("click", (event) => {
    const button = event.target.closest?.(".h3-drama3-cast-card");
    const grid = button?.closest(".h3-drama3-cast");
    if (!grid) return;
    if (button.dataset.slot === grid.dataset.picked) return open(button.dataset.slot);
    grid.h3Pick = button.dataset.slot;
    render(grid);
  });
}
"""


def _voice_slot(speaker, names):
    with gr.Accordion(drama3_slot_title(speaker), open=True, elem_id=f"h3-drama3-voice-{speaker}",
                      visible=speaker < DRAMA3_VISIBLE_SLOTS,
                      elem_classes=["h3-drama3-slot", "h3-drama3-card-slot"]) as group:
        with gr.Row(equal_height=True):
            preset = gr.Dropdown(drama3_voice_choices(names), value=UPLOAD_VOICE,
                                 label=f"Speaker {speaker} voice")
            save_name = gr.Textbox(label=f"Speaker {speaker} name", placeholder="Character or role",
                                   info=f'Script lines start with this name, or with "Speaker {speaker}:".')
        description = gr.Textbox(label=f"Speaker {speaker} voice description", lines=2,
                                  placeholder="Age range, pitch, timbre, accent, articulation and pace.")
        # A saved speaker hides the upload controls, so its clip plays here instead.
        saved_audio = gr.Audio(label=f"Speaker {speaker} saved voice", type="filepath", interactive=False,
                               visible=False)
        with gr.Group(elem_classes=["h3-drama3-clip"]) as upload:
            with gr.Row(equal_height=False):
                with gr.Column(min_width=220):
                    preview_text = gr.Textbox(label=f"Speaker {speaker} preview speech", lines=2,
                                             placeholder="Words a generated reference clip will speak.")
                    language = gr.Dropdown(list(VOICE_DESIGN_LANGUAGES), value="Auto", label="Preview language")
                    with gr.Row(elem_classes=["h3-drama3-inline-actions"]):
                        generate = gr.Button("Generate voice", size="sm", variant="primary")
                        stop = gr.Button("Interrupt", size="sm", interactive=False)
                        generate.h3_stop = stop
                with gr.Column(min_width=220):
                    audio = gr.Audio(label=f"Speaker {speaker} reference clip", type="filepath")
                    transcribe = gr.Button("Auto transcribe", size="sm")
        transcript = gr.Textbox(label=f"Speaker {speaker} exact transcript", lines=2,
                                placeholder="Every word spoken in the clip.")
        with gr.Row(elem_classes=["h3-drama3-inline-actions"]):
            with gr.Row() as save_row:
                save = gr.Button("Save speaker", size="sm")
            delete = gr.Button("Delete saved speaker", size="sm", variant="stop", visible=False)
        status = gr.Markdown("", elem_classes=["h3-drama3-note"])
        has_clip = gr.Checkbox(False, visible=False)
    return Drama3VoiceSlot(
        speaker=speaker,
        group=group,
        preset=preset,
        upload=upload,
        audio=audio,
        transcribe=transcribe,
        transcript=transcript,
        save_row=save_row,
        save_name=save_name,
        save=save,
        delete=delete,
        description=description,
        preview_text=preview_text,
        language=language,
        generate=generate,
        stop=stop,
        status=status,
        has_clip=has_clip,
        saved_audio=saved_audio,
    )


def _sound_slot(index, names):
    with gr.Accordion(
        drama3_sound_slot_title(index),
        open=True,
        elem_id=f"h3-drama3-sound-{index}",
        visible=index < DRAMA3_VISIBLE_SOUND_SLOTS,
        elem_classes=["h3-drama3-slot", "h3-drama3-card-slot"],
    ) as group:
        with gr.Row(equal_height=True):
            preset = gr.Dropdown(
                drama3_sound_choices(names),
                value=UPLOAD_SOUND,
                label=f"Sound {index} preset",
            )
            save_name = gr.Textbox(
                label=f"Sound {index} name · <sound name=\"...\">",
                placeholder="e.g. rain, thunder, noir_piano",
            )
        description = gr.Textbox(
            label=f"Sound {index} prompt (Stable Audio 3 Medium)",
            lines=2,
            placeholder="Detailed English prompt for sound effect or instrumental background music.",
        )
        duration = gr.Slider(
            1,
            47,
            value=5,
            step=1,
            label=f"Sound {index} duration (seconds)",
            info="About 3–6 s for one-shot effects, 10–15 s for looping ambience or music.",
        )
        with gr.Group(elem_classes=["h3-drama3-clip"]) as upload:
            audio = gr.Audio(label=f"Sound {index} clip", type="filepath")
            with gr.Row(elem_classes=["h3-drama3-inline-actions"]):
                generate = gr.Button("Generate sound", size="sm", variant="primary")
                stop = gr.Button("Interrupt", size="sm", interactive=False)
                generate.h3_stop = stop
        with gr.Row(elem_classes=["h3-drama3-inline-actions"]):
            with gr.Row() as save_row:
                save = gr.Button("Save sound", size="sm")
            delete = gr.Button("Delete saved sound", size="sm", variant="stop", visible=False)
        status = gr.Markdown("", elem_classes=["h3-drama3-note"])
        has_clip = gr.Checkbox(False, visible=False)
    return Drama3SoundSlot(
        index=index,
        group=group,
        preset=preset,
        upload=upload,
        audio=audio,
        save_row=save_row,
        save_name=save_name,
        save=save,
        delete=delete,
        description=description,
        duration=duration,
        generate=generate,
        stop=stop,
        status=status,
        has_clip=has_clip,
    )


def build_drama3_view(root, speaker_names=(), sound_names=(), speaker_groups=("Global",), group_size=None):
    root.elem_classes = [*root.elem_classes, "h3-song", "h3-drama3"]
    with root:
        intro("Soundstage · powered by Drama 3", "Turn any text into an audio drama.",
              "Write and direct a script, cast the voices and sounds, speak it, then review every line. "
              "Or paste any text and let <strong>Auto</strong> do every step.", "AUDIO DRAMA")
        steps = step_strip(STEPS_ID)
        with steps.tabs:
            with steps.step(SCRIPT, "Script"):
                with gr.Row(equal_height=False, elem_classes=["h3-generator-shell"]):
                    with gr.Column(scale=3, min_width=360, elem_classes=["h3-composer"]):
                        card_heading("What should they say?",
                                     "Start each line with a speaker's name from the cast, such as Mia:, and add "
                                     "[cues] in brackets. Or paste any text and let the writer script it.")
                        with gr.Row(elem_classes=["h3-drama3-inline-actions"]):
                            load_example = gr.Button("Load example story", size="sm",
                                                     elem_id="h3-drama3-load-example")
                            load_example_zh = gr.Button("Load Chinese example (中文)", size="sm",
                                                        elem_id="h3-drama3-load-example-zh")
                        title = gr.Textbox(label="Title", max_lines=1, placeholder="Chapter or story title",
                                           info="Names the output file. The writer fills it from the script.")
                        text = gr.Textbox(label="Speech text", lines=18, max_lines=60,
                                          elem_classes=["h3-drama3-script"],
                                          placeholder="Mia: Welcome [with warm excitement] back!\n"
                                                      "Leo: Took you long enough. [laughs]")
                        script_mode = gr.Radio(
                            list(DRAMA3_SCRIPT_MODES),
                            value=DRAMA3_SCRIPT_ADVANCED,
                            label="Script mode",
                            info="Basic is dialogue with expression cues. Advanced adds sound effects, music and "
                                 "vocal effects; Basic keeps those tags in the script but ignores them.",
                        )
                    with gr.Column(scale=2, min_width=300, elem_classes=STAGE):
                        card_heading("Rewrite as a Drama 3 script",
                                     "Splits narration from dialogue, adds speaker names and cues, and fills the "
                                     "cast. Advanced also stages sound effects, music and vocal effects.")
                        speaker_mode = gr.Radio(
                            list(DRAMA3_SPEAKER_MODES), value=DRAMA3_SPEAKER_AUTO, label="Speakers",
                            info="Auto decides from the text. Adapt to voices uses one speaker per voice in the "
                                 "cast. Single keeps one voice throughout.",
                        )
                        writer_workflow = gr.Radio(
                            list(DRAMA3_WORKFLOWS), value=DRAMA3_WORKFLOW_SINGLE, label="Writer workflow",
                            info="Single-pass writes the script in one call, best with strong models. Guided "
                                 "writes it in short steps (casting, voices, acting, sound), so smaller, faster "
                                 "models give reliable scripts; it makes 3–6 calls.",
                        )
                        enhance = gr.Button("Rewrite script", variant="primary", elem_classes=["h3-primary-action"])
                        enhance_status = gr.Textbox(label="Writer status", lines=2, interactive=False)
                        with gr.Accordion("Writing service", open=False):
                            writer = build_remote_prompt_writer_controls(GEMINI_PROMPT_MODELS,
                                                                         DEFAULT_GEMINI_PROMPT_MODEL)
                        with gr.Accordion("Use another chat model", open=False):
                            gr.Markdown("Copy the full writer prompt into any chat model, then paste its JSON reply. "
                                        "It is checked like the built-in writer before anything changes.",
                                        elem_classes=["h3-drama3-note"])
                            copy_prompt = gr.Button("Copy full writer prompt", size="sm",
                                                    elem_id="h3-drama3-copy-prompt")
                            # A hidden Textbox, not gr.State: State values stay on the server, so the copy JS got None.
                            writer_prompt = gr.Textbox(visible="hidden", container=False, show_label=False)
                            pasted_reply = gr.Textbox(label="Pasted writer reply", lines=6, max_lines=20,
                                                      placeholder='{"script": "Mia: ...", "speakers": [{"speaker_id": 0, "name": "Mia", ...}]}')
                            apply_reply = gr.Button("Apply pasted reply", size="sm", variant="primary")
                        steps.button("Auto: rewrite, voice, sound and speak →", SPEAK, press=AUTO_ID,
                                     classes=("h3-song-skip",))
                        gr.Markdown("**Auto** does every step with the settings on Speak; follow it there.",
                                    elem_classes=["h3-song-hint"])
                        with gr.Row(elem_classes=["h3-song-step-nav"]):
                            steps.button("Next: cast & sounds →", CAST, variant="primary")
            with steps.step(CAST, "Cast & sounds"):
                with gr.Row(equal_height=False, elem_classes=["h3-generator-shell"]):
                    with gr.Column(scale=3, min_width=360, elem_classes=["h3-composer"]):
                        card_heading("Who speaks, and what do we hear?",
                                     "Every speaker needs a voice: upload a reference clip, generate one from a "
                                     "description, or reuse a saved speaker. Advanced scripts also need sounds.")
                        with gr.Accordion(drama3_voices_title(group="Global", group_size=group_size),
                                          open=True, elem_classes=["h3-drama3-voices"]) as voices_panel:
                            with gr.Group(elem_classes=["h3-drama3-toolbar"]):
                                with gr.Row(equal_height=True):
                                    speaker_group = gr.Dropdown(list(speaker_groups), value="Global",
                                        label="Speaker preset group",
                                        info="Saves go here. Names match this group first, then Global.")
                                    speaker_group_size = gr.Number(group_size, precision=0, visible=False)
                                    new_speaker_group = gr.Textbox(label="New speaker group",
                                                                   placeholder="Story or series name",
                                                                   info="Reuse one group across chapters.")
                                with gr.Row(elem_classes=["h3-drama3-inline-actions"]):
                                    create_speaker_group = gr.Button("Create group", size="sm")
                                    save_all_speakers = gr.Button("Save all speakers to selected group", size="sm")
                                    library_open = gr.Button("Manage saved speakers", size="sm")
                                speaker_group_status = gr.Markdown(
                                    "Global voices are available in every group.", elem_classes=["h3-drama3-note"])
                                with gr.Row(elem_classes=["h3-drama3-inline-actions"]):
                                    generate_voices = gr.Button("Generate all speaker voices", size="sm", variant="primary")
                                    stop_voices = gr.Button("Interrupt", size="sm", interactive=False)
                                    generate_voices.h3_stop = stop_voices
                                voice_status = gr.Markdown("", elem_classes=["h3-drama3-note"])
                            speaker_library = build_speaker_library(speaker_groups, library_open)
                            # Filled in the browser by DRAMA3_CAST_JS from the slots below.
                            gr.HTML('<div class="h3-drama3-cast" data-kind="voices" role="group" '
                                    'aria-label="Cast: pick a speaker to edit"></div>',
                                    elem_classes=["h3-drama3-cast-block"])
                            voices = tuple(_voice_slot(speaker, list(speaker_names))
                                           for speaker in range(DRAMA3_VOICE_SLOTS))
                            shown_slots = gr.Number(DRAMA3_VISIBLE_SLOTS, precision=0, visible=False)
                            with gr.Row(elem_classes=["h3-drama3-inline-actions", "h3-drama3-slot-count"]):
                                add_speaker = gr.Button("Add speaker", size="sm")
                                remove_speaker = gr.Button("Remove last speaker", size="sm")
                        with gr.Accordion(drama3_sounds_title(), open=False,
                                          elem_classes=["h3-drama3-sounds"]) as sounds_panel:
                            with gr.Group(elem_classes=["h3-drama3-toolbar"]):
                                gr.Markdown(
                                    "Clips for `<sound name=\"...\">` tags, made with "
                                    "[Stable Audio 3 Medium](https://huggingface.co/stabilityai/stable-audio-3-medium). "
                                    "An Advanced rewrite fills these slots; any sound without a clip is generated "
                                    "during speech.",
                                    elem_classes=["h3-drama3-note"],
                                )
                                skip_existing_sounds = gr.Checkbox(
                                    True,
                                    label="Skip sounds with a clip",
                                    info="Uncheck to regenerate existing sound clips.",
                                )
                                with gr.Row(elem_classes=["h3-drama3-inline-actions"]):
                                    generate_sounds = gr.Button("Generate all sounds", size="sm", variant="primary")
                                    stop_sounds = gr.Button("Interrupt", size="sm", interactive=False)
                                    generate_sounds.h3_stop = stop_sounds
                                sound_status = gr.Markdown("", elem_classes=["h3-drama3-note"])
                            gr.HTML('<div class="h3-drama3-cast" data-kind="sounds" role="group" '
                                    'aria-label="Sounds: pick one to edit"></div>',
                                    elem_classes=["h3-drama3-cast-block"])
                            sounds = tuple(_sound_slot(i, list(sound_names)) for i in range(DRAMA3_SOUND_SLOTS))
                            shown_sound_slots = gr.Number(DRAMA3_VISIBLE_SOUND_SLOTS, precision=0, visible=False)
                            with gr.Row(elem_classes=["h3-drama3-inline-actions", "h3-drama3-slot-count"]):
                                add_sound = gr.Button("Add sound", size="sm")
                                remove_sound = gr.Button("Remove last sound", size="sm")
                    with gr.Column(scale=2, min_width=300, elem_classes=STAGE):
                        card_heading("Prepare voices & sounds",
                                     "Generate all speaker voices and the sounds for an Advanced script in one job. "
                                     "Each panel also has its own generation button.")
                        voice_engine = gr.Radio(
                            list(VOICE_DESIGN_ENGINES), value=ENGINE_QWEN, label="Voice generator",
                            info="[Qwen3-TTS VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) "
                                 "designs a voice from its description (downloaded on first use). Drama 3 uses "
                                 "SGLang-Omni and the sampling settings on Speak.",
                        )
                        skip_existing = gr.Checkbox(
                            True, label="Skip speakers with a voice",
                            info="Uncheck to regenerate, e.g. after changing the backend.")
                        asr_model = gr.Dropdown(list(DRAMA3_ASR_CHOICES), value=DEFAULT_DRAMA3_ASR,
                                                label="Voice transcript model",
                                                info="Used by Auto transcribe; 0.6B uses less memory.")
                        with gr.Column(elem_classes=["h3-action-dock"]):
                            with gr.Row(elem_classes=["h3-generation-buttons"]):
                                prepare_cast = gr.Button("Generate all speaker voices and sounds", variant="primary",
                                                         scale=3, elem_classes=["h3-primary-action"])
                                stop_prepare_cast = gr.Button("Interrupt", scale=1, interactive=False)
                                prepare_cast.h3_stop = stop_prepare_cast
                            prepare_status = gr.Markdown("", elem_classes=["h3-drama3-note"])
                        with gr.Row(elem_classes=["h3-song-step-nav"]):
                            steps.button("← Script", SCRIPT)
                            steps.button("Next: speak →", SPEAK, variant="primary")
            with steps.step(SPEAK, "Speak"):
                # The result (player, timeline, subtitles and stage) outgrows the mostly folded
                # settings, so here the stage column takes the wider share and scrolls with the page.
                with gr.Row(equal_height=False, elem_classes=["h3-generator-shell"]):
                    with gr.Column(scale=2, min_width=300, elem_classes=["h3-composer"]):
                        card_heading("How it should sound", "Unchanged lines reuse their saved takes, so speaking "
                                                            "again after a few fixes is quick.")
                        sampling_preset = gr.Radio(
                            list(DRAMA3_SAMPLING_PRESETS), value="Expressive", label="Speech preset",
                            info="Strict keeps delivery controlled; Expressive adds variation.",
                        )
                        word_timing = gr.Checkbox(
                            False, label="Word-by-word subtitles",
                            info="Times every word with Qwen3's aligner after speaking. To add it to a result, "
                                 "turn it on and Remix or generate again: saved takes are only aligned, not "
                                 "spoken again.")
                        with gr.Accordion("Speech settings", open=False):
                            gr.Markdown("Drama 3 runs on SGLang-Omni (Linux, CUDA). Lines go out in parallel with "
                                        "their speaker's clip. The checkpoint downloads on first use. "
                                        "[Model and license](https://huggingface.co/fishaudio/drama-3)",
                                        elem_classes=["h3-drama3-note"])
                            with gr.Row():
                                parallel = gr.Slider(1, 32, value=16, step=1, label="Parallel requests")
                                retries = gr.Slider(0, 4, value=0, step=1, label="Retries per chunk",
                                    info="0 keeps the first take. Retries only near-silent audio.")
                            with gr.Row():
                                temperature = gr.Slider(0, 2, value=2.0, step=0.05, label="Temperature")
                                top_p = gr.Slider(0.05, 1, value=1.0, step=0.05, label="Top-p")
                            top_k = gr.Slider(0, 4096, value=30, step=1, label="Top-k",
                                info="0 keeps all candidates. Small caps can make speech stall at pauses.")
                            max_frames = gr.Slider(1, 8192, value=8192, step=1, label="Maximum audio frames per take",
                                info="21.5 frames per second; 8192 is about 6.3 minutes. Only limits a take that "
                                     "runs on.")
                        with gr.Column(elem_classes=["h3-drama3-auto"]):
                            gr.HTML('<div class="h3-drama3-auto-heading"><strong>Auto</strong>'
                                    '<p>Rewrites the script, generates every voice (and every sound in Advanced), '
                                    'then speaks it with these settings.</p></div>')
                            with gr.Row(elem_classes=["h3-generation-buttons"]):
                                auto = gr.Button("Auto: rewrite, voice, sound and speak", size="sm",
                                                 variant="primary", elem_id=AUTO_ID)
                                stop_auto = gr.Button("Interrupt", size="sm", interactive=False)
                            auto_save_speakers = gr.Checkbox(False, label="Save generated speakers to the selected group",
                                info="After the voices are generated, Auto saves them like Save all speakers.")
                            auto_status = gr.Markdown("Paste any text on Script, pick a script mode, then run Auto.",
                                                      elem_classes=["h3-drama3-note"])
                            # Models Auto loads on the GPU while the writer rewrites the script.
                            auto_preload_status = gr.Markdown("", elem_classes=["h3-drama3-note"])
                            auto_sound_slots = gr.Number(0, precision=0, visible=False)
                    with gr.Column(scale=3, min_width=360, elem_classes=[*STAGE, "h3-drama3-result"]):
                        card_heading("Speak it", "Every line goes out with its speaker's voice; the mix adds "
                                                 "sounds and effects.")
                        with gr.Column(elem_classes=["h3-action-dock"]):
                            run, stop, status = generation_actions("Generate speech", elem_id=GENERATE_ID)
                        output = gr.Audio(label="Generated speech", type="filepath", interactive=False)
                        subtitles = gr.HTML("", visible=False, elem_classes=["h3-drama3-subtitle-panel"])
                        output.h3_metadata_root = gr.Column()
                        with gr.Row(elem_classes=["h3-song-step-nav"]):
                            steps.button("← Cast & sounds", CAST)
                            steps.button("Review lines →", REVIEW, variant="primary")
                        gr.Markdown("Every production is also saved to **Media**.", elem_classes=["h3-song-hint"])
                sampling_controls = [temperature, top_p, top_k]
                presets_js = json.dumps(DRAMA3_SAMPLING_PRESETS)
                sampling_preset.input(fn=None, js=f"(name) => ({presets_js})[name]",
                                      inputs=sampling_preset, outputs=sampling_controls, **QUIET)
                gr.on(
                    [control.change for control in sampling_controls], fn=None,
                    js=f"""(...values) => Object.entries({presets_js}).find(
                        ([, preset]) => preset.every((value, index) => value === values[index]))?.[0] ?? null""",
                    inputs=sampling_controls, outputs=sampling_preset, **QUIET,
                )
            with steps.step(REVIEW, "Review lines"):
                card_heading("Review every line",
                             "Click a line to hear its take. Regenerate one you don't like, or remix after "
                             "changing sounds or effects. Lines appear here once speech is generated.")
                with gr.Column(visible=False, elem_classes=["h3-drama3-lines"]) as take_actions:
                    with gr.Row(equal_height=False, elem_classes=["h3-generator-shell"]):
                        with gr.Column(scale=3, min_width=320, elem_classes=["h3-composer"]):
                            takes = gr.Dataframe(headers=["#", "Speaker", "Line", "Seconds", "Take"], label="Takes",
                                                 interactive=False, wrap=True, visible=False, max_height=560,
                                                 column_widths=["6%", "16%", "58%", "10%", "10%"])
                        with gr.Column(scale=2, min_width=280, elem_classes=STAGE):
                            card_heading("Line take", "The picked line before staging and effects.")
                            take_line = gr.Dropdown([], label="Line", info="Generate speech reuses unchanged lines.",
                                                    allow_custom_value=True)
                            take_audio = gr.Audio(label="Line take (before staging and effects)", type="numpy",
                                                  interactive=False)
                            with gr.Row(elem_classes=["h3-generation-buttons"]):
                                regenerate_line = gr.Button("Regenerate line", variant="primary")
                                remix = gr.Button("Remix")
                            regenerate_line.h3_stop = remix.h3_stop = stop
                            alignment_result = gr.State(None)
                            with gr.Row():
                                repair_words = gr.Button("Generate / repair word alignment", interactive=False)
                                stop_repair_words = gr.Button("Interrupt alignment", interactive=False)
                            repair_words.h3_stop = stop_repair_words
                            alignment_status = gr.Markdown("Word alignment has not been checked for this result.",
                                                           elem_classes=["h3-drama3-note"])
                            gr.Markdown("Align this result from its saved takes without regenerating speech or "
                                        "remixing. Missing timings are retried; successful ones are reused.",
                                        elem_classes=["h3-song-hint"])
                with gr.Row(elem_classes=["h3-song-step-nav"]):
                    steps.button("← Speak", SPEAK)
                    steps.button("Generate speech again", SPEAK, press=GENERATE_ID)
    return Drama3View(
        text=text,
        speaker_mode=speaker_mode,
        writer=writer,
        enhance=enhance,
        enhance_status=enhance_status,
        voices=voices,
        shown_slots=shown_slots,
        add_speaker=add_speaker,
        remove_speaker=remove_speaker,
        voice_status=voice_status,
        prepare_cast=prepare_cast,
        stop_prepare_cast=stop_prepare_cast,
        prepare_status=prepare_status,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_frames=max_frames,
        parallel=parallel,
        retries=retries,
        output=output,
        run=run,
        stop=stop,
        status=status,
        asr_model=asr_model,
        voice_engine=voice_engine,
        generate_voices=generate_voices,
        stop_voices=stop_voices,
        skip_existing=skip_existing,
        voices_panel=voices_panel,
        copy_prompt=copy_prompt,
        writer_prompt=writer_prompt,
        pasted_reply=pasted_reply,
        apply_reply=apply_reply,
        script_mode=script_mode,
        writer_workflow=writer_workflow,
        sounds_panel=sounds_panel,
        sounds=sounds,
        shown_sound_slots=shown_sound_slots,
        add_sound=add_sound,
        remove_sound=remove_sound,
        generate_sounds=generate_sounds,
        stop_sounds=stop_sounds,
        skip_existing_sounds=skip_existing_sounds,
        sound_status=sound_status,
        auto=auto,
        stop_auto=stop_auto,
        auto_status=auto_status,
        auto_preload_status=auto_preload_status,
        auto_sound_slots=auto_sound_slots,
        auto_save_speakers=auto_save_speakers,
        speaker_group=speaker_group,
        speaker_group_size=speaker_group_size,
        new_speaker_group=new_speaker_group,
        create_speaker_group=create_speaker_group,
        save_all_speakers=save_all_speakers,
        speaker_group_status=speaker_group_status,
        speaker_library=speaker_library,
        takes=takes,
        take_actions=take_actions,
        take_line=take_line,
        regenerate_line=regenerate_line,
        remix=remix,
        word_timing=word_timing,
        repair_words=repair_words,
        stop_repair_words=stop_repair_words,
        alignment_result=alignment_result,
        alignment_status=alignment_status,
        title=title,
        take_audio=take_audio,
        subtitles=subtitles,
        steps=steps,
        load_example=load_example,
        load_example_zh=load_example_zh,
    )
