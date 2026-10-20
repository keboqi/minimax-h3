"""Drama 3 slot contracts and browser-side presentation helpers."""

from dataclasses import dataclass
import json
import gradio as gr
from h3_ui.optional_engine_controls import _SKIP_JS


DRAMA3_VOICE_SLOTS = 16


DRAMA3_VISIBLE_SLOTS = 4


DRAMA3_SLOT_INPUT_NAMES = tuple(
    f"speaker_{speaker}_{field}"
    for speaker in range(DRAMA3_VOICE_SLOTS)
    for field in ("voice", "audio", "transcript")
)


DRAMA3_CAST_INPUT_NAMES = tuple(
    f"speaker_{speaker}_{field}"
    for speaker in range(DRAMA3_VOICE_SLOTS)
    for field in ("name", "description", "preview_text")
)


UPLOAD_VOICE = "Upload a clip"


DRAMA3_SOUND_SLOTS = 16


DRAMA3_VISIBLE_SOUND_SLOTS = 4


UPLOAD_SOUND = "Upload a clip"


DRAMA3_SOUND_SLOT_INPUT_NAMES = tuple(
    f"sound_{index}_{field}"
    for index in range(DRAMA3_SOUND_SLOTS)
    for field in ("preset", "audio", "name", "description", "duration")
)


DRAMA3_SOUND_META_INPUT_NAMES = tuple(
    f"sound_{index}_{field}"
    for index in range(DRAMA3_SOUND_SLOTS)
    for field in ("name", "description", "duration")
)


DRAMA3_SAMPLING_PRESETS = {
    "Strict": (0.7, 0.7, 30),
    "Balance": (1.0, 0.9, 0),
    "Expressive": (2.0, 1.0, 30),
}


def drama3_voice_choices(names):
    # Keep the existing API value while showing both ways to supply a reference.
    return [("Upload or generate a voice", UPLOAD_VOICE), *names]


def drama3_sound_choices(names):
    return [("Upload or generate a sound", UPLOAD_SOUND), *names]


@dataclass(frozen=True)
class _SlotKind:
    """Wording and limits shared by a slot list's titles and their browser twins."""
    slot: str
    upload: str
    saved: str
    ready: str
    panel: str
    noun: str
    asset: str
    total: int


_VOICE_KIND = _SlotKind("Speaker", UPLOAD_VOICE, "saved voice", "voice ready",
                        "Speaker voices", "speaker", "voice", DRAMA3_VOICE_SLOTS)


_SOUND_KIND = _SlotKind("Sound", UPLOAD_SOUND, "saved sound", "clip ready",
                        "Sound effects and music", "sound", "clip", DRAMA3_SOUND_SLOTS)


def _clamp_count(count, total):
    try:
        return max(1, min(total, int(count or 1)))
    except (TypeError, ValueError):
        return 1


def _slot_title(kind, index, name, preset, has_clip):
    title = f"{kind.slot} {index}"
    if str(name or "").strip():
        title += f" \u00b7 {str(name).strip()}"
    if preset and preset != kind.upload:
        title += f" \u00b7 {kind.saved}"
    elif has_clip:
        title += f" \u00b7 {kind.ready}"
    return title


def _panel_title(kind, shown, presets, has_clips, group=None, group_size=None):
    shown = _clamp_count(shown, kind.total)
    ready = sum(1 for index in range(shown)
                if (index < len(presets) and presets[index] and presets[index] != kind.upload)
                or (index < len(has_clips) and has_clips[index]))
    group = str(group or "").strip()
    grouped = f" \u00b7 {group} group" if group else ""
    if group and group_size is not None:
        grouped += f" ({int(group_size)} saved)"
    return f"{kind.panel}{grouped} \u00b7 {shown} {kind.noun}{'s' if shown != 1 else ''} \u00b7 {ready} with a {kind.asset}"


def _count_js(total):
    return f"Math.max(1, Math.min({total}, Math.trunc(Number(shown)) || 1))"


def _slot_title_js(kind, index):
    """Browser twin of ``_slot_title``; also derives the slot's has_clip flag from its audio."""
    return f"""(name, preset, audio) => {{
  const ready = Boolean(audio);
  const trimmed = String(name ?? "").trim();
  let title = {json.dumps(f"{kind.slot} {index}")};
  if (trimmed) title += " \u00b7 " + trimmed;
  if (preset && preset !== {json.dumps(kind.upload)}) title += {json.dumps(f" · {kind.saved}")};
  else if (ready) title += {json.dumps(f" · {kind.ready}")};
  return [{{__type__: "update", label: title}}, ready];
}}"""


def _panel_title_js(kind, grouped=False):
    """Browser twin of ``_panel_title``; inputs are the group and its saved count when
    ``grouped``, the shown count, each preset, then each has_clip flag."""
    return f"""({'group, size, ' if grouped else ''}shown, ...values) => {{
  const count = {_count_js(kind.total)};
  const groupName = {'String(group ?? "").trim()' if grouped else '""'};
  const saved = {'size === null || size === undefined || size === "" ? "" : " (" + Math.trunc(Number(size)) + " saved)"' if grouped else '""'};
  let ready = 0;
  for (let index = 0; index < count; index += 1) {{
    const preset = values[index];
    if ((preset && preset !== {json.dumps(kind.upload)}) || values[{kind.total} + index]) ready += 1;
  }}
  const label = {json.dumps(kind.panel)} + (groupName ? " · " + groupName + " group" + saved : "")
    + " · " + count + " " + {json.dumps(kind.noun)}
    + (count === 1 ? "" : "s") + " \u00b7 " + ready + {json.dumps(f" with a {kind.asset}")};
  return {{__type__: "update", label}};
}}"""


def _slot_count_js(total, step, cleared=()):
    """Browser twin of ``_slot_visibility`` for Add (+1) and Remove (-1).

    Outputs: the count, each slot group, Add, Remove, then one output per slot
    for each field group in ``cleared``. A removed slot is reset to those values
    so a hidden slot never sends anything.
    """
    return f"""(shown) => {{
  const skip = {_SKIP_JS};
  const before = {_count_js(total)};
  const count = Math.max(1, Math.min({total}, before + ({step})));
  const outputs = [count];
  for (let index = 0; index < {total}; index += 1) {{
    outputs.push((index < count) === (index < before) ? skip : {{__type__: "update", visible: index < count}});
  }}
  outputs.push({{__type__: "update", interactive: count < {total}}}, {{__type__: "update", interactive: count > 1}});
  for (const fields of {json.dumps([list(group) for group in cleared])}) {{
    for (let index = 0; index < {total}; index += 1) {{
      for (const value of fields) {{
        outputs.push(count < before && index === before - 1 ? {{__type__: "update", value}} : skip);
      }}
    }}
  }}
  return outputs;
}}"""


def drama3_slot_title(speaker, name="", preset=UPLOAD_VOICE, has_clip=False):
    """Panel title: index, character name, and whether a voice is ready."""
    return _slot_title(_VOICE_KIND, speaker, name, preset, has_clip)


def drama3_voices_title(shown=DRAMA3_VISIBLE_SLOTS, presets=(), has_clips=(), group=None, group_size=None):
    """Speaker voices panel title: preset group and its saved count, shown speakers, and how many have a voice."""
    return _panel_title(_VOICE_KIND, shown, presets, has_clips, group, group_size)


def drama3_sound_slot_title(index, name="", preset=UPLOAD_SOUND, has_clip=False):
    """Sound slot title: index, sound name, and whether a clip is ready."""
    return _slot_title(_SOUND_KIND, index, name, preset, has_clip)


def drama3_sounds_title(shown=DRAMA3_VISIBLE_SOUND_SLOTS, presets=(), has_clips=()):
    """Sound effects and music panel title with shown count and ready count."""
    return _panel_title(_SOUND_KIND, shown, presets, has_clips)


def _slot_visibility(count, previous=None, total=DRAMA3_VOICE_SLOTS):
    """Show ``count`` slots; disable Add at the limit and Remove at one slot.

    With the ``previous`` count only slots whose visibility changes are sent:
    each visibility update of an unrendered slot makes Gradio re-walk its layout.
    """
    count = _clamp_count(count, total)
    if previous is not None:
        previous = _clamp_count(previous, total)
        if previous == count:
            return (gr.skip(),) * (total + 3)
    groups = [gr.skip() if previous is not None and (index < count) == (index < previous)
              else gr.update(visible=index < count) for index in range(total)]
    return (count, *groups, gr.update(interactive=count < total), gr.update(interactive=count > 1))


def _sound_slot_visibility(count, previous=None):
    """Show ``count`` sound slots; disable Add at the limit and Remove at one slot."""
    return _slot_visibility(count, previous, DRAMA3_SOUND_SLOTS)

