"""What Dub & Translate's custom panels show: pure functions from sessions and settings.

The line board, speaker cards and earlier sessions render in the browser from the
values built here; every text in those values is HTML-escaped already. The step
plan, source card and dub readiness are plain HTML.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
import re
import time

from h3_app.dubbing import pipeline, session as sessions
from h3_app.dubbing.segments import DEFAULT_SPEAKER

# Speakers take name colors in order of first line; the ninth reuses the first.
TONES = 8
SEPARATION = "separate"
TRANSCRIPTION = "transcribe"
TRANSLATION = "translate"


def clock(seconds: float) -> str:
    """``73.25`` → ``1:13.3``; an hour or more → ``1:02:03.0``."""
    tenths = max(0, int(round(float(seconds) * 10)))
    hours, rest = divmod(tenths, 36000)
    minutes, rest = divmod(rest, 600)
    text = f"{minutes:02d}:{rest / 10:04.1f}" if hours else f"{minutes}:{rest / 10:04.1f}"
    return f"{hours}:{text}" if hours else text


def speaker_label(speaker: str) -> str:
    found = re.fullmatch(r"speaker(\d+)", str(speaker or ""))
    return f"Speaker {found.group(1)}" if found else str(speaker or DEFAULT_SPEAKER)


def _plural(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


# ---------------------------------------------------------------------------
# Review lines
# ---------------------------------------------------------------------------


def summary_html(session: sessions.Session, long: set[int] | None = None) -> str:
    """One line on the session's lines: how many, who speaks, what is ready and what needs a look."""
    lines = session.segments
    if not lines:
        return "No lines yet."
    long = set(pipeline.long_lines(lines)) if long is None else long
    ready = sum(bool(s["translation"]) and s.get("dub", True) for s in lines)
    untranslated = sum(not s["translation"] for s in lines)
    target = escape(session.data.get("target_language") or "the target language")
    parts = [_plural(len(lines), "line"), _plural(len({s["speaker"] for s in lines}), "speaker"),
             f"<strong>{ready} ready</strong> to dub into {target}"]
    if untranslated:
        parts.append(f"{untranslated} untranslated")
    issues = pipeline.translation_issues(lines)
    if issues:
        parts.append(f"{_plural(len(issues), 'translation')} {'needs' if len(issues) == 1 else 'need'} review")
    failed = pipeline.take_problems(session)
    if failed:
        parts.append(f"{len(failed)} with a failed take")
    rushed = pipeline.timing_problems(session)
    if rushed:
        parts.append(f"{len(rushed)} too long in the last dub")
    if long:
        parts.append(f"{len(long)} may be too long to say in time")
    return " · ".join(parts)


def board_value(session: sessions.Session, previous=None, *, focus: int | None = None,
                picked: list[int] | None = None, keep_focus: bool = False) -> dict:
    """The line board: escaped lines with their flags, the speakers, and the viewer's focus, picks and filter.

    The focus, picks and filter carry over from ``previous`` while it shows the same session.
    With ``keep_focus`` the browser keeps whatever line the viewer has focused since
    ``previous`` was sent, so a save that arrives after a click does not move the focus back.
    """
    lines = session.segments
    previous = previous if isinstance(previous, dict) and previous.get("session") == session.id else {}
    order = list(dict.fromkeys(s["speaker"] for s in lines))
    index = {speaker: number for number, speaker in enumerate(order)}
    long = set(pipeline.long_lines(lines))
    failed = pipeline.take_problems(session)
    rushed = pipeline.timing_problems(session)
    items = []
    issues = set(pipeline.translation_issues(lines))
    for number, line in enumerate(lines):
        flags = []
        if not line.get("dub", True):
            flags.append("off")
        elif not line["translation"]:
            flags.append("empty")
        else:
            if number in issues:
                flags.append("translation")
            if number in failed:
                flags.append("failed")
            if number in rushed:
                flags.append("rushed")
            if number in long:
                flags.append("long")
        items.append({
            "s": line["start"], "e": line["end"], "time": f"{clock(line['start'])} – {clock(line['end'])}",
            "speaker": index[line["speaker"]], "label": escape(speaker_label(line["speaker"])),
            "tone": index[line["speaker"]] % TONES, "source": escape(line["source"]),
            "translation": escape(line["translation"]), "dub": bool(line.get("dub", True)), "flags": flags,
            "problem": escape(failed.get(number, ""), quote=True),
            "timing": escape(rushed.get(number, ""), quote=True),
        })
    count = len(items)
    focus = previous.get("focus") if focus is None else focus
    if count:
        focus = min(max(int(focus), 0), count - 1) if isinstance(focus, (int, float)) else 0
    else:
        focus = None
    picked = (previous.get("picked") or []) if picked is None else picked
    picked = sorted({int(number) for number in picked if isinstance(number, (int, float)) and 0 <= number < count})
    speakers = [{"k": number, "label": escape(speaker_label(speaker)), "tone": number % TONES,
                 "count": sum(line["speaker"] == speaker for line in lines)} for number, speaker in enumerate(order)]
    duration = float(session.data.get("duration") or 0.0)
    return {"session": session.id, "lines": items, "speakers": speakers,
            "duration": max([duration] + [line["end"] for line in lines]), "summary": summary_html(session, long),
            "focus": focus, "picked": picked, "filter": previous.get("filter") or "all", "keepFocus": keep_focus}


def empty_board() -> dict:
    return {"session": "", "lines": [], "speakers": [], "duration": 0.0, "summary": "", "focus": None,
            "picked": [], "filter": "all", "keepFocus": False}


def board_focus(board) -> int | None:
    focus = board.get("focus") if isinstance(board, dict) else None
    return int(focus) if isinstance(focus, (int, float)) else None


def board_picks(board) -> list[int]:
    picked = board.get("picked") if isinstance(board, dict) else None
    return sorted({int(number) for number in picked or [] if isinstance(number, (int, float))})


def line_numbers(numbers: list[int]) -> str:
    """Zero-based indices as the one-based ranges the line actions read: ``[2, 4, 5, 6]`` → ``"3, 5-7"``."""
    ranges, numbers = [], sorted(set(numbers))
    for number in numbers:
        if ranges and number == ranges[-1][1] + 1:
            ranges[-1][1] = number
        else:
            ranges.append([number, number])
    return ", ".join(str(low + 1) if low == high else f"{low + 1}-{high + 1}" for low, high in ranges)


def fit_note(line: dict) -> str:
    note = _fit_note(line)
    if pipeline.translation_issues([line]):
        return ("**Review translation:** this text may contain an explanation instead of only the translation. "
                "Edit it or translate again before dubbing.\n\n" + note)
    return note


def _fit_note(line: dict) -> str:
    """How the translation fits its line's time, from Drama 3's estimate of spoken length."""
    from h3_app.drama3.serving.planner import expected_seconds

    if not line.get("translation"):
        return "No translation yet: translate this line to dub it."
    if not line.get("dub", True):
        return "This line keeps its original voice."
    slot = max(0.05, line["end"] - line["start"])
    speed = expected_seconds(line["translation"]) / slot
    if speed > pipeline.TEMPO_WARNING[1]:
        return (f"**May sound rushed:** the translation needs about {speed:.1f}× speed to fit {slot:.1f} s. "
                "Shorter wording sounds more natural than more speed.")
    if speed < pipeline.TEMPO_WARNING[0]:
        return f"**May sound slow:** the translation fills only about {speed:.0%} of its {slot:.1f} s."
    return f"Fits its {slot:.1f} s naturally (about {speed:.1f}× speed)."


def inspector_heading(session: sessions.Session | None, focus: int | None) -> str:
    lines = session.segments if session is not None else []
    if focus is None or not 0 <= focus < len(lines):
        return '<div class="h3-dub-inspector-head"><h3>No line chosen</h3><p>Click a line or the timeline.</p></div>'
    line = lines[focus]
    return (f'<div class="h3-dub-inspector-head"><h3>Line {focus + 1} of {len(lines)}</h3>'
            f'<p>{clock(line["start"])} – {clock(line["end"])} · {line["end"] - line["start"]:.1f} s · '
            f'{escape(speaker_label(line["speaker"]))}</p></div>')


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------


def source_card(path, *, probe=None) -> str:
    """The picked source: its name, kind, length and size."""
    if not path:
        return ('<div class="h3-dub-source-card empty"><p>No source yet. Upload a video or recording, '
                "pick one from Media, or reopen an earlier session.</p></div>")
    source = Path(path)
    kind = "Video" if pipeline.is_video(source) else "Audio"
    details = [kind]
    try:
        size = source.stat().st_size
        details.append(f"{size / 2**20:.1f} MB" if size < 2**30 else f"{size / 2**30:.2f} GB")
    except OSError:
        pass
    if probe is None:
        from h3_app.media_tools import probe_audio_seconds as probe
    try:
        seconds = probe(source)
    except Exception:
        seconds = None
    if seconds:
        details.insert(1, clock(seconds).rsplit(".", 1)[0])
    return (f'<div class="h3-dub-source-card"><span class="h3-dub-source-kind" aria-hidden="true">{kind}</span>'
            f'<div><strong>{escape(source.name)}</strong><p>{" · ".join(details)}</p></div></div>')


def plan_steps(*, separate: bool, separator_model: str, reuse: bool, transcription: str, translate: bool,
               translator: str, target_language: str) -> list[tuple[str, str, str]]:
    """The analysis steps as ``(key, title, detail)``."""
    steps = []
    if separate:
        steps.append((SEPARATION, "Separate vocals",
                      separator_model + (" · reused when this file was analyzed before" if reuse else "")))
    detail = {pipeline.PIPELINE_LOCAL: "Qwen3-ASR with word timing, on this server",
              pipeline.PIPELINE_MOSS: "MOSS, telling speakers apart, on this server",
              pipeline.PIPELINE_GEMINI: "Gemini, telling speakers apart"}.get(transcription, transcription)
    steps.append((TRANSCRIPTION, "Transcribe" + (" and translate" if transcription == pipeline.PIPELINE_GEMINI
                                                 and translate else ""), detail))
    if translate:
        target = target_language or "the chosen language"
        steps.append((TRANSLATION, f"Translate into {target}",
                      "Gemini fills any line it left" if transcription == pipeline.PIPELINE_GEMINI
                      else ("Hy-MT2 on this server" if translator == pipeline.TRANSLATOR_HYMT else "Gemini")))
    return steps


def step_for_status(status: str) -> str | None:
    """Which analysis step a status line belongs to."""
    text = str(status or "").lower()
    if "separat" in text or "vocals" in text:
        return SEPARATION
    if "translat" in text and "transcrib" not in text:
        return TRANSLATION
    if any(word in text for word in ("transcrib", "speakers", "speech", "gemini", "align")):
        return TRANSCRIPTION
    return None


def plan_html(steps: list[tuple[str, str, str]], *, active: str | None = None, finished: bool = False) -> str:
    """The analysis steps; while it runs, those before ``active`` are done and ``active`` is current."""
    keys = [key for key, _title, _detail in steps]
    current = keys.index(active) if active in keys else (len(keys) if finished else -1)
    items = []
    for number, (key, title, detail) in enumerate(steps):
        state = "done" if number < current else ("current" if number == current else "")
        mark = {"done": "✓", "current": "•"}.get(state, f"{number + 1:02d}")
        items.append(f'<li class="{state}"><span class="h3-dub-plan-mark" aria-hidden="true">{mark}</span>'
                     f'<span><strong>{escape(title)}</strong><span>{escape(detail)}</span></span></li>')
    return f'<ol class="h3-dub-plan" aria-label="Analysis steps">{"".join(items)}</ol>'


def options_label(*, separate: bool, separator_model: str, reuse: bool, transcription: str, aligner: bool,
                  merge_gap: float) -> str:
    """The More options accordion's title: its settings at a glance."""
    parts = [f"Separate vocals · {separator_model.split(' (')[0]}" if separate else "No vocal separation",
             "reuse earlier results" if reuse else "redo earlier results"]
    if transcription == pipeline.PIPELINE_LOCAL:
        parts += ["word timing" if aligner else "no word timing", f"join pauses < {float(merge_gap):.2f} s"]
    return "More options: " + ", ".join(parts)


def recent_value(runtime, limit: int = 30) -> dict:
    """Analyzed sessions to reopen, most recent first."""
    found = []
    for data in sessions.recent(runtime, limit):
        lines = data.get("segments") or []
        when = time.strftime("%b %d, %H:%M", time.localtime(data.get("updated") or data.get("created") or 0))
        target = data.get("target_language") or "untranslated"
        speakers = len({line.get("speaker") for line in lines})
        found.append({"id": data["id"], "name": escape(data.get("name") or "Untitled"),
                      "detail": escape(f"{target} · {_plural(len(lines), 'line')} · {_plural(speakers, 'speaker')}"),
                      "when": escape(when)})
    return {"sessions": found, "chosen": None}


# ---------------------------------------------------------------------------
# Dub
# ---------------------------------------------------------------------------


def speakers_value(session: sessions.Session, voices: list[tuple[str, str]]) -> dict:
    """One card per speaker: its line count, description, voice and volume."""
    settings = pipeline.speaker_settings(session)
    described = session.data.get("speakers") or {}
    cards = []
    choices = list(voices)
    for number, speaker in enumerate(pipeline.session_speakers(session)):
        entry = settings.get(speaker, {"voice": pipeline.ORIGINAL_VOICE, "volume": 100.0})
        if entry["voice"] and entry["voice"] not in {key for _label, key in choices}:
            choices.append((f"{entry['voice']} (no longer saved)", entry["voice"]))
        count = sum(line["speaker"] == speaker for line in session.segments)
        detail = _plural(count, "line") + (f" · {described[speaker]}" if described.get(speaker) else "")
        label = speaker_label(speaker)
        cards.append({"k": number, "label": escape(label), "detail": escape(detail), "tone": number % TONES,
                      "initials": escape("".join(part[0] for part in label.split()[:2]).upper() or "S"),
                      "voice": escape(entry["voice"], quote=True), "volume": entry["volume"]})
    return {"speakers": cards, "voices": [[escape(label), escape(key, quote=True)] for label, key in choices],
            "changed": None}


def readiness_html(session: sessions.Session | None) -> str:
    """What the next dub will do, and the lines worth a look first."""
    if session is None or not session.segments:
        return '<div class="h3-dub-readiness empty"><p>Analyze a source first.</p></div>'
    lines = session.segments
    voiced = sum(bool(s["translation"]) and s.get("dub", True) for s in lines)
    kept = sum(not s.get("dub", True) for s in lines)
    untranslated = sum(not s["translation"] and s.get("dub", True) for s in lines)
    target = escape(session.data.get("target_language") or "the target language")
    rows = [("Lines to dub", f"{voiced} of {len(lines)}"), ("Keep their original voice", str(kept)),
            ("Untranslated (kept as is)", str(untranslated)),
            ("Speakers", str(len({s["speaker"] for s in lines})))]
    facts = "".join(f"<dt>{name}</dt><dd>{value}</dd>" for name, value in rows)
    failed = sorted(pipeline.take_problems(session))
    warning = ""
    issues = pipeline.translation_issues(lines)
    if issues:
        warning += (f'<p class="h3-dub-readiness-warning"><strong>Review translations: '
                    f'{line_numbers(issues)}.</strong> These lines may contain explanations. '
                    'Edit them or translate again before dubbing.</p>')
    if failed:
        numbers = line_numbers(failed)
        warning += (f'<p class="h3-dub-readiness-warning"><strong>{_plural(len(failed), "line")} with a failed '
                    f"take</strong><span>{'Lines' if len(failed) > 1 else 'Line'} {numbers}: generation failed "
                    "or its take failed an audio check. Use Preview on "
                    f"{'them' if len(failed) > 1 else 'it'} in Review lines, or enter {numbers} under Fresh "
                    "takes for lines, then dub again; the other lines reuse their takes.</span></p>")
    rushed = sorted(pipeline.timing_problems(session))
    if rushed:
        numbers = line_numbers(rushed)
        warning += (f'<p class="h3-dub-readiness-warning"><strong>{_plural(len(rushed), "line")} too long in '
                    f"the last dub</strong><span>{'Lines' if len(rushed) > 1 else 'Line'} {numbers} did not "
                    "fit even after Smart timing used the silence around "
                    f"{'them' if len(rushed) > 1 else 'it'} and sped up to Maximum speed-up. Shorten "
                    f"{'their translations' if len(rushed) > 1 else 'its translation'} in Review lines, or "
                    "raise Maximum speed-up.</span></p>")
    long = pipeline.long_lines(lines)
    if long:
        numbers = ", ".join(str(index + 1) for index in long[:12]) + (", …" if len(long) > 12 else "")
        warning += (f'<p class="h3-dub-readiness-warning"><strong>{_plural(len(long), "line")} may sound rushed</strong>'
                   f"<span>{'Lines' if len(long) > 1 else 'Line'} {numbers} "
                   f"{'look' if len(long) > 1 else 'looks'} too long for the time. Shorten "
                   f"{'them' if len(long) > 1 else 'it'} in Review lines, or use Natural pace.</span></p>")
    return (f'<div class="h3-dub-readiness"><p>Dubbing into <strong>{target}</strong>. Unchanged lines reuse '
            f"their saved takes.</p><dl>{facts}</dl>{warning}</div>")
