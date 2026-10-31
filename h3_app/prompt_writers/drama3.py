"""Soundstage script writer and director: speaker plans, reply parsing and direction passes."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from h3_app.catalog import DEFAULT_OPENAI_PROMPT_MODEL
from h3_app.config import RuntimeConfig
from h3_app.errors import H3Error
from . import common


def validate_drama3_script(text: str) -> str:
    """Reject scripts the Drama 3 prompt builder would refuse or speak wrongly."""
    from h3_app.workflows.drama3 import validate_drama3_text

    return validate_drama3_text(text, "The writer's script")


DRAMA3_SPEAKER_AUTO = "Auto"


DRAMA3_SPEAKER_ADAPT = "Adapt to voices"


DRAMA3_SPEAKER_SINGLE = "Single"


DRAMA3_SPEAKER_MODES = (DRAMA3_SPEAKER_AUTO, DRAMA3_SPEAKER_ADAPT, DRAMA3_SPEAKER_SINGLE)


DRAMA3_SCRIPT_BASIC = "Basic"


DRAMA3_SCRIPT_ADVANCED = "Advanced (sound enhanced)"


DRAMA3_SCRIPT_MODES = (DRAMA3_SCRIPT_BASIC, DRAMA3_SCRIPT_ADVANCED)

# Single-pass asks one call for the whole script; Guided writes it in short steps for smaller models.
DRAMA3_WORKFLOW_SINGLE = "Single-pass"
DRAMA3_WORKFLOW_GUIDED = "Guided (step by step)"
DRAMA3_WORKFLOWS = (DRAMA3_WORKFLOW_SINGLE, DRAMA3_WORKFLOW_GUIDED)


def drama3_voice_context(voices: Iterable[tuple[int, str]]) -> str:
    """Describe cloned-voice slots by speaker ID and saved name only."""
    lines = [
        f"- speaker_id {speaker}"
        + (f' ("{name}")' if name else "")
        + " uses a cloned reference voice; direct its delivery only."
        for speaker, name in voices
    ]
    if not lines:
        return "Voice context: no reference voices; cast every voice with a style prefix."
    return (
        "Voice context (give these speaker IDs to the matching characters; any other speaker "
        "gets an invented voice cast with a style prefix):\n" + "\n".join(lines)
    )


def drama3_speaker_plan(mode: str, voices: Iterable[tuple[int, str]]) -> tuple[tuple[int, ...] | None, str]:
    """Return the allowed speaker numbers (None for any) and the writer instruction."""
    cloned = sorted({int(speaker) for speaker, _name in voices})
    if mode == DRAMA3_SPEAKER_AUTO:
        return None, "Speaker count: decide from the text as described in your instructions."
    if mode == DRAMA3_SPEAKER_SINGLE:
        speaker = cloned[0] if cloned else 0
        return (speaker,), (
            f"Speaker count: exactly ONE speaker, speaker_id {speaker}; start every line with its name. "
            "Do not split the text between characters or add a narrator; one voice performs "
            "everything, including any quoted dialogue, changing only its delivery."
        )
    if mode == DRAMA3_SPEAKER_ADAPT:
        if not cloned:
            raise H3Error("Adapt to voices needs at least one reference voice or saved speaker; "
                          "add one or choose Auto or Single.")
        ids = ", ".join(str(speaker) for speaker in cloned)
        return tuple(cloned), (
            f"Speaker count: exactly {len(cloned)} speaker(s), one per reference voice: speaker_id {ids}. "
            "Use only these speaker IDs. Distribute the lines among them by role; when the "
            "text has more characters than voices, give the extra characters' lines to the "
            "closest-fitting voice instead of inventing new speakers."
        )
    raise H3Error(f"Unsupported Drama 3 speaker mode: {mode}")


def drama3_script_mode_instruction(script_mode: str) -> str:
    if script_mode == DRAMA3_SCRIPT_BASIC:
        return (
            "Script mode: Basic (dialogue and vocal expression only; do NOT include "
            "<parallel>, <sequential>, <sound>, <#...#>, or speaker vocal effect tags)."
        )
    if script_mode == DRAMA3_SCRIPT_ADVANCED:
        return (
            'Script mode: Advanced (sound enhanced — enrich the script with nested '
            '<parallel>/<sequential> staging, <sound [loop="true"] [volume="0.2"] name="...">...</sound> '
            "sound effects and atmosphere music, <#seconds#> pauses, and speaker vocal effect tags "
            "such as <phone_call> or <robotic_voice> where appropriate)."
        )
    raise H3Error(f"Unsupported Drama 3 script mode: {script_mode}")


_ENGLISH_WORD_RE = re.compile(r"\b(?:the|and|of|to|with|is|are|in|a|an)\b", re.IGNORECASE)


def drama3_language_instruction(text: str) -> str:
    """Name the user text's language, so a small model does not follow the language of the prompt's examples."""
    text = str(text or "")
    if re.search(r"[\u3040-\u30ff]", text):
        language = "Japanese"
    elif re.search(r"[\uac00-\ud7af]", text):
        language = "Korean"
    elif re.search(r"[\u4e00-\u9fff]", text):
        language = "Chinese"
    elif all(char.isascii() for char in text if char.isalpha()) and len(_ENGLISH_WORD_RE.findall(text)) >= 2:
        language = "English"
    else:
        return ("Language: write the title and every spoken line in the language of the user text, "
                "whatever language the instructions' examples use.")
    return (f"Language: the user text is {language}. Write the title and every spoken line in {language}, "
            "whatever language the instructions' examples use; delivery cues stay short plain English.")


def enforce_drama3_speakers(script: str, allowed: tuple[int, ...] | None) -> str:
    """Fold a single-speaker script onto its tag; reject other speakers otherwise."""
    from h3_app.drama3.serving.tree import CONTAINER_TAG_RE, PAUSE_TAG_RE, parse_sound_tag
    from h3_app.workflows.drama3 import SPEAKER_TAG, text_speakers

    if allowed is None:
        return script
    if len(allowed) == 1:
        tag = f"<|speaker:{allowed[0]}|>"
        rebuilt: list[str] = []
        for raw_line in script.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            if (
                CONTAINER_TAG_RE.fullmatch(stripped)
                or PAUSE_TAG_RE.fullmatch(stripped)
                or (stripped.lower().startswith("<sound") and parse_sound_tag(stripped) is not None)
            ):
                rebuilt.append(stripped)
                continue
            body = SPEAKER_TAG.sub("", stripped).strip()
            if body:
                rebuilt.append(f"{tag} {body}")
        return "\n".join(rebuilt)
    extra = sorted(text_speakers(script) - set(allowed))
    if extra:
        raise H3Error("The writer used speakers without a reference voice: "
                      + ", ".join(f"Speaker {speaker}" for speaker in extra) + ".")
    return script


DRAMA3_PROMPT_FAILED = "Prompt enhancement failed:"


def drama3_writer_context(
    voices: Iterable[tuple[int, str]],
    speaker_mode: str,
    cast_context: str = "",
    *,
    script_mode: str = DRAMA3_SCRIPT_BASIC,
    sound_context: str = "",
    text: str = "",
) -> tuple[tuple[int, ...] | None, str]:
    """The allowed speaker IDs and the context lines sent with the user's ``text``."""
    voices = tuple(voices)
    allowed, speaker_instruction = drama3_speaker_plan(speaker_mode, voices)
    mode_instruction = drama3_script_mode_instruction(script_mode)
    parts = [drama3_voice_context(voices), speaker_instruction, mode_instruction]
    if str(text or "").strip():
        parts.append(drama3_language_instruction(text))
    if cast_context:
        parts.append(cast_context)
    if sound_context and script_mode == DRAMA3_SCRIPT_ADVANCED:
        parts.append(sound_context)
    return allowed, "\n".join(parts)


def drama3_full_prompt(
    text: str,
    voices: Iterable[tuple[int, str]] = (),
    speaker_mode: str = DRAMA3_SPEAKER_AUTO,
    *,
    runtime: RuntimeConfig,
    cast_context: str = "",
    script_mode: str = DRAMA3_SCRIPT_BASIC,
    sound_context: str = "",
) -> str:
    """The exact system prompt and request the built-in writer sends, for use in another chat model."""
    original = str(text or "").strip()
    if not original:
        raise H3Error("Write the speech text or a scene brief first.")
    _allowed, context = drama3_writer_context(
        voices,
        speaker_mode,
        cast_context,
        script_mode=script_mode,
        sound_context=sound_context,
        text=original,
    )
    system_prompt = runtime.prompt_systems["Drama 3"].read_text(encoding="utf-8").strip()
    # Same wording as the built-in request, so pasted replies follow the same contract.
    request = (f"Create the final Drama 3 prompt from the following user input.\n"
               f"{context}\nUser text:\n{original}")
    return f"{system_prompt}\n\n----- USER REQUEST -----\n\n{request}\n"


def _strip_reply_wrapping(reply: str) -> str:
    """Accept a pasted JSON reply wrapped in a code fence or a sentence of chat."""
    reply = str(reply or "").strip()
    fenced = re.search(r"```[A-Za-z0-9_-]*\s*(.*?)\s*```", reply, flags=re.S)
    if fenced:
        reply = fenced.group(1).strip()
    if not reply.startswith("{") and "{" in reply and "}" in reply:
        reply = reply[reply.index("{"):reply.rindex("}") + 1]
    return reply


def parse_drama3_reply(
    reply: str,
    original: str,
    allowed: tuple[int, ...] | None,
    speaker_mode: str,
    *,
    require_cast: bool,
    script_mode: str = DRAMA3_SCRIPT_BASIC,
    return_sounds: bool = False,
    notes: list[str] | None = None,
    meta: dict | None = None,
    speaker_names: dict[int, str] | None = None,
    direct=None,
) -> tuple:
    """Validate a writer reply exactly as the built-in writer does; raises ``H3Error``.

    ``original`` may name its speakers ("Mia: Hello."); ``speaker_names`` maps the
    Cast's names to speaker IDs for the checks against it.

    Risky cue placements are repaired; what was repaired, and source phrases the
    script dropped, are appended to ``notes`` for the status line. The reply's
    title, if any, goes in ``meta["title"]``. ``direct(script, cast)``, when given,
    returns the validated script with more delivery direction and a note, or None.
    """
    from h3_app.drama3.cast import (
        parse_script_cast_and_sounds,
        parse_writer_source_mode,
        parse_writer_title,
        voices_without_sex,
    )
    from h3_app.drama3.serving.tree import extract_script_sounds, strip_drama3_sound_tags
    from h3_app.workflows.drama3 import names_to_tags

    original = names_to_tags(original, speaker_names)

    def review(script, source_mode=""):
        from h3_app.drama3.lint import coverage_note, lint_drama3_script

        script, messages = lint_drama3_script(script)
        if notes is not None:
            notes.extend(message[0].upper() + message[1:] + "." for message in messages)
            coverage = coverage_note(original, script, source_mode=source_mode)
            if coverage:
                notes.append(coverage)
        return script

    if require_cast or reply.lstrip().startswith("{"):
        script, cast, sounds = parse_script_cast_and_sounds(reply)
        if meta is not None:
            meta["title"] = parse_writer_title(reply)
        if script_mode == DRAMA3_SCRIPT_BASIC:
            script = strip_drama3_sound_tags(script)
            sounds = []
        if direct is not None:
            script, note = direct(script, cast)
            if note and notes is not None:
                notes.append(note)
        script = review(script, parse_writer_source_mode(reply))
        unsexed = voices_without_sex(cast)
        if unsexed and notes is not None:
            notes.append(f"The voice description for {', '.join(unsexed)} does not say male or female; "
                         "add it before generating voices, or the voice may come out wrong.")
        if speaker_mode == DRAMA3_SPEAKER_AUTO:
            from h3_app.workflows.drama3 import SPEAKER_TAG, text_speakers
            existing_ids = {int(value) for value in SPEAKER_TAG.findall(str(original or ""))}
            if not existing_ids <= text_speakers(script):
                raise H3Error("The writer removed or renumbered existing script speaker IDs.")
        if allowed is not None and {entry["speaker_id"] for entry in cast} != set(allowed):
            raise H3Error("The writer's cast does not match the selected speaker mode and reference IDs.")
        return (script, cast, sounds) if return_sounds else (script, cast)
    # Compatibility for callers with older custom, plain-script system prompts.
    reply = names_to_tags(reply, speaker_names)
    cleaned_reply = strip_drama3_sound_tags(reply) if script_mode == DRAMA3_SCRIPT_BASIC else reply
    script = review(enforce_drama3_speakers(validate_drama3_script(cleaned_reply), allowed))
    sounds = [] if script_mode == DRAMA3_SCRIPT_BASIC else extract_script_sounds(script)
    return (script, [], sounds) if return_sounds else (script, [])


def apply_drama3_reply(
    reply: str,
    text: str,
    voices: Iterable[tuple[int, str]] = (),
    speaker_mode: str = DRAMA3_SPEAKER_AUTO,
    *,
    script_mode: str = DRAMA3_SCRIPT_BASIC,
    return_sounds: bool = False,
    return_title: bool = False,
    speaker_names: dict[int, str] | None = None,
) -> tuple:
    """Validate a reply pasted from another chat model; the text is kept on failure.

    ``return_title`` appends the reply's title ("" when it has none or failed).
    """
    original = str(text or "")
    extra = (([],) if return_sounds else ()) + (("",) if return_title else ())
    if not str(reply or "").strip():
        return (original, f"{DRAMA3_PROMPT_FAILED} paste the writer's JSON reply first.", [], *extra)
    notes: list[str] = []
    meta: dict = {}
    try:
        allowed, _context = drama3_writer_context(voices, speaker_mode, script_mode=script_mode)
        parsed = parse_drama3_reply(
            _strip_reply_wrapping(reply),
            original,
            allowed,
            speaker_mode,
            require_cast=True,
            script_mode=script_mode,
            return_sounds=return_sounds,
            notes=notes,
            meta=meta,
            speaker_names=speaker_names,
        )
    except H3Error as exc:
        return (original, f"{DRAMA3_PROMPT_FAILED} {exc} Your text was kept.", [], *extra)
    status = _with_writer_notes("Applied the pasted Drama 3 script.", notes)
    title = (meta.get("title", ""),) if return_title else ()
    if return_sounds:
        script, cast, sounds = parsed
        return script, status, cast, sounds, *title
    script, cast = parsed
    return script, status, cast, *title


DRAMA3_DIRECTION_PROMPT = "Drama 3 direction"


def drama3_direction_request(script: str, cast: Iterable[dict], script_mode: str) -> str:
    """The director's request: the cast for context and the script to direct."""
    people = "\n".join(f"- {entry['name']}: {' '.join(str(entry.get('description', '')).split())[:300]}"
                       for entry in cast)
    return (f"Script mode: {script_mode}\n"
            f"Cast (the voices are fixed; direct only how each line is delivered):\n{people}\n\n"
            f"Script to direct (return every line, in order, changing nothing but inserted cues and tags):\n"
            f"{script}")


def direct_drama3_script(script: str, cast: list[dict], script_mode: str, ask) -> tuple[str, str]:
    """(script, note): an under- or shallowly directed tagged script with a director's cues merged in.

    ``ask(request_text)`` returns the director's (reply, status). A script already
    directed well enough, a failed request, or a reply that changes words keeps its lines.
    """
    from h3_app.drama3.lint import direction_share, merge_direction, needs_direction
    from h3_app.workflows.drama3 import names_to_tags, tags_to_names

    if not needs_direction(script):
        return script, ""
    names = {int(entry["speaker_id"]): entry["name"] for entry in cast}
    try:
        reply, status = ask(drama3_direction_request(tags_to_names(script, names), cast, script_mode))
    except H3Error as exc:
        return script, f"The direction pass failed ({exc}); the writer's cues were kept."
    if status.startswith(DRAMA3_PROMPT_FAILED):
        return script, f"The direction pass failed ({status[len(DRAMA3_PROMPT_FAILED):].strip()}); the writer's cues were kept."
    fenced = re.search(r"```[A-Za-z0-9_-]*\s*(.*?)\s*```", reply, flags=re.S)
    directed, taken, refused = merge_direction(script, names_to_tags(fenced.group(1) if fenced else reply, names))
    if not taken:
        return script, "The direction pass changed the script's words or lines, so the writer's cues were kept."
    share = direction_share(directed)
    refusal = f"; {refused} line(s) that changed words were kept as written" if refused else ""
    return directed, f"A direction pass added cues to {taken} line(s), {share[0]} of {share[1]} now directed{refusal}."


def _with_writer_notes(status: str, notes: list[str]) -> str:
    return f"{status} Writer check: {' '.join(notes)}" if notes else status


MAX_GUIDED_NOTES = 4


def _write_guided(text, voices, speaker_mode, script_mode, cast_context, sound_context, notes, *,
                  runtime, ask, model_label) -> tuple[str, str]:
    """The Guided writer's reply and status, or the original text and a failure status."""
    from h3_app.prompt_writers.drama3_guided import guided_prompt_paths, write_drama3_guided

    guided_notes: list[str] = []
    try:
        reply, _source_mode, calls = write_drama3_guided(
            text, ask, guided_prompt_paths(Path(runtime.prompt_systems["Drama 3"]).parent),
            voices=voices, speaker_mode=speaker_mode, advanced=script_mode == DRAMA3_SCRIPT_ADVANCED,
            cast_context=cast_context, sound_context=sound_context, notes=guided_notes)
    except H3Error as exc:
        return text, f"{DRAMA3_PROMPT_FAILED} {exc}"
    if guided_notes:
        shown = "; ".join(guided_notes[:MAX_GUIDED_NOTES])
        more = len(guided_notes) - MAX_GUIDED_NOTES
        notes.append(f"Guided steps left out: {shown}{f'; and {more} more' if more > 0 else ''}.")
    return reply, f"Enhanced Drama 3 prompt with {model_label} in {calls} Guided steps."


def enhance_drama3_text(
    text: str,
    model: str,
    temporary_api_key: str,
    voices: Iterable[tuple[int, str]] = (),
    backend: str = "Gemini",
    openai_api_key: str = "",
    openai_model: str = DEFAULT_OPENAI_PROMPT_MODEL,
    openai_base_url: str | None = None,
    speaker_mode: str = DRAMA3_SPEAKER_AUTO,
    *,
    runtime: RuntimeConfig,
    return_cast: bool = False,
    cast_context: str = "",
    script_mode: str = DRAMA3_SCRIPT_BASIC,
    sound_context: str = "",
    return_sounds: bool = False,
    return_title: bool = False,
    speaker_names: dict[int, str] | None = None,
    workflow: str = DRAMA3_WORKFLOW_SINGLE,
) -> tuple:
    """Rewrite a script and validate its cast as one result before updating the UI.

    ``return_title`` appends the writer's title ("" when it gave none or failed).
    ``workflow`` Guided writes the script in short steps for smaller models; its reply
    is validated exactly like the Single-pass writer's.
    """
    original = str(text or "")
    voices = tuple(voices)
    meta: dict = {}

    def result(script, status, cast=None, sounds=None):
        title = (meta.get("title", ""),) if return_title else ()
        if return_sounds:
            return (script, status, cast or [], sounds or [], *title)
        return (script, status, cast or [], *title) if return_cast else (script, status, *title)

    if not original.strip():
        return result(original, f"{DRAMA3_PROMPT_FAILED} write the speech text or a scene brief first.")
    try:
        allowed, context = drama3_writer_context(
            voices,
            speaker_mode,
            cast_context,
            script_mode=script_mode,
            sound_context=sound_context,
            text=original,
        )
    except H3Error as exc:
        return result(original, f"{DRAMA3_PROMPT_FAILED} {exc}")
    notes: list[str] = []
    if workflow == DRAMA3_WORKFLOW_GUIDED:
        script, status = _write_guided(
            original, voices, speaker_mode, script_mode, cast_context, sound_context, notes, runtime=runtime,
            ask=lambda system_path, request_text: common.write_prompt(
                prompt=request_text, model=model, temporary_api_key=temporary_api_key,
                target="Drama 3 guided", system_path=system_path, backend=backend,
                openai_api_key=openai_api_key, openai_model=openai_model, openai_base_url=openai_base_url,
                media_values=(), context="", request_text=request_text),
            model_label=openai_model if backend == "OpenAI-compatible" else model)
    else:
        script, status = common.write_prompt(
            prompt=original,
            model=model,
            temporary_api_key=temporary_api_key,
            target="Drama 3",
            system_path=runtime.prompt_systems["Drama 3"],
            backend=backend,
            openai_api_key=openai_api_key,
            openai_model=openai_model,
            openai_base_url=openai_base_url,
            media_values=(),
            context=context,
        )
    if status.startswith(DRAMA3_PROMPT_FAILED):
        return result(original, status)
    # The Guided writer directs every line in its own step, so it never needs the direction pass.
    direction_prompt = (None if workflow == DRAMA3_WORKFLOW_GUIDED
                        else getattr(runtime, "prompt_systems", {}).get(DRAMA3_DIRECTION_PROMPT))

    def ask_director(request_text: str) -> tuple[str, str]:
        return common.write_prompt(
            prompt=request_text, model=model, temporary_api_key=temporary_api_key,
            target="Drama 3 direction", system_path=direction_prompt, backend=backend,
            openai_api_key=openai_api_key, openai_model=openai_model, openai_base_url=openai_base_url,
            media_values=(), context="", request_text=request_text,
        )

    try:
        parsed = parse_drama3_reply(
            script,
            original,
            allowed,
            speaker_mode,
            require_cast=return_cast or return_sounds,
            script_mode=script_mode,
            return_sounds=return_sounds,
            notes=notes,
            meta=meta,
            speaker_names=speaker_names,
            # A second writer call that only adds delivery cues to an under-directed script.
            direct=(lambda script, cast: direct_drama3_script(script, cast, script_mode, ask_director))
            if direction_prompt is not None and Path(direction_prompt).is_file() else None,
        )
        status = _with_writer_notes(status.replace(" prompt ", " script ", 1), notes)
        if return_sounds:
            script, cast, sounds = parsed
            return result(script, status, cast, sounds)
        script, cast = parsed
        return result(script, status, cast)
    except H3Error as exc:
        meta.clear()  # a rejected reply's title does not apply
        return result(original, f"{DRAMA3_PROMPT_FAILED} {exc} Your text was kept; retry.")
