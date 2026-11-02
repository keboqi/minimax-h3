"""Music 3 and YuE2 song writers: captions or style tags plus lyrics sized to the song."""

from __future__ import annotations

import re
from typing import Any

from h3_app.catalog import DEFAULT_OPENAI_PROMPT_MODEL
from h3_app import song_brief
from h3_app.config import RuntimeConfig
from h3_app.llm import gemma
from . import common


def _song_label(name: str) -> str:
    # Writers decorate labels as **STYLE:**, **STYLE**: or ## Style:.
    return rf"[ \t>#*_]*{name}[ \t*_]*:[ \t*_]*"


def _song_reply(text: str, first: str) -> tuple[str, str] | None:
    """Split a writer reply into its two marked blocks; None if they are absent."""
    text = re.sub(r"(?m)^\s*```[a-zA-Z]*\s*$", "", str(text or ""))
    # The first label opens a line, so a short preamble before it is ignored.
    match = re.search(
        rf"(?is)(?:^|\n){_song_label(first)}(.*?)\s{_song_label('LYRICS')}(.*)$", text
    )
    return (match.group(1).strip(), match.group(2).strip()) if match else None


def _song_lyrics(
    original: str, candidate: str | None, *, policy: str
) -> str:
    """Code-enforced lyric lock: the writer never silently replaces user lyrics."""
    candidate = "" if candidate is None or candidate.strip().upper() == "N/A" else candidate.strip()
    if original and (policy != song_brief.POLISH_LYRICS or not candidate):
        return original
    return candidate


def _has_sung_words(lyrics: str) -> bool:
    return any(section.lines for section in song_brief.parse_lyrics(lyrics))


def _writer_lyrics(lyrics: str, instrumental: bool) -> str:
    """The lyrics Write should keep: bare section tags only while still instrumental.

    Tags without words are the form Write fills in for an instrumental, so once
    a voice is chosen they are not lyrics to keep.
    """
    return lyrics if instrumental or _has_sung_words(lyrics) else ""


_STYLE_REQUEST = """Classify the style of this music request for a song planner.
Profiles:
- verse-chorus: pop, rock, R&B, soul, jazz, metal and most songs built from verses and a hook
- rap: rap, hip-hop, trap, drill, grime
- dance: EDM, house, techno, trance and other club music built on builds and drops
- ballad: slow, restrained ballads, folk, acoustic, country, lullabies
- ambient: ambient, cinematic, soundtrack, lo-fi, meditation, drone
bpm is the tempo the request states, in any words or language, as one number; null if it states none.
Reply with only JSON: {{"genre": "<profile>", "bpm": <number or null>}}
Request:
{idea}"""


def _song_style(idea: str) -> tuple[song_brief.Genre | None, int | None]:
    """Gemma's style profile and stated tempo for ``idea``; (None, None) leaves it to the keyword rules."""
    idea = str(idea or "").strip()
    if not idea:
        return None, None
    # The writer waits on this, so give up quickly and let the keyword rules decide.
    answer = gemma.ask_json(_STYLE_REQUEST.format(idea=idea[:2000]), max_output_tokens=64, timeout=8)
    if answer is None:
        return None, None
    genre = song_brief.GENRES.get(str(answer.get("genre") or "").strip().lower())
    bpm = answer.get("bpm")
    valid = isinstance(bpm, (int, float)) and not isinstance(bpm, bool) and 40 <= bpm <= 220
    return genre, round(bpm) if valid else None


@gemma.uses_writer_key
def enhance_music3_prompt(
    prompt: str,
    model: str,
    temporary_api_key: str,
    lyrics: str,
    ref_image_1: Any,
    ref_image_2: Any,
    ref_image_3: Any,
    backend: str = "Gemini",
    openai_api_key: str = "",
    openai_model: str = DEFAULT_OPENAI_PROMPT_MODEL,
    duration: float = 120,
    vocals: str = song_brief.AUTO,
    language: str = song_brief.AUTO,
    use: str = song_brief.AUTO,
    lyrics_policy: str = song_brief.KEEP_LYRICS,
    notes: str = "",
    openai_base_url: str | None = None,
    *,
    runtime: RuntimeConfig,
) -> tuple[str, str, str]:
    original_caption = str(prompt or "").strip()
    original_lyrics = str(lyrics or "").strip()
    instrumental = song_brief.wants_instrumental(vocals, use, original_caption)
    draft_lyrics = _writer_lyrics(original_lyrics, instrumental)
    request_text = original_caption
    if not request_text and draft_lyrics:
        request_text = "Create a fitting music description for these lyrics."
    genre, bpm = _song_style(original_caption)
    brief = song_brief.build_brief(
        engine="MiniMax Music 3", max_seconds=duration, vocals=vocals,
        language=language, use=use, lyrics_policy=lyrics_policy, notes=notes,
        idea=original_caption, lyrics=draft_lyrics, genre=genre, bpm=bpm,
    )
    generated, status = common.write_prompt(
        prompt=request_text,
        model=model,
        temporary_api_key=temporary_api_key,
        target="MiniMax Music 3",
        system_path=runtime.prompt_systems["MiniMax Music 3"],
        backend=backend,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        openai_base_url=openai_base_url,
        media_values=(
            ("Music reference image 1", ref_image_1),
            ("Music reference image 2", ref_image_2),
            ("Music reference image 3", ref_image_3),
        ),
        context=f"{brief}\nExisting lyrics:\n{draft_lyrics or '(none)'}",
    )
    if status.startswith("Prompt enhancement failed:"):
        return original_caption, original_lyrics, status
    reply = _song_reply(generated, "CAPTION")
    if reply is None:
        if re.search(r"(?i)\bLYRICS[ \t*_]*:", generated):
            return (
                original_caption, original_lyrics,
                "Prompt enhancement failed: the writer reply could not be split "
                "into caption and lyrics. Try again.",
            )
        # Older or custom writer prompts may return a caption only.
        reply = (str(generated).strip(), None)
    caption, candidate = reply
    if not caption:
        return original_caption, original_lyrics, (
            "Prompt enhancement failed: the writer returned an empty music description."
        )
    new_lyrics = _song_lyrics(draft_lyrics, candidate, policy=lyrics_policy)
    if instrumental and not draft_lyrics:
        # Music 3 sings any words it is given; an instrumental gets none.
        new_lyrics = ""
    return caption, new_lyrics, status


@gemma.uses_writer_key
def enhance_yue2_prompt(
    style: str,
    model: str,
    temporary_api_key: str,
    lyrics: str,
    mode: str,
    duration: float,
    backend: str = "Gemini",
    openai_api_key: str = "",
    openai_model: str = DEFAULT_OPENAI_PROMPT_MODEL,
    vocals: str = song_brief.AUTO,
    language: str = song_brief.AUTO,
    use: str = song_brief.AUTO,
    lyrics_policy: str = song_brief.KEEP_LYRICS,
    notes: str = "",
    openai_base_url: str | None = None,
    *,
    runtime: RuntimeConfig,
) -> tuple[str, str, str]:
    original_style = str(style or "").strip()
    original_lyrics = str(lyrics or "").strip()
    instrumental = song_brief.wants_instrumental(vocals, use, original_style)
    draft_lyrics = _writer_lyrics(original_lyrics, instrumental)
    request_text = original_style
    if not request_text and draft_lyrics:
        request_text = "Create a fitting production style for these lyrics."
    genre, bpm = _song_style(original_style)
    brief = song_brief.build_brief(
        engine="YuE2", max_seconds=duration, vocals=vocals, language=language,
        use=use, lyrics_policy=lyrics_policy, notes=notes, idea=original_style,
        lyrics=draft_lyrics, mode=mode, genre=genre, bpm=bpm,
    )
    generated, status = common.write_prompt(
        prompt=request_text,
        model=model,
        temporary_api_key=temporary_api_key,
        target="YuE2",
        system_path=runtime.prompt_systems["YuE2"],
        backend=backend,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        openai_base_url=openai_base_url,
        media_values=(),
        context=f"{brief}\nExisting lyrics:\n{draft_lyrics or '(none)'}",
    )
    if status.startswith("Prompt enhancement failed:"):
        return original_style, original_lyrics, status
    reply = _song_reply(generated, "STYLE")
    if reply is None:
        return original_style, original_lyrics, (
            "Prompt enhancement failed: the writer reply could not be split "
            "into style and lyrics. Try again."
        )
    # YuE2 reads the style as tags: one comma-separated line.
    new_style = ", ".join(
        part.strip(" .")
        for part in re.split(r"[\n,]+", reply[0])
        if part.strip(" .")
    )
    if not new_style:
        return original_style, original_lyrics, (
            "Prompt enhancement failed: the writer returned an empty style."
        )
    new_lyrics = _song_lyrics(draft_lyrics, reply[1], policy=lyrics_policy)
    if not new_lyrics or (instrumental and not draft_lyrics and _has_sung_words(new_lyrics)):
        # YuE2 needs section tags even for an instrumental; follow the planned form.
        plan = song_brief.plan_song(
            duration, vocals=song_brief.INSTRUMENTAL, use=use,
            bpm=song_brief.detect_bpm(new_style),
        )
        new_lyrics = song_brief.instrumental_tags(plan)
    return new_style, new_lyrics, status
