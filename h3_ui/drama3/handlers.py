"""Soundstage event handlers, built from explicit runtime services.

The application constructs one :class:`Drama3Handlers` with its services; each
handler reads them through the accessors below on every call, so tests can patch
those on an instance.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Callable

import gradio as gr

from .batch_api import BatchApiHandlers
from .generation import GenerationHandlers
from .library_api import LibraryApiHandlers
from .sounds import SoundHandlers
from .speakers import SpeakerHandlers
from .voices import DRAMA3_VOICE_DESIGN_SETTINGS, VoiceHandlers
from .writer import WriterHandlers


@dataclass(frozen=True)
class Drama3Services:
    runtime_config: Callable[[], Any]
    stage_file: Callable[..., str]
    generation_services: Callable[[], Any]
    free_gpu: Callable[[], None]
    # Maps Gradio's cached copy of a result back to the output that has its snapshot.
    resolve_result: Callable[[Any], Any] = lambda path: path


class Drama3Handlers(GenerationHandlers, WriterHandlers, VoiceHandlers, SpeakerHandlers, SoundHandlers,
                     LibraryApiHandlers, BatchApiHandlers):
    """Every Soundstage handler, bound to one set of runtime services."""

    def __init__(self, services: Drama3Services):
        self.services = services

    def _runtime_config(self):
        return self.services.runtime_config()

    def stage_file(self, path, category, transcode_video=False, reuse=False):
        return self.services.stage_file(path, category, transcode_video, reuse)

    def _generation_services(self):
        return self.services.generation_services()

    def free_comfy_vram(self):
        """Release ComfyUI's VRAM before SGLang-Omni, VoiceDesign or Stable Audio takes the GPU."""
        self.services.free_gpu()


# The bound instance; Gradio sees each handler's signature without it.
SELF = inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)


def _drama3_signature(leading, trailing):
    """Named API parameters around the generated per-slot inputs."""
    import inspect
    from h3_ui.drama3.controls import DRAMA3_SLOT_INPUT_NAMES

    positional = inspect.Parameter.POSITIONAL_OR_KEYWORD
    parameters = [inspect.Parameter(name, positional) for name in (*leading, *DRAMA3_SLOT_INPUT_NAMES)]
    parameters += [inspect.Parameter(name, positional, **({"default": default} if default is not inspect.Parameter.empty else {}))
                   for name, default in trailing]
    return parameters


def _publish_drama3_signatures():
    import inspect
    from h3_ui.drama3.controls import (
        DRAMA3_CAST_INPUT_NAMES, DRAMA3_SOUND_SLOT_INPUT_NAMES, DRAMA3_VISIBLE_SLOTS, DRAMA3_VOICE_SLOTS,
        UPLOAD_SOUND,
    )

    empty = inspect.Parameter.empty
    sound_defaults = []
    for name in DRAMA3_SOUND_SLOT_INPUT_NAMES:
        if name.endswith("_preset"):
            sound_defaults.append((name, UPLOAD_SOUND))
        elif name.endswith("_audio"):
            sound_defaults.append((name, None))
        elif name.endswith("_duration"):
            sound_defaults.append((name, 5))
        else:
            sound_defaults.append((name, ""))
    progress = inspect.Parameter("progress", inspect.Parameter.KEYWORD_ONLY, default=gr.Progress(track_tqdm=False))
    generation = _drama3_signature(("text",), (
        ("temperature", empty), ("top_p", empty), ("top_k", empty), ("max_frames", empty),
        ("parallel", 16), ("retries", 0), ("script_mode", "Basic"),
        *sound_defaults, ("title", ""),
        # Cast names, so scripts can name their speakers ("Mia: Hello.") instead of tagging them.
        *((f"speaker_{speaker}_name", "") for speaker in range(DRAMA3_VOICE_SLOTS)),
    ))
    Drama3Handlers.generate_drama3.__signature__ = inspect.Signature([SELF, *generation, progress])
    Drama3Handlers.remix_drama3.__signature__ = inspect.Signature([SELF, *generation, progress])
    Drama3Handlers.regenerate_drama3_line.__signature__ = inspect.Signature([
        SELF,
        inspect.Parameter("line", inspect.Parameter.POSITIONAL_OR_KEYWORD), *generation, progress])
    Drama3Handlers.enhance_drama3_text.__signature__ = inspect.Signature([SELF, *_drama3_signature(
        ("text", "speaker_mode", "model", "temporary_api_key", "backend", "openai_api_key",
         "openai_model", "openai_base_url"), ()),
        *[inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default="")
          for name in DRAMA3_CAST_INPUT_NAMES],
        inspect.Parameter("shown_slots", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=DRAMA3_VISIBLE_SLOTS),
        inspect.Parameter("script_mode", inspect.Parameter.POSITIONAL_OR_KEYWORD, default="Basic"),
        inspect.Parameter("speaker_group", inspect.Parameter.POSITIONAL_OR_KEYWORD, default="Global"),
    ])


def _publish_drama3_voices_signature():
    import inspect
    from h3_ui.drama3.controls import DRAMA3_VOICE_SLOTS

    names = [*DRAMA3_VOICE_DESIGN_SETTINGS, "parallel", "shown_slots", "skip_existing",
             *[f"speaker_{speaker}_{field}" for speaker in range(DRAMA3_VOICE_SLOTS)
               for field in ("voice", "has_clip", "description", "preview_text", "language")]]
    Drama3Handlers.generate_drama3_voices.__signature__ = inspect.Signature(
        [SELF, *(inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in names)])


def _publish_drama3_sounds_signature():
    import inspect
    from h3_ui.drama3.controls import DRAMA3_SOUND_SLOTS

    names = [
        "shown_sound_slots",
        "skip_existing_sounds",
        *[
            f"sound_{index}_{field}"
            for index in range(DRAMA3_SOUND_SLOTS)
            for field in ("preset", "has_clip", "name", "description", "duration")
        ],
    ]
    Drama3Handlers.generate_drama3_sounds.__signature__ = inspect.Signature(
        [SELF, *(inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in names)]
    )


_publish_drama3_signatures()
_publish_drama3_voices_signature()
_publish_drama3_sounds_signature()
