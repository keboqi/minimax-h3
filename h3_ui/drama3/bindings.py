"""Drama 3 writer, cast, sound and generation event wiring."""

from dataclasses import dataclass, fields
import json
from typing import Callable

import gradio as gr
from h3_ui.drama3.speaker_library import bind_speaker_library, bind_voice_takes
from h3_app.drama3.voice_design import ENGINE_QWEN
from h3_app.prompt_writers.drama3 import DRAMA3_PROMPT_FAILED, DRAMA3_SCRIPT_ADVANCED
from h3_app.jobs import JobCancelled
from ..job_bindings import GPU_QUEUE, bind_gpu_action, bind_prompt_action, owned_generation, owned_interrupt
from h3_ui.engines.controls import (
    QUIET,
    bind_generation,
    owned_view_generation,
    writer_inputs,
)
from h3_ui.drama3.controls import (
    DRAMA3_SOUND_SLOTS,
    UPLOAD_SOUND,
    UPLOAD_VOICE,
    _SOUND_KIND,
    _VOICE_KIND,
    _clamp_count,
    _panel_title_js,
    _slot_count_js,
    _slot_title_js,
    drama3_sound_choices,
    drama3_voice_choices,
)


@dataclass(frozen=True)
class Drama3Actions:
    """Soundstage handlers bound by :func:`bind_drama3`."""

    generate_drama3: Callable
    enhance_drama3_text: Callable
    transcribe_drama3_voice: Callable
    drama3_speaker_names: Callable[[], list[str]]
    select_drama3_speaker: Callable
    save_drama3_speaker: Callable
    delete_drama3_speaker: Callable
    interrupt: Callable
    generate_drama3_voice: Callable
    generate_drama3_voices: Callable
    copy_drama3_writer_prompt: Callable
    apply_drama3_writer_reply: Callable
    speaker_library_actions: object | None = None
    generate_drama3_voice_takes: Callable | None = None
    drama3_speaker_groups: Callable = lambda: ["Global"]
    drama3_speaker_group_size: Callable | None = None
    create_drama3_speaker_group: Callable | None = None
    match_drama3_speaker: Callable | None = None
    match_drama3_speakers: Callable | None = None
    save_all_drama3_speakers: Callable | None = None
    drama3_sound_names: Callable[[], list[str]] = lambda: []
    select_drama3_sound: Callable | None = None
    save_drama3_sound: Callable | None = None
    delete_drama3_sound: Callable | None = None
    generate_drama3_sound: Callable | None = None
    generate_drama3_sounds: Callable | None = None
    sync_drama3_script_sounds: Callable | None = None
    switch_drama3_script_mode: Callable | None = None
    regenerate_drama3_line: Callable | None = None
    remix_drama3: Callable | None = None
    repair_drama3_words: Callable | None = None
    prepare_drama3_cast: Callable | None = None
    drama3_takes: Callable | None = None
    drama3_take_selected: Callable | None = None
    drama3_take_audio: Callable | None = None
    drama3_subtitles: Callable | None = None
    preload_drama3_models: Callable | None = None
    drama3_library_groups: Callable | None = None
    drama3_library_speakers: Callable | None = None
    drama3_library_save_speaker: Callable | None = None
    drama3_library_speaker_clip: Callable | None = None
    drama3_library_sounds: Callable | None = None
    drama3_library_save_sound: Callable | None = None
    drama3_library_sound_clip: Callable | None = None
    drama3_result_details: Callable | None = None
    drama3_result_audio: Callable | None = None
    generate_drama3_voices_batch: Callable | None = None
    generate_drama3_sounds_batch: Callable | None = None
    generate_drama3_batch: Callable | None = None
    load_drama3_example: Callable | None = None
    load_drama3_example_zh: Callable | None = None

    @classmethod
    def of(cls, handlers, *, interrupt: Callable, speaker_library_actions: object | None = None):
        """Every action ``handlers`` defines under its field name."""
        names = {field.name for field in fields(cls)} - {"interrupt", "speaker_library_actions"}
        return cls(
            interrupt=interrupt,
            speaker_library_actions=speaker_library_actions,
            **{name: getattr(handlers, name) for name in names if hasattr(handlers, name)},
        )


def drama3_voice_inputs(view):
    return [component for slot in view.voices for component in (slot.preset, slot.audio, slot.transcript)]


def drama3_cast_inputs(view):
    return [component for slot in view.voices
            for component in (slot.save_name, slot.description, slot.preview_text)]


def drama3_sound_inputs(view):
    return [
        component
        for slot in view.sounds
        for component in (slot.preset, slot.audio, slot.save_name, slot.description, slot.duration)
    ]


def drama3_sound_meta_inputs(view):
    return [
        component
        for slot in view.sounds
        for component in (slot.save_name, slot.description, slot.duration)
    ]


def _refresh_choices_on_open(panel, presets, choices, names):
    """Show presets saved from other sessions when the panel opens; saves and deletes refresh in place."""
    def refresh():
        current = choices(names())
        # The page was served with the build-time choices; resend only when the store differs.
        values = [choice[1] if isinstance(choice, tuple) else choice for choice in current]
        if values == [choice[1] for choice in presets[0].choices]:
            return [gr.skip()] * len(presets)
        return [gr.update(choices=current)] * len(presets)

    panel.expand(refresh, outputs=presets, **QUIET)


def _bind_slot_presentation(kind, slots, shown, panel, add, remove, cleared, group=None, group_size=None):
    """Slot titles, has_clip flags, the panel title and Add/Remove, all in the browser.

    ``cleared`` pairs a slot's field getter with the values a removed slot is reset to;
    an optional ``group`` dropdown and its saved count (``group_size``) are named in the panel title.
    """
    for index, slot in enumerate(slots):
        gr.on([slot.save_name.change, slot.preset.change, slot.audio.change], fn=None,
              js=_slot_title_js(kind, index), inputs=[slot.save_name, slot.preset, slot.audio],
              outputs=[slot.group, slot.has_clip], **QUIET)
    presets = [slot.preset for slot in slots]
    flags = [slot.has_clip for slot in slots]
    grouped = [group, group_size] if group is not None and group_size is not None else []
    gr.on([*[g.change for g in grouped], shown.change, *[preset.change for preset in presets],
           *[flag.change for flag in flags]],
          fn=None, js=_panel_title_js(kind, bool(grouped)), inputs=[*grouped, shown, *presets, *flags],
          outputs=panel, **QUIET)
    groups = [slot.group for slot in slots]
    add.click(fn=None, js=_slot_count_js(kind.total, 1), inputs=shown,
              outputs=[shown, *groups, add, remove], **QUIET)
    fields = [component for fields_of, _values in cleared for slot in slots for component in fields_of(slot)]
    remove.click(fn=None, js=_slot_count_js(kind.total, -1, [values for _fields_of, values in cleared]),
                 inputs=shown, outputs=[shown, *groups, add, remove, *fields], **QUIET)


def _bind_script_mode(view, actions):
    if view.script_mode is None or actions.switch_drama3_script_mode is None:
        return
    view.script_mode.change(
        actions.switch_drama3_script_mode,
        inputs=[view.script_mode, view.text],
        outputs=[view.sounds_panel, view.status], **QUIET,
    )


_COPY_PROMPT_JS = """async (prompt) => {
  if (!prompt) return;
  try {
    await navigator.clipboard.writeText(prompt);
  } catch (_error) {
    const area = document.createElement('textarea');
    area.value = prompt;
    document.body.appendChild(area);
    area.select();
    document.execCommand('copy');
    area.remove();
  }
  const button = document.getElementById('h3-drama3-copy-prompt');
  if (button) {
    const label = 'Copy full writer prompt';
    button.textContent = 'Copied';
    setTimeout(() => { button.textContent = label; }, 1500);
  }
}"""


def _writer_slot_inputs(view):
    mode_inputs = [view.script_mode] if view.script_mode is not None else []
    return [*drama3_voice_inputs(view), *drama3_cast_inputs(view), view.shown_slots, *mode_inputs,
            *([view.speaker_group] if view.speaker_group is not None else [])]


def _workflow_input(view):
    """The Writer workflow and Guided batch size, last among the rewrite's inputs; copy and paste stay Single-pass."""
    return [
        *([view.writer_workflow] if view.writer_workflow is not None else []),
        *([view.max_batch_lines] if view.max_batch_lines is not None else []),
    ]


def _writer_outputs(view):
    return [view.text, view.enhance_status, *drama3_cast_inputs(view),
            view.shown_slots, *[slot.group for slot in view.voices],
            view.add_speaker, view.remove_speaker,
            *([slot.preset for slot in view.voices] if view.speaker_group is not None else []),
            view.title]


def _bind_writer(view, actions):
    """Rewrite, copy and paste; returns the events that may change the script."""
    writer_slots = _writer_slot_inputs(view)
    writer_outputs = _writer_outputs(view)
    # Copying never touches the rewrite's outputs, so a running rewrite keeps going.
    view.copy_prompt.click(actions.copy_drama3_writer_prompt,
                           inputs=[view.text, view.speaker_mode, *writer_slots],
                           outputs=view.writer_prompt, **QUIET,
                           ).success(fn=None, js=_COPY_PROMPT_JS, inputs=view.writer_prompt, api_name=False)
    applied = view.apply_reply.click(
        actions.apply_drama3_writer_reply,
        inputs=[view.pasted_reply, view.text, view.speaker_mode, *writer_slots],
        outputs=writer_outputs, **QUIET,
    )
    enhanced = bind_prompt_action(
        view.enhance.click, actions.enhance_drama3_text,
        inputs=[view.text, view.speaker_mode, *writer_inputs(view.writer), *writer_slots, *_workflow_input(view)],
        outputs=writer_outputs, show_progress="minimal", api_name="enhance_drama3_text",
    )
    if view.load_example is not None and actions.load_drama3_example is not None:
        example_inputs = [view.speaker_group] if view.speaker_group is not None else []
        example_outputs = [
            *writer_outputs,
            *([view.script_mode] if view.script_mode is not None else []),
            view.speaker_mode,
            *(
                [
                    *drama3_sound_meta_inputs(view),
                    view.shown_sound_slots,
                    *[slot.group for slot in view.sounds],
                    view.add_sound,
                    view.remove_sound,
                    view.sounds_panel,
                ]
                if view.sounds
                else []
            ),
        ]
        view.load_example.click(
            actions.load_drama3_example,
            inputs=example_inputs,
            outputs=example_outputs,
            **QUIET,
        )
        if view.load_example_zh is not None and actions.load_drama3_example_zh is not None:
            view.load_example_zh.click(
                actions.load_drama3_example_zh,
                inputs=example_inputs,
                outputs=example_outputs,
                **QUIET,
            )
    return enhanced, applied


def _voice_settings(view):
    return [view.voice_engine, view.temperature, view.top_p, view.top_k, view.max_frames]


def _bind_voice_batch(trigger, view, actions):
    return bind_gpu_action(
        trigger, owned_generation(actions.generate_drama3_voices, "drama3_voice"),
        inputs=[*_voice_settings(view), view.parallel, view.shown_slots, view.skip_existing,
                *[component for slot in view.voices
                  for component in (slot.preset, slot.has_clip, slot.description, slot.preview_text, slot.language)]],
        outputs=[*[component for slot in view.voices for component in (slot.audio, slot.transcript, slot.status)],
                 view.voice_status],
        show_progress="minimal", api_name=False,
    )


def _save_all_inputs(view):
    return [view.speaker_group, view.shown_slots,
            *[c for slot in view.voices for c in
              (slot.preset, slot.save_name, slot.audio, slot.transcript, slot.description)]]


def _bind_group_size(view, actions, events):
    """Keep the selected group's saved-speaker count current for the Speaker voices title."""
    if view.speaker_group_size is None or actions.drama3_speaker_group_size is None:
        return
    view.speaker_group.change(actions.drama3_speaker_group_size, inputs=view.speaker_group,
                              outputs=view.speaker_group_size, **QUIET)
    for event in events:
        event.then(actions.drama3_speaker_group_size, inputs=view.speaker_group,
                   outputs=view.speaker_group_size, **QUIET)


def _bind_voice_slots(view, actions):
    """Returns the events that save or delete speaker presets."""
    presets = [slot.preset for slot in view.voices]
    store_events = []
    _bind_slot_presentation(
        _VOICE_KIND, view.voices, view.shown_slots, view.voices_panel, view.add_speaker, view.remove_speaker,
        cleared=[(lambda slot: (slot.preset, slot.audio, slot.transcript), (UPLOAD_VOICE, None, "")),
                 (lambda slot: (slot.save_name, slot.description, slot.preview_text), ("", "", ""))],
        group=view.speaker_group, group_size=view.speaker_group_size,
    )
    _refresh_choices_on_open(view.voices_panel, presets, drama3_voice_choices, actions.drama3_speaker_names)
    if actions.create_drama3_speaker_group is not None:
        view.create_speaker_group.click(actions.create_drama3_speaker_group,
            inputs=view.new_speaker_group, outputs=[view.speaker_group, view.speaker_group_status], **QUIET)
        store_events.append(view.save_all_speakers.click(actions.save_all_drama3_speakers,
            inputs=_save_all_inputs(view), outputs=[view.speaker_group_status, *presets], **QUIET))
        view.speaker_group.change(actions.match_drama3_speakers,
            inputs=[view.speaker_group, *[c for slot in view.voices
                    for c in (slot.save_name, slot.preset, slot.audio)]], outputs=presets, **QUIET)
        for slot in view.voices:
            slot.save_name.blur(actions.match_drama3_speaker,
                inputs=[view.speaker_group, slot.save_name, slot.preset, slot.audio],
                outputs=slot.preset, **QUIET)
    voice_settings = _voice_settings(view)
    generated_all = _bind_voice_batch(view.generate_voices.click, view, actions)
    stopped_all = view.stop_voices.click(owned_interrupt(actions.interrupt, "drama3_voice"),
                                         outputs=view.voice_status, queue=False, api_name=False)
    stopped_all.then(fn=None, cancels=[generated_all], queue=False, api_name=False)
    if view.prepare_cast is not None and actions.prepare_drama3_cast is not None:
        prepared = bind_gpu_action(
            view.prepare_cast.click, owned_generation(actions.prepare_drama3_cast, "drama3_prepare"),
            inputs=[view.script_mode, *voice_settings, view.parallel, view.shown_slots, view.skip_existing,
                    *[component for slot in view.voices for component in
                      (slot.preset, slot.has_clip, slot.description, slot.preview_text, slot.language)],
                    view.shown_sound_slots, view.skip_existing_sounds,
                    *[component for slot in view.sounds for component in
                      (slot.preset, slot.has_clip, slot.save_name, slot.description, slot.duration)]],
            outputs=[*[component for slot in view.voices for component in (slot.audio, slot.transcript, slot.status)],
                     view.voice_status, *[component for slot in view.sounds for component in (slot.audio, slot.status)],
                     view.sound_status, view.prepare_status], show_progress="minimal", api_name=False)
        view.stop_prepare_cast.click(owned_interrupt(actions.interrupt, "drama3_prepare"), outputs=view.prepare_status,
                                     queue=False, api_name=False).then(fn=None, cancels=[prepared],
                                                                      queue=False, api_name=False)
    view.voice_engine.change(
        fn=None,
        js=f"(engine) => Array.from({{length: {len(view.voices)}}}, "
           f"() => ({{__type__: 'update', visible: engine === {json.dumps(ENGINE_QWEN)}}}))",
        inputs=view.voice_engine, outputs=[slot.language for slot in view.voices], **QUIET,
    )
    for slot in view.voices:
        generated = bind_gpu_action(
            slot.generate.click, owned_generation(actions.generate_drama3_voice, "drama3_voice"),
            inputs=[slot.description, slot.preview_text, slot.language, *voice_settings],
            outputs=[slot.audio, slot.transcript, slot.status], show_progress="minimal",
            api_name="generate_drama3_voice" if slot.speaker == 0 else False,
        )
        stopped = slot.stop.click(owned_interrupt(actions.interrupt, "drama3_voice"),
                                  outputs=slot.status, queue=False, api_name=False)
        stopped.then(fn=None, cancels=[generated], queue=False, api_name=False)
        # A new upload needs its own transcript.
        slot.audio.upload(fn=None, js="() => ({__type__: 'update', value: '', interactive: true})",
                          outputs=slot.transcript, **QUIET)
        bind_gpu_action(
            slot.transcribe.click, actions.transcribe_drama3_voice, inputs=[slot.audio, view.asr_model],
            outputs=[slot.transcript, slot.status], show_progress="minimal",
            api_name="transcribe_drama3_voice" if slot.speaker == 0 else False,
        )
        slot.preset.change(
            actions.select_drama3_speaker, inputs=[slot.preset],
            outputs=[slot.upload, slot.transcript, slot.save_row, slot.delete, slot.status,
                     slot.save_name, slot.description, slot.preview_text, slot.audio, slot.saved_audio],
            **QUIET,
        )
        store_events.append(slot.save.click(
            lambda name, audio, transcript, description, group, speaker=slot.speaker: actions.save_drama3_speaker(
                speaker, name, audio, transcript, len(presets), description, group),
            inputs=[slot.save_name, slot.audio, slot.transcript, slot.description, view.speaker_group],
            outputs=[slot.status, *presets], queue=False, api_name=False,
        ))
        store_events.append(slot.delete.click(
            lambda name, speaker=slot.speaker: actions.delete_drama3_speaker(speaker, name, len(presets)),
            inputs=[slot.preset], outputs=[slot.status, *presets], queue=False, api_name=False,
        ))
    return store_events


def _sync_script_sounds(trigger, view, actions):
    return trigger(
        actions.sync_drama3_script_sounds,
        inputs=[view.text, view.script_mode, view.shown_sound_slots, *drama3_sound_meta_inputs(view)],
        outputs=[*drama3_sound_meta_inputs(view), view.shown_sound_slots,
                 *[slot.group for slot in view.sounds], view.add_sound, view.remove_sound],
        **QUIET,
    )


def _bind_sound_batch(trigger, view, actions, shown):
    return bind_gpu_action(
        trigger, owned_generation(actions.generate_drama3_sounds, "drama3_sound"),
        inputs=[shown, view.skip_existing_sounds,
                *[component for slot in view.sounds
                  for component in (slot.preset, slot.has_clip, slot.save_name, slot.description,
                                    slot.duration)]],
        outputs=[*[component for slot in view.sounds for component in (slot.audio, slot.status)],
                 view.sound_status],
        show_progress="minimal", api_name=False,
    )


def _bind_sound_slots(view, actions, script_events):
    presets = [slot.preset for slot in view.sounds]
    _bind_slot_presentation(
        _SOUND_KIND, view.sounds, view.shown_sound_slots, view.sounds_panel, view.add_sound, view.remove_sound,
        cleared=[(lambda slot: (slot.preset, slot.audio, slot.save_name, slot.description, slot.duration),
                  (UPLOAD_SOUND, None, "", "", 5))],
    )
    _refresh_choices_on_open(view.sounds_panel, presets, drama3_sound_choices, actions.drama3_sound_names)
    if actions.sync_drama3_script_sounds is not None:
        for event in script_events:
            _sync_script_sounds(event.then, view, actions)
    if actions.generate_drama3_sounds is not None:
        generated_all = _bind_sound_batch(view.generate_sounds.click, view, actions, view.shown_sound_slots)
        stopped_all = view.stop_sounds.click(owned_interrupt(actions.interrupt, "drama3_sound"),
                                             outputs=view.sound_status, queue=False, api_name=False)
        stopped_all.then(fn=None, cancels=[generated_all], queue=False, api_name=False)
    for slot in view.sounds:
        if actions.generate_drama3_sound is not None:
            generated = bind_gpu_action(
                slot.generate.click, owned_generation(actions.generate_drama3_sound, "drama3_sound"),
                inputs=[slot.description, slot.duration], outputs=[slot.audio, slot.status],
                show_progress="minimal", api_name="generate_drama3_sound" if slot.index == 0 else False,
            )
            stopped = slot.stop.click(owned_interrupt(actions.interrupt, "drama3_sound"),
                                      outputs=slot.status, queue=False, api_name=False)
            stopped.then(fn=None, cancels=[generated], queue=False, api_name=False)
        if actions.select_drama3_sound is not None:
            slot.preset.change(
                actions.select_drama3_sound, inputs=[slot.preset],
                outputs=[slot.upload, slot.save_row, slot.delete, slot.status, slot.save_name,
                         slot.description, slot.duration, slot.audio], **QUIET,
            )
        if actions.save_drama3_sound is not None:
            slot.save.click(
                lambda name, audio, description, duration, index=slot.index: actions.save_drama3_sound(
                    index, name, audio, description, duration, len(presets)),
                inputs=[slot.save_name, slot.audio, slot.description, slot.duration],
                outputs=[slot.status, *presets], queue=False, api_name=False,
            )
        if actions.delete_drama3_sound is not None:
            slot.delete.click(
                lambda name, index=slot.index: actions.delete_drama3_sound(index, name, len(presets)),
                inputs=[slot.preset], outputs=[slot.status, *presets], queue=False, api_name=False,
            )


def _bind_takes(view, actions, generation_inputs):
    """Show the result's takes, and regenerate one line or remix from saved takes."""
    if view.takes is None or actions.drama3_takes is None:
        return
    view.output.change(actions.drama3_takes, inputs=view.output,
                       outputs=[view.takes, view.take_line, view.take_actions], **QUIET)
    if view.subtitles is not None and actions.drama3_subtitles is not None:
        view.output.change(actions.drama3_subtitles, inputs=view.output, outputs=view.subtitles, **QUIET)
    if view.take_audio is not None and actions.drama3_take_audio is not None:
        # Clicking a row picks its line; picking a line plays that line's take.
        view.takes.select(actions.drama3_take_selected, inputs=view.output, outputs=view.take_line, **QUIET)
        view.take_line.change(actions.drama3_take_audio, inputs=view.take_line, outputs=view.take_audio,
                              **QUIET)
    jobs = [
        bind_gpu_action(
            button.click, owned_view_generation(callback, "drama3", inputs), inputs=inputs,
            outputs=[view.output, view.status], api_name=api_name, show_progress="minimal",
        )
        for button, callback, inputs, api_name in (
            (view.regenerate_line, actions.regenerate_drama3_line, [view.take_line, *generation_inputs],
             "regenerate_drama3_line"),
            (view.remix, actions.remix_drama3, generation_inputs, "remix_drama3"),
        )
    ]
    view.stop.click(fn=None, cancels=jobs, queue=False, api_name=False)
    if view.repair_words is not None and actions.repair_drama3_words is not None:
        # Keep the result locator as state: accepting an Audio input would rename
        # its pinned copy and lose the original output's snapshot identity.
        view.output.change(lambda output: (gr.update(interactive=bool(output)), output,
                                           "Word alignment has not been checked for this result."), inputs=view.output,
                           outputs=[view.repair_words, view.alignment_result, view.alignment_status], **QUIET)
        aligned = bind_gpu_action(
            view.repair_words.click, owned_generation(actions.repair_drama3_words, "drama3_words"),
            inputs=[view.alignment_result], outputs=[view.subtitles, view.alignment_status], api_name=False,
            show_progress="minimal")
        view.stop_repair_words.click(owned_interrupt(actions.interrupt, "drama3_words"), outputs=view.alignment_status,
                                     queue=False, api_name=False).then(fn=None, cancels=[aligned],
                                                                      queue=False, api_name=False)


def _bind_library_api(actions):
    """API-only endpoints for scripted clients: saved speakers, saved sounds, result details and audio.

    Each endpoint has its own hidden trigger, so none of them runs from the page.
    """
    if actions.drama3_library_speakers is None:
        return
    with gr.Group(visible=False):
        group = gr.Textbox(label="Speaker group", value="Global")
        name = gr.Textbox(label="Name")
        clip = gr.File(label="Clip", type="filepath")
        transcript = gr.Textbox(label="Exact transcript")
        description = gr.Textbox(label="Description")
        duration = gr.Number(label="Duration", value=5, precision=0)
        result = gr.Textbox(label="Result file name")
        listing = gr.JSON(label="Listing")
        entry = gr.JSON(label="Saved entry")
        details = gr.JSON(label="Result details")
        download = gr.File(label="Saved clip")
    quiet = {"queue": False, "show_progress": "hidden"}
    for fn, inputs, outputs, api_name in (
        (actions.drama3_library_groups, [], listing, "drama3_library_groups"),
        (actions.drama3_library_speakers, [group], listing, "drama3_library_speakers"),
        (actions.drama3_library_save_speaker, [group, name, clip, transcript, description], entry,
         "drama3_library_save_speaker"),
        (actions.drama3_library_speaker_clip, [group, name], [download, entry], "drama3_library_speaker_clip"),
        (actions.drama3_library_sounds, [], listing, "drama3_library_sounds"),
        (actions.drama3_library_save_sound, [name, clip, description, duration], entry,
         "drama3_library_save_sound"),
        (actions.drama3_library_sound_clip, [name], [download, entry], "drama3_library_sound_clip"),
        (actions.drama3_result_details, [result], details, "drama3_result_details"),
        (actions.drama3_result_audio, [result], download, "drama3_result_audio"),
    ):
        if fn is not None:
            gr.Button(visible=False).click(fn, inputs=inputs, outputs=outputs, api_name=api_name, **quiet)


def _preload_callback(actions, input_names):
    """The model preload as a GPU-queue job; an interrupt ends it quietly, since a preload is optional."""
    run = owned_generation(actions.preload_drama3_models, "drama3_preload", input_names)

    def preload(*args):
        try:
            yield from run(*args)
        except JobCancelled:
            return

    preload.__signature__, preload.__annotations__ = run.__signature__, run.__annotations__
    return preload


def _bind_batch_api(actions):
    """API-only GPU endpoints for scripted clients: batch voices, sounds and renders, and model preload.

    Each is one GPU-queue job, so a script pays one model load and one queue wait for a whole
    cast, soundscape or book. Hidden triggers keep them off the page.
    """
    if actions.generate_drama3_voices_batch is None:
        return
    with gr.Group(visible=False):
        briefs = gr.Textbox(label="Voice briefs (JSON list)")
        language = gr.Textbox(label="Default preview language", value="Auto")
        sound_items = gr.Textbox(label="Sounds (JSON list)")
        engine = gr.Textbox(label="Voice engine", value=ENGINE_QWEN)
        mode = gr.Textbox(label="Script mode", value="Advanced (sound enhanced)")
        clips = gr.File(label="Clips", file_count="multiple", type="filepath")
        listing = gr.JSON(label="Listing")
        preload_status = gr.Textbox(label="Preload status")
        jobs = gr.Textbox(label="Render jobs (JSON list)")
        fix_rounds = gr.Number(label="Failed-take rounds", value=1, precision=0)
        render_status = gr.Textbox(label="Batch status")
    for fn, family, inputs, outputs, api_name in (
        (actions.generate_drama3_voices_batch, "drama3_voice", [briefs, language], [clips, listing],
         "generate_drama3_voices_batch"),
        (actions.generate_drama3_sounds_batch, "drama3_sound", [sound_items], [clips, listing],
         "generate_drama3_sounds_batch"),
        (actions.generate_drama3_batch, "drama3", [jobs, fix_rounds], [listing, render_status],
         "generate_drama3_batch"),
    ):
        if fn is not None:
            bind_gpu_action(gr.Button(visible=False).click, owned_generation(fn, family), inputs=inputs,
                            outputs=outputs, show_progress="hidden", api_name=api_name)
    if actions.preload_drama3_models is not None:
        gr.Button(visible=False).click(_preload_callback(actions, ("voice_engine", "script_mode")),
                                       inputs=[engine, mode], outputs=preload_status, show_progress="hidden",
                                       api_name="preload_drama3_models", **GPU_QUEUE)


def bind_drama3(view, actions):
    _bind_library_api(actions)
    _bind_batch_api(actions)
    _bind_script_mode(view, actions)
    script_events = _bind_writer(view, actions)
    store_events = _bind_voice_slots(view, actions)
    if view.sounds:
        _bind_sound_slots(view, actions, script_events)
    generation_inputs = [
        view.text, *drama3_voice_inputs(view), view.temperature, view.top_p, view.top_k,
        view.max_frames, view.parallel, view.retries,
    ]
    if view.script_mode is not None:
        generation_inputs.append(view.script_mode)
    if view.sounds:
        generation_inputs.extend(drama3_sound_inputs(view))
        # The title follows the full set of sound slots.
        if view.title is not None:
            generation_inputs.append(view.title)
            # Then the Cast names, which the script's "Name:" lines refer to.
            generation_inputs.extend(slot.save_name for slot in view.voices)
            if view.word_timing is not None:
                generation_inputs.append(view.word_timing)
    event =bind_generation(view, "drama3", actions.generate_drama3, actions.interrupt, generation_inputs)
    _bind_takes(view, actions, generation_inputs)
    store_events += _bind_auto(view, actions, generation_inputs)
    library_events = []
    if view.speaker_library is not None and actions.speaker_library_actions is not None:
        library_events = bind_speaker_library(view.speaker_library, view.speaker_group, view.voices,
                                              actions.speaker_library_actions, view.speaker_group_status,
                                              store_events) or []
        if actions.generate_drama3_voice_takes is not None:
            bind_voice_takes(view.speaker_library, actions.generate_drama3_voice_takes,
                             [*_voice_settings(view), view.parallel], actions.interrupt)
    _bind_group_size(view, actions, [*store_events, *library_events])
    return event


def drama3_auto_rewritten(status):
    """Stop Auto when the rewrite failed, keeping the writer's reason."""
    status = str(status or "")
    if status.startswith(DRAMA3_PROMPT_FAILED):
        raise gr.Error(f"Auto stopped: {status}")
    return "Auto: script rewritten. Generating speaker voices\u2026"


def drama3_auto_sound_slots(script_mode, shown_sound_slots):
    """Advanced generates the shown sound slots; Basic has no sounds, so none are generated."""
    if script_mode != DRAMA3_SCRIPT_ADVANCED:
        return 0, "Auto: voices done. Generating speech\u2026"
    return (_clamp_count(shown_sound_slots, DRAMA3_SOUND_SLOTS),
            "Auto: voices done. Generating sound effects\u2026")


def _bind_auto_save(voiced, view, actions):
    """When Auto save is checked, save every voiced speaker to the selected group like Save all."""
    presets = [slot.preset for slot in view.voices]

    def save(enabled, *values):
        if not enabled:
            return [gr.skip()] * (1 + len(presets))
        return actions.save_all_drama3_speakers(*values)

    return voiced.success(save, inputs=[view.auto_save_speakers, *_save_all_inputs(view)],
                          outputs=[view.speaker_group_status, *presets], **QUIET)


def _bind_preload(started, view, actions):
    """While the writer works, load Auto's models on the GPU queue, ahead of the voice step.

    A failed preload never stops Auto: the step that needs the model loads it as usual.
    """
    if view.auto_preload_status is None or actions.preload_drama3_models is None:
        return None
    sounds = bool(view.sounds) and view.script_mode is not None and actions.generate_drama3_sounds is not None
    inputs = [view.voice_engine, *([view.script_mode] if sounds else [])]
    preload = _preload_callback(actions, ("voice_engine", "script_mode")[:len(inputs)])
    return started.then(preload, inputs=inputs, outputs=view.auto_preload_status, show_progress="hidden",
                        api_name=False, **GPU_QUEUE)


def _bind_auto(view, actions, generation_inputs):
    """Auto: rewrite the script, generate every speaker voice, every sound, then the speech.

    Each step reuses its button's callback and job family and starts only after
    the previous step succeeds; the models they need preload during the rewrite.
    Interrupt auto stops the running job and the chain.
    Returns the events that save speaker presets.
    """
    if view.auto is None:
        return []
    # Auto status, Interrupt auto, then the Auto button itself, disabled until the chain ends.
    # Its label never changes: after long jobs over a tunnel the label reset could be
    # lost while the rest of the idle update applied, leaving a clickable "running" button.
    controls = [view.auto_status, view.stop_auto, view.auto]
    idle = (gr.update(interactive=False), gr.update(interactive=True))
    started = view.auto.click(
        lambda: ("Auto: rewriting the script\u2026", gr.update(interactive=True), gr.update(interactive=False)),
        outputs=controls, **QUIET,
    )
    preloaded = _bind_preload(started, view, actions)
    rewritten = bind_prompt_action(
        started.then, actions.enhance_drama3_text,
        inputs=[view.text, view.speaker_mode, *writer_inputs(view.writer), *_writer_slot_inputs(view),
                *_workflow_input(view)],
        outputs=_writer_outputs(view), show_progress="minimal", api_name=False,
    )
    checked = rewritten.success(drama3_auto_rewritten, inputs=view.enhance_status, outputs=view.auto_status,
                                **QUIET)
    # Each step with where its status shows: Auto runs on Speak, but the writer, voice and
    # sound statuses stay on their own steps, so a failure names the one to read.
    writer, cast = "Writer status on the Script step", "The status on the Cast & sounds step"
    steps = [(rewritten, writer), (checked, writer)]
    if view.sounds and actions.sync_drama3_script_sounds is not None:
        checked = _sync_script_sounds(checked.success, view, actions)
        steps.append((checked, cast))
    voiced = _bind_voice_batch(checked.success, view, actions)
    steps.append((voiced, cast))
    store_events = []
    if view.auto_save_speakers is not None and actions.save_all_drama3_speakers is not None:
        voiced = _bind_auto_save(voiced, view, actions)
        steps.append((voiced, cast))
        store_events.append(voiced)
    if view.sounds and actions.generate_drama3_sounds is not None:
        counted = voiced.success(drama3_auto_sound_slots, inputs=[view.script_mode, view.shown_sound_slots],
                                 outputs=[view.auto_sound_slots, view.auto_status], **QUIET)
        sounded = _bind_sound_batch(counted.success, view, actions, view.auto_sound_slots)
        ready = sounded.success(lambda: "Auto: generating speech\u2026", outputs=view.auto_status, **QUIET)
        steps += [(counted, cast), (sounded, cast), (ready, cast)]
    else:
        ready = voiced.success(lambda: "Auto: voices done. Generating speech\u2026",
                               outputs=view.auto_status, **QUIET)
        steps.append((ready, cast))
    spoken = bind_gpu_action(
        ready.success, owned_view_generation(actions.generate_drama3, "drama3", generation_inputs),
        inputs=generation_inputs, outputs=[view.output, view.status], show_progress="minimal",
        api_name=False,
    )
    steps.append((spoken, "Generation progress"))
    spoken.success(lambda: ("Auto: done.", *idle), outputs=controls, **QUIET)
    for step, where in steps:
        step.failure(lambda where=where: (f"Auto stopped. {where} says why.", *idle), outputs=controls, **QUIET)

    def stop_auto(request: gr.Request):
        for family in ("drama3_preload", "drama3_voice", "drama3_sound", "drama3"):
            actions.interrupt(request, family=family)
        return "Auto interrupted.", *idle

    stopped = view.stop_auto.click(stop_auto, outputs=controls, queue=False, api_name=False)
    # Gradio can cancel queued work only; status and slot updates run immediately.
    queued = [step for step, _where in steps if step.get("queue") is not False]
    stopped.then(fn=None, cancels=queued + ([preloaded] if preloaded is not None else []),
                 queue=False, api_name=False)
    return store_events
