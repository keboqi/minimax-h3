"""Drama 3 composer, voice and sound controls; event wiring lives in drama3_bindings."""

from dataclasses import dataclass
import json
import gradio as gr
from h3_ui.drama3_speaker_library import SpeakerLibraryView, build_speaker_library
from h3_app.drama3_voice_design import ENGINE_QWEN, VOICE_DESIGN_ENGINES
from h3_app.prompt_service import DRAMA3_SCRIPT_ADVANCED, DRAMA3_SCRIPT_MODES, DRAMA3_SPEAKER_AUTO, DRAMA3_SPEAKER_MODES
from h3_app.voice_design import LANGUAGES as VOICE_DESIGN_LANGUAGES
from h3_models import DEFAULT_DRAMA3_ASR, DRAMA3_ASR_CHOICES
from .prompt_writer_controls import RemotePromptWriterControls
from h3_ui.optional_engine_controls import (
    _QUIET,
    _actions,
    _writer,
)
from h3_ui.drama3_controls import (
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
    take_audio: gr.Audio | None = None


def _heading(step, title, detail=""):
    """Numbered section heading shared with the other generator views."""
    detail = f"<p>{detail}</p>" if detail else ""
    return gr.HTML(f'<div class="h3-panel-heading"><span class="h3-step" aria-hidden="true">{step}</span>'
                   f'<div><h3>{title}</h3>{detail}</div></div>')


def _voice_slot(speaker, names):
    with gr.Accordion(drama3_slot_title(speaker), open=speaker == 0,
                      visible=speaker < DRAMA3_VISIBLE_SLOTS, elem_classes=["h3-drama3-slot"]) as group:
        with gr.Row(equal_height=True):
            preset = gr.Dropdown(drama3_voice_choices(names), value=UPLOAD_VOICE,
                                 label=f"Speaker {speaker} voice · <|speaker:{speaker}|>")
            save_name = gr.Textbox(label=f"Speaker {speaker} name", placeholder="Character or role")
        description = gr.Textbox(label=f"Speaker {speaker} voice description", lines=2,
                                  placeholder="Age range, pitch, timbre, accent, articulation and pace.")
        with gr.Group(elem_classes=["h3-drama3-clip"]) as upload:
            with gr.Row(equal_height=False):
                with gr.Column(min_width=220):
                    preview_text = gr.Textbox(label=f"Speaker {speaker} preview speech", lines=2,
                                             placeholder="Words a generated reference clip will speak.")
                    language = gr.Dropdown(list(VOICE_DESIGN_LANGUAGES), value="Auto", label="Preview language")
                    with gr.Row(elem_classes=["h3-drama3-inline-actions"]):
                        generate = gr.Button("Generate voice", size="sm", variant="primary")
                        stop = gr.Button("Interrupt voice generation", size="sm", interactive=False)
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
    )


def _sound_slot(index, names):
    with gr.Accordion(
        drama3_sound_slot_title(index),
        open=index == 0,
        visible=index < DRAMA3_VISIBLE_SOUND_SLOTS,
        elem_classes=["h3-drama3-slot"],
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
                stop = gr.Button("Interrupt sound generation", size="sm", interactive=False)
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
    root.elem_classes = [*root.elem_classes, "h3-drama3"]
    with root:
        gr.HTML('<div class="h3-drama3-intro"><p class="h3-eyebrow">Drama 3 · expressive speech</p>'
                '<h2>Speech studio</h2><p>Write a script, cast the voices, then speak it and review it line '
                'by line. Or paste any text and let <strong>Auto</strong> do every step.</p></div>')
        # Script over Cast on the left, with the Speak stage beside both.
        with gr.Row(equal_height=False, elem_classes=["h3-generator-shell", "h3-drama3-studio"]):
            with gr.Column(scale=3, min_width=360, elem_classes=["h3-drama3-left"]):
                with gr.Column(elem_classes=["h3-composer", "h3-drama3-script-panel"]):
                    _heading("01", "Script", "Tag each line with <code>&lt;|speaker:N|&gt;</code> and add "
                             "[cues] in brackets, or let the writer do it.")
                    title = gr.Textbox(label="Title", max_lines=1, placeholder="Chapter or story title",
                                       info="Names the output file. The writer fills it from the script.")
                    text = gr.Textbox(label="Speech text", lines=18, max_lines=60, elem_classes=["h3-drama3-script"],
                        placeholder="<|speaker:0|> [with warm excitement] Welcome back!\n<|speaker:1|> [laughs] Took you long enough.")
                    with gr.Column(elem_classes=["h3-drama3-auto"]):
                        with gr.Row(equal_height=True, elem_classes=["h3-drama3-auto-bar"]):
                            gr.HTML('<div class="h3-drama3-auto-heading"><strong>Auto</strong>'
                                    '<p>Rewrites the script, generates every voice (and every sound in Advanced), '
                                    'then speaks it with the Speak settings.</p></div>')
                            with gr.Row(elem_classes=["h3-generation-buttons"]):
                                auto = gr.Button("Auto: rewrite, voice, sound and speak", size="sm",
                                                 variant="primary")
                                stop_auto = gr.Button("Interrupt auto", size="sm", interactive=False)
                        auto_save_speakers = gr.Checkbox(False, label="Save generated speakers to the selected group",
                            info="After the voices are generated, Auto saves them like Save all speakers.")
                        auto_status = gr.Markdown("Paste any text, pick a script mode, then run Auto.",
                                                  elem_classes=["h3-drama3-note"])
                        auto_sound_slots = gr.Number(0, precision=0, visible=False)
                    script_mode = gr.Radio(
                        list(DRAMA3_SCRIPT_MODES),
                        value=DRAMA3_SCRIPT_ADVANCED,
                        label="Script mode",
                        info="Basic is dialogue with expression cues. Advanced adds sound effects, music and "
                             "vocal effects; Basic keeps those tags in the script but ignores them.",
                    )
                    with gr.Accordion("Rewrite as a Drama 3 script", open=False):
                        gr.Markdown("Splits narration from dialogue, adds speaker tags and expression cues, and fills "
                                    "the voice slots in Cast. Advanced mode also stages sound effects, music and vocal "
                                    "effects such as `<phone_call>`.", elem_classes=["h3-drama3-note"])
                        speaker_mode = gr.Radio(
                            list(DRAMA3_SPEAKER_MODES), value=DRAMA3_SPEAKER_AUTO, label="Speakers",
                            info="Auto decides from the text. Adapt to voices uses one speaker per voice in Cast. "
                                 "Single keeps one voice throughout.",
                        )
                        writer, enhance, enhance_status = _writer("Rewrite script")
                        with gr.Accordion("Use another chat model", open=False):
                            gr.Markdown("Copy the full writer prompt into any chat model, then paste its JSON reply. "
                                        "It is checked like the built-in writer before anything changes.",
                                        elem_classes=["h3-drama3-note"])
                            copy_prompt = gr.Button("Show full writer prompt", size="sm")
                            writer_prompt = gr.Textbox(label="Full writer prompt", lines=8, max_lines=20,
                                                       interactive=False, visible=False, buttons=["copy"])
                            pasted_reply = gr.Textbox(label="Pasted writer reply", lines=6, max_lines=20,
                                                      placeholder='{"script": "<|speaker:0|> ...", "speakers": [...]}')
                            apply_reply = gr.Button("Apply pasted reply", size="sm", variant="primary")
                with gr.Column(elem_classes=["h3-composer", "h3-drama3-cast-panel"]):
                    _heading("02", "Cast", "Every speaker needs a voice: upload a reference clip, generate one from a "
                             "description, or reuse a saved speaker.")
                    with gr.Accordion(drama3_voices_title(group="Global", group_size=group_size),
                                      open=True) as voices_panel:
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
                        speaker_library = build_speaker_library(speaker_groups, library_open)
                        with gr.Group(elem_classes=["h3-drama3-toolbar"]):
                            voice_engine = gr.Radio(
                                list(VOICE_DESIGN_ENGINES), value=ENGINE_QWEN, label="Voice generator",
                                info="[Qwen3-TTS VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) "
                                     "designs a voice from its description (downloaded on first use). Drama 3 uses "
                                     "SGLang-Omni and the sampling settings in Speak.",
                            )
                            with gr.Row(equal_height=True):
                                skip_existing = gr.Checkbox(
                                    True, label="Skip speakers with a voice",
                                    info="Uncheck to regenerate, e.g. after changing the backend.")
                                asr_model = gr.Dropdown(list(DRAMA3_ASR_CHOICES), value=DEFAULT_DRAMA3_ASR,
                                                        label="Voice transcript model",
                                                        info="Used by Auto transcribe; 0.6B uses less memory.")
                            with gr.Row(elem_classes=["h3-drama3-inline-actions"]):
                                generate_voices = gr.Button("Generate all speaker voices", size="sm", variant="primary")
                                stop_voices = gr.Button("Interrupt", size="sm", interactive=False)
                                generate_voices.h3_stop = stop_voices
                            voice_status = gr.Markdown("", elem_classes=["h3-drama3-note"])
                        voices = tuple(_voice_slot(speaker, list(speaker_names))
                                       for speaker in range(DRAMA3_VOICE_SLOTS))
                        shown_slots = gr.Number(DRAMA3_VISIBLE_SLOTS, precision=0, visible=False)
                        with gr.Row(elem_classes=["h3-drama3-inline-actions", "h3-drama3-slot-count"]):
                            add_speaker = gr.Button("Add speaker", size="sm")
                            remove_speaker = gr.Button("Remove last speaker", size="sm")
                    with gr.Accordion(drama3_sounds_title(), open=False) as sounds_panel:
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
                        sounds = tuple(_sound_slot(i, list(sound_names)) for i in range(DRAMA3_SOUND_SLOTS))
                        shown_sound_slots = gr.Number(DRAMA3_VISIBLE_SOUND_SLOTS, precision=0, visible=False)
                        with gr.Row(elem_classes=["h3-drama3-inline-actions", "h3-drama3-slot-count"]):
                            add_sound = gr.Button("Add sound", size="sm")
                            remove_sound = gr.Button("Remove last sound", size="sm")
            with gr.Column(scale=2, min_width=320, elem_classes=["h3-preview-panel", "h3-drama3-stage"]):
                _heading("03", "Speak", "Unchanged lines reuse their saved takes.")
                output = gr.Audio(label="Generated speech", type="filepath", interactive=False)
                output.h3_metadata_root = gr.Column()
                sampling_preset = gr.Radio(
                    list(DRAMA3_SAMPLING_PRESETS), value="Expressive", label="Speech preset",
                    info="Strict keeps delivery controlled; Expressive adds variation.",
                )
                with gr.Accordion("Speech settings", open=False):
                    gr.Markdown("Drama 3 runs on SGLang-Omni (Linux, CUDA). Lines go out in parallel with their "
                                "speaker's clip. The checkpoint downloads on first use. "
                                "[Model and license](https://huggingface.co/fishaudio/drama-3)",
                                elem_classes=["h3-drama3-note"])
                    with gr.Row():
                        parallel = gr.Slider(1, 32, value=8, step=1, label="Parallel requests")
                        retries = gr.Slider(0, 4, value=0, step=1, label="Retries per chunk",
                            info="0 keeps the first take. Retries only near-silent audio.")
                    with gr.Row():
                        temperature = gr.Slider(0, 2, value=2.0, step=0.05, label="Temperature")
                        top_p = gr.Slider(0.05, 1, value=1.0, step=0.05, label="Top-p")
                    top_k = gr.Slider(0, 4096, value=30, step=1, label="Top-k",
                        info="0 keeps all candidates. Small caps can make speech stall at pauses.")
                    max_frames = gr.Slider(1, 8192, value=8192, step=1, label="Maximum audio frames per take",
                        info="21.5 frames per second; 8192 is about 6.3 minutes. Only limits a take that runs on.")
                sampling_controls = [temperature, top_p, top_k]
                presets_js = json.dumps(DRAMA3_SAMPLING_PRESETS)
                sampling_preset.input(fn=None, js=f"(name) => ({presets_js})[name]",
                                      inputs=sampling_preset, outputs=sampling_controls, **_QUIET)
                gr.on(
                    [control.change for control in sampling_controls], fn=None,
                    js=f"""(...values) => Object.entries({presets_js}).find(
                        ([, preset]) => preset.every((value, index) => value === values[index]))?.[0] ?? null""",
                    inputs=sampling_controls, outputs=sampling_preset, **_QUIET,
                )
                with gr.Column(elem_classes=["h3-action-dock"]):
                    run, stop, status = _actions("Generate speech")
        # Line review under the studio, like a story's chunk table: pick a line, hear it, redo it.
        with gr.Column(visible=False, elem_classes=["h3-composer", "h3-drama3-lines"]) as take_actions:
            _heading("04", "Lines", "Click a line to hear its take. Regenerate a line you don't like, or remix "
                     "after changing sounds or effects.")
            with gr.Row(equal_height=False):
                with gr.Column(scale=3, min_width=320):
                    takes = gr.Dataframe(headers=["#", "Speaker", "Line", "Seconds", "Take"], label="Takes",
                                         interactive=False, wrap=True, visible=False, max_height=520,
                                         column_widths=["6%", "16%", "58%", "10%", "10%"])
                with gr.Column(scale=2, min_width=280, elem_classes=["h3-drama3-line-panel"]):
                    take_line = gr.Dropdown([], label="Line", info="Generate speech reuses unchanged lines.")
                    take_audio = gr.Audio(label="Line take (before staging and effects)", type="numpy",
                                          interactive=False)
                    with gr.Row(elem_classes=["h3-generation-buttons"]):
                        regenerate_line = gr.Button("Regenerate line", size="sm", variant="primary")
                        remix = gr.Button("Remix", size="sm")
                regenerate_line.h3_stop = remix.h3_stop = stop
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
        title=title,
        take_audio=take_audio,
    )
