"""Drama 3 composer, voice and sound controls; event wiring lives in drama3_bindings."""

from dataclasses import dataclass
import json
import gradio as gr
from h3_ui.drama3_speaker_library import SpeakerLibraryView, build_speaker_library
from h3_app.drama3_voice_design import ENGINE_DRAMA3, VOICE_DESIGN_ENGINES
from h3_app.generation.optional_engines import DRAMA3_BACKENDS
from h3_app.prompt_service import DRAMA3_SCRIPT_BASIC, DRAMA3_SCRIPT_MODES, DRAMA3_SPEAKER_AUTO, DRAMA3_SPEAKER_MODES
from h3_app.voice_design import LANGUAGES as VOICE_DESIGN_LANGUAGES
from h3_models import DEFAULT_DRAMA3_ASR, DEFAULT_DRAMA3_MODEL, DRAMA3_ASR_CHOICES, DRAMA3_MODEL_CHOICES
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
    seed: gr.Number
    temperature: gr.Slider
    top_p: gr.Slider
    top_k: gr.Slider
    max_frames: gr.Slider
    split_long_text: gr.Checkbox
    experts: gr.Dropdown
    expert_cache_gb: gr.Slider
    backend: gr.Radio
    parallel: gr.Slider
    retries: gr.Slider
    output: gr.Audio
    run: gr.Button
    stop: gr.Button
    status: gr.Textbox
    model: gr.Dropdown
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
    speaker_group: gr.Dropdown | None = None
    new_speaker_group: gr.Textbox | None = None
    create_speaker_group: gr.Button | None = None
    save_all_speakers: gr.Button | None = None
    speaker_group_status: gr.Markdown | None = None
    speaker_library: SpeakerLibraryView | None = None


def _voice_slot(speaker, names):
    with gr.Accordion(drama3_slot_title(speaker), open=speaker == 0,
                      visible=speaker < DRAMA3_VISIBLE_SLOTS) as group:
        with gr.Row(equal_height=True):
            preset = gr.Dropdown(drama3_voice_choices(names), value=UPLOAD_VOICE,
                                 label=f"Speaker {speaker} voice · <|speaker:{speaker}|>")
            save_name = gr.Textbox(label=f"Speaker {speaker} name", placeholder="Character or role")
        description = gr.Textbox(label=f"Speaker {speaker} voice description", lines=3,
                                  placeholder="Age range, pitch, timbre, accent, articulation and pace.")
        with gr.Group() as upload:
            preview_text = gr.Textbox(label=f"Speaker {speaker} preview speech", lines=2,
                                     placeholder="Words a generated reference clip will speak.")
            language = gr.Dropdown(list(VOICE_DESIGN_LANGUAGES), value="Auto", label="Preview language",
                                   visible=False)
            with gr.Row():
                generate = gr.Button("Generate voice", size="sm")
                stop = gr.Button("Interrupt voice generation", size="sm", interactive=False)
                generate.h3_stop = stop
            audio = gr.Audio(label=f"Speaker {speaker} reference clip", type="filepath")
            transcribe = gr.Button("Auto transcribe", size="sm")
        transcript = gr.Textbox(label=f"Speaker {speaker} exact transcript", lines=2,
                                placeholder="Every word spoken in the clip.")
        with gr.Row() as save_row:
            save = gr.Button("Save speaker", size="sm")
        delete = gr.Button("Delete saved speaker", size="sm", visible=False)
        status = gr.Markdown("")
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
            info="Use ~3–6s for one-shot SFX and ~10–15s for looping ambience or background music.",
        )
        with gr.Group() as upload:
            with gr.Row():
                generate = gr.Button("Generate sound", size="sm")
                stop = gr.Button("Interrupt sound generation", size="sm", interactive=False)
                generate.h3_stop = stop
            audio = gr.Audio(label=f"Sound {index} clip", type="filepath")
        with gr.Row() as save_row:
            save = gr.Button("Save sound", size="sm")
        delete = gr.Button("Delete saved sound", size="sm", visible=False)
        status = gr.Markdown("")
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


def build_drama3_view(root, speaker_names=(), sound_names=(), speaker_groups=("Global",)):
    root.elem_classes = [*root.elem_classes, "h3-drama3"]
    with root:
        gr.Markdown("## Drama 3 speech studio\nWrite a script, choose your cast, then generate speech. "
                    "Use **Auto** to rewrite your draft and prepare voices in one run.")
        with gr.Row(equal_height=False, elem_classes=["h3-generator-shell"]):
            with gr.Column(scale=3, min_width=320, elem_classes=["h3-composer"]):
                text = gr.Textbox(label="Speech text", lines=8,
                    placeholder="<|speaker:0|> [with warm excitement] Welcome back!\n<|speaker:1|> [laughs] Took you long enough.")
                script_mode = gr.Radio(
                    list(DRAMA3_SCRIPT_MODES),
                    value=DRAMA3_SCRIPT_BASIC,
                    label="Script mode",
                    info="Basic is dialogue with expression cues. Advanced adds sound effects and music "
                         "and switches to SGLang-Omni (Linux). Switching to Basic removes sound tags.",
                )
                with gr.Row(elem_classes=["h3-generation-buttons"]):
                    auto = gr.Button("Auto: rewrite, voice, sound and speak", variant="primary", size="sm")
                    stop_auto = gr.Button("Interrupt auto", size="sm", interactive=False)
                auto_status = gr.Markdown(
                    "Paste the content and pick Basic or Advanced. Auto rewrites the script, generates every "
                    "speaker voice, generates every sound effect (Advanced only), then generates the speech, "
                    "using the writer and settings below.")
                auto_sound_slots = gr.Number(0, precision=0, visible=False)
                with gr.Accordion("Rewrite as a Drama 3 script", open=False):
                    gr.Markdown("Separates narration and dialogue, adds `<|speaker:N|>` tags and expression cues, "
                                "and fills matching voice slots with speaker names, descriptions and preview speech. "
                                "In Advanced mode, also stages `<parallel>`/`<sequential>` sound effects, atmosphere music, "
                                "and speaker vocal effects (`<phone_call>`, `<robotic_voice>`, etc.).")
                    speaker_mode = gr.Radio(
                        list(DRAMA3_SPEAKER_MODES), value=DRAMA3_SPEAKER_AUTO, label="Speakers",
                        info="Auto decides from the text. Adapt to voices uses exactly one speaker per "
                             "reference voice or saved speaker below. Single keeps one voice throughout.",
                    )
                    writer, enhance, enhance_status = _writer("Rewrite script")
                    with gr.Accordion("Use another chat model", open=False):
                        gr.Markdown("Copy the full writer prompt (system prompt, speaker slots and speech text) "
                                    "into any chat model. Paste its JSON reply here; it is checked like the "
                                    "built-in writer before the script and speaker slots change.")
                        copy_prompt = gr.Button("Show full writer prompt", size="sm")
                        writer_prompt = gr.Textbox(label="Full writer prompt", lines=8, max_lines=20,
                                                   interactive=False, visible=False, buttons=["copy"])
                        pasted_reply = gr.Textbox(label="Pasted writer reply", lines=6, max_lines=20,
                                                  placeholder='{"script": "<|speaker:0|> ...", "speakers": [...]}')
                        apply_reply = gr.Button("Apply pasted reply", size="sm")
                with gr.Accordion(drama3_voices_title(), open=False) as voices_panel:
                    gr.Markdown("Each speaker has its own panel. Upload a reference clip or generate one from "
                                "the voice description and preview speech; a generated clip uses the preview speech "
                                "as its transcript. Save either kind as a reusable speaker preset. Speakers without "
                                "clips use Drama 3's invented voices.")
                    with gr.Row():
                        speaker_group = gr.Dropdown(list(speaker_groups), value="Global", label="Speaker preset group",
                            info="Save voices to this group. Matching names use this group first, then Global.")
                        new_speaker_group = gr.Textbox(label="New speaker group", placeholder="Story or series name")
                        create_speaker_group = gr.Button("Create group", size="sm")
                    save_all_speakers = gr.Button("Save all speakers to selected group", size="sm")
                    speaker_group_status = gr.Markdown("Reuse the same group across chapters. Global voices remain available in every group.")
                    speaker_library = build_speaker_library(speaker_groups)
                    voice_engine = gr.Radio(
                        list(VOICE_DESIGN_ENGINES), value=ENGINE_DRAMA3, label="Voice generator",
                        info="Drama 3 casts the voice from the description as a delivery cue, on the selected "
                             "backend with the model and sampling settings below. "
                             "[Qwen3-TTS VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) "
                             "runs in its own isolated runtime, downloaded on first use.",
                    )
                    with gr.Row(equal_height=True):
                        skip_existing = gr.Checkbox(
                            True, label="Skip speakers with a voice",
                            info="Uncheck to regenerate clips, for example after changing the backend.")
                        asr_model = gr.Dropdown(list(DRAMA3_ASR_CHOICES), value=DEFAULT_DRAMA3_ASR,
                                                label="Voice transcript model",
                                                info="For Auto transcribe. 0.6B uses less memory. "
                                                     "Check names and every spoken word.")
                    with gr.Row():
                        generate_voices = gr.Button("Generate all speaker voices", size="sm")
                        stop_voices = gr.Button("Interrupt", size="sm", interactive=False)
                        generate_voices.h3_stop = stop_voices
                    voice_status = gr.Markdown("")
                    voices = tuple(_voice_slot(speaker, list(speaker_names))
                                   for speaker in range(DRAMA3_VOICE_SLOTS))
                    shown_slots = gr.Number(DRAMA3_VISIBLE_SLOTS, precision=0, visible=False)
                    with gr.Row():
                        add_speaker = gr.Button("Add speaker", size="sm")
                        remove_speaker = gr.Button("Remove last speaker", size="sm")
                with gr.Accordion(drama3_sounds_title(), open=False, visible=False) as sounds_panel:
                    gr.Markdown(
                        "Manage and generate `<sound name=\"...\">` clips for Advanced mode "
                        "using [Stable Audio 3 Medium](https://huggingface.co/stabilityai/stable-audio-3-medium). "
                        "Rewriting a script in Advanced mode populates these slots automatically. "
                        "Any `<sound>` without a pre-generated clip is generated automatically during speech synthesis."
                    )
                    skip_existing_sounds = gr.Checkbox(
                        True,
                        label="Skip sounds with a clip",
                        info="Uncheck to regenerate existing sound clips.",
                    )
                    with gr.Row():
                        generate_sounds = gr.Button("Generate all sounds", size="sm")
                        stop_sounds = gr.Button("Interrupt", size="sm", interactive=False)
                        generate_sounds.h3_stop = stop_sounds
                    sound_status = gr.Markdown("")
                    sounds = tuple(_sound_slot(i, list(sound_names)) for i in range(DRAMA3_SOUND_SLOTS))
                    shown_sound_slots = gr.Number(DRAMA3_VISIBLE_SOUND_SLOTS, precision=0, visible=False)
                    with gr.Row():
                        add_sound = gr.Button("Add sound", size="sm")
                        remove_sound = gr.Button("Remove last sound", size="sm")
                sampling_preset = gr.Radio(
                    list(DRAMA3_SAMPLING_PRESETS), value="Balance", label="Speech preset",
                    info="Strict keeps delivery controlled; Balance is the default; Expressive adds variation.",
                )
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 random)")
                with gr.Accordion("Speech and memory settings", open=False):
                    gr.Markdown("Models download on first use (~40 GB for FP8/INT8). With experts in RAM, "
                                "allow about 48 GB system RAM for FP8/INT8 or 40 GB for W6A8. "
                                "[Model and license](https://huggingface.co/fishaudio/drama-3)")
                    backend = gr.Radio(list(DRAMA3_BACKENDS), value=DRAMA3_BACKENDS[0], label="Backend",
                        info="ComfyUI generates the whole text in one take and can invent voices. "
                             "SGLang-Omni (Linux) is for long scripts where every speaker has a "
                             "voice below: each speaker's lines go out with that speaker's clip, in parallel, and "
                             "near-silent takes can be retried when retries are enabled. Advanced sound-enhanced mode "
                             "works exclusively with SGLang-Omni.")
                    with gr.Group(visible=False) as served_settings:
                        parallel = gr.Slider(1, 32, value=8, step=1, label="Parallel requests")
                        retries = gr.Slider(0, 4, value=0, step=1, label="Retries per chunk",
                            info="Default 0 keeps the first take. Enable to retry near-silent audio; duration warnings do not trigger retries.")
                    temperature = gr.Slider(0, 2, value=1, step=0.05, label="Temperature")
                    top_p = gr.Slider(0.05, 1, value=0.9, step=0.05, label="Top-p")
                    top_k = gr.Slider(0, 4096, value=0, step=1, label="Top-k",
                        info="0 keeps all candidates. Small caps can make speech stall at pauses.")
                    max_frames = gr.Slider(1, 8192, value=8192, step=1, label="Maximum audio frames per take",
                        info="21.5 frames per second; 8192 is about 6.3 minutes. A take stops when the "
                             "speech ends, so this only limits a take that runs on.")
                    with gr.Group() as comfy_settings:
                        model = gr.Dropdown(list(DRAMA3_MODEL_CHOICES), value=DEFAULT_DRAMA3_MODEL,
                                            label="Drama model",
                                            info="FP8 preserves the original weights. INT8 is useful with RAM experts; "
                                                 "W6A8 reduces model RAM to about 40 GB including working memory.")
                        split = gr.Checkbox(value=False, label="Split long text into connected chunks",
                            info="Off generates the whole text in one take, avoiding silent gaps at chunk joins.")
                        experts = gr.Dropdown(["in RAM", "in VRAM"], value="in VRAM", label="Expert weights",
                            info="In VRAM requires room for the whole model plus working memory.")
                        cache = gr.Slider(0, 256, value=0, step=0.5, label="Expert VRAM cache (GB, 0 automatic)")
                    backend.change(
                        fn=None,
                        js="(choice) => [{__type__: 'update', visible: choice !== 'ComfyUI'}, "
                           "{__type__: 'update', visible: choice === 'ComfyUI'}]",
                        inputs=backend, outputs=[served_settings, comfy_settings], **_QUIET,
                    )
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
            with gr.Column(scale=2, min_width=320, elem_classes=["h3-preview-panel"]):
                output = gr.Audio(label="Generated speech", type="filepath", interactive=False)
                output.h3_metadata_root = gr.Column()
                run, stop, status = _actions("Generate speech")
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
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_frames=max_frames,
        split_long_text=split,
        experts=experts,
        expert_cache_gb=cache,
        backend=backend,
        parallel=parallel,
        retries=retries,
        output=output,
        run=run,
        stop=stop,
        status=status,
        model=model,
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
        speaker_group=speaker_group,
        new_speaker_group=new_speaker_group,
        create_speaker_group=create_speaker_group,
        save_all_speakers=save_all_speakers,
        speaker_group_status=speaker_group_status,
        speaker_library=speaker_library,
    )
