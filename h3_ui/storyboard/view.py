"""The Storyboard task: every scene is a MiniMax H3 clip, plus how it starts.

A start card until a storyboard exists; then the scene list beside one scene
editor and its takes, and the full video below. The editor only offers what the
selected scene can use: a scene that continues has no first frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr

from h3_app.storyboard import FRAMES, OVERLAP_FRAMES, REFERENCES, SCENE_SECONDS, TEXT

from ..image_library import build_image_file_input, build_image_input
from .presentation import CLICK_JS, cast_html

CONNECT_CHOICES = [
    ("Continue one into the next: one flowing sequence", "flow"),
    ("Every scene starts fresh: separate shots joined by cuts", "cut"),
]
MODE_CHOICES = [(TEXT, TEXT), (FRAMES, FRAMES), (REFERENCES, REFERENCES)]

# Components every edit refreshes; handlers return a dict keyed by these names.
RENDER_FIELDS = (
    "start_group", "board_group", "storyboard", "scene_id", "profile_summary", "status",
    "scene_list", "scene_heading", "scene_context", "start", "opening_note",
    "mode", "frames_group", "first_image", "first_frame_box", "no_first_note", "last_image",
    "refs_group", "cast", "ref_images", "ref_videos", "voices_group", "ref_audios", "tag_hint",
    "image_previews", *(f"{kind}_preview_{i}" for kind in ("video", "audio") for i in range(3)),
    "prompt", "prompt_hint", "title", "seconds", "added_seconds", "checklist", "generate",
    "overlap", "seed", "takes_per_run",
    "take_video", "takes", "use_take", "take_note", "confirm_group", "confirm_text",
    "timeline", "generate_remaining", "cast_list", "remove_cast_name", "rename_name", "versions",
)


@dataclass(frozen=True)
class StoryboardView:
    start_group: gr.Group
    new_name: gr.Textbox
    new_story: gr.Textbox
    new_connect: gr.Radio
    new_seconds: gr.Slider
    new_scene_count: gr.Slider
    rewrite_story: gr.Button
    rewrite_status: gr.Markdown
    create: gr.Button
    cancel_new: gr.Button
    board_group: gr.Column
    storyboard: gr.Dropdown
    new_storyboard: gr.Button
    profile_summary: gr.Markdown
    status: gr.Markdown
    scene_id: gr.Textbox
    scene_list: gr.HTML
    scene_heading: gr.Markdown
    scene_context: gr.Markdown
    start: gr.Radio
    opening_note: gr.Markdown
    mode: gr.Radio
    frames_group: gr.Row
    first_image: gr.Image
    first_frame_box: gr.Column
    no_first_note: gr.Markdown
    last_image: gr.Image
    refs_group: gr.Group
    cast: gr.CheckboxGroup
    ref_images: gr.File
    ref_videos: gr.File
    voices_group: gr.Group
    ref_audios: gr.File
    image_previews: gr.Gallery
    reference_players: dict
    scene_completed: gr.Textbox
    tag_hint: gr.Markdown
    prompt: gr.Textbox
    enhance: gr.Button
    prompt_hint: gr.Markdown
    title: gr.Textbox
    seconds: gr.Slider
    added_seconds: gr.Markdown
    checklist: gr.Markdown
    generate: gr.Button
    stop: gr.Button
    overlap: gr.Dropdown
    seed: gr.Number
    takes_per_run: gr.Slider
    duplicate_scene: gr.Button
    move_up: gr.Button
    move_down: gr.Button
    delete_scene: gr.Button
    take_video: gr.Video
    takes: gr.Radio
    use_take: gr.Button
    take_note: gr.Markdown
    confirm_group: gr.Group
    confirm_text: gr.Markdown
    confirm_regenerate: gr.Button
    confirm_later: gr.Button
    confirm_cancel: gr.Button
    timeline: gr.HTML
    generate_remaining: gr.Button
    play_all: gr.Button
    export: gr.Button
    include_outdated: gr.Checkbox
    full_video: gr.Video
    cast_list: gr.HTML
    cast_image: gr.Image
    cast_name: gr.Textbox
    add_cast: gr.Button
    remove_cast_name: gr.Dropdown
    remove_cast: gr.Button
    use_h3_settings: gr.Button
    rename_name: gr.Textbox
    rename: gr.Button
    duplicate: gr.Button
    confirm_delete: gr.Checkbox
    delete: gr.Button
    version_label: gr.Textbox
    save_version: gr.Button
    versions: gr.Dropdown
    branch: gr.Button
    # Shared hosted storyboard writer; Studio's per-scene enhancement uses the H3 tab.
    gpu_note: gr.Markdown
    writer: dict

    def components(self, names):
        return [self.reference_players[name] if name in self.reference_players else getattr(self, name) for name in names]


GPU_NOTE = (
    "**Scenes are generated on the GPU endpoint.** Prepare every scene here, then open the GPU "
    "endpoint's Storyboard and press **Generate all remaining scenes**."
)


def build_writer_settings() -> dict:
    """Hosted storyboard writer settings shared by Workshop and Studio."""
    from h3_app.catalog import DEFAULT_GEMINI_PROMPT_MODEL, GEMINI_PROMPT_MODELS, OPENAI_PROMPT_MODELS

    with gr.Accordion("Prompt writer", open=False):
        gr.Markdown("Rewrite / extend uses Gemini or an OpenAI-compatible service. "
                    "Leave a key empty to use the server's key.")
        backend = gr.Radio(["Gemini", "OpenAI-compatible"], value="Gemini", label="Writer service")
        with gr.Row():
            gemini_model = gr.Dropdown(list(GEMINI_PROMPT_MODELS), value=DEFAULT_GEMINI_PROMPT_MODEL,
                                       label="Gemini model")
            gemini_key = gr.Textbox(label="Gemini API key (optional)", type="password")
        with gr.Row():
            openai_model = gr.Dropdown(list(OPENAI_PROMPT_MODELS), value=OPENAI_PROMPT_MODELS[0],
                                       allow_custom_value=True, label="OpenAI-compatible model")
            openai_key = gr.Textbox(label="OpenAI-compatible API key (optional)", type="password")
            openai_url = gr.Textbox(label="Base URL (optional)", max_lines=1)
    unused = {name: gr.State(value) for name, value in (
        ("local_prompt_base_model", ""), ("local_prompt_max_tokens", 512), ("local_prompt_temperature", 0.7),
        ("local_prompt_top_p", 0.9), ("local_prompt_greedy", False), ("local_prompt_seed", 0),
    )}
    return {
        "prompt_writer_backend": backend, **unused, "gemini_prompt_model": gemini_model,
        "gemini_api_key": gemini_key, "openai_api_key": openai_key, "openai_model": openai_model,
        "openai_base_url": openai_url,
    }


def build_storyboard_view(root, *, gpu: bool = True) -> StoryboardView:
    """``gpu=False`` builds the Workshop's page: everything except generating."""
    with root:
        gr.Markdown(
            "## Storyboard\nA storyboard is a list of scenes, and each scene is one MiniMax H3 clip. A scene can "
            "**continue** the one before it, so motion and sound carry on, or **start fresh** as a new shot. "
            "At the end the scenes are joined into one video."
        )
        writer = build_writer_settings()

        with gr.Group(elem_classes=["h3-sb-start"]) as start_group:
            gr.Markdown("### Start a storyboard")
            new_name = gr.Textbox(label="Name", max_lines=1, placeholder="Rooftop chase")
            new_story = gr.Textbox(
                label="Story", lines=7,
                info="Enter a full story or short idea, then use Rewrite / extend into scenes. "
                     "Or separate your own scenes with --- on its own line; ---6--- also makes the scene "
                     "after it 6 seconds long. Blank lines stay inside a scene. You can edit everything afterwards.",
                placeholder=(
                    "---8---\n\nWide shot over neon rooftops at dusk. Mei stands at the edge as sirens close in."
                    "\n\n---6---\n\nShe sprints and leaps across the gap to the next roof.\n\n---5---\n\n"
                    "Close-up: she lands hard and catches her breath."
                ),
            )
            new_connect = gr.Radio(CONNECT_CHOICES, value="flow", label="How should the scenes connect?",
                                   info="You can change this for any scene later.")
            new_seconds = gr.Slider(SCENE_SECONDS[0], SCENE_SECONDS[-1], value=10, step=1,
                                    label="Usual scene length (seconds)",
                                    info="Used by scenes without ---N---. Rewriting picks each scene's "
                                         "length around it.")
            with gr.Row():
                new_scene_count = gr.Slider(2, 24, value=5, step=1, label="Scenes to write")
                rewrite_story = gr.Button("Rewrite / extend into scenes")
            rewrite_status = gr.Markdown()
            gr.Markdown("Rewriting follows the MiniMax H3 prompt guidelines for each scene, including "
                        "visual action, soundscape, and music.")
            gr.Markdown("Rewriting uses the **Prompt writer** settings above. "
                        "Review the rewritten scenes before creating the storyboard.")
            gr.Markdown(
                "Videos use your current **MiniMax H3** settings, kept for this storyboard so every "
                "scene matches. Prompts and pictures are saved on this server." if gpu else
                "Videos use **MiniMax H3's default settings**, kept for this storyboard so every scene "
                "matches. To use your own, open the storyboard on the GPU endpoint and choose "
                "**Storyboard settings › Use my current MiniMax H3 settings**."
            )
            with gr.Row():
                create = gr.Button("Create storyboard", variant="primary", scale=0, min_width=200)
                cancel_new = gr.Button("Cancel", scale=0, min_width=120)

        with gr.Column(visible=False, elem_classes=["h3-storyboard"]) as board_group:
            with gr.Row(equal_height=True):
                storyboard = gr.Dropdown([], label="Storyboard", value=None, scale=4, min_width=240)
                new_storyboard = gr.Button("New storyboard", scale=0, min_width=160)
            profile_summary = gr.Markdown()
            gpu_note = gr.Markdown(GPU_NOTE, visible=not gpu)
            status = gr.Markdown(elem_classes=["h3-job-action-status"])
            scene_id = gr.Textbox(visible=False)
            scene_completed = gr.Textbox(visible=False)

            with gr.Row(equal_height=False):
                with gr.Column(scale=3, min_width=260):
                    gr.Markdown("### Scenes\nClick a scene to edit it.")
                    scene_list = gr.HTML(js_on_load=CLICK_JS)

                with gr.Column(scale=6, min_width=380):
                    scene_heading = gr.Markdown("### Scene")
                    scene_context = gr.Markdown()
                    start = gr.Radio([], label="How does this scene start?")
                    opening_note = gr.Markdown(
                        "The opening scene works exactly like a MiniMax H3 clip, with every input.", visible=False)
                    mode = gr.Radio(MODE_CHOICES, value=TEXT, label="Conditioning")
                    with gr.Row(visible=False, equal_height=False) as frames_group:
                        with gr.Column(min_width=200):
                            first_image = build_image_input(label="First frame (optional)", type="filepath", height=200)
                            # The input and its media library picker, shown or hidden together.
                            first_frame_box = first_image.h3_library_container
                            no_first_note = gr.Markdown(visible=False)
                        with gr.Column(min_width=200):
                            last_image = build_image_input(label="Last frame (optional)", type="filepath", height=200)
                    with gr.Group(visible=False) as refs_group:
                        cast = gr.CheckboxGroup([], label="Cast in this scene")
                        ref_images = build_image_file_input(label="Extra images", file_count="multiple",
                                                            file_types=["image"], type="filepath")
                        ref_videos = gr.File(label="Reference videos (up to 3)", file_count="multiple",
                                             file_types=["video"], type="filepath")
                        image_previews = gr.Gallery(label="Reference pictures", columns=3, height="auto",
                                                    interactive=False, visible=False)
                        reference_players = {}
                        with gr.Row():
                            for i in range(3):
                                reference_players[f"video_preview_{i}"] = gr.Video(
                                    label=f"Reference video {i + 1}", interactive=False, visible=False)
                        tag_hint = gr.Markdown()
                    # Reference media audio, or a frames scene's FL2VA voices: <Audio 1> to <Audio 3> either way.
                    with gr.Group(visible=False) as voices_group:
                        ref_audios = gr.File(label="Reference audio (up to 3)", file_count="multiple",
                                             file_types=["audio"], type="filepath")
                        with gr.Row():
                            for i in range(3):
                                reference_players[f"audio_preview_{i}"] = gr.Audio(
                                    label=f"Reference audio {i + 1}", interactive=False, visible=False)
                    prompt = gr.Textbox(label="Prompt", lines=4,
                                        placeholder="Describe what happens, the camera and the sound.")
                    with gr.Row(equal_height=True):
                        prompt_hint = gr.Markdown()
                        enhance = gr.Button("Improve with the prompt writer", size="sm", scale=0, min_width=240)
                    title = gr.Textbox(label="Scene title", max_lines=1)
                    with gr.Row(equal_height=True):
                        seconds = gr.Slider(SCENE_SECONDS[0], SCENE_SECONDS[-1], value=10, step=1,
                                            label="Length (seconds)")
                        added_seconds = gr.Markdown()
                    checklist = gr.Markdown(elem_classes=["h3-sb-checklist"])
                    with gr.Row(visible=gpu):
                        generate = gr.Button("Generate this scene", variant="primary",
                                             elem_classes=["h3-primary-action"])
                        stop = gr.Button("Interrupt", interactive=False, scale=0, min_width=120)
                        generate.h3_stop = stop
                    with gr.Accordion("More scene options", open=False):
                        with gr.Row():
                            overlap = gr.Dropdown(
                                [(f"{frames} frames", frames) for frames in OVERLAP_FRAMES], value=39,
                                label="Frames shared with the previous scene",
                                info="Only for continuing scenes. More frames: smoother, less new footage.",
                            )
                            seed = gr.Number(value=-1, precision=0, label="Seed (-1 is random)")
                            takes_per_run = gr.Slider(1, 4, value=1, step=1, label="Takes per click")
                        with gr.Row():
                            duplicate_scene = gr.Button("Duplicate scene", size="sm")
                            move_up = gr.Button("Move earlier", size="sm")
                            move_down = gr.Button("Move later", size="sm")
                            delete_scene = gr.Button("Delete scene", variant="stop", size="sm")

                with gr.Column(scale=4, min_width=300):
                    gr.Markdown("### Takes of this scene")
                    take_video = gr.Video(label="Take in use", interactive=False)
                    takes = gr.Radio([], label="Takes", info="The take in use is the one in the full video.")
                    use_take = gr.Button("Use this take")
                    take_note = gr.Markdown()
                    with gr.Group(visible=False, elem_classes=["h3-sb-confirm"]) as confirm_group:
                        confirm_text = gr.Markdown()
                        confirm_regenerate = gr.Button("Switch and regenerate them", variant="primary", visible=gpu)
                        confirm_later = gr.Button("Switch, regenerate later")
                        confirm_cancel = gr.Button("Cancel")

            gr.Markdown("### Full video\nContinuing scenes join seamlessly; fresh starts join as cuts.")
            timeline = gr.HTML(js_on_load=CLICK_JS)
            with gr.Row():
                generate_remaining = gr.Button("Generate all remaining scenes", visible=gpu)
                play_all = gr.Button("Play full video")
                export = gr.Button("Export to Media")
            include_outdated = gr.Checkbox(value=True, label="Include scenes whose take is out of date")
            full_video = gr.Video(label="Full video", interactive=False)

            with gr.Accordion("Cast", open=False):
                gr.Markdown("Add a picture of each person. Scenes using **Reference media** can then tick them, "
                            "so they look the same in every scene.")
                cast_list = gr.HTML(cast_html(None))
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        cast_image = build_image_input(label="Picture", type="filepath", height=180)
                    with gr.Column(scale=2):
                        cast_name = gr.Textbox(label="Name", max_lines=1, placeholder="Mei")
                        add_cast = gr.Button("Add to cast")
                        remove_cast_name = gr.Dropdown([], label="Remove someone", value=None)
                        remove_cast = gr.Button("Remove", size="sm")

            with gr.Accordion("Storyboard settings", open=False):
                with gr.Group(visible=gpu):
                    use_h3_settings = gr.Button("Use my current MiniMax H3 settings")
                    gr.Markdown("Scenes made with the old settings will need generating again.")
                with gr.Row(equal_height=True):
                    rename_name = gr.Textbox(label="Storyboard name", max_lines=1, scale=3)
                    rename = gr.Button("Rename", scale=0, min_width=120)
                duplicate = gr.Button("Duplicate storyboard")
                with gr.Row(equal_height=True):
                    confirm_delete = gr.Checkbox(label="Yes, delete this storyboard and all its takes")
                    delete = gr.Button("Delete storyboard", variant="stop", scale=0, min_width=180)

            with gr.Accordion("Versions", open=False):
                gr.Markdown("Save the scene list to come back to, or branch it into a new storyboard.")
                with gr.Row(equal_height=True):
                    version_label = gr.Textbox(label="Version name", max_lines=1, scale=3)
                    save_version = gr.Button("Save version", scale=0, min_width=140)
                with gr.Row(equal_height=True):
                    versions = gr.Dropdown([], label="Saved versions", value=None, scale=3)
                    branch = gr.Button("Branch into a new storyboard", scale=0, min_width=220)

    return StoryboardView(
        start_group=start_group,
        new_name=new_name,
        new_story=new_story,
        new_connect=new_connect,
        new_seconds=new_seconds,
        new_scene_count=new_scene_count,
        rewrite_story=rewrite_story,
        rewrite_status=rewrite_status,
        create=create,
        cancel_new=cancel_new,
        board_group=board_group,
        storyboard=storyboard,
        new_storyboard=new_storyboard,
        profile_summary=profile_summary,
        status=status,
        scene_id=scene_id,
        scene_list=scene_list,
        scene_heading=scene_heading,
        scene_context=scene_context,
        start=start,
        opening_note=opening_note,
        mode=mode,
        frames_group=frames_group,
        first_image=first_image,
        first_frame_box=first_frame_box,
        no_first_note=no_first_note,
        last_image=last_image,
        refs_group=refs_group,
        cast=cast,
        ref_images=ref_images,
        ref_videos=ref_videos,
        voices_group=voices_group,
        ref_audios=ref_audios,
        image_previews=image_previews, reference_players=reference_players, scene_completed=scene_completed,
        tag_hint=tag_hint,
        prompt=prompt,
        enhance=enhance,
        prompt_hint=prompt_hint,
        title=title,
        seconds=seconds,
        added_seconds=added_seconds,
        checklist=checklist,
        generate=generate,
        stop=stop,
        overlap=overlap,
        seed=seed,
        takes_per_run=takes_per_run,
        duplicate_scene=duplicate_scene,
        move_up=move_up,
        move_down=move_down,
        delete_scene=delete_scene,
        take_video=take_video,
        takes=takes,
        use_take=use_take,
        take_note=take_note,
        confirm_group=confirm_group,
        confirm_text=confirm_text,
        confirm_regenerate=confirm_regenerate,
        confirm_later=confirm_later,
        confirm_cancel=confirm_cancel,
        timeline=timeline,
        generate_remaining=generate_remaining,
        play_all=play_all,
        export=export,
        include_outdated=include_outdated,
        full_video=full_video,
        cast_list=cast_list,
        cast_image=cast_image,
        cast_name=cast_name,
        add_cast=add_cast,
        remove_cast_name=remove_cast_name,
        remove_cast=remove_cast,
        use_h3_settings=use_h3_settings,
        rename_name=rename_name,
        rename=rename,
        duplicate=duplicate,
        confirm_delete=confirm_delete,
        delete=delete,
        version_label=version_label,
        save_version=save_version,
        versions=versions,
        branch=branch,
        gpu_note=gpu_note,
        writer=writer,
    )
