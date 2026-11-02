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
    "refs_group", "cast", "ref_images", "ref_videos", "ref_audios", "tag_hint",
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
    new_seconds: gr.Radio
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
    ref_audios: gr.File
    tag_hint: gr.Markdown
    prompt: gr.Textbox
    enhance: gr.Button
    prompt_hint: gr.Markdown
    title: gr.Textbox
    seconds: gr.Radio
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

    def components(self, names):
        return [getattr(self, name) for name in names]


def build_storyboard_view(root) -> StoryboardView:
    with root:
        gr.Markdown(
            "## Storyboard\nA storyboard is a list of scenes, and each scene is one MiniMax H3 clip. A scene can "
            "**continue** the one before it, so motion and sound carry on, or **start fresh** as a new shot. "
            "At the end the scenes are joined into one video."
        )

        with gr.Group(elem_classes=["h3-sb-start"]) as start_group:
            gr.Markdown("### Start a storyboard")
            new_name = gr.Textbox(label="Name", max_lines=1, placeholder="Rooftop chase")
            new_story = gr.Textbox(
                label="Story", lines=7,
                info="Each paragraph becomes one scene, and its text becomes that scene's prompt. "
                     "Leave a blank line between paragraphs. You can edit everything afterwards.",
                placeholder=(
                    "Wide shot over neon rooftops at dusk. Mei stands at the edge as sirens close in.\n\n"
                    "She sprints and leaps across the gap to the next roof.\n\n"
                    "Close-up: she lands hard and catches her breath."
                ),
            )
            new_connect = gr.Radio(CONNECT_CHOICES, value="flow", label="How should the scenes connect?",
                                   info="You can change this for any scene later.")
            new_seconds = gr.Radio(list(SCENE_SECONDS), value=10, label="Scene length (seconds)")
            gr.Markdown("Videos use your current **MiniMax H3** settings, kept for this storyboard so every "
                        "scene matches. Prompts and pictures are saved on this server.")
            with gr.Row():
                create = gr.Button("Create storyboard", variant="primary", scale=0, min_width=200)
                cancel_new = gr.Button("Cancel", scale=0, min_width=120)

        with gr.Column(visible=False) as board_group:
            with gr.Row(equal_height=True):
                storyboard = gr.Dropdown([], label="Storyboard", value=None, scale=4, min_width=240)
                new_storyboard = gr.Button("New storyboard", scale=0, min_width=160)
            profile_summary = gr.Markdown()
            status = gr.Markdown(elem_classes=["h3-job-action-status"])
            scene_id = gr.Textbox(visible=False)

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
                        with gr.Row():
                            ref_videos = gr.File(label="Reference videos (up to 3)", file_count="multiple",
                                                 file_types=["video"], type="filepath")
                            ref_audios = gr.File(label="Reference audio (up to 3)", file_count="multiple",
                                                 file_types=["audio"], type="filepath")
                        tag_hint = gr.Markdown()
                    prompt = gr.Textbox(label="Prompt", lines=4,
                                        placeholder="Describe what happens, the camera and the sound.")
                    with gr.Row(equal_height=True):
                        prompt_hint = gr.Markdown()
                        enhance = gr.Button("Improve with the prompt writer", size="sm", scale=0, min_width=240)
                    title = gr.Textbox(label="Scene title", max_lines=1)
                    with gr.Row(equal_height=True):
                        seconds = gr.Radio(list(SCENE_SECONDS), value=10, label="Length (seconds)")
                        added_seconds = gr.Markdown()
                    checklist = gr.Markdown(elem_classes=["h3-sb-checklist"])
                    with gr.Row():
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
                        confirm_regenerate = gr.Button("Switch and regenerate them", variant="primary")
                        confirm_later = gr.Button("Switch, regenerate later")
                        confirm_cancel = gr.Button("Cancel")

            gr.Markdown("### Full video\nContinuing scenes join seamlessly; fresh starts join as cuts.")
            timeline = gr.HTML(js_on_load=CLICK_JS)
            with gr.Row():
                generate_remaining = gr.Button("Generate all remaining scenes")
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
        ref_audios=ref_audios,
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
    )
