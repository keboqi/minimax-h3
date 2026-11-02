"""The Storyboard task: scenes generated one by one with MiniMax H3, joined into one video."""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr

from h3_app.storyboard import OVERLAP_FRAMES, SCENE_SECONDS

from ..image_library import build_image_input
from .presentation import STRIP_JS, cast_html, strip_html, timeline_html

LINK_CHOICES = [("Flow on from the previous scene", "flow"), ("Cut to a new shot", "cut")]


@dataclass(frozen=True)
class StoryboardView:
    storyboard: gr.Dropdown
    scene_id: gr.Textbox
    profile_summary: gr.Markdown
    use_h3_settings: gr.Button
    new_name: gr.Textbox
    new_story: gr.Textbox
    new_seconds: gr.Radio
    create: gr.Button
    rename: gr.Button
    duplicate: gr.Button
    confirm_delete: gr.Checkbox
    delete: gr.Button
    cast_list: gr.HTML
    cast_name: gr.Textbox
    cast_image: gr.Image
    add_cast: gr.Button
    remove_cast_name: gr.Dropdown
    remove_cast: gr.Button
    strip: gr.HTML
    scene_heading: gr.Markdown
    title: gr.Textbox
    prompt: gr.Textbox
    seconds: gr.Radio
    link: gr.Radio
    cast: gr.CheckboxGroup
    overlap: gr.Dropdown
    seed: gr.Number
    takes_per_run: gr.Slider
    generate: gr.Button
    stop: gr.Button
    duplicate_scene: gr.Button
    move_up: gr.Button
    move_down: gr.Button
    delete_scene: gr.Button
    take_video: gr.Video
    takes: gr.Radio
    use_take: gr.Button
    take_note: gr.Markdown
    timeline: gr.HTML
    play_all: gr.Button
    generate_remaining: gr.Button
    include_outdated: gr.Checkbox
    export: gr.Button
    full_video: gr.Video
    version_label: gr.Textbox
    save_version: gr.Button
    versions: gr.Dropdown
    branch: gr.Button
    status: gr.Markdown

    def render_outputs(self):
        """Components every action refreshes, in ``StoryboardHandlers.render`` order."""
        return [
            self.storyboard, self.scene_id, self.profile_summary, self.cast_list, self.remove_cast_name,
            self.strip, self.scene_heading, self.title, self.prompt, self.seconds, self.link, self.cast,
            self.overlap, self.seed, self.takes_per_run, self.takes, self.take_video, self.take_note,
            self.timeline, self.versions, self.generate_remaining, self.status,
        ]


def build_storyboard_view(root) -> StoryboardView:
    with root:
        with gr.Row(equal_height=False, elem_classes=["h3-view-heading"]):
            gr.Markdown(
                "## Storyboard\nPlan a long video as scenes, generate them one by one with MiniMax H3, "
                "and join them into one video. Flow continues from the previous scene's last frames; "
                "a cut starts a new shot that keeps the cast.",
            )
        with gr.Row(equal_height=False):
            storyboard = gr.Dropdown([], label="Storyboard", value=None, scale=3, min_width=240)
            with gr.Column(scale=5, min_width=320):
                profile_summary = gr.Markdown("Render settings come from MiniMax H3 when you create a storyboard.")
                use_h3_settings = gr.Button("Apply current MiniMax H3 settings", size="sm")
        scene_id = gr.Textbox(visible=False)
        with gr.Accordion("New storyboard", open=False):
            new_name = gr.Textbox(label="Name", max_lines=1, placeholder="Rooftop chase")
            new_story = gr.Textbox(
                label="Story (optional)", lines=4,
                info="Each paragraph becomes a draft scene. Leave empty to start with one scene.",
            )
            new_seconds = gr.Radio(list(SCENE_SECONDS), value=10, label="Usual scene length (seconds)")
            gr.Markdown(
                "Uses your current **MiniMax H3** render settings. They stay fixed for this storyboard so "
                "every scene joins cleanly. Prompts and cast pictures are saved on this server."
            )
            create = gr.Button("Create storyboard", variant="primary")
        with gr.Accordion("Storyboard and cast", open=False):
            with gr.Row():
                rename = gr.Button("Rename to the name above", size="sm")
                duplicate = gr.Button("Duplicate storyboard", size="sm")
            with gr.Row():
                confirm_delete = gr.Checkbox(label="Delete this storyboard and its takes")
                delete = gr.Button("Delete storyboard", variant="stop", size="sm")
            cast_list = gr.HTML(cast_html(None))
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    cast_image = build_image_input(label="Cast picture", type="filepath", height=180)
                with gr.Column(scale=2):
                    cast_name = gr.Textbox(label="Cast name", max_lines=1, placeholder="Mei")
                    add_cast = gr.Button("Add to cast")
                    remove_cast_name = gr.Dropdown([], label="Remove from cast", value=None)
                    remove_cast = gr.Button("Remove", size="sm")
        gr.Markdown("### Scenes")
        strip = gr.HTML(strip_html(None, None), js_on_load=STRIP_JS, elem_classes=["h3-sb-strip-root"])
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=360):
                scene_heading = gr.Markdown("### Scene")
                title = gr.Textbox(label="Title", max_lines=1)
                prompt = gr.Textbox(label="What happens in this scene", lines=4)
                with gr.Row():
                    seconds = gr.Radio(list(SCENE_SECONDS), value=5, label="Length (seconds)")
                    link = gr.Radio(LINK_CHOICES, value="flow", label="Starts from the previous scene")
                cast = gr.CheckboxGroup([], label="Cast in this scene")
                with gr.Accordion("Scene options", open=False):
                    with gr.Row():
                        overlap = gr.Dropdown(
                            [(f"{frames} frames", frames) for frames in OVERLAP_FRAMES], value=39,
                            label="Overlap with the previous scene",
                        )
                        seed = gr.Number(value=-1, precision=0, label="Seed (-1 is random)")
                        takes_per_run = gr.Slider(1, 4, value=1, step=1, label="Takes per run")
                with gr.Row():
                    generate = gr.Button("Generate scene", variant="primary", elem_classes=["h3-primary-action"])
                    stop = gr.Button("Interrupt", interactive=False)
                    generate.h3_stop = stop
                with gr.Row():
                    duplicate_scene = gr.Button("Duplicate", size="sm")
                    move_up = gr.Button("Move earlier", size="sm")
                    move_down = gr.Button("Move later", size="sm")
                    delete_scene = gr.Button("Delete scene", variant="stop", size="sm")
            with gr.Column(scale=4, min_width=320):
                gr.Markdown("### Takes")
                take_video = gr.Video(label="Take in use", interactive=False)
                takes = gr.Radio([], label="Takes of this scene")
                use_take = gr.Button("Use this take")
                take_note = gr.Markdown("Every take is kept. The one in use is what later scenes continue from.")
        gr.Markdown("### Full video")
        timeline = gr.HTML(timeline_html(None, None), js_on_load=STRIP_JS)
        with gr.Row():
            play_all = gr.Button("Play all")
            generate_remaining = gr.Button("Generate remaining")
            include_outdated = gr.Checkbox(value=True, label="Include outdated scenes")
            export = gr.Button("Export to Media")
        full_video = gr.Video(label="Full video", interactive=False)
        with gr.Accordion("Versions", open=False):
            gr.Markdown("Save a version to come back to, or branch the story into a new storyboard.")
            with gr.Row():
                version_label = gr.Textbox(label="Version name", max_lines=1, scale=3)
                save_version = gr.Button("Save version", scale=1)
            with gr.Row():
                versions = gr.Dropdown([], label="Saved versions", value=None, scale=3)
                branch = gr.Button("Branch into a new storyboard", scale=1)
        status = gr.Markdown(elem_classes=["h3-job-action-status"])
    return StoryboardView(
        storyboard=storyboard,
        scene_id=scene_id,
        profile_summary=profile_summary,
        use_h3_settings=use_h3_settings,
        new_name=new_name,
        new_story=new_story,
        new_seconds=new_seconds,
        create=create,
        rename=rename,
        duplicate=duplicate,
        confirm_delete=confirm_delete,
        delete=delete,
        cast_list=cast_list,
        cast_name=cast_name,
        cast_image=cast_image,
        add_cast=add_cast,
        remove_cast_name=remove_cast_name,
        remove_cast=remove_cast,
        strip=strip,
        scene_heading=scene_heading,
        title=title,
        prompt=prompt,
        seconds=seconds,
        link=link,
        cast=cast,
        overlap=overlap,
        seed=seed,
        takes_per_run=takes_per_run,
        generate=generate,
        stop=stop,
        duplicate_scene=duplicate_scene,
        move_up=move_up,
        move_down=move_down,
        delete_scene=delete_scene,
        take_video=take_video,
        takes=takes,
        use_take=use_take,
        take_note=take_note,
        timeline=timeline,
        play_all=play_all,
        generate_remaining=generate_remaining,
        include_outdated=include_outdated,
        export=export,
        full_video=full_video,
        version_label=version_label,
        save_version=save_version,
        versions=versions,
        branch=branch,
        status=status,
    )
