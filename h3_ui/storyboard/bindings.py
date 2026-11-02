"""Event wiring for the Storyboard task."""

from __future__ import annotations

import functools
import inspect

import gradio as gr

from h3_app.storyboard import DEFAULTED_FIELDS, PROFILE_FIELDS

from ..job_bindings import PROMPT_QUEUE, bind_gpu_action, owned_generation, owned_interrupt
from .handlers import STORY_WRITER_FIELDS, WRITER_FIELDS, StoryboardHandlers
from .view import RENDER_FIELDS, StoryboardView

QUIET = {"queue": False, "api_name": False, "show_progress": "hidden"}
# Edits run on the queue so their success and error toasts reach the page.
ACTION = {"api_name": False, "show_progress": "hidden", "concurrency_id": "h3-storyboard-edit",
          "concurrency_limit": 4}


def as_outputs(callback, fields):
    """Turn a handler's ``{field: update}`` dict into a tuple for ``fields``; missing ones are skipped."""

    @functools.wraps(callback)
    def run(*args, **kwargs):
        result = callback(*args, **kwargs)
        return tuple(result.get(field, gr.skip()) for field in fields)

    return run


def with_leading(callback, leading, fields, *, pack):
    """``callback`` fed by ``leading`` inputs then ``fields`` components, with Gradio's request.

    ``pack`` passes the ``fields`` values as one dict (the H3 profile) instead of one by one.
    """

    def run(*args):
        *values, request = args
        head, tail = values[:len(leading)], values[len(leading):]
        if pack:
            return callback(*head, dict(zip(fields, tail, strict=True)), request)
        return callback(*head, *tail, request)

    parameters = [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                  for name in (*leading, *fields)]
    parameters.append(inspect.Parameter("request", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                        annotation=gr.Request, default=None))
    run.__signature__ = inspect.Signature(parameters)
    # Gradio injects the request from annotations; without this it passes None.
    run.__annotations__ = {"request": gr.Request}
    run.__name__ = callback.__name__
    return run


def bind_storyboard(view: StoryboardView, handlers: StoryboardHandlers, h3_values: dict | None = None, *,
                    interrupt=None, load_trigger=None, gpu: bool = True):
    """``h3_values`` maps MiniMax H3 setting names to their components: the profile and prompt writer.

    Without them (the Workshop) storyboards start from the default settings and the
    view's own writer settings. ``gpu=False`` binds everything except generating.
    """
    h3_values = h3_values or {}
    if h3_values:
        missing = [name for name in PROFILE_FIELDS if name not in h3_values and name not in DEFAULTED_FIELDS]
        if missing:
            raise ValueError(f"Storyboard needs these MiniMax H3 settings: {missing}")
    fields = [name for name in PROFILE_FIELDS if name in h3_values]
    profile_inputs = [h3_values[name] for name in fields]
    current = [view.storyboard, view.scene_id]

    def on(trigger, callback, inputs, extra=(), options=ACTION, exclude=()):
        names = tuple(name for name in (*RENDER_FIELDS, *extra) if name not in exclude)
        return trigger(as_outputs(callback, names), inputs=inputs, outputs=view.components(names), **options)

    if load_trigger is not None:
        on(load_trigger, handlers.refresh, current, options=QUIET)
    on(view.storyboard.input, handlers.select_storyboard, [view.storyboard])
    on(view.create.click,
       with_leading(handlers.create, ("name", "story", "connect", "seconds"), fields, pack=True),
       [view.new_name, view.new_story, view.new_connect, view.new_seconds, *profile_inputs])
    on(view.new_storyboard.click, handlers.show_start, [], extra=("new_story", "new_name", "rewrite_status"),
       options=QUIET)
    on(view.cancel_new.click, handlers.refresh, current)
    if h3_values:
        on(view.use_h3_settings.click,
           with_leading(handlers.apply_profile, ("storyboard_id", "scene_id"), fields, pack=True),
           [*current, *profile_inputs])
    on(view.rename.click, handlers.rename, [*current, view.rename_name])
    on(view.duplicate.click, handlers.duplicate, current)
    on(view.delete.click, handlers.delete, [*current, view.confirm_delete], extra=("confirm_delete",))
    on(view.add_cast.click, handlers.add_cast, [*current, view.cast_name, view.cast_image],
       extra=("cast_name", "cast_image"))
    on(view.remove_cast.click, handlers.remove_cast, [*current, view.remove_cast_name])

    for clickable in (view.scene_list, view.timeline):
        on(clickable.click, handlers.list_action, current)
    text_fields = [view.prompt, view.title, view.seconds, view.overlap, view.seed, view.takes_per_run]
    for component in text_fields:
        trigger = component.blur if isinstance(component, (gr.Textbox, gr.Number)) else component.input
        on(trigger, handlers.save_text, [*current, *text_fields])
    on(view.start.input, handlers.set_start, [*current, view.start])
    on(view.mode.input, handlers.set_mode, [*current, view.mode])
    on(view.cast.input, handlers.set_cast, [*current, view.cast])
    # Change, not upload: the media library picker sets values too. Unchanged values do nothing,
    # and these events never write media back, so a render can't trigger itself again.
    media = ("first_image", "last_image", "ref_images", "ref_videos", "ref_audios")
    for field in media:
        component = getattr(view, field)
        on(component.change, handlers.set_media(field), [*current, component], exclude=media)
    story_writer = [view.writer.get(name) for name in STORY_WRITER_FIELDS]
    if all(component is not None for component in story_writer):
        view.rewrite_story.click(
            as_outputs(with_leading(handlers.rewrite_story, ("story", "connect", "seconds", "count"),
                                    STORY_WRITER_FIELDS, pack=False), ("new_story", "rewrite_status")),
            inputs=[view.new_story, view.new_connect, view.new_seconds, view.new_scene_count, *story_writer],
            outputs=[view.new_story, view.rewrite_status],
            api_name=False, show_progress="minimal", **PROMPT_QUEUE,
        )
    else:
        view.rewrite_story.visible = False
    writer = [(h3_values if gpu else view.writer).get(name) for name in WRITER_FIELDS]
    if all(component is not None for component in writer):
        on(view.enhance.click,
           with_leading(handlers.enhance, ("storyboard_id", "scene_id"), WRITER_FIELDS, pack=False),
           [*current, *writer], options={"api_name": False, "show_progress": "minimal", **PROMPT_QUEUE})
    else:
        view.enhance.visible = False
    on(view.duplicate_scene.click, handlers.duplicate_scene, current)
    on(view.move_up.click, handlers.move_scene(-1), current)
    on(view.move_down.click, handlers.move_scene(1), current)
    on(view.delete_scene.click, handlers.delete_scene, current)
    on(view.use_take.click, handlers.use_take, [*current, view.takes])
    on(view.confirm_later.click, handlers.use_take_later, [*current, view.takes])
    on(view.confirm_cancel.click, handlers.cancel_take, current)
    on(view.save_version.click, handlers.save_version, [*current, view.version_label], extra=("version_label",))
    on(view.branch.click, handlers.branch, [*current, view.versions])

    joined = [view.full_video, view.status]
    view.play_all.click(handlers.play_all, inputs=[view.storyboard, view.include_outdated], outputs=joined,
                        api_name=False, concurrency_id="h3-storyboard-join", concurrency_limit=1)
    exported = view.export.click(handlers.export, inputs=[view.storyboard, view.include_outdated], outputs=joined,
                                 api_name=False, concurrency_id="h3-storyboard-join", concurrency_limit=1)
    on(exported.then, handlers.refresh, current, options=QUIET)

    if not gpu:
        return
    # The GPU buttons share Interrupt; it must be attached before binding.
    view.generate_remaining.h3_stop = view.stop
    view.confirm_regenerate.h3_stop = view.stop
    progress = [view.status, view.take_video]
    runs = (
        bind_gpu_action(view.generate.click, owned_generation(handlers.generate_scene, "storyboard"),
                        inputs=current, outputs=progress, show_progress="minimal", api_name=False),
        bind_gpu_action(view.generate_remaining.click, owned_generation(handlers.generate_remaining, "storyboard"),
                        inputs=[view.storyboard], outputs=progress, show_progress="minimal", api_name=False),
        bind_gpu_action(view.confirm_regenerate.click,
                        owned_generation(handlers.switch_and_regenerate, "storyboard"),
                        inputs=[*current, view.takes], outputs=progress, show_progress="minimal", api_name=False),
    )
    for event in runs:
        on(event.then, handlers.refresh, current, options=QUIET)
    stopped = view.stop.click(owned_interrupt(interrupt, "storyboard"), outputs=view.status, **QUIET)
    on(stopped.then, handlers.refresh, current, options=QUIET)
