"""Event wiring for the Storyboard task."""

from __future__ import annotations

import inspect

import gradio as gr

from h3_app.storyboard import DEFAULTED_FIELDS, PROFILE_FIELDS

from ..job_bindings import bind_gpu_action, owned_generation, owned_interrupt
from .handlers import StoryboardHandlers
from .view import StoryboardView

QUIET = {"queue": False, "api_name": False, "show_progress": "hidden"}


def with_profile(callback, leading, fields):
    """``callback(*leading, profile_values, request)`` fed by the MiniMax H3 settings components."""

    def run(*args):
        *values, request = args
        profile = dict(zip(fields, values[len(leading):], strict=True))
        return callback(*values[:len(leading)], profile, request)

    parameters = [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                  for name in (*leading, *fields)]
    parameters.append(inspect.Parameter("request", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                        annotation=gr.Request, default=None))
    run.__signature__ = inspect.Signature(parameters)
    run.__name__ = callback.__name__
    return run


def bind_storyboard(view: StoryboardView, handlers: StoryboardHandlers, h3_values: dict, *, interrupt, load_trigger=None):
    """``h3_values`` maps MiniMax H3 setting names to their components, the profile's source."""
    missing = [name for name in PROFILE_FIELDS if name not in h3_values and name not in DEFAULTED_FIELDS]
    if missing:
        raise ValueError(f"Storyboard needs these MiniMax H3 settings: {missing}")
    fields = [name for name in PROFILE_FIELDS if name in h3_values]
    profile_inputs = [h3_values[name] for name in fields]
    rendered = view.render_outputs()
    current = [view.storyboard, view.scene_id]

    if load_trigger is not None:
        load_trigger(handlers.refresh, inputs=current, outputs=rendered, **QUIET)
    view.storyboard.input(handlers.select_storyboard, inputs=[view.storyboard], outputs=rendered, **QUIET)
    view.create.click(with_profile(handlers.create, ("name", "story", "seconds"), fields),
                      inputs=[view.new_name, view.new_story, view.new_seconds, *profile_inputs],
                      outputs=rendered, **QUIET)
    view.use_h3_settings.click(with_profile(handlers.apply_profile, ("storyboard_id", "scene_id"), fields),
                               inputs=[*current, *profile_inputs], outputs=rendered, **QUIET)
    view.rename.click(handlers.rename, inputs=[*current, view.new_name], outputs=rendered, **QUIET)
    view.duplicate.click(handlers.duplicate, inputs=current, outputs=rendered, **QUIET)
    view.delete.click(handlers.delete, inputs=[*current, view.confirm_delete],
                      outputs=[*rendered, view.confirm_delete], **QUIET)
    view.add_cast.click(handlers.add_cast, inputs=[*current, view.cast_name, view.cast_image],
                        outputs=[*rendered, view.cast_name, view.cast_image], **QUIET)
    view.remove_cast.click(handlers.remove_cast, inputs=[*current, view.remove_cast_name],
                           outputs=rendered, **QUIET)

    for strip in (view.strip, view.timeline):
        strip.click(handlers.strip_action, inputs=current, outputs=rendered, **QUIET)
    scene_fields = [view.title, view.prompt, view.seconds, view.link, view.cast, view.overlap,
                    view.seed, view.takes_per_run]
    for component in scene_fields:
        trigger = component.blur if isinstance(component, (gr.Textbox, gr.Number)) else component.input
        trigger(handlers.save_scene, inputs=[*current, *scene_fields], outputs=rendered, **QUIET)
    view.duplicate_scene.click(handlers.duplicate_scene, inputs=current, outputs=rendered, **QUIET)
    view.move_up.click(handlers.move_scene(-1), inputs=current, outputs=rendered, **QUIET)
    view.move_down.click(handlers.move_scene(1), inputs=current, outputs=rendered, **QUIET)
    view.delete_scene.click(handlers.delete_scene, inputs=current, outputs=rendered, **QUIET)
    view.use_take.click(handlers.use_take, inputs=[*current, view.takes], outputs=rendered, **QUIET)
    view.save_version.click(handlers.save_version, inputs=[*current, view.version_label],
                            outputs=[*rendered, view.version_label], **QUIET)
    view.branch.click(handlers.branch, inputs=[*current, view.versions], outputs=rendered, **QUIET)

    joined = [view.full_video, view.status]
    view.play_all.click(handlers.play_all, inputs=[view.storyboard, view.include_outdated], outputs=joined,
                        api_name=False, concurrency_id="h3-storyboard-join", concurrency_limit=1)
    exported = view.export.click(handlers.export, inputs=[view.storyboard, view.include_outdated], outputs=joined,
                                 api_name=False, concurrency_id="h3-storyboard-join", concurrency_limit=1)
    exported.then(handlers.refresh, inputs=current, outputs=rendered, **QUIET)

    # Both GPU buttons share Interrupt; it must be attached before binding.
    view.generate_remaining.h3_stop = view.stop
    generated = bind_gpu_action(
        view.generate.click, owned_generation(handlers.generate_scene, "storyboard"),
        inputs=current, outputs=[view.status, view.take_video], show_progress="minimal", api_name=False,
    )
    remaining = bind_gpu_action(
        view.generate_remaining.click, owned_generation(handlers.generate_remaining, "storyboard"),
        inputs=[view.storyboard], outputs=[view.status, view.take_video], show_progress="minimal",
        api_name=False,
    )
    for event in (generated, remaining):
        event.then(handlers.refresh, inputs=current, outputs=rendered, **QUIET)
    stopped = view.stop.click(owned_interrupt(interrupt, "storyboard"), outputs=view.status, **QUIET)
    stopped.then(handlers.refresh, inputs=current, outputs=rendered, **QUIET)
