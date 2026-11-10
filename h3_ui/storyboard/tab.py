"""The Storyboard tab: the page in a frame, the scripted API and the H3 settings it needs from Gradio."""

from __future__ import annotations

from functools import partial
import inspect

import gradio as gr

from h3_app.storyboard import DEFAULTED_FIELDS, PROFILE_FIELDS

from ..job_admission import require_owner
from .handlers import WRITER_FIELDS, StoryboardHandlers
from .page import install_storyboard_page

FRAME = ('<iframe class="h3-storyboard-frame" src="/storyboard/" title="Storyboard" '
         'allow="fullscreen" scrolling="no"></iframe>')


def _bind_api(handlers: StoryboardHandlers):
    """API-only endpoints for scripted clients: import or update a storyboard, list them, read one.

    Each endpoint has its own hidden trigger, so none of them runs from the page.
    """
    with gr.Group(visible=False):
        spec = gr.Textbox(label="Storyboard spec (JSON)")
        files = gr.File(label="Scene and cast files", file_count="multiple", type="filepath")
        storyboard_id = gr.Textbox(label="Storyboard id")
        result = gr.JSON(label="Storyboard")
        options = {"show_progress": "hidden", "concurrency_id": "h3-storyboard-edit", "concurrency_limit": 4}
        for fn, inputs, api_name in (
            (handlers.storyboard_import, [spec, files], "storyboard_import"),
            (handlers.storyboard_update, [storyboard_id, spec, files], "storyboard_update"),
            (handlers.storyboard_list, [], "storyboard_list"),
            (handlers.storyboard_details, [storyboard_id], "storyboard_details"),
        ):
            gr.Button(visible=False).click(fn, inputs=inputs, outputs=result, api_name=api_name, **options)


def _bind_settings_capture(handlers, h3_values, triggers):
    """Hand the MiniMax H3 tab's values to the page's server side on each of ``triggers``."""
    missing = [name for name in PROFILE_FIELDS if name not in h3_values and name not in DEFAULTED_FIELDS]
    if missing:
        raise ValueError(f"Storyboard needs these MiniMax H3 settings: {missing}")
    profile = [name for name in PROFILE_FIELDS if name in h3_values]
    writer = [name for name in WRITER_FIELDS if name in h3_values]

    def capture(*args):
        *values, request = args
        handlers.capture_settings(
            require_owner(request),
            dict(zip(profile, values[:len(profile)], strict=True)),
            dict(zip(writer, values[len(profile):], strict=True)),
        )

    parameters = [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in (*profile, *writer)]
    parameters.append(inspect.Parameter("request", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                        annotation=gr.Request, default=None))
    capture.__signature__ = inspect.Signature(parameters)
    # Gradio injects the request from annotations; without this it passes None.
    capture.__annotations__ = {"request": gr.Request}
    gr.on(triggers=triggers, fn=capture, inputs=[h3_values[name] for name in (*profile, *writer)], outputs=None,
          queue=False, api_name=False, show_progress="hidden")


def build_storyboard_tab(root, demo: gr.Blocks, handlers: StoryboardHandlers, *, h3_values=None, triggers=(),
                         interrupt=None, gpu=True):
    """Fill ``root``. ``h3_values`` maps MiniMax H3 setting names to their components (Studio only).

    Without them (the Workshop) storyboards start from the default settings and the page's
    own writer settings, and ``gpu=False`` leaves out generating.
    """
    with root:
        gr.HTML(FRAME)
        _bind_api(handlers)
    if h3_values and triggers:
        _bind_settings_capture(handlers, h3_values, list(triggers))
    # Read by build_server, which owns the application these routes join.
    demo.h3_routes = (*getattr(demo, "h3_routes", ()),
                      partial(install_storyboard_page, handlers=handlers, gpu=gpu, interrupt=interrupt))
