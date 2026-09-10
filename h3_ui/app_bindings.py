"""Application event graph separated from view composition."""

from __future__ import annotations

from .contracts import AppComponents, AppServices
from .events.gallery import bind_gallery
from .events.generation import bind_generation
from .events.image_results import bind_image_results
from .events.input_upscale import bind_input_upscale
from .events.layout import bind_layout
from .events.ltx import bind_ltx
from .events.other_generation import bind_other_generation
from .events.prompt import bind_prompt
from .events.resolution import bind_resolution
from .events.system import bind_system
from .job_bindings import owned_interrupt
from .settings_controller import SettingsController, SettingsServices


def bind_app(
    components: AppComponents,
    services: AppServices,
) -> SettingsController:
    controller = SettingsController(
        components.as_mapping(),
        SettingsServices(
            services.resolve_request_settings,
            services.describe_settings,
            services.fbcache_preset_defaults,
        ),
    )
    controller.bind()
    bind_layout(components, services)
    ltx25_event = bind_ltx(components, services)
    bind_resolution(components, services, controller)
    input_upscale_event = bind_input_upscale(components, services)
    event, advanced_api_event = bind_generation(components, services)
    bind_image_results(components, services)
    bind_prompt(components, services)
    music3_event, api_event = bind_other_generation(components, services)
    for button, status, family, events in (
        (
            components.stop,
            components.status,
            "h3",
            [event, advanced_api_event, input_upscale_event],
        ),
        (components.ltx25_stop, components.ltx25_status, "ltx", [ltx25_event]),
        (components.music3_stop, components.music3_status, "music", [music3_event]),
        (components.api_stop, components.api_status, "api", [api_event]),
    ):
        stopped = button.click(
            owned_interrupt(services.interrupt, family),
            outputs=status,
            queue=False,
            api_name=False,
        )
        stopped.then(fn=None, cancels=events, queue=False, api_name=False)
    bind_system(components, services)
    bind_gallery(components, services)

    return controller
