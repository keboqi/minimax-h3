"""Bind system actions."""

from __future__ import annotations

from ..contracts import AppComponents, AppServices
from ..job_bindings import bind_gpu_action


def bind_system(components: AppComponents, services: AppServices) -> None:
    components.refresh.click(
        services.refresh_backend_views,
        outputs=[components.system_summary, components.health],
        queue=False,
        show_progress="hidden",
    )
    bind_gpu_action(
        components.unload_models.click,
        services.unload_all_models,
        outputs=[components.memory_status, components.health],
        show_progress="minimal",
        api_name=False,
    ).then(
        services.refresh_backend_views,
        outputs=[components.system_summary, components.health],
        queue=False,
        show_progress="hidden",
    )
