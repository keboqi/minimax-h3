"""Bind system actions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr

from ..job_bindings import bind_gpu_action

if TYPE_CHECKING:
    from ..contracts import BootstrapServices
    from ..system_view import SystemView


def bind_system(
    demo: gr.Blocks,
    view: SystemView,
    *,
    summary: gr.HTML,
    create_refresh: gr.Button,
    services: BootstrapServices,
) -> None:
    backend_outputs = [summary, view.health]
    refresh_from_create = create_refresh.click(
        services.refresh_backend_views,
        outputs=backend_outputs,
        queue=False,
        show_progress="hidden",
    )
    refresh_after_unload = bind_gpu_action(
        view.unload_models.click,
        services.unload_all_models,
        outputs=[view.memory_status, view.health],
        show_progress="minimal",
        api_name=False,
    ).then(
        services.refresh_backend_views,
        outputs=backend_outputs,
        queue=False,
        show_progress="hidden",
    )
    # Preserve the two-output API of both existing refresh events.
    for event in (refresh_from_create, refresh_after_unload):
        event.success(
            None,
            inputs=summary,
            outputs=view.runtime,
            js="(summary) => summary",
            queue=False,
            api_name=False,
            show_progress="hidden",
        )

    def refresh_page():
        status, detail = services.refresh_backend_views()
        return status, detail, status

    gr.on(
        triggers=[demo.load, view.refresh.click],
        fn=refresh_page,
        outputs=[*backend_outputs, view.runtime],
        queue=False,
        api_name=False,
        show_progress="hidden",
        trigger_mode="always_last",
    )
    if services.drama3_settings_state is not None:
        demo.load(
            services.drama3_settings_state,
            outputs=[
                view.drama3_backend,
                view.drama3_model,
                view.drama3_comfy_experts,
                view.drama3_status,
            ],
            queue=False,
            api_name=False,
            show_progress="hidden",
        )
    if services.update_drama3_settings is not None:
        drama3_inputs = [view.drama3_backend, view.drama3_model, view.drama3_comfy_experts]
        drama3_outputs = [view.drama3_model, view.drama3_comfy_experts, view.drama3_status]
        gr.on(
            triggers=[
                view.drama3_backend.input,
                view.drama3_model.input,
                view.drama3_comfy_experts.input,
            ],
            fn=services.update_drama3_settings,
            inputs=drama3_inputs,
            outputs=drama3_outputs,
            queue=False,
            api_name=False,
            show_progress="hidden",
        )
