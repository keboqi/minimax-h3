"""Application event graph separated from view composition."""

from __future__ import annotations

import gradio as gr
from .contracts import AppComponents, AppServices
from .settings_controller import SettingsController, SettingsServices
from .job_bindings import owned_generation, owned_interrupt
from h3_app.contracts import GENERATION_FIELDS, GENERATION_COMPONENTS


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
    components.mode.change(
        services.mode_layout_updates,
        inputs=components.mode,
        outputs=[
            components.help_text,
            components.frame_group,
            components.reference_group,
            components.generation_mode,
            components.preset,
            components.steps,
            components.scheduler,
            components.cache_mode,
            components.attention_mode,
        ],
    )
    components.result_format.change(
        services.result_format_layout_updates,
        inputs=[
            components.result_format,
            components.width,
            components.height,
            components.first,
            components.latent_upscale,
        ],
        outputs=[
            components.duration,
            components.image_frames,
            components.image_vae,
            components.output,
            components.output_2,
            components.output_3,
            components.output_4,
            components.batch_count,
            components.image_output_group,
            components.audio_output,
            components.generation_postprocess,
            components.latent_upscale,
            components.width,
            components.height,
            components.resolution_info,
            components.run,
        ],
        queue=False,
        show_progress="hidden",
    )
    components.trt_vae_compile.click(
        services.compile_trt_video_vae,
        outputs=components.status,
        show_progress="full",
    )
    components.image_vae.change(
        services.image_vae_frame_updates,
        inputs=components.image_vae,
        outputs=components.image_frames,
        queue=False,
        show_progress="hidden",
    )
    components.latent_upscale.change(
        services.latent_upscale_layout_updates,
        inputs=[
            components.latent_upscale,
            components.width,
            components.height,
            components.result_format,
        ],
        outputs=[
            components.latent_upscale_settings,
            components.width,
            components.height,
            components.resolution_info,
        ],
    )
    components.latent_upscale_method.change(
        services.latent_upscale_method_layout_update,
        inputs=components.latent_upscale_method,
        outputs=components.latent_split_settings,
        queue=False,
        show_progress="hidden",
    )
    ltx25_event = services.bind_ltx_view(
        components.ltx25_components,
        render_workflow=services.render_ltx25_workflow_details,
        prepare_workflow=services.prepare_ltx25_official_workflow,
        prepare_all_models=services.prepare_all_ltx25_official_models,
        render_inventory=services.render_ltx25_official_model_inventory,
        enhance_prompt=services.enhance_ltx25_prompt,
        generate=services.generate_ltx25,
    )
    components.generation_postprocess.change(
        lambda value: (
            gr.update(visible=value in services.AI_POSTPROCESS_OPTIONS),
            gr.update(visible=value == services.SEEDVR2_UPSCALE),
            gr.update(visible=value == services.LTX25_UPSCALE),
            gr.update(visible=value == services.LTX25_UPSCALE),
            gr.update(visible=value == services.LTX25_UPSCALE),
        ),
        inputs=components.generation_postprocess,
        outputs=[
            components.generation_postprocess_settings,
            components.generation_seedvr2_model,
            components.generation_ltx25_note,
            components.generation_split_upscale,
            components.generation_split_seconds,
        ],
        queue=False,
        show_progress="hidden",
    )
    draft_resolution_event = components.draft_resolution.change(
        lambda name, latent_upscale, result_format: services.resolution_choice_updates(
            name, "draft", latent_upscale, result_format
        ),
        inputs=[
            components.draft_resolution,
            components.latent_upscale,
            components.result_format,
        ],
        outputs=[
            components.width,
            components.height,
            components.resolution_info,
        ],
    )
    fast_resolution_event = components.fast_resolution.change(
        lambda name, latent_upscale, result_format: services.resolution_choice_updates(
            name, "fast", latent_upscale, result_format
        ),
        inputs=[
            components.fast_resolution,
            components.latent_upscale,
            components.result_format,
        ],
        outputs=[
            components.width,
            components.height,
            components.resolution_info,
        ],
    )
    large_resolution_event = components.large_resolution.change(
        lambda name, latent_upscale, result_format: services.resolution_choice_updates(
            name, "large", latent_upscale, result_format
        ),
        inputs=[
            components.large_resolution,
            components.latent_upscale,
            components.result_format,
        ],
        outputs=[
            components.width,
            components.height,
            components.resolution_info,
        ],
    )
    for resolution_event in (
        draft_resolution_event,
        fast_resolution_event,
        large_resolution_event,
    ):
        resolution_event.then(
            controller.refresh,
            inputs=[controller.memory, *controller.inputs],
            outputs=controller.outputs,
            queue=False,
            show_progress="hidden",
        )
    components.first.upload(
        fn=None,
        inputs=[
            components.first,
            components.width,
            components.height,
            components.result_format,
            components.latent_upscale,
            components.auto_megapixels,
        ],
        outputs=[
            components.width,
            components.height,
            components.resolution_info,
        ],
        js=services.AUTO_RESOLUTION_JS,
        queue=False,
        show_progress="hidden",
    )
    first_change_event = components.first.change(
        fn=services.auto_resolution_from_start_frame,
        inputs=[
            components.first,
            components.width,
            components.height,
            components.result_format,
            components.latent_upscale,
            components.auto_megapixels,
        ],
        outputs=[
            components.width,
            components.height,
            components.resolution_info,
        ],
        queue=False,
        show_progress="hidden",
    )
    first_change_event.then(
        controller.refresh,
        inputs=[controller.memory, *controller.inputs],
        outputs=controller.outputs,
        queue=False,
        show_progress="hidden",
    )
    auto_megapixels_change = components.auto_megapixels.input(
        fn=services.auto_resolution_from_start_frame,
        inputs=[
            components.first,
            components.width,
            components.height,
            components.result_format,
            components.latent_upscale,
            components.auto_megapixels,
        ],
        outputs=[
            components.width,
            components.height,
            components.resolution_info,
        ],
        queue=False,
        show_progress="hidden",
    )
    auto_megapixels_change.then(
        controller.refresh,
        inputs=[controller.memory, *controller.inputs],
        outputs=controller.outputs,
        queue=False,
        show_progress="hidden",
    )
    components.width.input(
        services.resolution_info_preview,
        inputs=[
            components.width,
            components.height,
            components.latent_upscale,
            components.result_format,
        ],
        outputs=[components.resolution_info],
        queue=False,
        show_progress="hidden",
    )
    components.height.input(
        services.resolution_info_preview,
        inputs=[
            components.width,
            components.height,
            components.latent_upscale,
            components.result_format,
        ],
        outputs=[components.resolution_info],
        queue=False,
        show_progress="hidden",
    )
    for res_event_trigger in (
        components.width.blur,
        components.height.blur,
        components.width.submit,
        components.height.submit,
    ):
        res_snap_event = res_event_trigger(
            services.resolution_control_updates,
            inputs=[
                components.width,
                components.height,
                components.latent_upscale,
                components.result_format,
            ],
            outputs=[
                components.width,
                components.height,
                components.resolution_info,
            ],
            queue=False,
            show_progress="hidden",
        )
        res_snap_event.then(
            controller.refresh,
            inputs=[controller.memory, *controller.inputs],
            outputs=controller.outputs,
            queue=False,
            show_progress="hidden",
        )
    components.input_upscale_frame_preset.change(
        services.input_image_frame_preset_updates,
        inputs=[
            components.input_upscale_frame_preset,
            components.input_upscale_frame_width,
            components.input_upscale_frame_height,
        ],
        outputs=[
            components.input_upscale_frame_width,
            components.input_upscale_frame_height,
        ],
        queue=False,
        show_progress="hidden",
        api_name=False,
    )
    input_upscale_event = components.input_upscale_run.click(
        owned_generation(services.upscale_selected_input_images, "h3-input"),
        inputs=[
            components.input_upscale_slots,
            components.input_upscale_model,
            components.input_upscale_seed,
            components.input_upscale_force_offload,
            components.input_upscale_frame_width,
            components.input_upscale_frame_height,
            components.first,
            components.last,
            components.ref_image_1,
            components.ref_image_2,
            components.ref_image_3,
            components.ref_image_4,
            components.ref_image_5,
            components.ref_image_6,
            components.ref_image_7,
            components.ref_image_8,
            components.ref_image_9,
        ],
        outputs=[
            components.first,
            components.last,
            components.ref_image_1,
            components.ref_image_2,
            components.ref_image_3,
            components.ref_image_4,
            components.ref_image_5,
            components.ref_image_6,
            components.ref_image_7,
            components.ref_image_8,
            components.ref_image_9,
            components.input_upscale_downloads,
            components.input_upscale_status,
        ],
        concurrency_id="h3-gpu",
        concurrency_limit=1,
        show_progress="minimal",
        api_name="upscale_h3_input_images",
    )
    generation_inputs = [
        components.batch_count,
        *(getattr(components, name) for name in GENERATION_COMPONENTS),
    ]
    generation_outputs = [
        components.output,
        components.output_2,
        components.output_3,
        components.output_4,
        components.image_output_group,
        components.image_output,
        components.image_selection,
        components.image_frame_paths,
        components.image_saved_files,
        components.audio_output,
        components.image_save_status,
        components.status,
    ]
    event = components.run.click(
        owned_generation(
            services.generate_for_ui,
            "h3",
            ("batch_count", *GENERATION_FIELDS, "preset"),
            metadata_output=True,
        ),
        inputs=[*generation_inputs, components.preset],
        outputs=[*generation_outputs, components.settings_used],
        concurrency_id="h3-gpu",
        concurrency_limit=1,
        show_progress="minimal",
        api_name=False,
    )
    advanced_api_event = gr.Button(visible=False).click(
        owned_generation(
            services.generate_for_ui, "h3", ("batch_count", *GENERATION_FIELDS)
        ),
        inputs=generation_inputs,
        outputs=generation_outputs,
        concurrency_id="h3-gpu",
        concurrency_limit=1,
        show_progress="minimal",
        api_name="generate_video_advanced",
    )
    components.image_select_all.click(
        services.select_all_image_frames,
        inputs=components.image_frame_paths,
        outputs=components.image_selection,
        queue=False,
        show_progress="hidden",
        api_name=False,
    )
    components.image_clear_selection.click(
        lambda: [],
        outputs=components.image_selection,
        queue=False,
        show_progress="hidden",
        api_name=False,
    )
    components.image_save_selected.click(
        services.save_selected_image_frames,
        inputs=[components.image_frame_paths, components.image_selection],
        outputs=[components.image_saved_files, components.image_save_status],
        concurrency_id="h3-gpu",
        concurrency_limit=1,
        show_progress="minimal",
        api_name="save_h3_image_frames",
    )
    components.prompt_writer_backend.change(
        services.prompt_writer_backend_visibility,
        inputs=components.prompt_writer_backend,
        outputs=[
            components.local_prompt_writer_group,
            components.gemini_prompt_writer_group,
            components.lightning_prompt_writer_group,
        ],
        queue=False,
        show_progress="hidden",
        api_name=False,
    )
    components.enhance_prompt_button.click(
        services.enhance_h3_prompt,
        inputs=[
            components.prompt,
            components.prompt_writer_backend,
            components.local_prompt_base_model,
            components.local_prompt_max_tokens,
            components.local_prompt_temperature,
            components.local_prompt_top_p,
            components.local_prompt_greedy,
            components.local_prompt_seed,
            components.gemini_prompt_model,
            components.gemini_api_key,
            components.lightning_api_key,
            components.mode,
            components.first,
            components.last,
            components.ref_image_1,
            components.ref_image_2,
            components.ref_image_3,
            components.ref_image_4,
            components.ref_image_5,
            components.ref_image_6,
            components.ref_image_7,
            components.ref_image_8,
            components.ref_image_9,
            components.ref_video_1,
            components.ref_video_2,
            components.ref_video_3,
            components.ref_audio_1,
            components.ref_audio_2,
            components.ref_audio_3,
            components.duration,
            components.width,
            components.height,
            components.result_format,
            components.image_frames,
        ],
        outputs=[components.prompt, components.enhance_prompt_status],
        concurrency_id="h3-gpu",
        concurrency_limit=1,
        show_progress="minimal",
        api_name="enhance_prompt",
    )
    music3_event = services.bind_music_view(
        components.music3_components,
        enhance_prompt=services.enhance_music3_prompt,
        generate=services.generate_music3,
    )
    api_event = services.bind_api_view(
        components.api_components, generate=services.generate_with_ui_defaults
    )
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
    components.refresh.click(
        services.refresh_backend_views,
        outputs=[components.system_summary, components.health],
        queue=False,
        show_progress="hidden",
    )
    components.unload_models.click(
        services.unload_all_models,
        outputs=[components.memory_status, components.health],
        concurrency_id="h3-gpu",
        concurrency_limit=1,
        show_progress="minimal",
        api_name=False,
    ).then(
        services.refresh_backend_views,
        outputs=[components.system_summary, components.health],
        queue=False,
        show_progress="hidden",
    )
    services.bind_gallery_view(
        components.gallery_components,
        tab=components.gallery_tab,
        selected_ltx_model=components.ltx25_model,
        ai_options=services.AI_POSTPROCESS_OPTIONS,
        seedvr_option=services.SEEDVR2_UPSCALE,
        ltx_option=services.LTX25_UPSCALE,
        refresh=services.refresh_gallery,
        select=services.select_gallery_video,
        import_video=services.import_gallery_video,
        postprocess=services.postprocess_selected_gallery_video,
        interrupt=services.interrupt,
        delete=services.delete_selected_gallery_video,
        empty=services.empty_generated_gallery,
    )

    return controller
