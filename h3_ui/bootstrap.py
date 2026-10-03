"""Composition root with explicit catalogs and callbacks; no generation policy."""

from dataclasses import dataclass
from contextlib import nullcontext
from typing import Any, Callable
import gradio as gr
from .app_bindings import bind_app
from .contracts import AppComponents, AppServices
from .h3_view import build_h3_view, H3ViewServices
from .layout import create_app_views, bind_workspace_navigation
from .persistence import bind_browser_settings
from .ltx_view import build_ltx_view
from .styles import H3_SETUP_CSS, H3_WORKSPACE_CSS
from .api_view import build_api_view
from .media_view import build_gallery_view
from .music_view import build_music_view
from .qwen_view import build_qwen_image21_view
from .yue2_view import build_yue2_view
from .api_bindings import bind_api_view
from .media_bindings import bind_gallery_view
from .music_bindings import bind_music_view
from .ltx_bindings import bind_ltx_view
from .qwen_bindings import bind_qwen_image21_view
from .yue2_bindings import bind_yue2_view


@dataclass(frozen=True)
class BootstrapCatalog:
    AI_POSTPROCESS_OPTIONS: Any
    AUTO_RESOLUTION_MEGAPIXEL_PRESETS: Any
    AUTO_SOL_TOKEN_THRESHOLD: Any
    COMFY_DIR: Any
    DEFAULT_AUTO_RESOLUTION_MEGAPIXELS: Any
    DEFAULT_GEMINI_PROMPT_MODEL: Any
    DEFAULT_INPUT_IMAGE_FRAME_PRESET: Any
    DEFAULT_LOCAL_PROMPT_BASE_MODEL: Any
    DEFAULT_LTX25_MODEL: Any
    DEFAULT_PROMPT_WRITER_BACKEND: Any
    DEFAULT_UPSCALE_RESOLUTION: Any
    DEFAULT_VIDEO_BATCH_COUNT: Any
    DRAFT_RESOLUTIONS: Any
    FAST_RESOLUTIONS: Any
    GEMINI_PROMPT_MODELS: Any
    GENERATION_POSTPROCESS_OPTIONS: Any
    H3_LATENT_UPSCALER_MODEL_CHOICES: Any
    H3_LATENT_UPSCALE_METHODS: Any
    H3_LATENT_UPSCALE_SPLIT: Any
    H3_TEXT_ENCODER_CHOICES: Any
    IMAGE_VAE_CHOICES: Any
    INPUT_IMAGE_FRAME_PRESETS: Any
    INPUT_IMAGE_UPSCALE_SLOTS: Any
    LARGE_RESOLUTIONS: Any
    LIGHTNING_PROMPT_MODEL: Any
    LOCAL_PROMPT_BASE_MODELS: Any
    LTX25_DEFAULTS: Any
    LTX25_MODEL_CHOICES: Any
    LTX25_UPSCALE: Any
    LTX25_WORKFLOWS: Any
    MAX_IMAGE_FRAMES: Any
    MAX_VIDEO_BATCH_COUNT: Any
    MIN_IMAGE_FRAMES: Any
    MIN_VIDEO_BATCH_COUNT: Any
    MODEL_PROFILE_CHOICES: Any
    MUSIC3_DEFAULTS: Any
    MUSIC3_MODEL_CHOICES: Any
    POSTPROCESS_OPTIONS: Any
    PROMPT_WRITER_BACKENDS: Any
    QWEN_IMAGE21_DEFAULTS: Any
    QWEN_IMAGE21_MODEL_CHOICES: Any
    QWEN_IMAGE21_TEXT_ENCODER_CHOICES: Any
    RESULT_FORMATS: Any
    SEEDVR2_MODEL_CHOICES: Any
    SEEDVR2_UPSCALE: Any
    SERVER_ATTENTION_BACKEND: Any
    SERVER_DENSE_ATTENTION_BACKEND: Any
    SLA_PRESET_INPUTS: Any
    TURBO_SETTINGS: Any
    UI_DEFAULTS: Any
    UPSCALE_RESOLUTION_PRESETS: Any
    YUE2_DEFAULTS: Any
    YUE2_MODEL_CHOICES: Any


@dataclass(frozen=True)
class BootstrapServices:
    api_get: Callable
    api_guide: Callable
    api_post: Callable
    auto_resolution_from_start_frame: Callable
    backend_status: Callable
    compact_backend_status: Callable
    compact_settings_summary: Callable
    compile_trt_video_vae: Callable
    delete_selected_gallery_media: Callable
    describe_settings: Callable
    empty_generated_media_gallery: Callable
    enhance_h3_prompt: Callable
    enhance_ltx25_prompt: Callable
    enhance_music3_prompt: Callable
    enhance_qwen_image21_prompt: Callable
    enhance_yue2_prompt: Callable
    fbcache_preset_defaults: Callable
    generate_for_ui: Callable
    generate_ltx25: Callable
    generate_music3: Callable
    generate_qwen_image21: Callable
    generate_with_ui_defaults: Callable
    generate_yue2: Callable
    generation_readiness_state: Callable
    image_vae_frame_updates: Callable
    import_gallery_media: Callable
    input_image_frame_preset_updates: Callable
    interrupt: Callable
    latent_upscale_layout_updates: Callable
    latent_upscale_method_layout_update: Callable
    list_media_paths: Callable
    mode_help: Callable
    mode_layout_updates: Callable
    postprocess_selected_gallery_media: Callable
    prepare_all_ltx25_official_models: Callable
    prepare_ltx25_official_workflow: Callable
    prompt_writer_backend_visibility: Callable
    reference_prompt_help: Callable
    refresh_backend_views: Callable
    refresh_media_gallery: Callable
    refresh_media_page: Callable
    render_ltx25_official_model_inventory: Callable
    render_ltx25_workflow_details: Callable
    render_snapshot: Callable
    resolution_choice_updates: Callable
    resolution_control_updates: Callable
    resolution_info_preview: Callable
    resolution_summary: Callable
    resolve_request_settings: Callable
    result_format_layout_updates: Callable
    result_settings_for_media: Callable
    save_selected_image_frames: Callable
    select_all_image_frames: Callable
    select_gallery_media: Callable
    unload_all_models: Callable
    upscale_selected_input_images: Callable


def build_ui(catalog: BootstrapCatalog, services: BootstrapServices) -> gr.Blocks:
    defaults = catalog.UI_DEFAULTS
    initial_backend = services.backend_status()
    with gr.Blocks(title="MiniMax H3 Local") as demo, gr.Column(
        elem_classes=["h3-workspace"]
    ):
        with gr.Row(elem_classes=["h3-workspace-header"]):
            gr.HTML(
                '<section class="h3-hero"><h1>MiniMax H3 Local</h1><p>Create video, images, audio, and music on the shared ComfyUI backend · <a href="/comfyui/" target="_blank" rel="noopener noreferrer">Open ComfyUI ↗</a></p></section>',
                scale=3,
            )
            summary_root = gr.Column(scale=2, elem_classes=["h3-header-status"])
            with summary_root:
                system_summary = gr.HTML(
                    services.compact_backend_status(initial_backend)
                )
        app_views = create_app_views(summary_root=summary_root)
        with app_views.system if app_views is not None else nullcontext():
            with gr.Accordion("System details and VRAM", open=False):
                with gr.Row(equal_height=True):
                    health = gr.Markdown(initial_backend)
                    unload_models = gr.Button("Unload all models / free VRAM", scale=0)
                memory_status = gr.Markdown()
        app_views = app_views or create_app_views()
        generation_view = app_views.generation
        qwen_image21_view = app_views.qwen_image21
        ltx25_view = app_views.ltx25
        music3_view = app_views.music3
        yue2_view = app_views.yue2
        gallery_view = app_views.gallery
        api_view = app_views.api
        gallery_tab = app_views.gallery_tab
        h3_components = build_h3_view(
            generation_view,
            defaults,
            H3ViewServices(
                AUTO_RESOLUTION_MEGAPIXEL_PRESETS=catalog.AUTO_RESOLUTION_MEGAPIXEL_PRESETS,
                AUTO_SOL_TOKEN_THRESHOLD=catalog.AUTO_SOL_TOKEN_THRESHOLD,
                DEFAULT_AUTO_RESOLUTION_MEGAPIXELS=catalog.DEFAULT_AUTO_RESOLUTION_MEGAPIXELS,
                DEFAULT_GEMINI_PROMPT_MODEL=catalog.DEFAULT_GEMINI_PROMPT_MODEL,
                DEFAULT_INPUT_IMAGE_FRAME_PRESET=catalog.DEFAULT_INPUT_IMAGE_FRAME_PRESET,
                DEFAULT_LOCAL_PROMPT_BASE_MODEL=catalog.DEFAULT_LOCAL_PROMPT_BASE_MODEL,
                DEFAULT_LTX25_MODEL=catalog.DEFAULT_LTX25_MODEL,
                DEFAULT_PROMPT_WRITER_BACKEND=catalog.DEFAULT_PROMPT_WRITER_BACKEND,
                DEFAULT_UPSCALE_RESOLUTION=catalog.DEFAULT_UPSCALE_RESOLUTION,
                DEFAULT_VIDEO_BATCH_COUNT=catalog.DEFAULT_VIDEO_BATCH_COUNT,
                DRAFT_RESOLUTIONS=catalog.DRAFT_RESOLUTIONS,
                FAST_RESOLUTIONS=catalog.FAST_RESOLUTIONS,
                GEMINI_PROMPT_MODELS=catalog.GEMINI_PROMPT_MODELS,
                GENERATION_POSTPROCESS_OPTIONS=catalog.GENERATION_POSTPROCESS_OPTIONS,
                H3_LATENT_UPSCALE_METHODS=catalog.H3_LATENT_UPSCALE_METHODS,
                H3_LATENT_UPSCALE_SPLIT=catalog.H3_LATENT_UPSCALE_SPLIT,
                H3_LATENT_UPSCALER_MODEL_CHOICES=catalog.H3_LATENT_UPSCALER_MODEL_CHOICES,
                H3_TEXT_ENCODER_CHOICES=catalog.H3_TEXT_ENCODER_CHOICES,
                IMAGE_VAE_CHOICES=catalog.IMAGE_VAE_CHOICES,
                INPUT_IMAGE_FRAME_PRESETS=catalog.INPUT_IMAGE_FRAME_PRESETS,
                INPUT_IMAGE_UPSCALE_SLOTS=catalog.INPUT_IMAGE_UPSCALE_SLOTS,
                LARGE_RESOLUTIONS=catalog.LARGE_RESOLUTIONS,
                LIGHTNING_PROMPT_MODEL=catalog.LIGHTNING_PROMPT_MODEL,
                LOCAL_PROMPT_BASE_MODELS=catalog.LOCAL_PROMPT_BASE_MODELS,
                MAX_IMAGE_FRAMES=catalog.MAX_IMAGE_FRAMES,
                MAX_VIDEO_BATCH_COUNT=catalog.MAX_VIDEO_BATCH_COUNT,
                MIN_IMAGE_FRAMES=catalog.MIN_IMAGE_FRAMES,
                MIN_VIDEO_BATCH_COUNT=catalog.MIN_VIDEO_BATCH_COUNT,
                MODEL_PROFILE_CHOICES=catalog.MODEL_PROFILE_CHOICES,
                PROMPT_WRITER_BACKENDS=catalog.PROMPT_WRITER_BACKENDS,
                RESULT_FORMATS=catalog.RESULT_FORMATS,
                LTX25_MODEL_CHOICES=catalog.LTX25_MODEL_CHOICES,
                SEEDVR2_MODEL_CHOICES=catalog.SEEDVR2_MODEL_CHOICES,
                SERVER_ATTENTION_BACKEND=catalog.SERVER_ATTENTION_BACKEND,
                SERVER_DENSE_ATTENTION_BACKEND=catalog.SERVER_DENSE_ATTENTION_BACKEND,
                SLA_PRESET_INPUTS=catalog.SLA_PRESET_INPUTS,
                TURBO_SETTINGS=catalog.TURBO_SETTINGS,
                UPSCALE_RESOLUTION_PRESETS=catalog.UPSCALE_RESOLUTION_PRESETS,
                compact_settings_summary=services.compact_settings_summary,
                generation_readiness_state=services.generation_readiness_state,
                mode_help=services.mode_help,
                reference_prompt_help=services.reference_prompt_help,
                resolution_summary=services.resolution_summary,
            ),
            advanced_root=app_views.generation_settings,
        )
        ltx25_components = build_ltx_view(
            ltx25_view,
            model_choices=catalog.LTX25_MODEL_CHOICES,
            defaults=catalog.LTX25_DEFAULTS,
            prompt_models=catalog.GEMINI_PROMPT_MODELS,
            default_prompt_model=catalog.DEFAULT_GEMINI_PROMPT_MODEL,
            workflows=tuple(catalog.LTX25_WORKFLOWS),
            initial_workflow_details=services.render_ltx25_workflow_details(
                next(iter(catalog.LTX25_WORKFLOWS))
            ),
            model_inventory_text=services.render_ltx25_official_model_inventory(),
        )
        music3_components = build_music_view(
            music3_view,
            prompt_models=catalog.GEMINI_PROMPT_MODELS,
            default_prompt_model=catalog.DEFAULT_GEMINI_PROMPT_MODEL,
            model_choices=catalog.MUSIC3_MODEL_CHOICES,
            defaults=catalog.MUSIC3_DEFAULTS,
        )
        qwen_image21_components = build_qwen_image21_view(
            qwen_image21_view,
            model_choices=catalog.QWEN_IMAGE21_MODEL_CHOICES,
            text_encoder_choices=catalog.QWEN_IMAGE21_TEXT_ENCODER_CHOICES,
            prompt_models=catalog.GEMINI_PROMPT_MODELS,
            default_prompt_model=catalog.DEFAULT_GEMINI_PROMPT_MODEL,
            defaults=catalog.QWEN_IMAGE21_DEFAULTS,
        )
        yue2_components = build_yue2_view(
            yue2_view,
            prompt_models=catalog.GEMINI_PROMPT_MODELS,
            default_prompt_model=catalog.DEFAULT_GEMINI_PROMPT_MODEL,
            model_choices=catalog.YUE2_MODEL_CHOICES,
            defaults=catalog.YUE2_DEFAULTS,
        )
        gallery_components = build_gallery_view(
            gallery_view,
            postprocess_options=catalog.POSTPROCESS_OPTIONS,
            resolution_choices=tuple(catalog.UPSCALE_RESOLUTION_PRESETS),
            default_resolution=catalog.DEFAULT_UPSCALE_RESOLUTION,
            seedvr2_choices=catalog.SEEDVR2_MODEL_CHOICES,
            default_seedvr2=defaults["seedvr2_model"],
            ltx25_choices=tuple(catalog.LTX25_MODEL_CHOICES),
            default_ltx25=catalog.DEFAULT_LTX25_MODEL,
        )
        with gallery_view:
            gallery_settings_used = gr.HTML("Select an output to inspect its settings.")
        from .library_tools import build_library_tools

        build_library_tools(gallery_view, services.list_media_paths)
        gallery_components.selected.change(
            services.render_snapshot,
            inputs=gallery_components.selected,
            outputs=gallery_settings_used,
            queue=False,
            api_name=False,
        )
        for root, view in (
            (ltx25_view, ltx25_components),
            (music3_view, music3_components),
            (qwen_image21_view, qwen_image21_components),
            (yue2_view, yue2_components),
        ):
            with root:
                metadata_view = gr.HTML(
                    "Settings used will appear with the generated result."
                )
            view.output.change(
                services.result_settings_for_media,
                inputs=view.output,
                outputs=metadata_view,
                queue=False,
                api_name=False,
                show_progress="hidden",
            )
        api_components = build_api_view(api_view, services.api_guide())
        app_components = dict(h3_components.values)
        app_components.update(
            {
                "api_components": api_components,
                "api_status": api_components.status,
                "api_stop": api_components.stop,
                "gallery_components": gallery_components,
                "gallery_tab": gallery_tab,
                "health": health,
                "ltx25_components": ltx25_components,
                "ltx25_status": ltx25_components.status,
                "ltx25_stop": ltx25_components.stop,
                "memory_status": memory_status,
                "music3_components": music3_components,
                "music3_status": music3_components.status,
                "music3_stop": music3_components.stop,
                "qwen_image21_components": qwen_image21_components,
                "qwen_image21_status": qwen_image21_components.status,
                "qwen_image21_stop": qwen_image21_components.stop,
                "yue2_components": yue2_components,
                "yue2_status": yue2_components.status,
                "yue2_stop": yue2_components.stop,
                "system_summary": system_summary,
                "unload_models": unload_models,
            }
        )
        settings_controller = bind_app(
            AppComponents.from_mapping(app_components),
            AppServices(
                resolve_request_settings=services.resolve_request_settings,
                describe_settings=services.describe_settings,
                AI_POSTPROCESS_OPTIONS=catalog.AI_POSTPROCESS_OPTIONS,
                LTX25_UPSCALE=catalog.LTX25_UPSCALE,
                SEEDVR2_UPSCALE=catalog.SEEDVR2_UPSCALE,
                auto_resolution_from_start_frame=services.auto_resolution_from_start_frame,
                bind_api_view=bind_api_view,
                bind_gallery_view=bind_gallery_view,
                bind_ltx_view=bind_ltx_view,
                bind_music_view=bind_music_view,
                bind_qwen_image21_view=bind_qwen_image21_view,
                bind_yue2_view=bind_yue2_view,
                compile_trt_video_vae=services.compile_trt_video_vae,
                delete_selected_gallery_media=services.delete_selected_gallery_media,
                empty_generated_media_gallery=services.empty_generated_media_gallery,
                enhance_h3_prompt=services.enhance_h3_prompt,
                enhance_ltx25_prompt=services.enhance_ltx25_prompt,
                enhance_music3_prompt=services.enhance_music3_prompt,
                enhance_qwen_image21_prompt=services.enhance_qwen_image21_prompt,
                enhance_yue2_prompt=services.enhance_yue2_prompt,
                fbcache_preset_defaults=services.fbcache_preset_defaults,
                generate_for_ui=services.generate_for_ui,
                generate_ltx25=services.generate_ltx25,
                generate_music3=services.generate_music3,
                generate_qwen_image21=services.generate_qwen_image21,
                generate_yue2=services.generate_yue2,
                generate_with_ui_defaults=services.generate_with_ui_defaults,
                image_vae_frame_updates=services.image_vae_frame_updates,
                import_gallery_media=services.import_gallery_media,
                input_image_frame_preset_updates=services.input_image_frame_preset_updates,
                interrupt=services.interrupt,
                latent_upscale_layout_updates=services.latent_upscale_layout_updates,
                latent_upscale_method_layout_update=services.latent_upscale_method_layout_update,
                mode_layout_updates=services.mode_layout_updates,
                postprocess_selected_gallery_media=services.postprocess_selected_gallery_media,
                prepare_all_ltx25_official_models=services.prepare_all_ltx25_official_models,
                prepare_ltx25_official_workflow=services.prepare_ltx25_official_workflow,
                prompt_writer_backend_visibility=services.prompt_writer_backend_visibility,
                refresh_backend_views=services.refresh_backend_views,
                refresh_media_gallery=services.refresh_media_gallery,
                refresh_media_page=services.refresh_media_page,
                list_media_paths=services.list_media_paths,
                render_ltx25_official_model_inventory=services.render_ltx25_official_model_inventory,
                render_ltx25_workflow_details=services.render_ltx25_workflow_details,
                resolution_control_updates=services.resolution_control_updates,
                resolution_choice_updates=services.resolution_choice_updates,
                resolution_info_preview=services.resolution_info_preview,
                result_format_layout_updates=services.result_format_layout_updates,
                save_selected_image_frames=services.save_selected_image_frames,
                select_all_image_frames=services.select_all_image_frames,
                select_gallery_media=services.select_gallery_media,
                unload_all_models=services.unload_all_models,
                upscale_selected_input_images=services.upscale_selected_input_images,
            ),
        )
        navigation = bind_workspace_navigation(
            app_views, h3_components.result_format, settings_controller
        )
        if app_views.jobs is not None:
            from h3_ui.jobs_view import build_jobs_view

            build_jobs_view(
                app_views.jobs,
                get=services.api_get,
                post=services.api_post,
                summary=app_views.jobs_count,
            )
            from h3_ui.system_view import build_readiness_view
            from h3_models import MODEL_SPECS

            build_readiness_view(
                app_views.system,
                model_specs=MODEL_SPECS,
                models_root=catalog.COMFY_DIR / "models",
            )
        from h3_ui.canvas_controls import bind_canvas_controls

        bind_canvas_controls(
            h3_components,
            services.resolution_choice_updates,
            settings_controller,
            {
                "draft": catalog.DRAFT_RESOLUTIONS,
                "fast": catalog.FAST_RESOLUTIONS,
                "large": catalog.LARGE_RESOLUTIONS,
            },
        )
        browser_settings = {
            **{
                f"h3.{name}": component
                for name, component in h3_components.values.items()
            },
            **{
                f"ltx25.{name}": component
                for name, component in vars(ltx25_components).items()
            },
            **{
                f"music3.{name}": component
                for name, component in vars(music3_components).items()
            },
            **{
                f"qwen_image21.{name}": component
                for name, component in vars(qwen_image21_components).items()
            },
            **{
                f"yue2.{name}": component
                for name, component in vars(yue2_components).items()
            },
            **{
                f"gallery.{name}": component
                for name, component in vars(gallery_components).items()
            },
        }
        if app_views.task is not None:
            browser_settings.update(
                {
                    "workspace.task": app_views.task,
                    "workspace.engine": navigation.engine_value,
                }
            )
        preferences = bind_browser_settings(
            demo, browser_settings, controller=settings_controller
        )
        if navigation is not None:
            preferences.h3_restore_event.success(
                navigation.restore,
                inputs=list(navigation.inputs),
                outputs=list(navigation.outputs),
                queue=True,
                concurrency_id="h3-settings",
                concurrency_limit=1,
                api_name=False,
                show_progress="hidden",
            )
    demo.h3_css = H3_SETUP_CSS + H3_WORKSPACE_CSS
    return demo
