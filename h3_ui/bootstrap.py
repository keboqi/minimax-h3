"""Composition root with explicit catalogs and callbacks; no generation policy."""

from dataclasses import dataclass
from contextlib import nullcontext
from typing import Any, Callable
import gradio as gr
from .app_bindings import bind_app
from .contracts import AppComponents
from .h3_view import build_h3_view, H3ViewServices
from .layout import create_app_views, bind_workspace_navigation
from .persistence import (
    RESET_BROWSER_SETTINGS_JS,
    bind_browser_settings,
    create_browser_state,
)
from .ltx_view import build_ltx_view
from .styles import H3_SETUP_CSS, H3_WORKSPACE_CSS
from .api_view import build_api_view
from .media_view import build_gallery_view
from .music_view import build_music_view
from .qwen_view import build_qwen_image21_view
from .yue2_view import build_yue2_view
from h3_ui.drama3_bindings import bind_drama3
from h3_ui.optional_engine_views import bind_flux3, build_flux3_view, bind_moss_image, build_moss_image_view
from h3_ui.fibo2_view import bind_fibo2, build_fibo2_view
from h3_ui.drama3_view import DRAMA3_SCRIPT_HIGHLIGHT_JS, DRAMA3_SUBTITLES_JS, build_drama3_view
from h3_ui.optional_engine_controls import settings_components
from h3_ui import audio_probe_cache

# Ctrl/Cmd+Enter presses the visible engine's generate button. Disabled
# buttons swallow the shortcut so it never falls through to other handlers.
GENERATE_SHORTCUT_JS = """
() => {
  if (window.h3GenerateShortcut) return;
  window.h3GenerateShortcut = true;
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || !(event.ctrlKey || event.metaKey) || event.repeat) return;
    const button = [...document.querySelectorAll("button.h3-primary-action")]
      .find((candidate) => candidate.offsetParent !== null);
    if (!button) return;
    event.preventDefault();
    if (!button.disabled) button.click();
  });
}
"""


@dataclass(frozen=True)
class BootstrapCatalog:
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
    optional_engine_actions: Any
    generation_readiness_state: Callable
    gallery_media_download_path: Callable
    image_vae_frame_updates: Callable
    import_gallery_media: Callable
    input_image_frame_preset_updates: Callable
    interrupt: Callable
    latent_upscale_layout_updates: Callable
    latent_upscale_method_layout_update: Callable
    list_media_paths: Callable
    mode_help: Callable
    postprocess_selected_gallery_media: Callable
    prepare_all_ltx25_official_models: Callable
    prepare_ltx25_official_workflow: Callable
    prompt_writer_backend_visibility: Callable
    reference_prompt_help: Callable
    refresh_backend_views: Callable
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
    audio_probe_cache.install()
    defaults = catalog.UI_DEFAULTS
    initial_backend = services.backend_status()
    with (
        gr.Blocks(title="MiniMax H3 Local", fill_width=True) as demo,
        gr.Column(elem_classes=["h3-workspace"]),
    ):
        with gr.Row(elem_classes=["h3-workspace-header"]):
            gr.HTML(
                '<header class="h3-hero"><span class="h3-brand-mark" aria-hidden="true">H3</span>'
                '<div><p class="h3-eyebrow">Creative workspace</p>'
                '<h1>MiniMax H3 Local</h1></div></header>',
                scale=3,
            )
            summary_root = gr.Column(scale=2, elem_classes=["h3-header-status"])
            with summary_root:
                system_summary = gr.HTML(
                    services.compact_backend_status(initial_backend)
                )
        app_views = create_app_views(summary_root=summary_root)
        with app_views.system if app_views is not None else nullcontext():
            gr.Markdown(
                "## System\nManage your backend, memory, and browser preferences. "
                '[Open ComfyUI ↗](/comfyui/)'
            )
            with gr.Accordion("System details and VRAM", open=False):
                with gr.Row(equal_height=True):
                    health = gr.Markdown(initial_backend)
                    unload_models = gr.Button("Unload all models / free VRAM", scale=0)
                memory_status = gr.Markdown()
            with gr.Row(equal_height=True):
                gr.Markdown(
                    "Settings are saved in this browser. Resetting clears them "
                    "and reloads the page with the app defaults."
                )
                gr.Button(
                    "Reset all settings to defaults",
                    variant="stop",
                    scale=0,
                    elem_id="h3-reset-browser-settings",
                ).click(None, js=RESET_BROWSER_SETTINGS_JS, api_name=False)
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
        engine_actions = services.optional_engine_actions
        flux3_components = build_flux3_view(app_views.flux3)
        drama3_components = build_drama3_view(
            app_views.drama3,
            engine_actions.drama3_speaker_names(),
            getattr(engine_actions, "drama3_sound_names", lambda: [])(),
            engine_actions.drama3_speaker_groups(),
            (getattr(engine_actions, "drama3_speaker_group_size", None) or (lambda _group: None))("Global"),
        )

        bind_flux3(flux3_components, engine_actions)
        moss_image_components = build_moss_image_view(app_views.moss_image)
        bind_moss_image(moss_image_components, engine_actions)
        fibo2_components = build_fibo2_view(app_views.fibo2)
        bind_fibo2(fibo2_components, engine_actions)
        bind_drama3(drama3_components, engine_actions)
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
        from .engine_readiness import bind_engine_readiness

        for view, family in (
            (ltx25_components, "ltx"),
            (qwen_image21_components, "qwen"),
            (music3_components, "music"),
            (yue2_components, "yue2"),
        ):
            bind_engine_readiness(view, family)
        with gallery_components.inspector:
            gallery_settings_used = gr.HTML("Select an output to inspect its settings.")
        from .library_tools import build_library_tools

        gallery_components.selected.h3_snapshot = (
            services.render_snapshot,
            gallery_settings_used,
        )
        build_library_tools(
            gallery_view,
            services.list_media_paths,
            view=gallery_components,
            system_root=app_views.system,
            validate_path=services.gallery_media_download_path,
            preview_media=services.refresh_media_page,
        )
        from .image_library import bind_image_library, build_media_image_destinations

        image_inputs = [
            *h3_components.values.values(),
            *vars(ltx25_components).values(),
            *vars(qwen_image21_components).values(),
            *music3_components.reference_images,
            flux3_components.reference_images,
            fibo2_components.reference_images,
        ]
        for component in image_inputs:
            if hasattr(component, "h3_library_picker"):
                bind_image_library(
                    component,
                    index=gallery_components.search.h3_asset_index,
                    refresh_page=services.refresh_media_page,
                    validate_path=services.gallery_media_download_path,
                )
        destinations = {
            "H3 · Reference images": tuple(
                component for component in h3_components.values.values()
                if hasattr(component, "h3_reference_slot")
            ),
        }
        for family, inputs in (
            ("H3", h3_components.values.values()),
            ("LTX", vars(ltx25_components).values()),
            ("Qwen", vars(qwen_image21_components).values()),
            ("FLUX 3", vars(flux3_components).values()),
            ("Fibo 2.0", vars(fibo2_components).values()),
        ):
            for component in inputs:
                if hasattr(component, "h3_library_picker") and not hasattr(component, "h3_reference_slot"):
                    destinations[f"{family} · {component.label}"] = component
        destinations["Music · Reference images"] = music3_components.reference_images
        build_media_image_destinations(
            gallery_components, destinations,
            validate_path=services.gallery_media_download_path,
        )
        for root, view in (
            (ltx25_view, ltx25_components),
            (music3_view, music3_components),
            (qwen_image21_view, qwen_image21_components),
            (yue2_view, yue2_components),
            (app_views.flux3, flux3_components),
            (app_views.moss_image, moss_image_components),
            (app_views.fibo2, fibo2_components),
            (app_views.drama3, drama3_components),
        ):
            with view.output.h3_metadata_root:
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
        browser_state = create_browser_state()
        settings_controller = bind_app(
            AppComponents.from_mapping(app_components),
            services,
            preferences=browser_state,
        )
        navigation = bind_workspace_navigation(
            app_views,
            h3_components.result_format,
            settings_controller,
            format_layout=services.result_format_layout_updates,
            preferences=browser_state,
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
                f"flux3.{name}": component
                for name, component in settings_components(flux3_components).items()
            },
            **{
                f"moss_image.{name}": component
                for name, component in settings_components(moss_image_components).items()
            },
            **{
                f"fibo2.{name}": component
                for name, component in settings_components(fibo2_components).items()
            },
            **{
                f"drama3.{name}": component
                for name, component in settings_components(drama3_components).items()
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
        bind_browser_settings(
            demo,
            browser_settings,
            controller=settings_controller,
            browser_state=browser_state,
            navigation=navigation,
        )
        demo.load(None, js=GENERATE_SHORTCUT_JS, api_name=False)
        demo.load(None, js=DRAMA3_SUBTITLES_JS, api_name=False)
        demo.load(None, js=DRAMA3_SCRIPT_HIGHLIGHT_JS, api_name=False)
    demo.h3_css = H3_SETUP_CSS + H3_WORKSPACE_CSS
    return demo
