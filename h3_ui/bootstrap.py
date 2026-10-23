"""Composition root with explicit catalogs and callbacks; no generation policy."""

from contextlib import nullcontext
import gradio as gr
from .app_bindings import bind_app
from .contracts import AppComponents, BootstrapCatalog, BootstrapServices
from .h3.services import H3ViewServices
from .h3.view import build_h3_view
from .layout import create_app_views, bind_workspace_navigation
from .persistence import (
    RESET_BROWSER_SETTINGS_JS,
    bind_browser_settings,
    create_browser_state,
)
from .styles import H3_SETUP_CSS, H3_WORKSPACE_CSS
from .api_view import build_api_view
from .media_view import build_gallery_view
from h3_ui.drama3.bindings import bind_drama3
from h3_ui.engines import FEATURES
from h3_ui.engines.controls import settings_components
from h3_ui.drama3.view import DRAMA3_SCRIPT_HIGHLIGHT_JS, DRAMA3_SUBTITLES_JS, build_drama3_view
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


def build_ui(catalog: BootstrapCatalog, services: BootstrapServices) -> gr.Blocks:
    audio_probe_cache.install()
    defaults = catalog.UI_DEFAULTS
    initial_backend = services.backend_status()
    with (
        gr.Blocks(title="Prism Studio", fill_width=True) as demo,
        gr.Column(elem_classes=["h3-workspace"]),
    ):
        with gr.Row(elem_classes=["h3-workspace-header"]):
            gr.HTML(
                '<header class="h3-hero"><span class="h3-brand-mark" aria-hidden="true">PS</span>'
                '<div><p class="h3-eyebrow">Video · Image · Music · Audio</p>'
                '<h1>Prism Studio</h1></div></header>',
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
        engine_actions = services.drama3_actions
        engine_views = {}
        for feature in FEATURES:
            engine_views[feature.id] = feature.build(getattr(app_views, feature.key), services.engine_deps)
            feature.bind(engine_views[feature.id], services.engine_deps)
        drama3_components = build_drama3_view(
            app_views.drama3,
            engine_actions.drama3_speaker_names(),
            getattr(engine_actions, "drama3_sound_names", lambda: [])(),
            engine_actions.drama3_speaker_groups(),
            (getattr(engine_actions, "drama3_speaker_group_size", None) or (lambda _group: None))("Global"),
        )
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
            downloads=True,
        )
        if app_views.dubbing is not None:
            from .dubbing.bindings import bind_dubbing
            from .dubbing.view import build_dubbing_view

            dubbing_components = build_dubbing_view(
                app_views.dubbing,
                gemini_models=catalog.GEMINI_PROMPT_MODELS,
                default_gemini_model=catalog.DEFAULT_GEMINI_PROMPT_MODEL,
            )
            bind_dubbing(dubbing_components, task=app_views.task, create_tab=app_views.create_tab,
                         interrupt=services.interrupt, actions=services.dubbing_actions)
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
            *(component for feature in FEATURES for component in feature.library_inputs(engine_views[feature.id])),
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
        for component in h3_components.values.values():
            if hasattr(component, "h3_library_picker") and not hasattr(component, "h3_reference_slot"):
                destinations[f"H3 · {component.label}"] = component
        for feature in FEATURES:
            destinations.update(feature.destinations_of(engine_views[feature.id]))
        build_media_image_destinations(
            gallery_components, destinations,
            validate_path=services.gallery_media_download_path,
        )
        for view in (*engine_views.values(), drama3_components):
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
                "memory_status": memory_status,
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
                projects_root=app_views.projects,
                jobs_tab=app_views.jobs_tab,
                other_tabs=tuple(tab for tab in app_views.main_tabs if tab is not app_views.jobs_tab),
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
                f"{feature.key}.{name}": component
                for feature in FEATURES
                for name, component in feature.settings_of(engine_views[feature.id]).items()
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
