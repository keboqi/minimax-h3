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
from h3_ui.engines import FEATURES
from h3_ui import audio_probe_cache
from h3_ui.image_library import REFERENCE_TAG_JS
from h3_ui.result_views import load_result_view_scripts

# Ctrl/Cmd+Enter presses the visible engine's generate button. Disabled or
# busy buttons swallow the shortcut so it never falls through to other handlers.
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
    if (!button.disabled && !button.classList.contains("h3-busy")) button.click();
  });
}
"""

# The top bar's tasks stand in for the hidden Create tab: picking one, even the
# current task, opens Create from Media, Jobs or any other tab.
TASK_OPENS_CREATE_JS = """
() => {
  if (window.h3TaskOpensCreate) return;
  window.h3TaskOpensCreate = true;
  document.addEventListener("click", (event) => {
    if (!event.target.closest(".h3-workspace-header .h3-task-picker label")) return;
    const create = document.querySelector('#h3-main-tabs [role="tab"][data-tab-id="create"]');
    if (create && create.getAttribute("aria-selected") !== "true") create.click();
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
        # The top bar: brand, Create tasks, then the main tab row (placed by CSS), then status.
        with gr.Row(elem_classes=["h3-workspace-header"]):
            gr.HTML(
                '<header class="h3-hero"><span class="h3-brand-mark" aria-hidden="true">PS</span>'
                '<h1>Prism Studio</h1></header>',
                scale=0,
                min_width=0,
            )
            task_root = gr.Column(scale=0, min_width=0, elem_classes=["h3-task-picker"])
            summary_root = gr.Column(scale=0, min_width=0, elem_classes=["h3-header-status"])
            with summary_root:
                system_summary = gr.HTML(
                    services.compact_backend_status(initial_backend)
                )
        app_views = create_app_views(summary_root=summary_root, task_root=task_root)
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
                H3_LATENT_UPSCALER_MODEL_CHOICES=catalog.H3_LATENT_UPSCALER_MODEL_CHOICES,
                H3_TEXT_ENCODER_CHOICES=catalog.H3_TEXT_ENCODER_CHOICES,
                IMAGE_VAE_CHOICES=catalog.IMAGE_VAE_CHOICES,
                INPUT_IMAGE_FRAME_PRESETS=catalog.INPUT_IMAGE_FRAME_PRESETS,
                INPUT_IMAGE_UPSCALE_SLOTS=catalog.INPUT_IMAGE_UPSCALE_SLOTS,
                LARGE_RESOLUTIONS=catalog.LARGE_RESOLUTIONS,
                OPENAI_PROMPT_MODEL=catalog.OPENAI_PROMPT_MODEL,
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
        engine_handlers = {}
        for feature in FEATURES:
            engine_views[feature.id] = feature.build(getattr(app_views, feature.key), services.engine_deps)
            engine_handlers[feature.id] = feature.bind(engine_views[feature.id], services.engine_deps)
        from .image_workspace import build_image_workspace

        image_workspace, image_controller = build_image_workspace(
            app_views.image, app_views, engine_views, engine_handlers, h3_components,
            services.engine_deps, demo,
        )
        from functools import partial

        from .engines.remake import EditTarget, bind_h3_settings

        image_pages = []

        def hand_to_image(engine, owner, path):
            image_pages[0].receive(owner, engine, path)

        bind_h3_settings(
            engine_views["remake"], engine_handlers["remake"], h3_components.values,
            # The MiniMax H3 page hands Remake its settings; nothing is captured from Gradio.
            (),
            # The Image page takes the frame; its Gradio view stays unseen.
            edit_targets={engine: EditTarget(receive=partial(hand_to_image, engine)) for engine in ("h3", "qwen")},
        )

        from .remake_page import install_remake_page

        demo.h3_routes = (*getattr(demo, "h3_routes", ()),
                          partial(install_remake_page, handlers=engine_handlers["remake"], interrupt=services.interrupt))
        # Soundstage is its own page; the tab shows it and keeps the API endpoints.
        from .drama3.tab import build_soundstage_tab
        from .drama3.visuals import video_export as soundstage_video_export

        build_soundstage_tab(app_views.drama3, engine_actions)
        # The Image page hands the export its settings; nothing is captured from Gradio.
        video_export = soundstage_video_export(image_controller, services.engine_deps)
        if services.soundstage_workspace is not None:
            from h3_app.drama3.serving.takes import takes_root
            from .drama3.page import install_soundstage_page

            store, output_dir = services.soundstage_workspace
            demo.h3_routes = (*demo.h3_routes, partial(
                install_soundstage_page, actions=engine_actions, store=store, output_dir=output_dir,
                takes_dir=lambda: takes_root(services.engine_deps.runtime_config()),
                runtime_config=lambda: services.engine_deps.runtime_config(), interrupt=services.interrupt,
                video_export=video_export))
        if app_views.storyboard is not None and services.storyboard_handlers is not None:
            from h3_ui.storyboard.tab import build_storyboard_tab

            build_storyboard_tab(
                app_views.storyboard,
                demo,
                services.storyboard_handlers,
                h3_values=h3_components.values,
                # The MiniMax H3 page hands Storyboard its settings; nothing is captured from Gradio.
                triggers=(),
                interrupt=services.interrupt,
            )
        from .h3.page import H3PageServices, install_h3_page

        followers = tuple(handlers for handlers in (engine_handlers["remake"], services.storyboard_handlers)
                          if handlers is not None)
        h3_page_services = H3PageServices(
            root=lambda: services.engine_deps.runtime_config().outputs_dir / ".h3-page",
            resolve=services.resolve_request_settings,
            describe=services.describe_settings,
            enhance=services.enhance_h3_prompt,
            interrupt=services.interrupt,
            compile_trt=services.compile_trt_video_vae,
            library=lambda: engine_handlers["remake"].library,
            followers=followers,
            attention_backend=catalog.SERVER_ATTENTION_BACKEND,
        )
        demo.h3_routes = (*demo.h3_routes, partial(install_h3_page, services=h3_page_services))
        from .h3.page import H3PageState
        from .image.page import ImagePageServices, ImagePageState, install_image_page, private_image_root
        from .page_fields import spec_of
        from .image_controller import suggest_prompt
        from h3_app.h3_image import H3_IMAGE_16M_PRESETS
        from h3_app.catalog import H3_IMAGE_4M_RESOLUTION_PRESETS, H3_IMAGE_8M_RESOLUTION_PRESETS

        # Reads what the MiniMax H3 page saved: P-H3-Image runs on that page's settings.
        h3_saved = H3PageState(h3_page_services)
        image_fields = {name: component for name, component in image_controller.fields.items()
                        if not name.startswith("h3__")}
        # Shown with the model's options, though no generation takes them by name.
        image_fields["qwen__preset"] = engine_views["qwen"].preset
        image_fields["moss_image__rewrite_mode"] = engine_views["moss_image"].rewrite_mode
        from h3_app.jobs import JOBS

        image_root = private_image_root(services.engine_deps.runtime_config(), JOBS.store.root)
        image_pages.append(ImagePageState(ImagePageServices(
            root=lambda: image_root,
            fields={name: spec_of(component) for name, component in image_fields.items()},
            job_fields=tuple(image_controller.fields),
            presets={
                **{engine: tuple(tuple(value for _label, value in dropdown.choices) for dropdown in pair)
                   for engine, pair in {
                       "qwen": (engine_views["qwen"].output_resolution_1k, engine_views["qwen"].output_resolution_2k),
                       **{engine: (engine_views[engine].resolution_small, engine_views[engine].resolution_large)
                          for engine in ("flux3", "moss_image", "fibo2")},
                   }.items()},
                "h3": (H3_IMAGE_4M_RESOLUTION_PRESETS, H3_IMAGE_8M_RESOLUTION_PRESETS, H3_IMAGE_16M_PRESETS),
            },
            suggest=partial(suggest_prompt, engine_handlers, runtime=services.engine_deps.runtime_config()),
            h3_values=lambda owner: h3_saved.request_values(h3_saved.load(owner)),
            h3_defaults={name: getattr(component, "value", None)
                         for name, component in image_controller.h3_fields.items()},
            interrupt=services.interrupt,
            library=lambda: engine_handlers["remake"].library,
            followers=(video_export,),
        )))
        demo.h3_routes = (*demo.h3_routes, partial(install_image_page, pages=image_pages[0]))
        from .song.page import SongPageServices, SongPageState, fields_of, install_song_page, private_song_root

        # The Song page takes each model's options, their defaults and their limits from its unseen Gradio view.
        song_pages = SongPageState(SongPageServices(
            root=lambda: private_song_root(services.engine_deps.runtime_config(), JOBS.store.root),
            fields=fields_of(engine_views),
            enhance={"music": engine_handlers["music"].enhance_music3_prompt,
                     "yue2": engine_handlers["yue2"].enhance_yue2_prompt},
            take_facts={engine: engine_handlers[engine].take_facts for engine in ("music", "yue2")},
            interrupt=services.interrupt,
            library=lambda: engine_handlers["remake"].library,
        ))
        demo.h3_routes = (*demo.h3_routes, partial(install_song_page, pages=song_pages))
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
            # Dubbing is its own page; the tab shows it and keeps the API endpoints.
            from .dubbing.page import install_dubbing_page
            from .dubbing.tab import build_dubbing_tab

            build_dubbing_tab(app_views.dubbing, services.dubbing_actions)
            demo.h3_routes = (*getattr(demo, "h3_routes", ()), partial(
                install_dubbing_page, handlers=services.dubbing_actions, interrupt=services.interrupt))
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
            image_workspace.reference_images,
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
        if services.storyboard_handlers is not None:
            from h3_ui.storyboard.handlers import MediaLibrary

            services.storyboard_handlers.library = MediaLibrary(
                gallery_components.search.h3_asset_index,
                services.refresh_media_page,
                services.gallery_media_download_path,
            )
            engine_handlers["remake"].library = services.storyboard_handlers.library
        # MiniMax H3's inputs live on its page, which has its own Media library picker; so do Image's and Song's.
        destinations = {}
        for feature in FEATURES:
            if feature.id not in {"qwen", "flux3", "moss_image", "fibo2", "music"}:
                destinations.update(feature.destinations_of(engine_views[feature.id]))
        # The Image page has its own Media library picker, as MiniMax H3's does.
        build_media_image_destinations(
            gallery_components, destinations,
            validate_path=services.gallery_media_download_path,
            page_targets={"Song · Visual inspiration": song_pages.receive},
        )
        for view in (*engine_views.values(), image_workspace):
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
            **{f"image.{name}": component
               for name, component in image_workspace.preference_components().items()},
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
        # The UI can open before ComfyUI's API: each page load shows the backend as it is now.
        demo.load(services.refresh_backend_views, outputs=[system_summary, health], queue=False,
                  show_progress="hidden", api_name=False)
        demo.load(None, js=GENERATE_SHORTCUT_JS, api_name=False)
        demo.load(None, js=TASK_OPENS_CREATE_JS, api_name=False)
        demo.load(None, js=REFERENCE_TAG_JS, api_name=False)
        load_result_view_scripts(demo)
    demo.h3_css = H3_SETUP_CSS + H3_WORKSPACE_CSS
    return demo
