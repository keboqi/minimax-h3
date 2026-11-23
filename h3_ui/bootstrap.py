"""Composition root with explicit catalogs and callbacks; no generation policy."""

from functools import partial
from types import SimpleNamespace
import gradio as gr
from .api_bindings import bind_default_video_api
from .app_bindings import bind_app
from .contracts import AppComponents, BootstrapCatalog, BootstrapServices
from .events.system import bind_system
from .h3.services import H3ViewServices
from .h3.view import build_h3_view
from .layout import create_app_views, bind_workspace_navigation
from .persistence import (
    bind_browser_settings,
    create_browser_state,
)
from .system_view import build_readiness_view, build_system_view
from .styles import H3_SETUP_CSS, H3_WORKSPACE_CSS
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
# current task, opens Create from Media, Jobs or any other tab and switches the
# active Create page immediately without waiting for a server round trip.
TASK_OPENS_CREATE_JS = """
() => {
  if (window.h3TaskOpensCreate) return;
  window.h3TaskOpensCreate = true;
  const TAB_FOR_TASK = {
    "Image": "image",
    "Audio / Music": "song",
    "Remake": "remake",
    "Storyboard": "storyboard",
    "TTS": "drama3",
    "Dub": "dub",
  };
  const VIDEO_TABS = new Set(["h3", "ltx"]);
  const currentVideoEngine = () => {
    const checked = document.querySelector('.h3-workspace .h3-engine-bar input[type="radio"]:checked');
    if (checked && VIDEO_TABS.has(checked.value)) {
      window.__h3VideoEngine = checked.value;
      return checked.value;
    }
    const activeTab = document.querySelector('#h3-engine-tabs [role="tab"][aria-selected="true"]');
    const tabId = activeTab?.getAttribute("data-tab-id");
    if (tabId && VIDEO_TABS.has(tabId)) {
      window.__h3VideoEngine = tabId;
      return tabId;
    }
    return window.__h3VideoEngine || "h3";
  };
  const selectTab = (selector) => {
    const tab = document.querySelector(selector);
    if (tab && tab.getAttribute("aria-selected") !== "true") {
      tab.click();
      return true;
    }
    return false;
  };
  const selectEngineTab = (tabId) => {
    if (!tabId) return false;
    window.__h3DesiredEngineTab = tabId;
    return selectTab(`#h3-engine-tabs [role="tab"][data-tab-id="${tabId}"]`);
  };
  let lastInputTarget = null;
  document.addEventListener("input", (event) => {
    lastInputTarget = event.target;
  }, true);
  let pendingTaskValue = null;
  let pendingObserver = null;
  const watchPendingTask = () => {
    if (pendingObserver) return;
    const root = document.querySelector(".h3-workspace .h3-task-picker");
    if (!root) return;
    pendingObserver = new MutationObserver(() => {
      if (!pendingTaskValue) return;
      const input = [...root.querySelectorAll('input[type="radio"]')].find((item) => item.value === pendingTaskValue);
      if (!input || input.disabled) return;
      const value = pendingTaskValue;
      pendingTaskValue = null;
      const targetTab = value === "Video" ? currentVideoEngine() : TAB_FOR_TASK[value];
      selectTab('#h3-main-tabs [role="tab"][data-tab-id="create"]');
      selectEngineTab(targetTab);
      if (!input.checked) input.click();
      else input.dispatchEvent(new Event("input", {bubbles: true}));
    });
    pendingObserver.observe(root, {subtree: true, attributes: true, attributeFilter: ["disabled"]});
  };
  document.addEventListener("click", (event) => {
    const engineLabel = event.target.closest?.(".h3-workspace .h3-engine-bar label");
    if (engineLabel) {
      const engineInput = engineLabel.querySelector('input[type="radio"]');
      if (engineInput && VIDEO_TABS.has(engineInput.value)) {
        window.__h3VideoEngine = engineInput.value;
        const taskInput = document.querySelector('.h3-workspace .h3-task-picker input[type="radio"]:checked');
        if (!taskInput || taskInput.value === "Video") {
          selectEngineTab(engineInput.value);
        }
      }
      return;
    }
    const label = event.target.closest?.(".h3-workspace .h3-task-picker label");
    if (!label) return;
    selectTab('#h3-main-tabs [role="tab"][data-tab-id="create"]');
    const input = label.querySelector('input[type="radio"]');
    if (!input) return;
    const value = input.value;
    const targetTab = value === "Video" ? currentVideoEngine() : TAB_FOR_TASK[value];
    if (input.disabled) {
      pendingTaskValue = value;
      watchPendingTask();
      selectEngineTab(targetTab);
      return;
    }
    pendingTaskValue = null;
    const switchedTab = selectEngineTab(targetTab);
    const sawInput = lastInputTarget === input;
    lastInputTarget = null;
    queueMicrotask(() => {
      const fired = sawInput || lastInputTarget === input;
      lastInputTarget = null;
      if (input.disabled) return;
      if (!input.checked) input.click();
      else if (switchedTab && !fired) input.dispatchEvent(new Event("input", {bubbles: true}));
    });
  });
}
"""


def build_ui(
    catalog: BootstrapCatalog,
    services: BootstrapServices,
    *,
    build_models=None,
    on_workspace_import=None,
) -> gr.Blocks:
    audio_probe_cache.install()
    defaults = catalog.UI_DEFAULTS
    initial_backend = services.backend_status()
    initial_summary = services.compact_backend_status(initial_backend)
    with (
        gr.Blocks(title="Prism Studio", fill_width=True) as demo,
        gr.Column(elem_classes=["h3-workspace"]),
    ):
        # The top bar: brand, Create tasks, the main tab row (placed by CSS), status, and engine switcher.
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
                system_summary = gr.HTML(initial_summary)
            engine_root = gr.Column(scale=0, min_width=0, elem_classes=["h3-engine-bar"])
        app_views = create_app_views(
            summary_root=summary_root,
            task_root=task_root,
            engine_root=engine_root,
            include_models=build_models is not None,
        )
        system_view = build_system_view(app_views.system, initial_backend, initial_summary)
        from .project_transfer import (
            build_project_import_view,
            build_project_manage_view,
            install_project_transfer_routes,
        )
        from h3_app.jobs import JOBS as _JOBS

        build_project_manage_view(
            app_views.system,
            store=_JOBS.store,
            runtime_config=services.engine_deps.runtime_config,
            on_changed=on_workspace_import,
        )
        build_project_import_view(
            app_views.system,
            store=_JOBS.store,
            runtime_config=services.engine_deps.runtime_config,
            on_imported=on_workspace_import,
        )
        demo.h3_routes = (
            *getattr(demo, "h3_routes", ()),
            partial(
                install_project_transfer_routes,
                store=_JOBS.store,
                runtime_config=services.engine_deps.runtime_config,
                on_imported=on_workspace_import,
            ),
        )
        generation_view = app_views.generation
        gallery_view = app_views.gallery
        gallery_tab = app_views.gallery_tab
        h3_components = build_h3_view(
            generation_view,
            defaults,
            H3ViewServices.from_bootstrap(catalog, services),
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

        from .engines.remake import EditTarget, bind_h3_settings, media_targets as remake_media_targets

        image_pages = []

        def hand_to_image(engine, owner, path):
            image_pages[0].receive(owner, engine, path)

        bind_h3_settings(
            engine_views["remake"], engine_handlers["remake"], h3_components.values,
            # The MiniMax H3 page hands Remake its settings; nothing is captured from Gradio.
            (),
            # The Image page takes the frame; its Gradio view stays unseen.
            edit_targets={engine: EditTarget(receive=partial(hand_to_image, engine)) for engine in ("h3", "qwen", "gpt_image")},
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
                music=engine_handlers.get("music"),
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
        from .h3.page import H3PageServices, H3PageState, install_h3_page
        from .image.page import ImagePageState, install_image_page
        from .ltx.page import LtxPageState, install_ltx_page, media_targets as ltx_media_targets
        from .song.page import SongPageState, install_song_page

        remake_library = lambda: engine_handlers["remake"].library
        followers = tuple(handlers for handlers in (engine_handlers["remake"], services.storyboard_handlers)
                          if handlers is not None)
        h3_page_services = H3PageServices.from_bootstrap(
            catalog, services, library=remake_library, followers=followers,
        )
        demo.h3_routes = (*demo.h3_routes, partial(install_h3_page, services=h3_page_services))
        # Reads what the MiniMax H3 page saved: P-H3-Image runs on that page's settings.
        h3_saved = H3PageState(h3_page_services)
        image_pages.append(
            ImagePageState.from_bootstrap(
                services, image_controller, engine_views, engine_handlers,
                h3_saved=h3_saved, library=remake_library, followers=(video_export,),
            )
        )
        demo.h3_routes = (*demo.h3_routes, partial(install_image_page, pages=image_pages[0]))
        song_pages = SongPageState.from_bootstrap(
            services, engine_views, engine_handlers, library=remake_library,
        )
        demo.h3_routes = (*demo.h3_routes, partial(install_song_page, pages=song_pages))
        ltx_pages = LtxPageState.from_bootstrap(
            services, engine_views["ltx"], engine_handlers["ltx"], library=remake_library,
        )
        demo.h3_routes = (*demo.h3_routes, partial(install_ltx_page, pages=ltx_pages))
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
            from .dubbing.share import install_share_routes
            from .dubbing.tab import build_dubbing_tab

            dubbing_gpu = services.dubbing_actions.services.gpu
            build_dubbing_tab(app_views.dubbing, services.dubbing_actions, gpu=dubbing_gpu)
            demo.h3_routes = (*getattr(demo, "h3_routes", ()), partial(
                install_dubbing_page, handlers=services.dubbing_actions, gpu=dubbing_gpu,
                interrupt=services.interrupt,
                h3_values=(lambda owner: h3_saved.request_values(h3_saved.load(owner))) if dubbing_gpu else None))
            if not dubbing_gpu:
                demo.h3_routes = (*demo.h3_routes, partial(
                    install_share_routes, runtime=services.dubbing_actions.services.runtime_config()))
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
            system_root=system_view.maintenance,
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
            # Remake's pictures are delivered to its page directly, below.
            if feature.id not in {"qwen", "flux3", "moss_image", "fibo2", "music", "remake", "ltx"}:
                destinations.update(feature.destinations_of(engine_views[feature.id]))
        # The Image page has its own Media library picker, as MiniMax H3's does.
        set_as = build_media_image_destinations(
            gallery_components, destinations,
            validate_path=services.gallery_media_download_path,
            page_targets={**remake_media_targets(engine_handlers["remake"]),
                          "Song · Visual inspiration": song_pages.receive, **ltx_media_targets(ltx_pages)},
            apply_on=[app_views.create_tab.select] if app_views.create_tab is not None and destinations else [],
        )
        from .media.page import MediaPageState, install_media_page

        # The Media page browses, annotates, deletes and enhances through the services its Gradio view used.
        media_pages = MediaPageState.from_bootstrap(
            services,
            index=lambda: gallery_components.search.h3_asset_index,
            set_as=lambda: set_as,
            h3_saved=h3_saved,
            seedvr2_default=defaults["seedvr2_model"],
        )
        demo.h3_routes = (*demo.h3_routes, partial(install_media_page, pages=media_pages))
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
        bind_default_video_api(services.generate_with_ui_defaults)
        app_components = dict(h3_components.values)
        app_components.update(
            {
                "gallery_components": gallery_components,
                "gallery_tab": gallery_tab,
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
        from h3_models import MODEL_SPECS

        build_readiness_view(
            app_views.system,
            model_specs=MODEL_SPECS,
            models_root=catalog.COMFY_DIR / "models",
        )
        from .system.page import SystemPageState, install_system_page

        system_pages = SystemPageState.from_bootstrap(
            catalog,
            services,
            store=_JOBS.store,
            index=lambda: gallery_components.search.h3_asset_index,
            model_specs=MODEL_SPECS,
            on_changed=on_workspace_import,
        )
        demo.h3_routes = (*demo.h3_routes, partial(install_system_page, pages=system_pages))
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
        bind_system(
            demo, system_view, summary=system_summary,
            create_refresh=h3_components.refresh, services=services,
        )
        demo.load(None, js=GENERATE_SHORTCUT_JS, api_name=False)
        demo.load(None, js=TASK_OPENS_CREATE_JS, api_name=False)
        demo.load(None, js=REFERENCE_TAG_JS, api_name=False)
        load_result_view_scripts(demo)
        if build_models is not None and app_views.models is not None:
            with app_views.models:
                build_models(demo)
    demo.h3_css = H3_SETUP_CSS + H3_WORKSPACE_CSS
    return demo
