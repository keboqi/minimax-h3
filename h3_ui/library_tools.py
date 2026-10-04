"""Persistent asset annotations, lineage and CPU-only comparison controls."""

from html import escape
from pathlib import Path
from urllib.parse import quote
import json

import gradio as gr

from h3_app.jobs import JOBS
from h3_app.errors import H3Error

VIDEO_JS = """
const setup = () => {
 const a = element.querySelector('[data-video-a]'), b = element.querySelector('[data-video-b]');
 if (!a || !b || a.dataset.bound) return;
 a.dataset.bound = 'true';
 const play = element.querySelector('[data-play]'), seek = element.querySelector('[data-seek]');
 const audio = element.querySelector('[data-audio]'), state = element.querySelector('[data-state]');
 let syncing = false;
 const end = () => Math.min(a.duration || 0, b.duration || 0);
 const pause = () => { a.pause(); b.pause(); play.textContent = 'Play both'; };
 const mute = () => { a.muted = audio.value !== 'a'; b.muted = audio.value !== 'b'; };
 audio.addEventListener('change', mute); mute();
 play.addEventListener('click', async () => {
   if (!a.paused) { pause(); return; }
   b.currentTime = a.currentTime;
   try { await Promise.all([a.play(), b.play()]); play.textContent = 'Pause both'; }
   catch (_) { pause(); state.textContent = 'Playback unavailable. Check browser codec support.'; }
 });
 seek.addEventListener('input', () => { pause(); a.currentTime = b.currentTime = Number(seek.value); });
 for (const v of [a,b]) {
   v.addEventListener('loadedmetadata', () => { seek.max = end(); state.textContent = 'Shared timeline: ' + end().toFixed(2) + ' seconds. Audio: one source only.'; });
   v.addEventListener('waiting', pause);
   v.addEventListener('ended', pause);
 }
 a.addEventListener('timeupdate', () => {
   if (syncing) return;
   syncing = true;
   if (a.currentTime >= end()) pause();
   if (Math.abs(a.currentTime - b.currentTime) > 0.08) b.currentTime = a.currentTime;
   seek.value = a.currentTime;
   syncing = false;
 });
};
new MutationObserver(setup).observe(element, {childList:true, subtree:true}); setup();
"""


def comparison_html(left, right):
    def url(path):
        return "/gradio_api/file=" + quote(
            str(Path(path).resolve()).replace("\\", "/"), safe="/:="
        )

    return (
        '<div class="h3-compare-video"><div class="h3-compare-pair">'
        f'<figure><figcaption>A · {escape(Path(left).name)}</figcaption><video data-video-a preload="metadata" playsinline src="{escape(url(left), quote=True)}"></video></figure>'
        f'<figure><figcaption>B · {escape(Path(right).name)}</figcaption><video data-video-b preload="metadata" playsinline src="{escape(url(right), quote=True)}"></video></figure></div>'
        '<button type="button" data-play>Play both</button> '
        '<label>Comparison time <input data-seek type="range" min="0" max="0" step="0.01" value="0"></label> '
        '<label>Audio <select data-audio><option value="mute">Muted</option><option value="a">A only</option><option value="b">B only</option></select></label>'
        '<p data-state role="status" aria-live="polite">Loading media timing…</p></div>'
    )


def build_library_tools(root, list_paths, *, view, system_root, validate_path):
    from .asset_index import AssetIndex

    index = AssetIndex(JOBS.store, list_paths)
    view.search.h3_asset_index = index
    with view.inspector:
        gr.Markdown(
            "Tags and favorites are shared library metadata. Search matches filenames, "
            "tags and technical settings; prompts are excluded."
        )
        with gr.Accordion("Tags & lineage", open=True):
            with gr.Row():
                tags = gr.Textbox(label="Tags (comma separated)")
                favorite = gr.Checkbox(label="Favorite")
            save = gr.Button("Save asset annotations")
            status = gr.Markdown()
            with gr.Accordion("Asset identity and lineage", open=False, max_height=400):
                lineage = gr.JSON(label="Asset identity and lineage")
    comparison_open = gr.State(False)
    with root, gr.Accordion("Compare two outputs", open=False) as comparison:
        gr.Markdown(
            "Images use a reveal slider. Videos share elapsed playback time up to the "
            "shorter duration, with drift correction. Buffering pauses both players. "
            "Audio is muted by default; choose A or B. Choose either output from the "
            "managed image/video library, including outside the current search."
        )
        with gr.Row():
            left = gr.Dropdown([], label="Comparison A", value=None)
            right = gr.Dropdown([], label="Comparison B", value=None)
            compare = gr.Button("Compare selected outputs")
        images = gr.ImageSlider(
            label="Image comparison A / B",
            type="filepath",
            interactive=False,
            visible=False,
        )
        videos = gr.HTML("", js_on_load=VIDEO_JS)
    with system_root, gr.Accordion("Media index maintenance", open=False):
        gr.Markdown(
            "The library indexes new and changed files when you browse it. Rebuild after repairing sidecar metadata."
        )
        rebuild = gr.Button("Rebuild media index")
        index_status = gr.Markdown()

    rebuild.click(
        index.rebuild,
        outputs=index_status,
        concurrency_id="h3-media-index",
        concurrency_limit=1,
        api_name=False,
    )

    def is_managed(path, mode):
        try:
            validate_path(path, mode)
            return Path(path).is_file()
        except (H3Error, OSError, ValueError):
            return False

    def resolve(asset_id):
        with JOBS.store.connect() as db:
            row = db.execute(
                "SELECT a.*, COALESCE(n.tags, '[]') AS tags, COALESCE(n.favorite,0) AS favorite "
                "FROM assets a LEFT JOIN annotations n ON n.asset_id=a.id WHERE a.id=?",
                (asset_id,),
            ).fetchone()
        if row is None or not is_managed(row["path"], row["kind"]):
            raise gr.Error("This asset is unavailable in the current managed library.")
        return dict(row)

    def selected_id(mode, path):
        if not path or not is_managed(path, mode):
            return None
        with JOBS.store.connect() as db:
            row = db.execute(
                "SELECT id FROM assets WHERE path=?", (str(Path(path).resolve()),)
            ).fetchone()
        return row["id"] if row else None

    def inspect_asset(mode, path):
        asset_id = selected_id(mode, path)
        if not asset_id:
            return "", False, None, gr.update(value=None)
        row = resolve(asset_id)
        return (
            ", ".join(json.loads(row["tags"])),
            bool(row["favorite"]),
            {
                "asset_id": asset_id,
                "filename": Path(row["path"]).name,
                "settings_and_lineage": json.loads(row["metadata"]),
            },
            gr.update(value=asset_id if row["kind"] in {"Image", "Video"} else None),
        )

    def comparison_choices(a, b, current, *, refresh=False):
        # Comparison B should remain available outside a filtered thumbnail
        # page. Keep its inventory independent of search/type transitions;
        # replacing an open native dropdown's choices can race its filtering.
        if refresh:
            index.sync("Image")
            index.sync("Video")
        revised = current.get("comparison_revision") != index.revision
        choices = current.get("comparison_choices", [])
        if revised or refresh:
            allowed = index.cached_paths("Image") | index.cached_paths("Video")
            rows = [
                row
                for row in JOBS.store.search_assets(limit=None)
                if row["kind"] in {"Image", "Video"} and Path(row["path"]) in allowed
            ]
            choices = [
                (
                    row["kind"]
                    + " · "
                    + Path(row["path"]).name
                    + " · "
                    + row["id"][:8],
                    row["id"],
                )
                for row in rows
            ]
        ids = {item[1] for item in choices}
        changed = refresh or current.get("comparison_choices") != choices
        current["comparison_choices"] = choices
        current["comparison_revision"] = index.revision

        # An empty captured value can precede a thumbnail selection. Updating
        # choices alone preserves that newer selection when responses overlap.
        def update(current):
            props = {"choices": choices} if changed else {}
            if current is not None and current not in ids:
                props["value"] = None
            return gr.update(**props)

        return update(a), update(b)

    def open_comparison(a, b, current):
        return *comparison_choices(a, b, current, refresh=True), True

    comparison.expand(
        open_comparison,
        inputs=[left, right, view.filters],
        outputs=[left, right, comparison_open],
        queue=False,
        show_progress="minimal",
        api_name=False,
    )
    comparison.collapse(
        lambda: False,
        outputs=comparison_open,
        queue=False,
        show_progress="hidden",
        api_name=False,
    )

    view.mode.change(
        lambda: (None, None, gr.update(visible=False), ""),
        outputs=[left, right, images, videos],
        queue=False,
        show_progress="hidden",
        api_name=False,
    )

    snapshot, snapshot_component = view.selected.h3_snapshot

    def synchronize(mode, paths, selected, a, b, current, comparing=False):
        inspected = inspect_asset(mode, selected)
        if comparing:
            choices_a, choices_b = comparison_choices(a, b, current)
        else:
            # A follows the selected thumbnail even before comparison is opened.
            identity = inspected[2]
            choice = (
                [(mode + " · " + identity["filename"], identity["asset_id"])]
                if identity
                else []
            )
            choices_a, choices_b = gr.update(choices=choice), gr.skip()
        return (
            *inspected[:3],
            {**choices_a, **inspected[3]},
            choices_b,
            snapshot(selected),
        )

    view.selected.h3_sync = (
        synchronize,
        [
            view.mode,
            view.paths,
            view.selected,
            left,
            right,
            view.filters,
            comparison_open,
        ],
        [tags, favorite, lineage, left, right, snapshot_component],
    )

    def annotate(mode, path, text, starred):
        asset_id = selected_id(mode, path)
        if not asset_id:
            raise gr.Error("Select an item in the Media library first.")
        resolve(asset_id)
        JOBS.store.annotate(asset_id, tags=text.split(","), favorite=starred)
        return "Asset annotations saved."

    save.click(
        annotate,
        inputs=[view.mode, view.selected, tags, favorite],
        outputs=status,
        queue=False,
        api_name=False,
    )

    def compare_assets(a, b):
        first, second = resolve(a), resolve(b)
        if first["kind"] != second["kind"] or first["kind"] not in {"Image", "Video"}:
            raise gr.Error("Select two images or two videos to compare.")
        if first["kind"] == "Image":
            return gr.update(value=(first["path"], second["path"]), visible=True), ""
        return gr.update(visible=False), comparison_html(first["path"], second["path"])

    compare.click(
        compare_assets,
        inputs=[left, right],
        outputs=[images, videos],
        queue=False,
        api_name=False,
    )
