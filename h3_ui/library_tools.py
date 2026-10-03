"""Persistent asset annotations, lineage and CPU-only comparison controls."""

from html import escape
from pathlib import Path
from urllib.parse import quote
import json

import gradio as gr

from h3_app.jobs import JOBS
from h3_app.provenance import read_snapshot

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


def build_library_tools(root, list_paths):
    with root, gr.Accordion("Search, tags, lineage & compare", open=False):
        gr.Markdown(
            "The Media library is shared by users of this server. Tags and favorites are shared library metadata. "
            "Search uses filenames, tags and recorded technical settings; prompts are excluded."
        )
        with gr.Row():
            query = gr.Textbox(label="Search media")
            kind = gr.Dropdown(
                ["All", "Video", "Image", "Audio"], value="All", label="Media type"
            )
            favorite_only = gr.Checkbox(label="Favorites only")
        with gr.Row():
            search = gr.Button("Search library")
            more = gr.Button("Index next 200 files")
            rebuild = gr.Button("Rebuild media index")
        index_cursor = gr.State(0)
        status = gr.Markdown(
            "Index the library to search existing and newly generated files."
        )
        asset = gr.Dropdown([], label="Indexed asset", value=None)
        with gr.Row():
            tags = gr.Textbox(label="Tags (comma separated)")
            favorite = gr.Checkbox(label="Favorite")
            save = gr.Button("Save asset annotations")
        lineage = gr.JSON(label="Asset identity and lineage")
        with gr.Accordion("Compare two outputs", open=False):
            gr.Markdown(
                "Images use a reveal slider. Videos compare elapsed seconds from the beginning, "
                "at their native frame rates, on the shorter duration. This is browser playback synchronization "
                "with drift correction, not frame-accurate editing. Audio is muted by default; choose A or B. "
                "Buffering pauses both players."
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

    def scan(cursor):
        entries = [
            (path, mode)
            for mode in ("Video", "Image", "Audio")
            for path in list_paths(mode)
        ]
        for path, mode in entries[int(cursor) : int(cursor) + 200]:
            try:
                JOBS.store.index_asset(path, mode, read_snapshot(path))
            except (OSError, ValueError):
                continue
        end = min(len(entries), int(cursor) + 200)
        return (
            end,
            f"Indexed {end} of {len(entries)} files. Tags and favorites are preserved.",
        )

    def index_more(cursor):
        return scan(cursor)

    more.click(
        index_more,
        inputs=index_cursor,
        outputs=[index_cursor, status],
        queue=False,
        api_name=False,
    )

    def rebuild_index():
        # Metadata is processed in bounded batches and yields progress to the UI.
        with JOBS.store.connect() as db:
            db.execute("UPDATE assets SET available=0")
        cursor = 0
        while True:
            end, message = scan(cursor)
            yield end, message
            if end == cursor:
                break
            cursor = end

    rebuild.click(
        rebuild_index,
        outputs=[index_cursor, status],
        concurrency_id="h3-media-index",
        concurrency_limit=1,
        api_name=False,
    )

    def results(text, mode, favorites):
        rows = JOBS.store.search_assets(
            query=text, kind=None if mode == "All" else mode, favorite=favorites
        )
        allowed = {
            mode: {Path(p).resolve() for p in list_paths(mode)}
            for mode in {row["kind"] for row in rows}
        }
        rows = [
            row
            for row in rows
            if Path(row["path"]).is_file() and Path(row["path"]) in allowed[row["kind"]]
        ]
        choices = [
            (Path(row["path"]).name + " · " + row["id"][:8], row["id"]) for row in rows
        ]
        return (
            gr.update(choices=choices, value=None),
            gr.update(choices=choices, value=None),
            gr.update(choices=choices, value=None),
            f"{len(rows)} matching assets (up to 200).",
        )

    search.click(
        results,
        inputs=[query, kind, favorite_only],
        outputs=[asset, left, right, status],
        queue=False,
        api_name=False,
    )

    def resolve(asset_id):
        with JOBS.store.connect() as db:
            row = db.execute(
                "SELECT a.*, COALESCE(n.tags, '[]') AS tags, COALESCE(n.favorite,0) AS favorite FROM assets a LEFT JOIN annotations n ON n.asset_id=a.id WHERE a.id=?",
                (asset_id,),
            ).fetchone()
        if row is None or Path(row["path"]) not in {
            Path(p).resolve() for p in list_paths(row["kind"])
        }:
            raise gr.Error("This asset is unavailable in the current managed library.")
        return dict(row)

    def inspect_asset(asset_id):
        if not asset_id:
            return "", False, None
        row = resolve(asset_id)
        metadata = json.loads(row["metadata"])
        return (
            ", ".join(json.loads(row["tags"])),
            bool(row["favorite"]),
            {
                "asset_id": asset_id,
                "filename": Path(row["path"]).name,
                "settings_and_lineage": metadata,
            },
        )

    asset.change(
        inspect_asset,
        inputs=asset,
        outputs=[tags, favorite, lineage],
        queue=False,
        api_name=False,
    )

    def annotate(asset_id, text, starred):
        resolve(asset_id)
        JOBS.store.annotate(asset_id, tags=text.split(","), favorite=starred)
        return "Asset annotations saved."

    save.click(
        annotate,
        inputs=[asset, tags, favorite],
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
        return gr.update(value=None, visible=False), comparison_html(
            first["path"], second["path"]
        )

    compare.click(
        compare_assets,
        inputs=[left, right],
        outputs=[images, videos],
        queue=False,
        api_name=False,
    )
