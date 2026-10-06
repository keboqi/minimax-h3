"""Persistent asset annotations, lineage and CPU-only comparison controls."""

from html import escape
from pathlib import Path
from urllib.parse import quote
import json

import gradio as gr

from h3_app.jobs import JOBS
from h3_app.errors import H3Error
from .media_previews import browser_gallery_items, browser_file_update

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


def build_library_tools(
    root, list_paths, *, view, system_root, validate_path, preview_media
):
    from .asset_index import AssetIndex
    from h3_app.gallery_store import AssetInventory

    index = AssetIndex(JOBS.store, list_paths)
    view.search.h3_asset_index = index
    with view.inspector:
        with gr.Row(elem_classes=["h3-compare-selection-actions"]):
            add_a = gr.Button("Add to compare A", interactive=False)
            add_b = gr.Button("Add to compare B", interactive=False)
    with view.inspector:
        with gr.Accordion("Tags & lineage", open=False):
            gr.Markdown(
                "Organize this asset with tags and favorites. Annotations are shared across the library."
            )
            with gr.Row():
                tags = gr.Textbox(label="Tags (comma separated)")
                favorite = gr.Checkbox(label="Favorite")
            save = gr.Button("Save asset annotations")
            status = gr.Markdown()
            with gr.Accordion("Asset identity and lineage", open=False, max_height=400):
                lineage = gr.JSON(label="Asset identity and lineage")
    pair = gr.State({"a": None, "b": None})
    with root, gr.Accordion("Compare two outputs", open=False) as comparison:
        gr.Markdown(
            "Select a thumbnail in the library, then add it to A or B. Compare two different images or two different videos. "
            "Your picks stay here while you search and browse. Adding another media type starts a new pair."
        )
        with gr.Row(equal_height=True, elem_classes=["h3-compare-slots"]):
            with gr.Column(min_width=140):
                left = gr.Image(
                    label="Comparison A",
                    type="filepath",
                    height=180,
                    interactive=False,
                    buttons=[],
                    elem_id="h3-compare-a",
                    min_width=140,
                )
                clear_a = gr.Button("Clear A", size="sm")
            with gr.Column(min_width=140):
                right = gr.Image(
                    label="Comparison B",
                    type="filepath",
                    height=180,
                    interactive=False,
                    buttons=[],
                    elem_id="h3-compare-b",
                    min_width=140,
                )
                clear_b = gr.Button("Clear B", size="sm")
        comparison_status = gr.Markdown(
            "Choose an image or video in the library to start.",
            elem_classes=["h3-compare-status"],
        )
        compare = gr.Button(
            "Compare selected outputs", variant="primary", interactive=False
        )
        images = gr.ImageSlider(
            label="Image comparison A / B",
            type="filepath",
            interactive=False,
            visible=False,
        )
        videos = gr.HTML("", js_on_load=VIDEO_JS)
    with system_root, gr.Accordion("Media index maintenance", open=False):
        gr.Markdown(
            "New outputs are registered when created. Scan historical media to add older or externally created files and repair missing previews."
        )
        rebuild = gr.Button("Scan historical media", elem_id="h3-system-history-scan")
        index_status = gr.Markdown()

    def scan_history():
        for message in index.rebuild():
            yield message, message

    scan_event = gr.on(
        triggers=[rebuild.click, view.scan_history.click],
        fn=scan_history,
        outputs=[index_status, view.status],
        concurrency_id="h3-media-index",
        concurrency_limit=1,
        api_name=False,
    )
    view.search.h3_scan_event = scan_event

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
            return "", False, None
        row = resolve(asset_id)
        return (
            ", ".join(json.loads(row["tags"])),
            bool(row["favorite"]),
            {
                "asset_id": asset_id,
                "filename": Path(row["path"]).name,
                "settings_and_lineage": {key: value for key, value in json.loads(row["metadata"]).items() if key != "_media"},
            },
        )

    def slot_preview(asset_id, slot, request):
        if not asset_id:
            return gr.update(value=None, label=f"Comparison {slot.upper()}")
        row = resolve(asset_id)
        metadata = json.loads(row["metadata"])
        page = preview_media(
            row["kind"], 1, paths=AssetInventory(
                row["kind"], (Path(row["path"]),),
                ({**row, "metadata": metadata},), 1,
            )
        )
        version = metadata["_media"]["registered_ns"]
        items = browser_gallery_items(page.items, request, version=version)
        return browser_file_update(gr.update(
            value=items[0][0] if items else None,
            label=f"Comparison {slot.upper()} · {row['kind']} · {Path(row['path']).name}",
        ))

    def pair_status(current):
        filled = [slot.upper() for slot in ("a", "b") if current.get(slot)]
        if len(filled) == 2:
            return "A and B are ready. Compare them below."
        if filled:
            missing = "B" if filled[0] == "A" else "A"
            return (
                f"{filled[0]} is set. Select another thumbnail and add it to {missing}."
            )
        return "Choose an image or video in the library to start."

    pair_outputs = [
        pair,
        left,
        right,
        compare,
        comparison_status,
        images,
        videos,
        comparison,
    ]

    def update_pair(current, request, message=None):
        return (
            current,
            slot_preview(current["a"], "a", request),
            slot_preview(current["b"], "b", request),
            gr.update(interactive=bool(current["a"] and current["b"])),
            message or pair_status(current),
            gr.update(value=None, visible=False),
            "",
            gr.update(open=True),
        )

    def assign(slot, mode, path, current, request):
        asset_id = selected_id(mode, path)
        if not asset_id or mode not in {"Image", "Video"}:
            raise gr.Error("Select an image or video thumbnail in the library first.")
        candidate = resolve(asset_id)
        current = dict(current)
        other = "b" if slot == "a" else "a"
        message = None
        if current[other]:
            try:
                previous = resolve(current[other])
            except gr.Error:
                current[other] = None
            else:
                if previous["kind"] != candidate["kind"]:
                    current[other] = None
                    message = f"Started a new {mode.lower()} pair. Add another {mode.lower()} to {other.upper()}."
                elif current[other] == asset_id:
                    raise gr.Error(
                        "This media is already in the other slot. Choose a different thumbnail."
                    )
        current[slot] = asset_id
        return update_pair(current, request, message)

    for button, slot in ((add_a, "a"), (add_b, "b")):

        def add(mode, path, current, request: gr.Request, slot=slot):
            return assign(slot, mode, path, current, request)

        button.click(
            add,
            inputs=[view.mode, view.selected, pair],
            outputs=pair_outputs,
            postprocess=False,
            concurrency_id="h3-comparison",
            concurrency_limit=1,
            api_name=False,
            show_progress="hidden",
        )

    for button, slot in ((clear_a, "a"), (clear_b, "b")):

        def clear(current, request: gr.Request, slot=slot):
            return update_pair({**current, slot: None}, request)

        button.click(
            clear,
            inputs=pair,
            outputs=pair_outputs,
            postprocess=False,
            concurrency_id="h3-comparison",
            concurrency_limit=1,
            api_name=False,
            show_progress="hidden",
        )

    snapshot, snapshot_component = view.selected.h3_snapshot

    def synchronize(mode, paths, selected, current, request: gr.Request = None):
        inspected = inspect_asset(mode, selected)
        can_add = bool(inspected[2]) and mode in {"Image", "Video"}
        revised = dict(current)
        for slot in ("a", "b"):
            if revised[slot]:
                try:
                    resolve(revised[slot])
                except gr.Error:
                    revised[slot] = None
        updates = (
            (
                revised,
                slot_preview(revised["a"], "a", request),
                slot_preview(revised["b"], "b", request),
                gr.update(interactive=bool(revised["a"] and revised["b"])),
                "An unavailable comparison item was removed. " + pair_status(revised),
                gr.update(value=None, visible=False),
                "",
            )
            if revised != current
            else (gr.skip(),) * 7
        )
        return (
            *inspected,
            snapshot(selected),
            gr.update(interactive=can_add),
            gr.update(interactive=can_add),
            *updates,
        )

    view.selected.h3_sync = (
        synchronize,
        [view.mode, view.paths, view.selected, pair],
        [
            tags,
            favorite,
            lineage,
            snapshot_component,
            add_a,
            add_b,
            pair,
            left,
            right,
            compare,
            comparison_status,
            images,
            videos,
        ],
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

    def compare_assets(current):
        a, b = current.get("a"), current.get("b")
        if not a or not b or a == b:
            raise gr.Error("Add two different thumbnails to A and B first.")
        first, second = resolve(a), resolve(b)
        if first["kind"] != second["kind"] or first["kind"] not in {"Image", "Video"}:
            raise gr.Error("Select two images or two videos to compare.")
        if first["kind"] == "Image":
            return gr.update(value=(first["path"], second["path"]), visible=True), ""
        return gr.update(visible=False), comparison_html(first["path"], second["path"])

    compare.click(
        compare_assets,
        inputs=pair,
        outputs=[images, videos],
        concurrency_id="h3-comparison",
        concurrency_limit=1,
        api_name=False,
    )
