# Gallery response investigation — 4 October 2026

The delay comes from synchronous library work on the Media tab's interaction path. A local CPU-only profile reproduced slow opening and selection; GPU inference was not involved.

## Causes and changes

- `library_tools.py` previously scanned and indexed both images and videos for comparison dropdowns after every gallery refresh and selection, including while Compare was collapsed. Comparison inventory now loads when the panel opens and reuses completed scans until the index revision changes. Selection and annotation actions validate the current file directly through the existing managed download-path boundary.
- `AssetIndex.sync` committed every changed asset separately through SQLite with `synchronous=FULL`. New/changed assets now use one transaction per scan, preserving stable IDs, tags and favorites. Unchanged scans do not write the database.
- The thumbnail page repeated the scan already performed by the index. An internal typed, ordered inventory now carries that scan into page rendering. Ordinary path filters still intersect the managed library; inventories with a mismatched media type are rejected.
- Directory discovery statted sidecars and traversed preview/processing directories unnecessarily. The shared walker filters extensions first, prunes those directories and reuses modification times for sorting.
- The first 48 video posters ran FFmpeg sequentially, each with a 60-second timeout. A shared pool now allows at most four poster tasks across sessions, with one decoding/filtering thread per process and a 10-second timeout. Results retain gallery order. Worker contexts preserve cancellation when called from an owned finishing job.

## Local measurements

The isolated library contained 1,000 valid 960×640 PNGs with settings sidecars and 60 valid one-second 640×360 H.264 videos. The grid shows 48 cards. Timings are individual instrumented Python callback samples on Windows with Gradio 6.29.1; they exclude browser media transfer, Gradio postprocessing and production storage/network effects.

| Interaction | Before | After |
| --- | ---: | ---: |
| First Video opening, grid + inspector | 15.42 s | 1.49 s |
| Warm Video opening, grid + inspector | 2.17 s | 0.21 s |
| Warm Image opening, grid + inspector | 4.26 s | 1.46 s |
| Video selection inspector | 2.06–2.08 s | 0.006–0.010 s |
| Image selection inspector | 3.83–3.94 s | 0.009–0.010 s |

Initial image indexing is deferred until images are needed. The first Image opening after Video took 4.28 seconds after the change; the baseline took 4.64 seconds, having already indexed all images during Video opening. This work is deferred, not eliminated. Active-library discovery still scales with library size, and the first comparison expansion can index an unseen media type. Larger libraries or slow mounted storage may warrant a separate background-discovery design.

Evidence and the reproducible profiling script are under `.cache/gallery-performance/`: `baseline.json`, `final.json`, their cumulative profiles, and `profile_gallery.py`. These ignored artifacts use temporary media and workspace state and never load models or submit GPU jobs.

## Validation

90 relevant CPU tests passed, including published UI/API contracts, gallery restoration/HDR routing, batch indexing and annotation preservation, missing/unmanaged file rejection, typed inventory handling, and bounded poster concurrency with preserved order/context. The existing workspace browser acceptance passed, covering filtered/empty gallery selection, annotations, comparison choices outside the search, and image/video comparison playback. Ruff lint/format, Python compilation and Git whitespace checks passed.

No deployed server was measured or restarted. Restart or redeploy the application to use these changes.
