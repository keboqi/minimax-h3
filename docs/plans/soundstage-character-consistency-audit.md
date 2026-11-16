# Sound Stage character consistency audit

## Observed behavior

The supplied character shots have harsh background texture and highlights. Shots without declared
characters, and a user test with consistency disabled, avoid the problem. This isolates the additional
portrait-matching operation, but does not establish that Qwen editing inherently reduces quality.

The user reports Qwen Image 2.1 with default Image-page settings. The actual before/after PNGs and
ComfyUI execution history for the affected run are not available in this checkout. The root cause
therefore remains a hypothesis until the same draft is compared with its matching output.

## Paths checked

Both paths reach `QwenHandlers.generate_qwen_image21`, then the same
`generate_qwen_image21` orchestration and `build_qwen_image21_graph` graph builder.
The Image page supplies Sound Stage with live, owner-specific saved options through `Live(job_options)`;
Sound Stage is not relying on a stale settings snapshot from an earlier tab switch.

The matching inputs are ordered correctly: the complete generated scene first, followed by only the
portraits of declared characters visible in the shot. Props and locations are not image references.
Reference tags are generated using the same helper as the Image page.

Default Qwen options are native-resolution references (`reference_resolution=0`), enhancer Off,
lossless cache precision (`cache_dtype=default`), and the Fast sampling preset. Sound Stage uses those
same sampling options. This audit found no evidence of a separate lower-quality matching sampler,
an extra sharpening filter, or a quantized edit-cache override. Fast sampling is shared with the Image
page; changing it globally is not a justified fix for this difference.

The matching edit receives the original scene output. JPEG conversion happens only when the finished
images are kept for playback, so a JPEG round trip is not causing the matching input to deteriorate.

## Strongest remaining suspect: edit scope

The old matching prompt requests a local identity change and then appends the entire tagged scene:
camera, action, lighting, stone, wood, rain and other unchanged materials. The preceding patch also
inserted the complete rendering preset. That creates a second scene-rendering brief inside a local
edit request. Repeating unchanged materials is unnecessary, and can encourage repainting their
appearance rather than preservation. The full rendering-preset insertion has been removed.

The replacement instruction leads with the requested operation, identifies each character from its
own anchor, binds it to its portrait, and preserves the rest of the source with a compact clause.
It explicitly retains pose, expression, gaze, position, background, lighting, colors, rendering and
surface-detail level. No whole-scene narrative or rendering preset is repeated. Portraits supply
identity, hair and outfit design, not scene lighting or shading.

This matches the distinction already stated in the repository's Qwen edit-writer instructions:
local edits should avoid detailed descriptions of unchanged regions. It remains a model-behavior
hypothesis, not a visually demonstrated root cause.

## Source alignment safeguard

Previously, Sound Stage forced manual output dimensions for both scene generation and matching,
while inheriting reference resizing. With the reported defaults, the generated 1664x928 scene and its
reference already match, so this is not an explanation of the supplied default-settings failure.
With a nonzero reference resolution, however, encoded-source and output dimensions can differ.

The matching pass now uses original-size references and the native first-reference-sized output latent.
It also disables native prompt expansion for this already bounded instruction and clears named output
size presets. These options are applied to a copy of the captured settings; scene generation,
portrait generation, sampling steps, CFG, model, Turbo mode and cache precision are unchanged.

ComfyUI's native `TextEncodeQwenImage21` exposes a first-reference-sized latent and explicitly warns
that sampling at another size shifts an edit. Resolution zero preserves each reference's own size,
rounded to the model's grid. See the [native node implementation](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_qwen.py).

## Evidence retained for the next draw

Each freshly matched shot keeps a lossless `*-draft.png` beside its playback image under
`<output_dir>/.soundstage/stills/<mix stem>/`. Its `stills.json` entry includes `draft`, `matching.prompt`,
ordered reference filenames, requested matching settings and the original output filename. When a
Qwen execution snapshot exists, its job ID and effective settings, including resolved seed, are copied
into the matching record. Drafts and matching records survive an interrupted run and a partial redraw.
Older entries without this evidence remain readable; their historical drafts cannot be reconstructed.

Each completed shot is now published before the next shot starts. The player and shot list refresh
while the run is active, and completed shots remain available if a later generation fails. Existing
playback images and draft PNGs are reused by filename during checkpoints, avoiding repeated image
copies and recompression. Rendering presets and saved-style compatibility live in `visual_styles.py`.
Character consistency defaults off; explicitly saved project choices are retained.

After restarting the application, redraw one affected shot with consistency enabled. No replanning
or style change is needed. Compare that shot's saved draft against its matched output. If the new
bounded edit still changes the background, that is evidence to investigate reference conditioning or
sampling with the exact recorded request, rather than adding more texture or style adjectives.

## Validation

Focused tests cover per-engine reference tags and ordering, omission of unchanged scene text,
source-aligned Qwen options without sampling changes, exact request recording, lossless draft
retention, retry metadata, partial redraw, and the existing Qwen graph and request orchestration.
No live Qwen image generation or visual quality comparison was performed in this checkout.
