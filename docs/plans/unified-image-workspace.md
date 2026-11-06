# Unified image workspace: research and proposed design

Research date: 2026-11-06. Reviewed and implemented after user approval.

Implementation keeps the existing model option components and public API adapters,
but dispatches the common draft directly to handlers. Legacy API draft fields
are not synchronized with the common composer. Results precede optional settings
in the mobile stack. MOSS editing is removed throughout the application.

## Recommendation

Give Image one composer and one result area. Selecting a model changes its
capabilities and optional settings, while preserving the user's prompt,
ordered references, and visible results. Keep Qwen Image 2.1 as the default.

Use an explicit Generate / Edit choice. Edit requires references; Generate
does not submit references. This makes intent consistent across engines that
currently infer editing from references and Qwen, which has an explicit mode.
Keep references in the session when Generate is selected, but clearly mark
them inactive and exclude them from generation and prompt enhancement.

MiniMax H3 uses the existing H3 settings authority. Its Image workspace has no
duplicate video model, performance, or sampling controls. MOSS is generation
only, with editing removed from the UI and underlying application paths.

## Research evidence

External examples support the overall direction, rather than prescribing this
application's implementation:

- Adobe Firefly exposes model selection within Generate Image and states that
  generation settings vary by the selected model. Its Boards workflow places
  additional generation settings beside aspect ratio. This supports keeping
  common composition controls prominent and model controls secondary.
  [Generate with partner models](https://helpx.adobe.com/sg/firefly/web/generate-images-with-text-to-image/generate-images-using-text-prompts/generate-images-using-non-adobe-models.html),
  [Boards model settings](https://helpx.adobe.com/ph_en/firefly/web/create-mood-boards/firefly-boards/use-non-adobe-models-to-generate-images.html).
- BFL describes one Playground for exploring models, adjusting parameters,
  and generating images, including multi-reference editing. This is evidence
  for integrating creation and editing into a model-selectable workspace.
  [BFL Playground](https://help.bfl.ai/articles/8667153955-what-is-the-bfl-playground).
- Adobe's Image5 API uses one request shape for text generation and instruct
  editing, with references determining the operation. This supports a common
  internal image request shape; an explicit mode remains useful in our UI
  because users can retain uploaded references while generating from text.
  [Image5 feature guide](https://developer.adobe.com/firefly-services/docs/firefly-api/guides/how-tos/cm-generate-image/feature-guide).

These sources do not establish our model limits. The checked-in validators
and graph builders are the authority for the capability table below.

## Current application findings

- `h3_ui/layout.py`: engine selection selects a separate engine tab. H3's
  Image task also changes the shared H3 result format and settings layout.
- `h3_ui/bootstrap.py`: each `EngineFeature` builds and binds its own view;
  engine-specific inputs are registered for Media, settings, and Remake.
- `h3_ui/engines/qwen/view.py`, `flux3.py`, `moss_image.py`, and `fibo2.py`:
  each owns a prompt, references, dimensions, actions, progress, and gallery.
  Placement and editing behavior differ, even though some controls are shared.
- `h3_ui/engines/controls.py`: generation already uses shared GPU admission,
  job ownership, interruption, and progress helpers. Preserve these semantics.
- `h3_ui/persistence.py`: model options already have separate namespaces.
  Preserve independent model preferences and migrate common controls carefully.
- `h3_app/generation/ph3_image.py`: H3 Image already routes to P-H3-Image,
  resolving the selected H3 profile's FL2VA base through existing policy. The
  image workflow fixes sampling at 8 steps, Euler, CFG 1. Video sampling
  controls therefore must not be presented as effective image controls.
- Qwen's composer help broadly says Turbo supports three references, while
  `h3_app/generation/qwen.py` applies that limit only to the Viggle variants.
  The new workspace should derive help and validation from shared capability
  rules, rather than repeating these limits in independent labels.

## Capability matrix

| Model | Generate | Edit | References per combined edit | Optional model controls |
| --- | --- | --- | --- | --- |
| Qwen Image 2.1 | Yes | Yes | 10 normally; 3 for Viggle Turbo variants | Preset, diffusion model, text encoder, Turbo/Lightning, sampling, negative prompt, attention/cache/acceleration, native enhancer, reference resolution, batching |
| FLUX 3 Image Klein | Yes | Yes | 10 | Precision/model, local prompt expansion and upsampler, sampling, layout boxes |
| MiniMax H3 / P-H3-Image | Yes | Yes | 8 | Read-only effective base summary and navigation to Video → MiniMax H3 settings |
| MOSS-Image | Yes | Removed | None | Preset, SFT/Base, precision, encoder, VAE, Turbo, Spectrum, sampling, layout/rewrite mode |
| Fibo 2.0 Dev | Yes | Yes | 4 | Dev/Edit Turbo, precision, sampling, VGL JSON prompt support |
| Fibo 2.0 Edit Turbo | No | Yes | 4 | Precision; fixed 4 steps and CFG 1 |

Qwen also supports editing each input separately and 1–4 images per batch.
Keep these as Qwen-specific controls, with the total output count displayed.
Do not imply every model supports batching or identical reference semantics.

## Proposed layout and behavior

Desktop: composer on the left, preview/results on the right. Mobile: composer,
primary actions, results, then collapsed optional settings.

1. Model picker with a short capability description.
2. Generate / Edit selector and shared prompt field. Change the label and
   placeholder with intent, without rewriting the prompt.
3. In Edit, ordered reference thumbnails with upload and Media-library input,
   remove/reorder actions, and selected-model reference limit. Identify the
   first image as target where applicable; show model-specific tag guidance.
4. Output size and seed in consistent positions. Offer model-valid presets,
   custom dimensions, and edit sizing choices; show resolved pixel dimensions
   and any alignment changes. Fibo retains its trained-size choice. Do not
   label each model's different megapixel policies as the same resolution.
5. Generate image / Edit image, Interrupt, and concise readiness/progress.
6. One results area with existing download and settings metadata support.
   Provide Use as edit input to continue working on a selected result.
7. Collapsed Model settings, containing only the selected model's options.
   Keep the selected preset visible in its summary; do not add an ambiguous
   cross-model Fast/Quality control.
8. Collapsed Prompt assistance with a consistent entry point. Dispatch to the
   selected model's existing writer recipe and supported reference behavior.
   Retain Qwen's prompt review behavior; unsupported assistance is unavailable.

State rules:

- Prompt, intent, ordered references, and results belong to the image session.
  Model settings remain independent, including model-specific size/seed values
  displayed through the common controls. This avoids corrupting another
  model's valid size or silently resetting its tuned configuration.
- Switching models preserves the draft. If references exceed the selected
  variant's limit, retain them and block submission with a concrete message;
  never truncate or reinterpret the inputs silently.
- In Edit, MOSS remains identifiable as generation only and cannot run. Offer
  switching intent to Generate; do not silently ignore references or change
  the mode. Its reference inputs never reach a MOSS request.
- Fibo Edit Turbo cannot run in Generate. Show the reason and offer selecting
  Dev or switching to Edit; do not change the user's intent automatically.
- A running job captures its model, draft, and effective settings at submission.
  Changing the picker cannot reroute its result or Interrupt action. Keep the
  result's originating model visible after switching to another model.
- Media → Use in Image and Remake → image editing target the common composer.
  Existing model-targeted destinations can select their model before filling
  the composer, where compatibility requires retaining them.
- Persist settings through the existing allowlist. Do not newly persist upload
  paths or API secrets as part of the shared draft.

## H3 settings reuse

Follow the existing Remake settings-reuse pattern in
`h3_ui/engines/remake.py` / `h3_ui/bootstrap.py`: read the authoritative H3
component/settings state and resolve policy when the image job is submitted.
Do not copy H3 defaults once at UI construction and allow them to become stale.

Image-specific inputs are prompt, image references, output sizing, and seed.
Build request overrides for these without writing image draft values into
the user's video composer. The H3 adapter translates Edit references into the
existing image request path; users should not encounter First / last frame or
Reference media video terminology in the image composer.

Show: “Uses MiniMax H3 settings. Adjust the base model under Video → MiniMax H3.”
Include an actionable navigation control and a compact effective-base summary.
Returning to Image restores the image draft. Explain the fixed P-H3 eight-step
sampling in the summary. Reusing H3 settings does not make video-only sampling,
decoding, duration, or finishing options effective in the still-image graph.

## Complete MOSS editing removal

Remove the functionality, rather than merely hiding the upload:

- Remove reference/edit-size components, edit callbacks, Media destinations,
  handler staging, and descriptions from `h3_ui/engines/moss_image.py`.
- Remove edit fields from `MossImageRequest` and the generated public API.
  If an explicit legacy adapter is retained, reject nonempty reference inputs
  with a clear unsupported-operation error; never silently generate instead.
- Remove reference-size resolution, edit preparation/model keys, and edit
  validation branches from `h3_app/generation/moss_image.py`.
- Remove edit LoRA and image conditioning graph branches from
  `h3_app/workflows/moss_image.py`.
- Remove `moss_image_edit_lora` from `h3_models.py` and its provisioning group;
  remove the edit-only `H3MossEditReference` custom node and registration after
  checking callers. Leave downloaded files and historical outputs on disk.
- Drop obsolete MOSS edit-size preferences during migration, update docs and
  fixtures, and replace edit-success coverage with generation-only coverage
  and rejection of legacy edit requests where an adapter exists.
- Retain MOSS text rendering, layout generation, presets, Turbo and Spectrum.

## Implementation approach

Create a dedicated shared Image view and a small image-controller module.
Navigation selects this same Image surface for all image engine choices;
other tasks retain their existing surfaces. Model settings builders populate
only their option panels instead of constructing separate composers.

Introduce a UI-level image draft and adapters to the existing typed generation
requests. Keep graph builders, model provisioning, output metadata, job queue,
and engine-specific prompt writers in their current owning modules. H3's
adapter reads live H3 settings, while the other adapters read their model's
options. Avoid introducing a second backend generation framework.

Keep existing supported per-engine API endpoints through explicit adapters,
even when their visible composers are removed. Shared UI submission should
reuse owned GPU admission and job recording rather than chaining button events
or synchronizing hidden duplicate prompt/reference/gallery widgets.

Recommended order: define capability/state rules; build shared layout and
adapters; connect model option panels and H3 reuse; remove MOSS edit paths;
migrate persistence and Media/Remake routing; verify and update documentation.

## Focused verification for implementation

- Pure capability/adapter tests: Qwen variant limits and separate editing,
  H3 eight-reference limit and latest base selection, Fibo edit-only constraints,
  MOSS edit rejection, size policies, and exclusion of inactive references.
- UI contract/navigation checks: every Image model uses the same composer and
  result surface; model switching retains draft/results; optional settings
  remain independent; H3 settings navigation returns without losing the draft.
- Job checks: correct model dispatch, original-job interruption after switching
  models, output metadata, and no admission/ownership regression.
- Persistence checks: restore existing model preferences, discard MOSS edit
  preferences, and preserve video settings during image generation.
- Browser checks using the lightweight workspace fixture: desktop/mobile
  layout, keyboard access, reference ordering, Media/Remake routing, and the
  generate → result → edit loop. GPU quality tests are separate from UI tests.

Run targeted checks for the changed paths; do not run the full suite repeatedly.
