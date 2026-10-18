# FLUX 3 official workflow update — adaptation

Follow-up: [remaining engine adaptations](remaining-engine-updates-2026-10-18.md) now add Base models, layout boxes and W4A8 upsampling, and advance the coordinated Core/Kitchen/Drama pins. The implementation and coverage below describe the first FLUX adaptation step.

BFL acknowledged Comfy workflow problems and announced corrected workflows with ComfyUI v0.41.2 in [its model discussion](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/discussions/15). Our previous assessment checked GitHub Releases, which still lists v0.41.0; a separate tag check finds v0.41.1, v0.41.2 and v0.41.3. This follow-up implements the applicable changes rather than leaving the assessment as recommendations.

## Implemented

- **Core v0.41.3**, commit `8713616fcc321624e306bbd36be5906203dcb0dc`, with workflow templates 0.11.87. This maintenance branch retains the existing Qwen decode APIs: Kitchen 0.2.37, frontend 1.55.17, Torch and Drama pins remain compatible. Unreleased main's cache API migration is a separate upgrade.
- **Official hyphenated distilled checkpoint paths**, with unchanged SHA256 pins. Existing BF16/NVFP4 local filenames remain stable. A manifest with the same repository, pinned hash and file size accepts a remote rename; planning also reuses the same remote content identity, avoiding a redownload.
- **INT8 ConvRot distilled model option**, using the official 5.83 GB checkpoint and its published SHA256. Existing model defaults remain available.
- **Updated BFL sampling shift**: alpha 16, expressed as `ln(16)` in native `ModelSamplingFlux`. Equal base/max anchors keep it independent of canvas dimensions. Distilled sampling remains 16 steps, CFG 1, Euler/simple. This explicit graph override is needed because even the latest observed Core defaults to alpha 6.93.
- **Official caption-decoding defaults**: MTP off, greedy top-k 1, fixed seed, 3584-token limit. Existing INT8 prompt upsampler remains supported; templates also offer the smaller W4A8 quantization, which is not required for the workflow correction.
- **Local image-aware edit expansion**, enabled when references are selected. `H3Flux3EditPrompt` uses the same native Qwen3.5 tokenization/generation APIs and edit settings as the official workflow, passing up to ten separate RGB image tensors in reference order. This supports different image sizes without cropping them into a common batch. Image conditioning reuses the original load nodes. Empty/truncated reasoning outputs fail before an unconditional edit can be generated.
- **Separate edit/T2I instructions**, using the existing local prompt files that match the official templates. Remote rewriting still disables redundant local expansion after success.

## Coverage boundaries

The current generation/editing integration now covers the announced workflow corrections and the reference implementation's latest sampling parameter. This does not add Base-model sampling, bounding-box/layout controls, or FLUX video models. Base's new CFG 4 default does not apply to our distilled graph. The existing 4 MP application canvas cap remains; it is an application resource limit rather than full coverage of BFL's maximum-resolution capabilities.

No replacement model contents were identified for our existing BF16/NVFP4/encoder/VAE/INT8-upsampler files: their current published hashes match the configured pins. Only INT8 selection requires an additional image checkpoint. No packages or weights were downloaded, no deployment was performed, and no GPU quality/latency improvement is claimed.

## Evidence

- [BFL official announcement in discussion 15](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/discussions/15).
- [Core v0.41.0 → v0.41.3 comparison](https://github.com/Comfy-Org/ComfyUI/compare/98cc0bf84b7cb1383654c5b402e7612d875d6d22...8713616fcc321624e306bbd36be5906203dcb0dc).
- [Official distilled generation template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_flux3_klein_distilled_t2i.json) and [edit template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_flux3_klein_distilled_image_edit.json).
- [BFL alpha/CFG update](https://github.com/black-forest-labs/flux-3-image-klein/commit/62e90b9278b73ecc52927770c6965dc6b1966498).
- [Published model names and hashes](https://huggingface.co/api/models/black-forest-labs/flux-3-image-klein-comfy?blobs=true).

Raw comparisons and workflow snapshots are in ignored `.cache/flux-official-update-2026-10-18/`. Validation: 47 focused workflow/component tests, two Gradio UI contract checks, and model inventory/requirements selftests pass. Coverage includes graph wiring/sampling, image order and unequal sizes, empty edit captions, cache migration, model selection, UI edit-expansion controls, Drama graphs and existing component contracts. The UI checks ran outside the sandbox because its socket restrictions stalled Python's asyncio socketpair. No full suite was run. GPU inference remains unverified.
