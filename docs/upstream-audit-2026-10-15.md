# Component update audit — 2026-10-15

Refreshed all 15 configured generation/provisioning Git sources against their public default branches and queried 24 primary packages on PyPI. Eight repositories are ahead of our pins; seven match. This is a source/metadata assessment: no runtime pins, packages, models or deployments were changed.

**Recommendation:** keep the current production pins. The main new adaptation candidate is native H3 LightVAE support in unreleased ComfyUI. Evaluate that with the next Core release. T8's new AudioFirst and modality-scoped LoRA features deserve separate feature evaluation, not a blanket node-pack upgrade. The October 14 ComfyUI/Veda/Gradio upgrades are already implemented.

## New since the previous audit

### ComfyUI: native LightVAE is useful maintenance work

Latest published release remains **v0.40.0**, already pinned at `3541246a9c997f4b8c32f9f88f3c985d02021c79`. Observed default-branch HEAD `ba994879b88d0ebd9950b1e5c1e923d928f216ba` is eight commits ahead. Its requirements file is unchanged in the comparison. [Release](https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.40.0), [comparison](https://github.com/Comfy-Org/ComfyUI/compare/3541246a9c997f4b8c32f9f88f3c985d02021c79...ba994879b88d0ebd9950b1e5c1e923d928f216ba).

Commit `1d9cec9e` adds partial encoder/decoder loading and shortcut-only pruned encoder blocks to the native H3 VAE. This overlaps our `custom_nodes/H3LightVAE/h3_vae.py`, which currently constructs both halves, removes unused components, and replaces pruned blocks itself. Native constructor flags could simplify that implementation and avoid constructing unused components. No measured memory, speed or output-quality improvement is established. [Change](https://github.com/Comfy-Org/ComfyUI/commit/1d9cec9e).

Do not simply delete our loader: it also converts the configured checkpoint tensor layout, preserves original-component fallback, and lets users independently select the encoder and decoder. Those behaviors need explicit preservation. The new constructor arguments have backward-compatible defaults; this source review found no immediate reason our existing constructor call must change, but that is not runtime validation.

For a future adoption, compare original encoder/LightVAE decoder, LightVAE encoder/original decoder, both LightVAE components, INT8 ConvRot decoding, tiled operation, offload and 500K still decoding. Review exact checkpoint loading and graph contracts before replacing custom code. Prefer the next tagged Core release unless this maintenance work becomes urgent.

Other HEAD additions include ReferenceMultiLatent, latent geometry nodes and SDR-to-HDR image conversion. No current graph requirement or demonstrated generation benefit was identified. The LoRA-stack disabling commit was subsequently reverted within this comparison; it is not a remaining HEAD change.

### T8 Audio: new experimental features, not a current voice-path fix

Latest is **1.101.0**, `6377904c07b163c6ef76dccc45559bbe64d2c06d`, versus our 1.85.0 pin. Since yesterday's 1.99.0 observation:

- 1.100.0 adds AudioFirst diagnostics and native Wuli Hybrid workflows, plus a Wuli weight-verification cache fix.
- 1.101.0 adds generated-audio/video token-row LoRA scoping and invocation auditing.

These are possible future creative controls. They require graph/UI integration and, for Hybrid, a model review. Upstream reports acceptance of specific short clips and explicitly limits generalization; modality-scoped residuals do not guarantee the other final modality stays unchanged. No benefit to our existing voice-conditioning graph was established. Keep the pin for now. [Exact changelog](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/blob/6377904c07b163c6ef76dccc45559bbe64d2c06d/CHANGELOG.md).

### LTXVideo and KJNodes

LTXVideo is now four commits ahead. The two commits since yesterday add registry publishing automation; earlier changes remove legacy nodes and adjust templates. Keep our working tiled-fusion pin: no new runtime benefit was identified. [Comparison](https://github.com/Lightricks/ComfyUI-LTXVideo/compare/3bf3ca62595f1764c47d01c35c8e5dfe47e1a88f...44c8a36b7c31ee00b4271daed11dc2a4aa481140).

KJNodes is now four commits ahead. New changes fix Model Preview Override targeting inside loops/expanded nodes and clean up code. This is conditional value for direct ComfyUI graph users, with no identified benefit to the current Gradio workflows. [Comparison](https://github.com/kijai/ComfyUI-KJNodes/compare/57105374f47d0fbb49c9c3926fb981702e0a4b5c...daa2bee6951527ed5600fa0c7263a1154ffbf630).

## Complete source inventory

| Source | Configured pin | Observed HEAD | Commits ahead |
|---|---|---|---:|
| ComfyUI | 3541246a | ba994879 | 8 |
| ControlNet Aux | 59b1fc41 | a587a4c1 | 7 |
| H3 latent upscaler | d7c01b90 | 40316cf0 | 1 |
| KJNodes | 57105374 | daa2bee6 | 4 |
| Larry Turbo | 4274783a | same | 0 |
| LTXVideo | 3bf3ca62 | 44c8a36b | 4 |
| PlagueKind SLA | d58d006a | same | 0 |
| Veda | 67e4eb55 | same | 0 |
| Sol public fork | 930a4d6e | same | 0 |
| Spectrum H3 | 5161f045 | 806fe498 | 1 |
| Spectrum Qwen | ddb5470f | same | 0 |
| Video Depth Anything | a0db08e6 | same | 0 |
| T8 Audio | 6063fafb | 6377904c | 23 |
| SwiftVR | 5ca168ce | 4fb82651 | 3 |
| H3VAE TRT | 4360e008 | same | 0 |

Unchanged deferred candidates retain the [October 14 assessment](upstream-audit-2026-10-14.md): Spectrum H3 0.2.29 targets Core BSA/Flow/Untwist compatibility; ControlNet Aux 1.1.6 has face/animal-pose fixes with dependency migration to review; the latent-upscaler delta is license-only and SwiftVR's delta is documentation-only.

## Packages and validation priority

Of the 24 queried packages, only OpenAI SDK registry latest changed since yesterday: **3.29.0 → 3.31.0**. Runtime already allows it (`>=3.16.2,<4`); tests pin 3.16.2. This metadata check does not establish a needed SDK migration. Gradio 6.30.0, frontend 1.55.17 and Kitchen 0.2.37 still match registry latest.

Keep Torch 2.11.0/CUDA 13.0 and the matched Sage wheel, NumPy 1.26.4/SciPy 1.15.3, Hub <2, Diffusers <0.37, kernels 0.16.0 and the paired Kornia pins. Newer registry versions are not a validated replacement stack. Before broad upgrades, capture a tested Linux/Python 3.12 resolution, especially Transformers/Hub/kernels/PEFT and TensorRT; broad ranges currently permit build drift. [OpenAI package metadata](https://pypi.org/pypi/openai/3.31.0/json).

Raw GitHub comparisons, release metadata, T8 changelog and PyPI dependency metadata are retained in ignored `.cache/component-audit-2026-10-15/`. The GitHub comparison file list may truncate at 300 entries, particularly for T8; its complete source was not audited. No model-registry-wide inventory, fresh Linux build, package installation, GPU inference or test suite was performed. Recommendations concern source compatibility and integration value, not measured GPU performance or exhaustive security review.
