# Component update audit — 2026-10-17

Checked the current configured pins against public GitHub default branches and queried 25 packages on PyPI. The automated inventory covers 17 provisioning repositories; the bundled ModelTC/LightVAE adapter was checked separately. Ten repositories are ahead of their pins, eight match including LightVAE. No runtime pins, installed packages, models or deployments were changed.

**Recommendation:** prioritize a targeted ComfyUI-Drama3 upgrade, then simplify the LightVAE adapter using the native API already present in our pinned ComfyUI v0.41.0. Keep the remaining generation stack pinned. Capture a tested Linux/Python 3.12 dependency resolution before broader package upgrades.

## 1. Drama 3: strongest upgrade candidate, requires integration work

Current pin `66c4fb9c`; observed HEAD `bc9b04a2acf1c553bec3b5e1d2339093eb826396`, two commits ahead. The substantive commit is `7a23df14`, which changes dialogue generation, sampling, reference preparation and decoding, and adds an editor and saved-code nodes. [Exact comparison](https://github.com/PedroMarinhoDev/ComfyUI-Drama3/compare/66c4fb9cd3c0151e1e2ccd207847cb8335e87ab4...bc9b04a2acf1c553bec3b5e1d2339093eb826396).

Useful changes:

- Windowed codec decoding uses 256-frame windows and 320 frames of left context. The decode node reserves activation memory when loading the codec. This addresses long-take decode memory pressure; upstream's reported fidelity/memory figures have not been measured here.
- Dialogue lines use the model's audio separator, with per-line, contextual per-line, per-speaker and joint generation modes. Fully referenced joint dialogue avoids earlier generated takes as context to reduce voice carryover.
- `top_k=0` now means uncapped sampling and becomes upstream's default. Upstream reports fewer silent/stalled takes than with 30. That is upstream evidence, not a guarantee for our clips.
- Reference clips are trimmed and peak-normalized. Incomplete/silent openings receive an internal second attempt.
- The new editor, code persistence and optional forced aligner are separate feature candidates; they are not prerequisites for adopting decoding fixes.

Required adaptations before changing the pin:

1. Our request, graph and UI default to `top_k=30`, and validation and the slider reject zero. Update these together if adopting uncapped sampling; confirm the SGLang route's semantics separately because the control is shared across backends.
2. Explicitly choose/pass the new `speakers` input. Our current ComfyUI graph describes a joint take, while upstream now defaults to one take per line. A pin-only update would silently change that behavior even with `split_long_text=False`.
3. Explicitly choose/pass `shorten_long_pauses`. Its new default trims leading/trailing silence and caps pauses at 0.75 seconds, which can alter intended dramatic timing.
4. Preserve our explicit speaker tags. Our graph enumerates reference slots independently of speaker IDs and embeds the actual speaker tag in each transcript. Upstream honors explicit tags, so slot numbering alone is not a reason to rewrite this mapping.
5. Characterize the new `Drama3Codes` dictionary (`codes`, `lines`, `takes`) instead of the previous tensor. The stock Generate → Decode pair changes together; any custom consumers or persisted codes need review.
6. Review retry/progress/cancellation behavior: upstream now retries an opening internally, while the app has separate optional SGLang chunk retries. These are different paths and policies.

Validation should cover single and multiple speakers, noncontiguous speaker IDs, invented voices, reference preparation, pause preservation, long decode, empty/stalled output, cancellation and graph schema. GPU checks should measure peak decode VRAM and listen for window seams and voice leakage. No extra aligner download is needed unless integrating the editor.

## 2. LightVAE: adapt to the API we already ship

ComfyUI's latest published release is **v0.41.0**, which matches our configured `98cc0bf8` pin. Its native H3 VAE already accepts `encoder`, `decoder` and `encoder_kept_blocks`. [Release](https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.41.0), [exact native source](https://github.com/Comfy-Org/ComfyUI/blob/98cc0bf84b7cb1383654c5b402e7612d875d6d22/comfy/ldm/minimax/vae.py).

Our `custom_nodes/H3LightVAE/h3_vae.py` still constructs both components, removes unselected components, and manually replaces pruned encoder blocks with `ResidualShortcut`. The next maintenance change should load the selected component metadata first and pass the native flags and kept-block list at construction. Preserve our tensor-name conversion, QKV packing, quantization handling, independent encoder/decoder choices and original-VAE fallback.

This can avoid constructing unused components and remove duplicate pruning logic. No measured speed or memory benefit is claimed. The bundled adapter's upstream ModelTC/LightVAE revision `38b0587d` still matches HEAD. Validate standard and INT8 decoding, encoder-only and decoder-only combinations, strict checkpoint loading, tiled operations, offload and 500K still routing before switching implementations.

## 3. Other repositories: keep pins unless adopting a specific feature

| Component | Current pin | Observed HEAD | Ahead | Assessment |
|---|---|---|---:|---|
| ComfyUI | 98cc0bf8 | fc04da91 | 7 | Unreleased MOSS-image support, partner nodes, labels/docs/templates; no identified fix for current graphs |
| ControlNet Aux | 59b1fc41 | a587a4c1 | 7 | Face/animal-pose fixes and dependency cleanup; evaluate if those preprocessors are needed |
| H3 latent upscaler | d7c01b90 | 40316cf0 | 1 | License-only change |
| KJNodes | 57105374 | 17195fcf | 5 | Adds FLUX 3 prompt builder; optional direct-ComfyUI feature, plus loop-preview fixes |
| Larry Turbo | 4274783a | same | 0 | Current |
| LTXVideo | 3bf3ca62 | 44c8a36b | 4 | Legacy-node removals/template changes and registry publishing; preserve bundled workflow contracts |
| PlagueKind SLA | d58d006a | same | 0 | Current |
| Veda | 67e4eb55 | same | 0 | Current |
| Sol public fork | 930a4d6e | same | 0 | Current |
| Spectrum H3 | 5161f045 | 806fe498 | 1 | v0.2.29 restores Core BSA/Flow/Untwist forecasting; current UI uses legacy mode |
| Spectrum Qwen | ddb5470f | same | 0 | Current |
| Video Depth Anything | a0db08e6 | same | 0 | Current |
| T8 Audio | 6063fafb | 6377904c | 23 | v1.101.0; experimental AudioFirst, native Hybrid and token-row LoRA scope deserve separate feature review |
| ComfyUI-Drama3 | 66c4fb9c | bc9b04a2 | 2 | Prioritize targeted adaptation described above |
| SGLang-Omni | caae7eb1 | 49dc4ba9 | 4 | Voxt VAD lifecycle fixes, CI and tests; no Drama 3 runtime change identified |
| SwiftVR | 5ca168ce | 4fb82651 | 3 | Documentation-only changes |
| H3VAE TRT | 4360e008 | same | 0 | Current |
| ModelTC/LightVAE | 38b0587d | same | 0 | Bundled adapter source current |

[ComfyUI comparison](https://github.com/Comfy-Org/ComfyUI/compare/98cc0bf84b7cb1383654c5b402e7612d875d6d22...fc04da91b2ab12f99e4716371a7e4f98ac982597), [KJNodes comparison](https://github.com/kijai/ComfyUI-KJNodes/compare/57105374f47d0fbb49c9c3926fb981702e0a4b5c...17195fcfa8a05f601b766a0f09c9733284f47416), [SGLang comparison](https://github.com/sgl-project/sglang-omni/compare/caae7eb18d8c8d22643d4cf8e377226a3b1259e9...49dc4ba9628b8a8827be060b4175afdea925c7c9), [T8 exact changelog](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/blob/6377904c07b163c6ef76dccc45559bbe64d2c06d/CHANGELOG.md).

## 4. Packages: control resolution drift before broad upgrades

Gradio **6.30.0**, frontend **1.55.17**, Kitchen **0.2.37**, and Kornia **0.8.3** match registry latest. Since the October 15 metadata snapshot, Transformers moved **5.19.0 → 5.20.1**, OpenAI SDK **3.31.0 → 3.32.0**, Modal **1.6.1 → 1.6.2**, and Pillow **12.3.0 → 13.0.0**. Their current runtime ranges admit these versions; the lightweight test requirements still pin OpenAI 3.16.2. A new registry version alone does not establish a necessary migration.

The main practical risk is reproducibility: fixed Git SHAs coexist with broad package ranges. Record a working Linux/Python 3.12 resolution, including transitive dependencies, then use it consistently in local setup and Modal. Review Pillow 13 as a major-version change before letting rebuilds adopt it without validation.

Keep the Torch 2.11.0/CUDA 13.0 stack and matched Sage wheel. Registry latest is Torch 2.14.1 and torchvision 0.29.1, but torchaudio latest remains 2.11.0: independently selecting latest does not provide a matched replacement trio. Keep NumPy 1.26.4/SciPy 1.15.3 and the HDR constraints until a coordinated compatibility review. Diffusers 0.41.0 and kernels 0.17.2 are outside our current selection policy; no immediate workflow need was established.

One policy comment needs revisiting: `h3_requirements.py` says Transformers rejects Hub 2.x, but latest Transformers 5.20.1 metadata allows `huggingface-hub>=1.31.0,<3`. Its optional kernels extra requests `kernels>=0.17.2,<0.18`, while this project pins 0.16.0 for its FP8 path. Keep our conservative Hub/kernel choices until testing the actual prompt-rewriter and model loading paths; the old comment is not sufficient justification for compatibility with today's latest Transformers. [Transformers metadata](https://pypi.org/pypi/transformers/5.20.1/json).

TensorRT 11.4.0.106 is already admitted by `>=11.2,<12`; it needs compiler/cache validation rather than a range bump. wsproto 1.3.2 and kornia-rs 0.2.0 are newer than our exact pins, with no demonstrated need to replace the working WebSocket and image-processing selections. [TensorRT metadata](https://pypi.org/pypi/tensorrt-cu13/11.4.0.106/json).

## Evidence and limits

Raw repository comparisons, package metadata and exact ComfyUI source are retained in ignored `.cache/component-audit-2026-10-17/`. Every automated repository/package lookup succeeded after running outside the network sandbox. GitHub comparison file lists may truncate at 300 files, particularly for T8; this is not an exhaustive source or security audit.

This assessment checks configured source and dependency policy, not a live deployed environment. No installs, fresh Linux dependency resolution, GPU inference, benchmarks or test suite were run. Model-registry-wide releases were not inventoried. The pre-existing October 15 audit and temporary integration directory were left unchanged.

## Implementation follow-up

Adopted ComfyUI-Drama3 `bc9b04a2` through the shared local/Modal source pin.
The graph explicitly uses `speakers="together"` and
`shorten_long_pauses=False`, preserving joint dialogue and authored pauses.
Balance/Expressive and new requests default to uncapped top-k 0; Strict
retains 30. The SGLang payload translates zero to its positive
whole-vocabulary sentinel (`1 << 30`) because the pinned Drama request builder
rejects zero. Saved explicit sampling values remain supported. Local setup
now checks for the new line-head asset; its pinned 8,456-byte file was verified
as actual safetensors data, not an LFS pointer.

LightVAE now loads selected component metadata before construction and passes
native component flags and kept-block pairs. The old construct/remove/prune
implementation is removed. Tensor conversion, mixed-precision selection,
strict loading and original-component fallback remain.

72 focused CPU checks passed, covering Drama graphs, voice slots, request
validation, SGLang payloads/lifecycle, LightVAE encoder/decoder combinations,
fallback, still-image routing and UI bindings. Exact-source characterization
also checked native encoder pruning, bounded codec windows and the new code
dictionary's empty/multiple-take decoding. Upstream characterization reads
the ignored upgrade cache and skips when those sources are unavailable.
The dependency-policy self-test passed. Sandbox fixture permission errors
were resolved by running the focused checks outside the sandbox.

No full suite, GPU inference, deployment or Linux dependency lock was produced.
The lock remains deferred until a working GPU environment can be resolved and
validated. Run updated setup and restart ComfyUI, or rebuild/redeploy Modal,
to activate these changes in an existing installation.
