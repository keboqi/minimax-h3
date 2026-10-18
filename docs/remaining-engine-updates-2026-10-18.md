# Remaining Qwen, FLUX and Drama adaptations — 2026-10-18

Implemented the applicable upgrades and feature adaptations from the recent component assessments. These are repository changes; no deployment, model download, fresh GPU environment build or GPU inference was performed.

## Sources and compatibility

| Component | New pin | Why |
|---|---|---|
| ComfyUI | `c9ea36b9d2140c89b2ef84f93f428df474e72ba5` | Qwen decoding optimizations and thinking-template/output fixes; workflow templates 0.11.89. |
| comfy-kitchen | `0.2.38` | Exact dependency required by that Core source. |
| ComfyUI-Drama3 | `67a733a96890637dc27b1914974ad00c569405fb` | CUDA expert kernels, graph replay, and compatibility with the new Qwen cache APIs. |

Core and Drama heads were rechecked against the audited revisions and matched. The source pins are coordinated: the new Core replaces `FixedKVBias` with `FixedKVCache`, and Drama detects the corresponding deferred Qwen3.5 decode interface. Frontend, Torch/CUDA, NumPy, Transformers policy and the isolated SGLang backend pins remain unchanged. Upstream performance results are not measurements of this application.

Evidence: [Core source](https://github.com/Comfy-Org/ComfyUI/tree/c9ea36b9d2140c89b2ef84f93f428df474e72ba5), [Core requirements](https://github.com/Comfy-Org/ComfyUI/blob/c9ea36b9d2140c89b2ef84f93f428df474e72ba5/requirements.txt), [Drama source](https://github.com/PedroMarinhoDev/ComfyUI-Drama3/tree/67a733a96890637dc27b1914974ad00c569405fb).

## Drama

- Added ComfyUI model choices for original FP8, INT8 ConvRot and W6A8. FP8 remains the default; a generation downloads only its selected checkpoint and the shared codec. INT8 is useful for evaluating the new RAM-expert/CPU path, while W6A8 reduces system memory requirements.
- Added 0.6B Qwen3-ASR alongside the existing 1.7B default. Every speaker's transcription button uses the shared selector and downloads only the selected ASR model.
- Added persistent encoded-reference caching under the existing private workspace, outside the public gallery. Keys include waveform content and shape, sample rate, exact transcript/speaker tag, codec SHA256 and encoding-node revision. Cache files contain int16 codec codes and transcript, matching upstream's voice-file representation. Writes are atomic; unreadable caches are re-encoded and unavailable storage falls back to normal encoding. Original clips and saved speaker presets remain intact for SGLang.
- Preserved joint dialogue generation and authored pauses (`speakers="together"`, `shorten_long_pauses=False`).

The SGLang official checkpoint remains pinned: the newer model revision changes documentation only. The audited SGLang delta did not identify a Drama-specific inference fix. A vLLM-Omni replacement remains a separate backend evaluation, as recommended by the audit.

## Qwen Image 2.1

Added an optional **Native prompt enhancer** control, default Off. It selects the separate official text-to-image or editing PE checkpoint in INT8 or W4A8. Models download only when selected. The generation graph uses the official workflow instructions, thinking mode, top-k 20, 4096 output tokens, repetition penalty 1 and the task-specific presence penalty.

Editing passes references individually and in order, preserving different dimensions. The image-conditioning branch still reconstructs the source alpha channel. A fixed writer seed allows image-seed rerolls to reuse the same expanded prompt; separate-image edits get separate captions for each source. Incomplete reasoning never becomes an image prompt. Output dimensions remain controlled by the existing UI. Existing remote writers remain available; keep native enhancement Off after rewriting remotely to avoid rewriting twice.

The instructions are bundled in `h3_app/workflows/qwen_prompt_instructions.py`, so the existing local/Modal package copy includes them. All four PE model entries have published SHA256 pins. Existing generation API arguments retain their order; the new option is appended with a default.

Evidence: [official model components](https://huggingface.co/Comfy-Org/Qwen-Image-2.1), [native generation workflow](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_t2i.json), [native edit workflow](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_image_edit.json).

No replacement Lightning weight was found, and upstream still has no verified alignment fix. The existing community six-step option remains experimental. Full Turbo checkpoints and Union ControlNet are alternative model/conditioning features, not required replacements for the current Turbo LoRA; no automatic substitution was made.

## FLUX 3

- Added the W4A8 prompt upsampler as an optional lower-memory choice. Existing INT8 remains the default and is not downloaded alongside W4A8.
- Added Base BF16, INT8 and NVFP4 checkpoints with published hashes. Selecting Base sets 50 steps/CFG 4; selecting distilled restores 16 steps/CFG 1. Base uses real empty-text conditioning for the negative branch. Both retain the explicit alpha-16 schedule, avoiding inherited Core defaults.
- Added validated layout boxes for generation and reference-guided editing. The advanced JSON control accepts descriptions and `[top, left, bottom, right]` coordinates on a 0–1000 grid, plus optional text elements. The native `Flux3LayoutPrompt` appends the layout after local expansion, so the writer cannot discard it. Invalid geometry, types, nonfinite coordinates and unsupported fields are rejected before downloads. Box descriptions can refer to `<ref_image_0>`, etc. Box placement guides composition; this is not a pixel mask or a source-region editor.
- New model/sampling preferences persist in browser settings. Layout descriptions are treated as prompt content: they are excluded from saved preferences and provenance settings, which record only the box count.

Evidence: [official component inventory](https://huggingface.co/api/models/black-forest-labs/flux-3-image-klein-comfy?blobs=true), [BFL current reference sampling parameters](https://github.com/black-forest-labs/flux-3-image-klein/commit/62e90b9278b73ecc52927770c6965dc6b1966498).

## Validation

- 123 existing focused tests passed across FLUX, Drama, Qwen Image 2.1, Drama serving and component loading contracts.
- 13 new update tests passed, covering download scope, model selection, Base guidance/conditioning, native layout serialization, Qwen thinking output/image order, voice-cache reuse/invalidation/corruption recovery and the exact upstream cache/dependency APIs. Drama's changed graph tests also pass.
- 49 UI contract checks passed, including the new selectors and persistence. Four H3 UI test methods were excluded after reproducing their failures on an untouched `git archive HEAD` baseline: encoder-attention input position, FL2VA voice-input positions, refinement-LoRA input position and the published H3 API fixture. The baseline also had an older Qwen enum fixture; only Qwen's fixture was refreshed for this change.
- Model inventory and requirements selftests pass. Python compilation and patch-whitespace checks complete the verification; no full repository suite was run.

Raw source snapshots and the baseline reproduction are retained in ignored `.cache/remaining-updates-2026-10-18/`. GPU speed, voice quality, image quality and actual CUDA-kernel availability remain unverified on deployment hardware. The updated pins and custom nodes take effect on the next normal setup/image rebuild and backend restart.
