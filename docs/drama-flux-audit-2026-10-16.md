# Drama 3 and FLUX 3 update assessment — 2026-10-16

Compared upstream with the current working tree, including the ongoing Drama serving changes. Recommendation: keep the source pins; prioritize feature adaptations below. No runtime code, dependencies, models or deployments were changed by this assessment.

## Source and checkpoint status

| Component | Current pin | Observed upstream | Assessment |
|---|---|---|---|
| ComfyUI-Drama3 | `66c4fb9c` | Identical | No node-pack upgrade available. |
| Official Drama 3 checkpoint | `e0a9218a` | Identical | No official checkpoint upgrade available. |
| SGLang-Omni | `caae7eb1` | `2c23b68f`, 2 commits ahead | Voxt live-ASR lifecycle and CI changes; no Drama inference changes in the comparison. |
| ComfyUI | `98cc0bf8` | `bdc63e72`, 5 commits ahead | Krea API integration, FLUX display labels, quantization docs and workflow-template bumps. No local FLUX inference fix identified. |
| BFL reference inference | Not a runtime dependency | `e53f4ea6` | Post-release commits update documentation, emphasizing prompt upsampling and linking alternative runtimes. |

Evidence: [Drama comparison](https://github.com/PedroMarinhoDev/ComfyUI-Drama3/compare/66c4fb9cd3c0151e1e2ccd207847cb8335e87ab4...main), [SGLang comparison](https://github.com/sgl-project/sglang-omni/compare/caae7eb18d8c8d22643d4cf8e377226a3b1259e9...2c23b68f4f763c14cfb845eadb4883a5e791b701), [Comfy comparison](https://github.com/Comfy-Org/ComfyUI/compare/98cc0bf84b7cb1383654c5b402e7612d875d6d22...bdc63e72f5ab29fde448c161db3057827148c0ef), [BFL history](https://github.com/black-forest-labs/flux-3-image-klein/commits/main/).

## Recommended adaptations, in order

### 1. Make FLUX edit captioning easier to use correctly

BFL explicitly recommends full scene captions for editing as well as generation. Our default text-to-image path already expands prompts, and the remote edit writer already uses the references and official instructions. However, selecting references disables local expansion and still allows an unexpanded short instruction to run.

Best immediate improvement: make the existing remote rewrite step prominent for edits and explain the complete-caption requirement at the point of generation. A local image-aware edit upsampler is also feasible to investigate: the pinned `TextGenerate` accepts an image input. Single-reference support is the simplest starting point; preserving reference order and supporting differently sized multiple images require implementation and inference checks. Do not treat the current text-only expansion branch as an edit upsampler.

Touch points: `h3_ui/optional_engine_views.py`, `h3_app/workflows/flux3.py`, `h3_app/generation/optional_engines.py`. [BFL prompting guidance](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy#prompting).

### 2. Add FLUX layout and region-edit controls

The pinned Core already includes `Flux3LayoutPrompt`; our graph does not expose structured boxes. Add a validated box input, then a drawing UI if useful. Comfy captions use `[top, left, bottom, right]` coordinates on a 0–1000 grid and matching object IDs. Append the layout after caption expansion so the writer cannot accidentally discard it. Reference-region edits also need source/target box fields and reference identity.

Boxes guide composition; they are not a hard pixel mask. The BFL Python API uses normalized boxes, so conversion must be explicit. Touch points: request dataclass, graph builder, UI and prompt writer. [Pinned node implementation](https://github.com/Comfy-Org/ComfyUI/blob/98cc0bf84b7cb1383654c5b402e7612d875d6d22/comfy_extras/nodes_flux.py).

### 3. Add FLUX distilled INT8 as a model option

We offer NVFP4 and BF16. Upstream also lists `flux3_image_klein_sd_int8_convrot.safetensors`, approximately 5.8 GB, useful on GPUs that cannot run NVFP4. This is a model-registry and selection adaptation using the current loader. Keep NVFP4 for Blackwell; establish performance and image quality on the target GPU before changing defaults. [Official inventory](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy#files).

### 4. Add optional Drama INT8/W6A8 and smaller ASR

Our graph fixes the Drama checkpoint to FP8 and ASR to 1.7B. Upstream offers INT8, W6A8, W4A8 and ASR 0.6B. INT8 is a speed candidate for older GPUs; W6A8 lowers RAM requirements. Prefer evaluating those before W4A8, for which upstream reports more silent takes. The smaller ASR is a useful selectable option, provided transcripts remain complete on our reference clips.

These are candidates already supported by the pinned pack, not newly landed source fixes. Keep FP8 as the fidelity baseline and benchmark on our deployment hardware. [Pinned Drama documentation](https://github.com/PedroMarinhoDev/ComfyUI-Drama3/blob/66c4fb9cd3c0151e1e2ccd207847cb8335e87ab4/README.md).

### 5. Cache encoded Drama speaker references

Our presets store audio and transcript; graphs encode the reference again when their inputs are not cached. The pinned pack already has `Drama3SaveVoice` and `Drama3LoadVoice` for persistent codec codes plus transcript. Adapt this as an internal cache keyed by reference content, transcript and codec revision. Preserve the original audio for SGLang, which uses a different reference interface, and preserve current private preset storage. Benefit is avoiding repeated reference encoding across restarts; it does not speed up autoregressive speech generation. [Pinned voice nodes](https://github.com/PedroMarinhoDev/ComfyUI-Drama3/blob/66c4fb9cd3c0151e1e2ccd207847cb8335e87ab4/nodes.py).

## Deferred and validation limits

FLUX Base needs a distinct sampling configuration (50 steps, CFG 3.5 versus distilled 16 steps, CFG 1). It is mainly a training/research option, so lower priority than the above. Alternative FLUX serving runtimes merit a separate throughput benchmark, not an automatic backend migration.

The configured Drama FP8/codec SHA256 values match current public model metadata. FLUX metadata exposes filenames but masks gated LFS hashes, so unchanged FLUX weight contents could not be verified anonymously. No gated access was bypassed. Raw public comparisons and source snapshots are saved in `.cache/drama-flux-audit-2026-10-16/`.

This is a source and integration assessment. No models were downloaded and no GPU inference or test suite was run. Suggested performance improvements remain unmeasured in this project.
