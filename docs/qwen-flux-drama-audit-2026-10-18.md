# Qwen Image 2.1, FLUX 3 and Drama 3 update assessment

Implementation follow-up: [remaining engine adaptations](remaining-engine-updates-2026-10-18.md) records the coordinated source upgrades, native Qwen enhancement, Drama models/ASR/reference cache and FLUX Base/layout controls now implemented.

Follow-up: the [official FLUX update adaptation](flux3-official-update-2026-10-18.md) implements the relevant FLUX changes and corrects this assessment's release discovery: GitHub Releases omits the newer maintenance tags, including v0.41.3. The Core/Drama cache migration described below concerns unreleased main, not that maintenance tag.

Checked 2026-10-18 against the current working tree and live public upstream metadata. This supersedes the source-status findings in the October 16 Drama/FLUX audit. Runtime code, dependency pins, installed models and deployments were not changed.

**Recommendation:** fix the FLUX download paths first; evaluate its changed sampling schedule next. Upgrade the Drama node pack for its CUDA optimizations after focused inference validation. Treat the newer ComfyUI/Kitchen pair as a coordinated upgrade candidate, with the newer Drama pack required for compatibility.

## Observed source status

| Component | Configured | Observed upstream | Decision |
|---|---|---|---|
| ComfyUI | `98cc0bf8` (v0.41.0) | `c9ea36b9`, 14 commits ahead; latest published release still v0.41.0 | Useful unreleased Qwen changes; evaluate with Kitchen 0.2.38 and updated Drama pack. |
| comfy-kitchen | 0.2.37 | 0.2.38 | Keep paired with the corresponding Core source. Python 3.12 Windows/Linux wheels exist. |
| ComfyUI-Drama3 | `bc9b04a2` | `67a733a9`, one commit ahead | Substantial performance and new-Core compatibility candidate. |
| SGLang-Omni | `caae7eb1` | `53e0bfdc`, 20 commits ahead | Shared serving changes deserve a separate integration check; no Drama-specific inference fix identified in this comparison. |
| Spectrum Qwen Proper | `ddb5470f` | Same | No source upgrade. |
| Official Drama checkpoint | `e0a9218a` | `b2b5ee99` | Only README changed: new vLLM-Omni instructions. Keep checkpoint pin. |
| LightX2V Qwen Lightning | `6b0a1176` observed previously | Same | No replacement adapter. |
| Viggle Turbo repository | v0.2.1/v0.3 adapters registered | `009a44a8`, last modified October 1 | Current registry already includes v0.3 six/nine-step modes. |

Evidence: [Core comparison](https://github.com/Comfy-Org/ComfyUI/compare/98cc0bf84b7cb1383654c5b402e7612d875d6d22...c9ea36b9d2140c89b2ef84f93f428df474e72ba5), [latest release observed](https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.41.0), [Drama comparison](https://github.com/PedroMarinhoDev/ComfyUI-Drama3/compare/bc9b04a2acf1c553bec3b5e1d2339093eb826396...67a733a96890637dc27b1914974ad00c569405fb), [Omni comparison](https://github.com/sgl-project/sglang-omni/compare/caae7eb18d8c8d22643d4cf8e377226a3b1259e9...53e0bfdcc9586fdcda055a6507a64356401a4060), [Kitchen metadata](https://pypi.org/pypi/comfy-kitchen/0.2.38/json).

## 1. FLUX: download paths need adaptation

The public model inventory at revision `feef60b0` no longer contains either diffusion filename configured in `h3_models.py`:

| Model | Configured remote filename | Current remote filename |
|---|---|---|
| Distilled NVFP4 | `diffusion_models/flux3_image_klein_sd_nvfp4.safetensors` | `diffusion_models/flux3-image-klein-nvfp4.safetensors` |
| Distilled BF16 | `diffusion_models/flux3_image_klein_sd_bf16.safetensors` | `diffusion_models/flux3-image-klein-bf16.safetensors` |

The new filenames have **exactly the SHA256 values already pinned** in our registry. This is a source-path migration, with no evidence of changed weight contents. Fresh provisioning through our model manager will fail its filename lookup with the old inventory entries. Existing validated local files can remain usable.

When implementing, update remote paths while retaining the old local names through `local_filename`. Also preserve existing downloaded files through manifest reconciliation: `model_file_matches_manifest` compares the recorded remote filename, so a rename alone can mark cached files stale. Reuse should be based on verified content identity. Check fresh provisioning metadata, both graph loader names and an existing old manifest; no model download is needed to test those contracts.

The text encoder, VAE and prompt upsampler still exist and match all three configured hashes. Upstream also offers distilled INT8 ConvRot (~5.83 GB) and a W4A8 prompt upsampler (~6.72 GB), useful optional choices for memory-constrained hardware. These are feature additions, not necessary replacements.

Evidence: [official model inventory](https://huggingface.co/api/models/black-forest-labs/flux-3-image-klein-comfy?blobs=true). The inventory returned names and hashes anonymously; fetching the gated README and commit history returned HTTP 401. Rename timing and rationale were therefore not established.

## 2. FLUX: sampling defaults have diverged

BFL reference commit `62e90b92` changes the default timestep parameter **alpha 6.93 → 16**, shared by base and distilled generation. It also changes **base CFG 3.5 → 4**. Our integration uses distilled weights at 16 steps and CFG 1, so the base CFG change does not apply today.

The alpha change does apply to the reference distilled path. Pinned ComfyUI's `Flux3.sampling_settings` still sets `shift=1.93586`, approximately `ln(6.93)`. Our graph uses the inherited schedule through `KSampler` with `simple` and provides no override. Current Core HEAD's supported-model diff does not update the FLUX setting either: a generic Core upgrade alone will not resolve this mismatch.

Evaluate an explicit FLUX schedule using alpha 16 (Comfy's logarithmic shift approximately 2.772589), preserving the distilled flow-map model behavior. Compare identical prompts/seeds, text rendering, reference fidelity and several canvas sizes before changing the default. An explicit sigma schedule can avoid depending on the inherited Core setting. Upstream intent is clear; improved quality in this app remains unmeasured.

BFL also published external-VLM prompts for text-to-image, editing and bounding-box tasks. Our remote writer already follows the same broad approach; review the new instructions for useful differences. Optional layout/region controls remain unexposed. The previous audit's blanket suggestion that edits always need a complete scene recaption should be refined: BFL's external edit prompt asks for a self-contained editing instruction, consistent with our current edit writer.

Evidence: [reference parameter change](https://github.com/black-forest-labs/flux-3-image-klein/commit/62e90b9278), [pinned Core FLUX settings](https://github.com/Comfy-Org/ComfyUI/blob/98cc0bf84b7cb1383654c5b402e7612d875d6d22/comfy/supported_models.py#L845), [external-VLM prompt addition](https://github.com/black-forest-labs/flux-3-image-klein/commit/458dcc00e4).

## 3. Drama: prioritize the new CUDA implementation

`67a733a9` adds custom NVRTC expert kernels, CUDA graph replay, a shared expert-cache session across takes, and CPU computation for INT8 cache misses. Upstream reports an RTX 3090 INT8 example improving from 68.8 to 23.2 ms per frame (a nine-second line from 16.7 to 6.7 seconds), and FP8 from 67.6 to 24.8 ms per frame. These are the author's hardware measurements, not this project's results.

The implementation detects both the old and new Comfy Qwen decode APIs, so this is a candidate node-pack-only upgrade on our current v0.41.0 Core. The commit message mentions post-v0.41.2 compatibility, but the currently observed latest Core release is v0.41.0; the compatibility branch in the code is the relevant evidence. CUDA kernels use `torch.cuda._compile_kernel`; missing NVRTC/helper support falls back rather than guaranteeing acceleration. Validate our Torch 2.11/CUDA 13 environment and target GPU, both resident and RAM experts, repeated cloned takes, dialogue, long text and cancellation.

Our graph's explicit `speakers="together"` and `shorten_long_pauses=False` still match the new node schema. Keep those policies during the upgrade. No FP8/codec model replacement is required: both configured hashes still match. INT8 is especially worth exposing for smaller GPUs with RAM experts; W6A8 and 0.6B ASR remain optional. Keep FP8 as a quality baseline and measure before changing defaults.

Evidence: [exact Drama revision and benchmarks](https://github.com/PedroMarinhoDev/ComfyUI-Drama3/blob/67a733a96890637dc27b1914974ad00c569405fb/README.md), [kernel availability check](https://github.com/PedroMarinhoDev/ComfyUI-Drama3/blob/67a733a96890637dc27b1914974ad00c569405fb/drama3/kernels.py), [decode compatibility branch](https://github.com/PedroMarinhoDev/ComfyUI-Drama3/blob/67a733a96890637dc27b1914974ad00c569405fb/drama3/model.py).

## 4. Qwen/Core: coordinated upgrade, optional native prompt enhancement

Core `4d7b5945` optimizes Qwen autoregressive decoding and changes KV-cache APIs; `ce1e922d` fixes Qwen thinking templates and reasoning extraction in `TextGenerate`. This is directly relevant to our FLUX local Qwen3.5 upsampler and potentially Qwen conditioning performance. Do not interpret text-generation benchmarks as image denoising speedups.

**Do not upgrade Core while retaining the current Drama pack.** Current Drama imports `FixedKVBias`, which is absent from the observed new Core `llama.py`; the newer pack handles `FixedKVCache` and deferred Qwen3.5 state. Pair new Core with Kitchen 0.2.38 and the newer Drama pack. Frontend remains 1.55.17; no Torch-stack bump is indicated by these Core requirements. Broader shared encoder changes warrant checking other enabled engines too.

Qwen's published native prompt-enhancement weights include separate T2I/I2I models in INT8 and W4A8, alongside full Turbo checkpoints and Union ControlNet patches. Our app already has a remote writer and the official eight-step Turbo LoRA, but does not register these additional components. Native local prompt enhancement is the strongest optional Qwen adaptation; it needs separate model selection and image-aware edit wiring. Full Turbo is an alternative loading path, not a mandatory quality upgrade over the existing extracted adapter. ControlNet requires a dedicated graph/UI feature.

All currently configured Qwen model filenames still exist. Official Turbo and Lightning adapter hashes match our pins. Lightning discussion 3 remains open without an author-provided drift fix; discussion 5 now includes another user reporting leftward edit drift above four steps. Our six-step reduced-strength option already exists and should remain a community experiment, not a resolved alignment fix. Spectrum is unchanged.

Evidence: [Qwen performance PR](https://github.com/Comfy-Org/ComfyUI/pull/17007), [template fix](https://github.com/Comfy-Org/ComfyUI/commit/ce1e922d86), [Qwen component card](https://huggingface.co/Comfy-Org/Qwen-Image-2.1), [Lightning drift discussion](https://huggingface.co/lightx2v/Qwen-Image-2.1-Lightning/discussions/3), [six-step discussion](https://huggingface.co/lightx2v/Qwen-Image-2.1-Lightning/discussions/5).

## 5. Drama serving: defer a blanket backend upgrade

The official checkpoint's newer revision changes README only, adding vLLM-Omni serving instructions; weights, tokenizer, config and codec metadata match the pinned revision. No 44 GB redownload is justified.

SGLang-Omni's 20-commit delta includes idle-stage scheduling, stage-owned HTTP routes and shared sampling seed uploads. The Fish-S2-Pro optimization is for a different model; it is not evidence of a Drama speedup. No Drama model implementation file changed in the comparison, although Drama tests and shared serving code did. If adopting it, validate speech requests, seeds/frame limits, cancellation and our sleep/wake lifecycle separately. The new vLLM instructions establish an alternative backend to evaluate, not a reason to replace our existing one automatically.

Evidence: [official checkpoint history](https://huggingface.co/fishaudio/drama-3/commits/main), [SGLang exact comparison](https://github.com/sgl-project/sglang-omni/compare/caae7eb18d8c8d22643d4cf8e377226a3b1259e9...53e0bfdcc9586fdcda055a6507a64356401a4060).

## Scope and validation limits

This assessment used live GitHub comparisons, exact source snapshots, Hugging Face filenames/LFS hashes and PyPI metadata. Raw evidence is retained in ignored `.cache/image-drama-audit-2026-10-18/`. No packages or model weights were downloaded or executed, no GPU inference was performed, and no test suite was run. Suggested speed/quality gains require target-hardware validation. Base Qwen and Viggle entries lack configured SHA256 pins, so current file existence alone does not establish historical weight identity.
