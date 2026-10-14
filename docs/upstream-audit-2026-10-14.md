# Component update audit — 2026-10-14

Refreshed all 15 generation/provisioning Git source comparisons from the October 12 audit and checked 25 primary packages on PyPI. Of these 15 sources, nine have newer default-branch commits and six match their pins. This assessment changes no runtime pins, models, installed packages or deployments.

**Recommended order:** adopt ComfyUI v0.40.0 with the single-frame decoder adaptation and matching dependencies; evaluate Veda 0.3.1 separately; then update Gradio. Keep the CUDA/Torch/NumPy stack. LTX tiled fusion is already implemented here, so the October 12 feature recommendation is complete.

**Implementation follow-up:** ComfyUI v0.40.0, frontend 1.55.17, Veda 0.3.1 and Gradio 6.30.0 are now pinned in the shared local/Modal provisioning policy. The 500K loader preserves base metadata and sets direct-decoder routing and output slice 3. Runtime and test Gradio pins move together. Core requirements supply templates 0.11.82, docs 0.5.14 and aimdo 0.5.6. Other upgrade candidates below remain deferred. Focused CPU and browser results are recorded at the end; no GPU deployment has been performed.

## 1. ComfyUI v0.40.0 is now the main upgrade candidate

The previous audit's unreleased candidate has become a published release. Use **v0.40.0**, immutable revision `3541246a9c997f4b8c32f9f88f3c985d02021c79`, rather than latest default-branch HEAD `e0a32bce84f4603f2b61b16b50f8958e68fd6171`. The release was published October 14 at 05:29 China time. HEAD is two commits beyond it, adding an SDR-to-HDR image node and a default-workflow route; neither is needed for the existing integration. [Release](https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.40.0).

The Qwen Image 2.1 low-VRAM fix and shared cache/offload changes remain relevant. The H3 single-image decode change still requires an adaptation in our `H3SingleFrameVAELoader`: after overlaying the 500K decoder, preserve base metadata and add string values `h3_t1_direct="true"` and `h3_t1_output_slice="3"` to the composite VAE metadata. The current loader forwards only base metadata, while the new native loader uses those keys to select direct single-latent decoding. This is an upgrade compatibility requirement for the optional 500K path, not evidence of a current-pin failure. TensorRT uses its own implementation. [Native loader](https://github.com/Comfy-Org/ComfyUI/blob/3541246a9c997f4b8c32f9f88f3c985d02021c79/comfy/sd.py), [500K reference](https://huggingface.co/iamkaikai/MiniMax-H3-Single-Frame-VAE-500K/blob/main/README.md).

Move the source and its matching dependency set together:

| Component | Current | v0.40.0 requires |
|---|---|---|
| Frontend | 1.53.10 | **1.55.17** |
| Workflow templates | 0.11.76 | **0.11.82** |
| Embedded docs | 0.5.13 | **0.5.14** |
| Comfy aimdo | 0.5.5 | **0.5.6** |
| Comfy Kitchen | 0.2.37 | 0.2.37 |

These replace the October 12 candidate's frontend/template/doc versions. Core also requires `av>=17.0.0`, `simpleeval>=1.0.0`, `blake3` and `comfy-angle`; resolve the full release requirements under our ABI constraints rather than copying only this table. The temporary direct `cryptography` requirement has been removed. [Exact requirements](https://github.com/Comfy-Org/ComfyUI/blob/3541246a9c997f4b8c32f9f88f3c985d02021c79/requirements.txt).

The asset-system assessment also changes: **assets are off by default** for our ordinary Linux/Modal launches. `--enable-assets` opts in; Windows standalone builds enable them, and `--disable-assets` overrides that. Avoid importing the earlier candidate's default-on assumption into this upgrade. [Release CLI policy](https://github.com/Comfy-Org/ComfyUI/blob/3541246a9c997f4b8c32f9f88f3c985d02021c79/comfy/cli_args.py).

Before adoption, review shared Linear/fusion and bypass-LoRA changes against our quantized Turbo loader, fused modulation and refinement/compiler guards. Update shared constants, self-tests, README and cached upstream contracts together. Run focused CPU contracts, browser queue/proxy checks, a fresh Linux/Python 3.12 dependency resolution, and GPU comparisons covering native/500K images, H3 VAE offload, Turbo/refinement and Qwen cache/offload. CPU source inspection cannot establish CUDA quality or performance.

## 2. Veda and Gradio remain useful secondary upgrades

**Veda 0.3.1:** candidate `67e4eb5541b648555563144401ebaba832bd73ec` is unchanged since October 12. Its explicit failure behavior prevents broken sparse configurations from silently finishing with dense attention, and preserves CUDA OOM typing for ComfyUI recovery. This improves failure visibility but can turn formerly completed requests into errors. Validate error/retry/cancellation, both sampling stages and Ref2VA; update the README's fallback description. Preserve SLA as default. No new NVIDIA performance or predictor-quality benefit was established. [Comparison](https://github.com/veda-sparse/Veda-on-ComfyUI/compare/8d7ac153ac215d636eaf1b5910b0536c3de9e6ed...67e4eb5541b648555563144401ebaba832bd73ec).

**Gradio 6.29.1 → 6.30.0:** maintenance candidate for hidden-accordion reveal/open rendering and Markdown LaTeX restoration. Move runtime and test pins together, then check settings, mode changes, uploads, gallery and custom WebSocket queue transport. ImageEditor alpha fixes do not demonstrate a benefit to our `gr.Image` upload path. [Release](https://github.com/gradio-app/gradio/releases/tag/gradio%406.30.0).

## 3. New LTX and T8 commits do not justify immediate adoption

**LTXVideo:** `7f22eb16ebe33757bac7ded283ebb4acaee180a7` is two commits beyond our newly adopted tiled-fusion pin. It removes legacy guide, normalization, mask, prompt-enhancer, VAE-patcher, APG and tricks nodes, adjusts example workflow inputs, and removes the Transformers `timm` extra requirement. Static comparison found no references in our runtime Python sources to the twelve removed decorated node IDs, and no removed root 2.5 workflow JSONs. Our keyframe graph uses native `LTXVAddGuide`, not the removed `LTXVAddGuideAdvanced` nodes. Tiled-fusion source files do not appear in this comparison's changed-file list.

Keep the present pin: no clear current-workflow quality, speed or reliability gain was identified. A future update still needs full official-workflow node coverage, dependency resolution and execution checks; the static absence-of-references check is not proof of complete compatibility. [Comparison](https://github.com/Lightricks/ComfyUI-LTXVideo/compare/3bf3ca62595f1764c47d01c35c8e5dfe47e1a88f...7f22eb16ebe33757bac7ded283ebb4acaee180a7).

**T8 Audio:** latest is now **1.99.0**, revision `2ccd8eb0a4160eef12f00e30b5aefda42d8be577`. The additional release introduces reviewed H11 Spark Wuli/modular reference workflows. These require deliberate feature/model integration; upgrading the node pack alone does not enable them in our voice-reference graph. Retain 1.85.0 for the current feature pending a separate review. The October 12 AST characterization applied to 1.98.2 and must not be treated as a new 1.99.0 validation. [Comparison](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/compare/6063fafbd9c3b85c5ff40aef435ae11b2844e558...2ccd8eb0a4160eef12f00e30b5aefda42d8be577).

## 4. Other sources and package policy

Spectrum H3 0.2.29 remains optional for Core BSA/Flow/Untwist compatibility; no direct gain for our explicit SLA/Veda/Sol paths was established. ControlNet Aux 1.1.6 remains conditional on its face/animal-pose fixes and AlbumentationsX/native dependency compatibility. KJNodes is cosmetic/metadata, the latent-upscaler update is license-only, and SwiftVR's newer commits are documentation-only. These observations remain unchanged from the [October 12 audit](upstream-audit-2026-10-12.md).

| Source | Configured pin | Observed HEAD | Commits ahead |
|---|---|---|---:|
| ComfyUI | b0b74356 | e0a32bce | 56 |
| ControlNet Aux | 59b1fc41 | a587a4c1 | 7 |
| H3 latent upscaler | d7c01b90 | 40316cf0 | 1 |
| KJNodes | 57105374 | d7641533 | 2 |
| Larry Turbo | 4274783a | same | 0 |
| LTXVideo | 3bf3ca62 | 7f22eb16 | 2 |
| PlagueKind SLA | d58d006a | same | 0 |
| Veda | 8d7ac153 | 67e4eb55 | 5 |
| Sol public fork | 930a4d6e | same | 0 |
| Spectrum H3 | 5161f045 | 806fe498 | 1 |
| Spectrum Qwen | ddb5470f | same | 0 |
| Video Depth Anything | a0db08e6 | same | 0 |
| T8 Audio | 6063fafb | 2ccd8eb0 | 21 |
| SwiftVR | 5ca168ce | 4fb82651 | 3 |
| H3VAE TRT | 4360e008 | same | 0 |

All 25 registry latest package versions match the previous audit except the frontend, now **1.55.17**, which matches v0.40.0. Keep Torch 2.11.0/CUDA 13.0 and its matching Sage wheel, NumPy 1.26.4/SciPy 1.15.3, Hub <2, Diffusers <0.37, kernels 0.16.0 and the paired Kornia pins until coordinated runtime validation. Avoid treating newer individual Torch ecosystem packages as a compatible set.

The strongest package-policy improvement is still recording a tested Linux/Python 3.12 resolution. Broad ranges for Transformers, PEFT, Accelerate and other dependencies permit build drift even with fixed Git SHAs. Latest versions being admitted by a range is not validation of this stack. Registry details and dependency metadata are retained in the audit cache.

## Evidence and limits

Raw GitHub comparisons, PyPI metadata, exact release source files and the LTX static characterization are in ignored `.cache/component-audit-2026-10-14/`. GitHub API rate limiting affected the final release lookups; the Gradio release was checked through its public release page instead. The T8 monolithic source lookup returned 404; this report makes no new conditioning-class equivalence claim. GitHub comparison file lists can cap at 300 entries, notably for T8.

This was a source/metadata assessment. No package installs, model downloads, deployment changes, GPU inference, fresh Linux build or broad regression suite were performed. The LTX characterization checks removed decorated IDs against runtime source text; it does not validate every runtime registration, official workflow or saved user graph.

### Upgrade validation follow-up

Gradio 6.30.0 was installed in the existing Windows/Python 3.12 test environment. The dependency-policy self-test and 140 focused tests passed, covering the overlay loader, exact upstream requirements, Veda graph routing, refinement/compiler guards, Qwen, encoder caching, standalone setup, proxy, queue and UI contracts. The new overlay regression covers absent/present metadata, preservation of architecture metadata, and replacement of the decoder tensors. A subsequent exact-source native decoder characterization verifies stock single-latent clip expansion versus direct 500K routing.

Headless Chromium acceptance passed for WebSocket progress with HTTP SSE blocked, settings/preset/reset/reload/session isolation and narrow layout, and per-action request budgets including mode/task changes. The native H3 transformer source is byte-identical to the previous configured pin. The upstream bypass hook now sets/restores `comfy_force_forward`, so the existing LightX2V adapter continues to use the upstream injection contract.

The initial sandboxed run hit Windows temporary-directory permission errors; its isolated rerun outside the sandbox passed. No complete suite was run. Linux/CUDA dependency resolution, GPU output quality, OOM recovery, memory/performance and Modal image rebuild/deployment remain unvalidated here.
