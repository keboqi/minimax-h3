# Component update audit — 2026-10-12

Compared all **15 configured Git source pins** with live upstream default-branch HEAD and checked **24 primary packages** on PyPI. Nine repositories have newer commits; six match. The working tree was clean at the start. The initial assessment changed no runtime pins, packages, model weights or deployments.

Following approval, the LTX pin was updated to `3bf3ca62595f1764c47d01c35c8e5dfe47e1a88f` and a separate **LTX-2.5 Tiled Upscale** method was implemented for Media and H3 finishing. It uses Refine Details, spatial fusion, matching 97-frame streaming guides, frame padding/trim, and original audio. Independent clip splitting remains an advanced fallback. Other upgrade candidates below remain recommendations; GPU quality/performance and deployed runtime validation remain outstanding.

Recommended order: stage the ComfyUI update with the single-frame decoder adaptation, evaluate Veda's explicit-failure behavior, then consider LTX tiled fusion as separate feature work. Gradio is a secondary maintenance update. Preserve the existing CUDA/Torch/NumPy stack and capture a tested Linux dependency resolution before broader package upgrades.

## 1. Stage ComfyUI with matching dependencies and adapt the 500K loader

Current `b0b743566f65daafc423b4fea8a2fbda94b3384a`; candidate `e8665ca8e4089ab917567cdb491308e1397663d4`, **43 commits ahead**. The latest published GitHub release still reports **v0.39.0**, our current release; the candidate contains subsequent default-branch changes and needs staging rather than being treated as a new stable release.

Relevant changes:

- **Qwen Image 2.1 under low VRAM:** upstream explicitly identifies a fix alongside aimdo **0.5.5 → 0.5.6**. Its commit also moves reusable prefix-cache staging allocation outside memory-compiler recording and adds stream ordering in the shared cache implementation. Test native generation/editing, prefix KV caches, repeated requests and offload. This is a reliability candidate; no speed improvement is established here. [Upstream commit](https://github.com/Comfy-Org/ComfyUI/commit/b28cb564e3aba27326257bb19d3a491d99e9fa97).
- **H3 single-latent decoding:** the stock VAE now repeats a still latent into a clip and selects its first real frame. A decoder trained for direct single-latent use needs `h3_t1_direct="true"` and `h3_t1_output_slice="3"` metadata. Our `H3SingleFrameVAELoader` overlays the 500K decoder but passes only the base VAE metadata. Updating Core without adapting this loader would route the fine-tuned decoder through the new repeated-clip path. Set the direct-decoder metadata on the in-memory composite VAE, preserving the other base metadata; retain direct decoding and its last-frame slice. [Core change](https://github.com/Comfy-Org/ComfyUI/commit/ec1537de684cac0785cd682f03db73083efcfe56), [500K reference decode](https://huggingface.co/iamkaikai/MiniMax-H3-Single-Frame-VAE-500K/blob/main/README.md).

The second finding is a concrete upgrade compatibility issue, not evidence that our current ordinary image output is broken. Our ordinary image graph samples at least five pixel frames and then selects output frames; the optional 500K graph explicitly isolates one latent slice. The metadata adaptation belongs to that optional loader. TensorRT has its own decode implementation and does not automatically inherit the native VAE change.

Move these dependencies together:

| Dependency | Current | Candidate Core requires |
|---|---|---|
| Frontend | 1.53.10 | **1.55.14** |
| Workflow templates | 0.11.76 | **0.11.78** |
| aimdo | 0.5.5 | **0.5.6** |
| Comfy Kitchen | 0.2.37 | 0.2.37 |
| Embedded docs | 0.5.13 | 0.5.13 |
| cryptography | No direct Core requirement | New direct requirement |

Do not substitute registry-latest frontend 1.55.16 for the reviewed Core requirement. Update shared constants, their self-tests, README and cached upstream contract fixtures together. [Candidate requirements](https://github.com/Comfy-Org/ComfyUI/blob/e8665ca8e4089ab917567cdb491308e1397663d4/requirements.txt).

Other migration work matters: Core changes the linear/fusion dispatch API and bypass-LoRA handling; review our quantized LightX2V loader, fused projections and refinement/compiler guards against those changes. The native H3 transformer file itself is byte-identical between our pin and candidate, but shared operations and patching are not. Core also enables its asset system by default and adds `--disable-assets`; explicitly choose the policy for local and Modal launches, considering our separate workspace catalog and cold-start behavior. Later commits disable output-directory scanning, so assess the final candidate behavior rather than assuming every output is rescanned. [Full comparison](https://github.com/Comfy-Org/ComfyUI/compare/b0b743566f65daafc423b4fea8a2fbda94b3384a...e8665ca8e4089ab917567cdb491308e1397663d4).

Validation before adopting: existing CPU contracts and browser/proxy checks; fresh Linux/Python 3.12 build; GPU tests for native versus 500K image decoding, H3 INT8/LynnReal VAE offload, quantized Turbo/refinement, Qwen cache/offload, and SeedVR2/LTX attention. Measure Modal readiness and repeated-generation memory behavior.

## 2. Evaluate Veda 0.3.1 for explicit failures and OOM recovery

Current `8d7ac153`; candidate `67e4eb5541b648555563144401ebaba832bd73ec`, five commits ahead. This adds AMD RDNA3/RDNA4 support, which does not benefit our NVIDIA deployment, plus meaningful changes to error handling:

- Missing or failing kernels stop execution with an explanation, rather than finishing with dense attention.
- Conflicting sparse nodes, overridden attention, incompatible layouts and head shapes become explicit errors.
- CUDA OOM retains its exception type so ComfyUI can unload models. Interrupts still propagate. Deliberately configured dense layers/steps remain supported.

This improves confidence that selecting Veda actually runs it, but changes availability behavior: some requests that previously completed without sparse acceleration now fail. Stage it with error/retry/cancellation tests, and update the README's fallback description. Keep it experimental and preserve SLA as default; no new NVIDIA speedup or predictor-quality improvement was established. Validate both independent sampling stages, Ref2VA and split refinement against the complete patch chain. [Exact comparison](https://github.com/veda-sparse/Veda-on-ComfyUI/compare/8d7ac153ac215d636eaf1b5910b0536c3de9e6ed...67e4eb5541b648555563144401ebaba832bd73ec), [upstream behavior](https://github.com/veda-sparse/Veda-on-ComfyUI/blob/67e4eb5541b648555563144401ebaba832bd73ec/README.md).

## 3. LTX tiled fusion is useful feature work, with an installer blocker

Candidate `3bf3ca62595f1764c47d01c35c8e5dfe47e1a88f`, eight commits beyond our pin; unchanged from the October 2 audit. It adds per-step spatial fusion, temporal streaming and tiling-size helpers. These are worth evaluating for large/long gallery upscales and HDR restoration, particularly where current independent split clips show discontinuities. They are not automatically used by our existing graphs.

**A pin-only update would fail provisioning:** current `sync_ltx25_workflows` requires exactly ten root JSON templates. Candidate has twelve, adding `LTX-2.5_V2V_TiledFusion_Native_4K_8K.json` and `LTX-2.5_V2V_TiledFusion_Upscale.json`. Adapt the explicit inventory and synchronization contract, or deliberately select the existing supported subset; separately decide whether to install nested HDR templates.

Integration requires a sampler/guide graph, not just new files. Upstream requires matching `tile_frames` on streaming guide and fusion sampler, a fresh guide encode per temporal window, and spatially untiled guide encoding. Start with Euler; history/adaptive/ancestral samplers have documented limitations. Validate seams, frame counts, audio, cancellation and peak memory. HDR EXR defaults change from ACEScct to ACEScg; specify the intended color space explicitly if exposing EXR output. Our current MP4 path does not gain an EXR feature from this upgrade. [Comparison](https://github.com/Lightricks/ComfyUI-LTXVideo/compare/f8387c893de3f652c3a416052c82bba593edcf2b...3bf3ca62595f1764c47d01c35c8e5dfe47e1a88f), [sampler contract](https://github.com/Lightricks/ComfyUI-LTXVideo/blob/3bf3ca62595f1764c47d01c35c8e5dfe47e1a88f/tiled_fusion_sampler.md).

## 4. Secondary updates and components to retain

| Component | Update | Decision |
|---|---|---|
| Gradio | 6.29.1 → **6.30.0** | Reasonable maintenance candidate, lower priority. Fixes reveal/open rendering for hidden accordions and Markdown LaTeX restoration; adds ImageEditor alpha preservation and Workflow features. We use `gr.Image`, so the ImageEditor fix does not establish an improvement to our reference-upload alpha path. Move runtime/test pins together and run settings, mode-switching, uploads, gallery and custom queue-transport browser checks. [Release](https://github.com/gradio-app/gradio/releases/tag/gradio%406.30.0). |
| Spectrum H3 | 0.2.28 → **0.2.29** | Optional. Restores forecasting compatibility with Core BSA/Flow/Untwist across source revisions and strengthens history identity. Our explicit PlagueKind SLA, Sol and Veda paths are different integrations; no direct current-path speedup established. Upstream has CPU compatibility coverage but no broad decoded GPU equivalence guarantee. [Release notes](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/blob/806fe498d574a726fecb8818917f00a4b993fd77/RELEASE_NOTES.md). |
| T8 Audio | 1.85.0 → **1.98.2** | Retain for the present voice feature. Exact AST of `MiniMaxH3AudioConditioningT8` is unchanged. The conditioning helper has new opt-in recipe/reference/marker branches; it is no longer byte-identical. New FreeVideo, temporal-dialogue/resume and Veda R2VA recipes require separate integration and model/runtime review. Do not infer tested gains for our existing voice path. [Comparison](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/compare/6063fafbd9c3b85c5ff40aef435ae11b2844e558...359a3569cbb58bdaf9ba5c459fb4729d4cc84975). |
| ControlNet Aux | 1.1.5 → **1.1.6** | Conditional: MediaPipe face-mesh and AP10K animal-pose JSON fixes. Requirements switch Albumentations to AlbumentationsX and alter native dependencies. Resolve against NumPy 1.26 before adoption; no H3/Qwen generation speed benefit established. [Comparison](https://github.com/Fannovel16/comfyui_controlnet_aux/compare/59b1fc411ede8623b2997855b8018f0b3b6cf49f...a587a4c1594524c1d2de5cd1a86cd868ff448e1c). |
| KJNodes | 1.5.1 → **1.5.2** | Low priority: version metadata and removal of HTML from resize progress text. |
| H3 latent upscaler | One new commit | MIT license file only; no runtime change. |
| SwiftVR | Three new commits | README changes only; pointing to LightX2V does not upgrade our `SwiftVRPipeline` runtime. |

Larry Turbo, PlagueKind SLA, the Sol public fork, Spectrum Qwen, Video Depth Anything and H3VAE TRT match their configured upstream HEADs. No source upgrade is indicated for those repositories. Existing removed/retired accelerator choices should not be reintroduced merely because another node pack includes them.

## Package policy

These are registry observations, not installed deployment versions or verified compatible upgrades. Records include Python/dependency metadata in the ignored audit cache.

| Package | Configured policy | Registry latest | Assessment |
|---|---|---|---|
| Torch / torchvision / torchaudio | 2.11.0 / 0.26.0 / 2.11.0, cu130 | 2.14.1 / 0.29.1 / 2.11.0 | Keep coherent ABI stack and the matching Sage 2.2.0 cp312 wheel. |
| NumPy / SciPy | 1.26.4 / 1.15.3 | 2.5.4 / 1.18.1 | Keep native/HDR compatibility pins. |
| huggingface-hub | >=1.5,<2 | 2.2.0 | Keep cap pending coordinated validation. Latest Transformers metadata now allows Hub <3, but latest Diffusers still requires <2; the existing code comment is version-dependent. |
| Transformers | >=4.57.1 | 5.19.0 | Choose an exact tested version with Hub, kernels and the local prompt model; current range permits major drift. |
| Diffusers | >=0.36,<0.37 | 0.41.0 | Keep cap until SwiftVR/transitive node imports and loading are validated. |
| kernels | 0.16.0 | 0.17.2 | Coordinate with Transformers FP8 support. Latest Transformers' kernels extra specifies >=0.17,<0.18; this does not alone prove our runtime needs a bump. |
| Comfy Kitchen | 0.2.37 | 0.2.37 | Current; consume matching Core requirement. |
| Frontend | 1.53.10 | 1.55.16 | Candidate Core requires 1.55.14. |
| Gradio | 6.29.1 | 6.30.0 | Secondary maintenance candidate above. |
| Kornia / Kornia RS | 0.8.3 / 0.1.14 | 0.8.3 / 0.2.0 | Keep paired LTX-compatible pins. |
| accelerate / peft / safetensors | >=1.12 / >=0.18 / >=0.7 | 1.15.0 / 0.21.2 / 0.8.0 | Already allowed; record tested resolution. |
| TensorRT cu13 | >=11.2,<12 | 11.4.0.106 | Already allowed; validate engine build, reuse and decode before standardizing. |
| wsproto | 1.2.0 | 1.3.2 | Optional after proxy/WebSocket checks. |
| OpenAI SDK | Runtime >=3.16.2,<4; tests 3.16.2 | 3.29.0 | Already allowed in runtime; capture a common tested resolution. |
| Modal | No central exact pin | 1.6.1 | Record SDK used for snapshots/build/deploy; no required API migration established. |
| Pillow / aiohttp / onnx | >=10 / >=3.11,<4 / >=1.19,<2 | 12.3.0 / 3.14.5 / 1.23.2 | Already allowed; record deployment resolution. |

Fixed Git SHAs plus broad Python ranges do not reproduce builds. Save the tested Linux/Python 3.12 dependency resolution for local and Modal provisioning, especially Transformers/Hub/kernels/PEFT and TensorRT. The Windows UI test environment cannot establish CUDA compatibility. [PyPI metadata example](https://pypi.org/pypi/transformers/5.19.0/json).

## Source inventory

| Component | Configured pin | Observed HEAD | Commits ahead |
|---|---|---|---:|
| ComfyUI | b0b74356 | e8665ca8 | 43 |
| ControlNet Aux | 59b1fc41 | a587a4c1 | 7 |
| H3 latent upscaler | d7c01b90 | 40316cf0 | 1 |
| KJNodes | 57105374 | d7641533 | 2 |
| Larry Turbo | 4274783a | same | 0 |
| LTXVideo | f8387c89 | 3bf3ca62 | 8 |
| PlagueKind SLA | d58d006a | same | 0 |
| Veda | 8d7ac153 | 67e4eb55 | 5 |
| Sol public fork | 930a4d6e | same | 0 |
| Spectrum H3 | 5161f045 | 806fe498 | 1 |
| Spectrum Qwen | ddb5470f | same | 0 |
| Video Depth Anything | a0db08e6 | same | 0 |
| T8 Audio | 6063fafb | 359a3569 | 20 |
| SwiftVR | 5ca168ce | 4fb82651 | 3 |
| H3VAE TRT | 4360e008 | same | 0 |

## Evidence and validation limits

Raw repository comparisons, package metadata, exact source snapshots and characterization results are in ignored `.cache/component-audit-2026-10-12/`. GitHub comparison file lists can cap at 300 entries (T8 reaches that limit); the base conditioning class/helper were fetched independently at both revisions rather than inferred from absence in the comparison.

CPU characterization executed the candidate native VAE's extracted `decode` method with shape stubs: absent direct metadata expands one latent to five; `still_frame=3` keeps one latent and selects slice 3. It also exercised our real workflow synchronization function against the twelve candidate filenames and reproduced its rejection of the two new templates. Exact-source checks confirmed the unchanged native H3 transformer bytes and unchanged T8 base-conditioning class AST.

These checks establish routing/contracts, not generated-image quality, CUDA numerical equivalence, speed or memory. No GPU inference, model download, package installation, fresh Linux build, deployment, remote installed-package inventory or comprehensive security audit was performed. The application regression suite was not rerun for this documentation-only assessment.
