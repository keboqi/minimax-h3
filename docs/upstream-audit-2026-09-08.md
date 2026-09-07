# Upstream component audit — 2026-09-08

Compared the 13 Git repository pins used by local/Modal provisioning with upstream default-branch heads. Reviewed diffs for all six changed repositories and checked selected official/community model announcements. This is a source audit, not a GPU compatibility certification or exhaustive Python-package/security audit. Runtime pins were not changed.

## Repository inventory

| Component | Current pin | Observed head | Commits ahead |
|---|---|---|---:|
| [Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI/compare/567275141678c9fd65bafef6aa9dcb4ac9bd70e3...41db8f4fa1587d139e412a57b9b69394e3b13f95) | `56727514` | `41db8f4f` | 44 |
| [T8mars/comfyui-minimax-h3-audio-T8](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/compare/91c1b4e9b680d07a6eacee6a3aa6b449a4697554...ad43472fc85e943aed895bf2246c6554177b9717) | `91c1b4e9` | `ad43472f` | 1 |
| [Saganaki22/ComfyUI-sol-attn](https://github.com/Saganaki22/ComfyUI-sol-attn/compare/930a4d6e432ff8b8ed5e30ff2f72519b92d69bdf...930a4d6e432ff8b8ed5e30ff2f72519b92d69bdf) | `930a4d6e` | `930a4d6e` | 0 |
| [PlagueKind/ComfyUI-PlagueKind-Nodes](https://github.com/PlagueKind/ComfyUI-PlagueKind-Nodes/compare/aaec055cd642b3292df18e69824c012d345ebfe8...59f54d359bbabff8bb813b1e3e381dd29843e720) | `aaec055c` | `59f54d35` | 4 |
| [xmarre/ComfyUI-Spectrum-MiniMax-H3](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/compare/beb32dd210ef9e95520453107f158241d4f2ecf3...a360f64fbfa54681ded100a64ded86a5713ddf17) | `beb32dd2` | `a360f64f` | 3 |
| [Larryvrh/ComfyUI-MiniMax-H3-Turbo](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/compare/4274783a23afcfdbea3b4876cb79effd6c510785...4274783a23afcfdbea3b4876cb79effd6c510785) | `4274783a` | `4274783a` | 0 |
| [LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/compare/d7c01b9011f2e8439493f6c02c29995a27df276f...d7c01b9011f2e8439493f6c02c29995a27df276f) | `d7c01b90` | `d7c01b90` | 0 |
| [Lightricks/ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo/compare/15d09abb5a187a8dcaea2fc31fe51ee96e6c9d0d...15d09abb5a187a8dcaea2fc31fe51ee96e6c9d0d) | `15d09abb` | `15d09abb` | 0 |
| [kijai/ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes/compare/e8e88f7c88e3f6205b122f5de87e69a09fbce5ac...c9869eade9920a1b949de07c4a197156006bcceb) | `e8e88f7c` | `c9869ead` | 2 |
| [Fannovel16/comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux/compare/59b1fc411ede8623b2997855b8018f0b3b6cf49f...59b1fc411ede8623b2997855b8018f0b3b6cf49f) | `59b1fc41` | `59b1fc41` | 0 |
| [yuvraj108c/ComfyUI-Video-Depth-Anything](https://github.com/yuvraj108c/ComfyUI-Video-Depth-Anything/compare/a0db08e63d1ea571601c45cde4aaee0acdd0544d...a0db08e63d1ea571601c45cde4aaee0acdd0544d) | `a0db08e6` | `a0db08e6` | 0 |
| [H-oliday/SwiftVR](https://github.com/H-oliday/SwiftVR/compare/5ca168cef6ca7200f135fdfea85e5e13d12c5b53...5ca168cef6ca7200f135fdfea85e5e13d12c5b53) | `5ca168ce` | `5ca168ce` | 0 |
| [lihaoyun6/ComfyUI-H3VAE_TRT](https://github.com/lihaoyun6/ComfyUI-H3VAE_TRT/compare/7131a316160b2f299239b9bc40621be46d8ce62f...4360e00867eca86ab61b3899216c0ec281367b46) | `7131a316` | `4360e008` | 4 |

## Recommended work

1. **Stage ComfyUI and its dependency set together.** Target observed head `41db8f4fa1587d139e412a57b9b69394e3b13f95`, frontend 1.51.10, Kitchen 0.2.33, aimdo 0.5.2, workflow templates 0.11.55, embedded docs 0.5.11. Changes include optional H3 reference VAEs/text-encoder-only references, DiffSynth/ModelScope H3 LoRA loading, native sparse attention, and the Comfy Compiler. These span memory allocation and execution: validate Blackwell generation, model unloading, H3 audio, LTX and Music before deployment. Update the shared constants and frontend checks together. Native sparse attention is a candidate to benchmark, not a proven replacement for our measured Sage 2 default. [Official sparse-attention PR](https://github.com/Comfy-Org/ComfyUI/pull/16072).
2. **Include KJNodes 1.5.1 in that upgrade.** Its narrow H3 low-memory block fix accepts the new optional attention callback and only uses list handoff for its own attention path. The app currently uses KJ's Sage patch; the fixed low-memory path is not directly selected by our graph, so this is compatibility maintenance rather than a demonstrated current crash.
3. **Evaluate Spectrum 0.2.24.** It removes an extra RES exact-tail floor and fixes qualified terminal PECE transition/bootstrap behavior. Our configured tail is one step. Benefits are workflow-dependent; keep current legacy/offline-audio settings and compare matched-seed RES/Turbo output before promotion. [Release notes](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/blob/a360f64fbfa54681ded100a64ded86a5713ddf17/RELEASE_NOTES.md).
4. **Adapt SLA before adopting 1.4.8.** Our graph explicitly supplies sparsity, block size, minimum sequence length, dense-last steps and audio protection, but omits engine, QK quantization and dense_steps. Upstream now defaults to Kitchen, experimental INT8 QK and dense_steps="1" instead of "0". Kitchen requires >=0.2.32 (ours is 0.2.31) and does not preserve multiple protected ranges, reference quotas or motion stabilization. To preserve the existing behavior during upgrade, explicitly select triton, use_int8_qk=False, tail_correction=False and dense_steps="0"; benchmark Kitchen as a separate choice. Do not assume a pin-only update preserves our audio-safe behavior.
5. **Rebase TensorRT compatibility patches before upgrading.** Upstream has W4A16/memory changes, single-frame and bounds fixes, encoder fixes and a compiler script. Our fail-closed patcher matches exact source for encode/decode padding, temporal trim, FP32 normalization and optional encoder loading. Reconcile these individually with upstream; retain decoder quality safeguards and verify single-frame plus odd-length clips and engine rebuilding. A bare SHA bump is not sufficient.
6. **T8 1.74.1 is low priority for this app.** Its sole new commit changes DLSS acknowledgement handling and workflow migration; the FL2VA voice-reference conditioning implementation is unchanged in the comparison. Our existing voice feature does not need this update.

## Model/community findings

- The recent LightX2V FL2V 4-step v1.2 and 8-step 768p/Ref2V additions are already represented in our model inventory. No replacement is justified by those announcements alone. [Publisher activity](https://huggingface.co/lightx2v/collections), [model repository](https://huggingface.co/lightx2v/Minimax-h3-Turbo).
- H3-World offers action/camera-controlled generation. T8 already contains support at our pin, but the Gradio app does not expose its dedicated graph or weights. The published Comfy bundle requires full FL2VA, per-action encoding, special attention routing, and a fixed initial 832x480/124-frame/50-step contract. Treat it as a separate experimental feature, not a generic LoRA addition. [Integration author's model card](https://huggingface.co/t8star/Minimax-H3-World-Comfy).
- Sol-Attn, Larry Turbo, LTXVideo, H3 latent upscaler, ControlNet Aux, Video Depth Anything and SwiftVR pins match the observed default-branch heads.

## Local dependency drift

`requirements-test.txt` pins openai==3.8.0, while both deployment scripts require openai>=1.109,<3. Tests therefore exercise a different SDK major than deployments. Align the test environment with the intended production major or deliberately migrate and validate both installers together. This is a local consistency finding, not a verified SDK migration recommendation.

## Validation boundary

Working tree was clean at audit start. No code, dependency pins, models or deployments were changed. GPU renders, package resolution and tests were not run because this deliverable is an upgrade assessment. Before applying upgrades, run existing graph/patch regression checks, resolve a clean local/Modal image, and compare matched-seed FL2VA, voice-reference, Ref2VA, RES/Turbo, LTX and Music output on the target GPU.


## Upgrade implementation — subsequent authorized change

The six requested component updates above are now applied to local and Modal
provisioning. SLA explicitly retains Triton, non-INT8 QK, no tail correction and
step-zero dense attention. TensorRT patch v5 retains single-frame decoder and
tail-trim behavior, inserts FP32 normalization after parsing, and supplies the
missing ONNX inspection import; both installers now include ONNX. Engine marker
v4 forces a lazy rebuild of older cached engines.

Validation: 61 unittest cases ran, 58 passed and three CPU-PyTorch-dependent
cases skipped. The five new upstream contract tests ran against the downloaded
pinned sources, including decoder frame plans for 1–100 latent frames. All
existing app, dependency, model, patch, attention and prompt self-tests passed;
compilation, local/Modal pin agreement and git diff whitespace checks passed.
The model self-test's stale inventory was corrected to include the existing
Semantic Bridge. A uv dry-run resolved ONNX 1.22.0 with NumPy 1.26.4 on Python
3.12; no package was installed. Full Linux/CUDA image resolution, GPU generation
and Modal deployment have not been performed.

The optional upstream tests use `H3_UPSTREAM_SOURCE_DIR` (default:
`.cache/upstream-upgrade`) containing `minimax_trt_node.py` from the TRT pin and
`comfy-requirements.txt` from the ComfyUI pin. They skip when sources are absent.
