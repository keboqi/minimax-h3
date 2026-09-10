# Component update audit — 2026-09-10

Compared all 13 repository pins against live upstream HEAD using GitHub's compare API, inspected relevant patches and exact source files, and queried PyPI metadata for 18 primary packages. Existing September 8 recommendations and the ComfyUI 0.35.0 upgrade are already applied. Working tree was clean at audit start. This is a source assessment, not a runtime or security certification; installed remote package versions were not inspected.

## Recommended order

1. **Upgrade Spectrum first: 0.2.24 → 0.2.26 (`be95adec`).** The intervening 0.2.25 change bypasses Comfy Compiler malloc graphs during Spectrum-managed H3 solver steps, addressing a reported repeated-prompt crash. Our ComfyUI pin includes the compiler infrastructure, so this is directly relevant when that path is active. 0.2.26 adds attention-backend policy/receipt tracking: transitions reset incompatible forecast history, unknown metadata falls back to actual computation, and CUDA OOM propagates. Update `SPECTRUM_REF` in both installers and version documentation. No new graph input was identified in the reviewed changes. Validate repeated requests, cancellation/error recovery, RES/Turbo, offline audio, and Sage/Sol/SLA combinations. Core BlockSparseAttention becomes actual-only without predictive backend metadata, so do not promise the same Spectrum speedup with that combination. The new provider contract does not establish that our existing Saganaki Sol plugin implements it. [Exact comparison](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/compare/a360f64fbfa54681ded100a64ded86a5713ddf17...be95adecec0b85c80d0c9fc5dd8d07386d50aaee), [release notes](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/blob/be95adecec0b85c80d0c9fc5dd8d07386d50aaee/RELEASE_NOTES.md).

2. **Stage T8 1.74.0 → 1.77.0 (`0eae2f22`) for compatibility maintenance.** Unlike the previous audit's DLSS-only update, this now includes Core attention callbacks, FinalLayer schedule/prefetch adapters, VDN ownership fixes and a conditioning probe correction. Our app uses `MiniMaxH3AudioConditioningT8` for Hybrid FL2VA voice references. Its node implementation/schema has no changes in the reviewed `nodes.py` diff; `conditioning.py` corrects the legacy first/last-frame sentinel probe. Thus no required voice graph rewrite was identified, but test node registration and first/last frame plus 1–3 voice references on our exact Core pin. The broader attention fixes mostly affect optional T8 algorithms that our graph does not currently select. Update the shared `H3_AUDIO_T8_REF`, then check imports and model-branch isolation. [Comparison](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/compare/91c1b4e9b680d07a6eacee6a3aa6b449a4697554...0eae2f22e6cb2115c9002ab6daa0c12c9636ccb1).

   New T8 features need separate integration work, beyond a pin bump:
   - Progressive 6+2 sampling needs a dedicated graph, schedule, learned latent upscaler and audio-quality checks; it resamples audio, unlike a first-pass-audio-preserving refinement route. Keep experimental: upstream has no validated 32-second route or TRT decoder integration and excludes VDN/FAST H3/SPEED/unknown wrappers and regional masks. Published timing gains are limited small-sample results on RTX 4060 Ti, not a prediction for our Blackwell deployment.
   - VDN two-pass refinement and outpainting need explicit graph/UI/model integration and their own compatibility checks.
   - DLSS frame interpolation requires separately supplied Windows runtime binaries and a Windows single-RTX path. It is not directly usable in the Linux Modal runtime. Do not add it to Modal provisioning as a generic dependency.
   [Release scope and limits](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/blob/0eae2f22e6cb2115c9002ab6daa0c12c9636ccb1/docs/RELEASE_1.77.0.md).

3. **KJNodes: optional update `c9869ead` → `57105374`.** Four commits add compiler-aware TinyVAE previews, reduce preview memory use and add MiniMax audio previews. The app currently selects `PathchSageAttentionKJ`, not PreviewOverride, so no current app fix or mandatory graph adaptation was identified. Update both installer pins if adopting it. Exposing audio previews requires wiring the audio VAE into PreviewOverride and handling its frontend output. Direct users of TinyVAE `decode_video` must adapt to `[T,3,H,W]` output; the 2D decoder now returns CPU uint8 video frames. Our app does not directly call that API. [Comparison](https://github.com/kijai/ComfyUI-KJNodes/compare/c9869eade9920a1b949de07c4a197156006bcceb...57105374f47d0fbb49c9c3926fb981702e0a4b5c).

4. **Keep ComfyUI 0.35.0 for now.** Six newer commits add/reorganize blueprints, AMD/ROCm adjustments and partner API nodes. No direct H3 NVIDIA inference fix was identified in this delta, and the exact `requirements.txt` is unchanged. If adopting HEAD later, retain its matched frontend/Kitchen dependency set and revalidate bundled workflow serving. [Comparison](https://github.com/Comfy-Org/ComfyUI/compare/40c4fcdf513a4523e39d54a9d391908af8df8171...a7b1d39d342d102f305797fb5ba12dc304d9c1f5).

## Repository inventory

| Component | Current pin | Observed HEAD | Commits ahead |
|---|---|---|---:|
| Comfy-Org/ComfyUI | `40c4fcdf` | `a7b1d39d` | 6 |
| Saganaki22/ComfyUI-sol-attn | `930a4d6e` | `930a4d6e` | 0 |
| PlagueKind/ComfyUI-PlagueKind-Nodes | `59f54d35` | `59f54d35` | 0 |
| xmarre/ComfyUI-Spectrum-MiniMax-H3 | `a360f64f` | `be95adec` | 3 |
| Larryvrh/ComfyUI-MiniMax-H3-Turbo | `4274783a` | `4274783a` | 0 |
| LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler | `d7c01b90` | `d7c01b90` | 0 |
| Lightricks/ComfyUI-LTXVideo | `15d09abb` | `15d09abb` | 0 |
| kijai/ComfyUI-KJNodes | `c9869ead` | `57105374` | 4 |
| Fannovel16/comfyui_controlnet_aux | `59b1fc41` | `59b1fc41` | 0 |
| yuvraj108c/ComfyUI-Video-Depth-Anything | `a0db08e6` | `a0db08e6` | 0 |
| T8mars/comfyui-minimax-h3-audio-T8 | `91c1b4e9` | `0eae2f22` | 4 |
| H-oliday/SwiftVR | `5ca168ce` | `5ca168ce` | 0 |
| lihaoyun6/ComfyUI-H3VAE_TRT | `4360e008` | `4360e008` | 0 |

The other nine pins match observed HEAD: Sol-Attn, SLA/PlagueKind, Larry Turbo, latent upscaler, LTXVideo, ControlNet Aux, Video Depth Anything, SwiftVR and TensorRT VAE. Keep the existing Larry/TRT compatibility patches; no new upstream rebase is indicated.

## Python packages: upgrade constraints

These are configured versions/ranges versus PyPI metadata, not installed-version comparisons. Packages with open ranges may already resolve to current versions during a fresh build. Latest does not imply compatible.

| Package | Project configuration | Observed latest | Recommendation / adaptation |
|---|---|---|---|
| Torch / torchvision / torchaudio | 2.11.0 / 0.26.0 / 2.11.0, cu130 | 2.14.0 / 0.29.0 / 2.11.0 | Hold the coherent existing stack. A migration requires confirming audio compatibility and replacing/rebuilding the Python 3.12, Torch 2.11, CUDA 13 SageAttention wheel; benchmark native/custom kernels together. |
| NumPy / SciPy | 1.26.4 / 1.15.3 | 2.5.3 / 1.18.1 | Hold pending a separate ABI migration. Validate node/native-extension compatibility and Python runtime constraints; do not lift these independent of the CUDA image. |
| Kornia | 0.8.1 | 0.8.3 | Hold: pinned LTXVideo imports the pyramid-module `pad` export removed in 0.8.2+. Adapt that import or update LTXVideo to a compatible revision first. |
| kernels | 0.16.0 | 0.16.1 | Patch-upgrade candidate only after checking the resolved Transformers FP8 loader's version requirement and loading the prompt-rewriter model. |
| Comfy frontend | 1.51.10 | 1.52.7 | Keep aligned with pinned Core requirements. A deliberate frontend override needs static-asset, proxy/WebSocket and nested-workflow tests. |
| comfy-kitchen | 0.2.33 | 0.2.33 | Current. |
| diffusers | >=0.36,<0.37 | 0.40.0 | Separate compatibility evaluation; do not remove the cap without checking transitive custom-node imports and resolving a clean image. |
| Transformers | >=4.57.1 | 5.17.0 | Already permits latest and crosses a major boundary. Pin a tested version/range with kernels and the prompt-rewriter model instead of assuming the minimum is what production uses. |
| Gradio | >=5,<7; tests 6.24.0 | 6.26.0 | Production permits latest. Bring the test pin to the chosen production version and run UI/browser contracts; ideally narrow production to the tested version. |
| TensorRT cu13 | >=11.2,<12 | 11.3.0.99 | Production already permits this. Pin/record the validated runtime; verify engine cache invalidation, FP32 normalization and single-frame/odd-length decoding on target GPU. |
| wsproto | 1.2.0 | 1.3.2 | Candidate after proxy/WebSocket testing, including compression-disabled operation; update the shared pin consumed by launcher checks. |
| huggingface-hub / accelerate / peft / safetensors | >=0.34 / >=1.12 / >=0.18 / >=0.7 | 1.30.0 / 1.15.0 / 0.20.0 / 0.8.0 | Already allowed; record resolved versions and check prompt/model loading as a group. |

Metadata sources: `https://pypi.org/pypi/<package>/json`; raw observed values saved in `.cache/component-packages-2026-09-10.json`. Primary package pages: [PyTorch](https://pypi.org/project/torch/), [TorchAudio](https://pypi.org/project/torchaudio/), [Kornia](https://pypi.org/project/kornia/), [Gradio](https://pypi.org/project/gradio/), [Transformers](https://pypi.org/project/transformers/), [TensorRT](https://pypi.org/project/tensorrt-cu13/).

**Outstanding reproducibility issue:** `requirements-test.txt` pins `openai==3.8.0`, while both deployment scripts require `openai>=1.109,<3`. Align tests with the deployed SDK major, or deliberately migrate both installers and validate the SDK calls. The lower-risk option is testing the current production major. Broad runtime dependency ranges also mean a fixed Git SHA alone does not reproduce a build. Capture a tested resolved package set for local and Modal installations.

## Validation before applying upgrades

Run existing graph, voice-reference and upstream patch/requirements contracts; build/resolve a fresh Linux Python 3.12 CUDA 13 environment; verify frontend assets and WebSockets; then run repeated matched-seed FL2VA/voice, Ref2VA, Spectrum RES/Turbo, LTX and Music jobs on target hardware. Include cancellation/retry and model unload/reload. TensorRT changes additionally require single-frame and odd-length decode plus cached-engine rebuild checks.

No runtime pins, application code, installed packages or deployments were changed. No GPU inference or test suite was run for this report. Full model catalog updates, all transitive packages and vulnerability advisories are outside this assessment. Large GitHub comparison patches were incomplete for some files; conclusions use commit metadata plus targeted exact-file retrieval for relevant contracts, not a claim of exhaustive review of every added experimental module.
