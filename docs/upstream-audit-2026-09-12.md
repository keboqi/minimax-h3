# Component update audit — 2026-09-12

Checked all 13 configured Git repository pins against live upstream HEAD and queried PyPI metadata for 20 primary packages. The September 10 Spectrum, T8 and KJNodes recommendations are already applied. Three repositories have newer commits; ten match their configured pins. The working tree was clean at the start.

This is an upgrade assessment. No runtime pins, installed packages, models or deployments were changed. Package comparisons describe configured requirements, not versions installed on a remote deployment.

## Recommended order

### 1. Stage Spectrum 0.2.26 → 0.2.27

Target `120d72e2f48b781235b34149e39bbdf0f1317d82`, three commits after our pin. This is the strongest inference-component update:

- Fixes retained actual-target CUDA views that could keep complete final hidden tensors alive through callbacks. References are now cleared when probes finish and when execution fails.
- Projects forecasts from system RAM through bounded CUDA workspaces, with a default 16 MiB hidden-workspace budget. This is relevant to our `history_storage="system_ram"` configuration in `h3_app/catalog.py`.
- Keeps the old one-shot projection for unrecognized FinalLayer wrappers. Native FinalLayer and a specifically reviewed H3-Optimizations contract can stream. Our bundled acceleration code does not directly replace FinalLayer, but the complete deployed patch combination still needs GPU validation.
- Adds narrowly qualified Core BlockSparseAttention cold-to-primed history recovery. Our explicit PlagueKind SLA path is a different integration; do not attribute upstream's Mixed-Grid/Core-BSA speedups to this app.

No node schema/config file changes appear in the comparison. Keep our existing graph inputs, patch ordering, legacy scheduling, offline replay and sampler choices. Update the shared `SPECTRUM_REF` in `h3_sources.py` and version documentation when adopting it; both installers consume that constant.

Treat this primarily as a memory/reliability improvement. Upstream's small matched benchmark reports about 67% lower incremental forecast-head CUDA peak, but the head itself was slower and whole-run timing was roughly flat. Output comparisons are not bitwise-parity guarantees. The 16 MiB budget is not a bound on total generation VRAM.

Validate repeated requests, cancellation/error recovery, unload/reload and peak memory; compare matched-seed normal RES and four/eight-step Turbo with SLA/Sage/Sol, FL2VA voice references, Ref2VA and latent refinement. Preserve unsupported-wrapper fallback behavior.

Sources: [exact comparison](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/compare/be95adecec0b85c80d0c9fc5dd8d07386d50aaee...120d72e2f48b781235b34149e39bbdf0f1317d82), [release notes](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/blob/120d72e2f48b781235b34149e39bbdf0f1317d82/RELEASE_NOTES.md), [projection implementation](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/blob/120d72e2f48b781235b34149e39bbdf0f1317d82/comfyui_spectrum_h3/minimax_h3.py).

### 2. Validate and align Gradio 6.27.0 and Hugging Face Hub 1.31.0

**Gradio:** production permits `>=5,<7`, while `requirements-test.txt` pins 6.24.0. Version 6.27.0 adds fixes for uploaded filenames/file URLs, staging uploads on the cache filesystem, audio previews and playback, plus launch/event JavaScript behavior. The upload and UI changes are relevant; streamed-media fixes only apply where that mode is used. Select one tested version for both installers and tests. Exercise settings restoration, mode switching, reference uploads, gallery playback/downloads, and the mounted/proxied UI. No mandatory app API rewrite was established by this source review. [Release notes](https://github.com/gradio-app/gradio/releases/tag/gradio%406.27.0).

**Hub:** production permits `>=0.34`, so a fresh resolver may already select 1.31.0. This release retries/resumes timeouts while opening streamed downloads and tolerates missing HEAD Content-Length. Those changes are useful for `h3_models.py`'s `hf_hub_download` path. It also fixes atomic snapshot cache references; the HfFileSystem path-validation fix concerns an API our downloader does not call. Validate model download/resume, cached reuse and identity checks, then record/pin the resolved version with Transformers. Existing installations are not proven current merely because the range permits the release. [Release notes](https://github.com/huggingface/huggingface_hub/releases/tag/v1.31.0).

**Existing consistency issue:** tests require `openai==3.8.0`, whereas both installers use `openai>=1.109,<3`. Align tests with the deployed major or deliberately migrate all three together. This finding does not depend on adopting the newest SDK. Similarly, `transformers>=4.57.1` permits major-version drift; capture a tested resolution with kernels and the prompt-rewriter model. Fixed source SHAs plus broad package ranges do not reproduce an image.

### 3. Keep ComfyUI 0.35.0 unless adopting its new features

HEAD `7193f5627f036701e5efc23beaea20fa37ceaadd` is 16 commits ahead. The relevant H3 change keeps Fun ControlNet state preparation and before/after-block work outside compiler allocation capture. Our current generation graphs do not select H3 Fun ControlNet, so this is not an identified fix for their present path. Other additions include Yue2 music, Marigold v2 and Video Concatenate; these need their own product/model integration if wanted.

If upgrading Core, move the matched frontend **1.51.10 → 1.52.7** and upstream workflow templates **0.11.57 → 0.11.59** together. Exact old/new requirements comparison shows these are the only requirements-file changes; Kitchen stays 0.2.33, aimdo 0.5.3 and embedded docs 0.5.11. Update `COMFY_REF` and `COMFY_FRONTEND_VERSION` in `h3_requirements.py`, its self-test expectations and the cached source used by upstream contract tests. Recheck static assets, nested bundled workflows, WebSockets/proxying and H3/LTX/Music generation.

Sources: [exact comparison](https://github.com/Comfy-Org/ComfyUI/compare/40c4fcdf513a4523e39d54a9d391908af8df8171...7193f5627f036701e5efc23beaea20fa37ceaadd), [Fun ControlNet compiler fix](https://github.com/Comfy-Org/ComfyUI/commit/6338e4bd428247a4a8843496aa98fb7f2a9d3632), [target requirements](https://github.com/Comfy-Org/ComfyUI/blob/7193f5627f036701e5efc23beaea20fa37ceaadd/requirements.txt).

### 4. T8 1.77.0 → 1.79.0 is optional

Target `b8574103574b7dbaa09f5e6350ff7d290bbd425c`, three commits ahead. Exact source comparison confirms `conditioning.py` is unchanged and the `MiniMaxH3AudioConditioningT8` class has an identical AST. Our FL2VA voice-reference feature does not require this update.

Version 1.78.1 moves implementation files under `h3_t8/` while preserving node IDs and package imports through the root entry point. Our graph uses the node ID. The local installer also validates root `nodes.py`, `conditioning.py` and `core.py` in `sync_external_nodes`; those `required_paths` must move under `h3_t8/` when adopting the new layout. Verify full package import/registration on Linux and voice graphs, and preserve the complete nested directory. Tools that directly read old implementation paths must adapt.

New capabilities are separate feature work:

- **Dual-model 4+4 long-video loops:** different models/LoRAs can serve initial sampling and learned-upscale refinement. Integrating this requires model residency, segment state, audio continuity, seam checks and explicit graph/UI work. It is not equivalent to just changing our existing refinement step count.
- **Optional TRT VAE:** separate runtime, flex decoder and dedicated single-frame/video encoder engines. Upstream tested Windows/4060 Ti with TensorRT 10.13.3.9.post1 and did not measure an overall output-stage speed benefit. Do not replace our Linux/Blackwell TRT 11.x integration or its compatibility patches based on this release alone.
- **Topaz:** requires separately installed licensed software/models. This does not provide a ready Modal/Linux enhancement backend.
- **VDN/Sol coexistence:** upstream now permits a retained attention override, but VDN still computes its own blocks. This is not evidence that Sol accelerates those VDN blocks.

Sources: [comparison](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/compare/0eae2f22e6cb2115c9002ab6daa0c12c9636ccb1...b8574103574b7dbaa09f5e6350ff7d290bbd425c), [1.79 notes](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/blob/b8574103574b7dbaa09f5e6350ff7d290bbd425c/docs/RELEASE_1.79.0.md), [layout contract](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/blob/b8574103574b7dbaa09f5e6350ff7d290bbd425c/docs/REPOSITORY_LAYOUT.md), [TRT scope](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/blob/b8574103574b7dbaa09f5e6350ff7d290bbd425c/docs/TRT_VAE_EXP.md).

## Repository inventory

| Component | Configured pin | Observed HEAD | Ahead |
|---|---|---|---:|
| ComfyUI | `40c4fcdf` | `7193f562` | 16 |
| Spectrum | `be95adec` | `120d72e2` | 3 |
| T8 Audio | `0eae2f22` | `b8574103` | 3 |
| KJNodes | `57105374` | same | 0 |
| Sol-Attn | `930a4d6e` | same | 0 |
| SLA / PlagueKind | `59f54d35` | same | 0 |
| Larry Turbo | `4274783a` | same | 0 |
| H3 latent upscaler | `d7c01b90` | same | 0 |
| LTXVideo | `15d09abb` | same | 0 |
| ControlNet Aux | `59b1fc41` | same | 0 |
| Video Depth Anything | `a0db08e6` | same | 0 |
| SwiftVR | `5ca168ce` | same | 0 |
| H3VAE TRT | `4360e008` | same | 0 |

## Remaining package policy

| Package | Configured | Observed latest | Recommendation |
|---|---|---|---|
| Torch / torchvision / torchaudio | 2.11.0 / 0.26.0 / 2.11.0, cu130 | 2.14.0 / 0.29.0 / 2.11.0 | Keep the coherent stack. Our Sage wheel explicitly targets Python 3.12/Torch 2.11/CUDA 13. A migration needs matching audio and rebuilt/replacement kernels. |
| NumPy / SciPy | 1.26.4 / 1.15.3 | 2.5.3 / 1.18.1 | Keep existing ABI constraints; evaluate separately with native dependencies. |
| Kornia | 0.8.1 | 0.8.3 | Retain documented LTXVideo compatibility pin; its upstream source has not changed. |
| kernels | 0.16.0 | 0.16.1 | Check the chosen Transformers FP8 loader and prompt model before changing the compatibility pin. |
| comfy-kitchen | 0.2.33 | 0.2.33 | Current. |
| diffusers | >=0.36,<0.37 | 0.40.0 | Keep cap until transitive node imports and a clean image resolution are tested. |
| Transformers | >=4.57.1 | 5.17.0 | Choose/record a tested version with kernels and Hub. |
| TensorRT cu13 | >=11.2,<12 | 11.3.0.99 | Already allowed; record deployed resolution and validate engine rebuilding/decoding before standardizing. |
| wsproto | 1.2.0 | 1.3.2 | Optional after proxy/WebSocket regression checks. |
| accelerate / peft / safetensors | >=1.12 / >=0.18 / >=0.7 | 1.15.0 / 0.20.0 / 0.8.0 | Already allowed; validate model loading as a set. |
| OpenAI SDK | production >=1.109,<3; tests 3.8.0 | 3.13.0 | Resolve the test/production mismatch before considering latest. |
| Modal SDK | no central exact pin identified | 1.5.5 | No upgrade recommendation established; record the SDK used to build/deploy. |

Gradio and Hub versions/recommendations are covered above. Registry metadata came from `https://pypi.org/pypi/<package>/json`, including [Gradio](https://pypi.org/pypi/gradio/json), [Hub](https://pypi.org/pypi/huggingface-hub/json), [Torch](https://pypi.org/pypi/torch/json) and [TensorRT](https://pypi.org/pypi/tensorrt-cu13/json). Latest metadata does not establish compatibility.

## Evidence and validation boundary

Raw repository comparisons and package metadata are saved locally in ignored `.cache/component-repos-2026-09-12.json` and `.cache/component-packages-2026-09-12.json`. Targeted exact source files are in `.cache/upstream-audit-2026-09-12/`. GitHub's large comparison responses omit some patches and cap the file list; conclusions about T8 conditioning and Core requirements use separately fetched exact old/new files, not zero-valued truncated diff statistics.

Performed source/AST comparison, dependency inventory and call-site inspection. No application tests, fresh Linux/CUDA resolution, GPU inference, installed-version inspection, deployment, full model-catalog refresh or comprehensive vulnerability audit was performed. Before promoting any candidate, run existing regression/browser checks and a clean deployment build, then the relevant GPU comparisons described above.


## Applied upgrade — 2026-09-12

The requested Spectrum and Gradio upgrades are now implemented:

- Spectrum is pinned to v0.2.27 at `120d72e2f48b781235b34149e39bbdf0f1317d82` in the shared source catalog. Existing graph inputs, sampler policy and attention ordering are preserved.
- Gradio is pinned to 6.27.0 through `GRADIO_VERSION` in both installers, and the test requirements match. The isolated Python 3.12 test environment now has Gradio 6.27.0 and its resolved gradio-client 2.7.0.
- The launcher rejects missing or mismatched Gradio versions so an existing local install refreshes on its next ordinary startup. Spectrum nodes already sync on startup, including the `--skip-env` path. Modal build inputs include both shared pin files; deployment requires rebuilding/redeploying.
- README and legacy diagnostic version text are updated. The settings browser test now waits for the server-rendered bridge state before the next edit, fixing an observed synchronization race between immediate checkbox changes and queued updates.

Validation: 93 CPU tests ran, 90 passed and three Torch-dependent tests skipped. This includes the existing workflow, attention, voice, upstream-source and service self-tests. Both settings and FL2VA voice browser acceptance checks passed on Gradio 6.27.0; the initial settings run hit the synchronization race described above. Exact pinned Spectrum input-schema/range and apply-signature checks accepted every configured graph input; its package declares no additional pip dependencies. Launcher checks accepted current versions and rejected stale/missing Gradio and missing frontend assets. Shared test/runtime Gradio version agreement, Python compilation, Bash syntax, `uv pip check` and Git whitespace checks passed.

Logs are in ignored `.cache/spectrum-gradio-upgrade-tests.log`, `.cache/gradio-627-settings.log` and `.cache/gradio-627-voice.log`. No GPU inference, full Linux/CUDA image build or Modal deployment was performed. The other component and SDK upgrades in the assessment remain recommendations.
