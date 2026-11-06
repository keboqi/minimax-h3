# Upstream status

## Native RelayVSR added — 2026-11-06

`COMFY_REF` pins `2f8943324e1eae89a5ee6c7db93602299246b5f1`, including
[native RelayVSR (#17511)](https://github.com/Comfy-Org/ComfyUI/pull/17511).
This is 11 commits ahead of the previous pin and retains adaptive SeedVR2 VAE slicing.
Kitchen moves to 0.2.41 for `window_attention`; frontend stays at 1.58.5.
Media exposes separate native 2x and 4x options for videos and single images,
without input/output resizing or target-resolution presets. Only the selected
INT8 checkpoint and matching Flash decoder download on first use, from pinned
`Comfy-Org/RelayVSR` revision `340486a3adf9544e660e9146911c3ced59e58dd3`.
Video graphs reuse the source audio and FPS. GPU inference and quality comparisons
remain unverified. Reprovision local ComfyUI or rebuild/redeploy Modal to apply it.

## ComfyUI SeedVR2 VAE pin updated — 2026-11-04

`COMFY_REF` now pins `56f179f6c50dad0c4c860ae4da427415c8e86415`, including
[adaptive SeedVR2 VAE slicing (#17508)](https://github.com/Comfy-Org/ComfyUI/pull/17508).
The two-commit upgrade changes no dependency requirements. Our native tiled
SeedVR2 encode/decode graphs use the updated implementation without graph changes.
It uses spare memory for longer slices and, where supported, device-held caches;
lower peak VRAM and a particular speedup are not guaranteed. GPU performance
remains unverified. Reprovision local ComfyUI or rebuild/redeploy Modal to apply it.

## Drama 3 FP8 pin updated — 2026-11-04

`DRAMA3_HF_REVISION` now pins `78e5ab79e49aa9a25abe8799eff2c4d73a1c3216`,
the audited October 29 FP8 repack. After redeploying, first use downloads this
revision into its own cache directory; the old checkpoint directory is not
reused. INT4 retains its existing pin. Parsed upstream model configuration is
unchanged, but GPU loading and listening comparisons remain unverified.
Focused validation passed: 96 Drama serving tests and five preparation tests.
The audit below records the pre-update decision.

## Model and LoRA audit — 2026-11-04

Checked all **134 configured files across 42 model repositories** in
`h3_models.py`, plus **nine speech/audio repositories** referenced by the
workers: **51 repositories total**. All configured filenames still exist;
all **59 configured SHA-256 values match** current upstream metadata. The
remaining 75 entries have no configured checksum baseline, so their presence
does not establish that their contents are unchanged. This compares the source
inventory, not files installed in a deployed container.

### Candidates and decisions

| Model / adapter | Current project state → upstream observation | Decision |
| --- | --- | --- |
| DMAD H3 | Uses `dmad_minimax_h3_4step_full_critic_comfyui.safetensors`. A separately trained eight-step student was uploaded October 30 UTC (the card announces October 29), including `minimax_h3/dmad_minimax_h3_8step_full_critic_comfyui.safetensors`. | Highest-priority new LoRA candidate. Add as a separate eight-step option with the author's sampler contract; compare video, audio and runtime before changing defaults. The author's quality claim is not a local benchmark. |
| Drama 3 FP8 | Pin `e0a9218a` → head `78e5ab79`. October 29 replaces eight weight shards with fifteen and changes the shard index. Parsed config objects are identical, including `bsgemm-moe`; shard bytes total 39.603574 GB → 39.603551 GB. | Review the repack before upgrading. Metadata does not establish tensor equality or improved speech quality. Keep the pin pending loading/listening checks. INT4 remains exactly at our `a6f58869` pin. |
| Fibo 2.0 | Configured Dev and Edit Turbo hashes match. Upstream also contains `loras/fibo_2.0_turbo_lora_avg_rank_270_bf16.safetensors`, uploaded October 21 and already mentioned by the October 28 audit. | Optional separate acceleration experiment; our merged Edit Turbo checkpoint is already available. This is not a newly discovered replacement for those checkpoints. |
| MOSS few-step | Our frozen four-step v2.0 file still matches. The same repository offers `moss_image_sft_3step_v1.0.safetensors` for exactly three steps, Euler/simple, CFG 1, strength 1. | Optional draft mode, not a four-step replacement. It predates this audit; retain the current four-step final mode. |
| P-H3 Image | Our `44a0fb73` pin is the current head, including native ComfyUI workflows. | Keep. Other precisions/merged files are alternatives, not a later release. |
| H3 base / Turbo / Singularity | Pika Flex, Sol INT8, both regenerate refiners, LightX2V four/eight-step adapters and Singularity V2 are already represented. All their configured checksums match. Larry remains v4-600 EMA; its latest experimental v5 upload is an older `.bin`, not a new ComfyUI release. | Keep existing selections. No newer drop-in adapter was found in these repositories. |
| Qwen Image 2.1 | Official Turbo, Pruna v0.2/v0.1 and Viggle v0.3 configured hashes match. October 29 adds Pruna adapters to the Comfy-Org repository; we already source them from Pruna directly. Lightning's configured hash also matches. | Keep. No replacement Lightning weight was published; community schedule suggestions do not establish an edit-drift fix. |
| FLUX 3 Klein | All eleven configured hashes match metadata at current head `9d36114c` (November 2 UTC). | No configured weight replacement indicated. Commit history requires authenticated access, so the latest repository change was not classified. |
| LTX / enhancement / upscaling | All configured LTX-2.5, legacy 2.3 IC-LoRAs, CQ enhancer, SeedVR2, H3 latent upscaler and VAE filenames exist; configured hashes match where present. | No newer replacement identified. Several gated histories could not be read; retain current compatibility choices. |
| Other speech / music | Music 3, YuE2, Qwen VoiceDesign, Qwen ASR/aligner and Hy-MT2 repositories show no newer activity since the October 30 cutoff. MOSS Transcribe head differs from our pin only in README. Nemotron's latest commits concern documentation/chat templates, not a newly uploaded checkpoint. | Keep. Nemotron is unpinned, so an already cached deployment need not contain the latest template; that is separate from a weight upgrade. |

Sources: [DMAD card and usage](https://huggingface.co/ZhengmingYu/DMAD),
[Drama 3 history](https://huggingface.co/fishaudio/drama-3/commits/main),
[Fibo history](https://huggingface.co/Comfy-Org/Fibo-2.0/commits/main),
[MOSS draft instructions](https://huggingface.co/akhaliq/MOSS-Image-SFT-4-steps),
[P-H3 Image](https://huggingface.co/PrunaAI/P-H3-Image-Comfy),
[Qwen adapters](https://huggingface.co/Comfy-Org/Qwen-Image-2.1),
[Lightning community report](https://huggingface.co/lightx2v/Qwen-Image-2.1-Lightning/discussions/5).
Exact heads, file metadata, recent commit summaries, configured checksum
comparisons and Drama/MOSS pinned-to-head comparisons are saved under
`.cache/model-audit-2026-11-04/`.

Public metadata was available for all 51 repositories. Commit-history requests
returned HTTP 401 for FLUX, Stable Audio and ten LTX repositories; those histories
remain unverified. No model weights were downloaded, model defaults or revisions
changed, tests run, deployment performed or GPU quality claims validated.

## Implemented upgrades — 2026-11-04

The five recommended upgrades from the audit below are now pinned in
`h3_requirements.py` for local setup and Modal builds:

| Component | New pin |
| --- | --- |
| Gradio | 6.31.0, also in `requirements-test.txt` |
| audio-separator | 0.48.0, isolated dubbing environment |
| ComfyUI / frontend | `56f179f6c50dad0c4c860ae4da427415c8e86415` / 1.58.5 |
| T8 Audio | `3a23f333537d9a715aad7ce79af8d9ec891d9ad8` |
| SGLang-Omni | v0.2.0 / `e8c635ecb9c68bcf2fbc32ceb3c40f85f3590657` |

Kitchen remains 0.2.40. ComfyUI's Torch/CUDA/NumPy stack and model defaults
remain unchanged; Omni's new Torch 2.14.1 dependency stays in its separate
environment. The existing environment fingerprints invalidate older Omni and
dubbing readiness markers when their pins change.

Validation completed with Gradio 6.31.0 and Brotli 1.2.1 installed in the local
UI test environment:

- The focused transport/proxy, Drama serving/checkpoint/snapshot, dubbing,
  audio-worker, FL2VA and standalone suite ran 306 tests: 302 passed and four
  optional Torch/audio numerical tests were skipped. Three additional file-range
  regressions passed, covering oversized ranges, starts past EOF/empty files,
  and media truncated after stat. The existing frontend bundle patch still applies.
- Browser queue acceptance passed, including reconnecting after dropped sockets
  with SSE blocked. Browser media acceptance passed for cached previews,
  image selection and audio/video/image switching.
- Exact-source checks verified Core's paired frontend/Kitchen requirements,
  T8's unchanged voice node schema/execution method, and audio-separator's used
  interfaces. The actual v0.2.0 Drama config accepts our launch overrides; its
  router methods honor H3's SHM platform override without using direct CUDA IPC.
- Omni's fishaudio dependencies resolved for Python 3.12, Linux glibc >=2.34
  and CUDA 13 using the existing protobuf override. The shared requirements
  self-test passed. Evidence is in `.cache/component-upgrade-2026-11-04/`.

No full test suite, GPU inference, speech listening comparison or deployment
was run. Rebuild/redeploy Modal to adopt the new build dependencies. Local
startup detects the changed Core/frontend/Gradio and refreshes setup; the
isolated speech/dubbing environments refresh on next use. GPU engine loading,
snapshot/sleep/wake behavior and separation output still need deployment checks.
The audit below records the pre-upgrade state and rationale.

## Component audit — 2026-11-04

This section supersedes the older recommendations below. Compared the actual
pins against all **18 Git repositories**: nine match their default-branch head
and nine have updates. Queried 25 PyPI packages; 24 returned a latest version,
while `kornia-rs` returned no usable version. These are upstream observations,
not an inventory of packages resolved in a deployed container. No runtime pins,
installed packages or model defaults were changed.

### Recommended upgrade order

1. **Gradio 6.30.0 → 6.31.0.** Highest-priority UI maintenance candidate:
   fixes an infinite loop on file ranges past EOF, requires `brotli>=1.2.1`
   for the security issue identified in its release notes, and puts Video
   conversions/watermarks in the cache. Update the shared pin and test
   requirements together. Run focused queue/WebSocket, media-range, video-cache
   and browser checks. Its metadata also requires Hub >=1.16,<3 and
   gradio-client 2.7.2; Hub 1.x remains allowed by both policies, but validate
   the resolved FastAPI/Starlette stack with our custom routes/proxy.
   [Release notes](https://github.com/gradio-app/gradio/releases/tag/gradio%406.31.0).
2. **audio-separator 0.47.1 → 0.48.0, in the dubbing venv only.** Restricts
   checkpoint deserialization, adds mirror-first downloads and SHA-256 checks
   where a published checksum exists, and uses temporary files for downloads.
   The two-commit comparison changes no dependencies besides the package's
   version. Check loading our chosen separation models, cached/fresh downloads
   and stem outputs before adoption; unknown-checksum files are still allowed.
   [Exact changes](https://github.com/nomadkaraoke/python-audio-separator/compare/v0.47.1...v0.48.0).
3. **ComfyUI/frontend as a pair:** `2c75dc1b` (already v0.44.0) →
   `7f09d3fa`, frontend 1.57.6 → 1.58.5. The single Core commit only changes
   the frontend requirement; Kitchen remains 0.2.40. Frontend 1.58.5 includes
   canvas recovery and picking up new node types after backend restarts.
   Validate `/comfyui/`, immutable assets, encoded nested workflows and a saved
   workflow reload. No inference improvement is established by this change.
   [Core diff](https://github.com/Comfy-Org/ComfyUI/compare/2c75dc1b379a59e0880619d029607af5169a62ef...7f09d3fa4a783868fe39d4d63cf4f815e58abecd),
   [frontend notes](https://github.com/Comfy-Org/ComfyUI_frontend/releases/tag/v1.58.5).
4. **T8 Audio:** evaluate `3a23f333` (37 commits beyond our pin). New
   compatibility fixes stop rejecting an unchanged native SigmaShift factory
   merely because other Core source changed, and handle explicitly exported
   nonpersistent buffers. This is more relevant than the earlier audit's
   feature-only assessment, particularly for composed semantic-bridge paths.
   Our voice graph uses `MiniMaxH3AudioConditioningT8`; it is not established
   that this ordinary path needs the fix. Check its input contract, reference
   conditioning and native/bridge sampling before adopting the larger update.
   Upstream's tiny-model CPU checks establish software compatibility, not GPU
   output quality. [Compatibility report](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/blob/3a23f333537d9a715aad7ce79af8d9ec891d9ad8/docs/CORE_COMPAT_20261031.md).
5. **SGLang-Omni:** evaluate released **v0.2.0** at `e8c635ec`
   (134 commits past our v0.1.9), rather than jumping directly to main
   `b3064b2e` (251 commits). The release includes fused Snake activation/padding
   for Drama 3's vocoder. Main additionally offers opt-in reference sampling
   with epsilon cutoff and a shorter first streaming window; those are separate
   candidates, not v0.2.0 features. Review install requirements, CLI/router and
   host-SHM platform compatibility, then run speech, GPU snapshot and
   sleep/wake checks. Keep reference sampling opt-in pending listening tests.
   [Release](https://github.com/sgl-project/sglang-omni/releases/tag/v0.2.0),
   [vocoder change](https://github.com/sgl-project/sglang-omni/commit/cc5ca8d8982bad577f19390d941f6bb455dc02d1),
   [later reference sampling](https://github.com/sgl-project/sglang-omni/commit/b1097644c09fd3dbe00c26e46d114dbec87f923d).

### Remaining source decisions

| Component | Pin → checked head | Decision |
| --- | --- | --- |
| KJNodes | `57105374` → `ab72eda7`, 31 commits | Optional: H3 audio-preview scaling, prompt builders and subgraph/UI fixes. No demonstrated fix to our generated inference graphs; review the LoRA/model-optimization changes when upgrading. |
| ControlNet Aux | `59b1fc41` → `a587a4c1`, 7 commits | Low priority; MediaPipe face-mesh, animal-pose JSON and dependency cleanup. |
| LTXVideo | `3bf3ca62` → `220e1d45`, 6 commits | Hold a pin-only update: still removes `LTXVSDRToHDRWorkingSpace` and `LTXVSaveHLG`, both used by `h3_app/workflows/upscale.py`. Migrate HDR graphs/templates first. |
| Spectrum-MOSS | `3fde0674` → `1e5eb359`, 15 commits | Hold a pin-only update: requires `warn_unverified_sampler`. ER-SDE support needs graph integration and benchmarking; preserve our explicit tail steps. |
| Drama3 nodes | `67a733a9` → `8cb3c643`, 1 commit | Registry publishing only; no runtime upgrade needed. |
| NeMo Speech | `a837f4e5` → `0fc5d7ad`, 3 commits | Word-confidence boundaries, codec training and duplex early-interruption fixes; no demonstrated benefit to our Nemotron vLLM path. |

Unchanged heads: Larry Turbo, SLA, Veda, Sol (public fork), H3 Spectrum,
Qwen Spectrum, Video Depth Anything, TensorRT VAE and stable-audio-tools.
Exact compared SHAs, commit summaries and changed filenames are recorded in
`.cache/component-audit-2026-11-04/repos.json`.

### Remaining package decisions

| Package | Policy/pin → checked latest | Decision |
| --- | --- | --- |
| Torch / torchvision / torchaudio | 2.11.0 / 0.26.0 / 2.11.0 → 2.15.0 / 0.29.2 / 2.11.1 | Hold the validated CUDA/attention ABI; do not independently choose each package's latest version. |
| NumPy / SciPy | 1.26.4 / 1.15.3 → 2.5.4 / 1.18.1 | Hold main environment's ABI/HDR constraints. Dubbing already uses an isolated NumPy 2 environment. |
| Transformers / kernels / Hub | <5.18 / 0.16.0 / <2 → 5.22.1 / 0.18.1 / 2.4.0 | Review jointly. Latest Transformers' kernels extra requires >=0.17.2,<0.18, so even the latest kernels is outside that range. Hub 2 is not required: Transformers still allows Hub >=1.31,<3. |
| Diffusers / Kornia | <0.37 / 0.8.3 → 0.42.1 / 0.9.0 | Hold pending model/node compatibility work. `kornia-rs` latest could not be determined. |
| Kitchen | 0.2.40 → same | Keep. |
| wsproto / Pedalboard | 1.2.0 / 0.9.19 → 1.3.2 / 0.9.26 | Secondary maintenance candidates; validate queue transport and speaker effects. |
| OpenAI SDK | runtime >=3.16.2,<4; tests 3.16.2 → 3.45.0 | Already allowed at runtime; align tested/deployed resolution in a separate prompt-writer check. |
| Playwright | tests 1.62.0 → 1.64.1 | Optional test tooling refresh; reinstall its matching browser if adopted. |
| TensorRT CU13 / yt-dlp | >=11.2,<12 / >=2026.8.19 → 11.4.0.106 / 2026.10.26 | Already allowed; a cached deployment may still need a rebuild to resolve them. |
| Qwen TTS / Qwen ASR / OmniVAD | 0.1.1 / 0.0.6 / 0.2.13 → same | Keep isolated environments. |
| Modal | queried latest 1.6.3 | Deployment tooling candidate; installed SDK version was not inspected. |

Package evidence: public [PyPI JSON metadata](https://docs.pypi.org/api/json/),
saved with dependency metadata in `.cache/component-audit-2026-11-04/packages.json`.
This audit reviewed source diffs and local graph usage. No test suite, package
installation, deployment or GPU inference was run. Model repositories were not
re-audited; model candidates below retain their earlier validation boundaries.

## Targeted recheck — 2026-10-30

Pika H3 Flex is integrated as a selectable six-step Turbo/refinement LoRA using
the official Comfy workflow's LCM + ManualSigmas contract. Both task adapters
are pinned to `Comfy-Org/MiniMax-H3@912b6e38b2092ec4d2e86d24183e0d59ea059d48`
and downloaded on demand with SHA256 verification. Existing native upscale and
other options remain available for user testing; deployment GPU results are
pending. The [integration record](plans/pika-h3-flex-integration-plan.md) lists
the reviewed sources and implementation details. No dependency pin was changed.

ComfyUI is pinned at `0df1646a9cd71d4c990a87014fb37797a2956ead`, the merged
[Sol H3 VAE support](https://github.com/Comfy-Org/ComfyUI/pull/17411) revision.
Frontend is paired at 1.57.6; Kitchen remains 0.2.40. Fast and Singularity use
Sol INT8 decoding and retain LightVAE Lite encoding. Balanced and Quality retain
Official INT8 decoding. Sol cannot tile, so compare peak memory on long and
high-resolution clips. The pin also includes the native H3 latent upscaler;
our refinement and H3 Regenerate graphs use its native optimized execution node
with a bundled precision-selectable loader. The old upscaler custom-node repository
is no longer installed by local setup or Modal builds.

## Targeted recheck — 2026-10-28

This recheck focused on the new Core release and model/LoRA candidates, not a
fresh comparison of every pinned repository or Python package.

- **ComfyUI v0.43.0** is now pinned and includes native Fibo 2.0, its negative-prompt fix,
  MOSS edit support and the MOSS memory change. Frontend is paired at 1.57.4.
  Validate graph execution and GPU workflows. [Release notes](https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.43.0).
- **PTD RefAsymOPD LoRAs** are promising experimental Ref2VA accelerators:
  the authors offer 6- and 8-step weights and claim improved quality over plain
  PTD. They use per-step adapters and PDD heads, and require their custom nodes
  plus ComfyUI-PTD; they are not a drop-in ordinary H3 LoRA. Evaluate on the
  official H3 base and benchmark video/audio quality and memory before considering
  compatibility with Singularity or a default change. [Model card](https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs).
- **MOSS edit and color-depth adapters** offer new functionality rather than
  checkpoint upgrades. Edit is integrated with Core's native reference conditioning;
  validate its output on a GPU. Color-depth still needs a conditioning graph and UI.
  [Edit model](https://huggingface.co/chinoll/Moss-Image-Edit),
  [color-depth model](https://huggingface.co/darask0/MOSS-Image-color-depth-control).
- **Singularity V2** remains an opt-in candidate. Its author reports stronger
  prompt adherence and an audio-reference fix; compare against v1.3 on our
  prompts and reference audio before changing the default. [Model card](https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity).
- **Fibo turbo LoRA** still has no documented settings in the Comfy-Org model
  card; keep waiting rather than assuming settings from the separate fused turbo
  checkpoint. [Files](https://huggingface.co/Comfy-Org/Fibo-2.0/tree/main).
- **Viggle v0.3** is already present in the working tree. Upstream describes it
  as a different quality tradeoff from v0.2.1, not a strict upgrade; its 9-step
  hybrid requires seven LoRA steps followed by two base steps. The local graph
  implements a split, but inference parity and quality still need GPU validation.
  [Model card](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo).

The Core and frontend pins were updated after this recheck. No model defaults
were changed, and no GPU inference or full test suite was run.

Open upstream items as of the 2026-10-25 component audit. Pins live in
`h3_sources.py` and `h3_requirements.py`. Earlier dated audits are in git
history (`git log --diff-filter=D -- 'docs/upstream-audit-*'`). Update this
page when an item is resolved or a new audit finds one.

## Recommended next changes

This audit checked all 18 pinned Git repositories against their default-branch
heads and queried 25 Python packages. Ten Git pins already match upstream;
eight have changes. Runtime pins were left unchanged apart from the
Transformers bound in the package table. Package metadata for
`kornia-rs` was unavailable in this check.

1. **Evaluate ComfyUI `067f8533` for MOSS memory use.** It is six commits
   beyond our `bc5f880d` pin, with shorter-lived MOSS attention intermediates,
   attention-bias conditioning, and edit image strength. Kitchen 0.2.40,
   frontend 1.57.1 and aimdo 0.6.0 remain unchanged. The memory change benefits
   existing MOSS generation; prompt weighting and image editing require
   separate graph/UI work. The update also includes an execution-engine Branch
   node change, so validate normal graph execution as well as MOSS with
   Spectrum on/off before adopting it. No GPU memory reduction was measured here.
   Source: [exact comparison](https://github.com/Comfy-Org/ComfyUI/compare/bc5f880dbdd43e426077f42ccec53bb577ffee9f...067f853305e74a051dd17f15964d5b0e7887abf5).
2. **Adapt Spectrum-MOSS when adding an ER-SDE option.** Upstream is now
   v0.8.1 at `1e5eb359` (15 commits ahead), with native ER-SDE bridging,
   synchronized CFG/SLG forecast decisions, and a fix respecting a user horizon
   cap of one. Add the required `warn_unverified_sampler=True` graph input
   alongside the pin update. Keep our explicit two tail steps: upstream now
   defaults to one. Our current Euler + `sgm_uniform` graph already matches
   the upstream recommendation, so this is not an urgent speed upgrade for
   today's default. Benchmark ER-SDE independently before changing presets.
   Sources: [source and input contract](https://github.com/Rkkss/ComfyUI-Spectrum-MOSS/blob/1e5eb359160b50be02ff95d1a93f22af2947888c/nodes.py),
   [sampler guidance](https://github.com/Rkkss/ComfyUI-Spectrum-MOSS/blob/1e5eb359160b50be02ff95d1a93f22af2947888c/README.md).

## Other source updates

| Component | Current → checked upstream | Assessment |
| --- | --- | --- |
| LTXVideo | `3bf3ca62` → `220e1d45`, 6 commits | Requires the HDR graph migration described below; avoid a pin-only update. |
| KJNodes | `57105374` → `37ca7aff`, 22 commits | H3 audio-preview scaling fix, FLUX/MOSS prompt builders and MOSS prompt weighting. Our generated graphs do not use the affected preview node; low urgency. New MOSS weighting should travel with compatible Core support. |
| ControlNet Aux | `59b1fc41` → `a587a4c1`, 7 commits | MediaPipe 1.0 face-mesh compatibility, animal-pose JSON encoding, dependency cleanup. Useful if exposing those preprocessors; no demonstrated benefit to current graphs. |
| T8 Audio | `6063fafb` → `5bea92d2`, 29 commits | Through v1.102.0: Director, sampling extensions, reference workflows and authoring improvements. Feature integration work, not a required voice-conditioning update. |
| SGLang-Omni | v0.1.9 / `8200f499` → `2fca2035`, 51 commits | v0.1.9 remains the latest backend release; newer Voxt tags are a different app. Hold while validating our current sleep/checkpoint path. The apparent Drama EOS fix in #3100 changes a test server, not production sampling. |
| H3 latent upscaler | `d7c01b90` → `40316cf0`, 1 commit | MIT license addition only; no runtime adaptation. |

Comparisons:
[LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo/compare/3bf3ca62595f1764c47d01c35c8e5dfe47e1a88f...220e1d45aa0469f518e982102fca4dca92a61862),
[KJNodes](https://github.com/kijai/ComfyUI-KJNodes/compare/57105374f47d0fbb49c9c3926fb981702e0a4b5c...37ca7aff7ed14dde6ba3c0b25713e21c18b28dcb),
[ControlNet Aux](https://github.com/Fannovel16/comfyui_controlnet_aux/compare/59b1fc411ede8623b2997855b8018f0b3b6cf49f...a587a4c1594524c1d2de5cd1a86cd868ff448e1c),
[T8](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/compare/6063fafbd9c3b85c5ff40aef435ae11b2844e558...5bea92d2e5c681fd8992eb8f297e9901158708d7),
[Omni test-only EOS fix](https://github.com/sgl-project/sglang-omni/commit/07367c2ae1308014d3c0a39c5b1cea6d1b3dbfbe).

A same-day recheck found three more heads, none needing adaptation: ComfyUI
`4281eb0a` (feature-flagged deploy-to-cloud endpoints), Drama3 `8cb3c643`
(Comfy Registry metadata only) and SGLang-Omni `44ba1227` (realtime duplex,
MiniCPM-o and Qwen3-Omni session fixes).

Already at the checked upstream head: Drama3 nodes, H3 Spectrum, Qwen Spectrum,
Larry Turbo, SLA, Veda, Sol, TensorRT VAE nodes, Video Depth Anything and
stable-audio-tools.

## Python package decisions

| Package | Project policy → current upstream | Decision |
| --- | --- | --- |
| Gradio / Kitchen | 6.30.0 / 0.2.40 → same | Keep. |
| Frontend | 1.57.1 → 1.57.3 | Keep aligned with Core's exact requirements, including the proposed Core head. |
| Torch / torchvision | 2.11.0 / 0.26.0 → 2.14.1 / 0.29.1 | Hold: Sage/FlashAttention wheels and CUDA extensions are tied to the existing ABI. Torchaudio's queried latest is still 2.11.0. |
| NumPy / SciPy | 1.26.4 / 1.15.3 → 2.5.4 / 1.18.1 | Hold the validated ABI stack and HDR dependency constraints. |
| Hub / kernels | Hub >=1.5,<2; kernels 0.16.0 → 2.3.0 / 0.17.2 | Review with Transformers as a set. New Transformers still permits Hub 1.x; a Hub 2 migration is not inherently required. |
| Transformers | >=4.57.1,<5.18 → 5.20.2 | Bounded on 2026-10-25 and added to the shared install constraints with kernels 0.16.0. Transformers 5.18+ require kernels 0.17 for the fine-grained FP8 loader, so move both together and validate on a GPU. SGLang-Omni (5.19.0) and the audio tools (4.57.3) use separate environments. |
| Diffusers | >=0.36,<0.37 → 0.42.0 | Deliberately bounded; review model-loading compatibility separately. |
| Kornia | 0.8.3 → 0.9.0 | Hold until the LTX/node compatibility review. |
| wsproto / Pedalboard | 1.2.0 / 0.9.19 → 1.3.2 / 0.9.26 | Maintenance candidates, not established fixes for this app. Validate queue transport and speaker effects separately. |
| TensorRT CU13 | >=11.2,<12 → 11.4.0.106 | Already allowed by the range; validate VAE engine building and decoding on an actual resolved version. |
| Qwen TTS / Qwen ASR / OmniVAD / audio-separator | 0.1.1 / 0.0.6 / 0.2.13 / 0.47.1 → same | Keep isolated environments. |
| OpenAI SDK | Runtime >=3.16.2,<4; tests 3.16.2 → 3.37.0 | Runtime already allows it. Align the tested version with deployment resolution. |
| yt-dlp | >=2026.8.19 → 2026.10.21 | Already allowed; refresh installed environments when needed. |

Versions above are public PyPI metadata observed on the audit date, not an
inventory of an installed deployment. Modal's queried release was 1.6.2.
Sources: [PyPI](https://pypi.org/),
[Core requirements at the proposed commit](https://github.com/Comfy-Org/ComfyUI/blob/067f853305e74a051dd17f15964d5b0e7887abf5/requirements.txt).

Read-only AST/source checks confirmed the Transformers/kernels incompatibility,
the missing Spectrum-MOSS required input, and the unchanged Core Kitchen/frontend
requirements. No full test suite, package installation, or GPU inference was run.
Local evidence is in `.cache/component-audit-2026-10-25/`; model candidates below
are carried forward from the previous audit, not a fresh model-repository review.

## Previously waiting for an upstream release

- **ComfyUI** is pinned to official master `bc5f880d` (2026-10-23) because
  Fibo 2.0 support
  ([Comfy-Org/ComfyUI#17240](https://github.com/Comfy-Org/ComfyUI/pull/17240))
  is not in a release yet. The v0.42.x tags (through v0.42.5) are cut from a
  release branch without it. v0.43.0 now includes the merge; see the recheck above.

## Do not bump without graph changes

- **LTXVideo** (pinned `3bf3ca62`; upstream was 6 commits ahead on
  2026-10-21). Upstream removes `LTXVSDRToHDRWorkingSpace` and `LTXVSaveHLG`.
  Our SDR→HDR graph in `h3_app/workflows/upscale.py` uses both. Upstream
  replaced them with core colour-space conversion and Create/Save Video.
- **Spectrum-MOSS v0.8.1** (pinned `3fde0674`). Adds a required
  `warn_unverified_sampler` input that our graph does not pass.

## Model candidates

- **Singularity V2 as default.** WarmBloodAban's official
  `Minimax-h3_Singularity_ref2va_v2_pruned_int8.safetensors` (2026-10-23) is
  the opt-in "Singularity V2" base model. v1.3 stays the default and
  preloaded profile until V2 has been compared on our prompts. There is no
  official V2 W4A8 prune yet.
- **Fibo 2.0 turbo LoRA**
  (`loras/fibo_2.0_turbo_lora_avg_rank_270_bf16.safetensors`) is in
  `Comfy-Org/Fibo-2.0` without a model card entry or workflow. Wait for
  documented settings.
- **MOSS-Image `moss_image_color_depth_control` LoRA** in
  `Comfy-Org/MOSS-Image` is a feature candidate.

## Feature candidates that need integration work

- **T8 Audio 1.100+**: AudioFirst diagnostics, Wuli Hybrid workflows, and
  LoRA scoping by generated modality. These need graph and UI work, plus a
  model review for Hybrid. Upstream showed no benefit for our
  voice-conditioning graph.
- **Drama 3**: the upstream editor, code persistence and optional forced
  aligner are separate from the decoding fixes we already adopted.

## Unverified

- The 2026-10-23 bump (ComfyUI `bc5f880d`, Kitchen 0.2.40, aimdo 0.6.0,
  Spectrum v0.2.29) has only passed CPU tests. Smoke-test Fibo 2.0
  text-to-image, Qwen Image 2.1, Drama 3 and Spectrum H3 forecasting on a GPU.
  Drama 3 uses Core's `FixedKVBias`/`FixedKVCache` and Qwen3.5 layers, whose
  signatures were unchanged at the new pin.
- The Drama 3 SGLang venv now includes `torch_memory_saver==0.0.10`. How it
  interacts with the sleep/checkpoint path has not been tested on a GPU.
- Drama 3 sleep on drivers 580-609 (Colab) depends on Omni's host-SHM stage
  transport and on checkpointing without `--launch-job`; neither has run on a
  GPU yet. `tests/colab_drama3_sleep.py` is the check. SGLang-Omni has no
  transport setting, so H3 overrides its CUDA platform; drop that if upstream
  adds one.
- No Linux/Python 3.12 dependency lock exists. Git sources are pinned by SHA,
  but most packages use ranges. Record a working resolution once a GPU
  environment is validated.

## Known upstream issue

- [Qwen Image 2.1 Lightning edit drift](qwen-lightning-edit-drift.md): a
  leftward shift in Max resolution edits with the LightX2V Lightning adapter.
  It is reported upstream with no fix. The official eight-step Turbo is the
  available fallback.
