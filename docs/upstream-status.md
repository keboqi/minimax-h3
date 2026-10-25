# Upstream status

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

## Waiting for an upstream release

- **ComfyUI** is pinned to official master `bc5f880d` (2026-10-23) because
  Fibo 2.0 support
  ([Comfy-Org/ComfyUI#17240](https://github.com/Comfy-Org/ComfyUI/pull/17240))
  is not in a release yet. The v0.42.x tags (through v0.42.5) are cut from a
  release branch without it. Pin the first release that includes the merge.

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
