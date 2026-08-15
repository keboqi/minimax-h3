# Code review and refactor notes

## LTX 2.5 and shared-gallery review — 2026-08-12

1. **Consolidated lazy-model freshness checks.** LTX had grown a UI-local
   manifest validator while H3 profile, INT8 VAE, and SeedVR2 lazy downloads
   still treated any file larger than 1 MiB as current. The source, remote
   filename, pinned SHA-256, and recorded byte size are now validated once in
   `h3_models.py` and reused by every lazy model path.
2. **Hardened the Comfy-ready NVFP4 boundary.** The replacement checkpoint is
   pinned by SHA-256 and its safetensors header must contain `.comfy_quant`
   markers before a prompt can be submitted. INT8 ConvRot and BF16 remain
   direct Lightricks assets.
3. **Fixed full-gallery deletion.** Display remains capped by
   `GRADIO_GALLERY_LIMIT`, but **Empty gallery** now scans all managed outputs
   instead of deleting only the newest visible page.
4. **Preserved the validated graph.** The working LTX 2.5 node ordering,
   distilled sigma schedule, shared ComfyUI queue, and H3/LTX gallery family
   routing were reviewed and intentionally left unchanged.

## Current review

The repository-level review found and addressed these issues:

1. `setup_h3.py` called `re.split()` without importing `re`, so a clean local
   ComfyUI dependency install failed at runtime even though bytecode compilation
   passed.
2. Local and Modal installers duplicated the ABI-sensitive package names and
   requirement parsing. The policy now lives in `h3_requirements.py`, including
   a dependency-free self-test.
3. ComfyUI history entries could construct paths outside the configured output
   directory. Output lookup now resolves each candidate and rejects paths that
   are not descendants of `OUTPUT_DIR`.
4. Generated Python caches and the large local `h3/` runtime had no repository
   exclusions. A project-specific `.gitignore` now prevents accidental commits.
5. The README described only the latest NumPy fix. It now documents purpose,
   prerequisites, local and Modal deployment, security-relevant binding
   behavior, validation, and repository structure.

The large graph builders and FirstBlockCache execution path were intentionally
left structurally intact because they encode runtime-sensitive ComfyUI ordering
and node contracts. Their existing self-tests remain the safer regression
boundary.

## Community optimization review — 2026-08-08

The latest H3 and ComfyUI work was compared against this deployment:

1. **Adopted: H3-native zero-copy Sol-Attn.** The earlier Blackwell plugin used
   a generic FlexAttention path. The deployment now pins Saganaki22's
   Apache-2.0 `ComfyUI-sol-attn` commit
   `90467f1c633ce53af7d77e4a1cc243b5001d89b0` and builds the
   `MiniMaxH3MemoryEfficientSolAttentionPatch` node. Its strided fused-QKV path
   avoids three long-sequence contiguous copies. Conditioning KV remains exact,
   INT8 QK/PV stays disabled, and one tail transformer block remains dense by
   default.
2. **Adopted: ConvRot feed-forward chunking.** Quality checkpoints now receive
   `MiniMaxH3ChunkFeedForward` with two chunks above 8K tokens. Upstream reports
   bit-identical output for its tested INT8 ConvRot block and a 37% reduction in
   MLP peak activation memory.
3. **Adopted: hardware-aware memory mode.** Current ComfyUI enables its new
   dynamic residency, pinned-memory, and prefetch system by default, but its
   community thread also contains high-memory slowdown reports. Local launch
   now uses Dynamic VRAM below 64 GiB and `--gpu-only` at 64 GiB+, while the
   known 96 GiB Modal target stays GPU-only.
4. **Deferred: AdaptiveCache.** It has promising early RTX 5090 results and a
   more sophisticated partial-tail cache, but is new, lossy, GPL-licensed, and
   not yet validated against this repository's FL2VA/Ref2VA audio safeguards.
5. **Adopted: Spectrum forecasting as the normal default.** Community results
   now cover RTX 3090, 4090, 5090, RTX PRO 6000, and an AMD R9700 report, with
   typical reported sampler-time reductions around 30–45%. Spectrum v0.2.2 is
   pinned and uses its corrected default path: offline smoothing replay,
   video blend 0.5, and audio blend 0. FirstBlockCache remains the lower-memory
   fallback and EasyCache remains available; the three modes are mutually
   exclusive. Spectrum is still approximate, and its upstream documentation
   reports possible motion, anatomy, and trajectory changes, so quality-critical
   outputs still require uncached A/B review.

Primary sources:

- https://github.com/Saganaki22/ComfyUI-sol-attn
- https://github.com/Comfy-Org/ComfyUI/discussions/12699
- https://github.com/FFFFFFpy/ComfyUI-MiniMaxH3-AdaptiveCache
- https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3

## Turbo optimization update — 2026-08-09

1. **Adopted: Sol-Attn v0.6.0.** Local and Modal provisioning now pin commit
   `bd28909e6a152ba5db4e3590d2d9df1c249943f2`. This release adds the SM120
   pointer-kernel path and the H3 fused-modulation node while retaining the
   existing zero-copy H3 attention node contract.
2. **Adopted: bit-exact fused modulation for LightX2V Turbo.** LightX2V applies
   `MiniMaxH3FusedModulation` after its core LoRA. Larry deliberately remains
   unfused pending GPU validation with its runtime AdaLN forward patches. The
   pinned Larry node receives a fail-closed provisioning patch: `_unique_t()`
   now mirrors ComfyUI's dynamic set of video, audio, visual-conditioning, and
   audio-conditioning timesteps from the actual packed layout. This fixes both
   the initial 3-vs-2 row mismatch and the later 4-vs-3 mismatch when video and
   audio schedules diverge after the first denoising step.
   Larry still uses Sol attention and ConvRot FFN chunking where selected.
3. **Adopted: thresholded Sol for Turbo.** Auto attention now enables Sol for
   Turbo jobs at or above the existing 8K packed-token threshold. Switching the
   UI into Turbo selects Auto rather than forcing Dense; explicit Dense and
   Sol-Attn selections remain available. Reference Turbo always uses Sol because
   its uploaded-media conditioning cannot be included in the pre-encoding token
   estimate.
4. **Still deferred: step caching in Turbo.** Spectrum, FirstBlockCache, and
   EasyCache remain forced off because their approximation error is more
   sensitive in four- and six-step schedules. Sol INT8 QK/PV also stays off.

Primary source: https://github.com/Saganaki22/ComfyUI-sol-attn/releases/tag/v0.6.0

## Spectrum Turbo update — 2026-08-10

1. **Adopted: Spectrum v0.2.5.** Local and Modal provisioning now pin commit
   `4b9a7d1163348c67e7e475423f24f8b7abb23565`. The release separates the
   bounded causal history from the offline replay archive and defaults both to
   system RAM in this deployment, avoiding the earlier all-anchor CUDA archive
   growth during two-pass replay.
2. **Adopted as default: Spectrum with Turbo.** Spectrum now recognizes Larry's
   exact `_turbo_sampler` contract, while its existing allowlist covers the RES
   multistep sampler used by LightX2V. Both retain the conservative limit of one
   consecutive forecast followed by a completed native refresh. Entering Turbo
   selects Spectrum by default and displays an exact-seed A/B warning.
3. **Still blocked in Turbo: FirstBlockCache and EasyCache.** These modes are
   automatically replaced with Off because their low-step composition has not
   received the same sampler-specific upstream validation. Sol INT8 QK/PV also
   stays off.
4. **Still deferred: AdaptiveCache.** New RTX 5090 results show additional
   speed when combined with Sol and Turbo, but the published W4A8 Turbo samples
   also contain unresolved quality failures. The plugin remains a separately
   installable GPL experiment pending BF16/NVFP4 audiovisual A/B validation.

Primary sources:

- https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/releases/tag/v0.2.5
- https://github.com/FFFFFFpy/ComfyUI-MiniMaxH3-AdaptiveCache

## Upstream optimization refresh — 2026-08-11

1. **Adopted: Sol-Attn v0.6.1.** Local and Modal provisioning now pin commit
   `e1d211026583064d33dc4326207c6502e2442208`. The release adds compatibility
   with KJNodes' single-item low-VRAM activation handoff. Upstream reports no
   kernel, attention-math, weight, dispatch, or numerical-output changes, so
   the reviewed v0.6.0 performance and quality policy remains applicable.
2. **Already inherited: native chunked H3 VAE I/O.** ComfyUI commit
   `2a68ce33b4c9ea6ee4283e618a74560cefb32694` streams decoded temporal chunks
   into the intermediate-device output buffer and moves encode clips to the
   execution device one at a time. This reduces peak H3 VAE VRAM without a
   custom workflow change. Both local setup and Modal clone current ComfyUI
   HEAD, so new deployments already contain it.
3. **No change: Spectrum and Larry Turbo.** Their pinned commits are still the
   current upstream heads (`4b9a7d1` and `55fee86`, respectively).
4. **Still deferred: AdaptiveCache.** Its upstream head is unchanged since the
   previous review. The published RTX 5090 speed results remain promising, but
   the unresolved W4A8 Turbo quality failures and missing NVFP4 audiovisual
   exact-seed validation still make it unsuitable as a default here.

Primary sources:

- https://github.com/Saganaki22/ComfyUI-sol-attn/releases/tag/v0.6.1
- https://github.com/Comfy-Org/ComfyUI/commit/2a68ce33b4c9ea6ee4283e618a74560cefb32694
- https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/releases/tag/v0.2.5
- https://github.com/FFFFFFpy/ComfyUI-MiniMaxH3-AdaptiveCache

## Optional INT8 VAE and EasyCache Turbo update — 2026-08-12

1. **Added as default-off: INT8 ConvRot video VAE.** The experimental Kijai
   checkpoint is recorded in the generated model catalog but excluded from the
   preload set. Selecting it in the UI downloads it on demand; FP16 remains the
   default and reviewed fallback.
2. **Enabled as default-off: EasyCache with Turbo.** ComfyUI v0.31 fixes the H3
   audio-carry mutation that previously corrupted audio under EasyCache. Turbo
   continues to default to Spectrum, and selecting EasyCache displays an
   exact-seed A/B warning because low-step approximation quality remains
   experimental.
3. **Enabled as default-off: FirstBlockCache with Turbo.** The existing bounded
   cache window, temporal guard, and consecutive-hit limit remain active. The UI
   displays the same exact-seed A/B warning because low-step composition has not
   received sampler-specific upstream validation.

Primary sources:

- https://github.com/Comfy-Org/ComfyUI/pull/15334
- https://github.com/Comfy-Org/ComfyUI/pull/15390

## Earlier v37 → v38 review

## Findings addressed

### 1. Local and Modal model provisioning had duplicated sources and logic

`setup_h3.py` and `modal_h3.py` independently contained the same model repo
constants, eight model specifications, manifest parsing, Hugging Face metadata
queries, stale-file decisions, parallel downloader, atomic replacement, and
configuration generation.

This was the highest maintenance risk: a future model change could work locally
but silently leave Modal on the old file or repo.

**Refactor:** all of that is now in `h3_models.py`.

### 2. Model metadata was represented as positional tuples

The meaning of:

```text
(repo_id, folder, filename, source)
```

depended on tuple position and was repeated in two files.

**Refactor:** model entries now use a frozen `ModelSpec` dataclass.

### 3. Generated h3_models.json was constructed as dense inline dictionaries

That made schema changes hard to review and duplicated schema generation.

**Refactor:** one `_build_config()` path produces the schema for both
environments.

### 4. Modal mixed infrastructure concerns with generic model-management code

The Modal file contained several hundred lines unrelated to Modal itself.

**Refactor:** Modal now owns image/volume/service lifecycle; generic model
management is delegated to the shared module.

### 5. Resolution and sampling policy was partly encoded in callback conditionals

**Refactor:** `RESOLUTION_TIERS` and `SAMPLING_PRESETS` are canonical lookup
tables.

## Intentionally not rewritten

`generate()` and the ComfyUI graph builders are large, but they encode several
runtime fixes that have already been validated in production-style testing:

- Turbo is independent from the Speed, Quality, or Original base profile.
- Four-step LightX2V Reference Turbo uses its dedicated Ref2V adapter. Eight-step
  LightX2V and Larry temporarily reuse their FL2VA LoRAs through separate Ref2VA
  configuration keys.
- Turbo mode/variant defaults update outside the generation queue (Larry 6,
  LightX2V 4), while the resulting step control remains user-editable.
- FirstBlockCache must precede Sol-Attn.
- Turbo defaults to Spectrum; FirstBlockCache and EasyCache remain explicit,
  default-off experimental choices.
- H3 full-model compile is omitted because it conflicts with the active
  attention and cache optimizations without improving measured performance.
- SaveVideo codec must be the literal `auto` string.

A larger rewrite of that path would create more regression risk than immediate
maintenance value. The current graph helpers already separate model-stack,
FL2VA, Ref2VA, and sampling construction reasonably well.

## Validation

v38 passed:

```text
Python compile: h3_models/setup/modal/gradio/H3Acceleration
Bash syntax: run_h3.sh
h3_models selftest
setup_h3.py --help
Gradio selftest
Gradio build_ui()
Modal module construction with a fake Modal API
top-level duplicate-definition scan
```

A fixed-input behavioral comparison was also run between v37 and v38:

```text
FL2VA graph: identical, 16 nodes
Ref2VA graph: identical, 16 nodes
```

The comparison included FirstBlockCache + Sol-Attn, Quality model selection,
scheduler, cache parameters, VAE/CLIP loaders, and SaveVideo wiring.


## v39 attention composition

Native SageAttention is now deliberately separate from the Sol model patch.

`SERVER_ATTENTION_BACKEND=sol` means Sol controls sparse H3 routing.
`SERVER_DENSE_ATTENTION_BACKEND=sage|pytorch` reports the global ComfyUI backend
used by dense and fallback calls.

A GPU-side probe controls `--use-sage-attention`, preventing a bad Sage build
from breaking service startup.

The current ComfyUI 0.33.1 deployment supersedes this launch policy: both local
and Modal launchers now pass `--use-ck-attention` and report `comfy-kitchen` as
the dense/fallback backend. H3 workflows default to Sage 2 after local comparison
showed it slightly faster than Kitchen. This applies pinned KJNodes'
`PathchSageAttentionKJ` model override in `auto` mode against the installed
SageAttention 2.2.0 wheel. Kitchen remains selectable and backs Sol's dense
calls. Sol remains an explicit workflow override (including the optional
Auto routing policy). The
source pin is the immutable v0.33.1 commit
`72865f4f27eaf5396f8f36370e0a2be3a9a090ee`, paired with `comfy-kitchen==0.2.31`.


## v40 Turbo replacement

The Larry-specific Turbo implementation was replaced by Kijai's converted
LightX2V v0.1 LoRA:

```text
loras/minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors
```

Core `LoraLoaderModelOnly` now applies the LoRA at strength 0.75 and normal
`res_multistep` sampling is used. The Larry custom loader/sampler repository is
no longer provisioned.

## v42 SageAttention install optimization

The previous default source build was unnecessarily slow for the known target
environment. v42 first installs the exact prebuilt wheel:

```text
sageattention-2.2.0-cp312-cp312-linux_x86_64.whl
```

from `JahJedi/sageattention-flashattn-blackwell-cu130-torch211-cp312`. The pinned official source build remains fallback-only.


## v43 Torch/Sage ABI correction

Observed runtime: ComfyUI dependencies upgraded the initially pinned
`torch 2.11.0+cu130` to `torch 2.13.0+cu130`. The prebuilt SageAttention wheel
was compiled against the 2.11 C++ ABI, causing an undefined `c10` symbol.

v43 filters ComfyUI's unpinned Torch trio from dependency installation and
automatically reconciles existing local environments back to 2.11+cu130 before
Sage installation. Modal uses the same policy.


## v46 Sage validation fix

Sage validation now uses a brand-new Python subprocess. Automatic Sage source compilation has been removed.


## v47 dependency ABI repair

Pinned NumPy 1.26.4 / SciPy 1.15.3 and added an existing-install reconciliation
check. ComfyUI requirements now filter NumPy/SciPy as well as the Torch trio.
The missing dependency helper functions from the v46 refactor were restored.


## v48 selectable Turbo implementations

Larry v4-600 EMA is provisioned again alongside LightX2V v0.1 and is the new UI
default. Larry runs at a six-step default and strength 1.0 through the pinned
`MiniMaxH3TurboLoRA` and `MiniMaxH3TurboSampler` nodes. This preserves its
runtime AdaLN injection and activation-space LoRA path on the pruned quantized
H3 bases. LightX2V remains available at four steps and strength 0.75 through
core `LoraLoaderModelOnly` plus normal `res_multistep` sampling.

Both choices remain experimental for Ref2VA because both LoRAs were trained for
FL2VA. The model configuration keeps separate per-choice Ref2VA keys for future
dedicated weights.

## v49 LightX2V v1.0 Turbo adapters

The preview LightX2V v0.1 adapter is replaced by the official ComfyUI BF16 v1.0
release. The UI now exposes the 4-step 768p and 8-step 544p adapters separately,
uses strength 1.0 for both, and synchronizes the editable step control to the
selected adapter. Both official files are provisioned directly from
`lightx2v/Minimax-h3-Turbo`.

## v50 profile-aware Turbo loading

Original BF16 now applies both Larry and LightX2V Turbo adapters through runtime
bypass, preventing ComfyUI's reversible merge path from retaining up to 37.3 GiB
of original H3 projection weights alongside their patched copies. Speed and
Quality instead merge both adapter families: their compact quantized bases have
ample headroom, merging avoids per-step LoRA matrix multiplies, and it preserves
compatibility with fused kernels that read weights without calling a module's
forward method. The bundled LightX2V bypass accepts only the exact official
50-block plus two-refiner, four-projection layout and fails closed on missing keys, unexpected
ranks, unsupported extensions, or incomplete hook installation.

## v51 ComfyUI and Spectrum refresh

ComfyUI is pinned to `2220d111c8b036f094eb465400fdf962626e4afa`.
Relative to the previous pin, this enables automatic Dynamic VRAM on supported
NVIDIA WSL installations and carries only the intervening reviewed maintenance
fixes. Spectrum is updated from v0.2.5 to v0.2.7 at
`7911ec7827921de599492f21eade181211266029`. The new release supports native
ER-SDE and protects its offline replay tail, but the launcher continues to use
Larry and RES samplers. Its experimental model-aware controller is explicitly
set to `off`, preserving the existing scheduling, fitting, and correction
behavior.

Primary sources:

- https://github.com/Comfy-Org/ComfyUI/commit/2220d111c8b036f094eb465400fdf962626e4afa
- https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/releases/tag/v0.2.7

## v52 dedicated LightX2V Ref2V Turbo adapter

The four-step LightX2V option now routes reference-media generation to the
official `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` adapter.
The new adapter is trained for four-step 544p inference; FL2VA continues to use
the v1.0 768p adapter. Provisioning preloads and validates both mode-specific
files, while older generated configurations retain their existing FL2VA fallback
through the app's compatibility loader. No dedicated Ref2V files are currently
published for LightX2V eight-step or Larry Turbo, so those reference routes
remain experimental shared-LoRA paths.
