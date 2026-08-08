# Code review and refactor notes

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

- Turbo is independent from Speed/Quality.
- Turbo steps are user-editable.
- FirstBlockCache must precede Sol-Attn.
- Turbo disables cache by default.
- quantized H3 full-model torch.compile stays blocked.
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
