# Component update audit — 2026-10-22

Rechecked the pinned repositories, packages and main Hugging Face model repositories against the [2026-10-21 audit](upstream-audit-2026-10-21.md). Only SGLang-Omni and ComfyUI moved upstream. One upgrade was applied. Raw comparisons are in ignored `.cache/component-audit-2026-10-22/`.

## Applied

**SGLang-Omni `f31cf9e9` → v0.1.9 (`8200f499`).** This is the first release that includes Drama 3. The 3 commits after the tag touch only CosyVoice 3, Qwen3-TTS and MiniCPM-o.

- [#2844](https://github.com/sgl-project/sglang-omni/pull/2844): the Drama 3 `lm_head` keeps only the 4,097 main-token candidates (EOS plus 4,096 semantic tokens) instead of the 252,288-row text vocabulary, about 16 MB instead of 1.0 GB. The Triton semantic kernel and first-frame CUDA graph use the narrow row. Upstream measured RTF 0.141 → 0.135 on one H200 with unchanged sampling; a unit test checks that sampling from the narrow row matches sampling from the full vocabulary.
- [#2964](https://github.com/sgl-project/sglang-omni/pull/2964): SGLang 0.5.21 → 0.5.22, with torch 2.14.1, torchvision 0.29.1, transformers 5.19.0, flashinfer 0.7.1 and torchcodec 0.17.0. These land only in the isolated Drama 3 venv. `requires-python` is now `>=3.11`; we use 3.12. `torch_memory_saver==0.0.10` and the `[tool.uv]` protobuf override are unchanged.

SGLang 0.5.22 drops the `tc_piecewise` CUDA graph backend. Our `cuda_graph_backend_prefill=disabled` is still a valid choice (`full`, `breakable`, `disabled`). `first_frame_cuda_graph_max_requests`, `window_cuda_graphs`, `codec_path`, `max_running_requests` and the memory-fraction flags still exist. The top-k sentinel `1 << 30` is within `MAXIMUM_TOP_K` (int32 max). The pin change alters `drama3_sglang_fingerprint()`, so the next setup rebuilds the venv.

## No action

- **ComfyUI** (`7a283320`, 5 behind): partner nodes (HeyGen, Krea, VEED, Mirelo) and workflow templates 0.11.94 only. No release contains the Fibo merge yet; v0.42.1 is still the latest.
- **LTXVideo** and **Spectrum-MOSS**: unchanged since 2026-10-21; both still need graph changes before a bump.
- KJNodes, ControlNet Aux, the latent upscaler, Spectrum H3 and T8 audio have no commits beyond the 2026-10-21 assessment. The other repositories match HEAD. The only new package version is openai 3.36.0, which needs no change.

## Model candidates

- **Singularity V2 INT8** (`Minimax-h3_Singularity_ref2va_v2_int8.safetensors`, 34.0 GB): wider action coverage, closer prompt adherence and a fix for audio reference conditioning. We use the pruned v1.3 INT8 file (21.0 GB), and there is no pruned V2 yet.
- **Fibo 2.0 turbo LoRA** (`loras/fibo_2.0_turbo_lora_avg_rank_270_bf16.safetensors`, 2.0 GB) was added to `Comfy-Org/Fibo-2.0` without a model card entry or a workflow. Wait for documented settings before wiring it in.

## Validation

`h3_requirements.py` selftest passed. 170 focused tests passed (upstream upgrade, component upgrade, Drama 3 serving with the pin assertion updated, Drama 3, Drama 3 checkpoint, Modal lifecycle, setup worktree). No GPU inference, environment rebuild or deployment was run. Rerun setup and restart SGLang, or rebuild Modal, to activate the new pin.
