# Component and community audit — 2026-10-02

Reviewed the initial source and workflow contracts, compared all 14 configured Git repository pins with live upstream HEAD, and queried PyPI for 24 primary packages. The working tree was clean at the start. Seven repositories were ahead of our pins; seven matched. The initial assessment added only this report. The three subsequently requested priorities are now implemented in the repository, as described below; no deployment was performed.

## Implementation status

- ComfyUI is pinned to `65787d668397d230bf5839d69a0a7239e2dad378` with matched frontend `1.53.10`. Both setup and Modal builds consume the shared pins; existing installations need setup/image rebuild to receive them.
- Turbo counts native Qwen 2.1 spatial tokens and retains its published raw schedules, null terminal, and nine-step 7+2 handoff.
- Base sampling defaults to `Qwen 2.1 (resolution-aware)`, using output-latent dynamic shifting and terminal `0.02`. Legacy simple/normal/beta remain available.
- References pass through native `JoinImageWithAlpha`, restoring RGBA from RGB plus inverse alpha before conditioning. Original uploaded files are staged without image conversion.
- CPU regression checks cover square, rectangular and 4 MP schedules, native edit versus empty-latent metadata, six/nine-step Turbo, base graph wiring, batching/provenance, and pinned native soft-alpha reconstruction. GPU generation, offload behavior and VRAM/latency measurements remain unverified.

Implementation clarified the token-count finding: `EmptyLatentImage` produces an **8×** grid with `downscale_ratio_spacial=8`, which ComfyUI resizes to the native **16×** Qwen grid before sampling. The old division happened to count this empty grid correctly, but undercounted native edit latents by four. Both new schedulers resolve that metadata before counting tokens. The table below describes native edit latents, rather than every initial latent representation.

## Recommended order

### 1. Correct Qwen Turbo token counting

There is a concrete reference-pipeline mismatch in `custom_nodes/H3Acceleration/__init__.py:1707`. `H3Qwen21TurboSigmas.calculate` counts `(latent_height // 2) * (latent_width // 2)` tokens. Qwen Image 2.1 consumes **unpatched** latents: the correct count is `latent_height * latent_width`. Its VAE compresses spatial dimensions by 16 and its transformer configuration has `patch_size: 1`.

The official pipeline computes schedule mu from `latents.shape[1]`, after flattening all latent spatial positions. Our initial count for native edit latents was four times too small:

| Output | Current token count / mu | Reference token count / mu |
|---|---|---|
| 1024×1024 | 1024 / 0.538710 | 4096 / 0.693548 |
| 2048×2048 | 4096 / 0.693548 | 16384 / 1.312903 |

Both six-step and nine-step Viggle modes use this node. Retain their published raw sigma sequences, `shift_terminal=null`, and the nine-step 7+2 LoRA/base handoff; correct only the resolution calculation. The existing numeric test in `tests/test_qwen_image21.py:743` repeats the same incorrect formula, so passing it does not establish reference parity. Add independent reference expectations at 1 MP, 4 MP, and a nonsquare edit size, covering six/nine steps and continuity at the shared sigma.

Sources: [official pipeline at Viggle's published Diffusers revision](https://github.com/huggingface/diffusers/blob/80c7ed262aeffbeb43ef13ae04baeb9b84515a69/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py), [transformer configuration](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/transformer/config.json), [Viggle scheduler](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo/blob/main/scheduler/scheduler_config.json).

### 2. Upgrade ComfyUI with its matching frontend

Our pin `986c4d154ef8c288382ac87d956b52a2b640c8b3` is 20 commits behind observed HEAD `65787d668397d230bf5839d69a0a7239e2dad378`. Two changes directly affect our H3 paths:

- **VAE offload fix:** wraps RMS-normalization weights in `CastBiasWeightContext(..., offloadable=True)` before fused linear kernels. This is relevant to native video VAE execution, including the INT8/LynnReal choices; it does not establish a TRT-engine improvement.
- **Lower peak VRAM:** moves embedding/packing into `_embed_and_pack`, letting embedding temporaries go out of scope before transformer blocks. This also changes the native H3 forward implementation that our accelerator wrappers and Spectrum adapter interact with.

Target requirements change frontend **1.53.6 → 1.53.10**, workflow templates **0.11.70 → 0.11.74**, and embedded docs **0.5.12 → 0.5.13**. Kitchen remains **0.2.36**. Use the frontend matched to this Core revision rather than independently selecting the latest PyPI frontend, 1.54.8.

Update `COMFY_REF`, `COMFY_FRONTEND_VERSION`, self-test expectations and exact-source contract fixtures together. Recheck the full H3 wrapper stack, native INT8/LynnReal VAE under offload, repeated requests, refinement compiler lifecycle, Qwen generation/editing/cache, and frontend assets/proxy routes. Measure matched-seed quality, latency and peak VRAM on the deployment GPU before promotion; no numeric speedup was measured in this audit.

Sources: [exact comparison](https://github.com/Comfy-Org/ComfyUI/compare/986c4d154ef8c288382ac87d956b52a2b640c8b3...65787d668397d230bf5839d69a0a7239e2dad378), [VAE fix](https://github.com/Comfy-Org/ComfyUI/commit/83071e1a), [embedding lifetime change](https://github.com/Comfy-Org/ComfyUI/commit/2d6b7328), [target requirements](https://github.com/Comfy-Org/ComfyUI/blob/65787d668397d230bf5839d69a0a7239e2dad378/requirements.txt).

### 3. Add a reference-compatible base Qwen schedule

The pinned Core `QwenImage21.sampling_settings` fixes shift at **0.69**, approximately its 1024×1024 value. Our non-Turbo graph uses ordinary `KSampler` and defaults to `simple`; it does not calculate shift from the target latent or apply the official terminal stretching. The official scheduler uses dynamic shifting with anchors `(256, 0.5)` and `(8192, 0.9)` and `shift_terminal=0.02`.

The community [QwenImage21-Tools](https://github.com/xb1n0ry/ComfyUI-QwenImage21-Tools) implements latent-size scheduling and terminal stretching. Its token calculation agrees with the official pipeline. Adapt the math into our existing scheduler node family, or pin and validate the external node. Feed the **resolved target latent**, including match-input/max-resolution edits, into a custom sampling graph. Keep base-model terminal stretching separate from Viggle's null-terminal schedule. This restores sampling behavior; it is not a universal quality guarantee.

Sources: [our pinned Core definition](https://github.com/Comfy-Org/ComfyUI/blob/986c4d154ef8c288382ac87d956b52a2b640c8b3/comfy/supported_models.py), [official scheduler configuration](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/scheduler/scheduler_config.json), [community implementation](https://github.com/xb1n0ry/ComfyUI-QwenImage21-Tools/blob/main/qwen_image21_tools.py).

### 4. Preserve alpha when loading Qwen edit references

Our Qwen graph connects only output 0 of `LoadImage` to `TextEncodeQwenImage21`. The pinned loader returns RGB images and alpha separately as an inverse mask; that mask is unused. Native Qwen conditioning can accept four-channel images and keeps all four channels for the VAE while compositing over white for the vision encoder. Consequently, this graph does not provide the source alpha channel to that conditioning path.

Use an RGBA-preserving loader or recombine image and inverse mask before encoding. Verify transparent/semitransparent edit references, RGB references, multiple references and saved PNG alpha. The existing `VAEDecode → SaveImage` output route and file-path resolver do not themselves force RGB; this finding concerns reference input.

Sources: [pinned LoadImage](https://github.com/Comfy-Org/ComfyUI/blob/986c4d154ef8c288382ac87d956b52a2b640c8b3/nodes.py), [pinned Qwen conditioning](https://github.com/Comfy-Org/ComfyUI/blob/986c4d154ef8c288382ac87d956b52a2b640c8b3/comfy_extras/nodes_qwen.py).

## Community improvements worth evaluating

### Qwen Image 2.1

| Candidate | What it adds | Assessment for this project |
|---|---|---|
| [Official PE with community MTP backend](https://github.com/mozophe/ComfyUI-Qwen-Image-2.1-PromptEnhancer-MTP) | Local official T2I/edit prompt enhancers, MTP decoding, optional llama.cpp process | Useful optional backend. Current app enhancement uses hosted services/custom instructions. Author reports roughly 2× T2I / 3.5× two-image edit token throughput with Q8_0 llama.cpp at official lengths on a 4090 Laptop; these are enhancer measurements, not image-generation speedups or comparisons with our hosted backend. Validate model provisioning, subprocess cleanup, cancellation, memory release, multi-image ordering and output parsing. |
| [LanPaint](https://github.com/scraed/LanPaint) | Masked Qwen 2.1 editing and inpainting of transparency | Strong feature candidate for precise local edits. Requires mask input/UI, sampler graph and alpha handling. Start with the base model and acceleration Off; compatibility with our Spectrum/Viggle stack is unverified. |
| [Alibaba PAI Fun ControlNet Union](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union) | Eight structural controls plus masked inpainting | Our pinned Core already contains Union support, but our app graph has no control loader/application or mask inputs. Expose the native path rather than adding a redundant Core compatibility patch. Provision the matching control checkpoint and preprocessors; validate control sizing/strength and reference edits. |
| [Alibaba PAI Fun Acc 4-step PDD](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs/blob/main/README.md) | A smaller rank-64 few-step adapter for T2I/editing | Experimental alternative to Viggle, requiring its own PDD runtime. Published scripts use `QwenImage21PDDScheduler`, a per-step callback and `use_kv_cache=False`; loading the LoRA into our existing Viggle/Euler graph is insufficient. Evaluate small text, edit fidelity, blur/darkening and actual NFE/latency. |
| [Viggle v0.3](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo) | Cleaner six-step surfaces and a nine-step detail handoff | Already integrated, including custom nine-step handoff. v0.2.1 remains a reasonable choice for crisper texture; upstream explicitly describes v0.3 as a tradeoff. Fix our token count before comparing variants. |

Native prefix KV caching, INT8 ConvRot/BF16 model choices, W4A8/INT8/BF16 encoders, transparency prompting, and Spectrum quality/preview modes are already present. No verified Qwen 2.1-specific Nunchaku drop-in was established; older Qwen support alone is not compatibility evidence for the new 7B single-stream architecture.

### MiniMax H3

| Candidate | What it adds | Assessment for this project |
|---|---|---|
| [LongLive-Plug](https://github.com/NVlabs/LongLive) and [converted H3 few-step LoRA](https://huggingface.co/Kijai/MiniMax-H3-experimental/blob/main/loras/minimax_h3_ELM_longlive_plug_4step_lora_bf16.safetensors) | Reusable few-step distillation adapter | Worth an explicit experimental comparison with LightX2V/Larry/PDMD. Conversion availability alone does not prove schedule, conditioning-mode, quantized-loader or refinement compatibility. Official H3 cards say to use few-step and CFG adapters separately. Do not infer long-video continuation from the name. |
| [H3 Motion Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context) / [Context Loop](https://github.com/ethanfel/ComfyUI-MiniMaxH3-Context-Loop) / [Extender](https://github.com/tritant/ComfyUI_MiniMax_H3_Extender) | Motion/audio clip continuity, scene review, checkpoint/resume, per-clip references | Strong product extension for long videos. Choose one implementation and prove its native layout contract against our Core pin. Requires scene state, cache invalidation, cancellation/resume, audio clock and seam validation; installing a node alone will not add this to the Gradio app. |
| [LanPaint AV inpainting](https://github.com/scraed/LanPaint) | Masked H3 video/audio editing | Promising separate editing workflow. Requires per-frame masks/audio intervals and nested AV latents. Test bounded clip lengths and latency; leave stacking with distilled models and our accelerator combinations unproven. |
| [T8 1.88.1](https://github.com/T8mars/comfyui-minimax-h3-audio-T8) | HyperVAE examples, semantic bridge composition, Veda/separated sampling and split-audio fixes | Optional feature work. Exact old/new `h3_t8/conditioning.py` bytes match, so no upgrade is needed for our current voice-conditioning implementation. Adopt only with the corresponding workflow/model/runtime contract. |

Currently integrated: PDMD 4-step adapters, LightX2V/Larry choices, learned latent upscale/refinement, native optimized video VAE, LynnReal Light INT8, TRT VAE, conditioning cache, Spectrum and sparse attention options. FastH3 eight-step V2, TaoMate three-step, and PDMD two-step support was removed on 2026-10-03 after quality testing. These should not be presented as supported community upgrades.

## Other component decisions

- **Gradio 6.27.0 → 6.29.0:** reasonable secondary update. Release notes include sanitizer document-leak cleanup, clearer upload failures and event-validator fixes. Streaming microphone fixes apply only if recording is enabled; the NumPy 2 audio fix does not target our pinned NumPy 1.26.4. Upgrade runtime and test pins together, then run existing settings/upload/gallery/proxy browser checks. [Release notes](https://github.com/gradio-app/gradio/releases/tag/gradio@6.29.0).
- **LTXVideo:** eight commits ahead, adding per-step tiled fusion, temporal window/IC-LoRA streaming and tiling-size helpers; HDR EXR default changes from ACEScct to ACEScg. Useful if expanding the finishing pipeline, rather than a demonstrated current-H3 fix. Updating the pin requires reviewing our exact-ten-template synchronization contract; new bundled workflow files may change that set. Test streaming guide alignment, seams, audio and explicit color-space selection. [Comparison](https://github.com/Lightricks/ComfyUI-LTXVideo/compare/f8387c893de3f652c3a416052c82bba593edcf2b...3bf3ca62).
- **ControlNet Aux 1.1.6:** MediaPipe face-mesh compatibility and dependency cleanup. Current H3/Qwen graphs do not gain speed from this update. Its requirements replace Albumentations with AlbumentationsX and modify native/transitive dependencies, so perform a constrained resolution against our NumPy pin before adopting. [Comparison](https://github.com/Fannovel16/comfyui_controlnet_aux/compare/59b1fc411ede8623b2997855b8018f0b3b6cf49f...0cd29047).
- **KJNodes:** only package version metadata changes to 1.5.2. **H3 latent upscaler:** license-file addition only. **SwiftVR:** README-only commits point to a separate LightX2V backend; updating our source pin adds no runtime optimization. [KJNodes comparison](https://github.com/kijai/ComfyUI-KJNodes/compare/57105374f47d0fbb49c9c3926fb981702e0a4b5c...d3cfe216), [upscaler comparison](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/compare/d7c01b9011f2e8439493f6c02c29995a27df276f...40316cf0), [SwiftVR comparison](https://github.com/H-oliday/SwiftVR/compare/5ca168cef6ca7200f135fdfea85e5e13d12c5b53...dbec2f99).

## Repository inventory

| Component | Configured pin | Observed HEAD | Ahead |
|---|---|---|---:|
| ComfyUI | 986c4d15 | 65787d66 | 20 |
| ControlNet Aux | 59b1fc41 | 0cd29047 | 5 |
| H3 latent upscaler | d7c01b90 | 40316cf0 | 1 |
| KJNodes | 57105374 | d3cfe216 | 1 |
| Larry Turbo | 4274783a | same | 0 |
| LTXVideo | f8387c89 | 3bf3ca62 | 8 |
| PlagueKind SLA | d58d006a | same | 0 |
| Sol-Attn public fork | 930a4d6e | same | 0 |
| Spectrum H3 | 5161f045 | same | 0 |
| Spectrum Qwen | ddb5470f | same | 0 |
| Video Depth Anything | a0db08e6 | same | 0 |
| T8 Audio | 6063fafb | 70fb30f5 | 7 |
| SwiftVR | 5ca168ce | dbec2f99 | 2 |
| H3VAE TRT | 4360e008 | same | 0 |

## Package policy

These are registry latest versions, not installed deployment versions or proven compatible upgrades. Metadata is from [PyPI JSON](https://pypi.org/pypi/gradio/json); individual package records and dependencies are saved in the audit cache.

| Package | Configured policy | Registry latest | Decision |
|---|---|---|---|
| Torch / torchvision / torchaudio | 2.11.0 / 0.26.0 / 2.11.0, cu130 | 2.14.1 / 0.29.1 / 2.11.0 | Keep coherent ABI stack and Sage wheel. |
| NumPy / SciPy | 1.26.4 / 1.15.3 | 2.5.3 / 1.18.1 | Keep native compatibility constraints. |
| Gradio | 6.27.0 | 6.29.0 | Stage after higher-priority corrections. |
| huggingface-hub | >=1.5,<2 | 2.1.1 | Retain cap; select a tested 1.x resolution with Transformers. |
| Transformers | >=4.57.1 | 5.18.0 | Record a tested exact resolution; avoid unattended major drift. |
| diffusers | >=0.36,<0.37 | 0.40.0 | Keep until node imports are validated. Qwen generation currently uses native Comfy, not a Diffusers pipeline. |
| kernels | 0.16.0 | 0.17.1 | Retain FP8-loader compatibility pin until checked as a set. |
| comfy-kitchen | 0.2.36 | 0.2.36 | Current. |
| frontend | 1.53.6 | 1.54.8 | Target 1.53.10 with reviewed Core. |
| Kornia / Kornia RS | 0.8.3 / 0.1.14 | 0.8.3 / 0.2.0 | Keep paired LTX-compatible versions. |
| accelerate / peft / safetensors | >=1.12 / >=0.18 / >=0.7 | 1.15.0 / 0.21.2 / 0.8.0 | Already allowed; lock tested resolution with Transformers. |
| TensorRT cu13 | >=11.2,<12 | 11.3.0.99 | Already allowed; validate native engine build/decode. |
| wsproto | 1.2.0 | 1.3.2 | Optional after WebSocket/proxy checks. |
| OpenAI SDK | runtime >=3.16.2,<4; tests 3.16.2 | 3.23.0 | Major mismatch from old audit is resolved; capture a common tested resolution. |
| Modal | no central exact pin | 1.6.0 | Record deployed SDK; no required migration established. |
| Pillow / aiohttp / onnx | >=10 / >=3.11,<4 / >=1.19,<2 | 12.3.0 / 3.14.3 / 1.23.1 | Already allowed; capture deployment resolution. |

Broad package ranges plus fixed Git SHAs still do not reproduce an image. Save the resolved Linux/Python 3.12 dependency set used by both deployment paths. The lightweight Windows test environment does not establish CUDA/Blackwell compatibility.

## Evidence and validation boundary

Raw comparison responses, package metadata and targeted exact source/model-card files are in ignored `.cache/component-audit-2026-10-02/`. GitHub comparisons cap file lists: T8 conditioning was separately fetched at old/new revisions and compared byte-for-byte. Mutable community source snapshots are research evidence, not proposed production pins. Latest upstream revisions and registry versions are observations as of this audit.

Performed source/contract inspection and public upstream research. No GPU inference, benchmark, fresh Linux/CUDA build, deployed-package inspection, application regression suite, deployment or comprehensive security audit was performed. The scheduler mismatch is established from arithmetic and official source; output-quality effects remain unmeasured. Prioritize scheduler correction and Core/VAE reliability, then base scheduling/alpha fidelity, before expanding accelerators or editing features.
