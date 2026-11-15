# H3 Regenerate IC refinement feasibility

Reviewed 2026-11-14. Source review only; no GPU inference, model-weight download,
dependency changes or runtime changes were performed.

## Recommendation

Add **H3 Regenerate 2K IC (experimental)** as a separate fourth choice under
**High-resolution refinement method**, after the existing full-frame,
rank-16 and rank-384 choices. This is feasible engineering work, but is not a
drop-in replacement for either current refiner. Do not expose it as working
until its conditioning and HyperFlow execution contracts are implemented.

The official IC adapter is rank 384, alpha 384, FP32 and targets 312 linears on
`transformer_ref`. The existing rank-16 and rank-384 options are Kijai's
**non-IC refiner-only** conversions. Rank alone does not identify the method.
The separate Kijai average-rank-21 IC conversion noted in `docs/upstream-status.md`
is another candidate, not an equivalent replacement established by this review.

Sources: [model card](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K-IC),
[adapter configuration](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K-IC/blob/main/lora/lora_config.json).

## What must change

| Area | Current rank-16 / rank-384 integration | IC requirement |
| --- | --- | --- |
| Model | Uses the generation model; standalone existing-video route loads FL2VA | Dedicated Ref2VA refinement model, even after FL2VA generation |
| Adapters | LightX2V FL2VA Turbo + selected refiner; no LMS Sharp | HyperFlow 8-step + LMS Sharp + IC, applied in upstream order |
| Conditioning | No draft-as-reference injection in the refinement stage | Draft vision context in Qwen plus draft latent reference rows |
| Sampling | Euler, `0.909375, 0.857143, 0.666667, 0` | HyperFlow endpoint-aware model execution; up3 grid approximately `0.834942, 0.696852, 0.468753, 0` |
| Audio | Silent sampling audio, original audio retained | Fixed clean silence rows during refinement; source soundtrack restored on output |

HyperFlow is more than a normal linear LoRA: its `TwoTimeEmbedder` combines
the current and next-step time embeddings, with adapter weights for a separate
endpoint branch. Each step updates generated rows' endpoint to `1 - sigma_next`;
reference/audio rows keep their own times. Loading its weights through an
ordinary LoRA loader and changing `ManualSigmas` is insufficient.

The upstream packed layout contains draft vision/text context, draft reference
video, silence audio and target video. Draft reference rows receive 0.001 noise
and stay anchored. Up3 starts from the learned-upscaled draft mixed with noise.
An implementation needs to verify latent normalization, packed row ordering,
time conventions and noise scaling across Diffusers and native ComfyUI.

Sources: [inference loop](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K-IC/blob/main/infer/infer.py),
[HyperFlow implementation](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K-IC/blob/main/infer/ref2va_ic/hyperflow.py),
[packed layout](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K-IC/blob/main/infer/ref2va_ic/layout.py).

## Integration approach

Prefer a native ComfyUI implementation for the final feature, reusing the model
profiles, learned latent upscaler, offload machinery and media finishing flow.
Add dedicated IC conditioning and HyperFlow patches rather than extending the
current `H3_REGENERATE_MODELS` mapping alone: membership currently implies the
old adapter stack, schedule and audio preparation.

For native post-generation refinement, retain the draft latent for reference
conditioning and decode the draft for Qwen vision context. Run the refinement
on Ref2VA independently of the draft model. The existing-video entry point can
instead encode/decode the supplied clip. Preserve the original soundtrack in
both cases. Check the upstream prompt encoder before deciding whether normal
reference-video conditioning provides identical vision context.

Affected integration points include `h3_models.py`, `h3_app/catalog.py`,
`h3_app/settings.py`, `h3_app/model_service.py`, `h3_app/refinement.py`,
`h3_app/workflows/h3.py` and `h3_app/workflows/regenerate.py`, plus custom nodes
and focused graph/settings/model-download tests. Downloads should remain lazy.

An isolated upstream runner is useful first as a numerical/visual baseline.
Its Diffusers and Transformers environment should remain separate from the
app's pinned runtime. The standard VAE is supported upstream; the fast VAE can
be evaluated later instead of being a prerequisite for correctness.

## Limits and validation before exposure

The supplied reader samples at 24 fps, takes at most 124 frames, refuses fewer
than 113 real frames, and pads the tail. It cover-resizes and center-crops onto
a fixed draft canvas. Thus directly invoking it would truncate longer videos,
reject shorter ones and change other aspect ratios. Initial support should
state these constraints; arbitrary duration needs a separately validated window
strategy. A batch of jobs is not temporal window stitching.

Start with a landscape draft of about five seconds, up3, CFG 1, and 2K output.
Compare the upstream baseline against native conditioning, time embeddings and
one-step outputs before comparing complete videos. Check faces, lip sync, text,
motion, seams, frame count and audio alignment. Measure peak VRAM and latency;
one-GPU code support does not establish practical fit on a particular GPU.
Then test both generation modes and the existing-video route. Keep acceleration
patch combinations experimental until tested with the IC execution path.

Sources: [media reader/writer](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K-IC/blob/main/infer/ref2va_ic/media.py),
[recipe configuration](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K-IC/blob/main/config.json),
[runtime requirements](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K-IC/blob/main/infer/requirements.txt).

Review evidence: small upstream source/config copies are in
`.cache/h3-regenerate-ic-review/`. These are inspection snapshots, not pinned
release artifacts. Pin revisions and verify weight mappings/checksums during
implementation. The model card also specifies MiniMax H3 Community License
terms and separate component licenses; adding a UI choice does not change them.
