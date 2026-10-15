# Qwen Image 2.1 Lightning edit drift investigation

Checked 2026-10-15. The user observes a leftward shift with LightX2V Lightning
in Max resolution mode; the base model and other Turbo adapters do not show it
under the same settings. This makes Lightning-specific behavior the leading
explanation. It does not establish the underlying cause.

## Upstream status

- [Lightning discussion 3](https://huggingface.co/lightx2v/Qwen-Image-2.1-Lightning/discussions/3)
  reports the same leftward movement. The public discussion API returns an open
  thread with one initial comment and no replies or resolution.
- The repository is still at `6b0a1176327c01522a44276c5da67331a025b6fc`.
  The adapter's SHA-256 remains
  `53c69320d4163da51032f90275074028cb098556039c72b45f8dce9f0f9e34a3`,
  identical to our configured download. There is no replacement weight to adopt.
- In [discussion 2](https://huggingface.co/lightx2v/Qwen-Image-2.1-Lightning/discussions/2),
  users report better results around strength 0.5, including an eight-step run.
  Another user reports offset problems and says strength 1 is needed for text.
  These are informal reports, not a verified fix for discussion 3. The author
  confirms that training uses a shifted schedule and updated the model card.
- [ComfyUI issue 16496](https://github.com/Comfy-Org/ComfyUI/issues/16496)
  reported a similar symptom earlier. Its author subsequently attributed it to
  the old Viggle four-step LoRA. Closing that issue was not a ComfyUI code fix.

## Checks against our implementation

Our Lightning path already uses the updated training schedule: raw nodes
1/.75/.5/.25, dynamic exponential shifting with 256/8192 token anchors and
0.5/0.9 shifts, no terminal stretch, Euler, CFG 1, strength 1, and edit KV cache
off. Changing the scheduler dropdown does not affect this path.

The public pinned ComfyUI loader recognizes the adapter's attention tensor names
and `.lora.down.weight` / `.lora.up.weight` format. Its rank and alpha are both
16, so no extra alpha/rank correction is indicated by the header.

There is a size-related factor worth isolating. Max resolution makes a separate
output canvas near 4 MP but retains the chosen reference resolution. With a
1024-square source and reference resolution 0, a generated graph encodes the
1024-square reference and samples a 1984-square output. The native ComfyUI
encoder explicitly warns that a different target latent size can shift edits.
This difference exists for the other models too; it cannot by itself explain
the user's Lightning-only comparison. A Lightning-specific sensitivity to that
difference remains a hypothesis.

ComfyUI's reference-grid parity adjustment differs from the integer positions
in LightX2V/Diffusers, but our usual references and targets are multiples of
32 pixels, giving even native 16x latent dimensions. The adjustment is zero
there, so it does not explain a general left shift in those cases.

## Available mitigations

1. **Use the existing official eight-step Turbo for affected edits.** The user
   already has an effective control with the same settings. This is the
   immediately available fallback in our app.
2. **Test lower Lightning strength as an experiment.** Compare strength 1 and
   0.5 at four steps first; test an eight-step schedule separately so strength
   and step-count effects are not confused. Four steps remains the author's
   trained setup. Our current Lightning option intentionally fixes strength
   and steps, so the community experiment needs a diagnostic workflow or an
   explicit experimental option. Do not present it as a proven correction.
3. **Correct global drift after generation.** AusBoss's experimental
   [Realign to Source](https://github.com/ausboss/ComfyUI-AusBoss#realign-to-source-experimental-)
   estimates zoom/translation, with an affine mode for tilt. It cannot recover
   cropped-away content or independently displaced objects. Padding before
   editing gives it border room; confidence reporting and the empty-border
   mask are necessary. This is compensation, not a repaired Lightning model.
4. **Consistency LoRA is a separate candidate.** The
   [author's model card](https://huggingface.co/ausboss/Qwen-Image-2.1-Consistency-LoRA)
   reports better alignment on base and official eight-step Turbo, but explicitly
   says four-step adapters are untested. It can suppress requested movement and
   leave ghosts after large removals. Stacking it with Lightning needs testing.

## Controlled test needed to identify the best fix

Use one failing source/prompt at a fixed seed, then repeat with several seeds:

- Current Lightning Max resolution baseline.
- Same output dimensions, but resize the first reference to those exact
  dimensions before both VLM and VAE encoding; keep auxiliary references
  unchanged. This isolates reference/output size mismatch while keeping the
  output resolution and Lightning schedule fixed.
- Same aligned reference/output with strength 0.5, four steps.
- Official eight-step Turbo control using the same source and output size.

Measure displacement on unchanged background features rather than an object
the prompt intentionally moves. Record horizontal/vertical shift, scale,
alignment confidence, edit fidelity and text quality. A single attractive
image is insufficient evidence that geometric drift is fixed.

This investigation inspected upstream discussions, model metadata, inference
sources and a generated Max-resolution graph. It did not reproduce the failure
on a GPU, modify production behavior, push changes or redeploy.
