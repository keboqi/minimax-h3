"""Isolated Fibo inference. Invoked only by the bundled ComfyUI bridge."""

import json
import os
import sys
from pathlib import Path

MODEL_REPOS = {"Dev": "briaai/Fibo-2.0-dev", "Edit Turbo": "briaai/Fibo-2.0-edit-turbo"}


def run(request, *, on_step):
    import torch
    from diffusers import BriaFibo2Pipeline
    from huggingface_hub import get_token
    from PIL import Image, ImageOps

    model = request["model_choice"]
    if model not in MODEL_REPOS:
        raise ValueError("Unknown Fibo model.")
    paths = request.get("references", [])
    if len(paths) > 4:
        raise ValueError("Use at most four Fibo reference images.")
    if model == "Edit Turbo" and (not paths or request["steps"] != 4 or request["cfg"] != 1):
        raise ValueError("Edit Turbo requires reference images, 4 steps and CFG 1.")
    if request.get("mask") and len(paths) != 1:
        raise ValueError("A mask requires exactly one reference image.")
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(ImageOps.exif_transpose(image).convert("RGB"))
    mask = None
    if request.get("mask"):
        with Image.open(request["mask"]) as source:
            mask = ImageOps.exif_transpose(source).convert("L")
        if mask.size != images[0].size:
            raise ValueError("The mask must have the same dimensions as the reference image.")
    if not torch.cuda.is_available():
        raise RuntimeError("Fibo 2.0 requires a CUDA GPU in this integration.")
    try:
        pipe = BriaFibo2Pipeline.from_pretrained(
            MODEL_REPOS[model], dtype=torch.bfloat16, cache_dir=request["cache_dir"],
            token=os.environ.get("HF_TOKEN") or get_token(),
        ).to("cuda")
    except OSError as exc:
        raise RuntimeError(
            f"Could not load {MODEL_REPOS[model]}. Accept its Hugging Face access conditions and "
            "configure HF_TOKEN (or hf auth login) for the server. " + str(exc)
        ) from exc

    def callback(_pipe, step, _timestep, kwargs):
        on_step(step + 1)
        return kwargs

    result = pipe(
        prompt=request["prompt"], image=images or None, mask=mask,
        width=request["width"], height=request["height"],
        num_inference_steps=request["steps"], guidance_scale=request["cfg"],
        generator=torch.Generator(device="cuda").manual_seed(request["seed"]),
        callback_on_step_end=callback,
    )
    result.images[0].save(request["output"])


def main():
    path = Path(sys.argv[1])
    progress = path.with_suffix(".progress")

    def on_step(step):
        pending = progress.with_suffix(".tmp")
        pending.write_text(str(step), encoding="utf-8")
        try:
            pending.replace(progress)
        except PermissionError:
            # Windows refuses the rename while the bridge reads; the next step reports again.
            pass

    run(json.loads(path.read_text(encoding="utf-8")), on_step=on_step)


if __name__ == "__main__":
    main()
