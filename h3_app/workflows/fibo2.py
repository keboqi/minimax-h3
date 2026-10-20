"""Fibo 2.0 graphs through the isolated Diffusers ComfyUI bridge."""

import json

from h3_app.graph import Graph

FIBO2_MODELS = ("Dev", "Edit Turbo")
FIBO2_MAX_REFERENCES = 4


def validate_fibo2_inputs(model_choice, prompt, reference_images=(), mask_image=None, steps=50, cfg=5.0):
    if model_choice not in FIBO2_MODELS:
        raise ValueError("Choose Fibo 2.0 Dev or Edit Turbo.")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Write an image prompt, VGL JSON caption or edit instruction.")
    if prompt.lstrip().startswith(("{", "[")):
        try:
            caption = json.loads(prompt)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid Fibo VGL JSON: {exc.msg}") from exc
        if not isinstance(caption, dict) or not caption:
            raise ValueError("Fibo VGL must be a non-empty JSON object.")
    if len(reference_images) > FIBO2_MAX_REFERENCES:
        raise ValueError("Fibo supports up to four reference images in this integration.")
    if mask_image and len(reference_images) != 1:
        raise ValueError("A Fibo mask requires exactly one reference image.")
    if model_choice == "Edit Turbo":
        if not reference_images:
            raise ValueError("Fibo Edit Turbo requires a reference image; use Dev for text-to-image.")
        if steps != 4 or cfg != 1.0:
            raise ValueError("Fibo Edit Turbo requires 4 steps and CFG 1.")


def build_fibo2_graph(*, prompt, model_choice="Dev", reference_images=(), mask_image=None,
                      width=1024, height=1024, seed=0, steps=50, cfg=5.0,
                      output_stamp="0", output_nonce="fibo"):
    validate_fibo2_inputs(model_choice, prompt, reference_images, mask_image, steps, cfg)
    graph = Graph()
    inputs = dict(prompt=prompt, model_choice=model_choice, width=width, height=height,
                  seed=seed, steps=steps, cfg=cfg)
    for index, path in enumerate(reference_images, 1):
        inputs[f"image_{index}"] = Graph.out(graph.add("LoadImage", image=path))
    if mask_image:
        # Load the mask's RGB values, not LoadImage's inverted alpha channel.
        inputs["mask_image"] = Graph.out(graph.add("LoadImage", image=mask_image))
    image = Graph.out(graph.add("H3Fibo2Generate", **inputs))
    graph.add("SaveImage", images=image,
              filename_prefix=f"h3/image_staging/fibo2_{output_stamp}_{output_nonce}")
    return graph.nodes
