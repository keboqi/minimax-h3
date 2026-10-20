"""Fibo 2.0 graphs on ComfyUI's native Fibo 2.0 nodes."""

import json

from h3_app.graph import Graph
from h3_models import FIBO2_MODEL_CHOICES, FIBO2_TEXT_ENCODERS, MODEL_SPECS

FIBO2_MODELS = tuple(FIBO2_MODEL_CHOICES)
FIBO2_MAX_REFERENCES = 4


def validate_fibo2_inputs(model_choice, prompt, reference_images=(), steps=30, cfg=5.0, precision=None):
    if model_choice not in FIBO2_MODELS:
        raise ValueError("Choose Fibo 2.0 Dev or Edit Turbo.")
    if precision is not None and precision not in FIBO2_MODEL_CHOICES[model_choice]:
        raise ValueError("Choose a valid Fibo 2.0 precision.")
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
    if model_choice == "Edit Turbo":
        if not reference_images:
            raise ValueError("Fibo Edit Turbo requires a reference image; use Dev for text-to-image.")
        if steps != 4 or cfg != 1.0:
            raise ValueError("Fibo Edit Turbo requires 4 steps and CFG 1.")


def fibo2_model_keys(model_choice, precision, editing):
    return (FIBO2_MODEL_CHOICES[model_choice][precision],
            FIBO2_TEXT_ENCODERS["edit" if editing else "generate"], "flux3_vae")


def build_fibo2_graph(*, prompt, model_choice="Dev", precision="BF16", reference_images=(),
                      width=1024, height=1024, auto_size=True, seed=0, steps=30, cfg=5.0,
                      output_stamp="0", output_nonce="fibo"):
    validate_fibo2_inputs(model_choice, prompt, reference_images, steps, cfg, precision)
    unet, text_encoder, vae_key = fibo2_model_keys(model_choice, precision, bool(reference_images))
    g = Graph()

    def add(kind, **inputs):
        return Graph.out(g.add(kind, **inputs))

    model = add("UNETLoader", unet_name=MODEL_SPECS[unet].local_name, weight_dtype="default")
    clip = add("CLIPLoader", clip_name=MODEL_SPECS[text_encoder].local_name, type="fibo2", device="default")
    vae = add("VAELoader", vae_name=MODEL_SPECS[vae_key].local_name)
    images = {f"images.image_{index}": add("LoadImage", image=path)
              for index, path in enumerate(reference_images, 1)}
    # 0 x 0 lets the node pick Bria's trained ~1 MP size for the first reference's aspect ratio.
    auto = auto_size and bool(reference_images)
    encoded = g.add("TextEncodeFibo2", clip=clip, prompt=prompt, negative_prompt="", vae=vae, **images,
                    width=0 if auto else width, height=0 if auto else height, batch_size=1)
    samples = add("KSampler", model=model, positive=Graph.out(encoded, 0), negative=Graph.out(encoded, 1),
                  latent_image=Graph.out(encoded, 2), seed=seed, steps=steps, cfg=cfg,
                  sampler_name="euler", scheduler="simple", denoise=1.0)
    image = add("VAEDecode", samples=samples, vae=vae)
    g.add("SaveImage", images=image, filename_prefix=f"h3/image_staging/fibo2_{output_stamp}_{output_nonce}")
    return g.nodes
