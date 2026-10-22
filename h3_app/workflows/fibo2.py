"""Fibo 2.0 graphs on ComfyUI's native Fibo 2.0 nodes."""

import json

from h3_app.catalog import QWEN_EDIT_SIZE_MANUAL, QWEN_EDIT_SIZE_MATCH
from h3_app.graph import Graph
from h3_models import FIBO2_MODEL_CHOICES, FIBO2_TEXT_ENCODERS, MODEL_SPECS

FIBO2_MODELS = tuple(FIBO2_MODEL_CHOICES)
FIBO2_MAX_REFERENCES = 4
# Bria trained at ~1 MP; Max and the 2M presets stop at MOSS's 2M ceiling (1456 × 1456).
FIBO2_MAX_PIXELS = 1456 * 1456
FIBO2_EDIT_SIZE_TRAINED = "Trained size (~1 MP, recommended)"
FIBO2_EDIT_SIZE_MAX = "Max resolution (up to 2 MP)"
FIBO2_EDIT_SIZES = (FIBO2_EDIT_SIZE_TRAINED, QWEN_EDIT_SIZE_MATCH, FIBO2_EDIT_SIZE_MAX, QWEN_EDIT_SIZE_MANUAL)
# Bria's trained sizes for each aspect ratio, as ComfyUI's TextEncodeFibo2 computes them.
FIBO2_1M_RESOLUTION_PRESETS = (
    "1M · 1:1 · 1024×1024", "1M · 4:3 · 1152×864", "1M · 3:2 · 1248×832", "1M · 16:9 · 1424×800",
    "1M · 3:4 · 864×1152", "1M · 2:3 · 832×1248", "1M · 9:16 · 800×1424",
)
FIBO2_2M_RESOLUTION_PRESETS = (
    "2M · 1:1 · 1456×1456", "2M · 4:3 · 1664×1248", "2M · 3:2 · 1776×1184", "2M · 16:9 · 1920×1088",
    "2M · 3:4 · 1248×1664", "2M · 2:3 · 1184×1776", "2M · 9:16 · 1088×1920",
)
# Bria's text-to-image negative caption, which improves fine detail and image quality.
# Edits keep TextEncodeFibo2's empty default: the null caption Fibo 2.0 was trained with.
FIBO2_NEGATIVE_T2I = (
    '{"short_description":"","objects":[{"description":"the main subject",'
    '"texture":"smooth, featureless, plastic-like, airbrushed, no fine surface detail",'
    '"appearance_details":"soft blurred edges, simplified shapes, no fine detail",'
    '"skin_tone_and_texture":"waxy airbrushed skin, poreless, plastic smooth"}],'
    '"background_setting":"a plain seamless studio backdrop, perfectly smooth gradient, '
    'blurred and featureless, no visible texture","lighting":{},"aesthetics":{"artifacts":'
    '["oversaturated colours","neon colour cast","overexposed highlights","lens flare",'
    '"slight blur","motion blur","haze"]},"photographic_characteristics":'
    '{"depth_of_field":"shallow, heavy bokeh obscuring most of the frame",'
    '"focus":"soft, slightly out of focus, lacking fine texture"},'
    '"style_medium":"","context":"","artistic_style":""}'
)


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
                      width=1024, height=1024, seed=0, steps=30, cfg=5.0,
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
    negative = "" if reference_images else FIBO2_NEGATIVE_T2I
    encoded = g.add("TextEncodeFibo2", clip=clip, prompt=prompt, negative_prompt=negative, vae=vae, **images,
                    width=width, height=height, batch_size=1)
    samples = add("KSampler", model=model, positive=Graph.out(encoded, 0), negative=Graph.out(encoded, 1),
                  latent_image=Graph.out(encoded, 2), seed=seed, steps=steps, cfg=cfg,
                  sampler_name="euler", scheduler="simple", denoise=1.0)
    image = add("VAEDecode", samples=samples, vae=vae)
    g.add("SaveImage", images=image, filename_prefix=f"h3/image_staging/fibo2_{output_stamp}_{output_nonce}")
    return g.nodes
