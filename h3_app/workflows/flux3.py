"""FLUX 3 distilled graphs for ComfyUI v0.41.0."""

from pathlib import Path

from h3_app.graph import Graph
from h3_models import FLUX3_MODEL_CHOICES, MODEL_SPECS


# top_k=1 decodes greedily, so the seed cannot change the caption. A constant
# seed keeps the upsampler inputs identical across image-seed rerolls, letting
# ComfyUI reuse the cached expansion instead of reloading the 9B upsampler.
UPSAMPLER_SEED = 0
# BFL's official text-to-image upsampler instructions, shared with the remote writers.
UPSAMPLER_SYSTEM_PATH = Path(__file__).resolve().parents[2] / "prompt_flux3_t2i.txt"
UPSAMPLER_MAX_LENGTH = 3584


def upsampler_system(path=UPSAMPLER_SYSTEM_PATH):
    return Path(path).read_text(encoding="utf-8").strip()


def flux3_model_keys(model_choice, expand_prompt):
    if model_choice not in FLUX3_MODEL_CHOICES:
        raise ValueError("Unknown FLUX 3 model choice.")
    return (FLUX3_MODEL_CHOICES[model_choice], "flux3_text_encoder", "flux3_vae",
            *(("flux3_upsampler",) if expand_prompt else ()))


def build_flux3_graph(*, model_choice, prompt, reference_images=(), width=1024,
                      height=1024, seed=0, steps=16, expand_prompt=True,
                      output_stamp="0", output_nonce="flux3", system_prompt=None):
    flux3_model_keys(model_choice, expand_prompt)
    if reference_images and expand_prompt:
        raise ValueError("For editing, disable prompt expansion and supply a complete scene caption with the edit included.")
    g = Graph()
    def add(kind, **inputs):
        return Graph.out(g.add(kind, **inputs))
    model = add("UNETLoader", unet_name=MODEL_SPECS[FLUX3_MODEL_CHOICES[model_choice]].local_name, weight_dtype="default")
    clip = add("CLIPLoader", clip_name=MODEL_SPECS["flux3_text_encoder"].local_name, type="flux3", device="default")
    vae = add("VAELoader", vae_name=MODEL_SPECS["flux3_vae"].local_name)
    if expand_prompt:
        writer = add("CLIPLoader", clip_name=MODEL_SPECS["flux3_upsampler"].local_name, type="flux3", device="default")
        prompt = add("TextGenerate", clip=writer, prompt=prompt,
                     system_prompt=system_prompt or upsampler_system(),
                     max_length=UPSAMPLER_MAX_LENGTH, sampling_mode="on", thinking=False,
                     use_default_template=True, mtp="auto", **{
                         "sampling_mode.temperature": 1.0, "sampling_mode.top_k": 1,
                         "sampling_mode.top_p": 0.95, "sampling_mode.min_p": 0.0,
                         "sampling_mode.repetition_penalty": 1.05,
                         "sampling_mode.presence_penalty": 1.5, "sampling_mode.seed": UPSAMPLER_SEED,
                     })
    positive = add("CLIPTextEncode", clip=clip, text=prompt)
    negative = add("ConditioningZeroOut", conditioning=positive)
    latent = add("EmptyFlux2LatentImage", width=width, height=height, batch_size=1)
    if reference_images:
        images = {f"images.image{i}": add("LoadImage", image=path)
                  for i, path in enumerate(reference_images)}
        refs = g.add("ReferenceMultiLatent", positive=positive, negative=negative, vae=vae,
                     resolution="resize", **{"resolution.megapixels": 1.0}, **images)
        positive, negative = Graph.out(refs, 0), Graph.out(refs, 1)
        # Keep the explicit output canvas; references condition, rather than set, its size.
    samples = add("KSampler", model=model, positive=positive, negative=negative,
                  latent_image=latent, seed=seed, steps=steps, cfg=1.0,
                  sampler_name="euler", scheduler="simple", denoise=1.0)
    image = add("VAEDecode", samples=samples, vae=vae)
    add("SaveImage", images=image, filename_prefix=f"h3/image_staging/flux3_{output_stamp}_{output_nonce}")
    return g.nodes
