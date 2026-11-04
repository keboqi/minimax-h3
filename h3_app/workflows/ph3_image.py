"""Pruna's native eight-step image workflow using an existing FL2VA base."""

from math import gcd

from h3_app.graph import Graph
from h3_models import MODEL_SPECS


PH3_IMAGE_KEYS = ("ph3_image_lora", "ph3_image_text_encoder", "ph3_image_vae", "ph3_image_upsampler")
PH3_IMAGE_SIGMAS = "1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0"


def build_ph3_image_graph(*, model_name, prompt, reference_images=(), width=2912,
                          height=2912, seed=0, output_stamp="0", output_nonce="ph3"):
    if len(reference_images) > 8:
        raise ValueError("P-H3-Image supports at most eight images.")
    g = Graph()

    def add(kind, **inputs):
        return Graph.out(g.add(kind, **inputs))

    def filename(key):
        return MODEL_SPECS[key].local_name

    model = add("UNETLoader", unet_name=model_name, weight_dtype="default")
    model = add("LoraLoaderBypassModelOnly", model=model,
                lora_name=filename("ph3_image_lora"), strength_model=1.0)
    model = add("MiniMaxH3SigmaShift", model=model, shift_video=1.0, shift_audio=1.0)
    clip = add("CLIPLoader", clip_name=filename("ph3_image_text_encoder"), type="minimax", device="default")
    vae = add("VAELoader", vae_name=filename("ph3_image_vae"))
    writer = add("CLIPLoader", clip_name=filename("ph3_image_upsampler"), type="minimax", device="default")
    images = [add("LoadImage", image=path) for path in reference_images]
    writer_inputs = {}
    if images:
        # Match the official edit workflow: one 1 MP image, then references
        # padded to that image's size, in the same order as conditioning.
        first = add("ImageScaleToTotalPixels", image=images[0], upscale_method="lanczos",
                    megapixels=1.0, resolution_steps=1)
        size = g.add("GetImageSize", image=first)
        batch = [first] + [add("ResizeAndPadImage", image=image,
                               target_width=Graph.out(size), target_height=Graph.out(size, 1),
                               padding_color="white", interpolation="lanczos") for image in images[1:]]
        writer_inputs["image"] = add("BatchImagesNode", **{
            f"images.image{i}": image for i, image in enumerate(batch)
        })
        pictures = "".join(f"<Picture {i}>: <|vision_start|><|image_pad|><|vision_end|>"
                           for i in range(1, len(images) + 1))
        writer_prompt = (f"<|im_start|>user\n{pictures}Edit <Picture 1>. {prompt}"
                         "<|im_end|>\n<|im_start|>assistant\n<think>\n")
    else:
        divisor = gcd(width, height)
        writer_prompt = f"Aspect ratio: {width // divisor}:{height // divisor}\nRequest: {prompt}"
    expanded = add("TextGenerate", clip=writer, prompt=writer_prompt,
                   max_length=3072 if images else 2048, sampling_mode="off",
                   thinking=not images, use_default_template=True, mtp="auto", **writer_inputs)
    conditioned = g.add("PH3ImageConditioning", clip=clip, vae=vae, prompt=expanded,
                        width=width, height=height, **{
                            f"images.image_{i}": image for i, image in enumerate(images, 1)
                        })
    guider = add("BasicGuider", model=model, conditioning=Graph.out(conditioned))
    sampled = add("SamplerCustomAdvanced", noise=add("RandomNoise", noise_seed=seed),
                  guider=guider, sampler=add("KSamplerSelect", sampler_name="euler"),
                  sigmas=add("ManualSigmas", sigmas=PH3_IMAGE_SIGMAS),
                  latent_image=Graph.out(conditioned, 1))
    decoded = add("VAEDecode", samples=sampled, vae=vae)
    add("SaveImage", images=decoded,
        filename_prefix=f"h3/image_staging/ph3_{output_stamp}_{output_nonce}")
    return g.nodes
