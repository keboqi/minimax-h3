"""Official MiniMax H3 Image on native ComfyUI image/PDD support."""

from h3_app.graph import Graph
from h3_models import MODEL_SPECS

from h3_app.h3_image import (
    H3_IMAGE_MAX_REFERENCES, h3_image_prompt_path, h3_image_canvas_description,
)

H3_IMAGE_KEYS = ("h3_image_dit", "text_encoder_int8", "h3_image_vae")


def build_h3_image_graph(*, prompt, reference_images=(), width=4096, height=4096, seed=0,
                         expand_prompt=True, system_prompt=None, output_stamp="0", output_nonce="h3image"):
    if len(reference_images) > H3_IMAGE_MAX_REFERENCES:
        raise ValueError("MiniMax H3 Image supports at most sixteen images.")
    g = Graph()

    def add(kind, **inputs):
        return Graph.out(g.add(kind, **inputs))

    model = add("UNETLoader", unet_name=MODEL_SPECS["h3_image_dit"].local_name, weight_dtype="default")
    clip = add("CLIPLoader", clip_name=MODEL_SPECS["text_encoder_int8"].local_name,
               type="minimax", device="default")
    vae = add("VAELoader", vae_name=MODEL_SPECS["h3_image_vae"].local_name)
    images = [add("LoadImage", image=path) for path in reference_images]
    if expand_prompt:
        writer = add("CLIPLoader", clip_name=MODEL_SPECS["h3_image_upsampler"].local_name,
                     type="minimax", device="default")
        writer_inputs = {}
        pictures = ""
        if images:
            writer_inputs = {
                f"images.image{i}": add("ImageScaleToTotalPixels", image=image, upscale_method="lanczos",
                                        megapixels=1.0, resolution_steps=1)
                for i, image in enumerate(images, 1)
            }
            pictures = "".join(f"<Picture {i}>: <|vision_start|><|image_pad|><|vision_end|>"
                               for i in range(1, len(images) + 1))
        instructions = (system_prompt if system_prompt is not None
                        else h3_image_prompt_path(bool(images)).read_text(encoding="utf-8").strip())
        context = h3_image_canvas_description(width, height) + "\n"
        prompt = add("TextGenerateV2", clip=writer,
                     prompt=f"<|im_start|>system\n{instructions}<|im_end|>\n"
                            f"<|im_start|>user\n{context}{pictures}\nUser request:\n{prompt}<|im_end|>\n"
                            "<|im_start|>assistant\n<think>\n\n</think>\n",
                     max_length=4096, sampling_mode="off", thinking=False,
                     use_default_template=False, mtp="auto", **writer_inputs)
    conditioned = g.add("MiniMaxH3ImageConditioning", clip=clip, vae=vae, prompt=prompt,
                        width=width, height=height, **{
                            f"images.image_{i}": image for i, image in enumerate(images, 1)})
    positive = Graph.out(conditioned)
    sampled = add("KSampler", model=model, positive=positive, negative=positive,
                  latent_image=Graph.out(conditioned, 1), seed=seed, steps=6, cfg=1.0,
                  sampler_name="euler", scheduler="simple", denoise=1.0)
    add("SaveImage", images=add("VAEDecode", samples=sampled, vae=vae),
        filename_prefix=f"h3/image_staging/h3image_{output_stamp}_{output_nonce}")
    return g.nodes
