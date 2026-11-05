"""Start a Remake window from the end of the render before it, as Viggle's hops do."""

import torch

import comfy.nested_tensor

from .nodes import lanczos

CHUNK_FRAMES, CHUNK_LATENTS, TAIL_FRAMES = 17, 5, 5
SPATIAL_SCALE = 16


class RemakeHoldFrames:
    """Make these frames the target's first frames: encoded clean, and never sampled."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent": ("LATENT",),
            "vae": ("VAE",),
            "frames": ("IMAGE", {"tooltip": "Frames of the previous render, a whole number of 17-frame chunks."}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "Viggle Remake"

    def run(self, latent, vae, frames):
        video, audio = latent["samples"].unbind()
        video_mask, audio_mask = latent["noise_mask"].unbind()
        count = frames.shape[0]
        steps = count // CHUNK_FRAMES * CHUNK_LATENTS
        assert count % CHUNK_FRAMES == 0 and 0 < steps < video.shape[2], f"can't hold {count} frames"
        height, width = video.shape[3] * SPATIAL_SCALE, video.shape[4] * SPATIAL_SCALE
        pixels = frames[..., :3]
        if pixels.shape[1:3] != (height, width):
            pixels = lanczos(pixels, width, height)
        # A clip's last latents are special: encode 17m + 5 frames and keep the first 5m latents.
        pixels = torch.cat((pixels, pixels[-1:].repeat(TAIL_FRAMES, 1, 1, 1)))
        held = vae.encode(pixels).to(video)
        video, video_mask = video.clone(), video_mask.clone()
        video[:, :, :steps] = held[:, :, :steps]
        video_mask[:, :, :steps] = 0.0
        return ({
            **latent,
            "samples": comfy.nested_tensor.NestedTensor((video, audio)),
            "noise_mask": comfy.nested_tensor.NestedTensor((video_mask, audio_mask)),
        },)


NODE_CLASS_MAPPINGS = {"RemakeHoldFrames": RemakeHoldFrames}
NODE_DISPLAY_NAME_MAPPINGS = {"RemakeHoldFrames": "Viggle Remake Hold Frames"}
