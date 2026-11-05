"""Keep the source soundtrack clean through H3's latent-upscale refinement."""

import torch

import comfy.nested_tensor


class RemakeHoldAudio:
    """Mark a combined H3 latent as Remake's target: video denoised, audio held clean."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "Viggle Remake"

    def run(self, latent):
        video, audio = latent["samples"].unbind()
        return ({**latent, "noise_mask": comfy.nested_tensor.NestedTensor(
            (torch.ones_like(video), torch.zeros_like(audio))
        )},)


NODE_CLASS_MAPPINGS = {"RemakeHoldAudio": RemakeHoldAudio}
NODE_DISPLAY_NAME_MAPPINGS = {"RemakeHoldAudio": "Viggle Remake Hold Audio"}
