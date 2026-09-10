"""Pure selection of active decoders, separate from saved preferences."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DecoderResources:
    video_decoder: str
    image_decoder: bool

    @property
    def use_trt_vae(self) -> bool:
        return self.video_decoder == "tensorrt"

    @property
    def use_int8_vae(self) -> bool:
        return self.video_decoder == "int8"

    @property
    def optional_model_keys(self) -> tuple[str, ...]:
        if self.image_decoder:
            return ("image_vae_500k",)
        if self.use_trt_vae:
            return ("video_vae_trt_decoder", "video_vae_trt_decoder_data")
        return ("video_vae_int8",) if self.use_int8_vae else ()


def resolve_decoders(
    result_format: str, image_vae: str, *, use_trt_vae: bool, use_int8_vae: bool
) -> DecoderResources:
    fmt = str(result_format).title()
    image_decoder = fmt == "Image" and "500K" in str(image_vae)
    if fmt == "Audio" or image_decoder:
        return DecoderResources("none", image_decoder)
    if use_trt_vae and use_int8_vae:
        raise ValueError("Select either INT8 ConvRot VAE or TensorRT VAE, not both.")
    return DecoderResources(
        "tensorrt" if use_trt_vae else "int8" if use_int8_vae else "official", False
    )
