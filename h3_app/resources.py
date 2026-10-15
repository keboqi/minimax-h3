"""Pure selection of active decoders, separate from saved preferences."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DecoderResources:
    video_decoder: str
    image_decoder: bool
    video_encoder: str = "Original"

    @property
    def use_trt_vae(self) -> bool:
        return self.video_decoder == "tensorrt"

    @property
    def use_int8_vae(self) -> bool:
        return self.video_decoder == "int8"

    @property
    def use_lynnreal_vae(self) -> bool:
        return self.video_decoder == "lynnreal_int8"

    @property
    def optional_model_keys(self) -> tuple[str, ...]:
        return self.decoder_model_keys + (("lightvae_encoder",) if self.video_encoder == "LightVAE Lite" else ())

    @property
    def decoder_model_keys(self) -> tuple[str, ...]:
        if self.video_decoder in {"lightvae_lite", "lightvae_int8"}:
            return ("lightvae_decoder_int8" if self.video_decoder == "lightvae_int8" else "lightvae_decoder",)
        if self.image_decoder:
            return ("image_vae_500k",)
        if self.use_trt_vae:
            return ("video_vae_trt_decoder", "video_vae_trt_decoder_data")
        if self.use_lynnreal_vae:
            return ("video_vae_lynnreal_int8",)
        return ("video_vae_int8",) if self.use_int8_vae else ()


def resolve_decoders(
    result_format: str,
    image_vae: str,
    *,
    use_trt_vae: bool,
    use_int8_vae: bool,
    use_lynnreal_vae: bool = False,
    video_encoder: str = "Original",
    lightvae_decoder: str = "none",
) -> DecoderResources:
    if video_encoder not in {"Original", "LightVAE Lite"}:
        raise ValueError("Unsupported video encoder.")
    if lightvae_decoder not in {"none", "lite", "int8"}:
        raise ValueError("Unsupported LightVAE decoder.")
    fmt = str(result_format).title()
    image_decoder = fmt == "Image" and "500K" in str(image_vae)
    if fmt == "Audio" or image_decoder:
        return DecoderResources("none", image_decoder, video_encoder)
    if sum((use_trt_vae, use_int8_vae, use_lynnreal_vae, lightvae_decoder != "none")) > 1:
        raise ValueError("Select only one video decoder.")
    decoder = (
        "lightvae_" + lightvae_decoder if lightvae_decoder != "none"
        else "tensorrt" if use_trt_vae
        else "lynnreal_int8" if use_lynnreal_vae
        else "int8" if use_int8_vae
        else "official"
    )
    return DecoderResources(decoder, False, video_encoder)
