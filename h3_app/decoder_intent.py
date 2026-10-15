"""Video decoder intent at the presentation/API compatibility boundary."""

from enum import StrEnum


class VideoDecoder(StrEnum):
    STANDARD = "Original FP16"
    OFFICIAL_INT8 = "Official INT8"
    LYNNREAL_INT8 = "LynnReal INT8"
    TENSORRT = "TensorRT"
    LIGHTVAE = "LightVAE Lite"
    LIGHTVAE_INT8 = "LightVAE Lite INT8 ConvRot"

    def workflow_flags(self) -> dict[str, bool | str]:
        return {
            "use_int8_vae": self == self.OFFICIAL_INT8,
            "use_lynnreal_vae": self == self.LYNNREAL_INT8,
            "use_trt_vae": self == self.TENSORRT,
            "lightvae_decoder": "lite" if self == self.LIGHTVAE else "int8" if self == self.LIGHTVAE_INT8 else "none",
        }

    @classmethod
    def from_flags(cls, official=False, lynnreal=False, trt=False, lightvae="none"):
        if lightvae not in {"none", "lite", "int8"}:
            raise ValueError("Unsupported LightVAE decoder.")
        if sum((bool(official), bool(lynnreal), bool(trt), lightvae != "none")) > 1:
            raise ValueError("Select only one video decoder.")
        if lightvae != "none":
            return cls.LIGHTVAE if lightvae == "lite" else cls.LIGHTVAE_INT8
        if trt:
            return cls.TENSORRT
        if lynnreal:
            return cls.LYNNREAL_INT8
        if official:
            return cls.OFFICIAL_INT8
        return cls.STANDARD
