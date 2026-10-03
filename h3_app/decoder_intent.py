"""Video decoder intent at the presentation/API compatibility boundary."""

from enum import StrEnum


class VideoDecoder(StrEnum):
    STANDARD = "Standard"
    OFFICIAL_INT8 = "Official INT8"
    LYNNREAL_INT8 = "LynnReal INT8"
    TENSORRT = "TensorRT"

    def workflow_flags(self) -> dict[str, bool]:
        return {
            "use_int8_vae": self == self.OFFICIAL_INT8,
            "use_lynnreal_vae": self == self.LYNNREAL_INT8,
            "use_trt_vae": self == self.TENSORRT,
        }

    @classmethod
    def from_flags(cls, official=False, lynnreal=False, trt=False):
        if sum((bool(official), bool(lynnreal), bool(trt))) > 1:
            raise ValueError("Select only one video decoder.")
        if trt:
            return cls.TENSORRT
        if lynnreal:
            return cls.LYNNREAL_INT8
        if official:
            return cls.OFFICIAL_INT8
        return cls.STANDARD
