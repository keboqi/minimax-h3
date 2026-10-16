"""Graph contract for ComfyUI-Drama3 revision 66c4fb9."""

from h3_app.graph import Graph
from h3_models import MODEL_SPECS


def build_drama3_graph(*, text, reference_audio=None, transcript="", seed=0,
                       temperature=1.0, top_p=0.9, top_k=30, max_frames=600,
                       split_long_text=True, experts="in RAM", expert_cache_gb=0.0):
    g = Graph()
    def add(kind, **inputs):
        return Graph.out(g.add(kind, **inputs))
    model = add("Drama3Loader", model_name=MODEL_SPECS["drama3_fp8"].local_name,
                experts=experts, expert_cache_gb=expert_cache_gb, second_gpu="none", split_layer=0)
    codec = add("Drama3CodecLoader", codec_name=MODEL_SPECS["drama3_codec"].local_name)
    voices = {}
    if reference_audio:
        audio = add("LoadAudio", audio=reference_audio)
        voices["voices.voice0"] = add("Drama3EncodeVoice", codec=codec, audio=audio, transcript=transcript)
    codes = add("Drama3Generate", model=model, text=text, seed=seed,
                temperature=temperature, top_p=top_p, top_k=top_k,
                max_frames=max_frames, split_long_text=split_long_text, **voices)
    audio = add("Drama3Decode", codec=codec, codes=codes)
    add("SaveAudio", audio=audio, filename_prefix="drama3/Drama3")
    return g.nodes
