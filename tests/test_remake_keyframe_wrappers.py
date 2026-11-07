"""CPU regression for Remake's conditioning boundary around layout observers."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest


class Model:
    def __init__(self):
        # Like ModelPatcher: wrappers stay here until the sampler merges them into
        # transformer_options, which has no "wrappers" entry while nodes run.
        self.model_options = {"transformer_options": {}}
        self.wrappers = {}

    def clone(self):
        return copy.deepcopy(self)

    def add_wrapper_with_key(self, kind, key, callback):
        self.wrappers.setdefault(kind, {}).setdefault(key, []).append(callback)


class Timestep(float):
    def flatten(self):
        return [self]


class Latent:
    def __init__(self, *shape):
        self.shape = shape


class PackedLayout:
    # The native signature: Spectrum's rebuild fails here because it adds frame_count.
    def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframes=None, refs=None):
        self.signature = (text_len, latent_t, latent_h, latent_w, audio_t)
        self.keyframes, self.refs = keyframes, refs


class Executor:
    def __init__(self, call):
        self.call = call
        self.class_obj = SimpleNamespace(patch_size=(1, 2, 2))

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)


def execute(model, payload, sigma, *, forecast=False):
    wrappers = model.wrappers["diffusion_model"]
    callbacks = [callback for group in wrappers.values() for callback in group]

    def native(x, timestep, context, options, *, minimax_payload, **kwargs):
        return (minimax_payload, kwargs)

    def call(index, *args, **kwargs):
        if index == len(callbacks):
            return native(*args, **kwargs)
        return callbacks[index](Executor(lambda *a, **k: call(index + 1, *a, **k)), *args, **kwargs)

    x = [Latent(1, 24, 7, 45, 81), Latent(1, 32, 300)]  # Odd height and width pad up to the 2x2 patch.
    return call(0, x, Timestep(sigma * 1000), Latent(1, 64, 5120), {"forecast": forecast},
                minimax_payload=payload, denoise_mask="held audio")


class KeyframeWrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the node class alone; this wrapper needs no torch or running ComfyUI.
        path = Path(__file__).resolve().parents[1] / "custom_nodes/H3Remake/nodes.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        node = next(item for item in tree.body if isinstance(item, ast.ClassDef)
                    and item.name == "RemakeKeyframeUntil")
        namespace = {"comfy": SimpleNamespace(
            ldm=SimpleNamespace(minimax=SimpleNamespace(model=SimpleNamespace(PackedLayout=PackedLayout))),
            patcher_extension=SimpleNamespace(
                WrappersMP=SimpleNamespace(DIFFUSION_MODEL="diffusion_model")))}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        cls.node = namespace["RemakeKeyframeUntil"]()

    def test_observers_and_native_forward_share_layout_across_release(self):
        original = Model()
        observed = []

        def spectrum(executor, x, timestep, context, options, *, minimax_payload, **kwargs):
            observed.append(minimax_payload)
            if options["forecast"]:
                return minimax_payload, kwargs  # Forecast skips downstream wrappers.
            result = executor(x, timestep, context, options, minimax_payload=minimax_payload, **kwargs)
            self.assertIs(result[0], minimax_payload)
            return result

        original.add_wrapper_with_key("diffusion_model", "spectrum", spectrum)
        patched, = self.node.run(original, 0.9)
        refs = [{"latent": "source"}, {"latent": "picture", "audio_latent": "audio"}]
        payload = {"keyframes": ["pin"], "layout": "pinned layout", "refs": refs, "other": "keep"}
        for sigma in (1.0, 0.9, 0.89, 0.5, 1.0):
            for forecast in (False, True):
                with self.subTest(sigma=sigma, forecast=forecast):
                    effective, kwargs = execute(patched, payload, sigma, forecast=forecast)
                    self.assertIs(observed[-1], effective)
                    self.assertEqual(kwargs, {"denoise_mask": "held audio"})
                    self.assertEqual(effective["other"], "keep")
                    self.assertIs(effective["refs"], refs)
                    if sigma >= 0.9:
                        self.assertIs(effective, payload)
                    else:
                        self.assertIsNone(effective["keyframes"])
                        self.assertEqual(effective["layout"].signature, (64, 7, 46, 82, 300))
                        self.assertIsNone(effective["layout"].keyframes)
                        self.assertIs(effective["layout"].refs, refs)
                        self.assertEqual(effective["cond_video_latents"], ["source", "picture"])
                        self.assertEqual(effective["cond_audio_latents"], ["audio"])
        self.assertEqual(payload["keyframes"], ["pin"])
        self.assertEqual(payload["layout"], "pinned layout")
        self.assertEqual(list(original.wrappers["diffusion_model"]), ["spectrum"])

    def test_no_pins_and_reapplying_the_node(self):
        original = Model()
        once, = self.node.run(original, 0.9)
        twice, = self.node.run(once, 0.8)
        payload = {"refs": [], "layout": "unpinned"}
        self.assertIs(execute(twice, payload, 0.5)[0], payload)
        pinned = {"keyframes": ["pin"], "layout": "pinned", "refs": []}
        self.assertIs(execute(twice, pinned, 0.85)[0], pinned)
        self.assertIsNone(execute(twice, pinned, 0.79)[0]["keyframes"])


if __name__ == "__main__":
    unittest.main()
