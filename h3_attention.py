#!/usr/bin/env python3
"""Runtime attention-backend detection for MiniMax H3.

Native ComfyUI SageAttention is deliberately used only as the global dense
backend. MiniMax H3 Sol-Attn remains the per-model sparse override, so Sol's
forced-dense and fallback calls naturally land on SageAttention.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class SageProbe:
    available: bool
    message: str


def probe_sageattention() -> SageProbe:
    try:
        import torch
    except Exception as exc:
        return SageProbe(False, f"torch import failed: {exc}")

    if not torch.cuda.is_available():
        return SageProbe(False, "CUDA is unavailable")

    try:
        from sageattention import sageattn
    except Exception as exc:
        return SageProbe(False, f"sageattention import failed: {exc}")

    try:
        device = torch.device("cuda")
        capability = torch.cuda.get_device_capability(device)
        name = torch.cuda.get_device_name(device)

        # H3's attention head_dim is 128. A small BF16 HND probe catches
        # architecture/build incompatibilities before ComfyUI is launched.
        with torch.inference_mode():
            q = torch.randn(
                (1, 4, 256, 128),
                device=device,
                dtype=torch.bfloat16,
            )
            k = torch.randn_like(q)
            v = torch.randn_like(q)
            output = sageattn(
                q,
                k,
                v,
                tensor_layout="HND",
                is_causal=False,
            )
            torch.cuda.synchronize()

        if output.shape != q.shape:
            return SageProbe(
                False,
                f"unexpected output shape {tuple(output.shape)}",
            )
        if not bool(torch.isfinite(output).all().item()):
            return SageProbe(False, "probe output contains non-finite values")

        return SageProbe(
            True,
            f"{name} sm_{capability[0]}{capability[1]} SageAttention probe OK",
        )
    except Exception as exc:
        return SageProbe(
            False,
            f"SageAttention CUDA probe failed: {exc.__class__.__name__}: {exc}",
        )


def selftest() -> None:
    ok = SageProbe(True, "ok")
    bad = SageProbe(False, "bad")
    assert ok.available is True
    assert bad.available is False
    print("h3_attention selftest OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return

    result = probe_sageattention()
    prefix = "[h3-attention]"
    print(
        f"{prefix} {'Sage enabled' if result.available else 'Sage unavailable'}: "
        f"{result.message}",
        flush=True,
    )
    raise SystemExit(0 if result.available else 1)


if __name__ == "__main__":
    main()
