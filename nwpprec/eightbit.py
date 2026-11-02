"""8-bit quantisation of the network, simulated inside the fp16 network (state
in fp32 as always): values are rounded to 8 bits with a scale and converted
back to fp16 before they are used, which reproduces the accuracy of 8-bit
inference but not its speed or memory.

Configurations:
    w-int8, w-fp8      weight-only: every matrix-product weight rounded to
                       int8 (symmetric, 127 levels) or fp8 (e4m3), with one
                       scale per output channel
    w8a8-int8          also the inputs of every Linear and convolution layer,
                       int8 with one scale per token (row)
    w8a8-fp8           also those inputs, fp8 (e4m3) with one scale per tensor

Matrix-product weights: Linear and convolution layers (and Aurora's linear
patch reconstruction), FourCastNet's spectral filter weights and SFNO's
complex spectral filters (real and imaginary parts, one scale per output
channel and wavenumber). Biases, normalisation, embeddings and Pangu-Weather's
position-dependent attention bias stay in fp16, as is usual. Activations
entering FourCastNet's and SFNO's spectral filters stay in fp16.

Forecasts of the 13 sensitivity dates are scored like the 52-date runs:

    .venv/bin/python -m nwpprec.eightbit --model pangu

Writes data/results/scores/<model>_fp16-<config>.csv.
"""

import argparse

import torch

from . import forecast
from .sensitivity import DATES, DEVICE, free

CONFIGS = ("w-int8", "w-fp8", "w8a8-int8", "w8a8-fp8")
LINEAR = {"Linear", "LinearPatchReconstruction"}
CONV = {"Conv1d", "Conv2d", "Conv3d"}
FP8_MAX, INT8_MAX = 448.0, 127.0


def quantise(x: torch.Tensor, fmt: str, reduce_dims) -> torch.Tensor:
    """x rounded to int8 or fp8 (e4m3) with one scale per slice (the maximum
    |x| over `reduce_dims`), returned in x's dtype."""
    amax = x.abs().amax(dim=reduce_dims, keepdim=True).float().clamp(min=1e-6)
    if fmt == "int8":
        scale = amax / INT8_MAX
        return (torch.round(x.float() / scale).clamp(-INT8_MAX, INT8_MAX) * scale).to(x.dtype)
    scale = amax / FP8_MAX
    return ((x.float() / scale).to(torch.float8_e4m3fn).float() * scale).to(x.dtype)


def _quantise_activation(x: torch.Tensor, fmt: str, conv: bool) -> torch.Tensor:
    if not isinstance(x, torch.Tensor) or not x.is_floating_point():
        return x
    if fmt == "fp8" or conv:  # one scale per tensor
        return quantise(x, fmt, tuple(range(x.dim())))
    rows = x.reshape(-1, x.shape[-1])
    out = torch.empty_like(rows)
    n = max(1, 2 ** 24 // x.shape[-1])  # in chunks: no full-size fp32 copies
    for i in range(0, rows.shape[0], n):
        out[i:i + n] = quantise(rows[i:i + n], fmt, (1,))  # one scale per token
    return out.reshape(x.shape)


def apply(name: str, model, config: str) -> None:
    """Quantise an fp16 model in place (and hook its layer inputs for w8a8)."""
    fmt = config.split("-")[1]
    with torch.no_grad():
        for module in model.modules():
            cls = type(module).__name__
            if cls in LINEAR | CONV:
                w = module.weight
                w.copy_(quantise(w, fmt, tuple(range(1, w.dim()))))
                if config.startswith("w8a8"):
                    module.register_forward_pre_hook(
                        lambda mod, args, _c=cls in CONV: (_quantise_activation(args[0], fmt, _c),) + args[1:] if args else None)
            elif cls == "AFNO2DLayer":  # einsum "...bi,bio->...bo": input axis -2
                for w in (module.w1, module.w2):
                    w.copy_(quantise(w, fmt, (w.dim() - 2,)))


def run(name: str, configs=CONFIGS) -> None:
    m = forecast.adapter(name)
    for config in configs:
        fmt = config.split("-")[1]
        if name == "sfno":  # complex filters: quantised where the adapter splits them into real and imaginary parts
            model = m.load("fp16", DEVICE, filter_transform=lambda w: quantise(w, fmt, (1,)))
        else:
            model = m.load("fp16", DEVICE)
        apply(name, model, config)
        forecast.run(name, "fp16", DATES, tag=f"fp16-{config}", model=model)
        print(f"[{name} 8-bit] {config} done", flush=True)
        del model
        free()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=forecast.MODELS)
    p.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=CONFIGS)
    a = p.parse_args()
    if a.model == "fastnet":
        torch.use_deterministic_algorithms(True)
    run(a.model, a.configs)


if __name__ == "__main__":
    main()
