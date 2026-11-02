"""Running parts of a network in reduced precision.

The physical state of a model, its normalisation and denormalisation stay in
fp32; only the network computes in bf16 or fp16. Each model adapter
(nwpprec/models) decides where the network begins and ends and uses these
primitives there:

    convert(module, dtype)    weights to dtype; floating-point inputs cast to
                              dtype on entry, outputs back to fp32 on exit
    cast_inputs(module, dtype)  inputs cast on entry only
    keep_fp32(module, fp32_weights, dtype)  the reverse of convert, for a
                              module inside an already converted network

convert() of one component is also what the one-at-a-time experiments use.
"""

import torch

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}

# type -> function(obj, dtype) for containers a model passes between its
# parts (e.g. Aurora's Batch); registered by the model adapter
CASTERS = {}


def cast(obj, dtype: torch.dtype):
    """Floating-point tensors in `obj` (also inside tuples, lists, dicts and
    registered types) converted to `dtype`; everything else unchanged."""
    if isinstance(obj, torch.Tensor):
        return obj.to(dtype) if obj.is_floating_point() else obj
    if type(obj) in CASTERS:
        return CASTERS[type(obj)](obj, dtype)
    if isinstance(obj, (tuple, list)):
        return type(obj)(cast(o, dtype) for o in obj)
    if isinstance(obj, dict):
        return {k: cast(v, dtype) for k, v in obj.items()}
    return obj


def cast_inputs(module: torch.nn.Module, dtype: torch.dtype) -> None:
    original = module.forward

    def forward(*args, **kwargs):
        return original(*cast(args, dtype), **cast(kwargs, dtype))

    module.forward = forward


def convert(module: torch.nn.Module, dtype: torch.dtype) -> None:
    module.to(dtype)
    original = module.forward

    def forward(*args, **kwargs):
        return cast(original(*cast(args, dtype), **cast(kwargs, dtype)), torch.float32)

    module.forward = forward


def keep_fp32(module: torch.nn.Module, fp32_weights: dict, dtype: torch.dtype) -> None:
    """Keep `module` in fp32 inside a network converted to `dtype`.

    fp32_weights: the module's state dict taken *before* the conversion --
    converting bf16 weights back to fp32 would keep their bf16 rounding.
    """
    module.float()
    module.load_state_dict(fp32_weights)
    original = module.forward

    def forward(*args, **kwargs):
        return cast(original(*cast(args, torch.float32), **cast(kwargs, torch.float32)), dtype)

    module.forward = forward
