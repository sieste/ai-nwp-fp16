"""SFNO (spherical Fourier neural operator, `sfno_73ch_small`, 6-hour steps),
via earth2studio and NVIDIA's makani (third_party/makani, third_party/torch-harmonics).

The model wrapper normalises the physical-unit state and adds its forcing
(solar zenith angle, static fields) in fp32; the network inside it is
converted. Its spherical harmonic transforms always run in fp32 (makani
disables reduced precision for them). 98% of its parameters are complex
spectral filters, for which PyTorch has neither a bf16 type nor half-precision
complex matrix products on the GPU: in bf16 and fp16 each filter is stored as
its real and imaginary parts in that precision and applied with four real
products (as FourCastNet's spectral filters do). Note that a plain
`model.to(torch.float16)` keeps only the real part of complex parameters.

Needs its own environment (.venv-sfno).
"""

import sys
import types
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from .. import era5
from ..precision import DTYPES, cast

STEP_HOURS = 6
LAT, LON = era5.LAT, era5.LON
SKIP_NAN = False
SURFACE = {"u10m": "u10", "v10m": "v10", "u100m": "u100", "v100m": "v100", "t2m": "t2m", "sp": "sp", "msl": "msl",
           "tcwv": "tcwv"}  # SFNO name -> ERA5 name
ATMOS = ("u", "v", "z", "t", "q")
LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
VARIABLES = list(SURFACE) + [f"{v}{lev}" for v in ATMOS for lev in LEVELS]
FIELDS = {"2t": "t2m", "msl": "msl", "10u": "u10m", "10v": "v10m", "z500": "z500", "t850": "t850"}
COMPONENTS = ("encoder",) + tuple(f"block_{i}" for i in range(8)) + ("decoder",)
EINSUM = {"diagonal": "bgixy,gioxy->bgoxy", "dhconv": "bgixy,giox->bgoxy"}


def _stub_torch_harmonics() -> None:
    """torch_harmonics imports compiled extensions for submodules SFNO does not
    use (attention, disco, and through them distributed); stand-ins are
    registered before it is imported, re-exporting the distributed utilities
    that makani imports."""
    if "torch_harmonics.attention" in sys.modules:
        return
    sys.path.insert(0, str(Path(__file__).parent / "sfno_stubs"))
    attention = types.ModuleType("torch_harmonics.attention")
    attention.AttentionS2 = attention.NeighborhoodAttentionS2 = object
    attention.attention_kernels = None
    disco = types.ModuleType("torch_harmonics.disco")
    disco.DiscreteContinuousConvS2 = disco.DiscreteContinuousConvTransposeS2 = object
    distributed = types.ModuleType("torch_harmonics.distributed")
    import importlib.util
    package = Path(importlib.util.find_spec("torch_harmonics").origin).parent  # without importing it
    distributed.__path__ = [str(package / "distributed")]
    sys.modules.update({"torch_harmonics.attention": attention, "torch_harmonics.disco": disco,
                        "torch_harmonics.distributed": distributed})
    import importlib
    for sub in ("primitives", "utils", "distributed_quadrature", "distributed_resample", "distributed_sht",
                "distributed_spectral_convolution"):
        mod = importlib.import_module(f"torch_harmonics.distributed.{sub}")
        for name in dir(mod):
            if not name.startswith("__"):
                setattr(distributed, name, getattr(mod, name))


# Imported here rather than in load: earth2studio keeps the stack frames of its
# failed optional imports alive, which would keep the calling frame's model alive.
_stub_torch_harmonics()
from earth2studio.models.px.sfno import SFNO  # noqa: E402


def components(model) -> dict:
    net = model.model.model
    return {"encoder": net.encoder, **{f"block_{i}": b for i, b in enumerate(net.blocks)}, "decoder": net.decoder}


def _spectral_filters_in(module: torch.nn.Module, dtype: torch.dtype, transform=None) -> None:
    """Complex spectral filters inside `module` stored as real and imaginary
    parts in `dtype` (each passed through `transform`, if given) and applied
    with four real products in `dtype`."""
    for m in module.modules():
        weight = getattr(m, "weight", None)
        if not (hasattr(m, "_contract") and isinstance(weight, torch.Tensor) and weight.is_complex()):
            continue
        assert not m._contract.keywords["separable"]
        eq = EINSUM[m._contract.keywords["operator_type"]]
        wr, wi = weight.data.real.to(dtype).contiguous(), weight.data.imag.to(dtype).contiguous()
        if transform is not None:
            wr, wi = transform(wr), transform(wi)
        m.weight = None

        def contract(x, _weight, _wr=wr, _wi=wi, _eq=eq):
            xr, xi = x.real.to(_wr.dtype), x.imag.to(_wr.dtype)
            re = torch.einsum(_eq, xr, _wr) - torch.einsum(_eq, xi, _wi)
            im = torch.einsum(_eq, xr, _wi) + torch.einsum(_eq, xi, _wr)
            return torch.complex(re.float(), im.float())
        m._contract = contract


def _convert(module: torch.nn.Module, dtype: torch.dtype, filters: bool = True, transform=None) -> None:
    """As precision.convert, but real parameters only: complex filters are
    handled by _spectral_filters_in (unless filters=False: left in complex64)."""
    for t in list(module.parameters()) + list(module.buffers()):
        if t.is_floating_point():
            t.data = t.data.to(dtype)
    if filters:
        _spectral_filters_in(module, dtype, transform)
    original = module.forward

    def forward(*args, **kwargs):
        return cast(original(*cast(args, dtype), **cast(kwargs, dtype)), torch.float32)
    module.forward = forward


def load(precision: str, device: str = "cuda", components_: tuple | None = None, keep_filters: bool = False,
         filter_transform=None):
    """components_: convert only these components to `precision`, the rest of
    the network staying fp32 (the one-at-a-time experiments). keep_filters:
    leave the complex spectral filters in complex64 (the layer-type
    experiment). filter_transform: applied to the real and imaginary parts
    of the filters (the 8-bit experiment)."""
    wrapper = SFNO.load_model(SFNO.load_default_package())
    assert list(wrapper.input_coords()["variable"]) == VARIABLES
    model = wrapper.model.to(device).eval()
    if precision == "fp32":
        return model
    if components_ is None:
        _convert(model.model.model, DTYPES[precision], not keep_filters, filter_transform)  # the network; wrapper (normalisation, forcing) stays fp32
    else:
        for name, module in components(model).items():
            if name in components_:
                _convert(module, DTYPES[precision])
    return model


def initial_state(init_time, device: str = "cuda") -> torch.Tensor:
    x = [era5.read(init_time, "surface-level", SURFACE[v]) for v in SURFACE]
    x += [era5.read(init_time, "atmospheric", v, lev) for v in ATMOS for lev in LEVELS]
    return torch.from_numpy(np.stack(x)[None]).to(device)


def forecast(model, init_time, max_lead: int = 240):
    """Yields (lead hours, physical-unit state) every 6 hours."""
    x = initial_state(init_time, next(model.parameters()).device)
    t = init_time.replace(tzinfo=timezone.utc)
    for lead in range(STEP_HOURS, max_lead + 1, STEP_HOURS):
        if x.is_cuda:  # the fp32 step needs nearly all of a 16 GB GPU
            torch.cuda.empty_cache()
        with torch.inference_mode():
            x = model(x, time=t, normalized_data=False)
        t += timedelta(hours=STEP_HOURS)
        yield lead, x


def fields(state) -> dict:
    """{verification variable: (721, 1440) float32 array}."""
    x = state.float().cpu()[0]
    return {var: x[VARIABLES.index(name)].numpy() for var, name in FIELDS.items()}
