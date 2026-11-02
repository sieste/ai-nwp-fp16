"""FourCastNet (AFNO, 6-hour steps), via earth2studio, whose wrapper normalises
the state and denormalises the output in fp32. In bf16 and fp16, each of the
network's components (COMPONENTS) is converted; the learned positional
embedding, outside every component, stays fp32.
"""

from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from earth2studio.models.px import FCN
from earth2studio.models.px.fcn import VARIABLES
from physicsnemo.nn.module import fft
from physicsnemo.nn.module.afno_layers import AFNO2DLayer

from .. import era5
from ..precision import DTYPES, convert

STEP_HOURS = 6
LAT, LON = era5.LAT[:720], era5.LON  # FourCastNet's grid leaves out the south pole row
SKIP_NAN = False
COMPONENTS = ("patch_embed",) + tuple(f"block_{i}" for i in range(12)) + ("head",)
FIELDS = {"2t": "t2m", "msl": "msl", "10u": "u10m", "10v": "v10m", "z500": "z500", "t850": "t850"}
# FourCastNet input channel -> (ERA5 file, name, level)
INPUTS = {"u10m": ("surface-level", "u10", None), "v10m": ("surface-level", "v10", None),
          "t2m": ("surface-level", "t2m", None), "sp": ("surface-level", "sp", None),
          "msl": ("surface-level", "msl", None), "tcwv": ("surface-level", "tcwv", None),
          "u100m": ("surface-level", "u100", None), "v100m": ("surface-level", "v100", None),
          "r500": ("relative-humidity", "r", 500), "r850": ("relative-humidity", "r", 850)}
INPUTS.update({f"{v}{lev}": ("atmospheric", v, lev) for v, levs in
               {"t": (850, 500, 250), "u": (1000, 850, 500, 250), "v": (1000, 850, 500, 250),
                "z": (1000, 850, 500, 250, 50)}.items() for lev in levs})
assert set(INPUTS) == set(VARIABLES)


def _afno_forward(self, x):
    """AFNO2DLayer.forward with its spectral weights applied in their own
    dtype. The original casts to fp32 for the Fourier transform (which has
    no bf16 kernel) and allocates its intermediate buffers in fp32, which
    fails with bf16/fp16 weights; here the transformed values and buffers
    take the weights' dtype, and are cast back to fp32 for the inverse
    transform. With fp32 weights this is the original computation."""
    bias, dtype = x, x.dtype
    x = x.float()
    B, H, W, C = x.shape
    x = fft.rfft2(x, dim=(1, 2), norm="ortho")
    wd = self.w1.dtype
    x_real = fft.real(x).reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size).to(wd)
    x_imag = fft.imag(x).reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size).to(wd)
    shape = [B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor]
    o1_real = torch.zeros(shape, device=x.device, dtype=wd)
    o1_imag = torch.zeros(shape, device=x.device, dtype=wd)
    o2 = torch.zeros(x_real.shape + (2,), device=x.device, dtype=wd)
    total = H // 2 + 1
    kept = int(total * self.hard_thresholding_fraction)
    lo, hi = total - kept, total + kept
    xr, xi = x_real[:, lo:hi, :kept], x_imag[:, lo:hi, :kept]
    o1_real[:, lo:hi, :kept] = F.relu(torch.einsum("nyxbi,bio->nyxbo", xr, self.w1[0])
                                      - torch.einsum("nyxbi,bio->nyxbo", xi, self.w1[1]) + self.b1[0])
    o1_imag[:, lo:hi, :kept] = F.relu(torch.einsum("nyxbi,bio->nyxbo", xi, self.w1[0])
                                      + torch.einsum("nyxbi,bio->nyxbo", xr, self.w1[1]) + self.b1[1])
    o1r, o1i = o1_real[:, lo:hi, :kept], o1_imag[:, lo:hi, :kept]
    o2[:, lo:hi, :kept, ..., 0] = (torch.einsum("nyxbi,bio->nyxbo", o1r, self.w2[0])
                                   - torch.einsum("nyxbi,bio->nyxbo", o1i, self.w2[1]) + self.b2[0])
    o2[:, lo:hi, :kept, ..., 1] = (torch.einsum("nyxbi,bio->nyxbo", o1i, self.w2[0])
                                   + torch.einsum("nyxbi,bio->nyxbo", o1r, self.w2[1]) + self.b2[1])
    x = F.softshrink(o2.float(), lambd=self.sparsity_threshold)
    x = fft.view_as_complex(x).reshape(B, H, W // 2 + 1, C)
    x = fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
    return x.type(dtype) + bias


AFNO2DLayer.forward = _afno_forward


def components(model) -> OrderedDict:
    net = model.model
    return OrderedDict([("patch_embed", net.patch_embed)] + [(f"block_{i}", b) for i, b in enumerate(net.blocks)]
                       + [("head", net.head)])


def load(precision: str, device: str = "cuda", components_: tuple | None = None):
    """components_: convert only these components to `precision`, the rest
    staying fp32 (the one-at-a-time experiments); default: all of them."""
    model = FCN.load_model(FCN.load_default_package()).to(device=device, dtype=torch.float32).eval()
    if precision != "fp32":
        names = COMPONENTS if components_ is None else components_
        for name, module in components(model).items():
            if name in names:
                convert(module, DTYPES[precision])
    return model


def initial_state(init_time, model, device: str = "cuda") -> tuple:
    x = np.stack([era5.read(init_time, *INPUTS[v])[:720] for v in VARIABLES])[None, None]
    coords = OrderedDict(model.input_coords())
    coords["batch"] = np.array([0])
    return torch.from_numpy(x).to(device), coords


def forecast(model, init_time, max_lead: int = 240):
    """Yields (lead hours, state) every 6 hours."""
    x, coords = initial_state(init_time, model, next(model.parameters()).device)
    it = model.create_iterator(x, coords)
    with torch.inference_mode():
        next(it)  # the initial condition
        for lead in range(STEP_HOURS, max_lead + 1, STEP_HOURS):
            yield lead, next(it)[0]


def fields(state) -> dict:
    """{verification variable: (720, 1440) float32 array}."""
    x = state.float().cpu()[0, 0]
    return {var: x[VARIABLES.index(name)].numpy() for var, name in FIELDS.items()}
