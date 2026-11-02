"""Pangu-Weather (24-hour model), via the pangu-pytorch reimplementation in
third_party/pangu-pytorch, with the weights in checkpoints/pangu.

The network normalises its inputs inside its first layer. To keep the state
and its normalisation in fp32, the inputs are normalised here in fp32 and
the network is given mean-0 / std-1 statistics, so that its own
normalisation does nothing; it returns normalised values, denormalised here
in fp32.
"""

import sys

import numpy as np
import torch

from .. import era5
from ..paths import CHECKPOINTS, THIRD_PARTY
from ..precision import DTYPES, convert

sys.path.insert(0, str(THIRD_PARTY / "pangu-pytorch"))
from models.layers import EarthSpecificBlock  # noqa: E402
from models.pangu_model import PanguModel  # noqa: E402

STEP_HOURS = 24
LAT, LON = era5.LAT, era5.LON  # grid of fields()
SKIP_NAN = False  # fields() covers every grid point
UPPER = ("z", "q", "t", "u", "v")
SURFACE = ("msl", "u10", "v10", "t2m")
LEVELS = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)
FIELDS = {"2t": ("surface", "t2m", None), "msl": ("surface", "msl", None), "10u": ("surface", "u10", None),
          "10v": ("surface", "v10", None), "z500": ("upper", "z", 500), "t850": ("upper", "t", 850)}
AUX = CHECKPOINTS / "pangu" / "aux_data"


def _gen_mask(self, x):
    """EarthSpecificBlock.gen_mask building the attention mask in x.dtype:
    the original builds it in fp32, which fails in a bf16/fp16 network."""
    ws = self.window_size
    mask = torch.zeros((1, x.shape[1], x.shape[2], x.shape[3], 1), dtype=x.dtype, device=self.device)
    mz, mh, mw = mask.shape[1:4]
    cnt = 0
    for z in (slice(0, -ws[0]), slice(-ws[0], -ws[0] // 2), slice(-ws[0] // 2, None)):
        for h in (slice(0, -ws[1]), slice(ws[1], -ws[1] // 2), slice(-ws[1] // 2, None)):
            mask[:, z, h, :, :] = cnt
            cnt += 1
    mask = mask.reshape(1, mz // ws[0], ws[0], mh // ws[1], ws[1], mw // ws[2], ws[2], 1)
    mask = torch.permute(mask, (0, 5, 1, 3, 2, 4, 6, 7))
    windows = mask.reshape(-1, self.type_of_windows, ws[0], ws[1], ws[2], 1)
    windows = windows.view(-1, self.type_of_windows, ws[0] * ws[1] * ws[2])
    attn = windows.unsqueeze(2) - windows.unsqueeze(3)
    return attn.masked_fill(attn != 0, float(-100.0)).masked_fill(attn == 0, float(0.0))


EarthSpecificBlock.gen_mask = _gen_mask


def _aux(name, device):
    return torch.from_numpy(np.load(AUX / name).astype(np.float32)).to(device)


class Pangu(torch.nn.Module):
    """Physical-unit fp32 state in, physical-unit fp32 state out; the network
    (self.net) computes in `dtype`."""

    def __init__(self, dtype: torch.dtype, device: str = "cuda"):
        super().__init__()
        net = PanguModel(device=torch.device(device))
        net.load_state_dict(torch.load(CHECKPOINTS / "pangu" / "pretrained_model" / "pangu_weather_24_torch.pth",
                                       map_location="cpu", weights_only=False)["model"])
        self.net = net.to(dtype).eval().to(device)
        self.dtype = dtype
        s_mean, s_std, u_mean, u_std = (_aux(f, device) for f in ("surface_mean.npy", "surface_std.npy",
                                                                    "upper_mean.npy", "upper_std.npy"))
        # input statistics: per surface variable; per level and upper variable, stored with levels reversed
        self.s_mean, self.s_std = s_mean.view(1, -1, 1, 1), s_std.view(1, -1, 1, 1)
        self.u_mean = u_mean.flip(0)[:, 0, 0, :].T.reshape(1, 5, 13, 1, 1)
        self.u_std = u_std.flip(0)[:, 0, 0, :].T.reshape(1, 5, 13, 1, 1)
        # output statistics: same values, broadcast against (B, 5, 13, H, W) with levels reversed
        self.out_u_mean = u_mean.cpu().flip(0).permute(1, 3, 0, 2)[..., None].to(device)
        self.out_u_std = u_std.cpu().flip(0).permute(1, 3, 0, 2)[..., None].to(device)
        self.identity = (torch.zeros_like(s_mean), torch.ones_like(s_std), torch.zeros_like(u_mean), torch.ones_like(u_std))
        self.maps, self.const_h = _aux("constantMaks3.npy", device), _aux("Constant_17_output_0.npy", device)

    def forward(self, upper, surface):
        upper_n = (upper - self.u_mean) / self.u_std
        surface_n = (surface - self.s_mean) / self.s_std
        d = self.dtype
        out_u, out_s = self.net(upper_n.to(d), surface_n.to(d), tuple(s.to(d) for s in self.identity),
                                self.maps.to(d), self.const_h.to(d))
        return out_u.float() * self.out_u_std + self.out_u_mean, out_s.float() * self.s_std + self.s_mean


COMPONENTS = ("patch_embedding", "stage_0", "downsample", "stage_1", "stage_2", "upsample", "stage_3", "patch_recovery")


def components(model: Pangu) -> dict:
    net = model.net
    return dict(zip(COMPONENTS, (net._input_layer, net.layers[0], net.downsample, net.layers[1], net.layers[2],
                                 net.upsample, net.layers[3], net._output_layer)))


def load(precision: str, device: str = "cuda", components_: tuple | None = None) -> Pangu:
    """components_: convert only these components to `precision`, the rest of
    the network staying fp32 (the one-at-a-time experiments)."""
    if components_ is None:
        return Pangu(DTYPES[precision], device)
    model = Pangu(torch.float32, device)
    for name, module in components(model).items():
        if name in components_:
            convert(module, DTYPES[precision])
    return model


def initial_state(init_time, device: str = "cuda") -> tuple:
    x = era5.inputs(init_time, surface=SURFACE, atmos=UPPER, levels=LEVELS)
    upper = torch.from_numpy(np.stack([x[v] for v in UPPER])[None]).to(device)
    surface = torch.from_numpy(np.stack([x[v] for v in SURFACE])[None]).to(device)
    return upper, surface


def forecast(model: Pangu, init_time, max_lead: int = 240):
    """Yields (lead hours, state) every 24 hours."""
    state = initial_state(init_time, next(model.net.parameters()).device)
    with torch.inference_mode():
        for lead in range(STEP_HOURS, max_lead + 1, STEP_HOURS):
            state = model(*state)
            yield lead, state


def fields(state) -> dict:
    """{verification variable: (721, 1440) float32 array}."""
    upper, surface = (t[0].float().cpu().numpy() for t in state)
    out = {}
    for var, (kind, name, level) in FIELDS.items():
        out[var] = surface[SURFACE.index(name)] if kind == "surface" else upper[UPPER.index(name), LEVELS.index(level)]
    return out
