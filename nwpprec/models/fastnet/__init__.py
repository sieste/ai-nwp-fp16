"""FastNet (graph network on the O96 grid, 6-hour steps), via the
fastnet_inference package. Its state is normalised and stays in fp32
between steps; in bf16 and fp16 the network is converted as a whole.

Inputs are interpolated from the 0.25-degree ERA5 files to the O96 points;
forecasts are interpolated back (linearly, on the triangulation of the O96
points) for verification, which leaves the few 0.25-degree points outside
that triangulation as NaN (SKIP_NAN). The static fields (land-sea mask,
orography and two sub-grid orography statistics) are read from
data/fastnet_static_o96.npz, saved from ECMWF's anemoi dataset
era5-o96-1979-2023-6h-v8.
"""

import importlib.resources
import json
from datetime import timedelta
from functools import cache

import numpy as np
import torch
from earthkit.meteo.solar import cos_solar_zenith_angle
from fastnet_inference.data import get_fastnet_var_order
from fastnet_inference.model import load_model
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator

from ... import era5
from ...paths import DATA
from ...precision import DTYPES, convert

STEP_HOURS = 6
LAT, LON = era5.LAT, era5.LON
SKIP_NAN = True
FIELDS = {"2t": "2t", "msl": "msl", "10u": "10u", "10v": "10v", "z500": "z_500", "t850": "t_850"}
SOURCE = {"2t": ("surface-level", "t2m"), "msl": ("surface-level", "msl"), "10u": ("surface-level", "u10"),
          "10v": ("surface-level", "v10"), "2d": ("surface-level", "d2m"), "skt": ("surface-level", "skt"),
          "sp": ("surface-level", "sp"), "z": ("atmospheric", "z"), "q": ("atmospheric", "q"),
          "t": ("atmospheric", "t"), "u": ("atmospheric", "u"), "v": ("atmospheric", "v")}
NOT_STANDARDISED = {"cos_latitude", "cos_longitude", "sin_longitude"}
FORECAST_VARS, FORCING_VARS = get_fastnet_var_order()


@cache
def static() -> dict:
    return dict(np.load(DATA / "fastnet_static_o96.npz"))


@cache
def stats() -> tuple:
    with (importlib.resources.files("fastnet_inference") / "stats.json").open() as f:
        s = json.load(f)
    return np.array(list(s["mean"].values()), dtype=np.float32), np.array(list(s["stdev"].values()), dtype=np.float32)


@cache
def _to_regular():
    return LinearNDInterpolator(np.column_stack([static()["lon"], static()["lat"]]), np.zeros(len(static()["lat"])))


def to_regular(values: np.ndarray) -> np.ndarray:
    """O96 field -> (721, 1440) field, NaN outside the O96 triangulation."""
    interp = _to_regular()
    interp.values = values.astype(np.float64)[:, None]
    lon2d, lat2d = np.meshgrid(LON, LAT)
    return np.squeeze(interp(lon2d, lat2d))


def to_o96(field: np.ndarray) -> np.ndarray:
    """(721, 1440) field -> O96 points (bilinear)."""
    interp = RegularGridInterpolator((LAT[::-1], LON), field[::-1].astype(np.float64), method="linear",
                                     bounds_error=False, fill_value=None)
    return interp(np.column_stack([static()["lat"], static()["lon"]])).astype(np.float32)


def _forecast_var(time, var):
    if "_" in var:
        base, level = var.split("_")
        kind, name = SOURCE[base]
        return to_o96(era5.read(time, kind, name, int(level)))
    return to_o96(era5.read(time, *SOURCE[var]))


def _forcing_var(time, var):
    lat, lon = static()["lat"], static()["lon"]
    day = 2 * np.pi * ((time - time.replace(month=1, day=1, hour=0)).total_seconds() / 86400.0) / 365.25
    local = 2 * np.pi * ((time.hour + time.minute / 60.0 + lon / 15.0) % 24.0) / 24.0
    return {"cos_latitude": lambda: np.cos(np.radians(lat)), "cos_longitude": lambda: np.cos(np.radians(lon)),
            "sin_longitude": lambda: np.sin(np.radians(lon)), "lsm": lambda: static()["lsm"],
            "orography": lambda: static()["orography"], "sdor": lambda: static()["sdor"],
            "slor": lambda: static()["slor"], "cos_julian_day": lambda: np.cos(day), "sin_julian_day": lambda: np.sin(day),
            "cos_local_time": lambda: np.cos(local), "sin_local_time": lambda: np.sin(local),
            "insolation": lambda: cos_solar_zenith_angle(time, lat, lon).astype(np.float32)}[var]()


def _normalise(x: np.ndarray, variables: list) -> torch.Tensor:
    """(..., n_points, len(variables)) raw values -> standardised (except NOT_STANDARDISED)."""
    mean, std = stats()
    idx = [(FORECAST_VARS + FORCING_VARS).index(v) for v in variables]
    raw = torch.from_numpy(x)
    out = (raw - mean[idx]) / std[idx]
    keep = [i for i, v in enumerate(variables) if v in NOT_STANDARDISED]
    out[..., keep] = raw[..., keep]
    return out.float()


COMPONENTS = ("mesh_edge_embedder", "encoder") + tuple(f"processor_{i}" for i in range(16)) + ("decoder",)


def components(net) -> dict:
    return {"mesh_edge_embedder": net.mesh_edge_embedder, "encoder": net.encoder,
            **{f"processor_{i}": m for i, m in enumerate(net.processor.interaction_nets)}, "decoder": net.decoder}


def load(precision: str, device: str = "cuda", components_: tuple | None = None):
    """components_: convert only these components to `precision`, the rest of
    the network staying fp32 (the one-at-a-time experiments)."""
    net = load_model(device=device)
    if precision == "fp32":
        return net
    if components_ is None:
        convert(net, DTYPES[precision])
        return net
    for name, module in components(net).items():
        if name in components_:
            convert(module, DTYPES[precision])
    return net


def inputs(init_time, steps: int) -> tuple:
    """(normalised initial state (1, points, forecast vars), normalised
    forcing for each step (1, steps, points, forcing vars)), on the CPU."""
    n_points = len(static()["lat"])
    x0 = np.zeros((1, n_points, len(FORECAST_VARS)), dtype=np.float32)
    for i, var in enumerate(FORECAST_VARS):
        x0[0, :, i] = _forecast_var(init_time, var)
    forcing = np.zeros((steps, n_points, len(FORCING_VARS)), dtype=np.float32)
    for t in range(steps):
        for i, var in enumerate(FORCING_VARS):
            forcing[t, :, i] = _forcing_var(init_time + timedelta(hours=STEP_HOURS * t), var)
    return _normalise(x0, FORECAST_VARS), _normalise(forcing, FORCING_VARS).unsqueeze(0)


def forecast(model, init_time, max_lead: int = 240):
    """Yields (lead hours, normalised state) every 6 hours."""
    device = next(model.parameters()).device
    steps = max_lead // STEP_HOURS
    state, forcing = (t.to(device) for t in inputs(init_time, steps))
    with torch.inference_mode():
        for t in range(steps):
            out = model(forcing[:, t:t + 1], state)
            state = out.squeeze(1) if out.dim() == 4 else out
            yield STEP_HOURS * (t + 1), state


def fields(state) -> dict:
    """{verification variable: (721, 1440) float32 array}, NaN outside the O96 triangulation."""
    mean, std = stats()
    n = len(FORECAST_VARS)
    x = state.float().squeeze(0).cpu().numpy() * std[:n] + mean[:n]
    return {var: to_regular(x[:, FORECAST_VARS.index(name)].astype(np.float32)) for var, name in FIELDS.items()}
