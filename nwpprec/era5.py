"""ERA5 verification fields on the 0.25-degree grid (721 x 1440, latitude
90 to -90, longitude 0 to 359.75), from the daily files in paths.ERA5."""

from datetime import datetime
from functools import lru_cache

import numpy as np
import xarray as xr

from .paths import ERA5

LAT = np.linspace(90, -90, 721)
LON = np.arange(0, 360, 0.25)

# verification variable -> (file, ERA5 name, pressure level)
VARIABLES = {
    "2t": ("surface-level", "t2m", None),
    "msl": ("surface-level", "msl", None),
    "10u": ("surface-level", "u10", None),
    "10v": ("surface-level", "v10", None),
    "z500": ("atmospheric", "z", 500),
    "t850": ("atmospheric", "t", 850),
    "ws10": None,  # 10 m wind speed, from 10u and 10v
}


def inputs(time: datetime, surface: tuple = (), atmos: tuple = (), levels: tuple = ()) -> dict:
    """{name: float32 array} at `time`: surface fields (721, 1440) by ERA5 name,
    and pressure-level fields (len(levels), 721, 1440) in the order of
    `levels`. Missing values (NaN) are set to 0."""
    out = {}
    for kind, names in (("surface-level", surface), ("atmospheric", atmos)):
        if not names:
            continue
        with xr.open_dataset(ERA5 / f"{time:%Y-%m-%d}-{kind}.nc", engine="netcdf4") as ds:
            step = ds.sel(valid_time=time)
            for name in names:
                field = step[name] if kind == "surface-level" else step[name].sel(pressure_level=list(levels))
                out[name] = np.nan_to_num(field.values, nan=0.0).astype(np.float32)
    return out


def read(time: datetime, kind: str, name: str, level: int | None = None) -> np.ndarray:
    """One (721, 1440) float32 field from the file `kind` ("surface-level",
    "atmospheric", "relative-humidity"), NaN set to 0."""
    with xr.open_dataset(ERA5 / f"{time:%Y-%m-%d}-{kind}.nc", engine="netcdf4") as ds:
        field = ds[name].sel(valid_time=time)
        if level is not None:
            field = field.sel(pressure_level=level)
        return np.nan_to_num(field.values, nan=0.0).astype(np.float32)


@lru_cache(maxsize=32)
def load(valid_time: datetime, var: str) -> np.ndarray:
    """(721, 1440) float32 field of `var` at `valid_time` (cached: do not modify it)."""
    if var == "ws10":
        return np.hypot(load(valid_time, "10u"), load(valid_time, "10v"))
    kind, name, level = VARIABLES[var]
    path = ERA5 / f"{valid_time:%Y-%m-%d}-{kind}.nc"
    with xr.open_dataset(path, engine="netcdf4") as ds:
        field = ds[name].sel(valid_time=valid_time)
        if level is not None:
            field = field.sel(pressure_level=level)
        return field.values.astype(np.float32)
