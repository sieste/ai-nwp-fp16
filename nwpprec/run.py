"""Run one forecast and write it to NetCDF: for users who want the models
in fp16 (or fp32, bf16) without the paper's experiments.

    .venv/bin/python -m nwpprec.run --model aurora --precision fp16 \\
        --init 2024-06-01T00 --lead 240 --out aurora_fp16.nc

The forecast starts from ERA5 in data/inputs/era5, which

    .venv/bin/python -m nwpprec.download era5 --init 2024-06-01T00

downloads (from the Copernicus Climate Data Store; needs ~/.cdsapirc). And
the network computes in the chosen precision while the physical state,
normalisation and denormalisation stay in fp32. The output holds 2 m
temperature, mean sea-level pressure, the 10 m wind components and speed,
geopotential at 500 hPa and temperature at 850 hPa, at the requested lead
times, on the 0.25-degree grid (Aurora and FourCastNet leave out the south
pole row; FastNet's forecasts are interpolated from its O96 grid, which leaves
about 1.6% of the points, those outside the O96 triangulation, as NaN). Each model
runs in its own environment (envs/README.md).

From Python:

    from nwpprec.run import run
    ds = run("pangu", "fp16", "2024-06-01T00", lead=120)   # an xarray.Dataset
"""

import argparse
from datetime import datetime, timezone

import numpy as np
import xarray as xr

from . import forecast
from .download import days_needed
from .paths import ERA5

UNITS = {"2t": "K", "msl": "Pa", "10u": "m s-1", "10v": "m s-1", "ws10": "m s-1", "z500": "m2 s-2", "t850": "K"}
NAMES = {"2t": "2 m temperature", "msl": "mean sea-level pressure", "10u": "10 m eastward wind",
         "10v": "10 m northward wind", "ws10": "10 m wind speed", "z500": "geopotential at 500 hPa",
         "t850": "temperature at 850 hPa"}


def inputs(model: str, start: datetime) -> list:
    """The ERA5 files a forecast from `start` reads (Aurora also the state 6 hours before)."""
    days = days_needed([(f"{start:%Y-%m-%d}", f"{start:%H}")], max_lead_hours=0)
    files = [ERA5 / f"{d:%Y-%m-%d}-{kind}.nc" for d in days for kind in ("surface-level", "atmospheric")]
    if model == "fourcastnet":
        files.append(ERA5 / f"{start:%Y-%m-%d}-relative-humidity.nc")
    return files


def run(model: str, precision: str, init: str, lead: int = 240, every: int | None = None, seed: int = 0) -> xr.Dataset:
    """Forecast from `init` (e.g. "2024-06-01T00", UTC; 00, 06, 12 or 18) to
    `lead` hours, keeping every `every` hours (default: every model step).
    `seed` sets the noise of Aurora's ensemble checkpoint (one member per seed)."""
    m = forecast.adapter(model)
    step = m.STEP_HOURS
    every = every or step
    if every % step or lead % every:
        raise ValueError(f"{model} steps {step} h: `every` must be a multiple of {step}, and `lead` of `every`")
    start = datetime.strptime(init, "%Y-%m-%dT%H")
    missing = [p.name for p in inputs(model, start) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"ERA5 missing in {ERA5}: {', '.join(missing)}; "
                                f"python -m nwpprec.download era5 --init {init}")
    net = m.load(precision)
    kwargs = {"seed": seed} if model == "aurora_ens" else {}
    leads, data = [], {}
    for hours, state in m.forecast(net, start, lead, **kwargs):
        if hours % every:
            continue
        for var, x in forecast.with_wind_speed(m.fields(state)).items():
            data.setdefault(var, []).append(np.asarray(x, np.float32))
        leads.append(hours)
    lat, lon = np.asarray(m.LAT, np.float32), np.asarray(m.LON, np.float32)
    ds = xr.Dataset(
        {var: (("lead_time", "latitude", "longitude"), np.stack(v), {"units": UNITS[var], "long_name": NAMES[var]})
         for var, v in data.items()},
        coords={"lead_time": ("lead_time", np.array(leads, np.int32), {"units": "hours"}),
                "latitude": ("latitude", lat, {"units": "degrees_north"}),
                "longitude": ("longitude", lon, {"units": "degrees_east"})})
    ds.attrs.update({"model": model, "precision": precision, "init_time": start.strftime("%Y-%m-%dT%H:00Z"),
                     "initial_conditions": "ERA5",
                     "note": "network in the stated precision; physical state, normalisation and "
                             "denormalisation in fp32",
                     "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "source": "nwpprec.run"})
    if model == "aurora_ens":
        ds.attrs["seed"] = seed
    if model == "fastnet":
        ds.attrs["missing"] = "NaN at points outside the triangulation of FastNet's O96 grid"
    return ds


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=forecast.MODELS)
    p.add_argument("--precision", default="fp16", choices=("fp32", "fp16", "bf16"))
    p.add_argument("--init", required=True, help="initial time, UTC, e.g. 2024-06-01T00")
    p.add_argument("--lead", type=int, default=240, help="forecast length in hours (default 240)")
    p.add_argument("--every", type=int, help="keep every N hours (default: every model step)")
    p.add_argument("--seed", type=int, default=0, help="ensemble member for aurora_ens")
    p.add_argument("--out", required=True, help="output NetCDF file")
    a = p.parse_args()
    try:
        ds = run(a.model, a.precision, a.init, a.lead, a.every, a.seed)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(e)
    ds.to_netcdf(a.out, encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})
    print(f"Wrote {a.out}: {a.model} {a.precision} from {ds.attrs['init_time']}, "
          f"{len(ds.lead_time)} lead times to {int(ds.lead_time[-1])} h", flush=True)


if __name__ == "__main__":
    main()
