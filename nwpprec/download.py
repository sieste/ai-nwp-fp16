"""Download ERA5 data and model checkpoints.

    .venv/bin/python -m nwpprec.download era5          # every day the 52 forecasts need
    .venv/bin/python -m nwpprec.download era5 --init 2024-06-01T00   # one initial state (nwpprec.run)
    .venv/bin/python -m nwpprec.download checkpoints   # Aurora and Pangu-Weather

ERA5 comes from the Copernicus Climate Data Store (needs ~/.cdsapirc): per
day, a surface file and a pressure-level file, and for the initialisation
days a relative-humidity file (FourCastNet's input), each with 00/06/12/18
UTC, into data/inputs/era5. With --init, only the days that the given
initial times' states need (no verification). Existing
files are skipped; a failed download leaves no file behind.

Checkpoints come from Hugging Face at pinned revisions. FastNet's and
FourCastNet's weights (Hugging Face, the latter via earth2studio) and
SFNO's (NVIDIA NGC, via earth2studio) are downloaded automatically when
the models are first loaded. FastNet's static
fields (data/fastnet_static_o96.npz) were saved from ECMWF's anemoi dataset
era5-o96-1979-2023-6h-v8, which now requires authentication.
"""

import argparse
import zipfile
from datetime import timedelta

from .dates import init_time, weekly_dates
from .paths import CHECKPOINTS, ERA5

SURFACE = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure",
           "2m_dewpoint_temperature", "total_column_water_vapour", "total_cloud_cover", "100m_u_component_of_wind",
           "100m_v_component_of_wind", "surface_pressure", "low_cloud_cover", "medium_cloud_cover",
           "high_cloud_cover", "skin_temperature", "soil_temperature_level_1", "volumetric_soil_water_layer_1",
           "sea_ice_cover", "snow_depth"]
ATMOSPHERIC = ["temperature", "u_component_of_wind", "v_component_of_wind", "specific_humidity", "geopotential"]
LEVELS = ["50", "100", "150", "200", "250", "300", "400", "500", "600", "700", "850", "925", "1000"]
FILES = {  # file -> (CDS dataset, variables, pressure levels)
    "surface-level": ("reanalysis-era5-single-levels", SURFACE, None),
    "atmospheric": ("reanalysis-era5-pressure-levels", ATMOSPHERIC, LEVELS),
    "relative-humidity": ("reanalysis-era5-pressure-levels", ["relative_humidity"], ["500", "850"]),
}
AURORA = ("microsoft/aurora", "a96afd7ee6d65e3bd2d476f3be798a25a56f2296",
          ["aurora-0.25-v1.5.ckpt", "aurora-0.25-v1.5-ensemble.ckpt", "aurora-0.25-v1.5-static.pickle"])
PANGU = ("zhaoshan/pangu_pytorch", "65cd779c938be4e307c96447f517e11ab619b629", ["aux_data.zip", "pretrained_model.zip"])


def days_needed(dates: list, max_lead_hours: int = 240) -> list:
    """Every calendar day from 6 hours before each initialisation (Aurora's
    history step) to its last lead time."""
    days = set()
    for date, hour in dates:
        start = init_time(date, hour)
        t = start - timedelta(hours=6)
        while t <= start + timedelta(hours=max_lead_hours):
            days.add(t.date())
            t += timedelta(hours=6)
    return sorted(days)


def era5(dates: list, max_lead_hours: int = 240) -> None:
    """Surface and pressure-level files for every day needed; relative
    humidity (FourCastNet's input only) for the initialisation days."""
    import cdsapi
    client = cdsapi.Client()
    ERA5.mkdir(parents=True, exist_ok=True)
    init_days = {init_time(d, h).date() for d, h in dates}
    for day in days_needed(dates, max_lead_hours):
        for kind, (dataset, variables, levels) in FILES.items():
            if kind == "relative-humidity" and day not in init_days:
                continue
            path = ERA5 / f"{day:%Y-%m-%d}-{kind}.nc"
            if path.exists() and path.stat().st_size > 0:
                continue
            request = {"product_type": "reanalysis", "variable": variables, "year": f"{day.year:04d}",
                       "month": f"{day.month:02d}", "day": f"{day.day:02d}",
                       "time": ["00:00", "06:00", "12:00", "18:00"], "data_format": "netcdf"}
            if levels:
                request["pressure_level"] = levels
            print(f"downloading {path.name}", flush=True)
            part = path.with_name(path.name + ".part")
            try:
                client.retrieve(dataset, request, str(part))
            except BaseException:
                part.unlink(missing_ok=True)
                raise
            part.replace(path)


def checkpoints() -> None:
    from huggingface_hub import hf_hub_download
    repo, revision, files = AURORA
    for f in files:
        hf_hub_download(repo_id=repo, filename=f, revision=revision, local_dir=str(CHECKPOINTS))
    repo, revision, files = PANGU
    for f in files:
        path = hf_hub_download(repo_id=repo, repo_type="dataset", filename=f, revision=revision,
                               local_dir=str(CHECKPOINTS / "pangu"))
        with zipfile.ZipFile(path) as z:
            z.extractall(CHECKPOINTS / "pangu")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=("era5", "checkpoints"))
    p.add_argument("--init", nargs="+", help="era5 for these initial times only, e.g. 2024-06-01T00")
    a = p.parse_args()
    if a.what == "era5" and a.init:
        era5([tuple(t.split("T")) for t in a.init], max_lead_hours=0)
    elif a.what == "era5":
        era5(weekly_dates())
    else:
        checkpoints()


if __name__ == "__main__":
    main()
