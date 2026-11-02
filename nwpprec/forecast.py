"""Run one model at one precision from many initial dates and score it.

Every forecast is scored against ERA5 at LEADS (daily, days 1-10): RMSE,
bias and MAE; globally, per region and per location (scoring.py). If the
fp32 forecast fields of the same model and dates have been saved
(--save-fields with --precision fp32), each forecast is also scored against
the fp32 forecast (column rmse_vs_fp32: its departure). The forecast values
at the 3 x 3 grid points of each location are saved for every model step.

    .venv/bin/python -m nwpprec.forecast --model pangu --precision fp32 --save-fields
    .venv/bin/python -m nwpprec.forecast --model pangu --precision fp16 --regional

Scores: data/results/scores/<model>_<tag>.csv; fields: data/results/fields/<model>_<tag>/;
location time series: data/results/points/<model>_<tag>/;
regional statistics and spectra (--regional, nwpprec/regional.py): data/results/regional/<model>_<tag>.npz.
Model environments: aurora and pangu .venv, fourcastnet .venv-fourcastnet,
fastnet .venv-fastnet, sfno .venv-sfno.
"""

import argparse
import csv
import importlib
import time
from datetime import timedelta

import numpy as np

from . import era5
from .dates import balanced_subset, init_time, weekly_dates
from .paths import RESULTS
from .scoring import LOCATIONS, location_masks, regional_scores

MODELS = ("aurora", "aurora_ens", "pangu", "fourcastnet", "fastnet", "sfno")
LEADS = tuple(range(24, 241, 24))
COLUMNS = ("date", "hour", "var", "lead", "region", "rmse", "bias", "mae", "rmse_vs_fp32")


def with_wind_speed(fields: dict) -> dict:
    return {**fields, "ws10": np.hypot(fields["10u"], fields["10v"])}


def adapter(name: str):
    return importlib.import_module(f"nwpprec.models.{name}")


def fields_dir(name: str, tag: str):
    return RESULTS / "fields" / f"{name}_{tag}"


def run(name: str, precision: str, dates: list, tag: str | None = None, model=None,
        save_fields: bool = False, on_fields=None) -> list:
    """Forecasts from `dates`, scored; returns the rows and writes them to
    RESULTS/scores/<name>_<tag>.csv. `model`: an already built (e.g. partly
    converted) model instead of adapter.load(precision); tag defaults to
    the precision. on_fields(date, hour, lead, fields), if given, is called
    with every scored forecast's fields."""
    m = adapter(name)
    tag = tag or precision
    model = model if model is not None else m.load(precision)
    ref_dir, out_dir = fields_dir(name, "fp32"), fields_dir(name, tag)
    points_dir = RESULTS / "points" / f"{name}_{tag}"
    boxes = {loc: np.ix_(*location_masks(m.LAT, m.LON, loc)) for loc in LOCATIONS}
    rows = []
    for i, (date, hour) in enumerate(dates):
        t0 = time.time()
        start = init_time(date, hour)
        ref_path = ref_dir / f"{date}_{hour}.npz"
        ref = np.load(ref_path) if ref_path.exists() and tag != "fp32" else None
        saved, points, steps = {}, {}, []
        for lead, state in m.forecast(model, start, max(LEADS)):
            fields = with_wind_speed(m.fields(state))
            steps.append(lead)
            for var, x in fields.items():
                for loc, box in boxes.items():
                    points.setdefault(f"{loc}_{var}", []).append(x[box])
            if lead not in LEADS:
                continue
            valid = start + timedelta(hours=lead)
            for var, x in fields.items():
                truth = era5.load(valid, var)[:len(m.LAT)]
                scores = regional_scores(x, truth, m.LAT, m.LON, skip_nan=m.SKIP_NAN)
                departure = (regional_scores(x, ref[f"{var}_{lead}"], m.LAT, m.LON, skip_nan=m.SKIP_NAN)
                             if ref is not None else None)
                for region, s in scores.items():
                    rows.append({"date": date, "hour": hour, "var": var, "lead": lead, "region": region, **s,
                                 "rmse_vs_fp32": departure[region]["rmse"] if departure else ""})
                if save_fields:
                    saved[f"{var}_{lead}"] = x.astype(np.float32)
            if on_fields is not None:
                on_fields(date, hour, lead, fields)
        if save_fields:
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez(out_dir / f"{date}_{hour}.npz", **saved)
        points_dir.mkdir(parents=True, exist_ok=True)
        np.savez(points_dir / f"{date}_{hour}.npz", leads=np.array(steps), **{k: np.stack(v) for k, v in points.items()})
        print(f"[{name} {tag} {i + 1}/{len(dates)}] {date} {hour}h: {time.time() - t0:.0f}s", flush=True)
    path = RESULTS / "scores" / f"{name}_{tag}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}", flush=True)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--precision", required=True, choices=("fp32", "bf16", "fp16"))
    p.add_argument("--dates", default="all", choices=("all", "oat"), help="the 52 weekly dates, or the 5 OAT dates")
    p.add_argument("--save-fields", action="store_true")
    p.add_argument("--regional", action="store_true", help="also accumulate regional statistics and spectra")
    a = p.parse_args()
    dates = weekly_dates() if a.dates == "all" else balanced_subset(weekly_dates())
    regional = None
    if a.regional:
        from .regional import RegionalStats
        m = adapter(a.model)
        regional = RegionalStats(a.model, dates, m.LAT, m.LON)
    run(a.model, a.precision, dates, save_fields=a.save_fields, on_fields=regional)
    if regional is not None:
        regional.save(RESULTS / "regional" / f"{a.model}_{a.precision}.npz")


if __name__ == "__main__":
    main()
