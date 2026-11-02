"""Ensemble verification of the Aurora ensemble checkpoint.

Every 4th of the 52 dates (13), 8 members per date with noise seeds 0-7,
the same seeds in every precision, so that member m of each precision is
the same stochastic draw. Per date, variable, lead time and region
(cos-lat weighted):

    ensmean_rmse   RMSE of the ensemble mean against ERA5
    member_rmse    mean over members of each member's RMSE
    spread         square root of the area mean of the ensemble variance
    crps           fair (finite-ensemble) CRPS
    outside_frac   fraction of the area where ERA5 lies outside the ensemble
    rmsd_vs_fp32   mean over members of the RMS difference from the
                   same-seed fp32 member (not for fp32)

    .venv/bin/python -m nwpprec.ensemble --precision fp32
    .venv/bin/python -m nwpprec.ensemble --precision fp16

Scores: data/results/scores/aurora_ens_ensemble_<precision>.csv; the fp32
members are saved to data/results/fields/aurora_ens_members_fp32/.
"""

import argparse
import csv
from datetime import timedelta

import numpy as np

from . import era5, forecast
from .dates import init_time, weekly_dates
from .models import aurora_ens as m
from .paths import RESULTS
from .scoring import REGIONS, region_masks

N_MEMBERS = 8
DATES = weekly_dates()[::4]


def fair_crps(ens: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per point: mean_i |x_i - y| - sum_{i<j} |x_i - x_j| / (M (M - 1))."""
    n = ens.shape[0]
    xs = np.sort(ens, axis=0)
    coef = (2 * np.arange(n) - n + 1).reshape((n,) + (1,) * (ens.ndim - 1))
    return np.abs(ens - truth).mean(axis=0) - (coef * xs).sum(axis=0) / (n * (n - 1))


def ensemble_scores(ens: np.ndarray, truth: np.ndarray, w: np.ndarray, ref: np.ndarray | None) -> dict:
    """ens, ref: (M, H, W); truth, w: (H, W)."""
    ens, truth = ens.astype(np.float64), truth.astype(np.float64)

    def mean(x):
        return float((x * w).sum() / w.sum())
    return {"ensmean_rmse": np.sqrt(mean((ens.mean(axis=0) - truth) ** 2)),
            "member_rmse": float(np.mean([np.sqrt(mean((e - truth) ** 2)) for e in ens])),
            "spread": np.sqrt(mean(ens.var(axis=0, ddof=1))),
            "crps": mean(fair_crps(ens, truth)),
            "outside_frac": mean(((truth < ens.min(axis=0)) | (truth > ens.max(axis=0))).astype(np.float64)),
            "rmsd_vs_fp32": "" if ref is None else
            float(np.mean([np.sqrt(mean((e - r) ** 2)) for e, r in zip(ens, ref.astype(np.float64))]))}


def members(model, start) -> dict:
    """{"<var>_<lead>": (M, H, W)}."""
    out = {}
    for k in range(N_MEMBERS):
        for lead, pred in m.forecast(model, start, max(forecast.LEADS), seed=k):
            if lead in forecast.LEADS:
                for var, x in m.fields(pred).items():
                    out.setdefault(f"{var}_{lead}", []).append(x)
    return {key: np.stack(v) for key, v in out.items()}


def run(precision: str) -> None:
    model = m.load(precision)
    w = np.broadcast_to(np.clip(np.cos(np.deg2rad(m.LAT)), 0, None)[:, None], (len(m.LAT), len(m.LON)))
    member_dir = forecast.fields_dir("aurora_ens", "members_fp32")
    member_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (date, hour) in enumerate(DATES):
        start = init_time(date, hour)
        ens = members(model, start)
        path = member_dir / f"{date}_{hour}.npz"
        if precision == "fp32":
            np.savez(path, **ens)
        ref = np.load(path) if precision != "fp32" else None
        for key, e in ens.items():
            var, lead = key.rsplit("_", 1)
            truth = era5.load(start + timedelta(hours=int(lead)), var)[:len(m.LAT)]
            r = None if ref is None else ref[key]
            rows.append({"date": date, "hour": hour, "var": var, "lead": int(lead), "region": "global",
                         **ensemble_scores(e, truth, w, r)})
            for region in REGIONS:
                rr, cc = region_masks(m.LAT, m.LON, region)
                sub = np.ix_(rr, cc)
                rows.append({"date": date, "hour": hour, "var": var, "lead": int(lead), "region": region,
                             **ensemble_scores(e[(slice(None),) + sub], truth[sub], w[sub],
                                               None if r is None else r[(slice(None),) + sub])})
        print(f"[ensemble {precision} {i + 1}/{len(DATES)}] {date} {hour}h", flush=True)
    out = RESULTS / "scores" / f"aurora_ens_ensemble_{precision}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--precision", required=True, choices=("fp32", "bf16", "fp16"))
    run(p.parse_args().precision)


if __name__ == "__main__":
    main()
