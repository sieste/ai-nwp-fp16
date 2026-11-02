"""Tracks and intensities of four named storms of 2024, in ERA5 and in forecasts.

Each storm is located in ERA5 at a reference time inside a box where it is
unambiguous and followed backwards and forwards in 6-hour steps by the
minimum of mean sea-level pressure within SEARCH_KM (per 6 hours) of its
previous position. Forecast tracks start from the ERA5 position at the
storm's first tracked lead time (Isha only becomes a separate low during 21
January) and are followed the same way (for 24-hour steps, the search radius
scales with the step). Position error, central-pressure error and the error
in the maximum 10 m wind within 300 km of the centre are relative to ERA5.

    .venv/bin/python -m nwpprec.storms --model pangu --precision fp16

Writes data/results/storms/<model>_<precision>.csv.
"""

import argparse
import csv
from datetime import datetime, timedelta

import numpy as np

from . import era5, forecast
from .dates import init_time
from .paths import RESULTS

SEARCH_KM = {"extratropical": 700.0, "tropical": 400.0}
EARTH_KM = 6371.0
# init (date, hour), track length (h), first tracked lead (h), reference time and box
# (lat min, lat max, lon min, lon max; lon -180..180), domain for cropping, kind
STORMS = {
    "isha": dict(init=("2024-01-20", "00"), hours=72, track_from=36, ref="2024-01-21T18", ref_box=(50, 66, -25, 0),
                 domain=(35, 75, -60, 30), kind="extratropical"),
    "kathleen": dict(init=("2024-04-05", "18"), hours=72, track_from=0, ref="2024-04-06T18", ref_box=(45, 62, -30, -5),
                     domain=(30, 70, -55, 20), kind="extratropical"),
    "beryl": dict(init=("2024-07-01", "00"), hours=168, track_from=0, ref="2024-07-02T00", ref_box=(10, 20, -75, -60),
                  domain=(5, 40, -110, -50), kind="tropical"),
    "milton": dict(init=("2024-10-07", "00"), hours=96, track_from=0, ref="2024-10-08T00", ref_box=(18, 27, -97, -84),
                   domain=(12, 38, -105, -65), kind="tropical"),
}


def great_circle_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    c = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(np.deg2rad(np.asarray(lon2) - np.asarray(lon1)))
    return EARTH_KM * np.arccos(np.clip(c, -1.0, 1.0))


def crop(field, lat, lon, domain):
    """(sub-field, lat, lon in -180..180) of a global field."""
    lon180 = ((np.asarray(lon, float) + 180.0) % 360.0) - 180.0
    la0, la1, lo0, lo1 = domain
    rows = np.flatnonzero((lat >= la0) & (lat <= la1))
    cols = np.flatnonzero((lon180 >= lo0) & (lon180 <= lo1))
    cols = cols[np.argsort(lon180[cols])]
    return np.asarray(field)[np.ix_(rows, cols)], np.asarray(lat, float)[rows], lon180[cols]


def find_min(msl, lat, lon, near=None, radius_km=None, box=None):
    """(lat, lon, value) of the msl minimum within radius_km of near=(lat, lon), or inside box."""
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    mask = np.isfinite(msl)
    if near is not None:
        mask &= great_circle_km(near[0], near[1], lat2d, lon2d) <= radius_km
    if box is not None:
        la0, la1, lo0, lo1 = box
        mask &= (lat2d >= la0) & (lat2d <= la1) & (lon2d >= lo0) & (lon2d <= lo1)
    if not mask.any():
        return np.nan, np.nan, np.nan
    i = np.nanargmin(np.where(mask, msl, np.inf))
    return lat2d.flat[i], lon2d.flat[i], msl.flat[i]


def max_wind_near(u, v, lat, lon, centre, radius_km=300.0):
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    mask = great_circle_km(centre[0], centre[1], lat2d, lon2d) <= radius_km
    return float(np.nanmax(np.where(mask, np.hypot(u, v), np.nan))) if mask.any() else np.nan


def follow(msl_at, start, times, name, step_hours=6):
    """{time: (lat, lon, min msl)} following the minimum from `start` through `times`."""
    out, pos = {}, start
    radius = SEARCH_KM[STORMS[name]["kind"]] * step_hours / 6.0
    for t in times:
        msl, lat, lon = msl_at(t)
        out[t] = find_min(msl, lat, lon, near=pos, radius_km=radius)
        if np.isfinite(out[t][0]):
            pos = out[t][:2]
    return out


def _era5(t, var, domain):
    return crop(era5.load(t, var), era5.LAT, era5.LON, domain)


def era5_track(name) -> dict:
    """{time: (lat, lon, min msl)} from the storm's first tracked time to the end of its track."""
    s = STORMS[name]
    start = init_time(*s["init"])
    first, last = start + timedelta(hours=s["track_from"]), start + timedelta(hours=s["hours"])
    ref = datetime.strptime(s["ref"], "%Y-%m-%dT%H")
    msl, lat, lon = _era5(ref, "msl", s["domain"])
    ref_pos = find_min(msl, lat, lon, box=s["ref_box"])
    step = timedelta(hours=6)
    back = [ref - k * step for k in range(1, int((ref - first) / step) + 1)]
    fwd = [ref + k * step for k in range(1, int((last - ref) / step) + 1)]
    msl_at = lambda t: _era5(t, "msl", s["domain"])  # noqa: E731
    track = {ref: ref_pos, **follow(msl_at, ref_pos[:2], back, name), **follow(msl_at, ref_pos[:2], fwd, name)}
    return dict(sorted(track.items()))


def run(name: str, precision: str) -> list:
    m = forecast.adapter(name)
    model = m.load(precision)
    rows = []
    for storm, s in STORMS.items():
        start = init_time(*s["init"])
        truth = era5_track(storm)
        crops = {}
        for lead, state in m.forecast(model, start, s["hours"]):
            f = m.fields(state)
            crops[start + timedelta(hours=lead)] = {v: crop(f[v], m.LAT, m.LON, s["domain"]) for v in ("msl", "10u", "10v")}
        times = [t for t in sorted(crops) if t > start + timedelta(hours=s["track_from"])]
        track = follow(lambda t: crops[t]["msl"], truth[start + timedelta(hours=s["track_from"])][:2], times,
                       storm, step_hours=m.STEP_HOURS)
        for t, (lat, lon, pmin) in track.items():
            if t not in truth:
                continue
            elat, elon, emin = truth[t]
            (u, la, lo), (v, _, _) = crops[t]["10u"], crops[t]["10v"]
            wind = max_wind_near(u, v, la, lo, (lat, lon))
            (eu, ela, elo), (ev, _, _) = _era5(t, "10u", s["domain"]), _era5(t, "10v", s["domain"])
            ewind = max_wind_near(eu, ev, ela, elo, (elat, elon))
            rows.append({"storm": storm, "lead": int((t - start).total_seconds() // 3600), "lat": lat, "lon": lon,
                         "pmin_hpa": pmin / 100, "pos_err_km": float(great_circle_km(elat, elon, lat, lon)),
                         "pmin_err_hpa": (pmin - emin) / 100, "wind_err": wind - ewind})
        print(f"[{name} {precision}] {storm}", flush=True)
    path = RESULTS / "storms" / f"{name}_{precision}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}", flush=True)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=forecast.MODELS)
    p.add_argument("--precision", required=True, choices=("fp32", "bf16", "fp16"))
    a = p.parse_args()
    run(a.model, a.precision)


if __name__ == "__main__":
    main()
