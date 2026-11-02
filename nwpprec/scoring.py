"""Area-weighted verification scores on a regular latitude-longitude grid.

Grid points are weighted by the cosine of their latitude. Scores are
computed globally, for each region in REGIONS and for each location in
LOCATIONS (the 3 x 3 grid points nearest to it, 0.75 x 0.75 degrees).
"""

import numpy as np
from scipy import stats as sps

# region -> ((lat min, lat max), (lon min, lon max)); longitudes in -180..180
REGIONS = {"uk_europe": ((35.0, 72.0), (-15.0, 40.0)),
           "north_america": ((25.0, 60.0), (-145.0, -50.0)),  # WMO verification area
           "south_asia": ((5.0, 35.0), (60.0, 100.0))}
# location -> (lat, lon), centre of a 3 x 3 grid-point box
LOCATIONS = {"london": (51.5, -0.1), "delhi": (28.6, 77.2), "sao_paulo": (-23.55, -46.65),
             "bay_area": (37.6, -122.2), "beijing": (39.9, 116.4)}


def _weights(lat: np.ndarray, shape: tuple) -> np.ndarray:
    return np.broadcast_to(np.clip(np.cos(np.deg2rad(lat)), 0.0, None)[:, None], shape)


def scores(pred: np.ndarray, truth: np.ndarray, lat: np.ndarray, skip_nan: bool = False) -> dict:
    """RMSE, bias (pred - truth) and MAE of (H, W) fields; lat: (H,) in degrees.

    NaNs propagate (a forecast that has blown up scores NaN) unless
    skip_nan, which leaves out points where either field is NaN -- for
    fields interpolated from a grid that does not cover every point.
    """
    w = _weights(lat, pred.shape)
    err = pred.astype(np.float64) - truth.astype(np.float64)
    if skip_nan:
        valid = ~(np.isnan(pred) | np.isnan(truth))
        w = np.where(valid, w, 0.0)
        err = np.where(valid, err, 0.0)
    wsum = w.sum()
    return {"rmse": float(np.sqrt((w * err**2).sum() / wsum)),
            "bias": float((w * err).sum() / wsum),
            "mae": float((w * np.abs(err)).sum() / wsum)}


def region_masks(lat: np.ndarray, lon: np.ndarray, region: str) -> tuple:
    """(row mask, column mask) of `region`; lon in 0..360."""
    (lat0, lat1), (lon0, lon1) = REGIONS[region]
    rows = (lat >= lat0) & (lat <= lat1)
    lo, hi = lon0 % 360, lon1 % 360
    cols = (lon >= lo) & (lon <= hi) if lo <= hi else (lon >= lo) | (lon <= hi)
    return rows, cols


def location_masks(lat: np.ndarray, lon: np.ndarray, location: str) -> tuple:
    """(row indices, column indices) of the 3 x 3 grid points around `location`; lon in 0..360."""
    la, lo = LOCATIONS[location]
    i = int(np.argmin(np.abs(lat - la)))
    j = int(np.argmin(np.abs((lon - lo % 360 + 180) % 360 - 180)))
    return np.arange(i - 1, i + 2), np.arange(j - 1, j + 2) % len(lon)


def regional_scores(pred: np.ndarray, truth: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                    skip_nan: bool = False) -> dict:
    """{"global": scores, region: scores, ..., location: scores, ...}."""
    out = {"global": scores(pred, truth, lat, skip_nan)}
    for region in REGIONS:
        rows, cols = region_masks(lat, lon, region)
        out[region] = scores(pred[rows][:, cols], truth[rows][:, cols], lat[rows], skip_nan)
    for location in LOCATIONS:
        rows, cols = location_masks(lat, lon, location)
        sub = np.ix_(rows, cols)
        out[location] = scores(pred[sub], truth[sub], lat[rows], skip_nan)
    return out


def paired_change(reference: np.ndarray, other: np.ndarray) -> dict:
    """Relative change of the mean, 100 (mean(other) - mean(reference)) /
    mean(reference), with a 95% confidence interval and p-value from the
    paired differences (e.g. per-date RMSE of two configurations)."""
    d = np.asarray(other, float) - np.asarray(reference, float)
    ref = np.mean(reference)
    half = sps.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return {"pct": 100 * d.mean() / ref, "lo": 100 * (d.mean() - half) / ref,
            "hi": 100 * (d.mean() + half) / ref, "p": float(sps.ttest_rel(other, reference).pvalue)}
