"""Regional error statistics and zonal spectra, accumulated during a forecast run.

For every forecast, variable and lead time, cos(lat)-weighted sums on
CELL x CELL degree cells (18 x 36 for 10 degrees):

    sse  sum w (forecast - ERA5)^2   -> RMSE of any union of cells
    se   sum w (forecast - ERA5)     -> bias
    ssd  sum w (forecast - fp32)^2   -> departure from the fp32 forecast
    sw   sum w

and, per 10-degree latitude band, the zonal power spectrum of the forecast
and of ERA5 (|rfft along longitude|^2 / n_lon^2, cos(lat)-weighted mean over
the band's rows).

Computed during a forecast run (nwpprec.forecast --regional), written to
data/results/regional/<model>_<tag>.npz. The departure needs the fp32 fields
(nwpprec.forecast --save-fields).
"""

from datetime import timedelta

import numpy as np

from . import era5, forecast
from .dates import init_time

CELL = 10
NLAT, NLON = 180 // CELL, 360 // CELL
VARIABLES = tuple(era5.VARIABLES)  # including ws10, which forecast.run adds to the fields


def land_fraction() -> np.ndarray:
    """(NLAT, NLON) cos(lat)-weighted land fraction of each cell, from the
    0.25-degree land-sea mask that comes with Aurora's checkpoint."""
    import pickle
    from .paths import CHECKPOINTS
    with open(CHECKPOINTS / "aurora-0.25-v1.5-static.pickle", "rb") as f:
        lsm = np.asarray(pickle.load(f)["lsm"], float)
    lat2d, lon2d = np.meshgrid(era5.LAT, era5.LON, indexing="ij")
    iy = np.clip(np.floor((90.0 - lat2d) / CELL).astype(int), 0, NLAT - 1)
    ix = np.floor(np.mod(lon2d, 360.0) / CELL).astype(int) % NLON
    w = np.clip(np.cos(np.deg2rad(lat2d)), 0, None)
    cell = (iy * NLON + ix).ravel()
    num = np.bincount(cell, weights=(lsm * w).ravel(), minlength=NLAT * NLON)
    den = np.bincount(cell, weights=w.ravel(), minlength=NLAT * NLON)
    return (num / den).reshape(NLAT, NLON)


class RegionalStats:
    """Callback for forecast.run: rs = RegionalStats(...); forecast.run(..., on_fields=rs)."""

    def __init__(self, name: str, dates: list, lat: np.ndarray, lon: np.ndarray, leads=forecast.LEADS):
        self.name, self.dates, self.leads = name, list(dates), tuple(leads)
        lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
        iy = np.clip(np.floor((90.0 - lat2d) / CELL).astype(int), 0, NLAT - 1)
        ix = np.floor(np.mod(lon2d, 360.0) / CELL).astype(int) % NLON
        self.cell = (iy * NLON + ix).ravel()
        self.w = np.clip(np.cos(np.deg2rad(lat2d)), 0, None).ravel()
        self.shape = lat2d.shape
        self.row_band = np.clip(np.floor((90.0 - lat) / CELL).astype(int), 0, NLAT - 1)
        self.row_w = np.clip(np.cos(np.deg2rad(lat)), 0, None)
        self.band_w = np.bincount(self.row_band, weights=self.row_w, minlength=NLAT)
        shape = (len(dates), len(VARIABLES), len(leads), NLAT, NLON)
        self.sse, self.se, self.ssd = (np.full(shape, np.nan) for _ in range(3))
        self.sw = None
        n_k = len(lon) // 2 + 1
        self.spec = np.full(shape[:3] + (NLAT, n_k), np.nan, np.float32)
        self.spec_truth = np.full_like(self.spec, np.nan)

    def _sum(self, values, weights):
        return np.bincount(self.cell, weights=weights * values, minlength=NLAT * NLON).reshape(NLAT, NLON)

    def spectrum(self, field):
        """Missing values (FastNet's interpolated fields have a gap at the
        0/360 degree seam and empty polar rows) are filled by periodic linear
        interpolation along the latitude circle; rows with no values at all are
        left out of the band average."""
        f = np.asarray(field, float).reshape(self.shape).copy()
        have = np.isfinite(f)
        for i in np.flatnonzero(~have.all(1) & have.any(1)):
            x = np.arange(f.shape[1])
            f[i, ~have[i]] = np.interp(x[~have[i]], x[have[i]], f[i, have[i]], period=f.shape[1])
        w = self.row_w * have.any(1)
        f[~have.any(1)] = 0.0
        power = np.abs(np.fft.rfft(f, axis=1)) ** 2 / self.shape[1] ** 2
        out = np.zeros((NLAT, power.shape[1]))
        np.add.at(out, self.row_band, power * w[:, None])
        band_w = np.bincount(self.row_band, weights=w, minlength=NLAT)
        return out / np.where(band_w > 0, band_w, 1)[:, None]

    def __call__(self, date, hour, lead, fields):
        di, li = self.dates.index((date, hour)), self.leads.index(lead)
        ref_path = forecast.fields_dir(self.name, "fp32") / f"{date}_{hour}.npz"
        ref = np.load(ref_path) if ref_path.exists() else None
        valid = init_time(date, hour) + timedelta(hours=lead)
        for vi, var in enumerate(VARIABLES):
            x = np.asarray(fields[var], float).ravel()
            truth = era5.load(valid, var)[:self.shape[0]]
            t = truth.astype(float).ravel()
            ok = np.isfinite(x) & np.isfinite(t)
            w = np.where(ok, self.w, 0.0)
            e = np.where(ok, x - t, 0.0)
            self.sw = self._sum(np.ones_like(w), w)
            self.sse[di, vi, li], self.se[di, vi, li] = self._sum(e * e, w), self._sum(e, w)
            if ref is not None:
                d = np.where(ok, x - np.asarray(ref[f"{var}_{lead}"], float).ravel(), 0.0)
                self.ssd[di, vi, li] = self._sum(d * d, w)
            self.spec[di, vi, li] = self.spectrum(fields[var])
            self.spec_truth[di, vi, li] = self.spectrum(truth)

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, sse=self.sse, se=self.se, ssd=self.ssd, sw=self.sw, spec=self.spec, spec_truth=self.spec_truth,
                 dates=np.array([f"{d}_{h}" for d, h in self.dates]), variables=np.array(VARIABLES),
                 leads=np.array(self.leads), cell_deg=CELL)
        print(f"Wrote {path}", flush=True)


def from_saved_fields(name: str, tag: str, dates: list) -> None:
    """Recompute data/results/regional/<name>_<tag>.npz from saved forecast
    fields (forecast --save-fields), without rerunning the model."""
    from .paths import RESULTS
    m = forecast.adapter(name)
    rs = RegionalStats(name, dates, m.LAT, m.LON)
    for date, hour in dates:
        saved = np.load(forecast.fields_dir(name, tag) / f"{date}_{hour}.npz")
        for lead in forecast.LEADS:
            rs(date, hour, lead, {var: saved[f"{var}_{lead}"] for var in VARIABLES})
    rs.save(RESULTS / "regional" / f"{name}_{tag}.npz")
