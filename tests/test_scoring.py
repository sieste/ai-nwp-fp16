import numpy as np
import pytest

from nwpprec import scoring
from nwpprec.era5 import LAT, LON


def test_constant_offset():
    truth = np.random.default_rng(0).normal(size=(721, 1440)).astype(np.float32)
    s = scoring.scores(truth + 2.0, truth, LAT)
    assert s["rmse"] == pytest.approx(2.0) and s["bias"] == pytest.approx(2.0) and s["mae"] == pytest.approx(2.0)


def test_cosine_weighting():
    # error 1 in the northern hemisphere only: weighted share of its area
    lat = np.array([60.0, 0.0, -60.0])
    err = np.array([[1.0], [0.0], [0.0]])
    w = np.cos(np.deg2rad(lat))
    assert scoring.scores(err, np.zeros_like(err), lat)["bias"] == pytest.approx(w[0] / w.sum())


def test_nan_propagates_unless_skipped():
    truth = np.zeros((721, 1440))
    pred = truth + 1.0
    pred[0, 0] = np.nan
    assert np.isnan(scoring.scores(pred, truth, LAT)["rmse"])
    assert scoring.scores(pred, truth, LAT, skip_nan=True)["rmse"] == pytest.approx(1.0)


def test_uk_europe_wraps_through_greenwich():
    rows, cols = scoring.region_masks(LAT, LON, "uk_europe")
    assert LAT[rows].min() == 35 and LAT[rows].max() == 72
    assert set(LON[cols]) == set(LON[(LON >= 345) | (LON <= 40)])


def test_fair_crps_of_two_members():
    # members 0 and 2, truth 1: mean |x - y| = 1, pair term |0 - 2| / 2 = 1, CRPS 0
    from nwpprec.ensemble import fair_crps
    assert fair_crps(np.array([[0.0], [2.0]]), np.array([1.0]))[0] == pytest.approx(0.0)


def test_paired_change():
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    s = scoring.paired_change(ref, ref * 1.1)
    assert s["pct"] == pytest.approx(10.0) and s["lo"] < 10 < s["hi"]


def test_location_boxes():
    for loc, (la, lo) in scoring.LOCATIONS.items():
        rows, cols = scoring.location_masks(LAT, LON, loc)
        assert len(rows) == len(cols) == 3
        assert abs(LAT[rows[1]] - la) <= 0.125 and abs((LON[cols[1]] - lo % 360 + 180) % 360 - 180) <= 0.125


def test_spectrum_fills_gaps():
    from nwpprec.regional import RegionalStats
    rs = RegionalStats("t", [("2024-01-07", "00")], LAT, LON)
    k = 7
    field = np.tile(np.sin(2 * np.pi * k * np.arange(1440) / 1440), (721, 1))
    gappy = field.copy()
    gappy[:, -3:] = np.nan  # a seam
    gappy[0] = np.nan  # an empty row
    a, b = rs.spectrum(field), rs.spectrum(gappy)
    assert np.argmax(b[5]) == k and b[5, k] == pytest.approx(a[5, k], rel=1e-3)
