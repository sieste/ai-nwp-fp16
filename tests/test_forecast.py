"""The forecast runner with a fake model whose forecast is ERA5 plus the lead
time in days, on the real ERA5 files."""

import types
from datetime import timedelta

import numpy as np
import pytest

from nwpprec import era5, forecast


def fake_adapter():
    m = types.SimpleNamespace(LAT=era5.LAT, LON=era5.LON, SKIP_NAN=False)
    m.load = lambda precision: None
    m.forecast = lambda model, start, max_lead: ((lead, (start, lead)) for lead in range(24, max_lead + 1, 24))
    m.fields = lambda state: {v: era5.load(state[0] + timedelta(hours=state[1]), v) + state[1] / 24
                              for v in ("2t", "10u", "10v")}
    return m


@pytest.fixture
def results(tmp_path, monkeypatch):
    monkeypatch.setattr(forecast, "RESULTS", tmp_path)
    monkeypatch.setattr(forecast, "adapter", lambda name: fake_adapter())
    return tmp_path


@pytest.mark.skipif(not (era5.ERA5 / "2024-01-17-surface-level.nc").exists(),
                    reason="needs the ERA5 files of the first forecast (nwpprec.download era5)")
def test_scores_and_departure(results):
    dates = [("2024-01-07", "00")]
    forecast.run("fake", "fp32", dates, save_fields=True)
    assert (results / "fields" / "fake_fp32" / "2024-01-07_00.npz").exists()
    rows = forecast.run("fake", "fp16", dates)
    assert {r["var"] for r in rows} == {"2t", "10u", "10v", "ws10"}
    assert {r["lead"] for r in rows} == set(forecast.LEADS)
    assert {"global", "uk_europe", "north_america", "south_asia", "london", "sao_paulo"} <= {r["region"] for r in rows}
    for r in rows:
        assert r["rmse_vs_fp32"] == 0
        if r["var"] == "2t":
            assert r["rmse"] == pytest.approx(r["lead"] / 24, rel=1e-5)
    points = np.load(results / "points" / "fake_fp16" / "2024-01-07_00.npz")
    assert points["london_2t"].shape == (len(points["leads"]), 3, 3)
    assert (results / "scores" / "fake_fp16.csv").read_text().count("\n") == len(rows) + 1
