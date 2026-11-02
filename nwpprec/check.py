"""Reproduction check: the first of the 52 dates, forecast to 24 h in fp32 and
fp16, compared with the stored scores of the 52-date runs.

    .venv/bin/python -m nwpprec.check --model pangu

Prints the largest relative difference in global RMSE over the verified
variables; it should be 0 (FastNet, whose forecasts are not bit-reproducible:
below 1e-3).
"""

import argparse
import csv
from datetime import timedelta

from . import era5, forecast
from .dates import init_time, weekly_dates
from .paths import RESULTS
from .scoring import scores

TOLERANCE = {"fastnet": 1e-3}  # not bit-reproducible: run-to-run differences up to ~7e-4


def stored(name: str, tag: str, date: str, hour: str) -> dict:
    with open(RESULTS / "scores" / f"{name}_{tag}.csv") as f:
        return {r["var"]: float(r["rmse"]) for r in csv.DictReader(f)
                if (r["date"], r["hour"], r["lead"], r["region"]) == (date, hour, "24", "global")}


def check(name: str) -> bool:
    m = forecast.adapter(name)
    date, hour = weekly_dates()[0]
    start = init_time(date, hour)
    ok = True
    for precision in ("fp32", "fp16"):
        model = m.load(precision)
        state = None
        for lead, state in m.forecast(model, start, 24):
            pass
        fields = forecast.with_wind_speed(m.fields(state))
        ref = stored(name, precision, date, hour)
        worst = 0.0
        for var, x in fields.items():
            rmse = scores(x, era5.load(start + timedelta(hours=24), var)[:len(m.LAT)], m.LAT, m.SKIP_NAN)["rmse"]
            worst = max(worst, abs(rmse / ref[var] - 1))
        passed = worst <= TOLERANCE.get(name, 0.0)
        ok &= passed
        print(f"{name} {precision}: largest relative RMSE difference {worst:.1e} -> {'ok' if passed else 'FAILED'}",
              flush=True)
        del model
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=forecast.MODELS)
    raise SystemExit(0 if check(p.parse_args().model) else 1)


if __name__ == "__main__":
    main()
