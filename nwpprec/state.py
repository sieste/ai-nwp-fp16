"""Why the physical state stays in fp32.

    .venv/bin/python -m nwpprec.state table          # ERA5 rounded to bf16 / fp16
    .venv/bin/python -m nwpprec.state whole fp16     # Aurora converted as a whole
    .venv/bin/python -m nwpprec.state whole bf16 --score   # ... scored on the 52 dates

table: for ERA5 initial states of the 13 sensitivity dates, the largest
value, the share of values above fp16's largest (65,504), and the RMS error
(cos-latitude weighted) of rounding each field to bf16 and fp16 in physical
units and after normalisation (here with the field's own global mean and
standard deviation, mapped back to physical units). Writes
tex/tables/state_rounding.tex.

whole: Aurora with model.to(dtype) applied to the whole model, so that it
casts the input batch to the precision of its weights before normalising
it; prints the share of non-finite values in each output field. With
--score, the converted model's forecasts of the 52 dates are scored like the
other runs (data/results/scores/aurora_<precision>-whole.csv).
"""

import argparse

import numpy as np
import torch

from . import era5
from .dates import init_time
from .paths import TABLES
from .sensitivity import DATES

FP16_MAX = 65504.0
FIELDS = {"2t": ("surface-level", "t2m", None, "K"), "t850": ("atmospheric", "t", 850, "K"),
          "msl": ("surface-level", "msl", None, "Pa"), "sp": ("surface-level", "sp", None, "Pa"),
          "z500": ("atmospheric", "z", 500, "m$^2$\\,s$^{-2}$"), "z400": ("atmospheric", "z", 400, "m$^2$\\,s$^{-2}$"),
          "z300": ("atmospheric", "z", 300, "m$^2$\\,s$^{-2}$"), "z50": ("atmospheric", "z", 50, "m$^2$\\,s$^{-2}$")}


def _rounded(x: np.ndarray, dtype) -> np.ndarray:
    return torch.from_numpy(x).to(dtype).float().numpy()


def rounding_stats() -> dict:
    w = np.cos(np.deg2rad(era5.LAT))[:, None] * np.ones((1, len(era5.LON)))
    w /= w.sum()
    rms = lambda d, m=None: float(np.sqrt((w * d ** 2).sum() / (1 if m is None else (w * m).sum())))  # noqa: E731
    out = {}
    for var, (kind, name, level, _) in FIELDS.items():
        acc = {k: [] for k in ("max", "overflow", "bf16", "fp16", "bf16_norm", "fp16_norm")}
        for date, hour in DATES:
            x = era5.read(init_time(date, hour), kind, name, level).astype(np.float32)
            mu = (w * x).sum()
            sd = np.sqrt((w * (x - mu) ** 2).sum())
            z = (x - mu) / sd
            acc["max"].append(np.abs(x).max())
            acc["overflow"].append((np.abs(x) > FP16_MAX).mean())
            for label, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
                r = _rounded(x, dtype)
                finite = np.isfinite(r)
                acc[label].append(rms(np.where(finite, r - x, 0), finite) if finite.any() else np.nan)
                acc[f"{label}_norm"].append(rms((_rounded(z, dtype) - z) * sd))
        out[var] = {k: float(np.max(v)) if k == "max" else float(np.mean(v)) for k, v in acc.items()}
    return out


def table() -> None:
    s = rounding_stats()
    fmt = lambda v: "--" if not np.isfinite(v) else f"{v:.3g}"  # noqa: E731
    rows = []
    for var, (_, _, _, unit) in FIELDS.items():
        r = s[var]
        fp16 = fmt(r["fp16"]) if r["overflow"] < 0.5 else "overflow"
        rows.append(f"{var} ({unit}) & {r['max']:.3g} & {100 * r['overflow']:.0f} & {fmt(r['bf16'])} & {fp16} & "
                    f"{fmt(r['bf16_norm'])} & {fmt(r['fp16_norm'])} \\\\")
    tex = "\n".join([
        r"\begin{table}[tbp]", r"\centering", r"\small",
        r"\caption{Rounding the physical state to 16 bits: for ERA5 initial states (13 dates in 2024), the largest "
        r"magnitude, the share of values above fp16's largest value (65,504), and the RMS error of rounding each field "
        r"to bf16 and fp16 in physical units (fp16: over the values that stay finite) and after normalisation with the "
        r"field's global mean and standard deviation (error expressed in physical units).}",
        r"\label{tab:state-rounding}", r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r" & & & \multicolumn{2}{c}{physical units} & \multicolumn{2}{c}{normalised} \\",
        r"Field & max & \% $>$ 65{,}504 & bf16 & fp16 & bf16 & fp16 \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (TABLES / "state_rounding.tex").write_text(tex + "\n")
    for var, r in s.items():
        print(var, {k: round(v, 4) for k, v in r.items()})


def whole(precision: str, score: bool = False) -> None:
    from .models import aurora
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    cls, checkpoint = aurora.VARIANTS["deterministic"]
    model = cls(autocast=False, use_fp16_safe_attention=False)
    model.load_checkpoint_local(str(aurora.CHECKPOINTS / checkpoint))
    model = model.eval().to(dtype).to("cuda")  # no fp32 state: Aurora casts the batch to dtype
    aurora._chunk_decoder(model)
    if score:
        from . import forecast
        from .dates import weekly_dates
        forecast.run("aurora", precision, weekly_dates(), tag=f"{precision}-whole", model=model)
        return
    for date, hour in DATES[:3]:
        for lead, pred in aurora.forecast(model, init_time(date, hour), 24):
            f = aurora.fields(pred)
            print(f"{precision} {date} {hour}h lead {lead}h, non-finite:",
                  {k: f"{100 * np.mean(~np.isfinite(v)):.0f}%" for k, v in f.items()}, flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("table")
    w = sub.add_parser("whole")
    w.add_argument("precision", choices=["fp16", "bf16"])
    w.add_argument("--score", action="store_true", help="score the forecasts of the 52 dates")
    a = p.parse_args()
    table() if a.cmd == "table" else whole(a.precision, a.score)


if __name__ == "__main__":
    main()
