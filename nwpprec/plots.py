"""Figures and tables for the paper, from data/results (models whose results
are not there yet are shown as pending).

    .venv/bin/python -m nwpprec.plots

Writes figures/*.png and tex/tables/*.tex.
"""

import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

from .paths import FIGURES, RESULTS, TABLES  # noqa: E402
from .scoring import LOCATIONS, REGIONS, paired_change  # noqa: E402

MODELS = {"fastnet": "FastNet", "pangu": "Pangu-Weather", "fourcastnet": "FourCastNet", "sfno": "SFNO",
          "aurora": "Aurora"}  # by size
VARS = ("2t", "msl", "ws10", "z500", "t850")
UNITS = {"2t": "K", "msl": "Pa", "ws10": "m s$^{-1}$", "10u": "m s$^{-1}$", "10v": "m s$^{-1}$",
         "z500": "m$^2$ s$^{-2}$", "t850": "K"}
AREAS = ("global",) + tuple(REGIONS) + tuple(LOCATIONS)
AREA_NAMES = {"global": "Global", "uk_europe": "UK/Europe", "north_america": "North America", "south_asia": "South Asia",
              "london": "London", "delhi": "Delhi", "bay_area": "Bay Area", "beijing": "Beijing", "sao_paulo": "São Paulo"}
COLOURS = {"fp32": "#000000", "fp16": "#0072B2", "bf16": "#D55E00"}


def scores(model: str, tag: str) -> dict | None:
    """{(var, lead, area): {"rmse": [per date], "mae": [...], "rmse_vs_fp32": [...]}} or None."""
    path = RESULTS / "scores" / f"{model}_{tag}.csv"
    if not path.exists():
        return None
    out = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (r["var"], int(r["lead"]), r["region"])
            for k in ("rmse", "mae", "rmse_vs_fp32"):
                if r[k] != "":
                    out[key][k].append(float(r[k]))
    return out


def leads_of(s: dict) -> list:
    return sorted({k[1] for k in s})


def fp32_skill(area: str = "global") -> None:
    """fp32 RMSE against ERA5 vs lead time, one panel per variable, one line per model."""
    fig, axes = plt.subplots(1, len(VARS), figsize=(13, 2.8))
    for ax, var in zip(axes, VARS):
        for model, title in MODELS.items():
            s = scores(model, "fp32")
            if s is None:
                continue
            leads = leads_of(s)
            mean = [np.mean(s[(var, ld, area)]["rmse"]) for ld in leads]
            se = [np.std(s[(var, ld, area)]["rmse"], ddof=1) / np.sqrt(len(s[(var, ld, area)]["rmse"])) for ld in leads]
            days = np.array(leads) / 24
            ax.plot(days, mean, marker="o", markersize=3, linewidth=1, label=title)
            ax.fill_between(days, np.array(mean) - se, np.array(mean) + se, alpha=0.2)
        ax.set_title(f"{var} ({UNITS[var]})", fontsize=9)
        ax.set_xlabel("lead time (days)", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel(f"RMSE, {AREA_NAMES[area]}", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / f"fp32_skill_{area}.png", dpi=200)
    plt.close(fig)


def relative_change(other: str = "fp16", area: str = "global", metric: str = "rmse", ylim=None) -> None:
    """(other - fp32) / fp32 of the mean RMSE (or MAE) vs lead time with 95%
    confidence intervals from the paired per-date differences; one panel per
    model (columns) and variable (rows). ylim: the same y-range in every
    panel (intervals may extend beyond it)."""
    fig, axes = plt.subplots(len(VARS), len(MODELS), figsize=(2.6 * len(MODELS) + 0.6, 9), sharex=True, sharey="row")
    for j, (model, title) in enumerate(MODELS.items()):
        ref, oth = scores(model, "fp32"), scores(model, other)
        for i, var in enumerate(VARS):
            ax = axes[i, j]
            ax.axhline(0, color="grey", linewidth=0.6)
            if ref is None or oth is None:
                ax.text(0.5, 0.5, "pending", ha="center", va="center", transform=ax.transAxes, color="0.5")
            else:
                leads = leads_of(ref)
                ch = [paired_change(ref[(var, ld, area)][metric], oth[(var, ld, area)][metric]) for ld in leads]
                days = np.array(leads) / 24
                ax.errorbar(days, [c["pct"] for c in ch], yerr=[[c["pct"] - c["lo"] for c in ch], [c["hi"] - c["pct"] for c in ch]],
                            color=COLOURS[other], marker="s", markersize=3, linewidth=1, capsize=2)
            if ylim is not None:
                ax.set_ylim(*ylim)
            if i == 0:
                ax.set_title(title, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{var}: {metric.upper()} change (%)", fontsize=8)
            if i == len(VARS) - 1:
                ax.set_xlabel("lead time (days)", fontsize=8)
            ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{other}_vs_fp32_{metric}_{area}.png", dpi=200)
    plt.close(fig)


def departure_ratio(other: str = "fp16", area: str = "global") -> None:
    """Departure of `other` from the fp32 forecast relative to the fp32 RMSE, vs lead time."""
    fig, axes = plt.subplots(1, len(VARS), figsize=(13, 2.8), sharey=True)
    for ax, var in zip(axes, VARS):
        for model, title in MODELS.items():
            ref, oth = scores(model, "fp32"), scores(model, other)
            if ref is None or oth is None:
                continue
            leads = leads_of(ref)
            ratio = [np.mean(oth[(var, ld, area)]["rmse_vs_fp32"]) / np.mean(ref[(var, ld, area)]["rmse"]) for ld in leads]
            ax.plot(np.array(leads) / 24, ratio, marker="o", markersize=3, linewidth=1, label=title)
        ax.set_yscale("log")
        ax.set_title(var, fontsize=9)
        ax.set_xlabel("lead time (days)", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel(f"departure / fp32 RMSE ({other})", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{other}_departure_{area}.png", dpi=200)
    plt.close(fig)


def area_table(other: str = "fp16", days=(1, 5, 10)) -> None:
    """LaTeX table: % change in RMSE (other vs fp32) per area, variable and lead
    time, for each model; significant values (p < 0.05) in bold."""
    lines = []
    for model, title in MODELS.items():
        ref, oth = scores(model, "fp32"), scores(model, other)
        if ref is None or oth is None:
            continue
        lines.append(rf"\multicolumn{{{1 + len(VARS) * len(days)}}}{{l}}{{\textit{{{title}}}}} \\")
        for area in AREAS:
            cells = []
            for var in VARS:
                for d in days:
                    c = paired_change(ref[(var, 24 * d, area)]["rmse"], oth[(var, 24 * d, area)]["rmse"])
                    txt = f"{c['pct']:+.1f}".replace("+0.0", "0.0").replace("-0.0", "0.0")
                    cells.append(rf"\textbf{{{txt}}}" if c["p"] < 0.05 else txt)
            lines.append(f"{AREA_NAMES[area]} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    head = " & ".join(rf"\multicolumn{{{len(days)}}}{{c}}{{{v}}}" for v in VARS)
    sub = " & ".join(" & ".join(f"d{d}" for d in days) for _ in VARS)
    tex = "\n".join([
        r"\begin{table}[tbp]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
        rf"\caption{{Change in RMSE (\%) of the {other} forecasts relative to fp32, $(\mathrm{{{other}}}-\mathrm{{fp32}})/\mathrm{{fp32}}$, "
        rf"at days {', '.join(map(str, days))}, per area; 52 dates in 2024. Bold: significant at the 5\% level (paired $t$-test).}}",
        rf"\label{{tab:{other}-areas}}", r"\begin{tabular}{l" + "r" * (len(VARS) * len(days)) + "}", r"\toprule",
        f" & {head} \\\\", f" & {sub} \\\\", r"\midrule", *lines[:-1], r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (TABLES / f"{other}_areas.tex").write_text(tex + "\n")


def power_table(other: str = "fp16") -> None:
    """LaTeX table of the smallest change in mean RMSE (% of fp32) that the
    two-sided paired t-test detects with probability 0.8 at the 5% level,
    (t_0.975 + t_0.8) s / sqrt(N), from the observed standard deviation s of
    the per-date differences; median and range over variables and lead times."""
    from scipy import stats
    areas = ("global",) + tuple(REGIONS)
    rows = []
    for model, title in MODELS.items():
        ref, oth = scores(model, "fp32"), scores(model, other)
        if ref is None or oth is None:
            continue
        cells = []
        for area in areas:
            mde = []
            for var in VARS:
                for ld in leads_of(ref):
                    a, b = np.array(ref[(var, ld, area)]["rmse"]), np.array(oth[(var, ld, area)]["rmse"])
                    n = len(a)
                    s = (b - a).std(ddof=1)
                    mde.append(100 * (stats.t.ppf(0.975, n - 1) + stats.t.ppf(0.8, n - 1)) * s / np.sqrt(n) / a.mean())
            med = np.median(mde)
            cells.append(f"{med:.2f} [{min(mde):.2f}, {max(mde):.1f}]" if area == "global" else f"{med:.2f}")
        rows.append(f"{title} & {n} & " + " & ".join(cells) + r" \\")
    tex = "\n".join([
        r"\begin{table}[htbp]", r"\centering", r"\small",
        rf"\caption{{Smallest change in mean RMSE, as \% of the fp32 RMSE, that the paired $t$-test detects with "
        rf"probability 0.8 at the 5\% level, given the $N$ dates and the observed date-to-date variability of the "
        rf"{other}$-$fp32 differences. Median over the 50 combinations of variable and lead time per area (global: "
        rf"with range).}}",
        r"\label{tab:power}", r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Model & $N$ & Global & UK/Europe & N.\ America & S.\ Asia \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (TABLES / "power.tex").write_text(tex + "\n")


def eightbit_table(days=(1, 5, 10), variables=("z500", "2t")) -> None:
    """LaTeX table: % change in global RMSE relative to fp32 of fp16 and the
    8-bit configurations (nwpprec.eightbit) on the 13 sensitivity dates."""
    from .eightbit import CONFIGS
    from .sensitivity import DATES
    keep = {(d, h) for d, h in DATES}
    names = {"fp16": "fp16", "fp16-w-int8": "int8 weights", "fp16-w-fp8": "fp8 weights",
             "fp16-w8a8-int8": "int8 weights + inputs", "fp16-w8a8-fp8": "fp8 weights + inputs"}

    def global_rmse(model, tag):
        path = RESULTS / "scores" / f"{model}_{tag}.csv"
        if not path.exists():
            return None
        out = defaultdict(list)
        with open(path) as f:
            for r in csv.DictReader(f):
                if r["region"] == "global" and (r["date"], r["hour"]) in keep:
                    out[(r["var"], int(r["lead"]))].append(float(r["rmse"]))
        return out
    lines = []
    for model, title in MODELS.items():
        ref = global_rmse(model, "fp32")
        if ref is None:
            continue
        for k, tag in enumerate(["fp16"] + [f"fp16-{c}" for c in CONFIGS]):
            oth = global_rmse(model, tag)
            if oth is None:
                continue
            cells = []
            for var in variables:
                for d in days:
                    c = paired_change(ref[(var, 24 * d)], oth[(var, 24 * d)])
                    txt = f"{c['pct']:+.1f}" if abs(c["pct"]) < 100 else f"{c['pct']:+.0f}"
                    cells.append(rf"\textbf{{{txt}}}" if c["p"] < 0.05 else txt)
            lines.append(f"{title if k == 0 else ''} & {names[tag]} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    head = " & ".join(rf"\multicolumn{{{len(days)}}}{{c}}{{{v}}}" for v in variables)
    sub = " & ".join(" & ".join(f"d{d}" for d in days) for _ in variables)
    tex = "\n".join([
        r"\begin{table}[tbp]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Change in global RMSE (\%) relative to fp32 of the fp16 network and of 8-bit quantisation "
        r"simulated inside it (weights only, or weights and the inputs of every linear and convolution layer; "
        r"\S\ref{sec:methods-further}), on the 13 sensitivity dates. Bold: significant at the 5\% level "
        r"(paired $t$-test).}",
        r"\label{tab:eightbit}", r"\begin{tabular}{ll" + "r" * (len(variables) * len(days)) + "}", r"\toprule",
        f" & & {head} \\\\", f"Model & Configuration & {sub} \\\\", r"\midrule",
        *lines[:-1], r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (TABLES / "eightbit.tex").write_text(tex + "\n")


def bf16_table(days=(1, 3, 5, 10), variables=("z500", "2t", "msl")) -> None:
    """LaTeX table: % change in global RMSE relative to fp32 of the bf16 and
    fp16 networks on the 52 dates."""
    lines = []
    for model, title in MODELS.items():
        ref = scores(model, "fp32")
        if ref is None:
            continue
        for k, tag in enumerate(("bf16", "fp16")):
            oth = scores(model, tag)
            if oth is None:
                continue
            cells = []
            for var in variables:
                for d in days:
                    c = paired_change(ref[(var, 24 * d, "global")]["rmse"], oth[(var, 24 * d, "global")]["rmse"])
                    txt = f"{c['pct']:+.1f}".replace("+0.0", "0.0").replace("-0.0", "0.0")
                    cells.append(rf"\textbf{{{txt}}}" if c["p"] < 0.05 else txt)
            lines.append(f"{title if k == 0 else ''} & {tag} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    head = " & ".join(rf"\multicolumn{{{len(days)}}}{{c}}{{{v}}}" for v in variables)
    sub = " & ".join(" & ".join(f"d{d}" for d in days) for _ in variables)
    tex = "\n".join([
        r"\begin{table}[htbp]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Change in global RMSE (\%) relative to fp32 with the network in bf16 and in fp16 (state in fp32 in "
        r"both), 52 dates in 2024. Bold: significant at the 5\% level (paired $t$-test).}",
        r"\label{tab:bf16}", r"\begin{tabular}{ll" + "r" * (len(variables) * len(days)) + "}", r"\toprule",
        f" & & {head} \\\\", f"Model & Precision & {sub} \\\\", r"\midrule",
        *lines[:-1], r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (TABLES / "bf16.tex").write_text(tex + "\n")


BANDS = (("90-60S", lambda c: c < -60), ("60-20S", lambda c: (c >= -60) & (c < -20)), ("20S-20N", lambda c: np.abs(c) <= 20),
         ("20-60N", lambda c: (c > 20) & (c <= 60)), ("60-90N", lambda c: c > 60))
CELL_LAT = 90 - 5 - 10 * np.arange(18)  # centres of the 10-degree cell rows


def regional(model: str, tag: str):
    path = RESULTS / "regional" / f"{model}_{tag}.npz"
    return np.load(path) if path.exists() else None


def band_change(other: str = "fp16", days=(1, 5, 10)) -> None:
    """% change in RMSE (other vs fp32) per latitude band; rows: variables, columns: models."""
    fig, axes = plt.subplots(len(VARS), len(MODELS), figsize=(2.6 * len(MODELS) + 0.6, 9), sharex=True, sharey="row")
    x = np.arange(len(BANDS))
    for j, (model, title) in enumerate(MODELS.items()):
        a, b = regional(model, "fp32"), regional(model, other)
        for i, var in enumerate(VARS):
            ax = axes[i, j]
            ax.axhline(0, color="grey", linewidth=0.6)
            if a is None or b is None:
                ax.text(0.5, 0.5, "pending", ha="center", va="center", transform=ax.transAxes, color="0.5")
            else:
                vi = list(a["variables"]).index(var)
                for d, colour in zip(days, ("#56B4E9", "#E69F00", "#000000")):
                    li = list(a["leads"]).index(24 * d)
                    pct = []
                    for _, rows in BANDS:
                        r = rows(CELL_LAT)
                        sw = a["sw"][r].sum()
                        ra = np.sqrt(a["sse"][:, vi, li][:, r].sum((1, 2)) / sw)
                        rb = np.sqrt(b["sse"][:, vi, li][:, r].sum((1, 2)) / sw)
                        pct.append(100 * (rb.mean() / ra.mean() - 1))
                    ax.plot(x, pct, color=colour, marker="s", markersize=3, linewidth=1, label=f"day {d}")
            if i == 0:
                ax.set_title(title, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{var}: RMSE change (%)", fontsize=8)
            ax.set_xticks(x, [n for n, _ in BANDS], rotation=45, fontsize=7)
            ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{other}_bands.png", dpi=200)
    plt.close(fig)


def whole_model_figure(model: str = "aurora", area: str = "global") -> None:
    """RMSE against ERA5 vs lead time of the fp32 forecasts, of the network in
    bf16 with the state in fp32, and of the whole model converted to bf16
    (nwpprec.state whole bf16 --score); one panel per variable."""
    runs = {"fp32": ("fp32", "black", "-"), "bf16, state in fp32": ("bf16", "#0072B2", "-"),
            "bf16, whole model": ("bf16-whole", "#D55E00", "--")}
    data = {label: scores(model, tag) for label, (tag, _, _) in runs.items()}
    if any(d is None for d in data.values()):
        return
    fig, axes = plt.subplots(1, len(VARS), figsize=(13, 2.9))
    for ax, var in zip(axes, VARS):
        for label, (_, colour, style) in runs.items():
            s = data[label]
            leads = leads_of(s)
            ax.plot(np.array(leads) / 24, [np.mean(s[(var, ld, area)]["rmse"]) for ld in leads], color=colour,
                    linestyle=style, marker="o", markersize=2.5, linewidth=1.2, label=label)
        ax.set_title(f"{var} ({UNITS[var]})", fontsize=9)
        ax.set_xlabel("lead time (days)", fontsize=8)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=7)
        ax.grid(color="0.92", linewidth=0.5)
    axes[0].set_ylabel(f"RMSE, {AREA_NAMES[area]}", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{model}_bf16_whole_{area}.png", dpi=200)
    plt.close(fig)


def land_sea(other: str = "fp16", days=(1, 5, 10), variables=("2t", "msl", "ws10", "z500", "t850")) -> None:
    """Prints the change in RMSE (other vs fp32, %) over mostly-land cells
    (land fraction >= 50%) and almost entirely sea cells (< 5%), and the fp32
    2t RMSE over both."""
    from .regional import land_fraction
    land = land_fraction()
    masks = {"land": land >= 0.5, "sea": land < 0.05}
    print(f"cells: land {masks['land'].sum()}, sea {masks['sea'].sum()}")
    for model in MODELS:
        a, b = regional(model, "fp32"), regional(model, other)
        if a is None or b is None:
            continue
        for var in variables:
            vi = list(a["variables"]).index(var)
            out = []
            for name, mask in masks.items():
                sw = (a["sw"] * mask).sum()
                cells = []
                for d in days:
                    li = list(a["leads"]).index(24 * d)
                    ra, rb = (np.sqrt((x["sse"][:, vi, li] * mask).sum((1, 2)) / sw) for x in (a, b))
                    c = paired_change(ra, rb)
                    cells.append(f"{c['pct']:+.2f}{'*' if c['p'] < 0.05 else ''}")
                    if var == "2t" and d == days[0]:
                        cells[-1] += f" (fp32 RMSE {ra.mean():.2f})"
                out.append(f"{name} " + " / ".join(cells))
            print(f"{model:11s} {var:5s} " + " | ".join(out))


def cell_map(var: str = "z500", day: int = 5, other: str = "fp16", lim: float = 5.0) -> None:
    """Map of the change in RMSE (other vs fp32, %) on the 10-degree cells, one
    panel per model; dots: cells where the paired t-test across dates is
    significant at the 5% level."""
    import cartopy.crs as ccrs
    from scipy import stats
    fig, grid = plt.subplots(3, 2, figsize=(10, 7.6),
                             subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=180)))
    axes = grid.flat
    lon_edges, lat_edges = np.arange(0, 361, 10), np.arange(90, -91, -10)
    mesh = None
    for ax, (model, title) in zip(axes, MODELS.items()):
        a, b = regional(model, "fp32"), regional(model, other)
        ax.coastlines(linewidth=0.4, color="0.3")
        ax.set_title(title, fontsize=9, loc="left")
        if a is None or b is None:
            ax.text(0.5, 0.5, "pending", ha="center", va="center", transform=ax.transAxes, color="0.5")
            continue
        vi, li = list(a["variables"]).index(var), list(a["leads"]).index(24 * day)
        sw = np.where(a["sw"] > 0, a["sw"], np.nan)
        ra, rb = (np.sqrt(x["sse"][:, vi, li] / sw) for x in (a, b))
        pct = 100 * (np.nanmean(rb, 0) / np.nanmean(ra, 0) - 1)
        p = stats.ttest_rel(rb, ra, axis=0).pvalue
        mesh = ax.pcolormesh(lon_edges, lat_edges, pct, cmap="RdBu_r", vmin=-lim, vmax=lim,
                             transform=ccrs.PlateCarree())
        i, j = np.nonzero(p < 0.05)
        ax.plot(5 + 10 * j, 85 - 10 * i, linestyle="none", marker=".", markersize=1.8, color="black",
                transform=ccrs.PlateCarree())
    spare = grid.flat[len(MODELS)]  # the sixth panel holds the colour bar
    spare.set_visible(False)
    pos = spare.get_position()
    cax = fig.add_axes([pos.x0 + 0.1 * pos.width, pos.y0 + 0.45 * pos.height, 0.8 * pos.width, 0.08 * pos.height])
    cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="both")
    cbar.set_label(f"{var} RMSE change at day {day},\n{other} vs fp32 (%)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.savefig(FIGURES / f"{other}_cells_{var}_day{day}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# Largest zonal wavenumber each model resolves. Aurora and Pangu-Weather: just
# below 1440 longitudes / (2 x 4-point patches), where their fp32 spectra have
# a spike from the patch structure. FourCastNet: 45, the largest zonal
# wavenumber its spectral filters act on (they keep 46 of the 91 zonal modes
# of its 90 x 180 patch grid; larger wavenumbers bypass them). FastNet: its
# grid spacing at the equator (about 0.9 degrees); above it, our
# interpolation to 0.25 degrees. SFNO: 239, the largest zonal wavenumber of
# its internal 480-longitude grid and spectral filters.
CUTOFF = {"fourcastnet": 45, "aurora": 179, "pangu": 179, "fastnet": 200, "sfno": 239}
FULL_MAX = 700  # full spectra: leave out the last wavenumbers, waves of about two grid points


def spectra(day: int = 5, other: str = "fp16", full: bool = False) -> None:
    """Zonal power spectra at `day`, mean over the 52 dates and the latitude
    band: ERA5 (black), each model's fp32 forecast (line) and `other` forecast
    (dots, same colour), up to each model's cutoff wavenumber (CUTOFF).
    z500 in geopotential metres, 2t in K. full: wavenumbers up to FULL_MAX
    (FastNet still up to its cutoff), with each model's cutoff (dotted) and
    the patch wavenumbers 180, 360, 540 (dashed) marked (*_spectra_full.png)."""
    bands = (("30-60N", [3, 4, 5]), ("20S-20N", [7, 8, 9, 10]))
    scale = {"z500": 1 / 9.80665 ** 2, "2t": 1.0}  # power of z/g in gpm^2
    units = {"z500": "gpm$^2$", "2t": "K$^2$"}
    colours = dict(zip(MODELS, ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7")))
    fig, axes = plt.subplots(2, len(bands), figsize=(9.5, 6.5), sharex=True)
    for i, var in enumerate(("z500", "2t")):
        for j, (band, rows) in enumerate(bands):
            ax = axes[i, j]
            truth_done = False
            for model, title in MODELS.items():
                a, b = regional(model, "fp32"), regional(model, other)
                if a is None or b is None:
                    continue
                vi, li = list(a["variables"]).index(var), list(a["leads"]).index(24 * day)
                pa, pb = (np.nanmean(x["spec"][:, vi, li][:, rows], axis=(0, 1)) * scale[var] for x in (a, b))
                if not truth_done:
                    pt = np.nanmean(a["spec_truth"][:, vi, li][:, rows], axis=(0, 1)) * scale[var]
                    k = np.arange(1, 640)  # ERA5 has no variance above wavenumber 639
                    ax.plot(k, pt[k], color="black", linewidth=1.2, label="ERA5")
                    truth_done = True
                kmax = FULL_MAX if full and model != "fastnet" else CUTOFF[model]  # FastNet above its cutoff: our interpolation
                k = np.arange(1, kmax + 1)
                ax.plot(k, pa[k], color=colours[model], linewidth=1, label=f"{title} fp32")
                if full and model != "fastnet":
                    ax.axvline(CUTOFF[model], color=colours[model], linestyle=":", linewidth=0.8)
                kd = np.unique(np.round(np.logspace(0, np.log10(kmax), 30)).astype(int))
                ax.plot(kd, pb[kd], linestyle="none", marker="o", markersize=2.5, color=colours[model],
                        label=f"{title} {other}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(color="0.92", linewidth=0.5)
            if full:
                for kp in (180, 360, 540):  # patch wavenumbers: 1440/8 and multiples
                    ax.axvline(kp, color="0.6", linewidth=0.6, linestyle="--", zorder=0)
            if i == 0:
                ax.set_title(band, fontsize=9)
            if i == 1:
                ax.set_xlabel("zonal wavenumber", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{var} power ({units[var]})", fontsize=8)
            ax.tick_params(labelsize=7)
    # one legend column per model (fp32 line above, fp16 dots below), ERA5 alone in the last column;
    # matplotlib fills legend columns top to bottom
    handles = []
    for model, title in MODELS.items():
        handles += [plt.Line2D([], [], color=colours[model], linewidth=1, label=f"{title} fp32"),
                    plt.Line2D([], [], color=colours[model], linestyle="none", marker="o", markersize=2.5,
                               label=f"{title} {other}")]
    handles += [plt.Line2D([], [], color="black", linewidth=1.2, label="ERA5"),
                plt.Line2D([], [], linestyle="none", label=" ")]
    fig.legend(handles=handles, loc="lower center", ncol=len(MODELS) + 1, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIGURES / f"{other}_spectra{'_full' if full else ''}.png", dpi=200)
    plt.close(fig)


def lead_time_shift(ref: dict, oth: dict, var: str, area: str) -> tuple:
    """(lead days, shift in hours, low, high): the change in mean RMSE expressed
    as the lead time over which the fp32 RMSE grows by the same amount,
    shift = (RMSE_other - RMSE_fp32) / (dRMSE_fp32 / dt), with dRMSE/dt from
    central differences of the fp32 RMSE between daily lead times (one-sided
    at days 1 and 10); the 95% interval is that of the paired change."""
    leads = leads_of(ref)
    rmse = np.array([np.mean(ref[(var, ld, area)]["rmse"]) for ld in leads])
    slope = np.gradient(rmse, np.array(leads, float))  # per hour
    shift, lo, hi = [], [], []
    for ld, r, g in zip(leads, rmse, slope):
        c = paired_change(ref[(var, ld, area)]["rmse"], oth[(var, ld, area)]["rmse"])
        shift.append(c["pct"] / 100 * r / g)
        lo.append(c["lo"] / 100 * r / g)
        hi.append(c["hi"] / 100 * r / g)
    return np.array(leads) / 24, np.array(shift), np.array(lo), np.array(hi)


def lead_time_figure(other: str = "fp16", area: str = "global") -> None:
    """Change in RMSE as an equivalent change in lead time (hours; positive: other is worse)."""
    fig, axes = plt.subplots(1, len(VARS), figsize=(13, 2.9), sharey=True)
    for ax, var in zip(axes, VARS):
        ax.axhline(0, color="grey", linewidth=0.6)
        for model, title in MODELS.items():
            ref, oth = scores(model, "fp32"), scores(model, other)
            if ref is None or oth is None:
                continue
            days, sh, lo, hi = lead_time_shift(ref, oth, var, area)
            ax.errorbar(days, sh, yerr=[sh - lo, hi - sh], marker="o", markersize=3, linewidth=1, capsize=2, label=title)
        ax.set_title(var, fontsize=9)
        ax.set_xlabel("lead time (days)", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel(f"equivalent lead-time loss (h), {other}", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{other}_lead_time_{area}.png", dpi=200)
    plt.close(fig)


def headroom_figure() -> None:
    """Largest |activation| of the fp16 network per model step, against fp16's largest value."""
    import json
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for model, title in MODELS.items():
        path = RESULTS / "headroom" / f"{model}.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        steps = np.arange(1, len(d["step_max_abs"]) + 1)
        hours = steps * 240 / len(steps)
        ax.plot(hours / 24, d["step_max_abs"], marker="o", markersize=2.5, linewidth=1,
                label=f"{title} ({d['margin']:.0f}x)")
    ax.axhline(65504, color="#D55E00", linewidth=1, linestyle="--")
    ax.text(0.2, 65504 * 0.75, "largest fp16 value (65,504)", fontsize=7, color="#D55E00", va="top")
    ax.set_yscale("log")
    ax.set_xlabel("lead time (days)", fontsize=8)
    ax.set_ylabel("largest |activation|", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False, loc="upper right", bbox_to_anchor=(1, 0.9))
    fig.tight_layout()
    fig.savefig(FIGURES / "fp16_headroom.png", dpi=200)
    plt.close(fig)


def storm_figure(other: str = "fp16") -> None:
    """Tracks of the four storms: ERA5 (black), fp32 (solid) and `other` (dashed); rows storms, columns models."""
    import cartopy.crs as ccrs
    from .storms import STORMS, era5_track
    fig, axes = plt.subplots(len(STORMS), len(MODELS), figsize=(2.6 * len(MODELS) + 0.6, 7.2), subplot_kw=dict(projection=ccrs.PlateCarree()))
    for i, (storm, spec) in enumerate(STORMS.items()):
        la0, la1, lo0, lo1 = spec["domain"]
        e = np.array(list(era5_track(storm).values()))
        for j, (model, title) in enumerate(MODELS.items()):
            ax = axes[i, j]
            ax.set_extent([lo0, lo1, la0, la1], crs=ccrs.PlateCarree())
            ax.coastlines(linewidth=0.4, color="0.5")
            ax.plot(e[:, 1], e[:, 0], color="black", linewidth=1.4, marker="o", markersize=1.5, transform=ccrs.PlateCarree())
            for tag, style in (("fp32", dict(color="#0072B2", linestyle="-")), (other, dict(color=COLOURS[other] if other != "fp16" else "#D55E00", linestyle="--"))):
                path = RESULTS / "storms" / f"{model}_{tag}.csv"
                if not path.exists():
                    continue
                rows = [r for r in csv.DictReader(open(path)) if r["storm"] == storm]
                ax.plot([float(r["lon"]) for r in rows], [float(r["lat"]) for r in rows], linewidth=1.1,
                        transform=ccrs.PlateCarree(), **style)
            if i == 0:
                ax.set_title(title, fontsize=9)
            if j == 0:
                ax.text(-0.04, 0.5, storm.replace("_", " ").title(), transform=ax.transAxes, rotation=90,
                        ha="right", va="center", fontsize=9)
    handles = [plt.Line2D([], [], color="black", linewidth=1.4, label="ERA5"),
               plt.Line2D([], [], color="#0072B2", label="fp32"), plt.Line2D([], [], color="#D55E00", linestyle="--", label=other)]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGURES / f"storm_tracks_{other}.png", dpi=200)
    plt.close(fig)


def location_series(loc: str = "london", date: str = "2024-01-20", hour: str = "00",
                    variables=("msl", "ws10", "2t")) -> None:
    """Forecasts at one location from one initial date: ERA5 (black), fp32
    (solid) and fp16 (dashed); rows: models, columns: variables. Mean over
    the location's 3 x 3 grid points (for FastNet, over those its
    interpolated fields cover, and the same for ERA5)."""
    from datetime import timedelta
    from . import era5
    from .dates import init_time
    from .scoring import location_masks
    start = init_time(date, hour)
    scale = {"msl": 0.01}  # Pa -> hPa
    units = {"msl": "hPa", "ws10": "m s$^{-1}$", "2t": "K"}
    styles = {"fp32": dict(color="#0072B2", linestyle="-"), "fp16": dict(color="#D55E00", linestyle="--")}
    fig, axes = plt.subplots(len(MODELS), len(variables), figsize=(10, 2.0 * len(MODELS)), sharex=True, sharey="col")
    box = np.ix_(*location_masks(era5.LAT, era5.LON, loc))
    for i, (model, title) in enumerate(MODELS.items()):
        runs = {tag: RESULTS / "points" / f"{model}_{tag}" / f"{date}_{hour}.npz" for tag in styles}
        for j, var in enumerate(variables):
            ax = axes[i, j]
            if not all(path.exists() for path in runs.values()):
                ax.text(0.5, 0.5, "pending", ha="center", va="center", transform=ax.transAxes, color="0.5")
                continue
            data = {tag: np.load(path) for tag, path in runs.items()}
            leads = data["fp32"]["leads"]
            valid = ~np.isnan(data["fp32"][f"{loc}_{var}"][0])
            truth = [era5.load(start + timedelta(hours=int(ld)), var)[box][valid].mean() for ld in leads]
            ax.plot(leads / 24, np.array(truth) * scale.get(var, 1), color="black", linewidth=1.0)
            for tag, d in data.items():
                ax.plot(d["leads"] / 24, d[f"{loc}_{var}"][:, valid].mean(1) * scale.get(var, 1), linewidth=1.2,
                        **styles[tag])
            ax.grid(color="0.92", linewidth=0.5)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(f"{var} ({units[var]})", fontsize=9)
            if i == len(MODELS) - 1:
                ax.set_xlabel("lead time (days)", fontsize=8)
            if j == 0:
                ax.set_ylabel(title, fontsize=9)
    handles = [plt.Line2D([], [], color="black", linewidth=1.0, label="ERA5")]
    handles += [plt.Line2D([], [], linewidth=1.2, label=tag, **style) for tag, style in styles.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGURES / f"location_{loc}_{date}_{hour}.png", dpi=200)
    plt.close(fig)


def model_skill(area: str = "global") -> None:
    """Figure 1: fp32 RMSE against ERA5 vs lead time (daily), one panel per
    variable, one line per model, shading +-1 standard error across dates."""
    variables = ("2t", "msl", "10u", "10v", "z500", "t850")
    style = {"aurora": ("#2F6FD0", "o"), "pangu": ("#E2653A", "D"), "fourcastnet": ("#1BA672", "^"),
             "fastnet": ("#F0A202", "v"), "sfno": ("#8E44AD", "s")}
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.9), sharex=True)
    for ax, var in zip(axes.flat, variables):
        for model, title in MODELS.items():
            s = scores(model, "fp32")
            if s is None:
                continue
            leads = leads_of(s)
            vals = [s[(var, ld, area)]["rmse"] for ld in leads]
            mean = np.array([np.mean(v) for v in vals])
            se = np.array([np.std(v, ddof=1) / np.sqrt(len(v)) for v in vals])
            days = np.array(leads) / 24
            colour, marker = style[model]
            ax.plot(days, mean, color=colour, marker=marker, markersize=4, markeredgecolor="white",
                    markeredgewidth=0.6, linewidth=1.6, label=title)
            ax.fill_between(days, mean - se, mean + se, color=colour, alpha=0.15, linewidth=0)
        ax.set_title(f"{var} ({UNITS[var]})", fontsize=11)
        ax.set_ylim(bottom=0)
        ax.set_xticks(range(1, 11))
        ax.grid(color="0.9", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("lead time (days)", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"RMSE, {AREA_NAMES[area]}", fontsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=10, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):  # the paper includes the pdf
        fig.savefig(FIGURES / f"model_comparison_rmse_{area}.{ext}", dpi=200)
    plt.close(fig)


def ensemble_scores(tag: str) -> dict | None:
    path = RESULTS / "scores" / f"aurora_ens_ensemble_{tag}.csv"
    if not path.exists():
        return None
    out = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in ("ensmean_rmse", "spread", "crps", "outside_frac", "rmsd_vs_fp32"):
                if r[k] != "":
                    out[(r["var"], int(r["lead"]), r["region"])][k].append(float(r[k]))
    return out


def ensemble_figure(other: str = "fp16", area: str = "global") -> None:
    """Aurora ensemble: change in CRPS and spread (other vs fp32) with 95%
    intervals, and the departure of `other` members from same-seed fp32
    members relative to the fp32 spread; one column per variable."""
    a, b = ensemble_scores("fp32"), ensemble_scores(other)
    if a is None or b is None:
        return
    variables = [v for v in ("2t", "msl", "z500", "t850") if (v, 24, area) in a]
    leads = sorted({k[1] for k in a})
    days = np.array(leads) / 24
    fig, axes = plt.subplots(2, len(variables), figsize=(11, 5), sharex=True, sharey="row")
    for j, var in enumerate(variables):
        ax = axes[0, j]
        ax.axhline(0, color="grey", linewidth=0.6)
        for metric, colour, marker in (("crps", "#0072B2", "s"), ("spread", "#E69F00", "o")):
            ch = [paired_change(a[(var, ld, area)][metric], b[(var, ld, area)][metric]) for ld in leads]
            ax.errorbar(days, [c["pct"] for c in ch], yerr=[[c["pct"] - c["lo"] for c in ch], [c["hi"] - c["pct"] for c in ch]],
                        color=colour, marker=marker, markersize=3, linewidth=1, capsize=2, label=metric.upper() if metric == "crps" else metric)
        ax.set_title(var, fontsize=9)
        ax.tick_params(labelsize=7)
        ax2 = axes[1, j]
        ratio = [np.mean(b[(var, ld, area)]["rmsd_vs_fp32"]) / np.mean(a[(var, ld, area)]["spread"]) for ld in leads]
        ax2.plot(days, ratio, color="#009E73", marker="o", markersize=3, linewidth=1)
        ax2.set_xlabel("lead time (days)", fontsize=8)
        ax2.tick_params(labelsize=7)
    axes[0, 0].set_ylabel(f"change {other} vs fp32 (%)", fontsize=8)
    axes[1, 0].set_ylabel(f"{other} - fp32 member difference\n/ fp32 spread", fontsize=8)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{other}_ensemble.png", dpi=200)
    plt.close(fig)


def cost_table() -> None:
    """LaTeX table of time per step, time and energy per 10-day forecast and
    peak memory in fp32 and fp16, from data/results/benchmark.json; the
    largest bf16 - fp16 differences are given in the caption."""
    import json
    path = RESULTS / "benchmark.json"
    if not path.exists():
        return
    d = json.loads(path.read_text())
    rows = []
    diff = {k: 0.0 for k in ("s_per_step", "peak_gb", "kj_per_10_days")}  # largest |bf16 / fp16 - 1|
    for model, title in MODELS.items():
        if model not in d:
            continue
        r32 = d[model]["fp32"]
        if "bf16" in d[model]:
            for k in diff:
                diff[k] = max(diff[k], abs(d[model]["bf16"][k] / d[model]["fp16"][k] - 1))
        label = title + (r"$^\dagger$" if model == "aurora" else "")  # footnote: weight streaming
        for prec in ("fp32", "fp16"):
            r = d[model][prec]
            rows.append(f"{label if prec == 'fp32' else ''} & {prec} & {r['s_per_step']:.3f} & "
                        f"{r32['s_per_step'] / r['s_per_step']:.1f} & {r['s_per_10_days']:.0f} & {r['peak_gb']:.1f} & "
                        f"{r['kj_per_10_days'] / 3.6:.2f} \\\\")  # kJ -> Wh
        rows.append(r"\midrule")
    tex = "\n".join([
        r"\begin{table}[tbp]", r"\centering", r"\small",
        r"\caption{Inference cost on one RTX 5080: mean time per model step (first step of each forecast untimed, "
        r"at least 60\,s of timed steps; FourCastNet, FastNet, SFNO and Aurora step 6\,h, Pangu-Weather 24\,h), speed-up over "
        r"fp32, time for a 10-day forecast, peak allocated GPU memory, and energy for a 10-day forecast (mean GPU power "
        r"within the timed steps, sampled every 50\,ms, times their duration). fp16: network in fp16, state in fp32. With the "
        rf"network in bf16 instead, time per step, peak memory and energy differ from fp16 by at most "
        rf"{100 * diff['s_per_step']:.0f}\%, {100 * diff['peak_gb']:.0f}\% and {100 * diff['kj_per_10_days']:.1f}\%.}}",
        r"\label{tab:benchmark}", r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"Model & Precision & s/step & speed-up & s/10\,d & GB & Wh/10\,d \\", r"\midrule",
        *rows[:-1], r"\bottomrule", r"\end{tabular}\\[2pt]",
        r"{\scriptsize $^\dagger$~In fp32, Aurora fits in 16\,GB only with its backbone weights streamed from CPU "
        r"memory one block at a time. This slows its fp32 runs and keeps their peak memory down, so that its speed-up "
        r"is partly due to the streaming, its memory saving is understated, and its energy comparison may be affected.}",
        r"\end{table}"])
    (TABLES / "benchmark.tex").write_text(tex + "\n")


def _sensitivity(model: str, exp: str, lead: int = 24) -> dict | None:
    """{(config, var): [departure per date]} at `lead`, or None."""
    path = RESULTS / "sensitivity" / f"{model}_{exp}.csv"
    if not path.exists():
        return None
    out = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            if int(r["lead"]) == lead:
                out[(r["config"], r["var"])].append(float(r["departure"]))
    return out


def sensitivity_oat_figure(var: str = "z500", lead: int = 24) -> None:
    """Departure from fp32 at `lead` with one component at a time in fp16,
    relative to the whole network in fp16; one row per model, components in
    execution order."""
    fig, axes = plt.subplots(len(MODELS), 1, figsize=(10, 2.25 * len(MODELS)))
    for ax, (model, title) in zip(axes, MODELS.items()):
        d = _sensitivity(model, "oat", lead)
        ax.axhline(1, color="grey", linewidth=0.6)
        ax.set_title(title, fontsize=9, loc="left")
        if d is None:
            ax.text(0.5, 0.5, "pending", ha="center", va="center", transform=ax.transAxes, color="0.5")
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        ref = np.mean(d[("fp16", var)])
        comps = [c[3:] for c in dict.fromkeys(c for c, _ in d) if c.startswith("in_")]
        x = np.arange(len(comps))
        ax.bar(x, [np.mean(d[(f"in_{c}", var)]) / ref for c in comps], 0.6, color="#0072B2")
        ax.set_xticks(x, [c.replace("_", " ") for c in comps], rotation=40, ha="right", fontsize=7)
        ax.set_ylim(0, 1.15)
        ax.tick_params(labelsize=7)
        ax.set_ylabel("departure\n/ fp16", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fp16_oat.png", dpi=200)
    plt.close(fig)


def sensitivity_depth_figure() -> None:
    """Relative fp16 - fp32 difference of every module's output in the first
    model step against its position in the execution order (median over dates)."""
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for model, title in MODELS.items():
        path = RESULTS / "sensitivity" / f"{model}_depth.csv"
        if not path.exists():
            continue
        pos = defaultdict(list)
        with open(path) as f:
            for r in csv.DictReader(f):
                pos[int(r["position"])].append(float(r["relative_difference"]))
        ps = sorted(pos)
        med = np.array([np.nanmedian(pos[p]) for p in ps])
        x = np.array(ps) / max(ps)
        k = max(1, len(ps) // 40)  # running median over about 1/40 of the network
        smooth = np.array([np.nanmedian(med[max(0, i - k):i + k + 1]) for i in range(len(med))])
        ax.plot(x, smooth, linewidth=1.2, label=title)
    ax.set_yscale("log")
    ax.set_xlabel("position in the network (fraction of modules executed)", fontsize=8)
    ax.set_ylabel("relative fp16 - fp32 difference", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fp16_depth.png", dpi=200)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    fp32_skill("global")
    model_skill("global")
    model_skill("uk_europe")
    relative_change("fp16", "global", "rmse", ylim=(-2, 2))
    relative_change("fp16", "global", "mae", ylim=(-2, 2))
    departure_ratio("fp16", "global")
    area_table("fp16")
    power_table("fp16")
    eightbit_table()
    bf16_table()
    whole_model_figure()
    lead_time_figure("fp16", "global")
    missing = [f"{m}_{t}" for m in MODELS for t in ("fp32", "fp16")
               if not (RESULTS / "regional" / f"{m}_{t}.npz").exists()]
    if missing:  # the archived results leave them out (forecast --regional makes them)
        print(f"latitude bands, cell map and spectra skipped: no regional statistics for {', '.join(missing)}")
    else:
        band_change("fp16")
        cell_map("z500", 5)
        spectra(5, "fp16")
        spectra(5, "fp16", full=True)
    headroom_figure()
    ensemble_figure("fp16")
    cost_table()
    sensitivity_oat_figure()
    sensitivity_depth_figure()
    storm_figure("fp16")
    location_series("london", "2024-01-20", "00")
    print("figures and tables written")


if __name__ == "__main__":
    main()
