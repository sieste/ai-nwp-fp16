"""Where does fp16 change the forecast? Four experiments on every 4th of the
52 dates (13), each testing a common belief about reduced precision:

  oat       first and last layers matter most: one component at a time in
            fp16 (and all components converted separately), scored by the
            departure from the fp32 forecast at every step up to 24 h
  types     normalisation (and softmax) need higher precision: all layers of
            one type (norm, attention or spectral filter, MLP) kept in fp32
            with their original weights inside the fp16 network
  outliers  errors concentrate in a few outlier channels: per-channel maximum
            |value| of every module's fp16 output
  depth     rounding error builds up with depth: relative fp16 - fp32
            difference of every module's output in the first step, in
            execution order, from a fixed sample of each output

    .venv/bin/python -m nwpprec.sensitivity --model pangu --experiment oat

FastNet runs in PyTorch's deterministic mode here (its default is not
bit-reproducible). Results: data/results/sensitivity/<model>_<experiment>.*
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # needed by deterministic mode

import argparse  # noqa: E402
import csv  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import zlib  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from . import forecast  # noqa: E402
from .dates import init_time, weekly_dates  # noqa: E402
from .paths import RESULTS  # noqa: E402
from .precision import cast  # noqa: E402
from .scoring import scores  # noqa: E402

DATES = weekly_dates()[::4]
DEVICE = "cuda"
OUT = RESULTS / "sensitivity"
# fp32 computation inside an fp16 network, chunked so that fp32 copies of large inputs never exist in full:
TOKEN_WISE = {"LayerNorm", "MLP", "Mlp", "AFNOMlp", "Linear"}  # act on each token (last axis) separately
ROW_WISE = {"PerceiverAttention"}  # rows (dim 0 of every input) are independent
CHUNK_VALUES, CHUNK_ROWS = 2 ** 26, 2048


def free() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def layer_types(name: str, model) -> dict:
    """{type: [modules]}, non-overlapping."""
    by_class, parent = {}, {}
    for module in model.modules():
        by_class.setdefault(type(module).__name__, []).append(module)
        for child in module.children():
            parent[id(child)] = type(module).__name__
    get = lambda *names: [m for n in names for m in by_class.get(n, [])]  # noqa: E731
    if name in ("aurora", "aurora_ens"):
        norms = get("AdaptiveLayerNorm") + [m for m in get("LayerNorm") if parent.get(id(m)) != "AdaptiveLayerNorm"]
        return {"norm": norms, "attention": get("WindowAttention", "PerceiverAttention"), "mlp": get("MLP")}
    if name == "pangu":
        return {"norm": get("LayerNorm"), "attention": get("EarthAttention3D"), "mlp": get("Mlp")}
    if name == "fourcastnet":
        return {"norm": get("LayerNorm"), "filter": get("AFNO2DLayer"), "mlp": get("AFNOMlp")}
    if name == "fastnet":
        return {"norm": get("LayerNorm"), "linear": get("Linear")}
    if name == "sfno":  # filters: kept by loading with keep_filters (see types)
        return {"norm": get("InstanceNorm2d"), "filter": get("SpectralConv"), "mlp": get("MLP")}
    raise ValueError(name)


def keep_fp32(model, modules, fp32_state: dict, dtype=torch.float16, rows: bool = False) -> None:
    """Keep `modules` in fp32, with their original weights from fp32_state (the
    fp32 model's state dict), inside a network converted to `dtype`. rows: the
    modules act on each grid point of (B, C, H, W) inputs separately (SFNO's
    1x1-convolution MLPs) and are computed in chunks of latitude rows."""
    names = {id(m): n for n, m in model.named_modules()}
    for module in modules:
        prefix = names[id(module)]
        module.float()
        with torch.no_grad():
            for pname, p in module.named_parameters():
                p.copy_(fp32_state[f"{prefix}.{pname}"].to(p.device))
        original = module.forward
        if rows:
            def forward(x, _f=original):
                n = max(1, CHUNK_VALUES // (4 * x.shape[1] * x.shape[3]))
                return torch.cat([_f(x[:, :, i:i + n].float()).to(dtype) for i in range(0, x.shape[2], n)], dim=2)
        elif type(module).__name__ in TOKEN_WISE:
            def forward(x, _f=original):
                flat = x.reshape(-1, x.shape[-1])
                rows = max(1, CHUNK_VALUES // (4 * x.shape[-1]))
                out = torch.cat([_f(flat[i:i + rows].float()).to(dtype) for i in range(0, flat.shape[0], rows)])
                return out.reshape(*x.shape[:-1], out.shape[-1])
        else:
            def whole(*args, _f=original, **kwargs):
                return cast(_f(*cast(args, torch.float32), **cast(kwargs, torch.float32)), dtype)
            def by_rows(*args, _whole=whole, **kwargs):
                n = args[0].shape[0]
                if n <= CHUNK_ROWS or kwargs:
                    return _whole(*args, **kwargs)
                return torch.cat([_whole(*(a[i:i + CHUNK_ROWS] for a in args)) for i in range(0, n, CHUNK_ROWS)])
            forward = by_rows if type(module).__name__ in ROW_WISE else whole
        module.forward = forward


def _tensor(out):
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
        return out[0]
    return None


def _write(rows: list, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}", flush=True)


def _departures(m, model, refs: dict, config: str, max_lead: int = 24) -> list:
    rows = []
    for date, hour in DATES:
        for lead, state in m.forecast(model, init_time(date, hour), max_lead):
            fields = forecast.with_wind_speed(m.fields(state))
            for var, x in fields.items():
                d = scores(x, refs[(date, hour, lead)][var], m.LAT, m.SKIP_NAN)["rmse"]
                rows.append({"config": config, "date": date, "hour": hour, "lead": lead, "var": var, "departure": d})
    return rows


def _fp32_refs(m, name, max_lead: int = 24) -> dict:
    model = m.load("fp32", DEVICE)
    refs = {}
    for date, hour in DATES:
        for lead, state in m.forecast(model, init_time(date, hour), max_lead):
            refs[(date, hour, lead)] = forecast.with_wind_speed(m.fields(state))
    del model
    free()
    return refs


def oat(name: str) -> None:
    m = forecast.adapter(name)
    refs = _fp32_refs(m, name)
    configs = {"fp16": None, "all": m.COMPONENTS}
    configs.update({f"in_{c}": (c,) for c in m.COMPONENTS})
    rows = []
    for config, comps in configs.items():
        model = m.load("fp16", DEVICE) if comps is None else m.load("fp16", DEVICE, components_=comps)
        rows += _departures(m, model, refs, config)
        print(f"[{name} oat] {config}", flush=True)
        del model
        free()
    _write(rows, OUT / f"{name}_oat.csv")


def _fp32_state(m) -> dict:
    model = m.load("fp32", device="cpu")
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    del model
    free()
    return state


def types(name: str) -> None:
    m = forecast.adapter(name)
    refs = _fp32_refs(m, name)
    fp32_state = _fp32_state(m)
    model = m.load("fp16", DEVICE)
    rows = _departures(m, model, refs, "fp16")
    groups = list(layer_types(name, model))
    del model
    free()
    for group in groups:
        if name == "sfno" and group == "filter":  # complex filters left in complex64
            model = m.load("fp16", DEVICE, keep_filters=True)
        else:
            model = m.load("fp16", DEVICE)
            keep_fp32(model, layer_types(name, model)[group], fp32_state, rows=name == "sfno" and group == "mlp")
        rows += _departures(m, model, refs, f"keep_{group}")
        print(f"[{name} types] {group}", flush=True)
        del model
        free()
    _write(rows, OUT / f"{name}_types.csv")


def outliers(name: str) -> None:
    m = forecast.adapter(name)
    model = m.load("fp16", DEVICE)
    maxima = {}

    def hook(mod_name):
        def h(module, args, out):
            t = _tensor(out)
            if t is not None and name == "sfno" and t.dim() == 4:
                t = t.movedim(1, -1)  # channels first (B, C, H, W)
            if t is None or t.dtype != torch.float16 or t.dim() < 2 or t.shape[-1] < 8:
                return
            flat = t.detach().reshape(-1, t.shape[-1])
            rows = max(1, CHUNK_VALUES // t.shape[-1])
            cm = torch.stack([flat[i:i + rows].abs().float().amax(0) for i in range(0, flat.shape[0], rows)]).amax(0)
            key = f"{mod_name} [{t.shape[-1]}]"  # a module may be applied to outputs of different widths
            maxima[key] = cm if key not in maxima else torch.maximum(maxima[key], cm)
        return h
    for n, mod in model.named_modules():
        mod.register_forward_hook(hook(n or "<model>"))
    for date, hour in DATES:
        for _ in m.forecast(model, init_time(date, hour), 24):
            pass
    summary = {}
    for n, cm in maxima.items():
        cm = cm.cpu().numpy()
        med = float(np.median(cm))
        summary[n] = {"channels": int(cm.size), "max": float(cm.max()), "median": med,
                      "max_over_median": float(cm.max() / med) if med > 0 else None,
                      "channels_over_6x_median": int((cm > 6 * med).sum())}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}_outliers.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"Wrote {OUT / f'{name}_outliers.json'}", flush=True)


SAMPLE = 2 ** 14  # values sampled from each module output


def _sample_outputs(m, model, start) -> tuple:
    """({(module, call): sampled values} for the first model step, execution order)."""
    samples, order, calls = {}, [], {}

    def hook(mod_name):
        def h(module, args, out):
            t = _tensor(out)
            if t is None or not t.is_floating_point() or t.numel() == 0:
                return
            k = calls.get(mod_name, 0)
            calls[mod_name] = k + 1
            g = torch.Generator().manual_seed(zlib.crc32(f"{mod_name}/{k}".encode()))
            idx = torch.randint(t.numel(), (min(SAMPLE, t.numel()),), generator=g).to(t.device)
            samples[(mod_name, k)] = t.detach().reshape(-1)[idx].double().cpu()
            order.append((mod_name, k, type(module).__name__))
        return h
    handles = [mod.register_forward_hook(hook(n or "<model>")) for n, mod in model.named_modules()]
    next(m.forecast(model, start, m.STEP_HOURS))
    for h in handles:
        h.remove()
    return samples, order


def depth(name: str) -> None:
    m = forecast.adapter(name)
    ref = {}
    model = m.load("fp32", DEVICE)
    for date, hour in DATES:
        ref[(date, hour)] = _sample_outputs(m, model, init_time(date, hour))
    del model
    free()
    model = m.load("fp16", DEVICE)
    rows = []
    for date, hour in DATES:
        samples, _ = _sample_outputs(m, model, init_time(date, hour))
        ref_samples, order = ref[(date, hour)]
        for i, (mod_name, k, cls) in enumerate(order):
            a, b = ref_samples[(mod_name, k)], samples.get((mod_name, k))
            if b is None:
                continue
            rel = float(torch.sqrt(((b - a) ** 2).mean()) / torch.sqrt((a ** 2).mean())) if (a ** 2).mean() > 0 else np.nan
            rows.append({"date": date, "hour": hour, "position": i, "module": mod_name, "call": k, "type": cls,
                         "relative_difference": rel})
    _write(rows, OUT / f"{name}_depth.csv")


EXPERIMENTS = {"oat": oat, "types": types, "outliers": outliers, "depth": depth}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=forecast.MODELS)
    p.add_argument("--experiment", required=True, choices=list(EXPERIMENTS))
    a = p.parse_args()
    if a.model == "fastnet":
        torch.use_deterministic_algorithms(True)
    EXPERIMENTS[a.experiment](a.model)


if __name__ == "__main__":
    main()
