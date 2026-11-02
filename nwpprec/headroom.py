"""How close an fp16 network comes to the limits of fp16's range.

Forward hooks on every module of the model (not only the leaf layers, so
that residual sums computed in a parent's forward are seen as that parent's
output) accumulate, for every fp16 tensor a module returns, the largest
finite |value|, the number of inf/NaN values (overflow), the number of
nonzero values below fp16's smallest normal number (which keep fewer
significant bits), and the largest |value| per model step. Values computed
and consumed inside a module without being returned (e.g. attention scores)
are not seen, but an overflow there would reach the module's output as inf
or NaN.

    .venv/bin/python -m nwpprec.headroom --model pangu

Writes data/results/headroom/<model>.json.
"""

import argparse
import json

import torch

from . import forecast
from .dates import init_time, weekly_dates
from .paths import RESULTS

FP16_MAX = 65504.0
FP16_MIN_NORMAL = 6.103515625e-05
CHUNK = 2 ** 26  # values per chunk: no full-size temporary copies of large outputs


class Headroom:
    def __init__(self, model: torch.nn.Module, n_steps: int):
        self.step = 0  # set by the caller: the model step being computed
        self.stats = {}
        self.n_steps = n_steps
        self.step_max = None  # on the model's device, created at the first hook call
        for name, module in model.named_modules():
            module.register_forward_hook(self._hook(name or "<model>"))

    def _hook(self, name):
        def hook(module, args, output):
            for t in _tensors(output):
                if t.dtype != torch.float16 or t.numel() == 0:
                    continue
                flat = t.detach().reshape(-1)
                m = torch.zeros((), device=t.device)
                nonfinite = torch.zeros((), dtype=torch.int64, device=t.device)
                subnormal = torch.zeros((), dtype=torch.int64, device=t.device)
                for i in range(0, flat.numel(), CHUNK):
                    a = flat[i:i + CHUNK].abs()
                    finite = torch.isfinite(a)
                    m = torch.maximum(m, torch.where(finite, a, torch.zeros((), dtype=a.dtype, device=a.device)).max().float())
                    nonfinite += (~finite).sum()
                    subnormal += ((a > 0) & (a < FP16_MIN_NORMAL)).sum()
                s = self.stats.setdefault(name, {"max_abs": torch.zeros((), device=t.device), "nonfinite": 0,
                                                 "subnormal": 0, "numel": 0})
                s["max_abs"] = torch.maximum(s["max_abs"], m)
                s["nonfinite"] = s["nonfinite"] + nonfinite
                s["subnormal"] = s["subnormal"] + subnormal
                s["numel"] += flat.numel()
                if self.step_max is None:
                    self.step_max = torch.zeros(self.n_steps, device=t.device)
                self.step_max[self.step] = torch.maximum(self.step_max[self.step], m)
        return hook

    def summary(self) -> dict:
        modules = {n: {"max_abs": s["max_abs"].item(), "nonfinite": int(s["nonfinite"]),
                       "subnormal_frac": int(s["subnormal"]) / max(s["numel"], 1)} for n, s in self.stats.items()}
        top = max(m["max_abs"] for m in modules.values())
        return {"max_abs": top, "margin": FP16_MAX / top, "overflowing_modules": [n for n, m in modules.items() if m["nonfinite"]],
                "step_max_abs": self.step_max.tolist(), "modules": modules}


def _tensors(obj):
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (tuple, list)):
        for o in obj:
            yield from _tensors(o)
    elif isinstance(obj, dict):
        for o in obj.values():
            yield from _tensors(o)


def run(name: str, dates: list, max_lead: int = 240) -> dict:
    m = forecast.adapter(name)
    model = m.load("fp16")
    h = Headroom(model, max_lead // m.STEP_HOURS)
    for i, (date, hour) in enumerate(dates):
        for lead, _ in m.forecast(model, init_time(date, hour), max_lead):
            h.step = min(lead // m.STEP_HOURS, h.n_steps - 1)  # the next step to be computed
        h.step = 0
        print(f"[{name} headroom {i + 1}/{len(dates)}] {date} {hour}h", flush=True)
    summary = h.summary()
    path = RESULTS / "headroom" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"max |x| {summary['max_abs']:.4g} (margin {summary['margin']:.0f}x); wrote {path}", flush=True)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=forecast.MODELS)
    run(p.parse_args().model, weekly_dates())


if __name__ == "__main__":
    main()
