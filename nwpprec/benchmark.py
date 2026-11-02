"""Inference time, peak GPU memory and energy per model and precision.

The model is built as in the forecast runs and stepped forward from one
initial date, restarting the 10-day forecast as often as needed, until at
least MIN_SECONDS of model steps (and at least MIN_STEPS) have been timed
(synchronised per step). The first step of every forecast, which also loads
the inputs, is not timed. Energy is the mean GPU power reported by
nvidia-smi (sampled every 50 ms) within the timed steps, times their total
duration. Needs the GPU to itself.

    .venv/bin/python -m nwpprec.benchmark --model pangu

Results are merged into data/results/benchmark.json.
"""

import argparse
import gc
import json
import subprocess
import threading
import time

import torch

from . import forecast
from .dates import init_time
from .paths import RESULTS

INIT = ("2024-01-07", "00")
MIN_SECONDS, MIN_STEPS = 60.0, 8


class PowerSampler:
    def __init__(self):
        self.samples = []
        self.proc = subprocess.Popen(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits",
                                      "-lms", "50"], stdout=subprocess.PIPE, text=True)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            try:
                self.samples.append((time.perf_counter(), float(line)))
            except ValueError:
                pass

    def mean_watts(self, t0, t1):
        watts = [w for t, w in self.samples if t0 <= t <= t1]
        return sum(watts) / len(watts) if watts else None

    def stop(self):
        self.proc.terminate()


def bench(name: str, precision: str) -> dict:
    m = forecast.adapter(name)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = m.load(precision)
    power = PowerSampler()
    intervals = []
    while sum(t1 - t0 for t0, t1 in intervals) < MIN_SECONDS or len(intervals) < MIN_STEPS:
        steps = m.forecast(model, init_time(*INIT), 240)
        next(steps)  # loads the inputs; not timed
        while True:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                next(steps)
            except StopIteration:
                break
            torch.cuda.synchronize()
            intervals.append((t0, time.perf_counter()))
    time.sleep(0.2)
    power.stop()
    times = [t1 - t0 for t0, t1 in intervals]
    inside = [w for t, w in power.samples if any(t0 <= t <= t1 for t0, t1 in intervals)]
    watts = sum(inside) / len(inside) if inside else None
    per_step = sum(times) / len(times)
    return {"s_per_step": per_step, "step_hours": m.STEP_HOURS, "s_per_10_days": per_step * 240 / m.STEP_HOURS,
            "peak_gb": torch.cuda.max_memory_allocated() / 1e9, "mean_watts": watts,
            "kj_per_10_days": watts * per_step * 240 / m.STEP_HOURS / 1e3 if watts else None,
            "timed_steps": len(times), "power_samples": len(inside),
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=forecast.MODELS)
    p.add_argument("--precisions", nargs="+", default=["fp32", "bf16", "fp16"])
    a = p.parse_args()
    path = RESULTS / "benchmark.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    # imported here, not first inside bench: earth2studio keeps the frames of
    # its failed optional imports alive, which would keep that call's model
    # on the GPU during the next configuration
    forecast.adapter(a.model)
    for precision in a.precisions:
        r = bench(a.model, precision)
        results.setdefault(a.model, {})[precision] = r
        print(f"{a.model} {precision}: {r['s_per_step']:.3f} s/step, {r['peak_gb']:.1f} GB, "
              f"{r['kj_per_10_days'] or 0:.1f} kJ per 10 days", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
