# nwpprec

Runs five AI weather models in fp32, fp16 and bf16 (and simulated 8-bit formats) and verifies them against
ERA5. In every precision the physical state, its normalisation and
denormalisation stay in fp32; only the network computes in reduced precision.

## Layout

| module | what it does |
|---|---|
| `models/<model>.py` | one file per model (FastNet: `models/fastnet/`, AGPL-3.0): `load(precision)`, `forecast(model, init_time)`, `fields(state)`, `COMPONENTS` |
| `precision.py` | `convert`, `cast_inputs`, `keep_fp32`: where a network switches precision |
| `run.py` | one forecast from any ERA5 initial time, written to NetCDF |
| `forecast.py` | runs a model from the 52 dates, scores it against ERA5 and against the fp32 forecast |
| `sensitivity.py` | where fp16 rounding enters: one component at a time, layer types, outlier channels, depth |
| `regional.py` | regional statistics and zonal spectra (`forecast --regional`) |
| `headroom.py` | how close fp16 activations come to fp16's numeric range |
| `storms.py` | tracks of four named storms |
| `ensemble.py` | ensemble verification of Aurora's ensemble checkpoint |
| `benchmark.py` | time, memory and energy per step |
| `state.py` | why the physical state stays fp32: rounding and overflow of ERA5 fields, whole-model conversion |
| `eightbit.py` | simulated 8-bit quantisation of weights and layer inputs |
| `plots.py` | the paper's figures and tables |
| `download.py` | ERA5 data (the 52 dates' or chosen initial times') and model checkpoints |
| `check.py` | reproduction check: one date per model against the stored scores |
| `era5.py`, `dates.py`, `scoring.py`, `paths.py` | ERA5 fields, initial dates, area-weighted scores, file locations |

Results go to `data/results/`.

## Environments

| model | environment |
|---|---|
| `aurora`, `aurora_ens`, `pangu` | `.venv` |
| `fourcastnet` | `.venv-fourcastnet` |
| `fastnet` | `.venv-fastnet` |
| `sfno` | `.venv-sfno` |

How to install them: `../envs/README.md`.

Run from the repository root, e.g.

    .venv/bin/python -m nwpprec.forecast --model pangu --precision fp32 --save-fields
    .venv/bin/python -m nwpprec.forecast --model pangu --precision fp16 --regional
    .venv/bin/python -m nwpprec.sensitivity --model pangu --experiment oat

The fp32 run with `--save-fields` comes first: the other runs and experiments
score their departure from it.

## Running one forecast

To use a model in fp16 without the paper's experiments:

    .venv/bin/python -m nwpprec.download checkpoints                 # Aurora, Pangu-Weather
    .venv/bin/python -m nwpprec.download era5 --init 2024-06-01T00
    .venv/bin/python -m nwpprec.run --model pangu --precision fp16 \
        --init 2024-06-01T00 --lead 240 --every 24 --out pangu_fp16.nc

with the model's environment (above). For each model:

    .venv/bin/python -m nwpprec.run --model aurora --init 2024-06-01T00 --out aurora_fp16.nc
    .venv/bin/python -m nwpprec.run --model aurora_ens --init 2024-06-01T00 --seed 1 --out aurora_ens_fp16_m1.nc
    .venv/bin/python -m nwpprec.run --model pangu --init 2024-06-01T00 --out pangu_fp16.nc
    .venv-fourcastnet/bin/python -m nwpprec.run --model fourcastnet --init 2024-06-01T00 --out fourcastnet_fp16.nc
    .venv-sfno/bin/python -m nwpprec.run --model sfno --init 2024-06-01T00 --out sfno_fp16.nc
    .venv-fastnet/bin/python -m nwpprec.run --model fastnet --init 2024-06-01T00 --out fastnet_fp16.nc

| model | step | grid | weights | also needs |
|---|---|---|---|---|
| `aurora` | 6 h | 720 x 1440 (no south pole row) | `download checkpoints` | the state 6 h before the initial time (fetched by `download era5 --init`); in fp32, weight streaming to fit 16 GB (automatic) |
| `aurora_ens` | 6 h | 720 x 1440 | `download checkpoints` | as `aurora`; one member per `--seed` |
| `pangu` | 24 h | 721 x 1440 | `download checkpoints` | `third_party/pangu-pytorch` (`../envs/README.md`) |
| `fourcastnet` | 6 h | 720 x 1440 (no south pole row) | on first load | ERA5 relative humidity (fetched by `download era5 --init`) |
| `sfno` | 6 h | 721 x 1440 | on first load | `makani` and `torch-harmonics` from GitHub (`../envs/README.md`); nearly all of a 16 GB GPU in fp32 |
| `fastnet` | 6 h | O96, interpolated to 721 x 1440 | on first load | `data/fastnet_static_o96.npz`; about 1.6% of points NaN; not bit-reproducible |

`--precision` defaults to fp16 and `--lead` to 240 h; `--every` keeps every
N hours (a multiple of the model's step). The output holds 2 m temperature,
mean sea-level pressure, 10 m wind (components and speed), z500 and t850 on
the 0.25-degree grid, one field per lead time. The same from Python:

    from nwpprec.run import run
    ds = run("pangu", "fp16", "2024-06-01T00", lead=240, every=24)   # xarray.Dataset

Initial times are 00, 06, 12 or 18 UTC on any day ERA5 covers. Other
variables are in the model's state but not written: see `fields()` in the
model's adapter. No converted weights are distributed: each run converts the
original checkpoint as it loads, which takes seconds.

## Using fp16 in your own code

The adapters in `models/` are short and can serve as templates. What they
do, and what goes wrong otherwise:

- **Keep the state in fp32; convert only the network.** Normalise the fp32
  state, cast the normalised input to fp16, run the network, cast its output
  back to fp32 and denormalise it (`models/pangu.py`, `Pangu.forward`).
  Calling `model.to(torch.float16)` on the whole model instead also rounds
  the state and its normalisation: mean sea-level and surface pressure in Pa
  and geopotential at 400 hPa and above exceed fp16's largest number
  (65504) and become infinite. In bf16 they fit but are rounded to 512 Pa
  (pressure). The paper's Section 4.1 and `state.py` show both.
- **Convert real-valued weights only.** SFNO's spectral filters are complex,
  and `module.to(torch.float16)` discards their imaginary parts (with a
  warning only). `models/sfno.py` runs each complex filter as real and
  imaginary fp16 parts; its spherical harmonic transforms stay fp32.
- **Keep the forecast loop in fp32.** Residual additions, time stepping and
  anything that accumulates over steps operate on the fp32 state outside the
  network.
- **Check the range once.** fp16 overflows above 65504. Before trusting a new
  model in fp16, record the largest absolute activation of each layer in an
  fp32 forecast over the full lead time (FourCastNet's largest values grow
  fivefold after day 6), e.g.

      peak = {}
      def hook(name):
          def f(module, args, out):
              if isinstance(out, torch.Tensor) and out.is_floating_point():
                  peak[name] = max(peak.get(name, 0.0), out.detach().abs().max().item())
          return f
      for name, module in net.named_modules():
          module.register_forward_hook(hook(name))

  and consider keeping in fp32 any layer whose peak comes close to 65504
  (`precision.keep_fp32`). `headroom.py` does this for the five models,
  whose largest values stay 38 (Aurora) to 544 (Pangu-Weather) times below
  the limit. An fp16 forecast that overflows shows it plainly:
  infinite or NaN fields within the first steps.
- **Compare with fp32 before relying on it**: a few initial times scored
  against the fp32 forecast and against ERA5 (`forecast.py`, `scoring.py`).

`precision.py` has the helpers: `convert(module, dtype)` runs one module in
fp16 with fp32 inputs and outputs; `keep_fp32` does the reverse inside a
converted network.

## Tests and checks

Unit tests (no GPU needed; one test also needs the ERA5 files and is skipped without them):

    .venv/bin/python -m pytest tests

Checking an installation:

    .venv/bin/python -m nwpprec.check --model pangu

reruns the first of the 52 dates to 24 h in fp32 and fp16 and compares the
global RMSE with the stored scores: identical for every model except FastNet,
whose forecasts are not bit-reproducible from run to run (within 1e-3).
