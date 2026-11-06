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

## Numerical details of the reduced-precision runs

- **16-bit arithmetic on the GPU.** On current NVIDIA GPUs, fp16 and bf16
  matrix multiplications and convolutions run on tensor cores, which multiply
  16-bit inputs but accumulate the products in fp32 within each operation,
  rounding only the output to 16 bits. PyTorch's 16-bit normalisation, softmax
  and reduction kernels likewise typically accumulate internally in fp32. The
  dominant error of a 16-bit network is therefore the rounding of every stored
  tensor (the weights once, the intermediate results after every operation),
  which compounds with network depth and, in an autoregressive forecast, from
  step to step. One PyTorch default qualifies this: 16-bit matrix
  multiplications may use reduced-precision reductions in some algorithms
  (`allow_fp16_reduced_precision_reduction` and
  `allow_bf16_reduced_precision_reduction`, both enabled by default).
- **TF32 in the fp32 runs.** PyTorch allows convolutions to use TF32, a format
  with fp32's 8-bit exponent and a 10-bit mantissa (`cudnn.allow_tf32`, enabled
  by default), whereas fp32 matrix multiplications do not use it by default.
  On the RTX 5080 this setting is active: an fp32 convolution's relative error
  against an fp64 reference is 3e-4 with it, 4e-7 without it, and 3e-3 in bf16.
  "fp32" therefore means fp32 for every matrix multiplication and TF32 for the
  convolutions, which occur in the patch embeddings of Aurora, Pangu-Weather and
  FourCastNet, in Pangu-Weather's patch recovery, and in SFNO's encoder, decoder
  and the 1x1 convolutions of its blocks, in the fp32 references and in the fp32
  components of the one-at-a-time runs alike. FastNet contains no convolutions.
  The default is kept deliberately, so that the reference is the model as a
  typical user runs it, not a strictly fp32 configuration that requires changing
  library settings.
- **FourCastNet.** Its spectral filter layer converts its intermediate results
  to fp32 before each Fourier transform, which therefore runs in fp32, and then
  multiplies them with the weights, which fails once the weights are in 16 bits.
  The adapter converts the results back to the precision of the weights after
  the transform. With fp32 weights, the modified model gives bit-identical
  results to the original.
- **SFNO.** Its spectral filters are complex-valued. PyTorch (version 2.14, used
  for every model) has no complex counterpart of bf16, and its complex fp16 type
  lacks the batched matrix products the filters need on the GPU. In bf16 and
  fp16, the real and imaginary parts of each filter are therefore stored as
  separate weights in that precision, and the complex multiplication is carried
  out with four real products. PyTorch's standard conversion of a whole model to
  16 bits cannot be used for SFNO: it keeps only the real parts of complex
  weights. SFNO's spherical harmonic transforms always run in fp32, as NVIDIA's
  implementation prescribes.
- **Integer and fp8 quantisation of Aurora.** Tested at full resolution with the
  `torchao` library. Weight-only int8 and fp8 quantisation was 27-28% *slower* per
  step than bf16, because the weights are converted back before every
  multiplication in a forward pass whose speed is limited by computation rather
  than memory access, and reduced peak memory by only about 10%, since the
  intermediate results, not the weights, dominate memory use. Quantising both
  weights and intermediate results to int8 or fp8 ran out of memory on the 16 GB
  GPU, and benchmarks of isolated matrix multiplications showed dynamic int8 two
  to seven times slower than bf16, with mixed results for fp8.

## Tests and checks

Unit tests (no GPU needed; one test also needs the ERA5 files and is skipped without them):

    .venv/bin/python -m pytest tests

Checking an installation:

    .venv/bin/python -m nwpprec.check --model pangu

reruns the first of the 52 dates to 24 h in fp32 and fp16 and compares the
global RMSE with the stored scores: identical for every model except FastNet,
whose forecasts are not bit-reproducible from run to run (within 1e-3).
