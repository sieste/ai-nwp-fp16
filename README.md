# Half-precision inference of AI weather prediction models

Code and paper for *Half-precision inference of AI weather prediction
models: negligible loss of skill at a fraction of the cost* (Siegert, 2026).

Archived at Zenodo: code [doi:10.5281/zenodo.23888734](https://doi.org/10.5281/zenodo.23888734),
results [doi:10.5281/zenodo.23887144](https://doi.org/10.5281/zenodo.23887144).

Five publicly available AI weather models -- Aurora (deterministic and
ensemble checkpoints), Pangu-Weather, FourCastNet, SFNO and FastNet -- are run
on a single 16 GB GPU from 52 initial dates in 2024, in fp32 and with their
networks in fp16 (and bf16, and simulated 8-bit formats), with the physical
state kept in fp32, and verified against ERA5.

## Contents

| | |
|---|---|
| `nwpprec/` | the Python package that runs every forecast, experiment, figure and table (see `nwpprec/README.md`) |
| `envs/` | the exact software environments, and how to install them |
| `tests/` | unit tests of the package (`pytest tests`) |
| `tex/` | the paper's LaTeX source and tables |
| `figures/` | the paper's figures |
| `data/` | FastNet's static input fields, which are no longer downloadable |

Results (scores and the other numbers behind the figures and tables) are
archived separately: see *Data* below.

## Running a model in fp16

To run one of the models in fp16 from an ERA5 initial state, without
reproducing the paper (install the model's environment first, `envs/README.md`):

    .venv/bin/python -m nwpprec.download checkpoints
    .venv/bin/python -m nwpprec.download era5 --init 2024-06-01T00
    .venv/bin/python -m nwpprec.run --model aurora --precision fp16 \
        --init 2024-06-01T00 --lead 240 --out aurora_fp16.nc

`nwpprec/README.md` describes the output, the Python interface, and how to
run other models in fp16 in your own code.

## Reproducing the results

You need an NVIDIA GPU with 16 GB of memory, about 400 GB of disk space for
ERA5 and the model checkpoints, and a Copernicus Climate Data Store account
(`~/.cdsapirc`) for ERA5.

1. Install the four environments: `envs/README.md`.
2. Download the data and checkpoints:

       .venv/bin/python -m nwpprec.download era5
       .venv/bin/python -m nwpprec.download checkpoints

   FourCastNet's, SFNO's and FastNet's weights are downloaded when the models
   are first loaded.
3. Check the installation against the stored scores (one date per model):

       .venv/bin/python -m nwpprec.check --model aurora

4. Run the experiments. For each model (with its environment, `envs/README.md`):

       python -m nwpprec.forecast --model pangu --precision fp32 --save-fields --regional
       python -m nwpprec.forecast --model pangu --precision fp16 --regional
       python -m nwpprec.forecast --model pangu --precision bf16 --regional
       python -m nwpprec.headroom --model pangu
       python -m nwpprec.storms --model pangu --precision fp32
       python -m nwpprec.storms --model pangu --precision fp16
       python -m nwpprec.sensitivity --model pangu --experiment oat    # also: types, outliers, depth
       python -m nwpprec.eightbit --model pangu
       python -m nwpprec.benchmark --model pangu

   and once:

       .venv/bin/python -m nwpprec.ensemble --precision fp32
       .venv/bin/python -m nwpprec.ensemble --precision fp16
       .venv/bin/python -m nwpprec.state table
       .venv/bin/python -m nwpprec.state whole fp16
       .venv/bin/python -m nwpprec.state whole bf16 --score

   The fp32 forecast with `--save-fields` comes first: the other runs score
   their departure from it. On an RTX 5080, a model's 52-date forecast takes
   between about 10 minutes (Pangu-Weather) and 3 hours (Aurora in fp32).

5. Make the figures and tables, and the paper:

       .venv/bin/python -m nwpprec.plots
       make -C tex

Data and checkpoints can be kept outside the repository by setting
`NWPPREC_DATA` and `NWPPREC_CHECKPOINTS`.

## Data

The scores, sensitivity, headroom, storm, benchmark and location results
behind the paper's figures and tables are archived at Zenodo:
[doi:10.5281/zenodo.23887144](https://doi.org/10.5281/zenodo.23887144). Unpacked into `data/results/`, they let `nwpprec.plots` redraw the
figures and tables without rerunning the forecasts (except the maps, bands
and spectra, which need the regional statistics of step 4).

## Licence

Code: MIT (`LICENSE`). Paper (`tex/`, `figures/`): CC BY 4.0. The
exception is the FastNet adapter, `nwpprec/models/fastnet/`,
which is AGPL-3.0 because it imports the Met Office's AGPL-licensed
inference code; see `NOTICE`, which also lists the terms of the model
weights and other third-party code, none of which is included here.

## Citation

See `CITATION.cff`.
