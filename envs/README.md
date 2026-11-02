# Environments

The models need four separate Python environments (their dependencies
conflict). Each `*.lock.txt` lists the exact packages the paper's results
were computed with (Python 3.14, PyTorch 2.14 with CUDA 13, on an NVIDIA
GeForce RTX 5080).

| environment | models | lock file |
|---|---|---|
| `.venv` | Aurora (deterministic and ensemble), Pangu-Weather | `main.lock.txt` |
| `.venv-fourcastnet` | FourCastNet | `fourcastnet.lock.txt` |
| `.venv-sfno` | SFNO | `sfno.lock.txt` + two packages from GitHub, below |
| `.venv-fastnet` | FastNet | `fastnet.lock.txt` |

From the repository root:

    python3 -m venv .venv && .venv/bin/pip install -r envs/main.lock.txt
    python3 -m venv .venv-fourcastnet && .venv-fourcastnet/bin/pip install -r envs/fourcastnet.lock.txt
    python3 -m venv .venv-fastnet && .venv-fastnet/bin/pip install -r envs/fastnet.lock.txt

    python3 -m venv .venv-sfno && .venv-sfno/bin/pip install -r envs/sfno.lock.txt
    git clone https://github.com/NVIDIA/makani third_party/makani
    git -C third_party/makani checkout a0aa4c4
    git clone https://github.com/NVIDIA/torch-harmonics third_party/torch-harmonics
    git -C third_party/torch-harmonics checkout dd27548
    .venv-sfno/bin/pip install --no-deps -e third_party/makani -e third_party/torch-harmonics

`makani` is not on PyPI, and the `torch-harmonics` release on PyPI lacks
functions `makani` needs; both are installed from the commits used here,
without their dependencies (`--no-deps`). The editable install (`-e`) leaves
out `torch-harmonics`' optional compiled extensions, which SFNO does not use
(a regular install tries to build them and fails);
`nwpprec/models/sfno.py` registers stand-ins for the modules that would
import them.

Pangu-Weather runs through a PyTorch reimplementation that cannot be
installed with pip; clone it into `third_party/`:

    git clone https://github.com/zhaoshan2/pangu-pytorch third_party/pangu-pytorch
    git -C third_party/pangu-pytorch checkout 97db96b

Then download the checkpoints and ERA5 (`python -m nwpprec.download`, see
the main README) and check that everything reproduces:

    .venv/bin/python -m nwpprec.check --model pangu
