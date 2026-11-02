"""Locations of checkpoints, input data, results, figures and tables.

Checkpoints and data default to checkpoints/ and data/ in the repository;
set NWPPREC_CHECKPOINTS and NWPPREC_DATA to keep them elsewhere (ERA5
inputs need about 370 GB, saved forecast fields about 150 GB).
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = Path(os.environ.get("NWPPREC_CHECKPOINTS", ROOT / "checkpoints"))
DATA = Path(os.environ.get("NWPPREC_DATA", ROOT / "data"))
ERA5 = DATA / "inputs" / "era5"  # one surface and one pressure-level file per day
RESULTS = DATA / "results"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tex" / "tables"
THIRD_PARTY = ROOT / "third_party"  # pangu-pytorch (not installable with pip)
