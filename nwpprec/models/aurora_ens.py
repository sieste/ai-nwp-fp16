"""Aurora 1.5 ensemble checkpoint: as aurora.py, with noise set by the seed."""

from functools import partial

from . import aurora
from .aurora import COMPONENTS, LAT, LON, SKIP_NAN, STEP_HOURS, components, fields, forecast  # noqa: F401

load = partial(aurora.load, variant="ensemble")

__all__ = ["COMPONENTS", "LAT", "LON", "SKIP_NAN", "STEP_HOURS", "components", "fields", "forecast", "load"]
