"""Core elements of module `dolfin_mech`."""

from .compute_error import compute_error
from .FOI import FOI
from .QOI import QOI

__all__ = ["FOI", "QOI", "compute_error"]
