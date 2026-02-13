"""Material module of `dolfin_mech`.

This sub-package provides the material law/stresses of dolfin_mech as well as the material factory.
"""

from . import elastic, inelastic
from .Material import (
	Material,
	material_factory,
)

__all__ = [
	"elastic",
	"inelastic",
	"Material",
	"material_factory",
]
