"""Material module of `dolfin_mech`.

This sub-package provides the material law/stresses of dolfin_mech as well as the material factory.
"""

from .material import (  # isort: skip
	Material,
	material_factory,
)
from . import elastic, inelastic

__all__ = [
	"elastic",
	"inelastic",
	"Material",
	"material_factory",
]
