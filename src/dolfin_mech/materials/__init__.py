"""Material module of `dolfin_mech`.

This sub-package provides the material law/stresses of dolfin_mech as well as the material factory.
"""

from .Elastic import ElasticMaterial
from .Inelastic import InelasticMaterial
from .Material import (
	Material,
	material_factory,
)

__all__ = [
	"InelasticMaterial",
	"ElasticMaterial",
	"Material",
	"material_factory",
]
