# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################
"""Inelastic Constitutive Laws.

This module provides the base abstractions and specific implementations for
inelastic material behavior within the ``dolfin_mech`` framework.

It extends the basic elastic material definitions to account for:

* **Plasticity**: Permanent deformations and yield surfaces.
* **Viscoelasticity**: Time-dependent stress-strain relations.
* **Damage**: Reduction of material integrity over loading cycles.

The hierarchy is rooted in the :class:`InelasticMaterial` base class, which
standardizes the handling of internal state variables.
"""

from dolfin_mech.materials import Material

################################################################################


class InelasticMaterial(Material):
	"""Base class for inelastic material models.

	This class serves as a fundamental abstraction for materials where the
	deformation is not fully reversible and may depend on the history of
	loading. In the context of ``dolfin_mech``, this includes:

	* **Plasticity:** Permanent deformation after exceeding a yield stress.
	* **Viscoelasticity:** Time-dependent material response.
	* **Damage:** Progressive degradation of material stiffness.

	Inherits from :class:`Material`.

	Subclasses are typically expected to implement internal state variables
	(e.g., back-stress, plastic strain, or damage variables) and their
	corresponding evolution laws.
	"""

	pass
