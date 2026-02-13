# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

r"""Elastic Constitutive Laws.

This module implements the hierarchy of elastic material models within
``dolfin_mech``. It provides a unified interface for both infinitesimal
strain (linear) and finite strain (hyperelastic) formulations.

The primary abstraction is the :class:`ElasticMaterial` base class, from which
all specific material laws (e.g., Hooke, Neo-Hookean, Mooney-Rivlin) derive.

Key Features:
-------------
* **Unified Interface**: Common methods for parameter normalization.
* **Hyperelasticity**: Support for strain energy density potentials $\Psi$.
* **Linear Elasticity**: Implementation of the classic Hooke's law.
"""

from dolfin_mech.materials import Material

################################################################################


class ElasticMaterial(Material):
	r"""Base class for all elastic material models.

	This class serves as an abstraction layer for both linearized elasticity
	(e.g., Hooke) and finite-strain hyperelasticity (e.g., Neo-Hookean,
	Mooney-Rivlin). It inherits from :py:class:`Material`
	to provide access to unified parameter conversion methods.

	Derived classes are expected to implement specific strain energy density
	functions :math:`\Psi` or stress-strain relationships.
	"""

	pass
