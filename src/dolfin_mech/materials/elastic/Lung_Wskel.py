# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Lung skeletal tissue elasticity implementation.

This module provides a composite hyperelastic model for lung parenchyma,
combining Exponential-Ogden bulk and Neo-Hookean deviatoric components.
"""

import dolfin_mech.materials as materials

from .Elastic import ElasticMaterial

################################################################################


class WskelLung(ElasticMaterial):
	r"""Class representing the skeletal (solid phase) elastic energy of the lung parenchyma.

	This model is a composite hyperelastic material that splits the strain energy
	density into a volumetric (bulk) part and a deviatoric (shear) part to
	capture the complex non-linear response of lung tissue.

	The total strain energy density is defined as:

	.. math::
	    \Psi_{total} = \Psi_{bulk} + \Psi_{dev}

	Where:
	    - :math:`\Psi_{bulk}` is modeled using an
	      :class:`ExponentialOgdenCiarletGeymonatElasticMaterial`.
	    - :math:`\Psi_{dev}` is modeled using a
	      :class:`NeoHookeanMooneyRivlinElasticMaterial`.

	Attributes:
	    kinematics (Kinematics): Kinematic variables associated with the deformation.
	    bulk (ElasticMaterial): The volumetric component of the lung skeleton elasticity.
	    dev (ElasticMaterial): The deviatoric component of the lung skeleton elasticity.
	    Psi (UFL expression): Total strain energy density.
	    Sigma (UFL expression): Total Second Piola-Kirchhoff stress tensor.
	    P (UFL expression): Total First Piola-Kirchhoff stress tensor.
	    sigma (UFL expression): Total Cauchy stress tensor.
	"""

	def __init__(self, kinematics, parameters):
		"""Initializes the WskelLungElasticMaterial.

		Args:
		    kinematics: Kinematics object containing deformation tensors.
		    parameters: Dictionary of material parameters. These must satisfy
		        the requirements for both the Exponential-Ogden and
		        Neo-Hookean sub-models.
		"""
		self.kinematics = kinematics

		self.bulk = materials.elastic.ExponentialOgdenCiarletGeymonat(kinematics, parameters)
		self.dev = materials.elastic.NeoHookeanMooneyRivlin(kinematics, parameters)

		self.Psi = self.bulk.Psi + self.dev.Psi
		self.Sigma = self.bulk.Sigma + self.dev.Sigma
		self.P = self.bulk.P + self.dev.P
		self.sigma = self.bulk.sigma + self.dev.sigma
