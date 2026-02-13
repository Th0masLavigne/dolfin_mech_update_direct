# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Cécile Patte, 2019-2021                                              ###
###                                                                          ###
### INRIA, Palaiseau, France                                                 ###
###                                                                          ###
################################################################################

"""Lung tissue volumetric elastic potential.

This module implements specific logarithmic energy potentials designed for
lung parenchymal tissue modeling.
"""

import dolfin

from .Elastic import ElasticMaterial

################################################################################


class WbulkLung(ElasticMaterial):
	r"""Class representing the bulk (volumetric) elastic energy contribution specifically tailored for lung tissue.

	This material model uses a logarithmic energy potential to penalize deviations
	from the reference solid volume fraction :math:`\Phi_{s0}`.

	The strain energy density function is defined as:

	.. math::
	    \Psi = \kappa \left( \frac{\Phi_s}{\Phi_{s0}} - 1 - \ln\left( \frac{\Phi_s}{\Phi_{s0}} \right) \right)

	Where:
	    - :math:`\kappa` is the bulk modulus (penalty parameter).
	    - :math:`\Phi_s` is the current solid volume fraction.
	    - :math:`\Phi_{s0}` is the reference solid volume fraction.

	Attributes:
	    kappa (dolfin.Constant): Bulk modulus parameter.
	    Psi (UFL expression): The calculated strain energy density.
	    dWbulkdPhis (UFL expression): The derivative of the energy with respect to
	        the solid volume fraction :math:`\frac{\partial \Psi}{\partial \Phi_s}`.
	"""

	def __init__(self, Phis, Phis0, parameters):
		"""Initializes the WbulkLungElasticMaterial.

		Args:
		    Phis: Current solid volume fraction.
		    Phis0: Reference solid volume fraction.
		    parameters: Dictionary containing material parameters.
		        Must include 'kappa'.

		Raises:
		    AssertionError: If 'kappa' is not present in the parameters.
		"""
		assert "kappa" in parameters
		self.kappa = dolfin.Constant(parameters["kappa"])

		Phis = dolfin.variable(Phis)
		self.Psi = self.kappa * (Phis / Phis0 - 1 - dolfin.ln(Phis / Phis0))
		self.dWbulkdPhis = dolfin.diff(self.Psi, Phis)
