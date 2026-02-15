# coding=utf8

################################################################################
###                                                                          ###
### Created by Mahdi Manoochehrtayebi, 2020-2024                             ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Martin Genet, 2018-2025                                              ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Defines the DeformedSolidVolume class.

Enforces the kinematic
relationship between the initial and current solid volume through the
Jacobian of the deformation gradient in a variational framework.
"""

import dolfin

from ..operator import Operator

# ################################################################################


class DeformedSolidVolume(Operator):
	r"""Operator representing the evolution of the solid phase volume in a deforming continuum.

	In the context of nonlinear continuum mechanics and porous media, the
	local volume ratio of the solid phase is governed by the Jacobian of the
	transformation. This operator enforces the weak relationship between the
	actual solid volume :math:`v_s`, the initial solid volume :math:`V_{s0}`,
	and the Jacobian :math:`J`.

	The governing equation in strong form is:

	.. math::
	    v_s = J \cdot V_{s0}

	Which is implemented here in weak form as:

	.. math::
	    \mathcal{R} = \int_{\Omega} \left( \frac{v_s}{V_{s0}} - J \right) \delta v_s \, d\Omega

	Attributes:
	    Vs0 (dolfin.Constant): The reference (initial) solid volume.
	    measure (dolfin.Measure): Integration measure (typically ``dx``).
	    res_form (UFL expression): The resulting residual variational form.
	"""

	def __init__(self, vs, vs_test, J, Vs0, measure):
		"""Initializes the DeformedSolidVolumeOperator.

		:param vs: Variable representing the current solid volume.
		:type vs: dolfin.Function or dolfin.Coefficient
		:param vs_test: Test function associated with the solid volume variable.
		:type vs_test: dolfin.TestFunction
		:param J: Determinant of the deformation gradient (Jacobian).
		:type J: ufl.algebra.Operator
		:param Vs0: Initial solid volume in the reference configuration.
		:type Vs0: float or dolfin.Constant
		:param measure: Integration measure for the domain.
		:type measure: dolfin.Measure
		"""
		self.Vs0 = dolfin.Constant(Vs0)
		self.measure = measure

		self.res_form = ((vs / self.Vs0 - J) * vs_test) * self.measure
