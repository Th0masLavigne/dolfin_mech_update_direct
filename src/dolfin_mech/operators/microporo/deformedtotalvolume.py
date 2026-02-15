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
"""Defines the DeformedTotalVolume class.

Enforces the kinematic
consistency between the total volume of a Representative Elementary Volume (REV)
and the macroscopic deformation gradient in multiscale homogenization problems.
"""

import dolfin

from ..operator import Operator

# ################################################################################


class DeformedTotalVolume(Operator):
	r"""Operator representing the total volume of a Representative Elementary Volume (REV) subjected to macroscopic deformation.

	In computational homogenization, the total volume of a porous or heterogeneous
	REV evolves according to the macroscopic deformation gradient :math:`\bar{\mathbf{F}}`.
	This operator enforces the kinematic relationship between the current total
	volume :math:`v` and the reference total volume :math:`V_0`.

	The relationship is governed by the macroscopic Jacobian :math:`\bar{J}`:

	.. math::
	    v = \bar{J} \cdot V_0

	The residual form is normalized by the initial solid volume :math:`V_{s0}`
	to maintain consistent scaling in multi-field porous media problems:

	.. math::
	    \mathcal{R} = \int_{\Omega} \frac{(v - \bar{J} V_0)}{V_{s0}} \delta v \, d\Omega

	Attributes:
	    measure (dolfin.Measure): Integration measure (typically ``dx``).
	    res_form (UFL expression): The resulting residual variational form.
	"""

	def __init__(self, v, v_test, U_bar, V0, Vs0, measure):
		r"""Initializes the DeformedTotalVolumeOperator.

		:param v: Variable representing the current total volume of the REV.
		:type v: dolfin.Function or dolfin.Coefficient
		:param v_test: Test function associated with the total volume variable.
		:type v_test: dolfin.TestFunction
		:param U_bar: Macroscopic displacement gradient tensor :math:`\bar{\mathbf{U}}`.
		:type U_bar: dolfin.Coefficient
		:param V0: Initial total volume of the REV in the reference configuration.
		:type V0: float or dolfin.Constant
		:param Vs0: Initial solid volume of the REV (used for scaling).
		:type Vs0: float or dolfin.Constant
		:param measure: Integration measure for the domain.
		:type measure: dolfin.Measure
		"""
		self.measure = measure

		dim = U_bar.ufl_shape[0]
		F_bar = dolfin.Identity(dim) + U_bar
		J_bar = dolfin.det(F_bar)

		self.res_form = ((v - J_bar * V0) * v_test) / Vs0 * self.measure
