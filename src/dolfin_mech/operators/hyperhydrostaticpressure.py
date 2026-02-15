# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Defines the HyperHydrostaticPressureOperator class.

Implements the
virtual work contribution of a hydrostatic pressure field, typically used
to enforce incompressibility or volumetric constraints in large deformation
mixed formulations.
"""

import dolfin

from .operator import Operator

################################################################################


class HyperHydrostaticPressure(Operator):
	r"""Operator representing the virtual work of a hydrostatic pressure in a large	deformation (hyperelastic).

	This operator accounts for the internal work contribution of a pressure field
	:math:`P`. In a mixed formulation, :math:`P` is often a Lagrange multiplier
	used to enforce incompressibility or a specific volumetric constraint.

	The residual contribution is the variation of the volumetric work:

	.. math::
	    \delta \Pi_{pres} = - \int_{\Omega_0} P \delta J \, d\Omega_0

	Where:
	    - :math:`P` is the hydrostatic pressure.
	    - :math:`J = \det(\mathbf{F})` is the volume ratio (Jacobian).
	    - :math:`\delta J` is the variation of the Jacobian with respect to the displacement.

	Attributes:
	    kinematics (Kinematics): Kinematic variables providing the Jacobian :math:`J`.
	    P (dolfin.Function or dolfin.Constant): The hydrostatic pressure field.
	    measure (dolfin.Measure): Integration measure (typically ``dx``).
	    res_form (UFL form): The resulting residual variational form.

	:param kinematics: Kinematics object providing the Jacobian and displacement.
	:type kinematics: dmech.Kinematics
	:param U_test: Test function associated with the displacement (virtual displacement).
	:type U_test: dolfin.TestFunction
	:param P: Hydrostatic pressure field or constant.
	:type P: dolfin.Function, dolfin.Constant, or dolfin.TrialFunction
	:param measure: Dolfin measure for domain integration.
	:type measure: dolfin.Measure
	"""

	def __init__(self, kinematics, U_test, P, measure):
		"""Initializes the HyperHydrostaticPressureOperator.

		:param kinematics: Kinematics object providing the Jacobian and displacement.
		:type kinematics: dmech.Kinematics
		:param U_test: Test function associated with the displacement (virtual displacement).
		:type U_test: dolfin.TestFunction
		:param P: Hydrostatic pressure field or constant.
		:type P: dolfin.Function, dolfin.Constant, or dolfin.TrialFunction
		:param measure: Dolfin measure for domain integration.
		:type measure: dolfin.Measure
		"""
		self.kinematics = kinematics
		self.P = P
		self.measure = measure

		dJ_test = dolfin.derivative(self.kinematics.J, self.kinematics.U, U_test)

		self.res_form = -self.P * dJ_test * self.measure
