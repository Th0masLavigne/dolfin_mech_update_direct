# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Defines the LinearizedHydrostaticPressure class.

Implements the
coupling term between the pressure field and volumetric strain within
a small-strain mixed variational framework.
"""

import dolfin

from .operator import Operator

################################################################################


class LinearizedHydrostaticPressure(Operator):
	r"""Operator representing the virtual work of hydrostatic pressure in a linearized (small strain) elasticity framework.

	In mixed displacement-pressure formulations, this operator defines the
	coupling term between the pressure field :math:`p` and the displacement
	test function :math:`\delta \mathbf{u}`. Physically, it represents the
	internal virtual work done by the pressure during a volumetric change.

	The internal virtual work contribution :math:`\delta \Pi_{pres}` is:

	.. math::
	    \delta \Pi_{pres} = - \int_{\Omega} p \, \text{tr}(\delta \mathbf{\epsilon}) \, d\Omega

	where:
	    - :math:`p` is the hydrostatic pressure (often a Lagrange multiplier).
	    - :math:`\delta \mathbf{\epsilon} = \text{sym}(\nabla \delta \mathbf{u})` is the
	      virtual infinitesimal strain tensor.
	    - :math:`\text{tr}(\delta \mathbf{\epsilon}) = \nabla \cdot \delta \mathbf{u}`
	      represents the virtual linearized volumetric strain.

	Attributes:
	    kinematics (Kinematics): Kinematic variables associated with the small strain problem.
	    p (dolfin.Function or dolfin.Constant): The hydrostatic pressure field.
	    measure (dolfin.Measure): Integration measure (typically ``dx``).
	    res_form (UFL form): The resulting residual variational form.


	:param kinematics: Kinematics object handling infinitesimal strains.
	:type kinematics: dmech.Kinematics
	:param u_test: The displacement test function (virtual displacement).
	:type u_test: dolfin.TestFunction
	:param p: Hydrostatic pressure field or constant.
	:type p: dolfin.Function, dolfin.Constant, or dolfin.TrialFunction
	:param measure: Dolfin measure for domain integration.
	:type measure: dolfin.Measure
	"""

	def __init__(self, kinematics, u_test, p, measure):
		"""Initializes the LinearizedHydrostaticPressureOperator."""
		self.kinematics = kinematics
		self.p = p
		self.measure = measure

		epsilon_test = dolfin.sym(dolfin.grad(u_test))
		self.res_form = -self.p * dolfin.tr(epsilon_test) * self.measure
