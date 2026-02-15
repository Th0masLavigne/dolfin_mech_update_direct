# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Defines the SurfacePressure operators.

Implement external pressure loads
for both finite strain (as a follower load that remains normal to the deformed
surface) and small strain frameworks.
"""

import dolfin

from ...core import TimeVaryingConstant
from ..operator import Operator

################################################################################


class SurfacePressure(Operator):
	r"""Operator representing an external surface pressure (follower load) in a large deformation (finite strain) framework.

	Unlike a simple traction, surface pressure is a "follower load," meaning
	its direction remains normal to the deformed surface. This operator
	uses Nanson's formula to pull back the spatial pressure contribution to
	the reference configuration.

	The residual contribution is:

	.. math::
	    \delta \Pi_{ext} = - \int_{\Gamma_0} -P \mathbf{N} \cdot \mathbf{F}^{-1} \cdot \delta \mathbf{u} \, J d\Gamma_0

	where:
	    - :math:`P` is the scalar pressure value.
	    - :math:`\mathbf{N}` is the unit normal in the reference configuration.
	    - :math:`\mathbf{F}^{-1}` is the inverse of the deformation gradient.
	    - :math:`J` is the determinant of the deformation gradient.

	Attributes:
	    measure (dolfin.Measure): Boundary measure (typically ``ds``).
	    tv_P (TimeVaryingConstant): Time-varying pressure value :math:`P`.
	    res_form (UFL form): The resulting residual variational form.
	"""

	def __init__(self, U_test, kinematics, N, measure, P_val=None, P_ini=None, P_fin=None):
		"""Initializes the SurfacePressureLoadingOperator.

		:param U_test: Test function (virtual displacement).
		:type U_test: dolfin.TestFunction
		:param kinematics: Kinematics object providing the deformation gradient and Jacobian.
		:type kinematics: dmech.Kinematics
		:param N: Unit normal vector in the reference configuration.
		:type N: dolfin.Constant or dolfin.Expression
		:param measure: Dolfin measure for the boundary integration.
		:type measure: dolfin.Measure
		:param P_val: Static pressure value.
		:type P_val: float, optional
		:param P_ini: Initial pressure value for time-varying loads.
		:type P_ini: float, optional
		:param P_fin: Final pressure value for time-varying loads.
		:type P_fin: float, optional
		"""
		self.measure = measure

		self.tv_P = TimeVaryingConstant(val=P_val, val_ini=P_ini, val_fin=P_fin)
		P = self.tv_P.val

		T = dolfin.dot(-P * N, dolfin.inv(kinematics.F))
		self.res_form = -dolfin.inner(T, U_test) * kinematics.J * self.measure

	def set_value_at_t_step(self, t_step):
		"""Updates the pressure value for the current time step.

		:param t_step: Current normalized time step (0.0 to 1.0).
		:type t_step: float
		"""
		self.tv_P.set_value_at_t_step(t_step)


################################################################################


class SurfacePressure0(Operator):
	r"""Operator representing an external surface pressure in a  small strain (linearized) framework.

	In the small strain assumption, the geometry is assumed to be undeformed.
	The pressure acts in the direction of the initial normal :math:`\mathbf{N}`,
	and no follower-load effects are considered.

	The residual contribution is:

	.. math::
	    \delta \Pi_{ext} = - \int_{\Gamma} -P \mathbf{N} \cdot \delta \mathbf{u} \, d\Gamma

	Attributes:
	    measure (dolfin.Measure): Boundary measure (typically ``ds``).
	    tv_P (TimeVaryingConstant): Time-varying pressure value :math:`P`.
	    res_form (UFL form): The resulting residual variational form.
	"""

	def __init__(self, U_test, N, measure, P_val=None, P_ini=None, P_fin=None):
		"""Initializes the SurfacePressure0LoadingOperator.

		:param U_test: Test function (virtual displacement).
		:type U_test: dolfin.TestFunction
		:param N: Unit normal vector.
		:type N: dolfin.Constant or dolfin.Expression
		:param measure: Dolfin measure for the boundary integration.
		:type measure: dolfin.Measure
		:param P_val: Static pressure value.
		:param P_ini: Initial pressure value.
		:param P_fin: Final pressure value.
		"""
		self.measure = measure

		self.tv_P = TimeVaryingConstant(val=P_val, val_ini=P_ini, val_fin=P_fin)
		P = self.tv_P.val

		self.res_form = -dolfin.inner(-P * N, U_test) * self.measure

	def set_value_at_t_step(self, t_step):
		"""Updates the pressure value for the current time step."""
		self.tv_P.set_value_at_t_step(t_step)
