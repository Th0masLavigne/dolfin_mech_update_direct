# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""suammary"""

import dolfin

from ..core import TimeVaryingConstant
from .operator import Operator

################################################################################


class Inertia(Operator):
	r"""Operator representing the inertial forces in a dynamic mechanical system.

	This operator assembles the residual contribution of the inertial terms
	using a backward Euler time-discretization scheme for the velocity.
	It is derived from the kinetic energy of the system.

	The kinetic energy functional :math:`\Pi_{kin}` is defined as:

	.. math::
	    \Pi_{kin} = \int_{\Omega} \frac{1}{2} \rho \mathbf{V} \cdot \mathbf{V} \, d\Omega

	where the velocity :math:`\mathbf{V}` is approximated by:

	.. math::
	    \mathbf{V} \approx \frac{\mathbf{u} - \mathbf{u}_{old}}{\Delta t}

	The residual form is obtained by taking the directional derivative of
	:math:`\Pi_{kin}` with respect to the displacement :math:`\mathbf{u}`
	in the direction of the test function :math:`\delta \mathbf{u}`.

	Attributes:
	    measure (dolfin.Measure): The integration measure (typically ``dx``).
	    tv_rho (TimeVaryingConstant): Time-varying mass density :math:`\rho`.
	    tv_dt (TimeVaryingConstant): Time step size :math:`\Delta t`.
	    res_form (UFL form): The resulting residual variational form.

	:param U: Current displacement field.
	:type U: dolfin.Function
	:param U_old: Displacement field from the previous time step.
	:type U_old: dolfin.Function
	:param U_test: Test function (virtual displacement).
	:type U_test: dolfin.TestFunction
	:param measure: Dolfin measure for domain integration.
	:type measure: dolfin.Measure
	:param rho_val: Static value for mass density.
	:type rho_val: float, optional
	:param rho_ini: Initial value for time-varying mass density.
	:type rho_ini: float, optional
	:param rho_fin: Final value for time-varying mass density.
	:type rho_fin: float, optional
	"""

	def __init__(self, U, U_old, U_test, measure, rho_val=None, rho_ini=None, rho_fin=None):
		"""Initializes the InertiaOperator."""
		self.measure = measure

		self.tv_rho = TimeVaryingConstant(val=rho_val, val_ini=rho_ini, val_fin=rho_fin)
		rho = self.tv_rho.val

		self.tv_dt = TimeVaryingConstant(0.0)
		dt = self.tv_dt.val

		# Pi = (rho/2/dt) * dolfin.inner(U, U)**2 * self.measure # MG20221108: What was that?!

		V = (U - U_old) / dt
		Pi = (rho / 2) * dolfin.inner(V, V) * self.measure
		self.res_form = dolfin.derivative(Pi, U, U_test)

	def set_value_at_t_step(self, t_step):
		"""Updates the time-varying mass density based on the current time step.

		:param t_step: Current normalized time step (0.0 to 1.0).
		:type t_step: float
		"""
		self.tv_rho.set_value_at_t_step(t_step)

	def set_dt(self, dt):
		r"""Updates the time step size used in the velocity approximation.

		:param dt: Current time step size :math:`\Delta t`.
		:type dt: float
		"""
		self.tv_dt.set_value(dt)
