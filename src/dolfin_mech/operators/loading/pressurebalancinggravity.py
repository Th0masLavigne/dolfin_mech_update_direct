# coding=utf8

################################################################################
###                                                                          ###
### Created by Alice Peyraut, 2023-2024                                      ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Martin Genet, 2018-2025                                              ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""summary"""

import dolfin

from ...core import TimeVaryingConstant
from ..operator import Operator

################################################################################


class PressureBalancingGravity(Operator):
	r"""Operator representing a gravity loading balanced by a surface pressure in a large deformation (finite strain) framework.

	This operator is typically used in lung mechanics to model the pleural
	pressure gradient that balances the weight of the lung parenchyma.
	It includes a "breathing constant" to account for non-hydrostatic
	effects observed in vivo.

	The balanced pressure :math:`\tilde{P}` is defined as:

	.. math::
	    \tilde{P} = P_0 + \rho_{solid} \mathbf{f} \cdot (\mathbf{x} - \mathbf{x}_0) - C_{breath} \rho_{solid} f_z (x_y - x_{0,y})^2

	where :math:`\mathbf{x}_0` is a reference position (center of mass)
	and :math:`C_{breath}` is the breathing constant.

	Attributes:
	    tv_f (TimeVaryingConstant): Time-varying gravity vector :math:`\mathbf{f}`.
	    tv_P0 (TimeVaryingConstant): Time-varying reference pressure :math:`P_0`.
	    res_form (UFL form): The resulting residual variational form including
	        volume gravity, surface pressure work, and Lagrange multipliers
	        for equilibrium constraints.



	:param X: Reference coordinates.
	:param x0: Reference position (Lagrange multiplier).
	:param x0_test: Test function for reference position.
	:param lmbda, mu, gamma: Lagrange multipliers for force/moment/pressure balancing.
	:param p: Pressure field variable.
	:param U: Displacement field.
	:param Phis0: Initial solid volume fraction.
	:param rho_solid: Material density of the solid phase.
	:param kinematics: Kinematics object for finite strain.
	:param N: Reference normal vector.
	:param dS: Surface measure (e.g., pleural surface).
	:param dV: Volume measure.
	:param breathing_constant: Empirical constant for the pressure gradient.
	"""

	def __init__(
		self,
		X,
		x0,
		x0_test,
		lmbda,
		lmbda_test,
		mu,
		mu_test,
		p,
		p_test,
		gamma,
		gamma_test,
		U,
		U_test,
		Phis,
		Phis0,
		rho_solid,
		kinematics,
		N,
		dS,
		dV,
		breathing_constant,
		P0_val=None,
		P0_ini=None,
		P0_fin=None,
		f_val=None,
		f_ini=None,
		f_fin=None,
	):
		"""Initializes the PressureBalancingGravityLoadingOperator."""
		self.measure = dV

		self.V0 = dolfin.assemble(dolfin.Constant(1) * self.measure)
		self.Vs0 = dolfin.assemble(Phis0 * self.measure)

		self.tv_f = TimeVaryingConstant(val=f_val, val_ini=f_ini, val_fin=f_fin)
		f = self.tv_f.val

		self.tv_P0 = TimeVaryingConstant(val=P0_val, val_ini=P0_ini, val_fin=P0_fin)
		P0 = self.tv_P0.val

		nf = dolfin.dot(N, dolfin.inv(kinematics.F))
		nf_norm = dolfin.sqrt(dolfin.inner(nf, nf))
		n = nf / nf_norm

		x = X + U
		x_tilde = x - x0

		P_tilde = P0
		P_tilde += dolfin.Constant(rho_solid) * dolfin.dot(f, x_tilde)
		P_tilde -= (
			dolfin.Constant(breathing_constant) * dolfin.Constant(rho_solid) * f[2] * x_tilde[1] ** 2
		)  # MG20241017: Directions hard coded from [Peyraut & Genet, 2024, BMMB]

		grads_p = dolfin.dot(dolfin.grad(p - P_tilde), dolfin.inv(kinematics.F)) - n * (
			dolfin.dot(n, dolfin.dot(dolfin.grad(p - P_tilde), dolfin.inv(kinematics.F)))
		)
		grads_p_test = dolfin.dot(dolfin.grad(p_test), dolfin.inv(kinematics.F)) - n * (
			dolfin.dot(n, dolfin.dot(dolfin.grad(p_test), dolfin.inv(kinematics.F)))
		)

		self.res_form = dolfin.Constant(1e-8) * p * p_test * kinematics.J * dV
		self.res_form -= dolfin.Constant(rho_solid) * Phis0 * dolfin.inner(f, U_test) * dV
		self.res_form -= dolfin.inner(-p * n, U_test) * nf_norm * kinematics.J * dS
		self.res_form += dolfin.Constant(rho_solid) * Phis0 * dolfin.inner(f, lmbda_test) * dV
		self.res_form += dolfin.inner(-p * n, lmbda_test) * nf_norm * kinematics.J * dS
		self.res_form -= dolfin.dot(lmbda, n) * p_test * nf_norm * kinematics.J * dS
		self.res_form -= dolfin.dot(mu, dolfin.cross(x_tilde, n)) * p_test * nf_norm * kinematics.J * dS
		self.res_form += gamma * p_test * nf_norm * kinematics.J * dS
		self.res_form += dolfin.inner(grads_p, grads_p_test) * nf_norm * kinematics.J * dS
		self.res_form += dolfin.inner(dolfin.cross(x_tilde, -p * n), mu_test) * nf_norm * kinematics.J * dS
		self.res_form += (p - P_tilde) * gamma_test * nf_norm * kinematics.J * dS

		self.res_form -= (
			dolfin.inner((Phis0 * x / dolfin.Constant(self.Vs0) - x0 / dolfin.Constant(self.V0)), x0_test) * dV
		)  # MG20250909: Why?!
		# self.res_form -= Phis * dolfin.inner((x - x0), x0_test) * dV                                                       # MG20250909: This looks more correct, right?

	def set_value_at_t_step(self, t_step):
		"""Updates gravity and reference pressure for the current time step.

		:param t_step: Current normalized time step (0.0 to 1.0).
		"""
		self.tv_f.set_value_at_t_step(t_step)
		self.tv_P0.set_value_at_t_step(t_step)


################################################################################


class PressureBalancingGravity0(Operator):
	r"""Operator representing the pressure-balanced gravity loading in a small strain (linearized) framework.

	Similar to :class:`PressureBalancingGravityLoadingOperator`, but formulated
	without the finite strain Jacobian (:math:`J`) or deformation gradient
	(:math:`\mathbf{F}`) pull-backs, assuming :math:`J \approx 1` and
	:math:`\mathbf{n} \approx \mathbf{N}`.

	Attributes:
	    res_form (UFL form): Linearized residual form.


	:param x: Spatial coordinates.
	:param phis: Solid volume fraction.
	:param n: Normal vector.
	:param dS: Surface measure.
	:param dV: Volume measure.
	"""

	def __init__(
		self,
		x,
		x0,
		x0_test,
		u_test,
		lmbda,
		lmbda_test,
		mu,
		mu_test,
		p,
		p_test,
		gamma,
		gamma_test,
		rho_solid,
		phis,
		n,
		dS,
		dV,
		breathing_constant,
		P0_val=None,
		P0_ini=None,
		P0_fin=None,
		f_val=None,
		f_ini=None,
		f_fin=None,
	):
		"""Initializes the PressureBalancingGravity0LoadingOperator."""
		self.measure = dV

		self.tv_f = TimeVaryingConstant(val=f_val, val_ini=f_ini, val_fin=f_fin)
		f = self.tv_f.val

		self.tv_P0 = TimeVaryingConstant(val=P0_val, val_ini=P0_ini, val_fin=P0_fin)
		P0 = self.tv_P0.val

		x_tilde = x - x0

		P_tilde = P0
		P_tilde += dolfin.Constant(rho_solid) * dolfin.dot(f, x_tilde)
		P_tilde -= (
			dolfin.Constant(breathing_constant) * dolfin.Constant(rho_solid) * f[2] * x_tilde[1] ** 2
		)  # MG20241017: Directions hard coded from [Peyraut & Genet, 2024, BMMB]

		grads_p = dolfin.grad(p - P_tilde) - n * (dolfin.dot(n, dolfin.grad(p - P_tilde)))
		grads_p_test = dolfin.grad(p_test) - n * (dolfin.dot(n, dolfin.grad(p_test)))

		self.res_form = dolfin.Constant(1e-8) * p * p_test * dV
		self.res_form -= dolfin.Constant(rho_solid) * phis * dolfin.inner(f, u_test) * dV
		self.res_form -= dolfin.inner(-p * n, u_test) * dS
		self.res_form += dolfin.Constant(rho_solid) * phis * dolfin.inner(f, lmbda_test) * dV
		self.res_form += dolfin.inner(-p * n, lmbda_test) * dS
		self.res_form -= dolfin.dot(lmbda, n) * p_test * dS
		self.res_form -= dolfin.dot(mu, dolfin.cross(x_tilde, n)) * p_test * dS
		self.res_form += gamma * p_test * dS
		self.res_form += dolfin.inner(grads_p, grads_p_test) * dS
		self.res_form += dolfin.inner(dolfin.cross(x_tilde, -p * n), mu_test) * dS
		self.res_form += (p - P_tilde) * gamma_test * dS

		self.res_form -= phis * dolfin.inner((x - x0), x0_test) * dV

	def set_value_at_t_step(self, t_step):
		"""Updates gravity and reference pressure for the current time step."""
		self.tv_f.set_value_at_t_step(t_step)
		self.tv_P0.set_value_at_t_step(t_step)
