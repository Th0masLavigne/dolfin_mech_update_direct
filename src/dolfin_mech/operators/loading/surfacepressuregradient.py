# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Defines the SurfacePressureGradient operators.

Implement external
pressure loads with linear spatial gradients (e.g., hydrostatic pressure)
for both finite strain follower-load and small strain frameworks.
"""

import dolfin

from ..operator import Operator

################################################################################


class SurfacePressureGradient(Operator):
	r"""Operator representing an external surface pressure with a spatial gradient in a large deformation (finite strain) framework.

	This operator applies a pressure field :math:`P` that varies linearly along
	a specified direction :math:`\mathbf{N}_0`. This is typically used to
	model hydrostatic pressure (e.g., :math:`P = P_0 + \rho g z`).

	The spatially varying pressure is defined as:

	.. math::
	    P(\mathbf{x}) = P_0 + \Delta P (\mathbf{x} - \mathbf{X}_0) \cdot \mathbf{N}_0

	where:
	    - :math:`P_0` is the reference pressure at point :math:`\mathbf{X}_0`.
	    - :math:`\Delta P` is the pressure gradient magnitude.
	    - :math:`\mathbf{N}_0` is the direction of the gradient.
	    - :math:`\mathbf{x}` is the current position (:math:`\mathbf{X} + \mathbf{u}`).

	As a "follower load," the pressure direction remains normal to the deformed
	surface using Nanson's formula.

	Attributes:
	    tv_X0 (TimeVaryingConstant): Reference position :math:`\mathbf{X}_0`.
	    tv_N0 (TimeVaryingConstant): Gradient direction :math:`\mathbf{N}_0`.
	    tv_P0 (TimeVaryingConstant): Reference pressure :math:`P_0`.
	    tv_DP (TimeVaryingConstant): Pressure gradient :math:`\Delta P`.
	    res_form (UFL form): The resulting residual variational form.
	"""

	def __init__(
		self,
		X,
		U,
		U_test,
		kinematics,
		N,
		measure,
		X0_val=None,
		X0_ini=None,
		X0_fin=None,
		N0_val=None,
		N0_ini=None,
		N0_fin=None,
		P0_val=None,
		P0_ini=None,
		P0_fin=None,
		DP_val=None,
		DP_ini=None,
		DP_fin=None,
	):
		r"""Initializes the SurfacePressureGradientLoadingOperator.

		:param X: Reference coordinates.
		:param U: Displacement field.
		:param U_test: Test function (virtual displacement).
		:param kinematics: Kinematics object for finite strain variables.
		:param N: Unit normal vector in the reference configuration.
		:param measure: Dolfin measure for boundary integration (e.g., ``ds``).
		:param X0_val, X0_ini, X0_fin: Values for reference position :math:`\mathbf{X}_0`.
		:param N0_val, N0_ini, N0_fin: Values for gradient direction :math:`\mathbf{N}_0`.
		:param P0_val, P0_ini, P0_fin: Values for reference pressure :math:`P_0`.
		:param DP_val, DP_ini, DP_fin: Values for pressure gradient :math:`\Delta P`.
		"""
		self.measure = measure

		self.tv_X0 = dmech.TimeVaryingConstant(val=X0_val, val_ini=X0_ini, val_fin=X0_fin)
		X0 = self.tv_X0.val
		self.tv_N0 = dmech.TimeVaryingConstant(val=N0_val, val_ini=N0_ini, val_fin=N0_fin)
		N0 = self.tv_N0.val
		self.tv_P0 = dmech.TimeVaryingConstant(val=P0_val, val_ini=P0_ini, val_fin=P0_fin)
		P0 = self.tv_P0.val
		self.tv_DP = dmech.TimeVaryingConstant(val=DP_val, val_ini=DP_ini, val_fin=DP_fin)
		DP = self.tv_DP.val

		x = X + U
		P = P0 + DP * dolfin.inner(x - X0, N0)

		T = dolfin.dot(-P * N, dolfin.inv(kinematics.F))
		self.res_form = -dolfin.inner(T, U_test) * kinematics.J * self.measure

	def set_value_at_t_step(self, t_step):
		"""Updates all time-varying constants for the gradient load."""
		self.tv_X0.set_value_at_t_step(t_step)
		self.tv_N0.set_value_at_t_step(t_step)
		self.tv_P0.set_value_at_t_step(t_step)
		self.tv_DP.set_value_at_t_step(t_step)


################################################################################


class SurfacePressureGradient0(Operator):
	r"""Operator representing an external surface pressure with a spatial gradient in a small strain (linearized) framework.

	Similar to :class:`SurfacePressureGradientLoadingOperator`, but formulated
	under the assumption of undeformed geometry. The pressure acts along the
	reference normal :math:`\mathbf{N}`.

	The pressure is defined spatially as:

	.. math::
	    P(\mathbf{x}) = P_0 + \Delta P (\mathbf{x} - \mathbf{X}_0) \cdot \mathbf{N}_0

	Attributes:
	    res_form (UFL form): The resulting residual variational form.
	"""

	def __init__(
		self,
		x,
		U_test,
		N,
		measure,
		X0_val=None,
		X0_ini=None,
		X0_fin=None,
		N0_val=None,
		N0_ini=None,
		N0_fin=None,
		P0_val=None,
		P0_ini=None,
		P0_fin=None,
		DP_val=None,
		DP_ini=None,
		DP_fin=None,
	):
		"""Initializes the SurfacePressureGradient0LoadingOperator.

		:param x: Spatial coordinates (fixed in linear theory).
		:param U_test: Test function.
		:param N: Unit normal vector.
		:param measure: Boundary measure.
		"""
		self.measure = measure

		self.tv_X0 = dmech.TimeVaryingConstant(val=X0_val, val_ini=X0_ini, val_fin=X0_fin)
		X0 = self.tv_X0.val
		self.tv_N0 = dmech.TimeVaryingConstant(val=N0_val, val_ini=N0_ini, val_fin=N0_fin)
		N0 = self.tv_N0.val
		self.tv_P0 = dmech.TimeVaryingConstant(val=P0_val, val_ini=P0_ini, val_fin=P0_fin)
		P0 = self.tv_P0.val
		self.tv_DP = dmech.TimeVaryingConstant(val=DP_val, val_ini=DP_ini, val_fin=DP_fin)
		DP = self.tv_DP.val

		P = P0 + DP * dolfin.inner(x - X0, N0)

		self.res_form = -dolfin.inner(-P * N, U_test) * self.measure

	def set_value_at_t_step(self, t_step):
		"""Updates all time-varying constants."""
		self.tv_X0.set_value_at_t_step(t_step)
		self.tv_N0.set_value_at_t_step(t_step)
		self.tv_P0.set_value_at_t_step(t_step)
		self.tv_DP.set_value_at_t_step(t_step)
