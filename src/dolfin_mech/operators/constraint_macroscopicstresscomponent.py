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

"""Defines the MacroscopicStressComponentConstraint class.

Enforces
specific macroscopic stress components on a Representative Elementary
Volume (REV) while accounting for microscopic pore pressure and
volumetric changes.
"""

import dolfin

from ..core import TimeVaryingConstant
from .operator import Operator

################################################################################


class MacroscopicStressComponentConstraint(Operator):
	r"""Operator to enforce a constraint on a single component of the macroscopic stress tensor.

	In multiscale mechanics, this operator is used to prescribe the macroscopic stress
	component :math:`\bar{\sigma}_{ij}` on a Representative Elementary Volume (REV).
	It relates the microscopic state (stresses and volume changes) to the
	desired macroscopic loading.

	The residual contribution follows the volume-averaged equilibrium:

	.. math::
	    \mathcal{R} = \delta\bar{U}_{ij} \int_{\Omega_0} \left( \tilde{\sigma}_{ij} - \frac{v}{V_{s0}} \bar{\sigma}_{ij} \right) d\Omega_0

	Where:
	    - :math:`\tilde{\sigma}` is the corrected microscopic stress accounting for pore pressure.
	    - :math:`v` is the current volume of the REV.
	    - :math:`V_{s0}` is the initial solid volume.
	    - :math:`\bar{\sigma}_{ij}` is the target macroscopic stress component.

	:param U_bar: Macroscopic displacement gradient tensor.
	:type U_bar: dolfin.Coefficient
	:param U_bar_test: Test function associated with the macroscopic displacement gradient.
	:type U_bar_test: dolfin.Argument
	:param kinematics: Microscopic kinematics object.
	:param material: Microscopic material object providing the stress tensor.
	:param V0: Initial total volume of the REV.
	:param Vs0: Initial solid volume of the REV.
	:param i: Row index of the stress component to constrain.
	:param j: Column index of the stress component to constrain.
	:param measure: Dolfin measure for integration.
	:param N: Normal vector (if applicable for surface constraints).
	:param sigma_bar_ij_val: Static value for target macroscopic stress.
	:param sigma_bar_ij_ini: Initial value for time-varying macroscopic stress.
	:param sigma_bar_ij_fin: Final value for time-varying macroscopic stress.
	:param pf_val: Static value for pore pressure.
	:param pf_ini: Initial value for time-varying pore pressure.
	:param pf_fin: Final value for time-varying pore pressure.

	Attributes:
	    kinematics (Kinematics): Microscopic kinematic variables.
	    material (Material): Microscopic material law.
	    measure (dolfin.Measure): Integration measure (typically ``dx``).
	    tv_pf (TimeVaryingConstant): Time-varying pore pressure :math:`p_f`.
	    tv_sigma_bar_ij (TimeVaryingConstant): Time-varying target macroscopic stress :math:`\bar{\sigma}_{ij}`.
	    res_form (UFL Form): The resulting residual variational form.
	"""

	def __init__(
		self,
		U_bar,
		U_bar_test,
		kinematics,
		material,
		V0,
		Vs0,
		i,
		j,
		measure,
		N,
		sigma_bar_ij_val=None,
		sigma_bar_ij_ini=None,
		sigma_bar_ij_fin=None,
		pf_val=None,
		pf_ini=None,
		pf_fin=None,
	):
		"""Initializes the MacroscopicStressComponentConstraintOperator."""
		self.kinematics = kinematics
		self.material = material
		self.measure = measure
		self.N = N

		self.tv_pf = TimeVaryingConstant(val=pf_val, val_ini=pf_ini, val_fin=pf_fin)
		pf = self.tv_pf.val

		self.tv_sigma_bar_ij = TimeVaryingConstant(
			val=sigma_bar_ij_val, val_ini=sigma_bar_ij_ini, val_fin=sigma_bar_ij_fin
		)
		sigma_bar_ij = self.tv_sigma_bar_ij.val

		dim = U_bar.ufl_shape[0]
		I_bar = dolfin.Identity(dim)
		F_bar = I_bar + U_bar
		J_bar = dolfin.det(F_bar)
		v = J_bar * V0

		sigma_tilde = self.material.sigma * self.kinematics.J - (v / Vs0 - self.kinematics.J) * pf * I_bar
		self.res_form = U_bar_test[i, j] * (sigma_tilde[i, j] - (v / Vs0) * sigma_bar_ij) * self.measure

	def set_value_at_t_step(self, t_step):
		"""Updates the target macroscopic stress and pore pressure constants for the current time step.

		:param t_step: Current time step (0.0 to 1.0).
		:type t_step: float
		"""
		self.tv_pf.set_value_at_t_step(t_step)
		self.tv_sigma_bar_ij.set_value_at_t_step(t_step)
