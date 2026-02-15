# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Defines the VolumeForce operator.

Implement the virtual work of external
body forces (such as gravity or electromagnetism) for both finite strain
(accounting for volumetric Jacobian scaling) and small strain frameworks.
"""

import dolfin

from ..operator import Operator

################################################################################


class VolumeForce(Operator):
	r"""Operator representing an external volume (body) force in a large deformation (finite strain) framework.

	This operator assembles the virtual work done by an external force density
	applied to the volume of the body. In a finite strain context, the
	integration is performed over the reference configuration, and the
	contribution is scaled by the Jacobian :math:`J` to account for volume
	changes.

	The residual contribution is:

	.. math::
	    \delta \Pi_{ext} = - \int_{\Omega_0} \mathbf{F} \cdot \delta \mathbf{u} \, J \, d\Omega_0

	where:
	    - :math:`\mathbf{F}` is the force vector per unit of spatial volume.
	    - :math:`J = \det(\mathbf{F}_{grad})` is the volume ratio.
	    - :math:`\delta \mathbf{u}` is the virtual displacement.

	Attributes:
	    measure (dolfin.Measure): Domain measure (typically ``dx``).
	    tv_F (TimeVaryingConstant): Time-varying force density vector :math:`\mathbf{F}`.
	    res_form (UFL form): The resulting residual variational form.
	"""

	def __init__(self, U_test, kinematics, measure, F_val=None, F_ini=None, F_fin=None):
		"""Initializes the VolumeForceLoadingOperator.

		:param U_test: Test function (virtual displacement).
		:type U_test: dolfin.TestFunction
		:param kinematics: Kinematics object providing the Jacobian :math:`J`.
		:type kinematics: dmech.Kinematics
		:param measure: Dolfin measure for domain integration (e.g., ``dx``).
		:type measure: dolfin.Measure
		:param F_val: Static body force value.
		:type F_val: list[float] or dolfin.Constant, optional
		:param F_ini: Initial body force value for time-varying loads.
		:type F_ini: list[float] or dolfin.Constant, optional
		:param F_fin: Final body force value for time-varying loads.
		:type F_fin: list[float] or dolfin.Constant, optional
		"""
		self.measure = measure

		self.tv_F = dmech.TimeVaryingConstant(val=F_val, val_ini=F_ini, val_fin=F_fin)
		F = self.tv_F.val

		self.res_form = -dolfin.inner(F, U_test) * kinematics.J * self.measure

	def set_value_at_t_step(self, t_step):
		"""Updates the body force vector for the current time step.

		:param t_step: Current normalized time step (0.0 to 1.0).
		:type t_step: float
		"""
		self.tv_F.set_value_at_t_step(t_step)


################################################################################


class VolumeForce0(Operator):
	r"""Operator representing an external volume (body) force in a small strain (linearized) framework.

	In the small strain framework, volume changes are neglected (:math:`J \approx 1`),
	and the force is integrated directly over the undeformed domain.

	The residual contribution is:

	.. math::
	    \delta \Pi_{ext} = - \int_{\Omega} \mathbf{F} \cdot \delta \mathbf{u} \, d\Omega

	Attributes:
	    measure (dolfin.Measure): Domain measure (typically ``dx``).
	    tv_F (TimeVaryingConstant): Time-varying force density vector :math:`\mathbf{F}`.
	    res_form (UFL form): The resulting residual variational form.
	"""

	def __init__(self, U_test, measure, F_val=None, F_ini=None, F_fin=None):
		"""Initializes the VolumeForce0LoadingOperator.

		:param U_test: Test function (virtual displacement).
		:type U_test: dolfin.TestFunction
		:param measure: Dolfin measure for domain integration.
		:type measure: dolfin.Measure
		:param F_val: Static body force value.
		:param F_ini: Initial body force value.
		:param F_fin: Final body force value.
		"""
		self.measure = measure

		self.tv_F = dmech.TimeVaryingConstant(val=F_val, val_ini=F_ini, val_fin=F_fin)
		F = self.tv_F.val

		self.res_form = -dolfin.inner(F, U_test) * self.measure

	def set_value_at_t_step(self, t_step):
		"""Updates the body force vector for the current time step."""
		self.tv_F.set_value_at_t_step(t_step)
