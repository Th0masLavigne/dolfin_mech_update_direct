# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Defines the SurfaceTension operators.

Implement the virtual work of
surface-bound tension forces for finite strain (including area-dependent
surfactant modeling) and small strain mechanical frameworks.
"""

import dolfin

from ...core import TimeVaryingConstant
from ..operator import Operator

################################################################################


class SurfaceTension(Operator):
	r"""Operator representing surface tension effects in a finite strain (large deformation) framework.

	This operator assembles the virtual work done by surface tension on a boundary.
	Surface tension acts as a membrane stress within the tangent plane of the deformed
	surface. It supports a constant tension coefficient or a surface-area-dependent
	tension (e.g., to model surfactant effects in lung alveoli).

	The virtual work of surface tension is given by:

	.. math::
	    \delta \Pi_{surf} = \int_{\Gamma} \gamma \mathbf{P} : \nabla_s \delta \mathbf{u} \, d\Gamma

	where:
	    - :math:`\gamma` is the surface tension coefficient.
	    - :math:`\mathbf{P} = \mathbf{I} - \mathbf{n} \otimes \mathbf{n}` is the
	      projector onto the tangent plane of the deformed surface.
	    - :math:`\mathbf{n}` is the current unit normal vector.
	    - :math:`d\Gamma = J \|\mathbf{F}^{-T} \mathbf{N}\| d\Gamma_0` is the deformed
	      surface element (Nanson's formula).

	Attributes:
	    measure (dolfin.Measure): Boundary measure (typically ``ds``).
	    tv_gamma (TimeVaryingConstant): Time-varying surface tension coefficient :math:`\gamma`.
	    kinematics (Kinematics): Kinematics object for deformation variables.
	    res_form (UFL form): The resulting residual variational form.
	"""

	def __init__(
		self, kinematics, N, measure, U_test, tension_params={}, gamma_val=None, gamma_ini=None, gamma_fin=None
	):
		"""Initializes the SurfaceTensionLoadingOperator.

		:param kinematics: Kinematics object providing F and J.
		:type kinematics: dmech.Kinematics
		:param N: Unit normal vector in the reference configuration.
		:type N: dolfin.Constant or dolfin.Expression
		:param measure: Dolfin measure for boundary integration.
		:type measure: dolfin.Measure
		:param U_test: Test function (virtual displacement).
		:type U_test: dolfin.TestFunction
		:param tension_params: Dictionary for area-dependency parameters (d1, d2, d3).
		:type tension_params: dict, optional
		:param gamma_val, gamma_ini, gamma_fin: Static or time-varying surface tension values.
		"""
		self.measure = measure

		self.tv_gamma = TimeVaryingConstant(val=gamma_val, val_ini=gamma_ini, val_fin=gamma_fin)
		gamma = self.tv_gamma.val

		self.N = N
		self.kinematics = kinematics

		self.tv_gamma = TimeVaryingConstant(val=gamma_val, val_ini=gamma_ini, val_fin=gamma_fin)
		gamma = self.tv_gamma.val

		dim = U_test.ufl_shape[0]
		I = dolfin.Identity(dim)
		FmTN = dolfin.dot(dolfin.inv(kinematics.F).T, N)
		T = dolfin.sqrt(dolfin.inner(FmTN, FmTN))
		n = FmTN / T
		P = I - dolfin.outer(n, n)

		S_hat = kinematics.J * T

		surface_dependancy = tension_params.get("surface_dependancy", None)
		if surface_dependancy == 1:
			d1 = dolfin.Constant(tension_params.get("d1"))
			d2 = dolfin.Constant(tension_params.get("d2"))
			d3 = dolfin.Constant(tension_params.get("d3"))

			gamma = gamma * (d1 / (1 + (S_hat / d2) ** (d3)))

		taus = gamma * P

		self.res_form = dolfin.inner(taus, dolfin.dot(P, (dolfin.grad(U_test)))) * self.kinematics.J * T * self.measure

	def set_value_at_t_step(self, t_step):
		"""Updates the surface tension coefficient for the current time step."""
		self.tv_gamma.set_value_at_t_step(t_step)


################################################################################


class SurfaceTension0(Operator):
	"""Operator representing surface tension effects in a linearized (small strain) framework.

	In small strain theory, the virtual work is derived by considering the
	variation of the surface area approximated by the trace of the strain
	within the tangent plane.

	Attributes:
	    res_form (UFL form): The resulting residual variational form.
	"""

	def __init__(self, u, u_test, kinematics, N, measure, gamma_val=None, gamma_ini=None, gamma_fin=None):
		"""Initializes the SurfaceTension0LoadingOperator.

		:param u: Displacement field.
		:param u_test: Test function.
		:param kinematics: Kinematics object for infinitesimal strain.
		:param N: Reference normal vector.
		:param measure: Boundary measure.
		"""
		self.measure = measure

		self.tv_gamma = TimeVaryingConstant(val=gamma_val, val_ini=gamma_ini, val_fin=gamma_fin)
		gamma = self.tv_gamma.val

		dim = u.ufl_shape[0]
		I = dolfin.Identity(dim)
		Pi = gamma * (1 + dolfin.inner(kinematics.epsilon, I - dolfin.outer(N, N))) * self.measure
		self.res_form = dolfin.derivative(Pi, u, u_test)  # MG20211220: Is that correct?!

	def set_value_at_t_step(self, t_step):
		"""Updates the surface tension coefficient for the current time step."""
		self.tv_gamma.set_value_at_t_step(t_step)
