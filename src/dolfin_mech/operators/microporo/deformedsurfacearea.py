# coding=utf8

################################################################################
###                                                                          ###
### Created by Mahdi Manoochehrtayebi, 2020-2023                             ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Defines the DeformedSurfaceArea class.

Tracks the evolution of
boundary surface area under large deformations using Nanson's formula,
enabling the simulation of area-dependent phenomena such as surface tension.
"""

import dolfin

from ..operator import Operator

# ################################################################################


class DeformedSurfaceArea(Operator):
	r"""Operator representing the evolution of surface area on a boundary under	large deformations.

	This operator facilitates the tracking of the current surface area :math:`S`
	relative to its initial area :math:`S_0`. It is particularly useful for
	problems involving surface-dependent phenomena like surface tension or
	surfactant concentration in lung mechanics.

	The change in area is governed by Nanson's formula, which relates the
	reference normal :math:`\mathbf{N}` and reference area element :math:`dS`
	to the deformed normal :math:`\mathbf{n}` and deformed area element :math:`ds`:

	.. math::
	    \mathbf{n} \, ds = J \mathbf{F}^{-T} \mathbf{N} \, dS

	The scalar ratio of area change :math:`T \cdot J` (where :math:`T = \|\mathbf{F}^{-T} \mathbf{N}\|`)
	is enforced in weak form:

	.. math::
	    \mathcal{R} = \int_{\Gamma} \left( \frac{S}{S_0} - J \|\mathbf{F}^{-T} \mathbf{N}\| \right) \delta S \, d\Gamma

	Attributes:
	    measure (dolfin.Measure): Boundary measure (typically ``ds``).
	    kinematics (Kinematics): Kinematic variables providing the deformation
	        gradient :math:`\mathbf{F}` and Jacobian :math:`J`.
	    N (dolfin.Constant): The unit normal vector in the reference configuration.
	    res_form (UFL expression): The resulting residual variational form.
	"""

	def __init__(self, S_area, S_area_test, kinematics, N, measure):
		"""Initializes the DeformedSurfaceAreaOperator.

		:param S_area: Variable representing the current surface area.
		:type S_area: dolfin.Function or dolfin.Coefficient
		:param S_area_test: Test function associated with the surface area variable.
		:type S_area_test: dolfin.TestFunction
		:param kinematics: Kinematics object for finite strain variables.
		:type kinematics: dmech.Kinematics
		:param N: Reference unit normal vector.
		:type N: dolfin.Constant
		:param measure: Integration measure for the boundary.
		:type measure: dolfin.Measure
		"""
		self.measure = measure
		self.kinematics = kinematics
		self.N = N

		FmTN = dolfin.dot(dolfin.inv(self.kinematics.F).T, self.N)
		T = dolfin.sqrt(dolfin.inner(FmTN, FmTN))
		S0 = dolfin.assemble(1 * self.measure)

		self.res_form = ((S_area / S0 - T * self.kinematics.J) * S_area_test) * self.measure
