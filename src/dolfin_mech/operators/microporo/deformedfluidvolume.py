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

"""Defines the DeformedFluidVolume class.

Computes the current volume of
an enclosed fluid region by integrating deformed boundary coordinates,
facilitating the coupling between internal fluid states and external solid
deformations.
"""

import dolfin

from ..operator import Operator

################################################################################


class DeformedFluidVolume(Operator):
	r"""Operator representing the volume of a fluid region enclosed by a deforming solid boundary.

	In large deformation mechanics, the volume of an enclosed cavity can be
	calculated using the surface integral of the spatial coordinates over the
	deformed boundary. This operator implements this relation, which is
	fundamental for problems involving pressurized cavities or fluid-structure
	coupling.

	The current volume :math:`v_f` is related to the deformed coordinates
	:math:`\mathbf{x} = \mathbf{X} + \mathbf{U}_{tot}` via the divergence theorem:

	.. math::
	    v_f = \frac{1}{dim} \int_{\Gamma} \mathbf{x} \cdot \mathbf{n} \, d\Gamma

	By pulling this back to the reference configuration :math:`\Gamma_0`, we use
	Nanson's formula (:math:`\mathbf{n} d\Gamma = J \mathbf{F}^{-T} \mathbf{N} d\Gamma_0`):

	.. math::
	    v_f = \frac{1}{dim} \int_{\Gamma_0} (\mathbf{X} + \mathbf{U}_{tot}) \cdot (J \mathbf{F}^{-T} \mathbf{N}) \, d\Gamma_0

	Attributes:
	    kinematics (Kinematics): Kinematic variables providing the deformation
	        gradient :math:`\mathbf{F}` and Jacobian :math:`J`.
	    U_tot (dolfin.Function): The total displacement field.
	    X (dolfin.SpatialCoordinate): The reference coordinates.
	    N (dolfin.Constant): The unit normal vector in the reference configuration.
	    measure (dolfin.Measure): Integration measure (typically ``dx``).
	    res_form (UFL expression): The residual variational form.
	"""

	def __init__(self, vf, vf_test, kinematics, N, dS, U_tot, X, measure):
		"""Initializes the DeformedFluidVolumeOperator.

		:param vf: Variable representing the fluid volume (scalar).
		:type vf: dolfin.Function or dolfin.Coefficient
		:param vf_test: Test function associated with the fluid volume variable.
		:type vf_test: dolfin.TestFunction
		:param kinematics: Kinematics object for finite strain variables.
		:type kinematics: dmech.Kinematics
		:param N: Reference unit normal vector.
		:type N: dolfin.Constant
		:param dS: Surface measure for the boundary of the fluid volume.
		:type dS: dolfin.Measure
		:param U_tot: Total displacement field.
		:type U_tot: dolfin.Function
		:param X: Spatial coordinates in the reference configuration.
		:type X: dolfin.SpatialCoordinate
		:param measure: Integration measure for the domain.
		:type measure: dolfin.Measure
		"""
		self.kinematics = kinematics
		self.U_tot = U_tot
		self.X = X
		self.N = N
		self.dS = dS
		self.measure = measure

		PN = self.kinematics.J * dolfin.dot(dolfin.inv(self.kinematics.F.T), self.N)

		self.res_form = (
			(vf + dolfin.assemble(dolfin.inner(self.U_tot + self.X, PN) * self.dS(0)) / 2) * vf_test
		) * self.measure  # MG20230203: This is not correct. Need to check.
