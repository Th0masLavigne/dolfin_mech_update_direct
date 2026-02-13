# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Mooney-Rivlin hyperelastic material implementation.

This module provides the Mooney-Rivlin model with support for both coupled
and isochoric-volumetric decoupled formulations in 2D and 3D.
"""

import dolfin

from .Elastic import ElasticMaterial

################################################################################


class MooneyRivlin(ElasticMaterial):
	r"""Class representing a Mooney-Rivlin hyperelastic material model.

	The Mooney-Rivlin model is a phenomenological model for hyperelastic material
	behavior where the strain energy density is a linear function of the first
	and second invariants of the Right Cauchy-Green deformation tensor.

	The strain energy density :math:`\Psi` depends on whether the formulation is
	coupled or decoupled (isochoric/volumetric split).

	**Standard Coupled Formulation (decoup=False):**

	.. math::
	    \Psi_{3D} = C_2 (I_C^{II} - 3 - 4\ln(J))

	.. math::
	    \Psi_{2D} = C_2 (I_C^{II} + I_C^{I} - 3 - 4\ln(J))

	**Decoupled Formulation (decoup=True):**

	Uses the isochoric invariants :math:`\bar{I}_C^{I}` and :math:`\bar{I}_C^{II}`:

	.. math::
	    \Psi_{3D} = C_2 (\bar{I}_C^{II} - 3)

	Attributes:
	    kinematics (Kinematics): Object containing kinematic variables and invariants.
	    C2 (float): Material constant associated with the second invariant.
	    Psi (UFL expression): Strain energy density.
	    Sigma (UFL expression): Second Piola-Kirchhoff stress tensor :math:`\mathbf{\Sigma}`.
	    P (UFL expression): First Piola-Kirchhoff stress tensor :math:`\mathbf{P}`.
	    sigma (UFL expression): Cauchy stress tensor :math:`\mathbf{\sigma}`.
	"""

	def __init__(self, kinematics, parameters, decoup=False):
		"""Initializes the MooneyRivlinElasticMaterial.

		:param kinematics: Kinematics object containing deformation tensors and invariants.
		:type kinematics: dmech.Kinematics
		:param parameters: Dictionary containing 'C2' (or parameters to derive it).
		:type parameters: dict
		:param decoup: If True, uses the isochoric-volumetric decoupled formulation.
		:type decoup: bool, optional
		"""
		self.kinematics = kinematics

		self.C2 = self.get_C2_from_parameters(parameters)

		if decoup:
			if self.kinematics.dim == 2:
				self.Psi = self.C2 * (
					self.kinematics.IIC_bar + self.kinematics.J ** (-2 / 3) * self.kinematics.IC_bar - 3
				)  # MG20200206: Plane strain
				self.Sigma = (
					2
					* self.C2
					* self.kinematics.J ** (-4 / 3)
					* (
						(self.kinematics.IC + 1) * self.kinematics.I
						- self.kinematics.C
						- 2 * (self.kinematics.IIC + self.kinematics.IC) / 3 * self.kinematics.C_inv
					)
				)  # MG20200206: Cannot differentiate Psi wrt to C because J is not defined as a function of C
				self.Sigma_ZZ = (
					2
					* self.C2
					* self.kinematics.J ** (-4 / 3)
					* (self.kinematics.IC - 2 * (self.kinematics.IIC + self.kinematics.IC) / 3)
				)
			elif self.kinematics.dim == 3:
				self.Psi = self.C2 * (self.kinematics.IIC_bar - 3)
				self.Sigma = (
					2
					* self.C2
					* self.kinematics.J ** (-4 / 3)
					* (
						self.kinematics.IC * self.kinematics.I
						- self.kinematics.C
						- 2 * self.kinematics.IIC / 3 * self.kinematics.C_inv
					)
				)  # MG20200206: Cannot differentiate Psi wrt to C because J is not defined as a function of C
		else:
			if self.kinematics.dim == 2:
				self.Psi = self.C2 * (
					self.kinematics.IIC + self.kinematics.IC - 3 - 4 * dolfin.ln(self.kinematics.J)
				)  # MG20200206: Plane strain
				self.Sigma = (
					2
					* self.C2
					* ((self.kinematics.IC + 1) * self.kinematics.I - self.kinematics.C - 2 * self.kinematics.C_inv)
				)  # MG20200206: Cannot differentiate Psi wrt to C because J is not defined as a function of C
				self.Sigma_ZZ = 2 * self.C2 * (self.kinematics.IC - 2)
			elif self.kinematics.dim == 3:
				self.Psi = self.C2 * (self.kinematics.IIC - 3 - 4 * dolfin.ln(self.kinematics.J))
				self.Sigma = (
					2
					* self.C2
					* (self.kinematics.IC * self.kinematics.I - self.kinematics.C - 2 * self.kinematics.C_inv)
				)  # MG20200206: Cannot differentiate Psi wrt to C because J is not defined as a function of C

		# self.P = dolfin.diff(self.Psi, self.kinematics.F) # MG20220426: Cannot do that for micromechanics problems
		self.P = self.kinematics.F * self.Sigma

		self.sigma = self.P * self.kinematics.F.T / self.kinematics.J

	# def get_free_energy(self,
	#         U=None,
	#         C=None):

	#     C     = self.get_C_from_U_or_C(U,C)
	#     IC    = dolfin.tr(C)
	#     IIC   = (dolfin.tr(C)*dolfin.tr(C) - dolfin.tr(C*C))/2
	#     JF    = dolfin.sqrt(dolfin.det(C)) # MG20200207: Watch out! This is well defined for inverted elements!
	#     # C_inv = dolfin.inv(C)

	#     assert (C.ufl_shape[0] == C.ufl_shape[1])
	#     dim = C.ufl_shape[0]
	#     # I = dolfin.Identity(dim)

	#     if   (dim == 2):
	#         Psi   =   self.C2 * (IIC + IC - 3 - 4*dolfin.ln(JF)) # MG20200206: plane strain
	#         # Sigma = 2*self.C2 * (IC * I - C + I - 2*C_inv)
	#     elif (dim == 3):
	#         Psi   =   self.C2 * (IIC - 3 - 4*dolfin.ln(JF))
	#         # Sigma = 2*self.C2 * (IC * I - C - 2*C_inv)
	#     Sigma = 2*dolfin.diff(Psi, C)

	#     return Psi, Sigma

	# def get_PK2_stress(self,
	#         U=None,
	#         C=None):

	#     Psi   = self.get_free_energy(U,C)
	#     C     = self.get_C_from_U_or_C(U,C)
	#     Sigma = 2*dolfin.diff(Psi, C)

	#     return Psi, Sigma

	# def get_PK1_stress(self,
	#         U=None,
	#         F=None):

	#     Psi = self.get_free_energy(U,F)
	#     F   = self.get_F_from_U_or_F(U,F)
	#     P   = dolfin.diff(Psi, F)

	#     return P
