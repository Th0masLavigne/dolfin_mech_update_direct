# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Kirchhoff elastic materials implementation.

This module provides classes for standard St. Venant-Kirchhoff materials,
including split formulations for bulk and deviatoric components.
"""

import dolfin

from .Elastic import ElasticMaterial

################################################################################


class Kirchhoff(ElasticMaterial):
	r"""Class representing a standard St. Venant-Kirchhoff elastic material.

	This model extends linear elasticity to large deformations by using the Green-Lagrange
	strain tensor :math:`\mathbf{E}` in the strain energy density function.

	The strain energy density is defined as:

	.. math::
	    \Psi = \frac{\lambda}{2} tr(\mathbf{E})^2 + \mu tr(\mathbf{E}^2)

	Attributes:
	    kinematics (Kinematics): Object containing kinematic variables (F, J, E, etc.).
	    lmbda (float): First Lamé parameter.
	    mu (float): Second Lamé parameter (shear modulus).
	    Psi (UFL expression): Strain energy density.
	    Sigma (UFL expression): Second Piola-Kirchhoff stress tensor.
	    P (UFL expression): First Piola-Kirchhoff stress tensor.
	    sigma (UFL expression): Cauchy stress tensor.
	"""

	def __init__(self, kinematics, parameters):
		"""Initializes the KirchhoffElasticMaterial.

		Args:
		    kinematics (dolfin_mech.Kinematics): Kinematics object containing
		        deformation tensors.
		    parameters (dict): Dictionary or object containing 'lambda'
		        and 'mu' (or equivalent).
		"""
		self.kinematics = kinematics

		self.lmbda = self.get_lambda_from_parameters(parameters)
		self.mu = self.get_mu_from_parameters(parameters)

		self.kinematics.E = dolfin.variable(self.kinematics.E)

		self.Psi = (self.lmbda / 2) * dolfin.tr(self.kinematics.E) ** 2 + self.mu * dolfin.inner(
			self.kinematics.E, self.kinematics.E
		)

		self.Sigma = dolfin.diff(self.Psi, self.kinematics.E)
		# self.Sigma = self.lmbda * dolfin.tr(self.kinematics.E) * self.kinematics.I + 2 * self.mu * self.kinematics.E

		if self.kinematics.dim == 2:
			self.Sigma_ZZ = self.lmbda * dolfin.tr(self.kinematics.E)

		self.P = self.kinematics.F * self.Sigma

		self.sigma = self.P * self.kinematics.F.T / self.kinematics.J

	# def get_free_energy(self,
	#         U=None,
	#         C=None,
	#         E=None):

	#     E = self.get_E_from_U_C_or_E(U, C, E)

	#     Psi = (self.lmbda/2) * dolfin.tr(E)**2 + self.mu * dolfin.inner(E, E)
	#     Sigma = dolfin.diff(Psi, E)

	#     # assert (E.ufl_shape[0] == E.ufl_shape[1])
	#     # dim = E.ufl_shape[0]
	#     # I = dolfin.Identity(dim)
	#     # Sigma = self.lmbda * dolfin.tr(E) * I + 2 * self.mu * E

	#     return Psi, Sigma


################################################################################


class KirchhoffBulk(ElasticMaterial):
	r"""Class representing the volumetric (bulk) component of a Kirchhoff elastic material.

	This model focuses on the spherical part of the strain tensor, typically used in
	penalty methods or split formulations.

	The strain energy density is defined as:

	.. math::
	    \Psi = \frac{d \cdot K}{2} tr(\mathbf{E}_{sph})^2

	Attributes:
	    K (float): Bulk modulus.
	    Psi (UFL expression): Volumetric strain energy density.
	    Sigma (UFL expression): Spherical part of the Second Piola-Kirchhoff stress.
	"""

	def __init__(self, kinematics, parameters):
		"""Initializes the KirchhoffBulkElasticMaterial.

		Args:
		kinematics: Kinematics object.
		parameters: Material parameters to derive Bulk modulus $K$.
		"""
		self.kinematics = kinematics

		# self.K = self.get_K_from_parameters(parameters)
		self.lmbda, self.mu = self.get_lambda_and_mu_from_parameters(parameters)
		self.K = (self.kinematics.dim * self.lmbda + 2 * self.mu) / self.kinematics.dim

		self.Psi = (self.kinematics.dim * self.K / 2) * dolfin.tr(self.kinematics.E_sph) ** 2
		self.Sigma = self.kinematics.dim * self.K * self.kinematics.E_sph

		if self.kinematics.dim == 2:
			self.Sigma_ZZ = self.K * dolfin.tr(self.kinematics.E)

		# self.P = dolfin.diff(self.Psi, self.kinematics.F) # MG20220426: Cannot do that for micromechanics problems
		self.P = self.kinematics.F * self.Sigma

		self.sigma = self.P * self.kinematics.F.T / self.kinematics.J

	# def get_free_energy(self,
	#         U=None,
	#         C=None,
	#         E=None,
	#         E_sph=None):

	#     E_sph = self.get_E_sph_from_U_C_E_or_E_sph(
	#         U, C, E, E_sph)
	#     assert (E_sph.ufl_shape[0] == E_sph.ufl_shape[1])
	#     dim = E_sph.ufl_shape[0]

	#     Psi   = (dim*self.K/2) * dolfin.tr(E_sph)**2
	#     Sigma =  dim*self.K    *           E_sph

	#     return Psi, Sigma

	# def get_PK2_stress(self,
	#         U=None,
	#         C=None,
	#         E=None,
	#         E_sph=None):

	#     E_sph = self.get_E_sph_from_U_C_E_or_E_sph(
	#         U, C, E, E_sph)
	#     assert (E_sph.ufl_shape[0] == E_sph.ufl_shape[1])
	#     dim = E_sph.ufl_shape[0]

	#     Sigma = dim * self.K * E_sph

	#     return Sigma


################################################################################


class KirchhoffDev(ElasticMaterial):
	r"""Class representing the deviatoric (shear) component of a Kirchhoff elastic material.

	Used to model the shape-changing part of the deformation, independent of volume change.

	The strain energy density is defined as:

	.. math::
	    \Psi = G \cdot (\mathbf{E}_{dev} : \mathbf{E}_{dev})

	Attributes:
	    G (float): Shear modulus.
	    Psi (UFL expression): Deviatoric strain energy density.
	    Sigma (UFL expression): Deviatoric part of the Second Piola-Kirchhoff stress.
	"""

	def __init__(self, kinematics, parameters):
		"""Initializes the KirchhoffDevElasticMaterial.

		Args:
		kinematics: Kinematics object.
		parameters: Material parameters to derive Shear modulus $G$.
		"""
		self.kinematics = kinematics

		self.G = self.get_G_from_parameters(parameters)

		self.Psi = self.G * dolfin.inner(self.kinematics.E_dev, self.kinematics.E_dev)
		self.Sigma = 2 * self.G * self.kinematics.E_dev

		if self.kinematics.dim == 2:
			self.Sigma_ZZ = -2 * self.G / 3 * dolfin.tr(self.kinematics.E)

		# self.P     = dolfin.diff(self.Psi, self.kinematics.F)
		self.P = self.kinematics.F * self.Sigma

		self.sigma = self.P * self.kinematics.F.T / self.kinematics.J

	# def get_free_energy(self,
	#         U=None,
	#         C=None,
	#         E=None,
	#         E_dev=None):

	#     E_dev = self.get_E_dev_from_U_C_E_or_E_dev(
	#         U, C, E, E_dev)

	#     Psi   =   self.G * dolfin.inner(E_dev, E_dev)
	#     Sigma = 2*self.G *              E_dev

	#     return Psi, Sigma

	# def get_PK2_stress(self,
	#         U=None,
	#         C=None,
	#         E=None,
	#         E_dev=None):

	#     E_dev = self.get_E_dev_from_U_C_E_or_E_dev(
	#         U, C, E, E_dev)

	#     Sigma = 2*self.G * E_dev

	#     return Sigma
