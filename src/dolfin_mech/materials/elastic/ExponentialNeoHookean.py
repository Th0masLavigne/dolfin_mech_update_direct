# coding=utf8

################################################################################
###                                                                          ###
### Created by Mahdi Manoochehrtayebi, 2020-2024                             ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################


"""Exponential Neo-Hookean Material Module.

This module defines the exponential-based Neo-Hookean hyperelastic model,
characterized by its non-linear stiffening under finite strains.
"""

import dolfin

from .Elastic import ElasticMaterial

################################################################################


class ExponentialNeoHookean(ElasticMaterial):
	r"""Class implementing an exponential-based Neo-Hookean elastic material model.

	This constitutive model is designed for finite strain elasticity and
	features an exponential strain energy density function. It is often used
	to capture the non-linear stiffening behavior of soft materials.

	The strain energy density :math:`\Psi` is defined as:

	.. math::

	    \Psi = \frac{\beta_1}{2\beta_2\alpha} \left( e^{\beta_2 (I_C - d - 2\ln J)^\alpha} - 1 \right) + \beta_3
	    (I_C - d - 2\ln J) + \beta_4 (J^2 - 1 - 2\ln J)

	where :math:`d` is the spatial dimension.

	Attributes:
	    kinematics (Kinematics): Kinematic quantities (F, J, C, etc.).
	    beta1, beta2, beta3, beta4, alpha (dolfin.Constant): Material parameters.
	    Psi (ufl.Form): Strain energy density function.
	    Sigma (ufl.Form): Second Piola-Kirchhoff stress tensor :math:`\mathbf{S}`.
	    P (ufl.Form): First Piola-Kirchhoff stress tensor :math:`\mathbf{P} = \mathbf{F}\mathbf{S}`.
	    sigma (ufl.Form): Cauchy stress tensor :math:`\boldsymbol{\sigma} = J^{-1} \mathbf{P}\mathbf{F}^T`.
	"""

	def __init__(self, kinematics, parameters):
		"""Initialize the Exponential Neo-Hookean material model.

		Args:
		    kinematics (Kinematics): Kinematic object providing deformation tensors
		        and invariants.
		    parameters (dict): Dictionary containing the material constants:
		        ``"beta1"``, ``"beta2"``, ``"beta3"``, ``"beta4"``, and ``"alpha"``.
		"""
		self.kinematics = kinematics

		self.beta1 = dolfin.Constant(parameters["beta1"])
		self.beta2 = dolfin.Constant(parameters["beta2"])
		self.beta3 = dolfin.Constant(parameters["beta3"])
		self.beta4 = dolfin.Constant(parameters["beta4"])
		self.alpha = dolfin.Constant(parameters["alpha"])

		if self.kinematics.dim == 2:
			self.Psi = (
				self.beta1
				/ self.beta2
				/ self.alpha
				/ 2
				* (
					dolfin.exp(self.beta2 * (self.kinematics.IC - 2 - 2 * dolfin.ln(self.kinematics.J)) ** self.alpha)
					- 1
				)
				+ self.beta3 * (self.kinematics.IC - 2 - 2 * dolfin.ln(self.kinematics.J))
				+ self.beta4 * (self.kinematics.J**2 - 1 - 2 * dolfin.ln(self.kinematics.J))
			)
			self.Sigma = (
				self.beta1
				* (self.kinematics.I - self.kinematics.C_inv)
				* (self.kinematics.IC - 2 - 2 * dolfin.ln(self.kinematics.J)) ** (self.alpha - 1)
				* dolfin.exp(self.beta2 * (self.kinematics.IC - 2 - 2 * dolfin.ln(self.kinematics.J)) ** self.alpha)
				+ 2 * self.beta3 * (self.kinematics.I - self.kinematics.C_inv)
				+ 2 * self.beta4 * (self.kinematics.J**2 - 1) * self.kinematics.C_inv
			)

		elif self.kinematics.dim == 3:
			self.Psi = (
				self.beta1
				/ self.beta2
				/ self.alpha
				/ 2
				* (
					dolfin.exp(self.beta2 * (self.kinematics.IC - 3 - 2 * dolfin.ln(self.kinematics.J)) ** self.alpha)
					- 1
				)
				+ self.beta3 * (self.kinematics.IC - 3 - 2 * dolfin.ln(self.kinematics.J))
				+ self.beta4 * (self.kinematics.J**2 - 1 - 2 * dolfin.ln(self.kinematics.J))
			)
			self.Sigma = (
				self.beta1
				* (self.kinematics.I - self.kinematics.C_inv)
				* (self.kinematics.IC - 3 - 2 * dolfin.ln(self.kinematics.J)) ** (self.alpha - 1)
				* dolfin.exp(self.beta2 * (self.kinematics.IC - 3 - 2 * dolfin.ln(self.kinematics.J)) ** self.alpha)
				+ 2 * self.beta3 * (self.kinematics.I - self.kinematics.C_inv)
				+ 2 * self.beta4 * (self.kinematics.J**2 - 1) * self.kinematics.C_inv
			)

		self.P = self.kinematics.F * self.Sigma
		self.sigma = self.P * self.kinematics.F.T / self.kinematics.J
