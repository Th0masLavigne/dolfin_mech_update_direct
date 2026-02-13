# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Linearized kinematics implementation for small-deformation mechanics.

This module provides the LinearizedKinematics class, which handles the
computation of infinitesimal strain tensors and their decompositions.
"""

import dolfin

################################################################################


class LinearizedKinematics:
	r"""Class to compute and store kinematic quantities for linearized solid mechanics.

	This class computes the infinitesimal strain tensor :math:`\boldsymbol{\varepsilon}`
	based on the displacement field :math:`\mathbf{u}` under the assumption of
	small displacement gradients. It provides access to the full, spherical,
	and deviatoric parts of the strain.

	Args:
		    u (dolfin.Function): Current displacement field.
		    u_old (dolfin.Function, optional): Displacement field from the previous
		        time step. Defaults to None.

	Attributes:
	    u (dolfin.Function): The displacement field.
	    dim (int): Spatial dimension.
	    I (dolfin.Identity): Identity tensor of dimension ``dim``.
	    epsilon (ufl.Form): Symmetric infinitesimal strain tensor
	        :math:`\boldsymbol{\varepsilon} = \frac{1}{2}(\nabla \mathbf{u} + \nabla \mathbf{u}^T)`.
	    epsilon_sph (ufl.Form): Spherical part of the strain tensor
	        :math:`\boldsymbol{\varepsilon}_{sph} = \frac{1}{d} \text{tr}(\boldsymbol{\varepsilon}) \mathbf{I}`.
	    epsilon_dev (ufl.Form): Deviatoric part of the strain tensor
	        :math:`\boldsymbol{\varepsilon}_{dev} = \boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_{sph}`.
	"""

	def __init__(self, u, u_old=None):
		"""Initialize the LinearizedKinematics object and compute strain tensors.

		Args:
		    u (dolfin.Function): Current displacement field.
		    u_old (dolfin.Function, optional): Displacement field from the previous
		        time step. Defaults to None.

		Notes:
		    If ``u_old`` is provided, the class also computes the strain at the
		    mid-point configuration, useful for certain time-integration schemes.
		"""
		self.u = u

		self.dim = self.u.ufl_shape[0]
		self.I = dolfin.Identity(self.dim)

		self.epsilon = dolfin.sym(dolfin.grad(self.u))
		self.epsilon = dolfin.variable(self.epsilon)

		self.epsilon_sph = dolfin.tr(self.epsilon) / self.dim * self.I
		self.epsilon_dev = self.epsilon - self.epsilon_sph

		if u_old is not None:
			self.u_old = u_old

			self.epsilon_old = dolfin.sym(dolfin.grad(self.u_old))

			self.epsilon_sph_old = dolfin.tr(self.epsilon_old) / self.dim * self.I
			self.epsilon_dev_old = self.epsilon_old - self.epsilon_sph_old

			self.epsilon_mid = (self.epsilon_old + self.epsilon) / 2
