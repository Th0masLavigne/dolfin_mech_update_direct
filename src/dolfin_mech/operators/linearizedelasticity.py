# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""summary"""

import dolfin

from ..materials import material_factory
from .operator import Operator

################################################################################


class LinearizedElasticity(Operator):
	r"""Operator representing the internal virtual work for linearized elasticity.

	This class assembles the residual variational form for a body under the
	assumption of small deformations and small strains. It relates the Cauchy
	stress tensor :math:`\mathbf{\sigma}` to the symmetric part of the
	displacement gradient (the infinitesimal strain tensor :math:`\mathbf{\epsilon}`).

	The internal virtual work :math:`\delta \Pi_{int}` is defined as:

	.. math::
	    \delta \Pi_{int} = \int_{\Omega} \mathbf{\sigma} : \delta \mathbf{\epsilon} \, d\Omega

	where:
	    - :math:`\mathbf{\sigma}` is the Cauchy stress tensor provided by the material model.
	    - :math:`\delta \mathbf{\epsilon} = \text{sym}(\nabla \delta \mathbf{u})` is the
	      virtual infinitesimal strain tensor.

	Attributes:
	kinematics (Kinematics): Kinematic variables associated with the small strain assumption.
	material (Material): Linear elastic material instance created via the factory.
	measure (dolfin.Measure): Integration measure (typically ``dx``).
	res_form (UFL form): The resulting residual variational form.


	:param kinematics: Kinematics object (typically handling infinitesimal strains).
	:type kinematics: dmech.Kinematics
	:param u_test: The test function (virtual displacement).
	:type u_test: dolfin.TestFunction
	:param material_model: Name of the material model (e.g., "Hooke").
	:type material_model: str
	:param material_parameters: Dictionary of material properties (e.g., E, nu).
	:type material_parameters: dict
	:param measure: Dolfin measure for domain integration.
	:type measure: dolfin.Measure
	"""

	def __init__(self, kinematics, u_test, material_model, material_parameters, measure):
		"""Initializes the LinearizedElasticityOperator."""
		self.kinematics = kinematics
		self.material = material_factory(kinematics, material_model, material_parameters)
		self.measure = measure

		epsilon_test = dolfin.sym(dolfin.grad(u_test))
		self.res_form = dolfin.inner(self.material.sigma, epsilon_test) * self.measure
