# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Defines the InverseHyperelasticity class.

Specialized problem manager for
reconstructing the stress-free reference configuration of a body from its
known deformed geometry, frequently used for analyzing medical imaging data.
"""

from .. import kinematics, operators
from .problem_hyperelasticity import Hyperelasticity

################################################################################


class InverseHyperelasticity(Hyperelasticity):
	r"""Problem class for solving inverse hyperelasticity problems.

	Inverse hyperelasticity is typically used to find the stress-free
	configuration (reference state) of a body when only the deformed geometry
	is known. This is a common requirement in medical imaging (e.g., estimating
	the unloaded shape of an organ from a CT scan).



	This class inherits from :class:`HyperelasticityProblem` but overrides the
	kinematic framework and specific operators to account for the fact that the
	computational mesh represents the deformed configuration.

	.. note::
	    Incompressibility constraints via mixed formulations are currently
	    not supported in this inverse implementation.
	"""

	def __init__(self, *args, **kwargs):
		"""Initializes the InverseHyperelasticityProblem.

		Accepts the same arguments as :class:`HyperelasticityProblem`.

		:raises AssertionError: If ``w_incompressibility`` is set to ``True``.
		"""
		if "w_incompressibility" in kwargs:
			assert bool(kwargs["w_incompressibility"]) == 0, (
				"Incompressibility not implemented for inverse problem. Aborting."
			)

		Hyperelasticity.__init__(self, *args, **kwargs)

	def set_kinematics(self):
		r"""Initializes the inverse kinematic framework.

		Instead of the standard deformation gradient :math:`\mathbf{F}`, this
		method sets up :class:`InverseKinematics` which treats the current
		mesh coordinates as the spatial frame.

		It registers the following inverse fields as Fields of Interest (FOI):
		    * **F**: Inverse deformation gradient :math:`\mathbf{f} = \partial \mathbf{X} / \partial \mathbf{x}`.
		    * **J**: Determinant of the inverse deformation gradient.
		    * **C**: Right Cauchy-Green deformation tensor in the inverse context.
		    * **E**: Green-Lagrange strain tensor derived from inverse mapping.
		"""
		self.kinematics = kinematics.InverseKinematics(
			u=self.displacement_subsol.subfunc, u_old=self.displacement_subsol.func_old
		)

		self.add_foi(expr=self.kinematics.F, fs=self.mfoi_fs, name="F")
		self.add_foi(expr=self.kinematics.J, fs=self.sfoi_fs, name="J")
		self.add_foi(expr=self.kinematics.C, fs=self.mfoi_fs, name="C")
		self.add_foi(expr=self.kinematics.E, fs=self.mfoi_fs, name="E")

	def add_elasticity_operator(self, material_model, material_parameters, subdomain_id=None):
		"""Adds an elasticity operator tailored for the inverse formulation.

		Even though the problem is hyperelastic, the inverse formulation often
		utilizes a linearized operator structure relative to the spatial
		coordinates of the deformed mesh.

		:param material_model: Name of the constitutive model.
		:type material_model: str
		:param material_parameters: Parameters for the material model.
		:type material_parameters: dict
		:param subdomain_id: Optional ID to restrict the operator to a subdomain.
		:type subdomain_id: int, optional
		:return: The added elasticity operator.
		"""
		operator = operators.LinearizedElasticity(
			kinematics=self.kinematics,
			u_test=self.displacement_subsol.dsubtest,
			material_model=material_model,
			material_parameters=material_parameters,
			measure=self.get_subdomain_measure(subdomain_id),
		)
		return self.add_operator(operator)
