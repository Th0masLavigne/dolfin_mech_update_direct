# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Defines the InversePoroHyperelasticity class.

Specialized manager for
reconstructing the stress-free reference configuration of porous media from
deformed spatial data. It adapts the biphasic poro-hyperelastic framework to
an inverse kinematic setting, coupling spatial porosity measurements with
reference geometry estimation.
"""

import dolfin
import numpy

from .. import kinematics, operators
from .problem_hyperelasticity_poro import PoroHyperelasticity

################################################################################


class InversePoroHyperelasticity(PoroHyperelasticity):
	r"""Problem class for solving inverse poro-hyperelasticity problems.

	Unlike the forward problem which predicts deformation from loads, the
	**inverse problem** seeks to determine the reference (stress-free)
	configuration :math:`\Omega_0` given a known deformed configuration
	:math:`\Omega` and a set of loads (e.g., blood pressure, gravity).



	This is particularly critical in biomechanics (e.g., estimating the
	zero-pressure geometry of the heart or lungs from in-vivo imaging).

	**Porosity Handling:**
	Since the reference configuration is unknown, the reference porosity
	:math:`\phi_{s0}` is also typically unknown. This class supports two
	formulations:

	1.  **Known Spatial Porosity (`phis` known)**: The current solid volume fraction
	    is measured (e.g., from CT Hounsfield units), and the reference
	    porosity is solved for.
	2.  **Known Material Porosity (`Phis0` known)**: The Lagrangian solid volume
	    fraction is assumed known, and the spatial porosity is derived.
	"""

	def __init__(self, *args, **kwargs):
		"""Initializes the InversePoroHyperelasticityProblem.

		Accepts the same arguments as :class:`PoroHyperelasticityProblem`,
		but sets up the problem in the spatial frame.
		"""
		PoroHyperelasticity.__init__(self, *args, **kwargs)

	def set_known_and_unknown_porosity(self, porosity_known):
		"""Configures which porosity field is treated as known data.

		:param porosity_known: A string indicating the known variable:

		    - ``"phis"``: The spatial solid volume fraction (current configuration)
		      is known. The reference porosity ``phis0`` becomes an unknown.
		    - ``"Phis0"``: The Lagrangian solid volume fraction (reference mass
		      density) is known. The spatial porosity ``phis`` becomes the unknown.
		"""
		self.porosity_known = porosity_known
		if self.porosity_known == "phis":
			self.porosity_unknown = "phis0"
		elif self.porosity_known == "Phis0":
			self.porosity_unknown = "phis"

	def get_deformed_center_of_mass(self):
		"""Calculates the center of mass of the body in the current (deformed) configuration.

		This is often required for setting up gravity loading or for pinning
		rigid body motions in the inverse problem.

		:return: A numpy array of the center of mass coordinates :math:`[x_c, y_c, z_c]`.
		"""
		M = dolfin.assemble(getattr(self, self.porosity_known) * self.dV)
		center_of_mass = numpy.empty(self.dim)
		for k_dim in range(self.dim):
			center_of_mass[k_dim] = dolfin.assemble(getattr(self, self.porosity_known) * self.x[k_dim] * self.dV) / M
		return center_of_mass

	def set_kinematics(self):
		r"""Initializes the inverse kinematic framework.

		Defines the deformation gradient :math:`\mathbf{F} = (\nabla_{\mathbf{x}} \mathbf{X})^{-1}`
		where the mesh represents the spatial domain :math:`\mathbf{x}` and the
		unknown is the reference coordinate mapping :math:`\mathbf{X}(\mathbf{x})`.

		Registers standard inverse kinematic fields (F, J, C, E) as Fields of Interest.
		"""
		self.kinematics = kinematics.InverseKinematics(
			u=self.displacement_subsol.subfunc, u_old=self.displacement_subsol.func_old
		)

		self.add_foi(expr=self.kinematics.F, fs=self.mfoi_fs, name="F")
		self.add_foi(expr=self.kinematics.J, fs=self.sfoi_fs, name="J")
		self.add_foi(expr=self.kinematics.C, fs=self.mfoi_fs, name="C")
		self.add_foi(expr=self.kinematics.E, fs=self.mfoi_fs, name="E")

	def set_porosity_fields(self):
		r"""Derives the dependent porosity field based on the mass conservation constraint.

		The relationship between reference solid volume fraction :math:`\phi_{s0}`
		and current solid volume fraction :math:`\phi_s` is governed by the Jacobian:

		.. math::
		    \phi_s = J^{-1} \Phi_{s0}

		where :math:`\Phi_{s0}` is the Lagrangian porosity.


		"""
		if self.porosity_known == "phis":
			self.phis0 = self.porosity_subsol.subfunc
			self.Phis0 = self.phis0 * self.kinematics.J
		elif self.porosity_known == "Phis0":
			self.phis = self.porosity_subsol.subfunc
			self.phis0 = self.Phis0 / self.kinematics.J

	def add_local_porosity_fois(self):
		r"""Registers Fields of Interest (FOI) for visualization of porosity distributions.

		Adds the following fields:
		    - **phis**: Current solid volume fraction.
		    - **phif**: Current fluid volume fraction (:math:`1 - \phi_s`).
		    - **phis0**: Reference solid volume fraction.
		    - **phif0**: Reference fluid volume fraction.
		"""
		self.add_foi(expr=self.phis, fs=self.porosity_subsol.fs.collapse(), name="phis")
		self.add_foi(expr=1.0 - self.phis, fs=self.porosity_subsol.fs.collapse(), name="phif")

		if self.porosity_known == "Phis0":
			self.add_foi(expr=self.phis0, fs=self.porosity_subsol.fs.collapse(), name="phis0")
		self.add_foi(expr=1 / self.kinematics.J - self.phis0, fs=self.porosity_subsol.fs.collapse(), name="phif0")

		self.add_foi(
			expr=self.kinematics.J * self.phis0,  # MG20250908: Todo: check!
			fs=self.porosity_subsol.fs.collapse(),
			name="Phis0",
		)
		self.add_foi(
			expr=1.0 - self.kinematics.J * self.phis0,  # MG20250908: Todo: check!
			fs=self.porosity_subsol.fs.collapse(),
			name="Phif0",
		)

	def add_Wskel_operator(self, material_parameters, material_scaling, subdomain_id=None):
		"""Adds the inverse strain energy operator for the solid skeleton.

		Calculates the virtual work contributions from the hyperelastic skeleton
		in the inverse configuration.
		"""
		operator = operators.poro.InverseWskel(
			kinematics=self.kinematics,
			u_test=self.displacement_subsol.dsubtest,
			phis0=self.phis0,
			material_parameters=material_parameters,
			material_scaling=material_scaling,
			measure=self.get_subdomain_measure(subdomain_id),
		)
		return self.add_operator(operator)

	def add_Wbulk_operator(self, material_parameters, material_scaling, subdomain_id=None):
		"""Adds the operator for the bulk compressibility of the mixture.

		This penalizes volume changes that deviate from the constitutive
		behavior of the constituents.
		"""
		operator = operators.poro.InverseWbulk(
			kinematics=self.kinematics,
			u_test=self.displacement_subsol.dsubtest,
			phis=self.phis,
			phis0=self.phis0,
			unknown_porosity_test=self.porosity_subsol.dsubtest,
			material_parameters=material_parameters,
			material_scaling=material_scaling,
			measure=self.get_subdomain_measure(subdomain_id),
		)
		return self.add_operator(operator)

	def add_Wpore_operator(self, material_parameters, material_scaling, subdomain_id=None):
		"""Adds the operator handling the pore pressure contribution.

		This links the porosity changes to the fluid pressure in the
		inverse formulation.
		"""
		operator = operators.poro.InverseWpore(
			kinematics=self.kinematics,
			phis=self.phis,
			phis0=self.phis0,
			unknown_porosity_test=self.porosity_subsol.dsubtest,
			material_parameters=material_parameters,
			material_scaling=material_scaling,
			measure=self.get_subdomain_measure(subdomain_id),
		)
		return self.add_operator(operator)

	def add_pressure_balancing_gravity0_loading_operator(self, k_step=None, **kwargs):
		"""Adds a specialized operator to balance internal pressure with gravity in the reference state.

		In many inverse problems (like finding the reference shape of a lung),
		the reference state is defined by an equilibrium between an internal
		pressure and gravity. This operator simultaneously solves for the
		geometry and the Lagrange multipliers required to satisfy this
		specific equilibrium.


		"""
		operator = operators.loading.PressureBalancingGravity0(
			x=self.x,
			x0=self.deformed_center_of_mass_subsol.subfunc,
			x0_test=self.deformed_center_of_mass_subsol.dsubtest,
			n=self.mesh_normals,
			u_test=self.displacement_subsol.dsubtest,
			lmbda=self.lmbda_subsol.subfunc,
			lmbda_test=self.lmbda_subsol.dsubtest,
			p=self.pressure_balancing_gravity_subsol.subfunc,
			p_test=self.pressure_balancing_gravity_subsol.dsubtest,
			gamma=self.gamma_subsol.subfunc,
			gamma_test=self.gamma_subsol.dsubtest,
			mu=self.mu_subsol.subfunc,
			mu_test=self.mu_subsol.dsubtest,
			**kwargs,
		)
		return self.add_operator(operator=operator, k_step=k_step)

	def add_global_porosity_qois(self):
		"""Adds Quantities of Interest (QOI) for global porosity metrics.

		Integrates the porosity fields over the domain to obtain total solid
		and fluid volumes in both reference and spatial configurations.
		"""
		self.add_qoi(name="phis", expr=self.phis * self.dV)

		self.add_qoi(name="phif", expr=(1.0 - self.phis) * self.dV)

		self.add_qoi(name="phis0", expr=self.phis0 * self.dV)

		self.add_qoi(name="phif0", expr=(1 / self.kinematics.J - self.phis0) * self.dV)

		self.add_qoi(name="Phis0", expr=(self.kinematics.J * self.phis0) * self.dV)

		self.add_qoi(name="Phif0", expr=(1 - self.kinematics.J * self.phis0) * self.dV)
