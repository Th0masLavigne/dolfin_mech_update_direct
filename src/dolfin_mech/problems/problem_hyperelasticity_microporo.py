# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Mahdi Manoochehrtayebi, 2020-2024                                    ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Defines the MicroPoroHyperelasticity class.

Multiscale solver for
poro-hyperelastic microstructures. It implements a displacement decomposition
framework that couples global macroscopic stretching with local periodic
fluctuations to model the nonlinear response of pressurized porous skeletons.
"""

import dolfin
import numpy

from .. import core, kinematics, operators
from .problem import Problem
from .problem_hyperelasticity import Hyperelasticity

################################################################################


class MicroPoroHyperelasticity(Hyperelasticity):
	r"""Problem class for multi-scale micro-poro-hyperelasticity.

	This class models the mechanical response of a porous microstructure (the
	micro-scale) subjected to macroscopic deformations. It uses a
	**displacement decomposition** approach:

	.. math::
	    \mathbf{U}_{tot}(\mathbf{X}) = \mathbf{U}_{bar}(\mathbf{X}) + \mathbf{U}_{tilde}(\mathbf{X})

	where:
	    - :math:`\mathbf{U}_{bar}` is the macroscopic displacement (linear mapping).
	    - :math:`\mathbf{U}_{tilde}` is the periodic or local fluctuation (perturbation).



	Attributes:
	    V0 (float): Initial total volume of the unit cell (bounding box volume).
	    Vs0 (float): Initial solid volume (volume of the mesh).
	    Vf0 (float): Initial fluid/pore volume.
	    kinematics (Kinematics): Finite strain kinematics based on total displacement.
	"""

	def __init__(
		self,
		w_solid_incompressibility=False,
		mesh=None,
		mesh_bbox=None,
		vertices=None,
		domains_mf=None,
		boundaries_mf=None,
		points_mf=None,
		displacement_perturbation_degree=None,
		solid_pressure_degree=None,
		quadrature_degree=None,
		foi_degree=0,
		solid_behavior=None,
		bcs="kubc",
	):  # "kubc" or "pbc"
		r"""Initializes the MicroPoroHyperelasticityProblem.

		:param w_solid_incompressibility: If True, uses a mixed u-p formulation for the solid phase.
		:param mesh_bbox: Bounding box [xmin, xmax, ymin, ymax, ...] defining the unit cell.
		:param vertices: Vertices for periodic point mapping.
		:param bcs: Boundary condition type: "kubc" (Kinematic Uniform) or "pbc" (Periodic).
		"""
		Problem.__init__(self)

		self.w_solid_incompressibility = w_solid_incompressibility
		self.vertices = vertices

		self.set_mesh(mesh=mesh, define_spatial_coordinates=1, define_facet_normals=1, compute_bbox=(mesh_bbox is None))
		self.X_0 = [0.0] * self.dim
		for k_dim in range(self.dim):
			self.X_0[k_dim] = dolfin.assemble(self.X[k_dim] * self.dV) / self.mesh_V0
		self.X_0 = dolfin.Constant(self.X_0)
		if mesh_bbox is not None:
			self.mesh_bbox = mesh_bbox
		d = [0] * self.dim
		for k_dim in range(self.dim):
			d[k_dim] = self.mesh_bbox[2 * k_dim + 1] - self.mesh_bbox[2 * k_dim + 0]

		self.V0 = numpy.prod(d)
		self.Vs0 = self.mesh_V0
		self.Vf0 = self.V0 - self.Vs0

		self.set_measures(domains=domains_mf, boundaries=boundaries_mf, points=points_mf)

		self.set_subsols(
			displacement_perturbation_degree=displacement_perturbation_degree,
			solid_pressure_degree=solid_pressure_degree,
		)
		self.set_solution_finite_element()
		if bcs == "pbc":
			periodic_sd = core.PeriodicSubDomain(self.dim, self.mesh_bbox, self.vertices)
			self.set_solution_function_space(constrained_domain=periodic_sd)
		else:
			self.set_solution_function_space()
		self.set_solution_functions()

		self.U_bar = dolfin.dot(self.macroscopic_stretch_subsol.subfunc, self.X - self.X_0)
		self.U_bar_old = dolfin.dot(self.macroscopic_stretch_subsol.func_old, self.X - self.X_0)
		self.U_bar_test = dolfin.dot(self.macroscopic_stretch_subsol.dsubtest, self.X - self.X_0)

		self.U_tot = self.U_bar + self.displacement_perturbation_subsol.subfunc
		self.U_tot_old = self.U_bar_old + self.displacement_perturbation_subsol.func_old
		self.U_tot_test = self.U_bar_test + self.displacement_perturbation_subsol.dsubtest

		self.set_quadrature_degree(quadrature_degree=quadrature_degree)

		self.set_foi_finite_elements_DG(degree=foi_degree)
		self.set_foi_function_spaces()

		self.add_foi(
			expr=self.U_bar, fs=self.displacement_perturbation_subsol.fs.collapse(), name="U_bar", update_type="project"
		)
		self.add_foi(
			expr=self.U_tot, fs=self.displacement_perturbation_subsol.fs.collapse(), name="U_tot", update_type="project"
		)

		self.set_kinematics()

		self.add_elasticity_operator(
			solid_behavior_model=solid_behavior["model"], solid_behavior_parameters=solid_behavior["parameters"]
		)
		if self.w_solid_incompressibility:
			self.add_hydrostatic_pressure_operator()
			self.add_incompressibility_operator()

		# self.add_macroscopic_stretch_symmetry_operator()
		self.add_macroscopic_stretch_symmetry_penalty_operator(pen_val=1e6)

		# self.add_deformed_total_volume_operator()
		# self.add_deformed_solid_volume_operator()
		# self.add_deformed_fluid_volume_operator()

		if bcs == "kubc":
			self.add_kubc()
		elif bcs == "pbc":
			pinpoint_sd = core.PinpointSubDomain(coords=mesh.coordinates()[-1], tol=1e-3)
			self.add_constraint(
				V=self.displacement_perturbation_subsol.fs,
				val=[0.0] * self.dim,
				sub_domain=pinpoint_sd,
				method="pointwise",
			)

	def add_macroscopic_stretch_subsol(self, degree=0, symmetry=None, init_val=None):
		"""Adds the macroscopic stretch (gradient) as a global Real (R) sub-solution."""
		self.macroscopic_stretch_subsol = self.add_tensor_subsol(
			name="U_bar", family="R", degree=degree, symmetry=symmetry, init_val=init_val
		)

	def add_displacement_perturbation_subsol(self, degree):
		r"""Adds the local displacement perturbation field :math:`\mathbf{U}_{tilde}`."""
		self.displacement_perturbation_degree = degree
		self.displacement_perturbation_subsol = self.add_vector_subsol(
			name="U_tilde", family="CG", degree=self.displacement_perturbation_degree
		)

	def add_deformed_total_volume_subsol(self):
		"""Adds a global scalar sub-solution to track the total volume of the unit cell.

		The total volume :math:`v` is defined using the **Real (R)** element family,
		meaning it is a single global scalar unknown across the entire domain.
		This variable is typically used in homogenization problems to couple
		the macroscopic stretch with the volume change of the unit cell.



		The initial value is set to :math:`V_0`, which represents the volume
		of the bounding box defining the unit cell in the reference configuration.

		:return: The added scalar sub-solution container.
		:rtype: SubSolution
		"""
		self.deformed_total_volume_subsol = self.add_scalar_subsol(name="v", family="R", degree=0, init_val=self.V0)

	def add_deformed_solid_volume_subsol(self):
		r"""Adds a global scalar sub-solution to track the volume of the solid phase.

		In the context of poro-hyperelasticity, this sub-solution tracks the
		current volume :math:`v_s` of the solid skeleton as it deforms. It is
		defined using a **Real (R)** element family, representing a single
		global degree of freedom for the unit cell.



		The value :math:`v_s` is typically computed by integrating the local
		Jacobian :math:`J` over the solid domain :math:`\Omega_s`:

		.. math::
		    v_s = \int_{\Omega_s} J \, d\Omega_0

		The initial value is set to :math:`V_{s0}` (the volume of the reference
		mesh), ensuring the problem starts from a state of zero deformation.

		:return: The added scalar sub-solution container.
		:rtype: SubSolution
		"""
		self.deformed_solid_volume_subsol = self.add_scalar_subsol(
			name="v_s", family="R", degree=0, init_val=self.mesh_V0
		)

	def add_deformed_fluid_volume_subsol(self):
		r"""Adds a global scalar sub-solution to track the volume of the fluid phase (pores).

		In a micro-poro-hyperelastic framework, the fluid volume :math:`v_f`
		represents the current volume of the void space within the unit cell.
		It is defined using the **Real (R)** element family, creating a single
		global scalar degree of freedom.



		In the context of displacement decomposition, the fluid volume is
		geometrically constrained by the difference between the total cell
		volume (governed by the macroscopic stretch) and the solid skeleton
		volume:

		.. math::
		    v_f = v_{tot} - v_s

		The initial value is set to :math:`V_{f0}`, the difference between
		the bounding box volume and the initial solid mesh volume.

		:return: The added scalar sub-solution container.
		:rtype: SubSolution
		"""
		self.deformed_fluid_volume_subsol = self.add_scalar_subsol(name="v_f", family="R", degree=0, init_val=self.Vf0)

	def add_surface_area_subsol(self, degree=0, init_val=None):
		r"""Adds a global scalar sub-solution to track the internal interfacial surface area.

		This sub-solution represents the current total area :math:`s` of the
		internal pores or interfaces (e.g., the alveolar surface area in lung
		parenchyma). It is defined using the **Real (R)** element family,
		creating a single global scalar degree of freedom for the entire
		Representative Elementary Volume (REV).



		Tracking the evolution of this surface area is critical for modeling
		surface tension effects or surfactant activity in pulmonary mechanics,
		where the energy contribution depends on the current area of the
		fluid-solid interface.

		:param degree: Polynomial degree for the Real element (defaults to 0).
		:type degree: int
		:param init_val: Initial value for the surface area. If None, it should
		    be updated later by an operator or solver.
		:type init_val: float, optional
		:return: The added scalar sub-solution container.
		:rtype: SubSolution
		"""
		self.surface_area_subsol = self.add_scalar_subsol(name="S_area", family="R", degree=degree, init_val=init_val)

	def set_subsols(self, displacement_perturbation_degree=None, solid_pressure_degree=None):
		r"""Configures the solution hierarchy for the micro-poro-hyperelastic problem.

		This method initializes the various sub-solutions required for the
		displacement decomposition and homogenization framework. It defines
		three main types of unknowns:

		1. **Macroscopic Stretch** (:math:`\mathbf{U}_{bar}`): A global tensor
		   representing the average deformation of the unit cell.
		2. **Displacement Perturbation** (:math:`\mathbf{U}_{tilde}`): A periodic
		   vector field representing local fluctuations within the microstructure.
		3. **Auxiliary Global Variables**: Such as the internal interfacial
		   surface area, used for surface tension/capillary energy.



		If ``w_solid_incompressibility`` is enabled, a pressure field is added
		for the solid phase, typically following the :math:`P_k / P_{k-1}`
		stable element pair logic.

		:param displacement_perturbation_degree: Polynomial degree for the
		    local fluctuation field (CG vector).
		:type displacement_perturbation_degree: int
		:param solid_pressure_degree: Polynomial degree for the solid pressure
		    (if incompressible). Defaults to ``displacement_perturbation_degree - 1``.
		:type solid_pressure_degree: int, optional
		"""
		self.add_macroscopic_stretch_subsol(
			symmetry=None
		)  # MG20220425: True does not work, cf. https://fenicsproject.discourse.group/t/writing-symmetric-tensor-function-fails/1136/2 & https://bitbucket.org/fenics-project/dolfin/issues/1065/cannot-store-symmetric-tensor-values

		self.add_displacement_perturbation_subsol(degree=displacement_perturbation_degree)

		if self.w_solid_incompressibility:
			if solid_pressure_degree is None:
				solid_pressure_degree = displacement_perturbation_degree - 1
			self.add_pressure_subsol(degree=solid_pressure_degree)

		# self.add_macroscopic_stress_lagrange_multiplier_subsol()

		# self.add_deformed_total_volume_subsol()
		# self.add_deformed_solid_volume_subsol()
		# self.add_deformed_fluid_volume_subsol()
		self.add_surface_area_subsol()

	def set_kinematics(self):
		"""Initializes finite strain kinematics for the total displacement field."""
		self.kinematics = kinematics.Kinematics(U=self.U_tot, U_old=self.U_tot_old)

		self.add_foi(expr=self.kinematics.F, fs=self.mfoi_fs, name="F_tot", update_type="project")
		self.add_foi(expr=self.kinematics.J, fs=self.sfoi_fs, name="J_tot", update_type="project")
		self.add_foi(expr=self.kinematics.C, fs=self.mfoi_fs, name="C_tot", update_type="project")
		self.add_foi(expr=self.kinematics.E, fs=self.mfoi_fs, name="E_tot", update_type="project")

	def add_elasticity_operator(self, solid_behavior_model, solid_behavior_parameters):
		r"""Adds a hyperelasticity operator for the solid phase in the micro-porous problem.

		This method defines the internal virtual work contribution of the solid
		skeleton. It utilizes the energy-based formulation (``formulation="ener"``)
		where the residual is the first variation of the strain energy density
		potential :math:`\Psi`.

		.. math::
		    \delta \Pi_{int} = \int_{\Omega_s} \frac{\partial \Psi}{\partial \mathbf{F}} : \delta \mathbf{F} \, d\Omega_0

		In this multiscale context, the operator specifically uses the
		**displacement perturbation** :math:`\mathbf{U}_{tilde}` as the primary
		unknown for the local fluctuation, while the kinematics account for the
		total displacement (macroscopic + perturbation).



		Additionally, this method automatically registers two stress fields as
		Fields of Interest (FOI):

		    1. **Sigma**: The Second Piola-Kirchhoff stress tensor (:math:`\mathbf{S}`).
		    2. **sigma**: The Cauchy stress tensor (:math:`\mathbf{\sigma}`).


		:param solid_behavior_model: The name of the hyperelastic constitutive
		    model (e.g., "NeoHookean", "HolzapfelOgden").
		:type solid_behavior_model: str
		:param solid_behavior_parameters: Dictionary of material constants
		    required by the model.
		:type solid_behavior_parameters: dict
		:return: The instantiated hyperelasticity operator.
		:rtype: dmech.HyperElasticityOperator
		"""
		operator = operators.HyperElasticity(
			U=self.displacement_perturbation_subsol.subfunc,
			U_test=self.displacement_perturbation_subsol.dsubtest,
			kinematics=self.kinematics,
			material_model=solid_behavior_model,
			material_parameters=solid_behavior_parameters,
			measure=self.dV,
			formulation="ener",
		)
		self.add_foi(expr=operator.material.Sigma, fs=self.mfoi_fs, name="Sigma", update_type="project")
		self.add_foi(expr=operator.material.sigma, fs=self.mfoi_fs, name="sigma", update_type="project")

		return self.add_operator(operator)

	def add_macroscopic_stretch_symmetry_penalty_operator(self, **kwargs):
		r"""Adds a penalty operator to enforce the symmetry of the macroscopic stretch tensor.

		In the multiscale formulation, the macroscopic stretch :math:`\mathbf{U}_{bar}`
		is often defined in a non-symmetric tensor space. To ensure it represents a
		purely symmetric deformation gradient (consistent with the physics of
		small or large strain stretching), this operator adds a penalty term to the
		global energy functional.

		The penalty contribution to the residual is:

		.. math::
		    \delta \Pi_{sym} = \int_{\Omega} \eta \, \text{skew}(\mathbf{U}_{bar}) : \text{skew}(\delta \mathbf{U}_{bar}) \, d\Omega

		where:
		    - :math:`\eta` is a large penalty parameter (typically ``pen_val=1e6``).
		    - :math:`\text{skew}(\mathbf{A}) = \frac{1}{2}(\mathbf{A} - \mathbf{A}^T)`.



		This effectively forces the off-diagonal components of :math:`\mathbf{U}_{bar}`
		to satisfy :math:`U_{ij} = U_{ji}`, preventing spurious rigid body
		rotations at the macroscopic level.

		:param kwargs: Keyword arguments passed to the penalty operator,
		    most notably ``pen_val`` (float) to define the penalty weight.
		:return: The instantiated symmetry penalty operator.
		:rtype: dmech.MacroscopicStretchSymmetryPenaltyOperator
		"""
		operator = operators.penalty.MacroscopicStretchSymmetry(
			U_bar=self.macroscopic_stretch_subsol.subfunc,
			sol=self.sol_func,
			sol_test=self.dsol_test,
			measure=self.dV,
			**kwargs,
		)
		return self.add_operator(operator)

	def add_macroscopic_stretch_component_penalty_operator(self, k_step=None, **kwargs):
		r"""Adds a penalty operator to prescribe specific components of the macroscopic stretch.

		This method is used to control the macroscopic loading of the Representative
		Elementary Volume (REV). It allows the user to "fix" one or more components
		of the macroscopic stretch tensor :math:`\mathbf{U}_{bar}` to a target value.

		The penalty contribution to the residual for a specific component :math:`(i,j)` is:

		.. math::
		    \delta \Pi_{comp} = \int_{\Omega} \eta (U_{ij} - \bar{U}_{ij}) \delta U_{ij} \, d\Omega

		where:
		    - :math:`\eta` is the penalty parameter (passed via ``pen_val``).
		    - :math:`\bar{U}_{ij}` is the target prescribed stretch value.



		This approach is often preferred over strong Dirichlet constraints for
		global variables in mixed formulations to maintain better solver
		conditioning.

		:param k_step: The index of the load step to which this operator belongs.
		:type k_step: int, optional
		:param kwargs: Keyword arguments used to define the constraint:
		    - ``component`` (tuple): Indices of the tensor component (e.g., (0,0)).
		    - ``val`` (float/Constant): Target value for the component.
		    - ``pen_val`` (float): Penalty stiffness (defaults to a high value).
		:return: The instantiated component penalty operator.
		:rtype: dmech.MacroscopicStretchComponentPenaltyOperator
		"""
		operator = operators.penalty.MacroscopicStretchComponent(
			U_bar=self.macroscopic_stretch_subsol.subfunc,
			U_bar_test=self.macroscopic_stretch_subsol.dsubtest,
			measure=self.dV,
			**kwargs,
		)
		return self.add_operator(operator, k_step=k_step)

	def add_macroscopic_stress_component_constraint_operator(self, k_step=None, **kwargs):
		r"""Adds an operator to prescribe a specific component of the macroscopic stress.

		In multi-scale modeling, this operator enables **stress-controlled loading**.
		It enforces that the volume average of the microscopic stress (Cauchy or
		First Piola-Kirchhoff) matches a target macroscopic stress value :math:`\bar{\Sigma}_{ij}`.

		The constraint is typically added to the virtual work equation as:

		.. math::
		    \delta \Pi_{stress} = (\langle \sigma_{ij} \rangle - \bar{\Sigma}_{ij}) \cdot \delta U_{bar, ij}

		where:
		    - :math:`\langle \sigma_{ij} \rangle` is the volume average of the
		      micro-stress over the unit cell.
		    - :math:`\delta U_{bar, ij}` is the virtual macroscopic stretch.



		This allows for the simulation of "free" boundary conditions in specific
		directions (by setting the target stress to zero) or specific pressure
		states.

		.. warning::
		    This method currently assumes the problem contains only a single
		    elasticity operator from which the material constitutive law
		    can be extracted.

		:param k_step: The index of the load step to which this operator belongs.
		:type k_step: int, optional
		:param kwargs: Keyword arguments to define the constraint:
		    - ``component`` (tuple): Indices of the stress tensor (e.g., (0,0)).
		    - ``val`` (float/Constant): The target macroscopic stress value.
		:return: The instantiated macroscopic stress constraint operator.
		:rtype: dmech.MacroscopicStressComponentConstraintOperator
		"""
		for (
			operator
		) in self.operators:  # MG20221110: Warning! Only works if there is a single operator with a material law!!
			if hasattr(operator, "material"):
				material = operator.material
				break

		operator = operators.MacroscopicStressComponentConstraint(
			U_bar=self.macroscopic_stretch_subsol.subfunc,
			U_bar_test=self.macroscopic_stretch_subsol.dsubtest,
			kinematics=self.kinematics,
			material=material,
			V0=self.V0,
			Vs0=self.Vs0,
			measure=self.dV,
			N=self.mesh_normals,
			**kwargs,
		)
		return self.add_operator(operator, k_step=k_step)

	def add_surface_pressure_loading_operator(self, k_step=None, **kwargs):
		r"""Adds a follower-force pressure loading operator to the internal pore surfaces.

		This operator represents the external work done by a fluid pressure :math:`p_f`
		acting on the boundaries of the solid phase. In a finite strain context,
		this is a "follower load," meaning the force vector evolves as the surface
		deforms and rotates.

		The contribution to the virtual work is:

		.. math::
		    \delta \Pi_{ext} = \int_{\partial \Omega} p_f \cdot J \mathbf{F}^{-T} \mathbf{N} \cdot \delta \mathbf{u} \, dA_0

		where:
		    - :math:`p_f` is the fluid pressure magnitude.
		    - :math:`\mathbf{F}^{-T} \mathbf{N}` accounts for the deformation of the
		      surface normal (Nanson's formula).
		    - :math:`J` is the determinant of the deformation gradient.



		This is particularly useful for modeling the inflation of lung alveoli or
		pressurized porous scaffolds where the internal geometry changes
		significantly during loading.

		:param k_step: The index of the load step to which this operator belongs.
		:type k_step: int, optional
		:param kwargs: Keyword arguments passed to the operator:
		    - ``P`` (float/Constant): The pressure value to apply.
		    - ``subdomain_id`` (int): The boundary ID representing the pore interface.
		:return: The instantiated surface pressure loading operator.
		:rtype: dmech.SurfacePressureLoadingOperator
		"""
		operator = operators.loading.SurfacePressure(
			U_test=self.displacement_perturbation_subsol.dsubtest,
			kinematics=self.kinematics,
			N=self.mesh_normals,
			**kwargs,
		)
		return self.add_operator(operator=operator, k_step=k_step)

	def add_surface_tension_loading_operator(self, k_step=None, **kwargs):
		r"""Adds a surface tension loading operator to the interfacial boundaries.

		This operator models the energy contribution of surface tension :math:`\gamma`
		acting on the pore-solid interface. Mathematically, it represents the
		first variation of the surface energy :math:`\Psi_s = \int \gamma \, da`.

		The contribution to the virtual work is:

		.. math::
		    \delta \Pi_{st} = \int_{\partial \Omega} \gamma \left( J \mathbf{F}^{-T} \mathbf{N} \cdot \text{grad}(\delta \mathbf{u}) \cdot \mathbf{n} \right) dA_0

		In a finite strain context, surface tension is a "configuration-dependent"
		load. As the surface area :math:`a` changes, the total energy changes,
		resulting in a force that is always tangent to the surface but normal
		to the interface boundaries (Laplace pressure effect).



		This is a critical component for modeling lung parenchyma, where the
		alveolar surface tension provides a significant portion of the lung's
		elastic recoil.

		:param k_step: The index of the load step to which this operator belongs.
		:type k_step: int, optional
		:param kwargs: Keyword arguments passed to the operator:
		    - ``gamma`` (float/Constant): The surface tension coefficient.
		    - ``subdomain_id`` (int): The boundary ID of the interface.
		:return: The instantiated surface tension loading operator.
		:rtype: dmech.SurfaceTensionLoadingOperator
		"""
		operator = operators.loading.SurfaceTension(
			kinematics=self.kinematics, N=self.mesh_normals, U_test=self.U_tot_test, **kwargs
		)
		return self.add_operator(operator=operator, k_step=k_step)

	def add_deformed_total_volume_operator(self, k_step=None):
		r"""Adds an operator to enforce the constraint on the total deformed volume.

		In multi-scale poro-hyperelasticity, the total volume of the unit cell
		:math:`v` is determined by the macroscopic deformation. This operator
		links the global scalar sub-solution :math:`v` to the macroscopic
		stretch :math:`\mathbf{U}_{bar}`.

		The constraint is enforced by ensuring that the current volume equals the
		reference volume :math:`V_0` scaled by the determinant of the macroscopic
		deformation gradient :math:`J_{bar} = \det(\mathbf{I} + \mathbf{U}_{bar})`.

		The contribution to the virtual work (residual) is:

		.. math::
		    \delta \Pi_{v} = \int_{\Omega} (v - J_{bar} V_0) \delta v \, d\Omega



		This operator is essential for problems where the pore fluid pressure or
		total porosity is coupled to the overall expansion or contraction of
		the unit cell.

		:param k_step: The index of the load step to which this operator belongs.
		:type k_step: int, optional
		:return: None. The operator is registered within the problem's operator list.
		"""
		operator = operators.microporo.DeformedTotalVolume(
			v=self.deformed_total_volume_subsol.subfunc,
			v_test=self.deformed_total_volume_subsol.dsubtest,
			U_bar=self.macroscopic_stretch_subsol.subfunc,
			V0=self.V0,
			measure=self.dV,
		)
		self.add_operator(operator=operator, k_step=k_step)

	def add_deformed_solid_volume_operator(self, k_step=None):
		r"""Adds an operator to enforce the constraint on the deformed solid volume.

		This operator links the global scalar sub-solution :math:`v_s` to the
		local kinematics of the solid phase. It ensures that the tracked solid
		volume is consistent with the spatial integration of the local
		Jacobian (determinant of the deformation gradient) over the solid domain.

		The constraint is enforced via the following variational contribution
		to the residual:

		.. math::
		    \delta \Pi_{v_s} = \int_{\Omega_s} (v_s - J) \delta v_s \, d\Omega_0

		where:
		    - :math:`v_s` is the global scalar for the deformed solid volume.
		    - :math:`J` is the local determinant of the deformation gradient
		      (:math:`\det \mathbf{F}`).
		    - :math:`\delta v_s` is the test function associated with the
		      global solid volume.



		This operator is critical for calculating the current solid volume fraction
		(solidity) and, by extension, the porosity in micro-porous hyperelastic
		simulations.

		:param k_step: The index of the load step to which this operator belongs.
		:type k_step: int, optional
		:return: None. The operator is registered within the problem's operator list.
		"""
		operator = operators.microporo.DeformedSolidVolume(
			vs=self.deformed_solid_volume_subsol.subfunc,
			vs_test=self.deformed_solid_volume_subsol.dsubtest,
			J=self.kinematics.J,
			Vs0=self.mesh_V0,
			measure=self.dV,
		)
		self.add_operator(operator=operator, k_step=k_step)

	def add_deformed_fluid_volume_operator(self, k_step=None):
		r"""Adds an operator to enforce the constraint on the deformed fluid (pore) volume.

		This operator links the global scalar sub-solution :math:`v_f` to the
		kinematics of the internal boundaries. Since the fluid phase is typically
		not meshed, its volume is calculated using a boundary integral based on
		the movement of the pore-solid interface.



		The calculation often relies on the identity for the volume of a region
		bounded by a surface :math:`\partial \Omega_f`:

		.. math::
		    v_f = \frac{1}{dim} \int_{\partial \Omega_f} \mathbf{x} \cdot \mathbf{n} \, da

		where :math:`\mathbf{x}` is the current position and :math:`\mathbf{n}` is
		the outward spatial normal. The operator ensures that the global variable
		:math:`v_f` satisfies this kinematic relationship as the pores expand
		or contract.

		:param k_step: The index of the load step to which this operator belongs.
		:type k_step: int, optional
		:return: None. The operator is registered within the problem's operator list.
		"""
		operator = operators.microporo.DeformedFluidVolume(
			vf=self.deformed_fluid_volume_subsol.subfunc,
			vf_test=self.deformed_fluid_volume_subsol.dsubtest,
			kinematics=self.kinematics,
			N=self.mesh_normals,
			dS=self.dS,
			U_tot=self.U_tot,
			X=self.X,
			measure=self.dV,
		)
		self.add_operator(operator=operator, k_step=k_step)

	def add_surface_area_operator(self, k_step=None, **kwargs):
		r"""Adds an operator to enforce the constraint on the deformed interfacial surface area.

		This operator links the global scalar sub-solution :math:`s` to the
		actual integrated area of the deformed mesh facets. It uses Nanson's
		formula to map the reference surface area elements to the spatial
		configuration.

		The constraint is enforced via the following variational contribution:

		.. math::
		    \delta \Pi_{s} = \int_{\partial \Omega} (s - \| J \mathbf{F}^{-T} \mathbf{N} \|) \delta s \, dA_0

		where:
		    - :math:`s` is the global scalar for the deformed surface area.
		    - :math:`J \mathbf{F}^{-T} \mathbf{N}` is the transformation of the
		      area element (Nanson's formula).
		    - :math:`\delta s` is the test function associated with the
		      global surface area.



		This operator allows the surface area to be treated as a primary
		unknown, which is essential for coupling mechanical deformation with
		surface-dependent physics like alveolar surface tension in lung models.

		:param k_step: The index of the load step to which this operator belongs.
		:type k_step: int, optional
		:param kwargs: Keyword arguments passed to the operator, typically
		    including ``subdomain_id`` to specify which boundary to integrate.
		:return: The instantiated surface area operator.
		:rtype: dmech.DeformedSurfaceAreaOperator
		"""
		operator = operators.microporo.DeformedSurfaceArea(
			S_area=self.surface_area_subsol.subfunc,
			S_area_test=self.surface_area_subsol.dsubtest,
			kinematics=self.kinematics,
			N=self.mesh_normals,
			**kwargs,
		)
		return self.add_operator(operator=operator, k_step=k_step)

	def add_kubc(self, xmin_id=1, xmax_id=2, ymin_id=3, ymax_id=4, zmin_id=5, zmax_id=6):
		r"""Applies Kinematic Uniform Boundary Conditions (KUBC) to the unit cell.

		KUBC is a standard homogenization boundary condition where the
		displacement fluctuation :math:`\mathbf{U}_{tilde}` is set to zero
		on all external faces of the Representative Elementary Volume (REV).

		.. math::
		    \mathbf{U}_{tilde} = \mathbf{0} \quad \text{on } \partial \Omega_{ext}



		By enforcing this, the total displacement on the boundary is purely
		dictated by the macroscopic stretch :math:`\mathbf{U}_{bar}`, satisfying
		the Hill-Mandel condition. This method specifically constrains the
		normal components of the fluctuation field on each respective face to
		ensure the cell remains compatible with a larger homogeneous medium.

		:param xmin_id: Subdomain ID for the negative X-face.
		:param xmax_id: Subdomain ID for the positive X-face.
		:param ymin_id: Subdomain ID for the negative Y-face.
		:param ymax_id: Subdomain ID for the positive Y-face.
		:param zmin_id: Subdomain ID for the negative Z-face (3D only).
		:param zmax_id: Subdomain ID for the positive Z-face (3D only).
		"""
		self.add_constraint(
			V=self.displacement_perturbation_subsol.fs.sub(0),
			sub_domains=self.boundaries,
			sub_domain_id=xmin_id,
			val=0.0,
		)
		self.add_constraint(
			V=self.displacement_perturbation_subsol.fs.sub(0),
			sub_domains=self.boundaries,
			sub_domain_id=xmax_id,
			val=0.0,
		)
		self.add_constraint(
			V=self.displacement_perturbation_subsol.fs.sub(1),
			sub_domains=self.boundaries,
			sub_domain_id=ymin_id,
			val=0.0,
		)
		self.add_constraint(
			V=self.displacement_perturbation_subsol.fs.sub(1),
			sub_domains=self.boundaries,
			sub_domain_id=ymax_id,
			val=0.0,
		)
		if self.dim == 3:
			self.add_constraint(
				V=self.displacement_perturbation_subsol.fs.sub(2),
				sub_domains=self.boundaries,
				sub_domain_id=zmin_id,
				val=0.0,
			)
			self.add_constraint(
				V=self.displacement_perturbation_subsol.fs.sub(2),
				sub_domains=self.boundaries,
				sub_domain_id=zmax_id,
				val=0.0,
			)

	def add_deformed_solid_volume_qoi(self):
		r"""Adds a Quantity of Interest (QOI) for the total deformed solid volume.

		This method registers an integral expression to track the evolution of
		the solid phase volume :math:`v_s` throughout the simulation. It
		integrates the local Jacobian :math:`J` (determinant of the deformation
		gradient) over the reference solid domain :math:`\Omega_{s,0}`.

		.. math::
		    v_s = \int_{\Omega_{s,0}} J \, dV



		The resulting value represents the current volume occupied by the
		material skeleton (e.g., the tissue in a lung model). This is
		distinguished from the total unit cell volume, which includes
		the voids or pores.

		:return: None. The QOI is added to the problem's internal QOI list.
		"""
		self.add_qoi(name="vs", expr=self.kinematics.J * self.dV)

	def add_deformed_fluid_volume_qoi(self):
		r"""Adds a Quantity of Interest (QOI) for the total deformed fluid (pore) volume.

		In the multiscale homogenization of porous media, the fluid volume :math:`v_f` 
		is calculated as the difference between the total deformed volume of the 
		unit cell :math:`v` and the integrated volume of the solid skeleton :math:`v_s`.

        

		The total volume :math:`v` is determined by the macroscopic deformation 
		gradient :math:`\mathbf{F}_{bar}`:

		.. math::
		    \mathbf{F}_{bar} = \mathbf{I} + \mathbf{U}_{bar} \
		    v = \det(\mathbf{F}_{bar}) V_0

		The QOI expression uses the fact that :math:`v_f = \int_{\Omega_{s,0}} (\phi_{f}/ \phi_{s,0} ) J dV` 
		is reformulated here as an integral over the solid mesh for numerical 
		consistency within the FEniCS integration framework.

		:return: None. The QOI is registered for post-processing.
		"""
		U_bar = self.macroscopic_stretch_subsol.subfunc
		I_bar = dolfin.Identity(self.dim)
		F_bar = I_bar + U_bar
		J_bar = dolfin.det(F_bar)
		v = J_bar * self.V0

		self.add_qoi(name="vf", expr=(v / self.Vs0 - self.kinematics.J) * self.dV)

	def add_deformed_volume_qoi(self):
		r"""Adds a Quantity of Interest (QOI) for the total deformed volume of the unit cell.

		This method calculates the current volume :math:`v` of the entire Representative 
		Elementary Volume (REV), including both the solid skeleton and the pore space. 
		It is derived from the macroscopic deformation gradient :math:`\mathbf{F}_{bar}`.

		The total deformed volume is given by:

		.. math::
		    \mathbf{F}_{bar} = \mathbf{I} + \mathbf{U}_{bar} \
		    v = \det(\mathbf{F}_{bar}) V_0

        

		To integrate this global scalar value within the FEniCS framework (which 
		integrates over the meshed solid domain :math:`\Omega_s`), the expression 
		is normalized by the initial solid volume :math:`V_{s0}`:

		.. math::
		    v = \int_{\Omega_s} \frac{v_{tot}}{V_{s0}} \, dV

		:return: None. The QOI is registered for post-processing and output.
		"""
		U_bar = self.macroscopic_stretch_subsol.subfunc
		I_bar = dolfin.Identity(self.dim)
		F_bar = I_bar + U_bar
		J_bar = dolfin.det(F_bar)
		v = J_bar * self.V0

		self.add_qoi(name="v", expr=(v / self.Vs0) * self.dV)

	def add_macroscopic_tensor_qois(self, basename, subsol, symmetric=False):
		r"""Helper method to add Quantities of Interest (QOI) for all components of a macroscopic tensor.

		This method iterates through the components of a global tensor sub-solution
		(defined in the **Real (R)** family) and registers each one as a separate
		scalar QOI. Since these are global variables, the value is extracted
		"pointwise" at a single mesh coordinate.



		The naming convention follows the format ``basename_IJ`` (e.g., ``U_bar_XX``,
		``U_bar_XY``).

		:param basename: The prefix used for the QOI names.
		:type basename: str
		:param subsol: The global tensor sub-solution to extract components from.
		:type subsol: SubSolution
		:param symmetric: If True, skips redundant off-diagonal components
		    (e.g., registers XY but skips YX).
		:type symmetric: bool, optional
		"""
		self.add_qoi(
			name=basename + "_XX", expr=subsol.subfunc[0, 0], point=self.mesh.coordinates()[0], update_type="direct"
		)
		if self.dim >= 2:
			self.add_qoi(
				name=basename + "_YY", expr=subsol.subfunc[1, 1], point=self.mesh.coordinates()[0], update_type="direct"
			)
			if self.dim >= 3:
				self.add_qoi(
					name=basename + "_ZZ",
					expr=subsol.subfunc[2, 2],
					point=self.mesh.coordinates()[0],
					update_type="direct",
				)
		if self.dim >= 2:
			self.add_qoi(
				name=basename + "_XY", expr=subsol.subfunc[0, 1], point=self.mesh.coordinates()[0], update_type="direct"
			)
			if not (symmetric):
				self.add_qoi(
					name=basename + "_YX",
					expr=subsol.subfunc[1, 0],
					point=self.mesh.coordinates()[0],
					update_type="direct",
				)
			if self.dim >= 3:
				self.add_qoi(
					name=basename + "_YZ",
					expr=subsol.subfunc[1, 2],
					point=self.mesh.coordinates()[0],
					update_type="direct",
				)
				if not (symmetric):
					self.add_qoi(
						name=basename + "_ZY",
						expr=subsol.subfunc[2, 1],
						point=self.mesh.coordinates()[0],
						update_type="direct",
					)
				self.add_qoi(
					name=basename + "_ZX",
					expr=subsol.subfunc[2, 0],
					point=self.mesh.coordinates()[0],
					update_type="direct",
				)
				if not (symmetric):
					self.add_qoi(
						name=basename + "_XZ",
						expr=subsol.subfunc[0, 2],
						point=self.mesh.coordinates()[0],
						update_type="direct",
					)

	def add_macroscopic_stretch_qois(self):
		r"""Registers all components of the macroscopic stretch tensor as Quantities of Interest.

		This method uses :meth:`add_macroscopic_tensor_qois` to extract the individual
		components of the global macroscopic stretch :math:`\mathbf{U}_{bar}`. In the
		context of displacement decomposition, :math:`\mathbf{U}_{bar}` represents the
		average displacement gradient across the Representative Elementary Volume (REV).



		The registered QOIs will be named ``U_bar_XX``, ``U_bar_YY``, etc. These
		values are essential for verifying the applied loading conditions and for
		constructing macroscopic stress-strain relationships.

		.. note::
		    Since the macroscopic stretch is generally enforced to be symmetric via
		    penalty operators, the off-diagonal terms (e.g., XY and YX) should
		    ideally be identical.

		:return: None. Component-wise QOIs are added to the problem.
		"""
		self.add_macroscopic_tensor_qois(basename="U_bar", subsol=self.macroscopic_stretch_subsol)

	def add_macroscopic_solid_stress_qois(self, symmetric=False):
		r"""Registers Quantities of Interest (QOI) for the homogenized solid Cauchy stress components.

		This method calculates the macroscopic contribution of the solid phase to the
		overall stress state. It integrates the local Cauchy stress :math:`\sigma` of the
		solid material, weighted by the local Jacobian :math:`J`, and averages it over
		the current total volume :math:`v` of the unit cell.

		The homogenization formula for the solid stress component :math:`ij` is:

		.. math::
		    \bar{\sigma}_{s, ij} = \frac{1}{v} \int_{\Omega_{s,0}} \sigma_{ij} J \, dV



		This calculation is fundamental for determining the effective mechanical
		properties of porous media, as it captures how the microscopic solid
		skeleton carries loads at the macroscopic scale.

		.. warning::
		    This method assumes a single elasticity operator is present in the
		    problem to retrieve the material constitutive law.

		:param symmetric: If True, only the independent components (XX, YY, ZZ, XY,
		    YZ, ZX) are registered to save output space.
		:type symmetric: bool, optional
		:return: None. Component-wise solid stress QOIs are added to the problem.
		"""
		for (
			operator
		) in self.operators:  # MG20221110: Warning! Only works if there is a single operator with a material law!!
			if hasattr(operator, "material"):
				material = operator.material
				break

		U_bar = self.macroscopic_stretch_subsol.subfunc
		I_bar = dolfin.Identity(self.dim)
		F_bar = I_bar + U_bar
		J_bar = dolfin.det(F_bar)
		v = J_bar * self.V0

		self.add_qoi(name="sigma_s_bar_XX", expr=(material.sigma[0, 0] * self.kinematics.J) / v * self.dV)
		if self.dim >= 2:
			self.add_qoi(name="sigma_s_bar_YY", expr=(material.sigma[1, 1] * self.kinematics.J) / v * self.dV)
			if self.dim >= 3:
				self.add_qoi(name="sigma_s_bar_ZZ", expr=(material.sigma[2, 2] * self.kinematics.J) / v * self.dV)
		if self.dim >= 2:
			self.add_qoi(name="sigma_s_bar_XY", expr=(material.sigma[0, 1] * self.kinematics.J) / v * self.dV)
			if not (symmetric):
				self.add_qoi(name="sigma_s_bar_YX", expr=(material.sigma[1, 0] * self.kinematics.J) / v * self.dV)
			if self.dim >= 3:
				self.add_qoi(name="sigma_s_bar_YZ", expr=(material.sigma[1, 2] * self.kinematics.J) / v * self.dV)
				if not (symmetric):
					self.add_qoi(name="sigma_s_bar_ZY", expr=(material.sigma[2, 1] * self.kinematics.J) / v * self.dV)
				self.add_qoi(name="sigma_s_bar_ZX", expr=(material.sigma[2, 0] * self.kinematics.J) / v * self.dV)
				if not (symmetric):
					self.add_qoi(name="sigma_s_bar_XZ", expr=(material.sigma[0, 2] * self.kinematics.J) / v * self.dV)

	def add_macroscopic_solid_hydrostatic_pressure_qoi(self):
		r"""Registers a Quantity of Interest (QOI) for the homogenized solid hydrostatic pressure.

		This method calculates the macroscopic hydrostatic pressure :math:`\bar{p}_{hydro}`
		by averaging the local hydrostatic pressure of the solid material over the
		current total volume :math:`v` of the unit cell.

		In continuum mechanics, the hydrostatic pressure is defined as the negative
		one-third of the trace of the Cauchy stress tensor:

		.. math::
		    p = -\frac{1}{3} \text{tr}(\sigma)



		The homogenization is performed by integrating the local pressure :math:`p`
		weighted by the local Jacobian :math:`J` over the reference solid domain:

		.. math::
		    \bar{p}_{hydro} = \frac{1}{v} \int_{\Omega_{s,0}} p_{hydro} J \, dV

		This value is a key indicator of the volumetric loading state of the
		homogenized material point.

		.. warning::
		    This method assumes a single elasticity operator is present in the
		    problem to retrieve the hydrostatic pressure from the material behavior.

		:return: None. The hydrostatic pressure QOI is added to the problem.
		"""
		for (
			operator
		) in self.operators:  # MG20221110: Warning! Only works if there is a single operator with a material law!!
			if hasattr(operator, "material"):
				material = operator.material
				break

		U_bar = self.macroscopic_stretch_subsol.subfunc
		I_bar = dolfin.Identity(self.dim)
		F_bar = I_bar + U_bar
		J_bar = dolfin.det(F_bar)
		v = J_bar * self.V0

		self.add_qoi(name="p_hydro", expr=(material.p_hydro * self.kinematics.J) / v * self.dV)

	def add_fluid_pressure_qoi(self):
		r"""Registers a Quantity of Interest (QOI) for the pore fluid pressure across all steps.

		This method extracts the fluid pressure value :math:`p_f` from the operators
		assigned to each simulation step. In micro-poro-hyperelasticity, :math:`p_f`
		represents the pressure within the voids/pores of the microstructure.



		The value is normalized by the initial solid volume :math:`V_{s0}` to
		maintain consistency with the integration over the solid mesh:

		.. math::
		    \bar{p}_f = \frac{1}{V_{s0}} \int_{\Omega_s} p_f \, dV

		Since the fluid pressure may vary or be defined by different time-varying
		functions in different steps, this method constructs a list of
		expressions (``expr_lst``), one for each step in the simulation.

		.. note::
		    The method searches for an operator possessing the ``tv_pf``
		    (time-varying pore fluid) attribute within each step.

		:return: None. The fluid pressure QOI is added to the problem's QOI list.
		"""
		expr_lst = []
		for i in range(len(self.steps)):
			for operator in self.steps[i].operators:
				if hasattr(operator, "tv_pf"):
					tv_pf = operator.tv_pf
					break
			expr_lst.append((tv_pf.val) / self.Vs0 * self.dV)

		self.add_qoi(name="p_f", expr_lst=expr_lst)
		# expr=(tv_pf.val)/self.Vs0 * self.dV)

	def add_macroscopic_stress_qois(self, symmetric=False):
		r"""Adds Quantities of Interest (QOI) for the homogenized macroscopic stress.

		This computes the effective stress :math:`\bar{\sigma}` by averaging the
		local solid stress and fluid pressure contributions over the deformed
		volume :math:`v`.

		.. math::
		    \bar{\sigma} = \frac{1}{v} \int_{\Omega} (\sigma_{s} J - \phi_{f} p_{f} J) d\Omega
		"""
		for (
			operator
		) in self.operators:  # MG20221110: Warning! Only works if there is a single operator with a material law!!
			if hasattr(operator, "material"):
				material = operator.material
				break

		for operator in self.steps[0].operators:  # MG20231124: Warning! Only works if there is a single step!!
			if hasattr(operator, "tv_pf"):
				tv_pf = operator.tv_pf
				break

		U_bar = self.macroscopic_stretch_subsol.subfunc
		I_bar = dolfin.Identity(self.dim)
		F_bar = I_bar + U_bar
		J_bar = dolfin.det(F_bar)
		v = J_bar * self.V0

		self.add_qoi(
			name="sigma_bar_XX",
			expr=(material.sigma[0, 0] * self.kinematics.J - (v / self.Vs0 - self.kinematics.J) * tv_pf.val)
			/ v
			* self.dV,
		)
		if self.dim >= 2:
			self.add_qoi(
				name="sigma_bar_YY",
				expr=(material.sigma[1, 1] * self.kinematics.J - (v / self.Vs0 - self.kinematics.J) * tv_pf.val)
				/ v
				* self.dV,
			)
			if self.dim >= 3:
				self.add_qoi(
					name="sigma_bar_ZZ",
					expr=(material.sigma[2, 2] * self.kinematics.J - (v / self.Vs0 - self.kinematics.J) * tv_pf.val)
					/ v
					* self.dV,
				)
		if self.dim >= 2:
			self.add_qoi(name="sigma_bar_XY", expr=(material.sigma[0, 1] * self.kinematics.J) / v * self.dV)
			if not (symmetric):
				self.add_qoi(name="sigma_bar_YX", expr=(material.sigma[1, 0] * self.kinematics.J) / v * self.dV)
			if self.dim >= 3:
				self.add_qoi(name="sigma_bar_YZ", expr=(material.sigma[1, 2] * self.kinematics.J) / v * self.dV)
				if not (symmetric):
					self.add_qoi(name="sigma_bar_ZY", expr=(material.sigma[2, 1] * self.kinematics.J) / v * self.dV)
				self.add_qoi(name="sigma_bar_ZX", expr=(material.sigma[2, 0] * self.kinematics.J) / v * self.dV)
				if not (symmetric):
					self.add_qoi(name="sigma_bar_XZ", expr=(material.sigma[0, 2] * self.kinematics.J) / v * self.dV)

	def add_interfacial_surface_qois(self):
		r"""Adds QOI for the deformed interfacial surface area.

		Uses Nanson's formula to track how the internal pore surface area
		evolves under large deformation.
		"""
		FmTN = dolfin.dot(dolfin.inv(self.kinematics.F).T, self.mesh_normals)
		T = dolfin.sqrt(dolfin.inner(FmTN, FmTN))
		expr = T * self.kinematics.J
		self.add_qoi(name="S_area", expr=expr * self.dS(0))
