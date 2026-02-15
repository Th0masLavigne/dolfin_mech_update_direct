# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Defines the ElasticityProblem class.

Specialized manager for linearized
elastic simulations that automates the setup of infinitesimal kinematics,
multi-material elasticity operators, and stable mixed displacement-pressure
formulations for incompressible media.
"""

from ..kinematics import LinearizedKinematics
from ..operators import LinearizedElasticity, LinearizedHydrostaticPressure, LinearizedIncompressibility
from .problem import Problem

################################################################################


class Elasticity(Problem):
	"""Problem class for linearized elasticity simulations.

	This class manages the lifecycle of a linear elastic simulation, including:
	    - Mesh and measure initialization.
	    - Finite element spaces for displacement and (optionally) pressure.
	    - Linearized kinematics (infinitesimal strain).
	    - Material operator assembly (e.g., Hooke's Law).
	    - Fields of Interest (FOI) and Quantities of Interest (QOI) for post-processing.

	The problem can handle mixed formulations (u-p) for incompressible materials
	by setting ``w_incompressibility=True``.

	Attributes:
	    w_incompressibility (bool): If True, a pressure field is added to enforce
	                                incompressibility.
	    kinematics (LinearizedKinematics): Object handling infinitesimal strain.
	    displacement_subsol (SubSolution): Container for displacement fields.
	    pressure_subsol (SubSolution): Container for pressure fields (if used).
	"""

	def __init__(
		self,
		w_incompressibility=False,
		mesh=None,
		define_facet_normals=False,
		domains_mf=None,
		boundaries_mf=None,
		points_mf=None,
		displacement_degree=None,
		pressure_degree=None,
		quadrature_degree=None,
		foi_degree=0,
		elastic_behavior=None,
		elastic_behaviors=None,
	):
		"""Initializes the ElasticityProblem.

		:param w_incompressibility: Enable mixed formulation for incompressibility.
		:type w_incompressibility: bool
		:param mesh: Dolfin mesh object.
		:param displacement_degree: Polynomial degree for displacement.
		:param pressure_degree: Polynomial degree for pressure.
		:param foi_degree: Polynomial degree for Fields of Interest (DG).
		:param elastic_behavior: Dictionary defining a single material behavior.
		:param elastic_behaviors: List of dictionaries for multi-material domains.
		"""
		Problem.__init__(self)

		self.w_incompressibility = w_incompressibility

		self.set_mesh(mesh=mesh, define_facet_normals=define_facet_normals)

		self.set_measures(domains=domains_mf, boundaries=boundaries_mf, points=points_mf)

		self.set_subsols(displacement_degree=displacement_degree, pressure_degree=pressure_degree)

		self.set_solution_finite_element()
		self.set_solution_function_space()
		self.set_solution_functions()

		self.set_quadrature_degree(quadrature_degree=quadrature_degree)

		self.set_foi_finite_elements_DG(degree=foi_degree)
		self.set_foi_function_spaces()

		self.set_kinematics()

		if elastic_behavior is not None:
			elastic_behaviors = [elastic_behavior]

		self.add_elasticity_operators(elastic_behaviors=elastic_behaviors)

	def add_displacement_subsol(self, degree):
		r"""Adds the displacement vector sub-solution to the problem's solution space.

		The displacement field :math:`\mathbf{u}` is defined using the **Continuous
		Galerkin (CG)** family, which ensures that the displacement is continuous
		across element boundaries (standard Lagrange elements).



		In a displacement-based finite element formulation, the polynomial degree
		of this sub-solution determines the spatial resolution and accuracy of
		the strain and stress calculations:

		* **Degree 1**: Linear elements. Strains and stresses are constant
		  within each element.
		* **Degree 2**: Quadratic elements. Strains and stresses vary linearly
		  within each element, often providing better convergence for complex
		  geometries and bending problems.

		:param degree: Polynomial degree for the displacement interpolation.
		:type degree: int
		"""
		self.displacement_degree = degree
		self.displacement_subsol = Problem.add_vector_subsol(
			self, name="u", family="CG", degree=self.displacement_degree
		)

	def add_pressure_subsol(self, degree):
		"""Adds the pressure scalar sub-solution to the problem's solution space.

		The choice of finite element family depends on the requested polynomial
		degree:

		* **Degree 0**: Uses a **Discontinuous Galerkin (DG)** family. This
		    results in a pressure field that is constant within each element
		    but discontinuous across cell boundaries.
		* **Degree > 0**: Uses a **Continuous Galerkin (CG)** family (Standard
		    Lagrange elements). This results in a pressure field that is
		    continuous across the entire domain.



		In mixed formulations, the choice of pressure space is critical for
		stability. For instance, a :math:`P_2-P_1` (Taylor-Hood) pair uses
		continuous linear pressure, while a :math:`P_1-P_0` pair uses
		discontinuous constant pressure.

		:param degree: Polynomial degree for the pressure interpolation.
		:type degree: int
		"""
		self.pressure_degree = degree
		if self.pressure_degree == 0:
			self.pressure_subsol = self.add_scalar_subsol(name="p", family="DG", degree=self.pressure_degree)
		else:
			self.pressure_subsol = self.add_scalar_subsol(name="p", family="CG", degree=self.pressure_degree)

	def set_subsols(self, displacement_degree=1, pressure_degree=None):
		r"""Configures and initializes the solution fields (sub-solutions) for the problem.

		This method defines the functional spaces for the unknowns being solved.
		It always initializes a displacement field :math:`\mathbf{u}`. If the problem
		is flagged for incompressibility (``w_incompressibility=True``), it also
		initializes a pressure field :math:`p`.

		For incompressible or nearly-incompressible materials, the method defaults
		to a **Taylor-Hood** element pair (e.g., :math:`P_k / P_{k-1}`) to satisfy
		the Inf-Sup (Babuska-Brezzi) stability condition, preventing spurious
		pressure oscillations.



		:param displacement_degree: Polynomial degree for the displacement Lagrange
		    elements. Defaults to 1 (linear).
		:type displacement_degree: int
		:param pressure_degree: Polynomial degree for the pressure elements.
		    If ``None`` and incompressibility is enabled, it defaults to
		    ``displacement_degree - 1``.
		:type pressure_degree: int, optional

		.. note::
		    Common stable pairs include:

		    * **P2-P1**: Quadratic displacement, Linear pressure.
		    * **P1-P0**: Linear displacement, Constant pressure (requires specific stabilization).
		"""
		self.add_displacement_subsol(degree=displacement_degree)

		if self.w_incompressibility:
			if pressure_degree is None:
				pressure_degree = displacement_degree - 1
			self.add_pressure_subsol(degree=pressure_degree)

	def set_quadrature_degree(self, quadrature_degree=None):
		r"""Configures the numerical integration precision for the variational forms.

		Quadrature (or Gauss) integration is used to evaluate the integrals in the
		residual and Jacobian. This method sets the polynomial degree that the
		quadrature rule should be able to integrate exactly.

		The ``quadrature_degree`` can be set using the following options:

		* **int**: A specific degree (e.g., ``2``, ``3``).
		* **"full"**: Allows FEniCS/FFC to automatically determine the degree
		    required for exact integration of the current UFL expressions.
		* **"default"**: Uses a heuristic based on the element type and polynomial
		    degree of the displacement to find a balance between speed and precision:

		    * For **Simplex** cells (Triangle/Tet): :math:`\max(2, 2(k-1))`
		    * For **Hypercube** cells (Quad/Hex): :math:`\max(2, 2(dim \cdot k-1))`

		    where :math:`k` is the displacement degree.



		:param quadrature_degree: The desired integration rule setting.
		:type quadrature_degree: int, str, or None
		:raises AssertionError: If an invalid string or type is provided.
		"""
		if (quadrature_degree is None) or (type(quadrature_degree) == int):
			pass
		elif quadrature_degree == "full":
			quadrature_degree = None
		elif quadrature_degree == "default":
			if self.mesh.ufl_cell().cellname() in ("triangle", "tetrahedron"):
				quadrature_degree = max(
					2, 2 * (self.displacement_degree - 1)
				)  # MG20211221: This does not allow to reproduce full integration results exactly, but it is quite close…
			elif self.mesh.ufl_cell().cellname() in ("quadrilateral", "hexahedron"):
				quadrature_degree = max(2, 2 * (self.dim * self.displacement_degree - 1))
		else:
			assert 0, 'Must provide an int, "full", "default" or None. Aborting.'

		Problem.set_quadrature_degree(self, quadrature_degree=quadrature_degree)

	def set_kinematics(self):
		r"""Initializes the linearized kinematic quantities and registers default kinematic fields of interest.

		This method instantiates a :class:`LinearizedKinematics` object using the
		current and previous displacement sub-solutions. This object computes the
		infinitesimal strain tensor :math:`\epsilon`, defined as the symmetric
		part of the displacement gradient:

		.. math::
		    \epsilon = \frac{1}{2} (\nabla \mathbf{u} + \nabla \mathbf{u}^T)

		Additionally, it registers the strain tensor as a Field of Interest (FOI),
		ensuring that :math:`\epsilon` is available for spatial visualization and
		post-processing (e.g., in XDMF files).


		"""
		self.kinematics = LinearizedKinematics(
			u=self.displacement_subsol.subfunc, u_old=self.displacement_subsol.func_old
		)

		self.add_foi(expr=self.kinematics.epsilon, fs=self.mfoi_fs, name="epsilon")

	def get_subdomain_measure(self, subdomain_id=None):
		"""Returns the integration measure associated with a specific subdomain ID.

		In finite element analysis, different material properties or constraints
		are often assigned to different regions of the mesh. This method provides
		the correct volume measure (:math:`dV`) for a given region.

		- If ``subdomain_id`` is ``None``, the measure represents the **entire domain**.
		- If ``subdomain_id`` is an integer, the measure is restricted to cells
		  tagged with that specific value in the domain ``MeshFunction``.



		:param subdomain_id: The marker value identifying the subdomain.
		    Corresponds to the values found in the ``domains_mf`` MeshFunction.
		:type subdomain_id: int, optional
		:return: A Dolfin measure object used in UFL variational forms.
		:rtype: dolfin.Measure
		"""
		if subdomain_id is None:
			return self.dV
		else:
			return self.dV(subdomain_id)

	def add_elasticity_operator(self, material_model, material_parameters, subdomain_id=None):
		r"""Adds a linearized elasticity operator to the variational problem.

		This method defines the internal virtual work contribution for the solid
		skeleton. It associates a specific constitutive behavior (material model)
		with the displacement field through the infinitesimal strain tensor.

		The contribution to the residual form is:

		.. math::
		    \delta \Pi_{int} = \int_{\Omega} \sigma(\epsilon) : \delta \epsilon \, d\Omega

		where:
		    - :math:`\sigma` is the Cauchy stress tensor defined by the
		      ``material_model``.
		    - :math:`\delta \epsilon` is the virtual infinitesimal strain
		      (symmetric part of the gradient of the test function).



		:param material_model: The name of the constitutive model to use (e.g., "Hooke").
		:type material_model: str
		:param material_parameters: A dictionary containing the necessary
		    coefficients for the chosen model (e.g., Young's modulus 'E' and
		    Poisson's ratio 'nu').
		:type material_parameters: dict
		:param subdomain_id: Optional ID to restrict the material behavior to
		    a specific part of the mesh. If None, the operator is applied
		    to the entire volume.
		:type subdomain_id: int, optional
		:return: The instantiated elasticity operator.
		:rtype: dmech.LinearizedElasticityOperator
		"""
		operator = LinearizedElasticity(
			kinematics=self.kinematics,
			u_test=self.displacement_subsol.dsubtest,
			material_model=material_model,
			material_parameters=material_parameters,
			measure=self.get_subdomain_measure(subdomain_id),
		)
		return self.add_operator(operator)

	def add_hydrostatic_pressure_operator(self, subdomain_id=None):
		r"""Adds the hydrostatic pressure coupling term to the virtual work equation.

		In a mixed :math:`\mathbf{u}-p` formulation, the internal virtual work is
		augmented by a pressure term. This operator represents the work done by
		the hydrostatic pressure :math:`p` against the virtual volumetric strain
		(the divergence of the test displacement :math:`\delta \mathbf{u}`).

		The contribution to the residual is:

		.. math::
		    \delta \Pi_{p} = - \int_{\Omega} p \cdot \text{div}(\delta \mathbf{u}) \, d\Omega

		This term ensures that the pressure :math:`p` acts as the mechanical
		hydrostatic stress component within the momentum equation, effectively
		balancing the external loads while satisfying the incompressibility
		constraint.



		:param subdomain_id: Optional ID to restrict the pressure coupling to
		    a specific subdomain.
		:type subdomain_id: int, optional
		:return: The instantiated hydrostatic pressure operator.
		:rtype: dmech.LinearizedHydrostaticPressureOperator
		"""
		operator = LinearizedHydrostaticPressure(
			kinematics=self.kinematics,
			u_test=self.displacement_subsol.dsubtest,
			p=self.pressure_subsol.subfunc,
			measure=self.get_subdomain_measure(subdomain_id),
		)
		return self.add_operator(operator)

	def add_incompressibility_operator(self, subdomain_id=None):
		r"""Adds the incompressibility constraint equation to the variational formulation.

		In a mixed :math:`\mathbf{u}-p` formulation, this operator enforces the kinematic
		constraint that prevents volume change. For linearized elasticity, this
		corresponds to the weak form of the divergence-free condition:

		.. math::
		    \int_{\Omega} \text{div}(\mathbf{u}) \cdot \delta p \, d\Omega = 0

		where:
		    - :math:`\mathbf{u}` is the displacement field.
		    - :math:`\delta p` is the pressure test function.

		This constraint is essential when the Poisson's ratio :math:`\nu` approaches
		0.5, as standard displacement-based elements suffer from volumetric locking.



		:param subdomain_id: Optional ID to restrict the constraint to a specific
		    subdomain. If None, it is applied to the entire domain.
		:type subdomain_id: int, optional
		:return: The instantiated incompressibility operator.
		:rtype: dmech.LinearizedIncompressibilityOperator
		"""
		operator = LinearizedIncompressibility(
			kinematics=self.kinematics,
			p_test=self.pressure_subsol.dsubtest,
			measure=self.get_subdomain_measure(subdomain_id),
		)
		return self.add_operator(operator)

	def add_elasticity_operators(self, elastic_behaviors):
		r"""Orchestrates the addition of multiple material behaviors and mixed-field constraints.

		This method processes a list of material definitions to build the problem's total
		internal virtual work. For each behavior, it:

		    1. Registers a :class:`LinearizedElasticityOperator`.
		    2. Automatically adds the resulting Cauchy stress :math:`\sigma` to the
		       Fields of Interest (FOI) for visualization.

		If ``w_incompressibility`` was enabled during problem initialization, this
		method also appends the hydrostatic pressure coupling and the volumetric
		constraint (continuity equation) to the variational form.



		:param elastic_behaviors: A list of dictionaries, where each dictionary contains:
		    - ``"model"`` (str): The material model name.
		    - ``"parameters"`` (dict): Parameters for the model (e.g., E, nu).
		    - ``"subdomain_id"`` (int, optional): The ID of the cell subdomain.
		    - ``"suffix"`` (str, optional): A string to append to the stress FOI name.
		:type elastic_behaviors: list[dict]
		"""
		for elastic_behavior in elastic_behaviors:
			operator = self.add_elasticity_operator(
				material_model=elastic_behavior["model"],
				material_parameters=elastic_behavior["parameters"],
				subdomain_id=elastic_behavior.get("subdomain_id", None),
			)
			suffix = "_" + elastic_behavior["suffix"] if "suffix" in elastic_behavior else ""
			self.add_foi(expr=operator.material.sigma, fs=self.mfoi_fs, name="sigma" + suffix)
		if self.w_incompressibility:
			self.add_hydrostatic_pressure_operator()
			self.add_incompressibility_operator()

	def add_global_strain_qois(self):
		r"""Adds integral Quantities of Interest (QOIs) for all independent infinitesimal strain components.

		This method extracts the strain tensor :math:`\mathbf{\epsilon}` from the problem's
		kinematics and defines a scalar QOI for each independent component by integrating
		it over the domain volume :math:`V`.

		The resulting QOIs are named using the pattern ``e_ij`` (e.g., ``e_XX``, ``e_XY``).
		The integrated values correspond to:

		.. math::
		    E_{ij} = \int_{\Omega} \epsilon_{ij} \, d\Omega

		The components added are determined by the spatial dimension of the mesh
		(:math:`d=1, 2, 3`). For example, in 3D, all six independent components
		(three axial, three shear) are registered.
		"""
		basename = "e_"
		strain = self.kinematics.epsilon

		self.add_qoi(name=basename + "XX", expr=strain[0, 0] * self.dV)
		if self.dim >= 2:
			self.add_qoi(name=basename + "YY", expr=strain[1, 1] * self.dV)
			if self.dim >= 3:
				self.add_qoi(name=basename + "ZZ", expr=strain[2, 2] * self.dV)
		if self.dim >= 2:
			self.add_qoi(name=basename + "XY", expr=strain[0, 1] * self.dV)
			if self.dim >= 3:
				self.add_qoi(name=basename + "YZ", expr=strain[1, 2] * self.dV)
				self.add_qoi(name=basename + "ZX", expr=strain[2, 0] * self.dV)

	def add_global_stress_qois(self):
		r"""Adds integral Quantities of Interest (QOIs) for all independent Cauchy stress components.

		This method iterates through all registered operators in the problem. For every
		operator that possesses a ``material`` attribute with a defined ``sigma`` (Cauchy stress),
		it accumulates the volume integral of each tensor component.

		The resulting QOIs are named using the pattern ``s_ij`` (e.g., ``s_XX``, ``s_XY``).
		These values represent the total force-like contribution over the domain:

		.. math::
		    \Sigma_{ij} = \sum_{k} \int_{\Omega_k} \sigma_{ij}^{(k)} \, d\Omega_k

		The number of components added depends on the spatial dimension of the problem
		(:math:`d=1, 2, 3`).
		"""
		basename = "s_"

		self.add_qoi(
			name=basename + "XX",
			expr=sum(
				[
					getattr(operator.material, "sigma")[0, 0] * operator.measure
					for operator in self.operators
					if (hasattr(operator, "material") and hasattr(operator.material, "sigma"))
				]
			),
		)
		if self.dim >= 2:
			self.add_qoi(
				name=basename + "YY",
				expr=sum(
					[
						getattr(operator.material, "sigma")[1, 1] * operator.measure
						for operator in self.operators
						if (hasattr(operator, "material") and hasattr(operator.material, "sigma"))
					]
				),
			)
			if self.dim >= 3:
				self.add_qoi(
					name=basename + "ZZ",
					expr=sum(
						[
							getattr(operator.material, "sigma")[2, 2] * operator.measure
							for operator in self.operators
							if (hasattr(operator, "material") and hasattr(operator.material, "sigma"))
						]
					),
				)
		if self.dim >= 2:
			self.add_qoi(
				name=basename + "XY",
				expr=sum(
					[
						getattr(operator.material, "sigma")[0, 1] * operator.measure
						for operator in self.operators
						if (hasattr(operator, "material") and hasattr(operator.material, "sigma"))
					]
				),
			)
			if self.dim >= 3:
				self.add_qoi(
					name=basename + "YZ",
					expr=sum(
						[
							getattr(operator.material, "sigma")[1, 2] * operator.measure
							for operator in self.operators
							if (hasattr(operator, "material") and hasattr(operator.material, "sigma"))
						]
					),
				)
				self.add_qoi(
					name=basename + "ZX",
					expr=sum(
						[
							getattr(operator.material, "sigma")[2, 0] * operator.measure
							for operator in self.operators
							if (hasattr(operator, "material") and hasattr(operator.material, "sigma"))
						]
					),
				)

	def add_global_pressure_qoi(self):
		r"""Adds a global Quantity of Interest (QOI) for the integrated pressure field.

		This method iterates through all registered operators in the problem and
		accumulates the contribution of their pressure fields :math:`p`. The
		resulting QOI is named "p" and represents the integral of the pressure
		over the relevant domains:

		.. math::
		    P_{global} = \sum_{i} \int_{\Omega_i} p_i \, d\Omega_i

		This is particularly useful for monitoring the average internal pressure
		in incompressible or poromechanical simulations.

		.. note::
		    For purely elastic operators where pressure is not an explicit
		    unknown, the hydrostatic pressure can alternatively be calculated
		    as :math:`p = -\frac{1}{3} \text{tr}(\sigma)`.
		"""
		self.add_qoi(
			name="p", expr=sum([operator.p * operator.measure for operator in self.operators if hasattr(operator, "p")])
		)
		# expr=sum([-dolfin.tr(operator.material.sigma)/3*operator.measure for operator in self.operators if (hasattr(operator, "material") and hasattr(operator.material, "sigma"))])+sum([operator.p*operator.measure for operator in self.operators if hasattr(operator, "p")]))
