# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Defines the Problem class.

Centralized manager for dolfin_mech simulations that
synchronizes mesh topology, mixed-element function spaces, physical operators,
and time-stepping logic to automate the assembly of nonlinear variational forms.
"""

import dolfin

from ..core import FOI, QOI, Constraint, Step, SubSol
from ..operators import Inertia, loading, penalty

################################################################################


class Problem:
	"""Base class for managing Finite Element Problems in FEniCS.

	This class acts as a container and manager for the essential components of a
	simulation: the mesh, the solution function spaces (unknowns), the operators
	(physics contributions), boundary conditions, and time stepping.

	It abstracts the boilerplate code required to set up mixed-element formulations,
	manage input/output fields (FOIs/QOIs), and assemble the global residual
	and Jacobian forms for the nonlinear solver.


	"""

	def __init__(self):
		"""Initializes an empty Problem instance."""
		self.subsols = []

		self.operators = []
		self.constraints = []

		self.inelastic_behaviors_mixed = []
		self.inelastic_behaviors_internal = []

		self.steps = []

		self.fois = []
		self.qois = []

		self.form_compiler_parameters = {}

	####################################################################### mesh ###

	def set_mesh(
		self,
		mesh,
		define_spatial_coordinates=True,
		define_facet_normals=False,
		compute_bbox=False,
		compute_local_cylindrical_basis=False,
	):
		r"""Sets the computational domain and associated geometric quantities.

		This method registers the mesh and initializes standard FEniCS measures
		(``dx`` for volume, ``ds`` for surface). It can also compute spatial
		coordinates (reference ``X`` or spatial ``x``) and local bases.

		:param mesh: The dolfin Mesh object.
		:param define_spatial_coordinates: If True, defines ``self.X`` (reference)
		    or ``self.x`` (spatial) depending on the problem type (Inverse vs Forward).
		:param define_facet_normals: If True, initializes ``self.mesh_normals``.
		:param compute_bbox: If True, computes the bounding box of the mesh.
		:param compute_local_cylindrical_basis: If True, computes local radial
		    (:math:`\mathbf{e}_R`) and tangential (:math:`\mathbf{e}_T`) vectors
		    and a rotation matrix ``Q_expr``.


		"""
		self.dim = mesh.ufl_domain().geometric_dimension()

		self.mesh = mesh
		self.dV = dolfin.Measure("dx", domain=self.mesh)
		self.mesh_V0 = dolfin.assemble(dolfin.Constant(1) * self.dV)

		if define_spatial_coordinates:
			if "Inverse" in str(self):
				self.x = dolfin.SpatialCoordinate(self.mesh)
			else:
				self.X = dolfin.SpatialCoordinate(self.mesh)

		if define_facet_normals:
			self.mesh_normals = dolfin.FacetNormal(mesh)

		if compute_bbox:
			coord = self.mesh.coordinates()
			self.mesh_bbox = []
			xmin = dolfin.MPI.min(mesh.mpi_comm(), min(coord[:, 0]))
			xmax = dolfin.MPI.max(mesh.mpi_comm(), max(coord[:, 0]))
			self.mesh_bbox += [xmin, xmax]
			if self.dim >= 2:
				ymin = dolfin.MPI.min(mesh.mpi_comm(), min(coord[:, 1]))
				ymax = dolfin.MPI.max(mesh.mpi_comm(), max(coord[:, 1]))
				self.mesh_bbox += [ymin, ymax]
				if self.dim >= 3:
					zmin = dolfin.MPI.min(mesh.mpi_comm(), min(coord[:, 2]))
					zmax = dolfin.MPI.max(mesh.mpi_comm(), max(coord[:, 2]))
					self.mesh_bbox += [zmin, zmax]

		if compute_local_cylindrical_basis:
			self.local_basis_fe = dolfin.VectorElement(
				family="DG",  # MG20220424: Why not CG?
				cell=mesh.ufl_cell(),
				degree=1,
			)

			self.eR_expr = dolfin.Expression(
				("+x[0]/sqrt(pow(x[0],2)+pow(x[1],2))", "+x[1]/sqrt(pow(x[0],2)+pow(x[1],2))"),
				element=self.local_basis_fe,
			)
			self.eT_expr = dolfin.Expression(
				("-x[1]/sqrt(pow(x[0],2)+pow(x[1],2))", "+x[0]/sqrt(pow(x[0],2)+pow(x[1],2))"),
				element=self.local_basis_fe,
			)

			self.Q_expr = dolfin.as_matrix([[self.eR_expr[0], self.eR_expr[1]], [self.eT_expr[0], self.eT_expr[1]]])

			self.local_basis_fs = dolfin.FunctionSpace(
				mesh, self.local_basis_fe
			)  # MG: element keyword don't work here…

			self.eR_func = dolfin.interpolate(v=self.eR_expr, V=self.local_basis_fs)
			self.eR_func.rename("eR", "eR")

			self.eT_func = dolfin.interpolate(v=self.eT_expr, V=self.local_basis_fs)
			self.eT_func.rename("eT", "eT")
		else:
			self.Q_expr = None

	def set_measures(self, domains=None, boundaries=None, points=None):
		"""Defines the integration measures for subdomains and boundaries.

		Registers ``self.dV`` (volume measure), ``self.dS`` (surface measure),
		and ``self.dP`` (point measure), linking them to specific subdomain data
		if provided.
		"""
		self.domains = domains
		self.dV = dolfin.Measure("cell", domain=self.mesh, subdomain_data=self.domains)
		# if (domains is not None):
		#     self.dV = dolfin.Measure(
		#         "dx",
		#         domain=self.mesh,
		#         subdomain_data=self.domains)
		# else:
		#     self.dV = dolfin.Measure(
		#         "dx",
		#         domain=self.mesh)

		self.boundaries = boundaries
		self.dS = dolfin.Measure("exterior_facet", domain=self.mesh, subdomain_data=self.boundaries)
		# if (boundaries is not None):
		#     self.dS = dolfin.Measure(
		#         "ds",
		#         domain=self.mesh,
		#         subdomain_data=self.boundaries)
		# else:
		#     self.dS = dolfin.Measure(
		#         "ds",
		#         domain=self.mesh)

		self.points = points
		self.dP = dolfin.Measure("vertex", domain=self.mesh, subdomain_data=self.points)
		# if (points is not None):
		#     self.dP = dolfin.Measure(
		#         "dP",
		#         domain=self.mesh,
		#         subdomain_data=self.points)
		# else:
		#     self.dP = dolfin.Measure(
		#         "dP",
		#         domain=self.mesh)

	################################################################### solution ###

	def add_subsol(self, name, *args, **kwargs):
		"""Generic method to add a sub-solution (unknown field) to the problem."""
		subsol = SubSol(name=name, *args, **kwargs)
		self.subsols += [subsol]
		return subsol

	def add_scalar_subsol(self, name, family="CG", degree=1, init_val=None, init_fun=None):
		"""Adds a scalar unknown field (e.g., pressure, porosity)."""
		fe = dolfin.FiniteElement(family=family, cell=self.mesh.ufl_cell(), degree=degree)

		subsol = self.add_subsol(name=name, fe=fe, init_val=init_val, init_fun=init_fun)
		return subsol

	def add_vector_subsol(self, name, family="CG", degree=1, init_val=None):
		"""Adds a vector unknown field (e.g., displacement)."""
		fe = dolfin.VectorElement(family=family, cell=self.mesh.ufl_cell(), degree=degree)

		subsol = self.add_subsol(name=name, fe=fe, init_val=init_val)
		return subsol

	def add_tensor_subsol(self, name, family="CG", degree=1, symmetry=None, init_val=None):
		"""Adds a tensor unknown field (e.g., internal variables)."""
		fe = dolfin.TensorElement(family=family, cell=self.mesh.ufl_cell(), degree=degree, symmetry=symmetry)

		subsol = self.add_subsol(name=name, fe=fe, init_val=init_val)
		return subsol

	def set_solution_finite_element(self):
		"""Constructs the (potentially mixed) finite element for the global solution.

		If multiple sub-solutions are defined (e.g., displacement and pressure),
		this creates a ``MixedElement``.


		"""
		if len(self.subsols) == 1:
			self.sol_fe = self.subsols[0].fe
		else:
			self.sol_fe = dolfin.MixedElement([subsol.fe for subsol in self.subsols])
		# print(self.sol_fe)

	def set_solution_function_space(self, constrained_domain=None):
		"""Creates the FunctionSpace on the mesh using the defined finite element."""
		self.sol_fs = dolfin.FunctionSpace(
			self.mesh, self.sol_fe, constrained_domain=constrained_domain
		)  # MG: element keyword don't work here…

		if len(self.subsols) == 1:
			self.subsols[0].fs = self.sol_fs
		else:
			for k_subsol, subsol in enumerate(self.subsols):
				subsol.fs = self.sol_fs.sub(k_subsol)

	def set_solution_functions(self):
		r"""Instantiates the actual FEniCS Functions for the solution.

		Creates:
		    - ``sol_func``: The current solution vector :math:`\mathbf{u}_{n+1}`.
		    - ``sol_old_func``: The solution at the previous step :math:`\mathbf{u}_n`.
		    - ``dsol_func``: The increment (Newton update) :math:`\delta \mathbf{u}`.
		    - ``dsol_test``: The test function :math:`\delta \mathbf{v}`.
		    - ``dsol_tria``: The trial function :math:`\Delta \mathbf{u}` (for the Jacobian).

		It also splits these functions back into their sub-solution components
		for easy access by operators.
		"""
		self.sol_func = dolfin.Function(self.sol_fs)
		self.sol_old_func = dolfin.Function(self.sol_fs)
		self.dsol_func = dolfin.Function(self.sol_fs)
		self.dsol_test = dolfin.TestFunction(self.sol_fs)
		self.dsol_tria = dolfin.TrialFunction(self.sol_fs)

		if len(self.subsols) == 1:
			subfuncs = (self.sol_func,)
			dsubtests = (self.dsol_test,)
			dsubtrias = (self.dsol_tria,)
			funcs = (self.sol_func,)
			funcs_old = (self.sol_old_func,)
			dfuncs = (self.dsol_func,)
		else:
			subfuncs = dolfin.split(self.sol_func)
			dsubtests = dolfin.split(self.dsol_test)
			dsubtrias = dolfin.split(self.dsol_tria)
			funcs = dolfin.Function(self.sol_fs).split(deepcopy=1)
			funcs_old = dolfin.Function(self.sol_fs).split(deepcopy=1)
			dfuncs = dolfin.Function(self.sol_fs).split(deepcopy=1)

		for k_subsol, subsol in enumerate(self.subsols):
			subsol.subfunc = subfuncs[k_subsol]
			subsol.dsubtest = dsubtests[k_subsol]
			subsol.dsubtria = dsubtrias[k_subsol]

			subsol.func = funcs[k_subsol]
			subsol.func.rename(subsol.name, subsol.name)
			subsol.func_old = funcs_old[k_subsol]
			subsol.func_old.rename(subsol.name + "_old", subsol.name + "_old")
			subsol.dfunc = dfuncs[k_subsol]
			subsol.dfunc.rename("d" + subsol.name, "d" + subsol.name)

		for k_subsol, subsol in enumerate(self.subsols):
			subsol.init()
		if len(self.subsols) > 1:
			dolfin.assign(self.sol_func, self.get_subsols_func_lst())
			dolfin.assign(self.sol_old_func, self.get_subsols_func_old_lst())

	def get_subsols_func_lst(self):

		return [subsol.func for subsol in self.subsols]

	def get_subsols_func_old_lst(self):

		return [subsol.func_old for subsol in self.subsols]

	def get_subsols_dfunc_lst(self):

		return [subsol.dfunc for subsol in self.subsols]

	def set_quadrature_degree(self, quadrature_degree):

		self.form_compiler_parameters["quadrature_degree"] = quadrature_degree

	######################################################################## FOI ###

	def set_foi_finite_elements_DG(
		self, degree=0
	):  # MG20180420: DG elements are simpler to manage than quadrature elements, since quadrature elements must be compatible with the expression's degree, which is not always trivial (e.g., for J…)
		"""Sets up Discontinuous Galerkin (DG) elements for Fields of Interest (FOI).
		These are typically used for post-processing variables like stress or strain.
		"""
		self.sfoi_fe = dolfin.FiniteElement(family="DG", cell=self.mesh.ufl_cell(), degree=degree)

		self.vfoi_fe = dolfin.VectorElement(family="DG", cell=self.mesh.ufl_cell(), degree=degree)

		self.mfoi_fe = dolfin.TensorElement(family="DG", cell=self.mesh.ufl_cell(), degree=degree)

	def set_foi_finite_elements_Quad(
		self, degree=0
	):  # MG20180420: DG elements are simpler to manage than quadrature elements, since quadrature elements must be compatible with the expression's degree, which is not always trivial (e.g., for J…)
		"""Sets up Quadrature elements for Fields of Interest (FOI).
		These allow evaluating fields exactly at integration points.
		"""
		self.sfoi_fe = dolfin.FiniteElement(
			family="Quadrature", cell=self.mesh.ufl_cell(), degree=degree, quad_scheme="default"
		)
		self.sfoi_fe._quad_scheme = "default"  # MG20180406: is that even needed?
		for sub_element in self.sfoi_fe.sub_elements():  # MG20180406: is that even needed?
			sub_element._quad_scheme = "default"  # MG20180406: is that even needed?

		self.vfoi_fe = dolfin.VectorElement(
			family="Quadrature", cell=self.mesh.ufl_cell(), degree=degree, quad_scheme="default"
		)
		self.vfoi_fe._quad_scheme = "default"  # MG20180406: is that even needed?
		for sub_element in self.vfoi_fe.sub_elements():  # MG20180406: is that even needed?
			sub_element._quad_scheme = "default"  # MG20180406: is that even needed?

		self.mfoi_fe = dolfin.TensorElement(
			family="Quadrature", cell=self.mesh.ufl_cell(), degree=degree, quad_scheme="default"
		)
		self.mfoi_fe._quad_scheme = "default"  # MG20180406: is that still needed?
		for sub_element in self.mfoi_fe.sub_elements():  # MG20180406: is that still needed?
			sub_element._quad_scheme = "default"  # MG20180406: is that still needed?

	def set_foi_function_spaces(self):
		"""Creates function spaces for Scalar, Vector, and Tensor Fields of Interest."""
		self.sfoi_fs = dolfin.FunctionSpace(self.mesh, self.sfoi_fe)  # MG: element keyword don't work here…

		self.vfoi_fs = dolfin.FunctionSpace(self.mesh, self.vfoi_fe)  # MG: element keyword don't work here…

		self.mfoi_fs = dolfin.FunctionSpace(self.mesh, self.mfoi_fe)  # MG: element keyword don't work here…

	def add_foi(self, *args, **kwargs):
		"""Adds a Field of Interest (FOI) to the problem.
		FOIs are updated at the end of every step for visualization/output.
		"""
		foi = FOI(*args, form_compiler_parameters=self.form_compiler_parameters, **kwargs)
		self.fois += [foi]
		return foi

	def get_foi(self, name):

		for foi in self.fois:
			if foi.name == name:
				return foi
		assert 0, 'No FOI named "' + name + '". Aborting.'

	def update_fois(self):
		"""Triggers the projection/interpolation of all registered FOIs."""
		for foi in self.fois:
			foi.update()

	def get_fois_func_lst(self):

		return [foi.func for foi in self.fois]

	######################################################################## QOI ###

	def add_qoi(self, *args, **kwargs):
		"""Adds a Quantity of Interest (QOI) to the problem.
		QOIs are scalar values (e.g., total volume, average stress) calculated per step.
		"""
		qoi = QOI(*args, form_compiler_parameters=self.form_compiler_parameters, **kwargs)
		self.qois += [qoi]
		return qoi

	def update_qois(self, dt=None, k_step=None):
		"""Updates the values of all registered QOIs."""
		for qoi in self.qois:
			qoi.update(dt, k_step)

	################################################################## operators ###

	def add_operator(self, operator, k_step=None):
		"""Adds a physical operator to the problem.

		Operators define the terms in the variational formulation (e.g., internal
		energy, external work). If ``k_step`` is provided, the operator is only
		active for that specific load step.
		"""
		if k_step is None:
			self.operators += [operator]
		else:
			self.steps[k_step].operators += [operator]
		return operator

	################################################################## operators ###

	# MG20230131: Loading operators should not be there,
	# but they are shared between Elasticity & HyperElasticity problems,
	# so it is more convenient for the moment.

	def add_volume_force0_loading_operator(self, k_step=None, **kwargs):
		"""Adds a dead volume force (reference configuration)."""
		operator = loading.VolumeForce0(U_test=self.displacement_subsol.dsubtest, **kwargs)
		return self.add_operator(operator=operator, k_step=k_step)

	def add_volume_force_loading_operator(self, k_step=None, **kwargs):
		"""Adds a follower volume force (current configuration)."""
		operator = loading.VolumeForce(U_test=self.displacement_subsol.dsubtest, kinematics=self.kinematics, **kwargs)
		return self.add_operator(operator=operator, k_step=k_step)

	def add_surface_force0_loading_operator(self, k_step=None, **kwargs):
		"""Adds a dead surface force vector (reference configuration)."""
		operator = loading.SurfaceForce0(U_test=self.displacement_subsol.dsubtest, **kwargs)
		return self.add_operator(operator=operator, k_step=k_step)

	def add_surface_force_loading_operator(self, k_step=None, **kwargs):
		"""Adds a follower surface force vector (current configuration)."""
		operator = loading.SurfaceForce(
			U_test=self.displacement_subsol.dsubtest, kinematics=self.kinematics, N=self.mesh_normals, **kwargs
		)
		return self.add_operator(operator=operator, k_step=k_step)

	def add_surface_pressure0_loading_operator(self, k_step=None, **kwargs):
		"""Adds a dead pressure load normal to the reference surface."""
		operator = loading.SurfacePressure0(U_test=self.displacement_subsol.dsubtest, N=self.mesh_normals, **kwargs)
		return self.add_operator(operator=operator, k_step=k_step)

	def add_surface_pressure_loading_operator(self, k_step=None, **kwargs):
		"""Adds a follower pressure load (normal to deformed surface)."""
		operator = loading.SurfacePressure(
			U_test=self.displacement_subsol.dsubtest, kinematics=self.kinematics, N=self.mesh_normals, **kwargs
		)
		return self.add_operator(operator=operator, k_step=k_step)

	def add_surface_pressure_gradient0_loading_operator(self, k_step=None, **kwargs):
		"""Adds a gradient-dependent pressure load on the reference surface."""
		operator = loading.SurfacePressureGradient0(
			x=dolfin.SpatialCoordinate(self.mesh),
			U_test=self.displacement_subsol.dsubtest,
			N=self.mesh_normals,
			**kwargs,
		)
		return self.add_operator(operator=operator, k_step=k_step)

	def add_surface_pressure_gradient_loading_operator(self, k_step=None, **kwargs):
		"""Adds a gradient-dependent pressure load on the deformed surface."""
		operator = loading.SurfacePressureGradient(
			X=dolfin.SpatialCoordinate(self.mesh),
			U=self.displacement_subsol.subfunc,
			U_test=self.displacement_subsol.dsubtest,
			kinematics=self.kinematics,
			N=self.mesh_normals,
			**kwargs,
		)
		return self.add_operator(operator=operator, k_step=k_step)

	def add_surface_tension0_loading_operator(self, k_step=None, **kwargs):
		"""Adds surface tension effects on the reference configuration."""
		operator = loading.SurfaceTension0(
			u=self.displacement_subsol.subfunc,
			u_test=self.displacement_subsol.dsubtest,
			kinematics=self.kinematics,
			N=self.mesh_normals,
			**kwargs,
		)
		return self.add_operator(operator=operator, k_step=k_step)

	def add_surface_tension_loading_operator(self, k_step=None, **kwargs):
		"""Adds surface tension effects on the current configuration."""
		operator = loading.SurfaceTension(
			# U=self.displacement_subsol.subfunc,
			U_test=self.displacement_subsol.dsubtest,
			kinematics=self.kinematics,
			N=self.mesh_normals,
			**kwargs,
		)
		return self.add_operator(operator=operator, k_step=k_step)

	def add_normal_displacement_penalty_operator(self, k_step=None, **kwargs):
		"""Adds a penalty operator to constrain displacement in the normal direction."""
		operator = penalty.NormalDisplacement(
			U=self.displacement_subsol.subfunc, U_test=self.displacement_subsol.dsubtest, N=self.mesh_normals, **kwargs
		)
		return self.add_operator(operator=operator, k_step=k_step)

	def add_directional_displacement_penalty_operator(self, k_step=None, **kwargs):
		"""Adds a penalty operator to constrain displacement in a specific global direction."""
		operator = penalty.DirectionalDisplacement(
			U=self.displacement_subsol.subfunc, U_test=self.displacement_subsol.dsubtest, **kwargs
		)
		return self.add_operator(operator=operator, k_step=k_step)

	def add_inertia_operator(self, k_step=None, **kwargs):
		"""Adds inertial forces for dynamic simulations."""
		operator = Inertia(
			U=self.displacement_subsol.subfunc,
			U_old=self.displacement_subsol.func_old,
			U_test=self.displacement_subsol.dsubtest,
			**kwargs,
		)
		return self.add_operator(operator=operator, k_step=k_step)

	################################################################ constraints ###

	def add_constraint(self, *args, k_step=None, **kwargs):
		"""Adds a Dirichlet boundary condition (constraint) to the problem."""
		constraint = Constraint(*args, **kwargs)
		if k_step is None:
			self.constraints += [constraint]
		else:
			self.steps[k_step].constraints += [constraint]
		return constraint

	###################################################################### steps ###

	def add_step(self, Deltat=1.0, **kwargs):
		"""Defines a new time step/load step for the simulation."""
		if len(self.steps) == 0:
			t_ini = 0.0
			t_fin = Deltat
		else:
			t_ini = self.steps[-1].t_fin
			t_fin = t_ini + Deltat
		step = Step(t_ini=t_ini, t_fin=t_fin, **kwargs)
		self.steps += [step]
		return len(self.steps) - 1

	###################################################################### forms ###

	def set_variational_formulation(self, k_step=None):
		r"""Assembles the global variational forms for the nonlinear solver.

		This aggregates the residual forms from all active operators and computes 
		the tangent Jacobian matrix using automatic differentiation.

		.. math::
		    \mathbf{R}(\mathbf{u}) = \sum \mathbf{R}_{op}(\mathbf{u}) \
		    \mathbf{J} = \frac{\partial \mathbf{R}}{\partial \mathbf{u}}



		:param k_step: Optional step index to include step-specific operators.
		"""
		self.res_form = sum(
			[operator.res_form for operator in self.operators if (operator.measure.integral_type() != "vertex")]
		)  # MG20190513: Cannot use point integral within assemble_system
		if k_step is not None:
			self.res_form += sum(
				[
					operator.res_form
					for operator in self.steps[k_step].operators
					if (operator.measure.integral_type() != "vertex")
				]
			)  # MG20190513: Cannot use point integral within assemble_system

		# print(self.res_form)
		# for operator in self.operators:
		#     if (operator.measure.integral_type() != "vertex"):
		#         print(type(operator))
		#         print(operator.res_form)

		self.jac_form = dolfin.derivative(self.res_form, self.sol_func, self.dsol_tria)

		# print(self.jac_form)
