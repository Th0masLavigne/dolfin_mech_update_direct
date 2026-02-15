# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Defines the RivlinCube_Hyperelasticity driver.

High-level simulation suite for
finite-strain nonlinear mechanics. It provides a modular interface to solve
forward and inverse hyperelastic problems on 2D or 3D domains, supporting multi-
material heterogeneity, incompressible mixed formulations, and complex loading
paths including follower pressure and surface tension.
"""

import dolfin

from .. import core, problems
from .rivlincube_mesh import RivlinCube_Mesh

################################################################################


def RivlinCube_Hyperelasticity(
	dim: int = 3,
	inverse: bool = 0,
	incomp: bool = 0,
	multimaterial: bool = 0,
	cube_params: dict = {},
	mat_params: dict = {},
	step_params: dict = {},
	const_params: dict = {},
	load_params: dict = {},
	move_params: dict = {},
	get_results: bool = 0,
	res_basename: str = "run_RivlinCube_Hyperelasticity",
	write_vtus_with_preserved_connectivity: bool = False,
	verbose: bool = 0,
):
	"""Runs the Rivlin Cube benchmark for Hyperelasticity (Forward or Inverse).

	This function simulates a unit cube (or square) under large deformations.
	It is the primary testbed for validating the solver's capabilities across
	different physics and kinematic assumptions.



	**Key Capabilities:**

	1.  **Inverse/Forward**: Can solve the standard forward problem (given load, find deformation)
	    or the inverse problem (given deformed shape, find reference configuration) by setting ``inverse=True``.
	2.  **Incompressibility**: Supports mixed finite element formulations (:math:`u-p`) for
	    incompressible materials via ``incomp=True``.
	3.  **Multimaterial**: Can split the domain into two distinct material regions
	    (e.g., stiff vs. soft) to test interface continuity and heterogeneous stress states.
	4.  **Mesh Morphing**: Supports Arbitrary Lagrangian-Eulerian (ALE) mesh movement
	    pre-solve via ``move_params``.

	**Loading Modes (`load_params["type"]`):**

	- ``"disp"``: Prescribed displacement (uniaxial/biaxial stretch).
	- ``"volu"`` / ``"volu0"``: Body force in current / reference configuration.
	- ``"surf"`` / ``"surf0"``: Surface traction vector in current / reference configuration.
	- ``"pres"`` / ``"pres0"``: Hydrostatic pressure (follower load / dead load).
	- ``"pgra"`` / ``"pgra0"``: Pressure gradient load (e.g., for hydrostatic fluid tanks).
	- ``"tens"``: Surface tension effects.

	:param dim: Dimension (2 or 3).
	:param inverse: If True, uses :class:`InverseHyperelasticityProblem`.
	:param incomp: If True, uses mixed formulation.
	:param multimaterial: If True, defines two material subdomains.
	:param cube_params: Mesh generation parameters.
	:param mat_params: Material constitutive parameters.
	:param step_params: Time-stepping configuration.
	:param const_params: Boundary constraints (e.g., symmetry planes).
	:param load_params: Loading configuration.
	:param move_params: Parameters for pre-simulation mesh movement (ALE).
	:param get_results: If True, returns the displacement function and measure at the end.
	:param res_basename: Output filename prefix.
	:return: (Optional) Tuple ``(displacement_function, measure)`` if ``get_results`` is True.
	"""
	################################################################### Mesh ###

	if dim == 2:
		mesh, boundaries_mf, xmin_id, xmax_id, ymin_id, ymax_id = RivlinCube_Mesh(dim=dim, params=cube_params)
	elif dim == 3:
		mesh, boundaries_mf, xmin_id, xmax_id, ymin_id, ymax_id, zmin_id, zmax_id = RivlinCube_Mesh(
			dim=dim, params=cube_params
		)

	if move_params.get("move", False):
		U = move_params.get("U")
		dolfin.ALE.move(mesh, U)

	if multimaterial:
		mat1_sd = dolfin.CompiledSubDomain("x[0] <= x0", x0=0.5)
		mat2_sd = dolfin.CompiledSubDomain("x[0] >= x0", x0=0.5)

		mat1_id = 1
		mat2_id = 2

		domains_mf = dolfin.MeshFunction(
			"size_t", mesh, mesh.topology().dim()
		)  # MG20180418: size_t looks like unsigned int, but more robust wrt architecture and os
		domains_mf.set_all(0)
		mat1_sd.mark(domains_mf, mat1_id)
		mat2_sd.mark(domains_mf, mat2_id)
	else:
		domains_mf = None

	################################################################ Problem ###

	if inverse:
		problem_type = problems.InverseHyperelasticity
	else:
		problem_type = problems.Hyperelasticity

	if incomp:
		displacement_degree = 2  # MG20211219: Incompressibility requires displacement_degree >= 2 ?!
		w_incompressibility = 1
	else:
		displacement_degree = 1
		w_incompressibility = 0

	quadrature_degree = "default"
	# quadrature_degree = "full"

	if multimaterial:
		elastic_behavior = None
		if incomp:
			mat1_mod = "NHMR"
			mat2_mod = "NHMR"
		else:
			mat1_mod = "CGNHMR"
			mat2_mod = "CGNHMR"
		mat1_params = {"E": 1.0, "nu": 0.5 * (incomp) + 0.3 * (1 - incomp)}

		mat2_params = {"E": 10.0, "nu": 0.5 * (incomp) + 0.3 * (1 - incomp)}
		elastic_behaviors = [
			{"subdomain_id": mat1_id, "model": mat1_mod, "parameters": mat1_params, "suffix": "1"},
			{"subdomain_id": mat2_id, "model": mat2_mod, "parameters": mat2_params, "suffix": "2"},
		]
	else:
		elastic_behavior = mat_params
		elastic_behaviors = None

	problem = problem_type(
		mesh=mesh,
		domains_mf=domains_mf,
		define_facet_normals=1,
		boundaries_mf=boundaries_mf,
		displacement_degree=displacement_degree,
		quadrature_degree=quadrature_degree,
		w_incompressibility=w_incompressibility,
		elastic_behavior=elastic_behavior,
		elastic_behaviors=elastic_behaviors,
	)

	########################################## Boundary conditions & Loading ###

	const_type = const_params.get("type", "sym")

	if const_type in ("symx", "sym"):
		problem.add_constraint(
			V=problem.displacement_subsol.fs.sub(0), sub_domains=boundaries_mf, sub_domain_id=xmin_id, val=0.0
		)
	if (const_type in ("symy", "sym")) and (dim >= 2):
		problem.add_constraint(
			V=problem.displacement_subsol.fs.sub(1), sub_domains=boundaries_mf, sub_domain_id=ymin_id, val=0.0
		)
	if (const_type in ("symz", "sym")) and (dim >= 3):
		problem.add_constraint(
			V=problem.displacement_subsol.fs.sub(2), sub_domains=boundaries_mf, sub_domain_id=zmin_id, val=0.0
		)
	if const_type in ("blox"):
		problem.add_constraint(
			V=problem.displacement_subsol.fs, sub_domains=boundaries_mf, sub_domain_id=xmin_id, val=[0.0] * dim
		)
	if const_type in ("bloy"):
		problem.add_constraint(
			V=problem.displacement_subsol.fs, sub_domains=boundaries_mf, sub_domain_id=ymin_id, val=[0.0] * dim
		)
	if const_type in ("bloz"):
		problem.add_constraint(
			V=problem.displacement_subsol.fs, sub_domains=boundaries_mf, sub_domain_id=zmin_id, val=[0.0] * dim
		)

	n_steps = step_params.get("n_steps", 1)
	Deltat_lst = step_params.get("Deltat_lst", [step_params.get("Deltat", 1.0) / n_steps] * n_steps)
	dt_ini_lst = step_params.get("dt_ini_lst", [step_params.get("dt_ini", 1.0) / n_steps] * n_steps)
	dt_min_lst = step_params.get("dt_min_lst", [step_params.get("dt_min", 1.0) / n_steps] * n_steps)

	load_type = load_params.get("type", "disp")

	if load_type == "disp":
		u_lst = load_params.get(
			"u_lst", [(k_step + 1) * load_params.get("u", +0.5) / n_steps for k_step in range(n_steps)]
		)
	elif load_type in ("volu0", "volu"):
		f_lst = load_params.get(
			"f_lst", [(k_step + 1) * load_params.get("f", +0.5) / n_steps for k_step in range(n_steps)]
		)
	elif load_type in ("surf0", "surf"):
		f_lst = load_params.get(
			"f_lst", [(k_step + 1) * load_params.get("f", +1.0) / n_steps for k_step in range(n_steps)]
		)
	elif load_type in ("pres0", "pres0_multi", "pres0_inertia", "pres"):
		p_lst = load_params.get(
			"f_lst", [(k_step + 1) * load_params.get("f", -0.5) / n_steps for k_step in range(n_steps)]
		)
	elif load_type in ("pgra0", "pgra"):
		X0_lst = load_params.get("X0_lst", [load_params.get("X0", [0.5] * dim)] * n_steps)
		N0_lst = load_params.get("N0_lst", [load_params.get("X0", [1.0] + [0.0] * (dim - 1))] * n_steps)
		P0_lst = load_params.get(
			"P0_lst", [(k_step + 1) * load_params.get("P0", -0.50) / n_steps for k_step in range(n_steps)]
		)
		DP_lst = load_params.get(
			"DP_lst", [(k_step + 1) * load_params.get("P0", -0.25) / n_steps for k_step in range(n_steps)]
		)
	elif load_type == "tens":
		gamma_lst = load_params.get(
			"gamma_lst", [(k_step + 1) * load_params.get("gamma", 0.01) / n_steps for k_step in range(n_steps)]
		)

	for k_step in range(n_steps):
		Deltat = Deltat_lst[k_step]
		dt_ini = dt_ini_lst[k_step]
		dt_min = dt_min_lst[k_step]

		k_step = problem.add_step(Deltat=Deltat, dt_ini=dt_ini, dt_min=dt_min)

		if load_type == "disp":
			u = u_lst[k_step]
			u_old = u_lst[k_step - 1] if (k_step > 0) else 0.0
			problem.add_constraint(
				V=problem.displacement_subsol.fs.sub(0),
				sub_domains=boundaries_mf,
				sub_domain_id=xmax_id,
				val_ini=u_old,
				val_fin=u,
				k_step=k_step,
			)
		elif load_type == "volu0":
			f = f_lst[k_step]
			f_old = f_lst[k_step - 1] if (k_step > 0) else 0.0
			problem.add_volume_force0_loading_operator(
				measure=problem.dV, F_ini=[f_old] + [0.0] * (dim - 1), F_fin=[f] + [0.0] * (dim - 1), k_step=k_step
			)
		elif load_type == "volu":
			f = f_lst[k_step]
			f_old = f_lst[k_step - 1] if (k_step > 0) else 0.0
			problem.add_volume_force_loading_operator(
				measure=problem.dV, F_ini=[f_old] + [0.0] * (dim - 1), F_fin=[f] + [0.0] * (dim - 1), k_step=k_step
			)
		elif load_type == "surf0":
			f = f_lst[k_step]
			f_old = f_lst[k_step - 1] if (k_step > 0) else 0.0
			problem.add_surface_force0_loading_operator(
				measure=problem.dS(xmax_id),
				F_ini=[f_old] + [0.0] * (dim - 1),
				F_fin=[f] + [0.0] * (dim - 1),
				k_step=k_step,
			)
		elif load_type == "surf":
			f = f_lst[k_step]
			f_old = f_lst[k_step - 1] if (k_step > 0) else 0.0
			problem.add_surface_force_loading_operator(
				measure=problem.dS(xmax_id),
				F_ini=[f_old] + [0.0] * (dim - 1),
				F_fin=[f] + [0.0] * (dim - 1),
				k_step=k_step,
			)
		elif load_type == "pres0":
			p = p_lst[k_step]
			p_old = p_lst[k_step - 1] if (k_step > 0) else 0.0
			problem.add_surface_pressure0_loading_operator(
				measure=problem.dS(xmax_id), P_ini=p_old, P_fin=p, k_step=k_step
			)
		elif load_type == "pres0_multi":
			p = p_lst[k_step]
			p_old = p_lst[k_step - 1] if (k_step > 0) else 0.0
			problem.add_surface_pressure0_loading_operator(
				measure=problem.dS(xmax_id), P_ini=p_old, P_fin=p, k_step=k_step
			)
			problem.add_surface_pressure0_loading_operator(
				measure=problem.dS(ymax_id), P_ini=p_old, P_fin=p, k_step=k_step
			)
			if dim == 3:
				problem.add_surface_pressure0_loading_operator(
					measure=problem.dS(zmax_id), P_ini=p_old, P_fin=p, k_step=k_step
				)
		elif load_type == "pres0_inertia":
			p = p_lst[k_step]
			p_old = p_lst[k_step - 1] if (k_step > 0) else 0.0
			problem.add_inertia_operator(measure=problem.dV, rho_val=1e-2, k_step=k_step)
			problem.add_surface_pressure0_loading_operator(
				measure=problem.dS(xmin_id), P_ini=p_old, P_fin=p, k_step=k_step
			)
			problem.add_surface_pressure0_loading_operator(
				measure=problem.dS(xmax_id), P_ini=p_old, P_fin=p, k_step=k_step
			)
			problem.add_surface_pressure0_loading_operator(
				measure=problem.dS(ymin_id), P_ini=p_old, P_fin=p, k_step=k_step
			)
			problem.add_surface_pressure0_loading_operator(
				measure=problem.dS(ymax_id), P_ini=p_old, P_fin=p, k_step=k_step
			)
			if dim == 3:
				problem.add_surface_pressure0_loading_operator(
					measure=problem.dS(zmin_id), P_ini=p_old, P_fin=p, k_step=k_step
				)
			if dim == 3:
				problem.add_surface_pressure0_loading_operator(
					measure=problem.dS(zmax_id), P_ini=p_old, P_fin=p, k_step=k_step
				)
		elif load_type == "pres":
			p = p_lst[k_step]
			p_old = p_lst[k_step - 1] if (k_step > 0) else 0.0
			problem.add_surface_pressure_loading_operator(
				measure=problem.dS(xmax_id), P_ini=p_old, P_fin=p, k_step=k_step
			)
		elif load_type == "pgra0":
			X0 = X0_lst[k_step]
			N0 = N0_lst[k_step]
			P0 = P0_lst[k_step]
			DP = DP_lst[k_step]
			P0_old = P0_lst[k_step - 1] if (k_step > 0) else 0.0
			DP_old = DP_lst[k_step - 1] if (k_step > 0) else 0.0
			problem.add_surface_pressure_gradient0_loading_operator(
				measure=problem.dS(),
				X0_val=X0,
				N0_val=N0,
				P0_ini=P0_old,
				P0_fin=P0,
				DP_ini=DP_old,
				DP_fin=DP,
				k_step=k_step,
			)
		elif load_type == "pgra":
			X0 = X0_lst[k_step]
			N0 = N0_lst[k_step]
			P0 = P0_lst[k_step]
			DP = DP_lst[k_step]
			P0_old = P0_lst[k_step - 1] if (k_step > 0) else 0.0
			DP_old = DP_lst[k_step - 1] if (k_step > 0) else 0.0
			problem.add_surface_pressure_gradient_loading_operator(
				measure=problem.dS(),
				X0_val=X0,
				N0_val=N0,
				P0_ini=P0_old,
				P0_fin=P0,
				DP_ini=DP_old,
				DP_fin=DP,
				k_step=k_step,
			)
		elif load_type == "tens":
			gamma = gamma_lst[k_step]
			gamma_old = gamma_lst[k_step - 1] if (k_step > 0) else 0.0
			problem.add_surface_tension_loading_operator(
				measure=problem.dS, gamma_ini=gamma_old, gamma_fin=gamma, k_step=k_step
			)

	################################################# Quantities of Interest ###

	problem.add_global_strain_qois()
	problem.add_global_stress_qois()
	if incomp:
		problem.add_global_pressure_qoi()
	if (inverse == 0) and (dim == 2):
		problem.add_global_out_of_plane_stress_qois()

	################################################################# Solver ###

	solver = core.NonlinearSolver(
		problem=problem,
		parameters={"sol_tol": [1e-6] * len(problem.subsols), "n_iter_max": 32},
		relax_type="constant",
		write_iter=0,
	)

	integrator = core.TimeIntegrator(
		problem=problem,
		solver=solver,
		parameters={"n_iter_for_accel": 4, "n_iter_for_decel": 16, "accel_coeff": 2, "decel_coeff": 2},
		print_out=res_basename * verbose,
		print_sta=res_basename * verbose,
		write_qois=res_basename + "-qois",
		write_qois_limited_precision=1,
		write_sol=res_basename * verbose,
		write_vtus=res_basename * verbose,
		write_vtus_with_preserved_connectivity=write_vtus_with_preserved_connectivity,
	)

	success = integrator.integrate()
	assert success, "Integration failed. Aborting."

	integrator.close()

	if get_results:
		return (problem.displacement_subsol.func, dolfin.Measure("dx", domain=mesh))
