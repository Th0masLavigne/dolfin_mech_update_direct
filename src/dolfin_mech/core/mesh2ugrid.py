# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
### This function inspired by Miguel A. Rodriguez                            ###
###  https://fenicsproject.org/qa/12933/                                     ###
###            making-vtk-python-object-from-solution-object-the-same-script ###
###                                                                          ###
################################################################################

"""Utilities for converting FEniCS objects to VTK Unstructured Grids."""

import dolfin
import numpy
import vtk

################################################################################


def mesh2ugrid(mesh, verbose=0):
	"""Converts a FEniCS mesh to a VTK Unstructured Grid (vtkUnstructuredGrid).

	This utility allows for the transition from FEniCS finite element meshes
	to VTK objects for visualization or further processing using VTK/PyVista.
	It supports 2D (triangles) and 3D (tetrahedrons) meshes.

	.. note::
	    The coordinates and connectivity arrays are stored in a global scope
	    temporarily within the function to prevent Python's garbage collector
	    from breaking the VTK pointers.

	:param mesh: The FEniCS mesh to convert.
	:type mesh: dolfin.Mesh
	:param verbose: Verbosity level for debugging prints, defaults to 0.
	:type verbose: int, optional
	:return: A VTK unstructured grid representation of the mesh.
	:rtype: vtk.vtkUnstructuredGrid
	:raises AssertionError: If the mesh dimension is not 2 or 3.
	"""
	if verbose:
		print("mesh2ugrid")

	n_dim = mesh.geometry().dim()
	assert n_dim in (2, 3)
	if verbose:
		print("n_dim = " + str(n_dim))

	n_verts = mesh.num_vertices()
	if verbose:
		print("n_verts = " + str(n_verts))
	n_cells = mesh.num_cells()
	if verbose:
		print("n_cells = " + str(n_cells))

	# Create function space
	fe = dolfin.FiniteElement(family="CG", cell=mesh.ufl_cell(), degree=1, quad_scheme="default")
	fs = dolfin.FunctionSpace(mesh, fe)

	# Store nodes coordinates as numpy array
	n_nodes = fs.dim()
	assert n_nodes == n_verts, "n_nodes (" + str(n_nodes) + ") ≠ n_verts (" + str(n_verts) + "). Aborting."
	if verbose:
		print("n_nodes = " + str(n_nodes))
	global np_coordinates  # MG20190621: if it disappears the vtk objects is broken
	np_coordinates = fs.tabulate_dof_coordinates().reshape([n_nodes, n_dim])
	if verbose:
		print("np_coordinates = " + str(np_coordinates))

	if n_dim == 2:
		np_coordinates = numpy.hstack((np_coordinates, numpy.zeros([n_nodes, 1])))
		if verbose:
			print("np_coordinates = " + str(np_coordinates))

	# Convert nodes coordinates to VTK
	vtk_coordinates = vtk.util.numpy_support.numpy_to_vtk(num_array=np_coordinates, deep=1)
	vtk_points = vtk.vtkPoints()
	vtk_points.SetData(vtk_coordinates)
	if verbose:
		print("n_points = " + str(vtk_points.GetNumberOfPoints()))

	# Store connectivity as numpy array
	if n_dim == 2:
		n_nodes_per_cell = 3
	elif n_dim == 3:
		n_nodes_per_cell = 4
	if verbose:
		print("n_nodes_per_cell = " + str(n_nodes_per_cell))
	global np_connectivity  # MG20190621: if it disappears the vtk objects is broken
	np_connectivity = numpy.empty([n_cells, 1 + n_nodes_per_cell], dtype=int)
	for i in range(n_cells):
		np_connectivity[i, 0] = n_nodes_per_cell
		np_connectivity[i, 1:] = fs.dofmap().cell_dofs(i)
	# if (verbose): print("np_connectivity = "+str(np_connectivity))

	# Add left column specifying number of nodes per cell and flatten array
	np_connectivity = np_connectivity.flatten()
	# if (verbose): print("np_connectivity = "+str(np_connectivity))

	# Convert connectivity to VTK
	vtk_connectivity = vtk.util.numpy_support.numpy_to_vtkIdTypeArray(np_connectivity)

	# Create cell array
	vtk_cells = vtk.vtkCellArray()
	vtk_cells.SetCells(n_cells, vtk_connectivity)
	if verbose:
		print("n_cells = " + str(vtk_cells.GetNumberOfCells()))

	# Create unstructured grid and set points and connectivity
	if n_dim == 2:
		vtk_cell_type = vtk.VTK_TRIANGLE
	elif n_dim == 3:
		vtk_cell_type = vtk.VTK_TETRA
	ugrid = vtk.vtkUnstructuredGrid()
	ugrid.SetPoints(vtk_points)
	ugrid.SetCells(vtk_cell_type, vtk_cells)

	return ugrid


################################################################################


def add_function_to_ugrid(function, ugrid, force_3d_field=1, verbose=0):
	"""Attaches a FEniCS Function's data to an existing VTK unstructured grid.

	The function values are extracted and added as point data to the VTK grid.
	Currently, only Lagrange (CG1) functions are supported as they map
	directly to mesh vertices.

	:param function: The FEniCS function containing the data (e.g., displacement, stress).
	:type function: dolfin.Function
	:param ugrid: The VTK unstructured grid to which data will be added.
	:type ugrid: vtk.vtkUnstructuredGrid
	:param force_3d_field: If 1 and the function is 2D, appends a zero Z-component, defaults to 1.
	:type force_3d_field: int, optional
	:param verbose: Verbosity level, defaults to 0.
	:type verbose: int, optional
	:raises AssertionError: If the function space is not compatible with mesh vertices (non-CG1).
	"""
	if verbose:
		print("add_function_to_ugrid")
	if verbose:
		print(ugrid.GetPoints())

	# Convert function values and add as scalar data
	n_dofs = function.function_space().dim()
	if verbose:
		print("n_dofs = " + str(n_dofs))
	n_dim = function.ufl_element().value_size()
	if verbose:
		print("n_dim = " + str(n_dim))
	assert n_dofs // n_dim == ugrid.GetNumberOfPoints(), "Only CG1 functions can be converted to VTK. Aborting."
	global np_array  # MG20190621: if it disappears the vtk object is broken
	np_array = function.vector().get_local()
	if verbose:
		print("np_array = " + str(np_array))
	np_array = np_array.reshape([n_dofs // n_dim, n_dim])
	if verbose:
		print("np_array = " + str(np_array))
	if (force_3d_field) and (n_dim == 2):
		np_array = numpy.hstack((np_array, numpy.zeros([n_dofs // n_dim, 1])))
		if verbose:
			print("np_array = " + str(np_array))
	vtk_array = vtk.util.numpy_support.numpy_to_vtk(num_array=np_array, deep=1)
	vtk_array.SetName(function.name())

	if verbose:
		print(ugrid.GetPoints())
	ugrid.GetPointData().AddArray(vtk_array)
	if verbose:
		print(ugrid.GetPoints())


################################################################################


def add_functions_to_ugrid(functions, ugrid):
	"""Helper function to add multiple FEniCS functions to a VTK grid.

	Iterates through a list of functions and calls :func:`add_function_to_ugrid`
	for each.

	:param functions: List of FEniCS functions to attach.
	:type functions: list[dolfin.Function]
	:param ugrid: The target VTK unstructured grid.
	:type ugrid: vtk.vtkUnstructuredGrid
	"""
	for function in functions:
		add_function_to_ugrid(function=function, ugrid=ugrid)  # removed the dmech in front
