# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Utility for exporting FEniCS functions to VTU format."""

import os
import shutil

import dolfin
import myVTKPythonLibrary as myvtk

from .mesh2ugrid import add_function_to_ugrid, mesh2ugrid

################################################################################


def write_VTU_file(filebasename, function, time=None, zfill=3, preserve_connectivity=False):
	"""Exports a FEniCS Function to a VTU file for visualization in ParaView.

	This function saves the state of a finite element field at a specific time step.
	It is typically called inside the time integration loop.



	**Export Modes:**

	1.  **Standard (`preserve_connectivity=False`)**: Uses FEniCS's built-in
	    `dolfin.File` mechanism. This is robust but may interpolate high-order
	    functions to a linear (P1) visualization mesh, potentially altering the
	    visual connectivity or smoothing results. The function performs file
	    renaming operations to ensure a clean naming convention (``name_00X.vtu``).

	2.  **Preserved Connectivity (`preserve_connectivity=True`)**: Uses a custom
	    conversion routine (`mesh2ugrid`) to map the FEniCS mesh directly to a
	    VTK Unstructured Grid. This is slower but ensures that the topology (nodes
	    and cells) in the output file exactly matches the computational mesh,
	    which is critical for visualizing Discontinuous Galerkin (DG) fields
	    or high-order elements correctly.

	:param filebasename: The prefix for the output file path (e.g., "results/u").
	:type filebasename: str
	:param function: The FEniCS Function object to export.
	:type function: dolfin.Function
	:param time: The current time step index or value (used for file suffixing).
	:type time: int or float
	:param zfill: Number of digits for zero-padding the step number (e.g., 3 -> "_001").
	:type zfill: int
	:param preserve_connectivity: If True, uses the custom writer to maintain exact mesh topology.
	:type preserve_connectivity: bool
	:return: None
	"""
	if preserve_connectivity:
		ugrid = mesh2ugrid(function.function_space().mesh())
		add_function_to_ugrid(function=function, ugrid=ugrid)
		myvtk.writeUGrid(
			ugrid=ugrid, filename=filebasename + ("_" + str(time).zfill(zfill) if (time is not None) else "") + ".vtu"
		)

	else:
		file_pvd = dolfin.File(filebasename + "__.pvd")
		file_pvd << (function, float(time) if (time is not None) else 0.0)
		os.remove(filebasename + "__.pvd")
		shutil.move(
			filebasename + "__" + "".zfill(6) + ".vtu",
			filebasename + ("_" + str(time).zfill(zfill) if (time is not None) else "") + ".vtu",
		)
