# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Felipe Álvarez-Barrientos, 2019-2021                                 ###
###                                                                          ###
### Pontificia Universidad Católica, Santiago, Chile                         ###
###                                                                          ###
###                                                                          ###
### And Mahdi Manoochehrtayebi, 2020-2024                                    ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Defines the PinpointSubDomain class.

A utility for isolating specific
coordinates within a mesh to apply point-wise constraints, concentrated
loads, or local tracking sensors.
"""

import dolfin
import numpy

################################################################################


class PinpointSubDomain(dolfin.SubDomain):
	"""SubDomain subclass for identifying a specific point in the mesh.

	This class selects a node (or nodes) that fall within a specified tolerance
	of a target coordinate. It is commonly used for:

	1.  **Point Constraints**: Preventing rigid body translations/rotations by pinning specific nodes.
	2.  **Point Loads**: Applying concentrated forces at specific locations.
	3.  **Sensors**: Identifying nodes for point-wise data extraction.



	:param coords: The target spatial coordinates :math:`(x, y, z)`.
	:type coords: list or numpy.ndarray
	:param tol: The search radius tolerance (default: 1e-3). Nodes within this distance are selected.
	:type tol: float
	"""

	def __init__(self, coords, tol=None):
		"""Initializes the PinpointSubDomain."""
		self.coords = numpy.asarray(coords)
		self.tol = tol if tol is not None else 1e-3

		dolfin.SubDomain.__init__(self)

	def move(self, coords):
		"""Updates the target coordinates.

		This allows the subdomain definition to move over time, which can be useful
		in dynamic simulations or when tracking a moving feature.

		:param coords: The new target coordinates.
		"""
		self.coords[:] = coords

	def inside(self, x, on_boundary):
		"""Checks if a point ``x`` is inside the subdomain.

		Returns True if the Euclidean distance between ``x`` and ``self.coords``
		is less than ``self.tol``.

		:param x: The point to check.
		:param on_boundary: Flag indicating if the point is on the global boundary (unused here).
		:return: bool
		"""
		return numpy.linalg.norm(x - self.coords) < self.tol

	def check_inside(self, mesh):
		"""Debugging utility to find which mesh nodes actually match the criteria.

		Iterates through all mesh coordinates and returns a list of those that
		satisfy the ``inside`` condition. This is useful for verifying that the
		tolerance is sufficient to catch exactly one node (or the intended nodes).

		:param mesh: The dolfin Mesh object to search.
		:return: List of coordinate arrays for matching nodes.
		"""
		x_lst = []
		for x in mesh.coordinates():
			if self.inside(x, True):
				x_lst.append(x)
		return x_lst
