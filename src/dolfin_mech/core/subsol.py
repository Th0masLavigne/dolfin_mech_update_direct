# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Defines the SubSol class.

Manages individual field variables (e.g.,
displacement, pressure) within a coupled system, including their
finite element definitions and initialization strategies.
"""

import dolfin
import numpy

################################################################################


class SubSol:
	r"""Class representing a single unknown field (sub-solution) in a mixed problem.

	A ``SubSol`` encapsulates the definition of a specific variable (e.g., displacement
	vector :math:`\mathbf{u}`, pressure scalar :math:`p`) within a larger coupled
	system. It manages the Finite Element (FE) type and the strategy for
	initialization (constant value vs. spatial field).



	This abstraction simplifies the assembly of mixed function spaces and the
	application of initial conditions in the ``Problem`` class.

	:param name: Name of the variable (e.g., "U", "p", "Phis").
	:type name: str
	:param fe: The FEniCS FiniteElement definition (e.g., VectorElement("CG", cell, 2)).
	:type fe: dolfin.FiniteElement
	:param init_val: (Optional) A constant value (scalar or array) to initialize the field everywhere.
	:type init_val: float or numpy.ndarray
	:param init_fun: (Optional) A FEniCS Function or Expression to initialize the field spatially.
	:type init_fun: dolfin.Function
	"""

	def __init__(self, name, fe, init_val=None, init_fun=None):
		"""Initializes the SubSol instance and determines the initialization strategy."""
		self.name = name
		self.fe = fe

		if (init_val is None) and (init_fun is None):
			self.init_val = numpy.zeros(fe.value_shape())
			self.init = self.init_with_val
		elif (init_val is not None) and (init_fun is None):
			assert numpy.shape(init_val) == self.fe.value_shape()
			self.init_val = numpy.asarray(init_val)
			self.init = self.init_with_val
		elif (init_val is None) and (init_fun is not None):
			self.init_fun = init_fun
			self.init = self.init_with_field
		else:
			assert 0, "Can only provide init_val or init_fun. Aborting."

	def init_with_val(self):
		"""Initializes the function with a constant value.

		This method creates a constant FEniCS Expression from ``self.init_val``
		and interpolates it into the function space. It updates both the current
		solution (``self.func``) and the previous step solution (``self.func_old``).
		"""
		init_val_str = self.init_val.astype(str).tolist()
		# print(self.func.vector().get_local())
		self.func.interpolate(dolfin.Expression(init_val_str, element=self.fe))
		# print(self.func.vector().get_local())
		self.func_old.interpolate(dolfin.Expression(init_val_str, element=self.fe))

	def init_with_field(self):
		"""Initializes the function using an existing field (Function or Expression).

		This copies the vector values from ``self.init_fun`` directly into
		``self.func`` and ``self.func_old``. This is useful for restarting
		simulations or mapping results from a previous step.
		"""
		self.func.vector()[:] = self.init_fun.vector().get_local()
		self.func_old.vector()[:] = self.init_fun.vector().get_local()
