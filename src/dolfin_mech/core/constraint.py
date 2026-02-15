# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Defines the Constraint class.

Integrates FEniCS Dirichlet boundary
conditions with time-varying magnitudes to enable dynamic prescription
of displacements, pressures, or other field variables.
"""

import dolfin

from dolfin_mech.core import TimeVaryingConstant

################################################################################


class Constraint:
	"""Class to manage Dirichlet boundary conditions with time-varying values.

	This class wraps the ``dolfin.DirichletBC`` object and associates it with a
	:py:class:`dolfin_mech.TimeVaryingConstant`. This allows for easy updates
	of the boundary condition values during time-stepping simulations.

	Args:
	        V (dolfin.FunctionSpace): The function space on which the BC is applied.
	        sub_domain (dolfin.SubDomain, optional): A FEniCS SubDomain object
	            defining the boundary. Defaults to None.
	        sub_domains (dolfin.MeshFunction, optional): A mesh function
	            defining sub-regions. Defaults to None.
	        sub_domain_id (int, optional): The ID within ``sub_domains`` to
	            apply the BC to. Defaults to None.
	        val (float, optional): A constant value for the entire simulation.
	        val_ini (float, optional): Initial value for time-varying BC.
	        val_fin (float, optional): Final value for time-varying BC.
	        method (str, optional): FEniCS BC method (e.g., 'topological',
	            'geometric', 'pointwise'). Defaults to None.

	Notes:
	        You must provide either (``val``) OR (``val_ini`` and ``val_fin``).
	        Similarly, you must provide either (``sub_domain``) OR
	        (``sub_domains`` and ``sub_domain_id``).

	Attributes:
	    tv_val (dolfin_mech.TimeVaryingConstant): Manager for the constant value
	        applied to the boundary.
	    bc (dolfin.DirichletBC): The underlying FEniCS Dirichlet boundary
	        condition object.
	"""

	def __init__(
		self,
		V,
		sub_domain=None,
		sub_domains=None,
		sub_domain_id=None,
		val=None,
		val_ini=None,
		val_fin=None,
		method=None,
	):  # topological, geometric, pointwise
		"""Initialize the Constraint and create the underlying DirichletBC."""
		if (val is not None) and (val_ini is None) and (val_fin is None):
			self.tv_val = TimeVaryingConstant(val_ini=val, val_fin=val)
		elif (val is None) and (val_ini is not None) and (val_fin is not None):
			self.tv_val = TimeVaryingConstant(val_ini=val_ini, val_fin=val_fin)

		if (sub_domain is not None) and (sub_domains is None) and (sub_domain_id is None):
			if method is None:
				self.bc = dolfin.DirichletBC(V, self.tv_val.val, sub_domain)
			else:
				self.bc = dolfin.DirichletBC(V, self.tv_val.val, sub_domain, method)
		elif (sub_domain is None) and (sub_domains is not None) and (sub_domain_id is not None):
			if method is None:
				self.bc = dolfin.DirichletBC(V, self.tv_val.val, sub_domains, sub_domain_id)
			else:
				self.bc = dolfin.DirichletBC(V, self.tv_val.val, sub_domains, sub_domain_id, method)

	def set_value_at_t_step(self, t_step):
		"""Update the boundary condition value for a specific time step.

		Args:
		    t_step (float): The current time step/progress (typically between 0 and 1).
		"""
		self.tv_val.set_dvalue_at_t_step(t_step)

	def restore_old_value(self):
		"""Restore the value of the constant to its previous state."""
		self.tv_val.restore_old_value()

	def homogenize(self):
		"""Set the boundary condition value to zero (homogeneous)."""
		self.tv_val.homogenize()
		# self.bc.homogenize() # MG20180508: seems to be changing the constant…
