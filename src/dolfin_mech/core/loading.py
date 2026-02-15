# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Defines the Loading base class.

Couples FEniCS integration measures
with time-varying scalar magnitudes to handle evolving boundary tractions,
pressures, or body forces.
"""

import dolfin

from dolfin_mech.core import TimeVaryingConstant

################################################################################


class Loading:
	"""Base class for managing time-dependent mechanical loads.

	This class serves as a container for loading parameters (such as pressure,
	traction, or body forces) that vary during a simulation. It associates
	a FEniCS measure with a :py:class:`dolfin_mech.TimeVaryingConstant`.

	Args:
		    measure (dolfin.Measure): FEniCS integration measure.
		    val (float, optional): A constant value for the entire simulation.
		    val_ini (float, optional): Initial value for time-varying loading.
		    val_fin (float, optional): Final value for time-varying loading.
		    xyz_ini (numpy.ndarray or list, optional): Initial position vector.
		    N (numpy.ndarray or list, optional): Directional vector (e.g., surface normal).

	Notes:
		    You must provide either (``val``) OR (``val_ini`` and ``val_fin``).
		    The magnitude is stored internally within a
		    :py:class:`dolfin_mech.TimeVaryingConstant`.

	Attributes:
	    measure (dolfin.Measure): The FEniCS measure (e.g., ``dx``, ``ds``)
	        defining the integration domain for the load.
	    tv_val (dolfin_mech.TimeVaryingConstant): Manager for the scalar load magnitude.
	    val (dolfin.Constant): The current value of the load magnitude,
	        linked to the underlying FEniCS constant.
	    N (dolfin.Constant, optional): Normal vector or direction associated with the load.
	    xyz_ini (dolfin.Constant, optional): Initial coordinates, often used for
	        position-dependent loading in the reference configuration.
	"""

	def __init__(self, measure, val=None, val_ini=None, val_fin=None, xyz_ini=None, N=None):
		"""Initialize the Loading object and set up the time-varying magnitude."""
		self.measure = measure

		if (val is not None) and (val_ini is None) and (val_fin is None):
			self.tv_val = TimeVaryingConstant(val_ini=val, val_fin=val)
		elif (val is None) and (val_ini is not None) and (val_fin is not None):
			self.tv_val = TimeVaryingConstant(val_ini=val_ini, val_fin=val_fin)
		self.val = self.tv_val.val

		if N is not None:
			self.N = dolfin.Constant(N)

		if xyz_ini is not None:
			self.xyz_ini = dolfin.Constant(xyz_ini)

	def set_value_at_t_step(self, t_step):
		"""Update the load magnitude for the given time step.

		Args:
		    t_step (float): The current simulation progress (typically
		        normalized between 0 and 1).
		"""
		self.tv_val.set_value_at_t_step(t_step)
