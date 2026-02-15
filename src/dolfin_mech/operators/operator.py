# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Defines the Operator base class.

Provides a unified interface for
variational form contributions (residual and Jacobian) and manages
time-dependent updates for loads, material properties, and discretization.
"""


################################################################################


class Operator:
	"""Base class for all variational operators in dolfin_mech.

	An operator encapsulates a contribution to the residual variational form
	(and potentially the Jacobian) of a mechanical problem. It provides an
	interface to update internal parameters, such as magnitudes of loads or
	material properties, throughout the simulation stages.



	Attributes:
	    res_form (UFL form): The residual variational form contribution of
	        the operator. This is typically defined in the __init__ of derived
	        classes.
	"""

	def set_value_at_t_step(self, *args, **kwargs):
		"""Updates internal time-varying parameters based on the current simulation time.

		This method is called by the solver at the beginning of each time step.
		Derived classes should override this to update magnitudes for ramping
		loads or time-dependent material coefficients.

		:param t_step: Typically the normalized time or specific time value.
		"""
		pass

	def set_dt(self, *args, **kwargs):
		r"""Updates the time step size within the operator.

		This is primarily used by operators involving time derivatives (e.g.,
		:class:`InertiaOperator` or :class:`ViscosityOperator`) to update the
		discretization parameter :math:`\Delta t`.

		:param dt: The current time step size.
		:type dt: float
		"""
		pass
