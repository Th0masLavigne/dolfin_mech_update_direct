# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Defines the Step class.

Manages a single temporal or loading interval
within a numerical simulation, including its time-stepping strategy,
governing operators, and boundary constraints.
"""
################################################################################


class Step:
	r"""Class representing a single time step or load step in a simulation.

	A ``Step`` defines a time interval :math:`[t_{ini}, t_{fin}]` and the
	associated time-stepping strategy (initial, minimum, and maximum ``dt``).
	It also acts as a container for all **Operators** (physics terms) and
	**Constraints** (boundary conditions) that are active during this specific interval.



	This structure allows for multi-stage simulations where boundary conditions
	or physics change over time (e.g., inflation followed by relaxation).

	:param t_ini: Start time of the step.
	:type t_ini: float
	:param t_fin: End time of the step.
	:type t_fin: float
	:param dt_ini: Initial time increment :math:`\Delta t` to attempt. Defaults to the full step duration.
	:type dt_ini: float
	:param dt_min: Minimum allowable time increment (for adaptive refinement). Defaults to ``dt_ini``.
	:type dt_min: float
	:param dt_max: Maximum allowable time increment. Defaults to ``dt_ini``.
	:type dt_max: float
	:param operators: List of operator objects active in this step.
	:type operators: list
	:param constraints: List of constraint objects active in this step.
	:type constraints: list
	"""

	def __init__(
		self,
		t_ini=0.0,
		t_fin=1.0,
		dt_ini=None,
		dt_min=None,
		dt_max=None,
		operators=None,  # MG20180508: Do not use list as default value because it is static
		constraints=None,
	):  # MG20180508: Do not use list as default value because it is static
		"""Initializes a Step instance."""
		self.t_ini = t_ini
		self.t_fin = t_fin

		self.dt_ini = dt_ini if (dt_ini is not None) else self.t_fin - self.t_ini
		self.dt_min = dt_min if (dt_min is not None) else self.dt_ini
		self.dt_max = dt_max if (dt_max is not None) else self.dt_ini

		self.operators = operators if (operators is not None) else []
		self.constraints = constraints if (constraints is not None) else []
