# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

################################################################################


def compute_error(val, ref):
	r"""Compute the relative error or ratio between a value and a reference.

	The error is calculated as:

	.. math::

	    e = \begin{cases} 0 & \text{if } \text{ref} \approx 0 \text{ and } \text{val} \approx 0 \\ 1 & \text{if }
	    \text{ref} \approx 0 \text{ and } \text{val} \neq 0 \\ \frac{\text{val}}{\text{ref}} & \text{otherwise}
	    \end{cases}

	Args:
	    val (float): The current or calculated value to compare.
	    ref (float): The reference or baseline value.

	Returns:
	    float: The computed relative error or ratio.

	Notes:
	    This function uses :py:func:`dolfin.near` with a tolerance of :math:`10^{-9}`
	    to handle floating point comparisons near zero.
	"""
	if dolfin.near(ref, 0.0, eps=1e-9):
		if dolfin.near(val, 0.0, eps=1e-9):
			return 0.0
		else:
			return 1.0
	else:
		return val / ref
