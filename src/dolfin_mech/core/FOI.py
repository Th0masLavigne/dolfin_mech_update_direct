# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Field of Interest (FOI) management module.

This module provides the FOI class to handle the projection and interpolation
of mathematical expressions onto FEniCS function spaces.
"""

import dolfin

################################################################################


class FOI:
	"""Class to manage Field of Interest (FOI) calculations and updates.

	An FOI represents a spatial field (like stress, strain, or porosity) derived
	from a UFL expression. This class handles the mapping of that expression
	onto a FEniCS Function Space using various projection or interpolation methods.

	Args:
		    expr (ufl.core.expr.Expr, optional): The mathematical expression to
		        be evaluated.
		    fs (dolfin.FunctionSpace, optional): The space onto which the
		        expression is mapped.
		    func (dolfin.Function, optional): An existing function to use
		        for storage.
		    name (str, optional): Name of the field (used for XDMF export).
		    form_compiler_parameters (dict, optional): Parameters passed to the
		        FEniCS form compiler.
		    update_type (str, optional): The numerical method for updating the
		        field. Options are:

		        * ``"local_solver"``: Efficient cell-wise projection (default).
		        * ``"project"``: Global L2 projection.
		        * ``"interpolate"``: Pointwise interpolation.

	Attributes:
	    func (dolfin.Function): The actual FEniCS function storing the field data.
	    expr (ufl.core.expr.Expr): The UFL expression used to compute the field.
	    fs (dolfin.FunctionSpace): The function space where the field is defined.
	    update (callable): Method used to refresh the field values (e.g.,
	        :py:meth:`update_local_solver`).
	"""

	def __init__(
		self, expr=None, fs=None, func=None, name=None, form_compiler_parameters={}, update_type="local_solver"
	):  # local_solver or project or interpolate
		"""Initialize the FOI and configure the update mechanism.

		Args:
		    expr (ufl.core.expr.Expr, optional): The mathematical expression to
		        be evaluated.
		    fs (dolfin.FunctionSpace, optional): The space onto which the
		        expression is mapped.
		    func (dolfin.Function, optional): An existing function to use
		        for storage.
		    name (str, optional): Name of the field (used for XDMF export).
		    form_compiler_parameters (dict, optional): Parameters passed to the
		        FEniCS form compiler.
		    update_type (str, optional): The numerical method for updating the
		        field. Options are:

		        * ``"local_solver"``: Efficient cell-wise projection (default).
		        * ``"project"``: Global L2 projection.
		        * ``"interpolate"``: Pointwise interpolation.
		"""
		if (expr is not None) and (fs is not None):
			self.expr = expr
			self.fs = fs
			self.func = func if func is not None else dolfin.Function(fs)
			if name is not None:
				self.name = name
				self.func.rename(self.name, self.name)

			if update_type == "local_solver":
				self.form_compiler_parameters = form_compiler_parameters

				self.func_test = dolfin.TestFunction(self.fs)
				self.func_tria = dolfin.TrialFunction(self.fs)

				self.a_expr = dolfin.inner(self.func_tria, self.func_test) * dolfin.dx(
					metadata=self.form_compiler_parameters
				)
				self.b_expr = dolfin.inner(self.expr, self.func_test) * dolfin.dx(
					metadata=self.form_compiler_parameters
				)
				self.local_solver = dolfin.LocalSolver(self.a_expr, self.b_expr)
				# t = time.time()
				self.local_solver.factorize()
				# t = time.time() - t
				# print("LocalSolver factorization = "+str(t)+" s")

				self.update = self.update_local_solver

			elif update_type == "project":
				self.form_compiler_parameters = form_compiler_parameters

				self.update = self.update_project

			elif update_type == "interpolate":
				self.update = self.update_interpolate

		elif (expr is None) and (fs is None) and (func is not None):
			self.func = func

			self.update = self.update_none

	def update_local_solver(self):
		"""Update the field using a local solver (cell-wise L2 projection).

		This is typically the fastest method for Quadrature spaces or
		Discontinuous Lagrange spaces where cells are independent.
		"""
		# print(self.name)
		# print(self.form_compiler_parameters)

		# t = time.time()
		self.local_solver.solve_local_rhs(self.func)
		# t = time.time() - t
		# print("LocalSolver solve = "+str(t)+" s")

	def update_project(self):
		"""Update the field using global L2 projection via ``dolfin.project``."""
		# print(self.name)
		# print(self.form_compiler_parameters)

		# t = time.time()
		dolfin.project(
			v=self.expr, V=self.fs, function=self.func, form_compiler_parameters=self.form_compiler_parameters
		)
		# t = time.time() - t
		# print("Project = "+str(t)+" s")

	def update_interpolate(self):
		"""Update the field using pointwise interpolation via ``dolfin.Function.interpolate``."""
		# print(self.name)

		# t = time.time()
		self.func.interpolate(self.expr)
		# t = time.time() - t
		# print("Projec = "+str(t)+" s")

	def update_none(self):
		"""Dummy update method for fields that do not require re-calculation."""
		pass
