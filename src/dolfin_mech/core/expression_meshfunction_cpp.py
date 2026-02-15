# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Colin Laville, 2021-2022                                             ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################


"""C++ backend utility for FEniCS.

Provides a high-performance C++ backend utility for FEniCS, enabling
the evaluation of MeshFunctions within variational forms by bypassing
Python-level overhead through pybind11-compiled Expressions.
"""


def get_ExprMeshFunction_cpp_pybind():
	"""Return the C++ source code for a pybind11-based dolfin Expression.

	This utility generates C++ code that defines a ``MeshExpr`` class. This class
	inherits from ``dolfin::Expression`` and allows for the efficient evaluation
	of scalar ``MeshFunction<double>`` data during FEniCS assembly.

	By using a C++ implementation via pybind11, the overhead of calling back into
	Python for every cell evaluation is avoided, significantly improving performance
	for large meshes.

	Returns:
	    str: The complete C++ source code string to be compiled by
	    :py:func:`dolfin.compile_cpp_code`.

	Note:
	    The generated C++ class ``MeshExpr`` expects a member ``mf`` (a shared pointer
	    to a ``dolfin::MeshFunction<double>``) to be set after instantiation in Python.

	Example:
	    >>> cpp_code = get_ExprMeshFunction_cpp_pybind()
	    >>> module = dolfin.compile_cpp_code(cpp_code)
	    >>> expr = dolfin.CompiledExpression(module.MeshExpr(), degree=0)
	    >>> expr.mf = my_mesh_function
	"""
	cpp_code = """
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <dolfin/function/Expression.h>
#include <dolfin/mesh/MeshFunction.h>

class MeshExpr : public dolfin::Expression
{
public:
    // The data stored in mesh functions
    std::shared_ptr<dolfin::MeshFunction<double>> mf;

    // Create scalar expression
    MeshExpr() : dolfin::Expression() {}

    // Function for evaluating expression on each cell
    void eval(
        Eigen::Ref<Eigen::VectorXd> values,
        Eigen::Ref<const Eigen::VectorXd> x,
        const ufc::cell& cell) const override
    {
        const uint cell_index = cell.index;
        values[0] = (*mf)[cell_index];
    }
};

PYBIND11_MODULE(SIGNATURE, m)
{
pybind11::class_<MeshExpr, std::shared_ptr<MeshExpr>, dolfin::Expression>
(m, "MeshExpr")
.def(pybind11::init<>())
.def_readwrite("mf", &MeshExpr::mf);
}
"""

	return cpp_code
