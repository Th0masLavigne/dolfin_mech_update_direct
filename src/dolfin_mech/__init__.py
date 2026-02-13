"""Dolfin Mech Package.

|doi| |pypi| |downloads|

A set of FEniCS-based Python tools for Computational Mechanics.

The library has notably been used in:

* `Genet (2019). A relaxed growth modeling framework for controlling growth-induced residual stresses. Clinical
   Biomechanics. <https://doi.org/10.1016/j.clinbiomech.2019.08.015>`_
* `Álvarez-Barrientos, Hurtado & Genet (2021). Pressure-driven micro-poro-mechanics: A variational framework for
   modeling the response of porous materials. International Journal of Engineering Science. <https://doi.org/10.1016/j.ijengsci.2021.103586>`_
* `Patte, Genet & Chapelle (2022). A quasi-static poromechanical model of the lungs. Biomechanics and Modeling in
   Mechanobiology. <https://doi.org/10.1007/s10237-021-01547-0>`_
* `Patte et al. (2022). Estimation of regional pulmonary compliance in idiopathic pulmonary fibrosis based on
   personalized lung poromechanical modeling. Journal of Biomechanical Engineering. <https://doi.org/10.1115/1.4054106>`_
* `Tueni, Allain & Genet (2023). On the structural origin of the anisotropy in the myocardium: Multiscale modeling
   and analysis. Journal of the Mechanical Behavior of Biomedical Materials. <https://doi.org/10.1016/j.jmbbm.2022.105600>`_
* `Laville et al. (2023). Comparison of optimization parametrizations for regional lung compliance estimation using
   personalized pulmonary poromechanical modeling. Biomechanics and Modeling in Mechanobiology. <https://doi.org/10.1007/s10237-023-01691-9>`_
* `Peyraut & Genet (2024). A model of mechanical loading of the lungs including gravity and a balancing heterogeneous
   pleural pressure. Biomechanics and Modeling in Mechanobiology. <https://doi.org/10.1007/s10237-024-01876-w>`_
* `Peyraut & Genet (2025). Finite strain formulation of the discrete equilibrium gap principle: application to direct
   parameter estimation from large full-fields measurements. Comptes Rendus Mécanique. <https://doi.org/10.5802/crmeca.279>`_
* `Manoochehrtayebi, Bel-Brunon & Genet (2025). Finite strain micro-poro-mechanics: Formulation and compared analysis
   with macro-poro-mechanics. International Journal of Solids and Structures. <https://doi.org/10.1016/j.ijsolstr.2025.113354>`_
* `Peyraut & Genet (2025). Inverse Uncertainty Quantification for Personalized Biomechanical Modeling: Application to
   Pulmonary Poromechanical Digital Twins. Journal of Biomechanical Engineering. <https://doi.org/10.1115/1.4068578>`_
* `Manoochehrtayebi, Genet & Bel-Brunon (2025). Micro-poro-mechanical modeling of lung parenchyma: Theoretical modeling
   and parameters identification. Journal of Biomechanical Engineering. <https://doi.org/10.1115/1.4070036>`_

Installation
------------

A working installation of `FEniCS <https://fenicsproject.org>`_ (version 2019.1.0) is required.
To setup a system, the simplest is to use **conda**:

1. Install `miniconda <https://docs.conda.io/projects/miniconda/en/latest>`_.
2. Install the necessary packages:

.. code-block:: bash

   conda create -y -c conda-forge -n dolfin_mech fenics=2019.1.0 matplotlib=3.5 meshio=5.3 mpi4py=3.1.3 numpy=1.23 pandas=1.3 pip python=3.10 vtk=9.2
   conda activate dolfin_mech

To install the library:

.. code-block:: bash

   pip install dolfin_mech

For development:

.. code-block:: bash

   git clone https://github.com/mgenet/dolfin_mech.git
   pip install -e dolfin_mech/.

.. |doi| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.8010870.svg?style=flat-square
   :target: https://doi.org/10.5281/zenodo.8010870

.. |pypi| image:: https://img.shields.io/pypi/v/dolfin-mech.svg?style=flat-square
   :target: https://pypi.org/project/dolfin-mech

.. |downloads| image:: https://static.pepy.tech/badge/dolfin-mech
   :target: https://pepy.tech/projects/dolfin-mech
"""

import importlib.metadata

# Try to get the version from pyproject.toml metadata
try:
	__version__ = importlib.metadata.version("dolfin_mech")
except importlib.metadata.PackageNotFoundError:
	__version__ = "0.0.0"

# Expose sub-packages
from . import kinematics, materials, operators

# Define what gets imported when someone uses `from rosaly_hydrogels import *`
__all__ = ["materials", "kinematics", "operators"]
