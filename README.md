




# dolfin_mech

`dolfin_mech` is a powerful Python library built on [FEniCS](https://fenicsproject.org) designed for **Computational Mechanics**. It provides a modular high-level interface to solve complex problems in nonlinear elasticity, poromechanics, and multi-scale homogenization.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8010870.svg?style=flat-square)](https://doi.org/10.5281/zenodo.8010870)
[![PyPi Version](https://img.shields.io/pypi/v/dolfin-mech.svg?style=flat-square)](https://pypi.org/project/dolfin-mech)
[![PyPI Downloads](https://static.pepy.tech/badge/dolfin-mech)](https://pepy.tech/projects/dolfin-mech)
[![Docs](https://img.shields.io/badge/docs-orange)](https://mgenet.github.io/dolfin_mech)

---

## The Core Idea: Compositional Mechanics

The design philosophy of `dolfin_mech` is to decouple the physics of a mechanical problem into independent, reusable components. Instead of writing monolithic solvers, users compose a simulation by selecting:

### 1. Kinematics (`kinematics/`)

Defines the **geometric transformation** and the measure of strain. It handles the mapping between reference and current configurations.

* **Role:** Computes quantities like the deformation gradient , Green-Lagrange strain , or specific kinematics for **Inverse Problems** (finding the stress-free state from a deformed image).

### 2. Operators (`operators/`)

These represent the **individual terms in the weak form** of the partial differential equations (PDEs).

* **Role:** Rather than defining one massive equation, `dolfin_mech` builds the equilibrium by summing operators (e.g., Internal Forces + Gravity + Surface Tension + Pressure). This allows for easy "plug-and-play" of different loading conditions.

### 3. Materials (`materials/`)

The **constitutive laws** that relate kinematics to stresses.

* **Role:** Given a kinematic measure, the material module returns the strain energy density  or the Stress tensor . It supports hyperelasticity (Neo-Hookean, Ogden) and complex porous media.

### 4. Problems (`problems/`)

The **orchestrator** that brings everything together.

* **Role:** This is the high-level class (e.g., `SolidMechanicsProblem`) that links the mesh, the kinematics, and the operators into a FEniCS `NonlinearVariationalProblem`.

### 5. Time Integrator (`core/`)

Manages the **evolution of the system** over time.

* **Role:** It handles the incremental stepping, Newton-Raphson iterations, and updates of state variables. It ensures that quasi-static or dynamic trajectories are followed accurately and saved (IO) consistently.


---

## Project Structure

The library is currently undergoing a transition to a modern `src/` layout to improve maintainability and extensibility.

```text
.
├── src/dolfin_mech/         # Core Library (New Structure)
│   ├── core/                # Time integrators, IO (XDMF/VTU), and error tools
│   ├── kinematics/          # Linear/Non-linear and Inverse kinematics
│   ├── materials/           # Constitutive laws (Neo-Hookean, Ogden, Porous, etc.)
│   ├── operators/           # Weak forms and FEM operators
│   └── problems/            # High-level problem definitions (Elasticity, Poro, etc.)
├── tests/                   # Extensive regression test suite with reference data
├── docker/                  # Reproducible environments (standard and dev)
├── docs/                    # Sphinx documentation sources
├── LEGACY_resources/        # Legacy scripts and material formulations
└── pyproject.toml           # Modern PEP 517 build configuration

```

---

## Installation

### Option 1: Conda Environment

Since FEniCS `2019.1.0` requires specific dependency versions, we recommend creating a dedicated environment:

```bash
conda create -y -c conda-forge -n dolfin_mech \
    fenics=2019.1.0 matplotlib=3.5 meshio=5.3 mpi4py=3.1.3 \
    numpy=1.23 pandas=1.3 pip python=3.10 vtk=9.2
conda activate dolfin_mech

```

### Option 2: Pip Installation

For standard usage:

```bash
pip install dolfin-mech

```


If you need to work with a specific version from github you can (for example changing `@{branch_name}` to `@devel`):
```
pip install "git+https://github.com/mgenet/dolfin_mech.git@{branch_name}"
```

For development (editable mode with testing/doc tools), adding the `{branch_name}` you wish to work with:
```bash
git clone https://github.com/mgenet/dolfin_mech.git@{branch_name}
cd dolfin_mech
pip install -e .[dev,docs]
```

### Option 3: Docker

If you prefer containerized execution, use the provided Dockerfiles:

```bash
docker build -t dolfin_mech:latest -f docker/dolfin_mech/Dockerfile .

```
**Run the container:**
Assuming the following folder organization:
```terminal
project
├── outputs
└── source
```
You can run the project image by mounting the `source` and the `outputs` as
```bash
docker run -ti --rm --mount type=bind,src=$(pwd)/source,dst=/hydrogels/source --mount type=bind,src=$(pwd)/outputs,dst=/hydrogels/outputs dolfin:v2019.2.0.dev0

```


## Running a Simulation

To run the a file:

```bash
# Inside the container or environment
python3 file.py

```

---

## Features

### 1. Constitutive Modeling

Supports a wide range of material behaviors:

* **Hyperelasticity:** Neo-Hookean, Mooney-Rivlin, Ogden, Ciarlet-Geymonat.
* **Biomechanics-specific:** Lung-specific  and  formulations.
* **Poromechanics:** Coupled fluid-solid interactions and micro-poro-mechanical models.

### 2. Advanced Solvers & Operators

* **Inverse Problems:** Solve for stress-free configurations or identify parameters.
* **Homogenization:** Tools for RVE analysis and macroscopic property estimation.
* **Loading Operators:** Surface tension, pressure-balancing gravity, and volume forces.

### 3. Verification & Testing

`dolfin_mech` includes a robust testing framework that compares simulation results against analytical benchmarks (e.g., Rivlin Cube) using `.dat` reference files.

---

## Documentation

The documentation is built via Sphinx. To generate the HTML docs locally (remember not to push them online):

```bash
sphinx-build -d html docs/src/ ../docs/build_docs/

```

## Key Publications

This framework has been validated and utilized in numerous peer-reviewed studies across biomechanics and materials science:

<sub>
* [[Genet (2019). A relaxed growth modeling framework for controlling growth-induced residual stresses. Clinical Biomechanics.](https://doi.org/10.1016/j.clinbiomech.2019.08.015)]
* [[Álvarez-Barrientos, Hurtado & Genet (2021). Pressure-driven micro-poro-mechanics: A variational framework for modeling the response of porous materials. International Journal of Engineering Science.](https://doi.org/10.1016/j.ijengsci.2021.103586)]
* [[Patte, Genet & Chapelle (2022). A quasi-static poromechanical model of the lungs. Biomechanics and Modeling in Mechanobiology.](https://doi.org/10.1007/s10237-021-01547-0)]
* [[Patte, Brillet, Fetita, Gille, Bernaudin, Nunes, Chapelle & Genet (2022). Estimation of regional pulmonary compliance in idiopathic pulmonary fibrosis based on personalized lung poromechanical modeling. Journal of Biomechanical Engineering.](https://doi.org/10.1115/1.4054106)]
* [[Tueni, Allain & Genet (2023). On the structural origin of the anisotropy in the myocardium: Multiscale modeling and analysis. Journal of the Mechanical Behavior of Biomedical Materials.](https://doi.org/10.1016/j.jmbbm.2022.105600)]
* [[Laville, Fetita, Gille, Brillet, Nunes, Bernaudin & Genet (2023). Comparison of optimization parametrizations for regional lung compliance estimation using personalized pulmonary poromechanical modeling. Biomechanics and Modeling in Mechanobiology.](https://doi.org/10.1007/s10237-023-01691-9)]
* [[Peyraut & Genet (2024). A model of mechanical loading of the lungs including gravity and a balancing heterogeneous pleural pressure. Biomechanics and Modeling in Mechanobiology.](https://doi.org/10.1007/s10237-024-01876-w)]
* [[Peyraut & Genet (2025). Finite strain formulation of the discrete equilibrium gap principle: application to direct parameter estimation from large full-fields measurements. Comptes Rendus Mécanique.](https://doi.org/10.5802/crmeca.279)]
* [[Manoochehrtayebi, Bel-Brunon & Genet (2025). Finite strain micro-poro-mechanics: Formulation and compared analysis with macro-poro-mechanics. International Journal of Solids and Structures.](https://doi.org/10.1016/j.ijsolstr.2025.113354)]
* [[Peyraut & Genet (2025). Inverse Uncertainty Quantification for Personalized Biomechanical Modeling: Application to Pulmonary Poromechanical Digital Twins. Journal of Biomechanical Engineering.](https://doi.org/10.1115/1.4068578)]
* [[Manoochehrtayebi, Genet & Bel-Brunon (2025). Micro-poro-mechanical modeling of lung parenchyma: Theoretical modeling and parameters identification. Journal of Biomechanical Engineering.](https://doi.org/10.1115/1.4070036)]
</sub>