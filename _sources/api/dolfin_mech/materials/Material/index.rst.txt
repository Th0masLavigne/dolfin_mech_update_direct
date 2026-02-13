dolfin_mech.materials.Material
==============================

.. py:module:: dolfin_mech.materials.Material

.. autoapi-nested-parse::

   Material Constitutive Laws Module.

   This module provides a unified framework for defining and instantiating material
   constitutive laws within the ``dolfin_mech`` ecosystem. It serves two primary roles:

   1. **Parameter Normalization**: A base ``Material`` class provides automated
      conversion between engineering constants (e.g., :math:`E, \nu`) and theoretical
      parameters (e.g., :math:`\lambda, \mu`).
   2. **The Material Factory**: A robust dispatcher function that maps string
      identifiers to specific elastic material classes.

   Crucial Material Factory Identifiers
   ------------------------------------

   The ``material_factory`` function supports an exhaustive list of aliases.
   These are categorized below by their underlying constitutive logic as depicted in the next table.
   Otherwise, you can use the Material name in lowercase.

   .. list-table:: Main Material Aliases
      :widths: 30 35 35
      :header-rows: 1

      * - Physical Model
        - Standard Aliases
        - Deviatoric / Bulk / Decoupled
      * - **Linear Elasticity**
        - ``"hooke"``, ``"Hooke"``, ``"H"``
        - ``"_dev"``, ``"_bulk"``
      * - **Saint-Venant Kirchhoff**
        - ``"SVK"``, ``"kirchhoff"``, ``"saintvenantkirchhoff"``
        - ``"_dev"``, ``"_bulk"``
      * - **Neo-Hookean**
        - ``"neohookean"``, ``"NH"``
        - ``"_bar"`` (Decoupled)
      * - **Mooney-Rivlin**
        - ``"mooneyrivlin"``, ``"MR"``
        - ``"_bar"`` (Decoupled)
      * - **Mixed NH-MR**
        - ``"NHMR"``
        - ``"_bar"`` (Decoupled)
      * - **Ogden-Ciarlet-Geymonat**
        - ``"OCG"``, ``"CG"``
        - ``"_bar"`` (Decoupled)

   Key Features
   ------------

   * **Lame Parameter Computation**: Automated handling of Plane Stress (``PS``)
     vs. Plane Strain/3D conditions when calculating $\lambda$ from Young's
     Modulus and Poisson's ratio.
   * **Hyperelastic Coefficients**: Built-in logic to derive hyperelastic constants
     (:math:`C_{10}, C_{01}`) from shear moduli if they are not explicitly provided.
   * **Decoupled Formulations**: The ``_bar`` suffix in the factory triggers
     the ``decoup=True`` flag, essential for nearly-incompressible materials
     where volumetric and isochoric responses are separated.

   .. note::

      The factory uses case-insensitive matching for most identifiers, but
      standardizing on the aliases above is recommended for readability.







Module Contents
---------------

.. py:class:: Material

   Base class for material models providing utility methods for parameter conversion.

   This class handles the translation between common engineering constants
   (Young's modulus :math:`E`, Poisson's ratio :math:`\nu`) and the theoretical
   parameters required by constitutive laws (Lamé constants :math:`\lambda, \mu`,
   bulk modulus :math:`K`, etc.).


   .. py:method:: get_lambda_from_parameters(parameters)

      Compute the first Lamé parameter :math:`\lambda`.

      :Parameters: **parameters** (:py:class:`dict`) -- Dictionary containing material constants.
                   Expected keys: ``"lambda"`` OR (``"E"`` AND ``"nu"``).
                   Optional key: ``"PS"`` (bool) for Plane Stress formulation.

      :returns: The value of :math:`\lambda`.
      :rtype: dolfin.Constant



   .. py:method:: get_mu_from_parameters(parameters)

      Compute the second Lamé parameter (shear modulus) :math:`\mu`.

      :Parameters: **parameters** (:py:class:`dict`) -- Dictionary containing material constants.
                   Expected keys: ``"mu"`` OR (``"E"`` AND ``"nu"``).

      :returns: The value of :math:`\mu`.
      :rtype: dolfin.Constant



   .. py:method:: get_lambda_and_mu_from_parameters(parameters)

      Compute both Lamé parameters simultaneously.

      :returns: (dolfin.Constant, dolfin.Constant) representing (:math:`\lambda, \mu`).
      :rtype: tuple



   .. py:method:: get_K_from_parameters(parameters)

      Compute the bulk modulus :math:`K`.

      If ``"K"`` is not in parameters, it is derived from :math:`\lambda` and :math:`\mu`
      as :math:`K = (3\lambda + 2\mu)/3`.



   .. py:method:: get_G_from_parameters(parameters)

      Compute the shear modulus :math:`G`. Equivalent to :math:`\mu`.



   .. py:method:: get_C0_from_parameters(parameters, decoup=False)

      Compute the hyperelastic coefficient :math:`C_0`.

      This parameter is typically associated with the volumetric or
      compressibility part of the strain energy density function.

      :Parameters: * **parameters** (:py:class:`dict`) -- Dictionary of material parameters.
                   * **decoup** (:py:class:`bool, optional`) -- If True, computes :math:`C_0` from the bulk
                     modulus :math:`K`: :math:`C_0 = K/4`. If False, computes it
                     from :math:`\lambda`: :math:`C_0 = \lambda/4`. Defaults to False.

      :returns: The computed :math:`C_0` value.
      :rtype: dolfin.Constant



   .. py:method:: get_C1_from_parameters(parameters)

      Compute the hyperelastic coefficient :math:`C_1`.

      In many models, :math:`C_1` is related to the shear response of the material.

      :Parameters: **parameters** (:py:class:`dict`) -- Dictionary of material parameters.
                   Expected keys: ``"C1"``, ``"c1"``, ``"mu"``, or (``"E"`` and ``"nu"``).

      :returns: The computed :math:`C_1` value.
      :rtype: dolfin.Constant

      .. rubric:: Notes

      If not explicitly provided, it is derived as :math:`C_1 = \mu/2`.



   .. py:method:: get_C2_from_parameters(parameters)

      Compute the hyperelastic coefficient :math:`C_2`.

      :Parameters: **parameters** (:py:class:`dict`) -- Dictionary of material parameters.
                   Expected keys: ``"C2"``, ``"c2"``, ``"mu"``, or (``"E"`` and ``"nu"``).

      :returns: The computed :math:`C_2` value.
      :rtype: dolfin.Constant

      .. rubric:: Notes

      If not explicitly provided, it is derived as :math:`C_2 = \mu/2`.



   .. py:method:: get_C1_and_C2_from_parameters(parameters)

      Compute Mooney-Rivlin coefficients :math:`C_1` and :math:`C_2`.

      If only ``"mu"`` is provided, it assumes :math:`C_1 = C_2 = \mu/4`.



.. py:function:: material_factory(kinematics, model, parameters)

   Factory function to instantiate the appropriate material model class.

   :Parameters: * **kinematics** (:py:class:`Kinematics`) -- An instance of a Kinematics class.
                * **model** (:py:class:`str`) -- The name of the material model (e.g., ``"Hooke"``, ``"NeoHookean"``, ``"SVK"``).
                * **parameters** (:py:class:`dict`) -- Material parameters passed to the model constructor.

   :returns: An instance of the requested material model.
   :rtype: Material

   .. rubric:: Example

   >>> mat = material_factory(kin, "NH", {"E": 10.0, "nu": 0.3})


