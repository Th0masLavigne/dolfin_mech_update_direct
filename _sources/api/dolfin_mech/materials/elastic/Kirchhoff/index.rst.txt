dolfin_mech.materials.elastic.Kirchhoff
=======================================

.. py:module:: dolfin_mech.materials.elastic.Kirchhoff

.. autoapi-nested-parse::

   Kirchhoff elastic materials implementation.

   This module provides classes for standard St. Venant-Kirchhoff materials,
   including split formulations for bulk and deviatoric components.





Module Contents
---------------

.. py:class:: Kirchhoff(kinematics, parameters)

   Bases: :py:obj:`dolfin_mech.materials.elastic.Elastic.ElasticMaterial`


   Class representing a standard St. Venant-Kirchhoff elastic material.

   This model extends linear elasticity to large deformations by using the Green-Lagrange
   strain tensor :math:`\mathbf{E}` in the strain energy density function.

   The strain energy density is defined as:

   .. math::
       \Psi = \frac{\lambda}{2} tr(\mathbf{E})^2 + \mu tr(\mathbf{E}^2)

   :ivar kinematics: Object containing kinematic variables (F, J, E, etc.).
   :vartype kinematics: :py:class:`Kinematics`
   :ivar lmbda: First Lamé parameter.
   :vartype lmbda: :py:class:`float`
   :ivar mu: Second Lamé parameter (shear modulus).
   :vartype mu: :py:class:`float`
   :ivar Psi: Strain energy density.
   :vartype Psi: :py:class:`UFL expression`
   :ivar Sigma: Second Piola-Kirchhoff stress tensor.
   :vartype Sigma: :py:class:`UFL expression`
   :ivar P: First Piola-Kirchhoff stress tensor.
   :vartype P: :py:class:`UFL expression`
   :ivar sigma: Cauchy stress tensor.

   :vartype sigma: :py:class:`UFL expression`


.. py:class:: KirchhoffBulk(kinematics, parameters)

   Bases: :py:obj:`dolfin_mech.materials.elastic.Elastic.ElasticMaterial`


   Class representing the volumetric (bulk) component of a Kirchhoff elastic material.

   This model focuses on the spherical part of the strain tensor, typically used in
   penalty methods or split formulations.

   The strain energy density is defined as:

   .. math::
       \Psi = \frac{d \cdot K}{2} tr(\mathbf{E}_{sph})^2

   :ivar K: Bulk modulus.
   :vartype K: :py:class:`float`
   :ivar Psi: Volumetric strain energy density.
   :vartype Psi: :py:class:`UFL expression`
   :ivar Sigma: Spherical part of the Second Piola-Kirchhoff stress.

   :vartype Sigma: :py:class:`UFL expression`


.. py:class:: KirchhoffDev(kinematics, parameters)

   Bases: :py:obj:`dolfin_mech.materials.elastic.Elastic.ElasticMaterial`


   Class representing the deviatoric (shear) component of a Kirchhoff elastic material.

   Used to model the shape-changing part of the deformation, independent of volume change.

   The strain energy density is defined as:

   .. math::
       \Psi = G \cdot (\mathbf{E}_{dev} : \mathbf{E}_{dev})

   :ivar G: Shear modulus.
   :vartype G: :py:class:`float`
   :ivar Psi: Deviatoric strain energy density.
   :vartype Psi: :py:class:`UFL expression`
   :ivar Sigma: Deviatoric part of the Second Piola-Kirchhoff stress.

   :vartype Sigma: :py:class:`UFL expression`


