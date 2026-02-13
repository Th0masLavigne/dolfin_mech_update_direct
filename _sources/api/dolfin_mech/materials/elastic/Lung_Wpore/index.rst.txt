dolfin_mech.materials.elastic.Lung_Wpore
========================================

.. py:module:: dolfin_mech.materials.elastic.Lung_Wpore

.. autoapi-nested-parse::

   Pore-level lung tissue elasticity implementation.

   This module provides energy potentials representing alveolar resistance to
   extreme compression (collapse) and extension (distension).





Module Contents
---------------

.. py:class:: WporeLung(Phif, Phif0, parameters)

   Bases: :py:obj:`dolfin_mech.materials.elastic.Elastic.ElasticMaterial`


   Class representing the pore-level elastic energy of lung tissue.

       Typically modeling the resistance of alveoli to collapse (atelectasis) and over-distension.

   This material model defines a potential based on the ratio of the current
   fluid volume fraction :math:`\Phi_f` to the reference fraction :math:`\Phi_{f0}`
   The energy is zero within a "physiological range" and grows according to a
   power law beyond specific upper and lower thresholds.

   The strain energy density :math:`\Psi` is defined using a triple conditional:

   .. math::
       \Psi = \eta \cdot
       \begin{cases}
       (r_{inf}/r - 1)^{n+1} & \text{if } r < r_{inf} \\
       (r/r_{sup} - 1)^{n+1} & \text{if } r > r_{sup} \\
       0 & \text{otherwise}
       \end{cases}

   Where:
       - :math:`r = \Phi_f / \Phi_{f0}` is the volume fraction ratio.
       - :math:`r_{inf} = \Phi_{f0}^{p-1}` is the lower activation threshold.
       - :math:`r_{sup} = \Phi_{f0}^{1/q-1}` is the upper activation threshold.
       - :math:`\eta` is the stiffness scaling parameter.
       - :math:`n, p, q` are shape and threshold parameters.

       Attributes:
       eta (dolfin.Constant): Stiffness parameter for the pore energy.
       n, p, q (dolfin.Constant): Power-law and threshold exponents.
       Psi (UFL expression): The calculated conditional strain energy density.
       dWporedPhif (UFL expression): Derivative of the energy with respect to the fluid volume fraction.




