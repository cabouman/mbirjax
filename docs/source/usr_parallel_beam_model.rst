.. _ParallelBeamModelDocs:

=================
ParallelBeamModel
=================

The ``ParallelBeamModel`` extends the functionalities provided by :ref:`TomographyModelDocs`.
This class inherits all behaviors and attributes of the TomographyModel and implements projectors specific
to parallel beam CT.

In addition, ``ParallelBeamModel`` includes filtered backprojection reconstruction as indicated below.

Constructor
-----------

.. autoclass:: mbirjax.ParallelBeamModel
   :show-inheritance:

Filtered Back Projection
------------------------

.. automethod:: mbirjax.ParallelBeamModel.fbp_recon

Parent Class
------------
:ref:`TomographyModelDocs`

