.. _ParallelBeamModelDocs:

=================
ParallelBeamModel
=================

The ``ParallelBeamModel`` class implements a geometry and reconstruction model for parallel-beam computed tomography.
This class inherits all behaviors and attributes of the :ref:`TomographyModelDocs`.
It also implements some parallel-beam specific functions such as FBP (Filtered Back Projection) reconstruction.

See the API docs for the :class:`~mbirjax.TomographyModel` class for details on a wide range
of functions that can be implemented using the ``ParallelBeamModel``.



Constructor
-----------

.. autoclass:: mbirjax.ParallelBeamModel
   :show-inheritance:

Filtered Back Projection
------------------------

.. automethod:: mbirjax.ParallelBeamModel.fbp_recon



