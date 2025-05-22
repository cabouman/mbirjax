.. _ConeBeamModelDocs:

=============
ConeBeamModel
=============

The ``ConeBeamModel`` class implements a geometry and reconstruction model for cone beam computed tomography.
This class inherits all behaviors and attributes of the :ref:`TomographyModelDocs`.
It also implements some cone-beam specific functions such as FDK (Feldkamp-Davis-Kress) reconstruction.

See the API docs for the :class:`~mbirjax.TomographyModel` class for details on a wide range
of functions that can be implemented using the ``ConeBeamModel``.

Constructor
-----------

.. autoclass:: mbirjax.ConeBeamModel
   :show-inheritance:

FDK Reconstruction
------------------

.. automethod:: mbirjax.ConeBeamModel.fdk_recon
