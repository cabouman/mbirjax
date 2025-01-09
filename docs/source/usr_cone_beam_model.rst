.. _ConeBeamModelDocs:

=============
ConeBeamModel
=============

The ``ConeBeamModel`` extends the functionalities provided by :ref:`TomographyModelDocs`.
This class inherits all behaviors and attributes of the TomographyModel and implements projectors specific
to cone beam CT.

In addition, ``ConeBeamModel`` includes FDK reconstruction as indicated below.

Constructor
-----------

.. autoclass:: mbirjax.ConeBeamModel
   :show-inheritance:

FDK Reconstruction
------------------

.. automethod:: mbirjax.ConeBeamModel.fdk_recon

Parent Class
------------
:ref:`TomographyModelDocs`

