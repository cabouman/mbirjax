TomographyModel
===============

The ``TomographyModel`` provides the basic interface for all specific geometries for tomographic projection
and reconstruction.

.. autoclass:: mbirjax.TomographyModel
   :members:
        mbirjax.TomographyModel.forward_project
        mbirjax.TomographyModel.get_params
   :show-inheritance:

Projection and Recon
--------------------

.. automethod:: mbirjax.TomographyModel.forward_project

Parameter Handling
------------------

.. automethod:: mbirjax.TomographyModel.get_params

