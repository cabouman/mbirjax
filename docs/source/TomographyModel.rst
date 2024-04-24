TomographyModel
===============

The ``TomographyModel`` provides the basic interface for all specific geometries for tomographic projection
and reconstruction.

.. autoclass:: mbirjax.TomographyModel
   :members:
        mbirjax.TomographyModel.recon
        mbirjax.TomographyModel.forward_project
        mbirjax.TomographyModel.back_project
        mbirjax.TomographyModel.reshape_recon
        mbirjax.TomographyModel.set_params
        mbirjax.TomographyModel.get_params
        mbirjax.TomographyModel.gen_weights
        mbirjax.TomographyModel.gen_3d_sl_phantom
   :show-inheritance:

Projection and Recon
--------------------

.. automethod:: mbirjax.TomographyModel.recon
.. automethod:: mbirjax.TomographyModel.forward_project
.. automethod:: mbirjax.TomographyModel.back_project
.. automethod:: mbirjax.TomographyModel.reshape_recon

Parameter Handling
------------------

.. automethod:: mbirjax.TomographyModel.set_params
.. automethod:: mbirjax.TomographyModel.get_params

Data Generation
---------------

.. automethod:: mbirjax.TomographyModel.gen_weights
.. automethod:: mbirjax.TomographyModel.gen_3d_sl_phantom
