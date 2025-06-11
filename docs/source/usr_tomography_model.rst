.. _TomographyModelDocs:


===============
TomographyModel
===============

The ``TomographyModel`` provides the basic interface for all specific geometries for tomographic projection
and reconstruction.

Constructor
-----------

.. autoclass:: mbirjax.TomographyModel
   :show-inheritance:

Reconstruction and Projection
-----------------------------

.. automethod:: mbirjax.TomographyModel.recon

.. automethod:: mbirjax.TomographyModel.direct_recon

.. automethod:: mbirjax.TomographyModel.scale_recon_shape

.. automethod:: mbirjax.TomographyModel.prox_map

.. automethod:: mbirjax.TomographyModel.forward_project

.. automethod:: mbirjax.TomographyModel.back_project

Weights and Metal Artifact Reduction
------------------------------------

.. automethod:: mbirjax.TomographyModel.gen_weights

.. automethod:: mbirjax.TomographyModel.gen_weights_mar

Saving and Loading
------------------

.. automethod:: mbirjax.TomographyModel.to_file

.. automethod:: mbirjax.TomographyModel.from_file

.. automethod:: mbirjax.TomographyModel.save_recon_hdf5

Parameter Handling
------------------

.. automethod:: mbirjax.TomographyModel.set_params

.. automethod:: mbirjax.ParameterHandler.get_params

.. automethod:: mbirjax.ParameterHandler.print_params


Data Generation
---------------

.. automethod:: mbirjax.TomographyModel.gen_modified_3d_sl_phantom


.. _detailed-parameter-docs:

Parameter Documentation
-----------------------

See the :ref:`Primary Parameters <ParametersDocs>` page.
