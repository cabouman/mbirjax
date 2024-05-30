.. _TomographyModelDocs:


===============
TomographyModel
===============

The ``TomographyModel`` provides the basic interface for all specific geometries for tomographic projection
and reconstruction.

Constructor
-----------

.. autoclass:: mbirjax.TomographyModel
   :no-index:
   :show-inheritance:

Recon and Projection
--------------------

.. automethod:: mbirjax.TomographyModel.recon
   :no-index:

.. automethod:: mbirjax.TomographyModel.prox_map
   :no-index:

.. automethod:: mbirjax.TomographyModel.forward_project
   :no-index:

.. automethod:: mbirjax.TomographyModel.back_project
   :no-index:

Saving and Loading
------------------

.. automethod:: mbirjax.TomographyModel.to_file
   :no-index:

.. automethod:: mbirjax.TomographyModel.from_file
   :no-index:

Parameter Handling
------------------

.. automethod:: mbirjax.TomographyModel.set_params
   :no-index:

.. automethod:: mbirjax.ParameterHandler.get_params
   :no-index:

.. automethod:: mbirjax.ParameterHandler.print_params
   :no-index:


Data Generation
---------------

.. automethod:: mbirjax.TomographyModel.gen_weights
   :no-index:

.. automethod:: mbirjax.TomographyModel.gen_modified_3d_sl_phantom
   :no-index:


.. _detailed-parameter-docs:

Parameter Documentation
-----------------------

See the :ref:`Primary Parameters <ParametersDocs>` page.
