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

Recon and Projection
--------------------

.. automethod:: mbirjax.TomographyModel.recon

.. automethod:: mbirjax.TomographyModel.prox_map

.. automethod:: mbirjax.TomographyModel.forward_project

.. automethod:: mbirjax.TomographyModel.back_project

Saving and Loading
------------------

.. automethod:: mbirjax.TomographyModel.to_file

.. automethod:: mbirjax.TomographyModel.from_file

Parameter Handling
------------------

.. automethod:: mbirjax.TomographyModel.set_params

.. automethod:: mbirjax.ParameterHandler.get_params

.. automethod:: mbirjax.ParameterHandler.print_params


Data Generation
---------------

.. automethod:: mbirjax.TomographyModel.gen_weights

.. automethod:: mbirjax.TomographyModel.gen_modified_3d_sl_phantom


.. _detailed-parameter-docs:

Parameter Documentation
-----------------------

See the :ref:`Primary Parameters <ParametersDocs>` page.
