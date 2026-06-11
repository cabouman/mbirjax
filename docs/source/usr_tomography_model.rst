.. _TomographyModelDocs:


================
Tomography Model
================

The ``TomographyModel`` provides the basic interface for all specific geometries for tomographic projection
and reconstruction.

Constructor
-----------

.. autoclass:: mbirjax.TomographyModel
   :show-inheritance:


Device Configuration
--------------------

On a machine with multiple GPUs, reconstruction is divided across them automatically
(currently for parallel-beam geometry) to increase the available memory and reduce
reconstruction time; these methods give explicit control and report the outcome.

.. automethod:: mbirjax.TomographyModel.configure_devices

.. automethod:: mbirjax.TomographyModel.prepare_sino_for_devices

.. autoproperty:: mbirjax.TomographyModel.device_summary


Reconstruction and Projection
-----------------------------

.. automethod:: mbirjax.TomographyModel.recon

.. automethod:: mbirjax.TomographyModel.direct_recon

.. automethod:: mbirjax.TomographyModel.prox_map

.. automethod:: mbirjax.TomographyModel.forward_project

.. automethod:: mbirjax.TomographyModel.back_project


Parameter Handling
------------------

.. automethod:: mbirjax.TomographyModel.set_params

.. automethod:: mbirjax.ParameterHandler.get_params

.. automethod:: mbirjax.ParameterHandler.print_params

.. automethod:: mbirjax.TomographyModel.get_recon_dict


Recon Shape and Voxel Spacing
-----------------------------

.. automethod:: mbirjax.TomographyModel.auto_set_recon_geometry

.. automethod:: mbirjax.TomographyModel.scale_recon_shape

.. automethod:: mbirjax.TomographyModel.get_magnification


.. _SaveLoadDocs:

Saving and Loading
------------------

.. automethod:: mbirjax.TomographyModel.save_recon_hdf5

.. automethod:: mbirjax.TomographyModel.load_recon_hdf5


Data Generation
---------------

.. automethod:: mbirjax.TomographyModel.gen_modified_3d_sl_phantom


.. _detailed-parameter-docs:

Parameter Documentation
-----------------------

See the :ref:`Primary Parameters <ParametersDocs>` page.
