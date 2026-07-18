.. _TomographyModelDocs:


================
Tomography Model
================

The ``TomographyModel`` provides the basic interface for all specific geometries for tomographic projection
and reconstruction.

Constructor
-----------

.. autoclass:: mbirjax.TomographyModel


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

.. automethod:: mbirjax.TomographyModel.get_all_params

.. automethod:: mbirjax.TomographyModel.get_recon_dict


Recon Shape and Voxel Spacing
-----------------------------

.. automethod:: mbirjax.TomographyModel.auto_set_recon_geometry

.. automethod:: mbirjax.TomographyModel.scale_recon_shape

.. automethod:: mbirjax.TomographyModel.get_magnification


Device Configuration
--------------------

On a machine with multiple GPUs, MBIRJAX automatically divides a reconstruction across them to
increase the available memory and reduce reconstruction time -- with no change to your script,
and for every geometry.  The methods below give explicit control over which devices are used
and report what was chosen.  See :doc:`usr_multi_gpu` for a full discussion.

.. automethod:: mbirjax.TomographyModel.configure_devices

.. automethod:: mbirjax.TomographyModel.prepare_sino_for_devices

.. autoproperty:: mbirjax.TomographyModel.device_summary


.. _SaveLoadDocs:

Saving and Loading
------------------

.. automethod:: mbirjax.TomographyModel.save_recon_hdf5

.. automethod:: mbirjax.TomographyModel.load_recon_hdf5


.. _detailed-parameter-docs:

Parameter Documentation
-----------------------

See the :ref:`Primary Parameters <ParametersDocs>` page.
