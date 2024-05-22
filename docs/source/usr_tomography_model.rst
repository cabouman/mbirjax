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
The following documents basic TomographyModel class parameters that are commonly used in reconstruction.
Other parameters may be used for specific geometries.

Basic Reconstruction Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

sharpness : float
    Specifies the sharpness of the reconstruction. Defaults to 0.0. Larger values produce sharper images. Smaller values produce softer images.

snr_db : float
    Specifies the assumed SNR of sinogram measurements in dB. Defaults to 30.0. Larger values produce sharper images.

verbose : int
    Larger values produce more status information. Defaults to 0 for silent operation.


Basic Geometry Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^

delta_det_channel : float
    Spacing between detector channels in ALU. Defaults to 1.0.

delta_det_row : float
    Spacing between detector rows in ALU. Defaults to 1.0.

det_channel_offset : float
    Assumed offset between center of rotation and center of detector between detector channels in ALU. Defaults to 0.0.

magnification : float
    Ratio of (source to detector distance)/(source to iso distance). Defaults to 1.0.

delta_voxel : float
    Spacing between voxels in ALU. Defaults to 1.0.


Proximal Map Parameters
^^^^^^^^^^^^^^^^^^^^^^^

sigma_y : float
    Assumed standard deviation of sinogram noise. Defaults to 1.0.

sigma_p : float
    Proximal map parameter. Defaults to 1.0.


Memory Allocation Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

pixel_batch_size : int
    Maximum number of pixels (i.e., voxel cylinders) processed simultaneously.

view_batch_size : int
    Maximum number of views processed simultaneously.




