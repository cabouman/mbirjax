.. _ParametersDocs:


==================
Primary Parameters
==================

The ``TomographyModel`` provides the basic interface for all specific geometries for tomographic projection
and reconstruction.

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




