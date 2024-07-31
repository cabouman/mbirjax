.. _ParametersDocs:


===============
Base Parameters
===============

The following documents the base parameters used by the :ref:`TomographyModelDocs` class.
Any of these parameters can be modified with :func:`TomographyModel.set_params`.

Parameters that are specific to particular geometries are documented in the geometry's documentation.

Reconstruction Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^

sharpness : float
    Specifies the sharpness of the reconstruction. Defaults to 0.0. Larger values produce sharper images. Smaller values produce softer images.

snr_db : float
    Specifies the assumed signal-to-noise ratio in dB of the sinogram data. Defaults to 30.0. Larger values produce sharper and more edgy images. Smaller values produce softer and less edgy images.

verbose : int
    Larger values produce more status information. Defaults to 0 for silent operation.


Geometry Parameters
^^^^^^^^^^^^^^^^^^^

recon_shape : tuple (num_rows, num_cols, num_slices)
    Array size of reconstruction. This is set automatically and is available from :meth:`get_params('recon_shape')`.
    It is recommended to use :func:`set_params` to increase this by a factor of 10-15% when the object extends beyond the field of view.

delta_det_channel : float
    Spacing between detector channels in ALU. Defaults to 1.0.

delta_det_row : float
    Spacing between detector rows in ALU. Defaults to 1.0.

det_channel_offset : float
    Assumed offset between center of rotation and center of detector between detector channels in ALU. Defaults to 0.0.

det_row_offset : float
    Assumed offset in rows of the source-to-detector line with center of detector in ALU. Defaults to 0.0.

delta_voxel : float
    Spacing between voxels in ALU. Defaults to 1.0.


Proximal Map Parameters
^^^^^^^^^^^^^^^^^^^^^^^

sigma_y : float
    Assumed standard deviation of sinogram noise. Defaults to 1.0.

sigma_p : float
    Proximal map parameter. Defaults to 1.0.




