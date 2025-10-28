.. _ParametersDocs:

===============
Base Parameters
===============

The following documents the base parameters used by the :ref:`TomographyModelDocs` class.
Any of these parameters can be modified with :func:`TomographyModel.set_params`.

Note that the default detector channel spacing is `delta_det_channel = 1 ALU`, and the voxel spacing is automatically
set to `delta_voxel = 1/magnifaction ALU` where `magnification` is the magnification of a voxel at iso.

Parameters that are specific to particular geometries are documented in the geometry's documentation.

Reconstruction Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^

.. _param-sharpness:

sharpness
"""""""""
:Type: float (Defaults to 1.0)

Specifies the sharpness of the reconstruction. Larger values produce sharper images. Smaller values produce softer images.
(For advanced users: This actually controls the underlying parameter ``sigma_x``.)

.. _param-snr_db:

snr_db
""""""
:Type: float (Defaults to 30.0)

Specifies the assumed signal-to-noise ratio in dB of the sinogram data. Larger values produce sharper and more edgy images.
Smaller values produce softer and less edgy images.
(For advanced users: This parameter actually controls the underlying parameter ``sigma_y``.)

.. _param-qggmrf_nbr_wts:

qggmrf_nbr_wts
""""""""""""""
:Type: list (num_rows, num_cols, num_slices)

This parameter controls the relative QGGMRF regularization strength along the row, column, and slice direction.
``qggmrf_nbr_wts = [1.0, 1.0, 1.0]`` corresponds to isotropic regularization.

.. _param-verbose:

verbose
"""""""
:Type: int (Defaults to 1)

Larger values produce more status information. Change to 0 for silent operation or 2 or 3 for more detailed output.

.. _param-use_gpu:

use_gpu
"""""""
:Type: string (Defaults to 'automatic')

Possible values are 'automatic', 'full', 'sinograms', 'worker', 'none'. 'full' tries to perform the entire reconstruction on the GPU;
'sinograms' uses the GPU for all of the sinogram storage and calculations but uses CPU memory for the reconstructions;
'worker' uses the CPU for some computations and the GPU for projections only (to conserve memory);
'automatic' tries to determine the appropriate choice based on available memory; 'none' disables GPU use.


Geometry Parameters
^^^^^^^^^^^^^^^^^^^

.. _param-recon_shape:

recon_shape
"""""""""""
:Type: tuple (num_rows, num_cols, num_slices)

Array size of reconstruction. This is set automatically and is available from :meth:`get_params('recon_shape')`.
It is recommended to use :func:`scale_recon_shape` to increase this by a factor of 10–15% when the object extends beyond the field of view.

.. _param-delta_det_channel:

delta_det_channel
"""""""""""""""""
:Type: float (Defaults to 1.0)

Spacing between detector channels in ALU.

.. _param-delta_det_row:

delta_det_row
"""""""""""""
:Type: float (Defaults to 1.0)

Spacing between detector rows in ALU.

.. _param-det_channel_offset:

det_channel_offset
""""""""""""""""""
:Type: float (Defaults to 0.0)

Assumed offset between center of rotation and center of detector between detector channels in ALU.

.. _param-det_row_offset:

det_row_offset
"""""""""""""""
:Type: float (Defaults to 0.0)

Assumed offset in rows of the source-to-detector line with center of detector in ALU.

.. _param-delta_voxel:

delta_voxel
"""""""""""
:Type: float (Defaults to None)

Spacing between voxels in ALU.
If None, then it is automatically set to `delta_voxel = delta_det_channel / magnification` where `magnification` is the
magnification of a voxel at iso determined from the function :func:`TomographyModel.get_magnification()`.

Proximal Map Parameters
^^^^^^^^^^^^^^^^^^^^^^^

.. _param-sigma_y:

sigma_y
"""""""
:Type: float (Defaults to 1.0)

Assumed standard deviation of sinogram noise.

.. _param-sigma_p:

sigma_prox
""""""""""
:Type: float (Defaults to 1.0)

Proximal map parameter.



