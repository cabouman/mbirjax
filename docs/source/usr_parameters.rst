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
:Type: list (Defaults to [1.0, 1.0, 1.0])

This parameter controls the relative QGGMRF regularization strength along the row, column, and slice direction.
``qggmrf_nbr_wts = [1.0, 1.0, 1.0]`` corresponds to isotropic regularization.

.. _param-positivity_flag:

positivity_flag
"""""""""""""""
:Type: boolean (Defaults to False)

This parameter determines if positivity is enforced in MBIR reconstruction.

.. _param-verbose:

max_overrelaxation
""""""""""""""""""
:Type: float (Defaults to 1.5)

This parameter limits the step size of VCD updates.

.. _param-max_overrelaxation:

verbose
"""""""
:Type: int (Defaults to 1)

Larger values produce more status information. Change to 0 for silent operation or 2 or 3 for more detailed output.

.. _param-use_gpu:

use_gpu
"""""""
:Type: string (Defaults to 'automatic')

Possible values are 'automatic', 'full', 'sinograms', 'none'.

 * 'automatic' - recommended setting in which MBIRJAX determines the appropriate use of the GPU based on the available memory;
 * 'full' - performs entire reconstruction on the GPU;
 * 'sinograms' - uses the GPU for the sinogram storage and calculations but uses CPU memory for the reconstructions;
 * 'none' - disables GPU use.


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


alu_unit
""""""""
:Type: string (Defaults to None)

alu_value
"""""""""
:Type: float (Defaults to 1.0)

These two parameters are used to store the unit and value of 1 ALU.
So for example, if `alu_unit = "cm"` and `alu_value = 0.5`, then we know that `1 ALU = 0.5 cm`.
With this information, quantities such as the detector channel spacing can be converted from ALU to physical units.
These parameters are set by the preprocessing function for various devices,
so they pass back information that can be used to compute quantitative reconstructions.


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



