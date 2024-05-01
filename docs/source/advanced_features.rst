=================
Advanced Features
=================


Once you have your first reconstruction, you can adjust the reconstruction parameters to improve results and performance for your application.
Parameters can be set by either using ``model.set_params('param_name=param_value')`` or when the model is initialized.

Below are tips on important and useful features:

- **Tune Image Quality:**

  You can tune image quality by setting the following parameters:

  - ``sharpness`` -  default = 0. A larger value of ``sharpness=1.0`` or greater will increase sharpness, and a negative value will reduce noise.
  - ``snr_db`` - default = 30.0. A larger value will increase resolution, but start with ``sharpness``.


- **Set Sinogram Weights:**

  As you become more experienced, you may want to set the sinogram weights to improve image quality.
  You can set weights for common scenarios by:

  - ``weights = model.gen_weights(sinogram, weight_type='transmission_root')`` can be used to generate a weight transmission weight array.
  - ``recon = model.recon(sinogram, weights=weights)`` can then be used to apply the desired sinogram weights.

  The weights array has the same shape as the sinogram, and it represents the assumed inverse noise variance for each sinogram entry.
  If you use the transmission options, it is critical that the sinogram be properly scaled to -log attenuation units, or you will get crazy results.

- **Change Reconstruction Size and Shape:**

  MBIRJAX will automatically set the reconstruction array size and voxel pitch to reasonable values.
  However, if you are doing tilt beam reconstruction or other specialized geometries, you may want to reconstruct a rectangular region-of-interest.
  You can do this by setting the following parameters:

  - ``num_recon_rows`` and  ``num_recon_cols`` -  These will default to the number of detector channels, but you can change their values to reconstruct a rectangular region.
  - ``delta_pixel_recon`` - This defaults to 1.0 and sets the spacing between voxels in ALU. (See :ref:`ALU conversion <ALU_conversion_label>`).
  - ``delta_det_channel`` and ``delta_det_row`` - These default to 1.0 and set the spacing between detector channels and rows in ALU. (See :ref:`ALU conversion <ALU_conversion_label>`).

- **Calibrate and Control Model:**

  MBIRJAX will allow you to compensate for non-ideal data with the following parameters:

  - ``det_channel_offset`` - This defaults to 0.0 and sets offset in the center of rotation in ALU. (See :ref:`ALU conversion <ALU_conversion_label>`).
  - ``verbose`` - default = 0 will be quiet. Set ``verbose=1`` or 2 for more feedback.
  - ``model.print_params()`` - This method will print out all the parameters in the model so you can monitor what's happening.

- **Manage Memory:**

  Large reconstruction can exceed the available memory, particular on GPUs.
  So MBIRJAX provides some parameters for processing data in manageable batches:

  - ``view_batch_size`` - This defaults to None and sets the maximum number of views that are processed together.
  - ``voxel_batch_size`` - This defaults to None and sets the maximum number of voxels that are processed together.

  If you are running out of memory, we recommend that you set these parameters to smaller values.
  Good starting points are ``view_batch_size=100`` and ``view_batch_size=10000``


