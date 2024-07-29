=================
Advanced Features
=================


Once you have your first reconstruction, you can adjust the reconstruction parameters to improve results and performance for your application.
Parameters can be set by either using ``model.set_params(param_name=param_value)``.

Below are tips on important and useful features:

- **Tune Image Quality:**

  You can tune image quality by setting the following parameters:

  - ``sharpness`` -  default = 0.0. A larger value of ``sharpness=1.0`` or greater will increase sharpness, and a negative value will reduce noise.  Any float is allowable, but anything outside [-5, 5] is probably not helpful.
  - ``snr_db`` - default = 30.0. A larger value will increase resolution, but try changing ``sharpness`` first.

- **Change Reconstruction Size and Shape:**

  MBIRJAX will automatically set the reconstruction array size and voxel pitch to reasonable values.
  However, if you are doing tilt beam reconstruction or other specialized geometries, you may want to reconstruct a rectangular region-of-interest.
  You can do this by setting the following parameters:

  - ``recon_shape`` -  The recon shape is a 3-tuple (num_rows, num_cols, num_slices). It defaults to something reasonable, but you can change its value to change the region of reconstruction.
  - ``delta_voxel`` - This defaults to 1.0 and sets the spacing between voxels in ALU in x, y, and z directions. (See :ref:`ALU conversion <ALU_conversion_label>`).
  - ``delta_det_channel`` and ``delta_det_row`` - These default to 1.0 and set the spacing between detector channels and rows in ALU. (See :ref:`ALU conversion <ALU_conversion_label>`).

- **Set Sinogram Weights:**

  As you become more experienced, you may want to set the sinogram weights to improve image quality.
  You can set weights for common scenarios by:

  - ``weights = model.gen_weights(sinogram, weight_type='transmission_root')`` can be used to generate a weight transmission weight array.
  - ``recon = model.recon(sinogram, weights=weights)`` can then be used to apply the desired sinogram weights.

  The weights array has the same shape as the sinogram, and it represents the assumed inverse noise variance for each sinogram entry.
  If you use the transmission options, it is critical that the sinogram be properly scaled to -log attenuation units, or you will get crazy results.

  There is also a method to generate weights to reduce metal artifacts, particularly for objects with some dense metal components and other components with much less attenuation:

  - ``weights = model.gen_weights_mar(sinogram, init_recon=None)``

  Supplying an initial recon gives a better estimate of the metal decomposition, but this is optional.

- **Calibrate and Control Model:**

  MBIRJAX will allow you to compensate for non-ideal data with the following parameters:

  - ``det_channel_offset`` - This defaults to 0.0 and sets offset in the center of rotation in ALU. (See :ref:`ALU conversion <ALU_conversion_label>`).
  - ``verbose`` - default = 1 to print out basic information. Set ``verbose=0`` for quiet or 2 or 3 for more feedback.
  - ``model.print_params()`` - This method will print out all the parameters in the model so you can monitor what's happening.



