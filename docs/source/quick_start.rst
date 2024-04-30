===========
Quick Start
===========

Demos
~~~~~


The best way to start is to:

- **Install MBIRJAX** in a conda environment and activate this environment using the instructions provided on the :ref:`Installation Page <installation_label>`.

- **Run Demo** using the Python program entitled ``demo_3D_shepp_logan.py`` located in `[mbirjax/demo] <https://github.com/cabouman/mbirjax/tree/main/demo>`__.

You can then modify the Python script to suit your needs.

Quick Start Guide
~~~~~~~~~~~~~~~~~

Below is a quick-start-guide to using MBIRJAX for your application:

- **Get your data:**

  - Import your ``sinogram`` data as a 3D numpy array organized by ``(views, detector rows, detector columns)``.

  - Create a 1D numpy array called ``angles`` that contains the rotation angle **in radians** of each view.

  - (Optional) Convert arrays both arrays to JAX format using the commands ``array = jnp.array(array)``.

  Note that each row of sinogram data is assumed to be perpendicular to the rotation axis and each view is assumed to be in conventional raster order (i.e., left-to-right, top-to-bottom) looking through the object from the source to the detector.


- **Initialize a model:**

  - Run ``model = mbirjax.parallel_beam.ParallelBeamModel(angles, sinogram.shape)`` to initialize a parallel beam model.

  You will then use the ``model`` object to perform the various reconstruction functions.


- **Reconstruct:**

  - Run ``recon = model.recon(sinogram)`` to reconstruct using MBIR.

  - Then run ``recon_3d = parallel_model.reshape_recon(recon)`` to reshape your reconstruction into ``(rows, columns, slices)`` format.

  Even the default parameter setting will usually produce a good quality reconstruction,


- **Tune image quality:**

  You can tune image quality by setting the following parameters:

  - ``sharpness`` -  default = 0. A larger value of ``sharpness=1.0`` or greater will increase sharpness, and a negative value will reduce noise.
  - ``snr_db`` - default = 30.0. A larger value will increase resolution, but we recommend you start with ``sharpness``.
  - ``verbose`` - default = 0 will be quiet. Set ``verbose=1`` or 2 for more feedback.

  Parameters can be set by either using ``model.set_params('param_name=param_value')`` or when the model is initialized.

- **Set sinogram weights:**

  As you become more experienced, you may want to set the sinogram weights to improve image quality.
  You can set weights for common scenarios by:

  - Generate a weight array using ``weights = model.gen_weights(sinogram, weight_type='transmission_root')``.
  - Then reconstruct using ``recon = model.recon(sinogram, weights=weights)``.

  The weights array has the same shape as the sinogram, and it represents the assumed inverse noise variance for each sinogram entry.
  If you use the transmission options, it is critical that the sinogram be properly scaled to -log attenuation units, or you will get crazy results.

- **Change default reconstruction size and shape:**

  MBIRJAX will automatically set the reconstruction array size and voxel pitch to reasonable values.
  However, if you are doing tilt beam reconstruction or other specialized geometries, you may want to reconstruct a rectangular region-of-interest.
  You can do this by setting the following parameters:

  - ``num_recon_rows`` and  ``num_recon_cols`` -  These will default to the number of detector channels, but you can change their values to reconstruct a rectangular region.
  - ``delta_pixel_recon`` - This defaults to 1.0 and sets the spacing between voxels in ALU. (See :ref:`ALU conversion <ALU_conversion_label>`).
  - ``delta_det_channel`` and ``delta_det_row`` - These default to 1.0 and set the spacing between detector channels and rows in ALU. (See :ref:`ALU conversion <ALU_conversion_label>`).

- **Calibrate model:**

  MBIRJAX will allow you to compensate for non-ideal data with the following parameters:

  - ``det_channel_offset`` - This defaults to 0.0 and sets offset in the center of rotation in ALU. (See :ref:`ALU conversion <ALU_conversion_label>`).


