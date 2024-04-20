=================
Quick Start Guide
=================

Getting Started
~~~~~~~~~~~~~~~


The best way to start is to:
- **Install MBIRJAX:** using the instructions provided on the :ref:`Installation Page <installation_label>`.

- **Run Demo:** using the simple Python program entitled ``demo_3D_shepp_logan.py`` located in `[mbirjax/demo] <https://github.com/cabouman/mbirjax/tree/main/demo>`__ that make it easy to get started.

You can then modify the Python script to suit your needs.

Quick Start Guide
~~~~~~~~~~~~~~~~~

Below, is a quick-start-guide to using MBIRJAX for your application:

- **Get your data:**

  - Import your data as a 3D numpy array called ``sinogram`` that is organized by ``(views, detector rows, detector columns)``.

  - Create a 1D numpy array called ``angles`` that contains the rotation angle **in radians** of each view.

  - Convert arrays to JAX format using the commands ``sinogram = jnp.array(sinogram)`` and ``angle = jnp.array(angle)``.

  Note, that each row of sinogram data is assumed to be perpendicular to the rotation axis, and the views are assumed to be in conventional raster order (i.e., left-to-right, top-to-bottom) looking through the object from the source to the detector.


- **Initialize a model:**

  - Run ``model = mbirjax.parallel_beam.ParallelBeamModel(angles, sinogram.shape)`` to initialize a parallel beam model.

  You will then use the ``model`` object to perform the various reconstruction functions.


- **Reconstruct:**

  - Run ``recon = model.recon(sinogram)`` to reconstruct using MBIR.

  - Run ``recon_3d = parallel_model.reshape_recon(recon)`` to reshape your reconstruction into ``(rows, columns, slices)`` format.

Even the default parameter setting will usually produce a good quality reconstruction,


- **Set reconstruction parameters:**
  If you would like to tune image quality, you can set parameters using ``model.set_params('param_name=param_value')``.
  Here is a list of parameters you may want to set:

  - ``sharpness`` - the default value is 0. However, you can set ``sharpness=1.0`` or greater to increase sharpness, and you can set it negative to reduce noise.
  - ``snr_db`` - the default value is 30.0. Again a larger value will increase resolution, lower will decrease it, but we recommend you start with ``sharpness``.
  - ``verbose`` - the default value is 0, which will be quite. But if you want more feedback, set ``verbose=1`` or 2.

- **Set sinogram weights:**
  As you become more experienced, you may want to set the sinogram weights to improve image quality.
  You can set weights for common scenarios by:

  - Use the method ``weights = model.gen_weights(sinogram, weight_type='transmission_root')``.

  - Then use ``recon = model.recon(sinogram, weights=weights)`` to perform a weighted reconstruction.

  The weights array has the same shape as the sinogram, and it represents the assumed inverse noise variance for each sinogram entry.
  If you use the transmission options, it is critical that the sinogram be properly scaled to physically meaningful units, or you will get crazy results.

