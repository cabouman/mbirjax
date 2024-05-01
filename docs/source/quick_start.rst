===========
Quick Start
===========

Demos
~~~~~


The best way to start is to:

- **Install MBIRJAX** in a conda environment and activate this environment using the instructions provided on the :ref:`Installation Page <installation_label>`.

- **Run Demo** using the Python program entitled ``demo_3D_shepp_logan.py`` located in `[mbirjax/demo] <https://github.com/cabouman/mbirjax/tree/main/demo>`__.

You can then modify the Python script to suit your needs.


Quick Start
~~~~~~~~~~~

Below are simple instructions on how to do your first reconstruction:

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

  Even the default parameter setting will usually produce a good quality reconstruction.

