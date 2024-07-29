===========
Quick Start
===========

Demos
~~~~~


The best way to start is to:

- **Install MBIRJAX** using the instructions provided on the :ref:`Installation Page <InstallationDocs>`.  We recommend installing in a conda environment or pip virtual environment.
- **View and run demos** either as `Jupyter notebooks <https://drive.google.com/drive/folders/1wVEsYtneTI83ZB8q-Ag4qk2gPxi_UfKA?usp=sharing>`__ or `python scripts <https://github.com/cabouman/mbirjax/tree/main/demo>`__

You can then adapt these demos to suit your needs.


Quick Start
~~~~~~~~~~~

Below are simple instructions on how to do your first reconstruction:

- **Get your data:**

  - Import your ``sinogram`` data as a 3D numpy array organized by ``(views, detector rows, detector channels (columns))``.
  - Create a 1D numpy array called ``angles`` that contains the rotation angle **in radians** of each view.

  Note that each row of sinogram data is assumed to be perpendicular to the rotation axis and each view is assumed to be in conventional raster order (i.e., left-to-right, top-to-bottom) looking through the object from the source to the detector.


- **Initialize a model:**

  - Run ``model = mbirjax.ParallelBeamModel(sinogram.shape, angles)`` to initialize a parallel beam model.

  You will then use the ``model`` object to perform the various reconstruction functions.


- **Reconstruct and visualize:**

  - Run ``recon, recon_params = model.recon(sinogram)`` to reconstruct a volume in  ``(rows, columns, slices)`` format using MBIR.
  - Call ``mbirjax.slice_viewer(recon, title='MBIRJAX reconstruction')``

Even the default parameter settings will usually produce a good quality reconstruction.

