.. _PreprocessDocs:

====================
Preprocess utilities
====================

The ``preprocess`` module provides scanner-specific preprocessing and more general preprocessing to compute and correct the sinogram data.
See `demo_nsi.py <https://github.com/cabouman/mbirjax_applications/tree/main/nsi>`__ in the
`mbirjax_applications <https://github.com/cabouman/mbirjax_applications>`__ repo for example uses.

NorthStar Instrument (NSI) functions
------------------------------------

.. automodule:: mbirjax.preprocess.nsi
   :members: compute_sino_and_params, load_scans_and_params
   :undoc-members:
   :show-inheritance:

   .. rubric:: **Functions:**

   .. autosummary::

      compute_sino_and_params
      load_scans_and_params

General preprocess functions
----------------------------

.. automodule:: mbirjax.preprocess
   :members: compute_sino_transmission, estimate_background_offset, interpolate_defective_pixels, correct_det_rotation, multi_threshold_otsu, export_recon_to_hdf5
   :undoc-members:
   :show-inheritance:

   .. rubric:: **Functions:**

   .. autosummary::

      compute_sino_transmission
      estimate_background_offset
      interpolate_defective_pixels
      correct_det_rotation
      multi_threshold_otsu
      export_recon_to_hdf5
