.. _PreprocessDocs:

====================
Preprocess utilities
====================

The ``preprocess`` module provides the basic preprocessing functionalities to compute and correct the sinogram data.

General preprocess functions
----------------------------

.. automodule:: mbirjax.preprocess
   :members: compute_sino_transmission, estimate_background_offset, interpolate_defective_pixels, correct_det_rotation
   :undoc-members:
   :show-inheritance:

   .. rubric:: **Functions:**

   .. autosummary::

      compute_sino_transmission
      estimate_background_offset
      interpolate_defective_pixels
      correct_det_rotation

NorthStar Instrument (NSI) functions
------------------------------------

.. automodule:: mbirjax.preprocess.NSI
   :members: load_scans_and_params
   :undoc-members:
   :show-inheritance:

   .. rubric:: **Functions:**

   .. autosummary::

      load_scans_and_params
