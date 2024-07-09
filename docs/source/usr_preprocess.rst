.. _PreprocessDocs:

====================
Preprocess utilities
====================

The ``preprocess`` module provides the basic preprocessing functionalities to compute and correct the sinogram data.

General preprocess functions
----------------------------

.. automodule:: mbirjax.preprocess
   :members: transmission_CT_compute_sino, estimate_background_offset, interpolate_defective_pixels, correct_det_rotation
   :undoc-members:
   :show-inheritance:

   .. rubric:: **Functions:**

   .. autosummary::

      transmission_CT_compute_sino
      estimate_background_offset
      interpolate_defective_pixels
      correct_det_rotation

NorthStart Instrument (NSI) functions
-------------------------------------

.. automodule:: mbirjax.preprocess_NSI
   :members: load_scans_and_params
   :undoc-members:
   :show-inheritance:

   .. rubric:: **Functions:**

   .. autosummary::

      load_scans_and_params
