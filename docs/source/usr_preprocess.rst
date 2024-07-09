.. _PreprocessDocs:

===============
preprocess
===============

The ``preprocess`` module provides the basic preprocessing functionalities to compute and correct the sinogram data.

General preprocess functions
----------------------------

.. automodule:: mbirjax.preprocess
   :members: transmission_CT_compute_sino, calc_background_offset, interpolate_defective_pixels, correct_det_rotation
   :undoc-members:
   :show-inheritance:

   .. rubric:: **Functions:**

   .. autosummary::

      transmission_CT_compute_sino
      calc_background_offset
      interpolate_defective_pixels
      correct_det_rotation

Functions specific to NorthStart Instrument (NSI) datasets
----------------------------------------------------------

.. automodule:: mbirjax.preprocess_NSI
   :members: load_scans_and_params
   :undoc-members:
   :show-inheritance:

   .. rubric:: **Functions:**

   .. autosummary::

      load_scans_and_params
