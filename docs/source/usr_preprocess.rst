.. _PreprocessDocs:

=============
Preprocessing
=============

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
   :members: compute_sino_transmission, interpolate_defective_pixels, correct_det_rotation_and_background, estimate_background_offset, downsample_view_data, crop_view_data
   :undoc-members:
   :show-inheritance:

   .. rubric:: **Functions:**

   .. autosummary::

      compute_sino_transmission
      interpolate_defective_pixels
      correct_det_rotation_and_background
      estimate_background_offset
      downsample_view_data
      crop_view_data

MAR preprocess functions
------------------------

.. automodule:: mbirjax.preprocess
   :members: multi_threshold_otsu, gen_huber_weights, BH_correction, segment_plastic_metal, get_opt_views, copy_ct_model
   :no-index:
   :undoc-members:
   :show-inheritance:

   .. rubric:: **Functions:**

.. autofunction::
   mbirjax.preprocess.multi_threshold_otsu

.. autofunction::
   mbirjax.preprocess.gen_huber_weights

.. autofunction::
   mbirjax.preprocess.BH_correction

.. autofunction::
   mbirjax.preprocess.segment_plastic_metal

View selection (VCLS) preprocess functions
------------------------------------------

.. autofunction::
   mbirjax.preprocess.get_opt_views

.. autofunction::
   mbirjax.preprocess.copy_ct_model


