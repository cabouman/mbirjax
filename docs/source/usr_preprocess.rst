.. _PreprocessDocs:

=============
Preprocessing
=============

The ``preprocess`` module provides scanner-specific preprocessing and more general preprocessing to compute and correct the sinogram data.
See `demo_nsi.py <https://github.com/cabouman/mbirjax_applications/tree/main/nsi>`__ in the
`mbirjax_applications <https://github.com/cabouman/mbirjax_applications>`__ repo for example uses.

NorthStar Instrument (NSI) functions
------------------------------------

.. currentmodule:: mbirjax.preprocess.nsi

.. autofunction:: compute_sino_and_params
.. autofunction:: load_scans_and_params

PYMBIR functions
----------------

.. currentmodule:: mbirjax.preprocess.pymbir

.. autofunction:: compute_sino_and_params

General preprocess functions
----------------------------

.. currentmodule:: mbirjax.preprocess

.. autofunction:: compute_sino_transmission
.. autofunction:: interpolate_defective_pixels
.. autofunction:: correct_det_rotation_and_background
.. autofunction:: estimate_background_offset
.. autofunction:: downsample_view_data
.. autofunction:: crop_view_data
.. autofunction:: apply_cylindrical_mask
.. autofunction:: read_scan_img

MAR utilities
-------------

.. currentmodule:: mbirjax.preprocess

.. autofunction:: gen_huber_weights
.. autofunction:: BH_correction
.. autofunction:: recon_BH_plastic_metal


Stripe/Ring/Offset Removal
--------------------------

.. currentmodule:: mbirjax.preprocess

.. autofunction:: remove_all_stripe
.. autofunction:: remove_stripe_fw
.. autofunction:: remove_sino_offset


Segmentation functions
----------------------

.. currentmodule:: mbirjax.preprocess

.. autofunction:: multi_threshold_otsu
.. autofunction:: segment_plastic_metal

View selection (VCLS) functions
-------------------------------

.. currentmodule:: mbirjax.preprocess

.. autofunction:: get_opt_views
.. autofunction:: show_image_with_projection_rays