.. _PreprocessDocs:

=============
Preprocessing
=============

The ``preprocess`` module provides scanner-specific preprocessing and more general preprocessing to compute and correct the sinogram data.
See `demo_nsi.py <https://github.com/cabouman/mbirjax_applications/tree/main/nsi>`__ in the
`mbirjax_applications <https://github.com/cabouman/mbirjax_applications>`__ repo for example uses.

One-Call Preprocessing
----------------------

Each supported scanner allows one-call preprocessing with the scanner's ``get_sino_and_model`` function, which loads a scan, computes its sinogram, and returns a ready-to-reconstruct model.

.. code-block:: python

    sino, model = mbirjax.preprocess.nsi.get_sino_and_model(dataset_dir)
    weights = mbirjax.gen_weights(sino, weight_type='transmission_root')
    recon, recon_dict = model.recon(sino, weights=weights)

The call selects the correct geometry class for the scanner (for example, the Zeiss reader picks
``ParallelBeamModel`` for an Ultra scan and ``ConeBeamModel`` for a Versa scan) and computes the
reconstruction geometry from the real detector parameters, so the returned model is ready to be used.
Reconstruction weights can be generated with :func:`mbirjax.gen_weights`.


NorthStar Instrument (NSI) reader
---------------------------------

.. currentmodule:: mbirjax.preprocess.nsi

.. autofunction:: get_sino_and_model
.. autofunction:: load_scans_and_params


Zeiss Versa and Ultra reader
----------------------------

.. currentmodule:: mbirjax.preprocess.zeiss

.. autofunction:: get_sino_and_model
.. autofunction:: load_scans_and_params


Zeiss translation tomography functions
--------------------------------------

.. currentmodule:: mbirjax.preprocess.zeiss_tct

.. autofunction:: get_sino_and_model
.. autofunction:: compute_weight
.. autofunction:: load_scans_and_params


PYMBIR functions
----------------

.. currentmodule:: mbirjax.preprocess.pymbir

.. autofunction:: get_sino_and_model


General preprocess functions
----------------------------

.. currentmodule:: mbirjax.preprocess

.. autofunction:: compute_sino_transmission
.. autofunction:: detect_blank_margins
.. autofunction:: apply_detector_crop
.. autofunction:: align_sino_views
.. autofunction:: interpolate_defective_pixels
.. autofunction:: correct_det_rotation
.. autofunction:: correct_background_offset
.. autofunction:: downsample_view_data
.. autofunction:: crop_view_data
.. autofunction:: apply_cylindrical_mask
.. autofunction:: save_cone_preprocessing
.. autofunction:: load_cone_preprocessing
.. autofunction:: read_tif_stack_dir
.. autofunction:: read_tif_img


MAR utilities
-------------

.. currentmodule:: mbirjax.preprocess

.. autofunction:: gen_huber_weights
.. autofunction:: recon_plastic_metal
.. autofunction:: BH_correction
.. autofunction:: fit_beam_hardening_curve
.. autofunction:: fit_inverse_beam_hardening_curve
.. autofunction:: apply_beam_hardening_curve
.. autofunction:: apply_inverse_beam_hardening_curve

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

