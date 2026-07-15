.. _PreprocessDocs:

=============
Preprocessing
=============

The ``preprocess`` module provides scanner-specific preprocessing and more general preprocessing to compute and correct the sinogram data.
See `demo_nsi.py <https://github.com/cabouman/mbirjax_applications/tree/main/nsi>`__ in the
`mbirjax_applications <https://github.com/cabouman/mbirjax_applications>`__ repo for example uses.

The one-call reader API
-----------------------

Each scanner reader exposes a single ``get_sino_and_model`` that loads a scan, computes its sinogram, and
returns a ready-to-reconstruct model::

    sino, model = mbirjax.preprocess.nsi.get_sino_and_model(dataset_dir)
    weights = mbirjax.gen_weights(sino, weight_type='transmission_root')
    recon, recon_dict = model.recon(sino, weights=weights)

The call selects the correct geometry class for the scanner (for example, the Zeiss reader picks
``ParallelBeamModel`` for an Ultra scan and ``ConeBeamModel`` for a Versa scan) and computes the
reconstruction geometry from the real detector parameters, so the returned model is always ready to
reconstruct -- there is no separate ``construct -> set_params -> auto_set_recon_geometry`` sequence to get
wrong.

Reconstruction weights are not returned: generate transmission weights with ``mbirjax.gen_weights``.  The
Zeiss translation-tomography reader is the exception -- it returns a data-specific ``weights`` mask
(from :func:`~mbirjax.preprocess.zeiss_tct.compute_weight`) alongside the model::

    sino, model, weights = mbirjax.preprocess.zeiss_tct.get_sino_and_model(dataset_dir)
    recon, recon_dict = model.recon(sino, weights=weights)

Pass ``auto_crop=True`` to the NSI, PYMBIR, or Zeiss reader (every reader except the
translation-tomography reader) to detect and remove blank sinogram margins before building the model,
shrinking the reconstruction volume.

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
.. autofunction:: save_preprocessing
.. autofunction:: load_preprocessing
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

