Developer API reference
=======================

**MBIRJAX** can be extended with new scanner geometries by subclassing
:ref:`TomographyModelDocs`.  The best reference is the existing geometry classes --
:class:`~mbirjax.ParallelBeamModel`, :class:`~mbirjax.ConeBeamModel`,
:class:`~mbirjax.TranslationModel`, and :class:`~mbirjax.MultiAxisParallelModel`.
They are the canonical, tested templates; this page summarizes what a new geometry
must provide.

Core projector interface
------------------------

A geometry supplies two ``jax.jit``-compilable per-view projector kernels plus a small
amount of geometry plumbing:

* ``forward_project_pixel_batch_to_one_view`` -- project a batch of voxel cylinders into one view.
* ``back_project_one_view_to_pixel_batch`` -- back-project one view onto a batch of voxel cylinders.
* ``get_geometry_parameters`` -- the view-independent parameters the kernels need, as a
  jit-friendly namedtuple.
* ``verify_valid_params``, ``get_magnification``, and ``auto_set_recon_geometry`` -- parameter
  validation, the iso-to-detector scale factor, and the default reconstruction geometry.

The skeleton at the bottom of this page shows these signatures.

Multi-device (sharded) support
------------------------------

Reconstructions are spread across the available devices automatically -- the recon by slice and
the sinogram by view (see :doc:`dev_sharding_overview` for the architecture).  Most of the
machinery is inherited from :ref:`TomographyModelDocs`; a new geometry usually needs only:

* **A banded back-projection kernel** (``back_project_one_view_to_band``) for the multi-device
  reduce-scatter -- but only when a single recon slice maps to a *data-dependent band* of detector
  rows, as in cone beam.  A separable geometry whose detector row ``r`` maps one-to-one to recon
  slice ``r`` (parallel beam) does not need one.
* **A detector-row padding override** (``_sino_row_padding``) only if its detector rows track the
  recon slices (again, the parallel-beam case).
* **Inert padding** -- any per-slice or per-view operation must be written against the device-form
  (padded) length, not the real count.  This is the one subtlety that must be correct for sharding,
  and it is what bit each geometry port.

:class:`~mbirjax.ParallelBeamModel` (separable) and :class:`~mbirjax.ConeBeamModel` (non-separable,
with a band kernel) are the two worked examples to study.

Geometry skeleton
-----------------

The following is a starting skeleton for the per-view projector interface above.  It is **not** the
whole picture -- study the existing geometry classes and :doc:`dev_sharding_overview` for the
multi-device pieces.

.. include:: _static/new_model_template.py
   :code: python
