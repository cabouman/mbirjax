========
User API
========

Most functions can be accessed by importing mbirjax and creating a model or through mbirjax directly.  Most
commonly used functions are described below.  See :ref:`DemosFAQs` for examples.

Geometry Models
---------------

The first step is to create an instance with a specific geometry. This is done by initializing one of the following geometry classes:

.. autosummary::

   mbirjax.ParallelBeamModel
   mbirjax.ConeBeamModel
   mbirjax.TranslationModel

Reconstruction and Projection
-----------------------------

Each geometry class is derived from :ref:`TomographyModelDocs`, which includes a number of powerful methods listed below for manipulating sinograms and reconstructions.
Detailed documentation for each geometry class is provided in :ref:`ParallelBeamModelDocs` and :ref:`ConeBeamModelDocs`.

Note that :ref:`ParallelBeamModelDocs` also includes ``fbp_recon`` and :ref:`ConeBeamModelDocs` includes ``fdk_recon``
for direct (non-iterative) reconstruction in the case of many views and low-noise data.

.. autosummary::

   mbirjax.TomographyModel.recon
   mbirjax.TomographyModel.scale_recon_shape
   mbirjax.TomographyModel.prox_map
   mbirjax.TomographyModel.forward_project
   mbirjax.TomographyModel.back_project
   mbirjax.TomographyModel.gen_weights
   mbirjax.TomographyModel.gen_weights_mar
   mbirjax.TomographyModel.gen_modified_3d_sl_phantom

Parameter Handling
------------------

See the :ref:`Primary Parameters <ParametersDocs>` page for a description of the primary parameters.
Parameter handling uses the following primary methods.

.. autosummary::

   mbirjax.ParameterHandler.set_params
   mbirjax.ParameterHandler.get_params
   mbirjax.ParameterHandler.print_params


.. automodule:: mbirjax
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

.. toctree::
   :hidden:
   :maxdepth: 4
   :caption: Classes

   usr_tomography_model
   usr_parameters
   usr_parallel_beam_model
   usr_cone_beam_model
   usr_translation_model
   usr_denoising
   usr_preprocess
   usr_utilities

Preprocessing
-------------

Preprocessing functions are implemented in :ref:`PreprocessDocs`. This includes various methods to compute and correct the sinogram data as needed.
The following are functions specific to NSI scanners.  See `demo_nsi.py <https://github.com/cabouman/mbirjax_applications/tree/main/nsi>`__ in the
`mbirjax_applications <https://github.com/cabouman/mbirjax_applications>`__ repo.

.. autosummary::

   preprocess.nsi.compute_sino_and_params
   preprocess.nsi.load_scans_and_params

The remaining functions can be used for multiple types of scan data.

.. autosummary::

   preprocess.compute_sino_transmission
   preprocess.interpolate_defective_pixels
   preprocess.correct_det_rotation_and_background
   preprocess.estimate_background_offset

Denoising
---------

MBIRJAX includes a Bayesian MAP denoising using the qGGMRF prior and a 3D median filter.

Bayesian denoising is implemented via the class :class:`QGGMRFDenoiser` with a using the QGGMRF prior loss function, which promotes
nearby voxels to have similar values while preserving edges.  Using :math:`Q(v)` to denote the loss function, the
denoiser is

        .. math::

            F(x) = \arg\min_v \left\{ Q(v) + \frac{1}{2}\|v - x\|^{2} \right\}.

.. autosummary::

   QGGMRFDenoiser.denoise

The median filter is implemented in jax using a fixed 3x3x3 neighborhood with replicated edges at the boundary.

.. autosummary::

   median_filter3d

Saving, Loading, and Display
----------------------------

* Saving and loading of the parameters needed to define a model are implemented in :meth:`TomographyModel.to_file` and :meth:`TomographyModel.from_file`.
* Saving and loading of the data and the dict of parameters/logs returned from :meth:`TomographyModel.recon` are implemented in :meth:`TomographyModel.save_recon_hdf5` and :meth:`TomographyModel.load_recon_hdf5`.
* Display of reconstructions and parameters/log dict is handled by :func:`viewer.slice_viewer`, which can be given these objects directly or which can be used to load hdf5 files interactively.

.. autosummary::

   TomographyModel.to_file
   TomographyModel.from_file
   TomographyModel.save_recon_hdf5
   TomographyModel.load_recon_hdf5
   viewer.slice_viewer


