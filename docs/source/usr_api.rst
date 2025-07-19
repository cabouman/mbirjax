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

Denoising
---------

See :ref:`DenoisingDocs` for details on Denoising Functions.
These includes functions for computing the MAP denoiser using the qGGMRF prior and a 3D median filter.

.. autosummary::

   mbirjax.QGGMRFDenoiser.denoise

The median filter is implemented in jax using a fixed 3x3x3 neighborhood with replicated edges at the boundary.

.. autosummary::

   mbirjax.denoising.median_filter3d

Parameter Handling
------------------

See :ref:`Primary Parameters <ParametersDocs>` page for a description of the primary parameters.
Users can set, get, and printout parameters using the following primary methods.

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
   usr_utilities
   usr_preprocess

Saving and Loading
------------------

* Saving and loading of the parameters needed to define a model are implemented in :meth:`TomographyModel.to_file` and :meth:`TomographyModel.from_file`.
* Saving and loading of the data and the dict of parameters/logs returned from :meth:`TomographyModel.recon` are implemented in :meth:`TomographyModel.save_recon_hdf5` and :meth:`TomographyModel.load_recon_hdf5`.

.. autosummary::

   TomographyModel.to_file
   TomographyModel.from_file
   TomographyModel.save_recon_hdf5
   TomographyModel.load_recon_hdf5


Utilities
---------

See :ref:`Utilities` for details on Utility Functions.
These include variety of functions for viewing, generating weights, exporting/importing data, and generating synthetic data.

.. autosummary::

   viewer.slice_viewer
   vcd_utils.gen_weights
   vcd_utils.gen_weights_mar
   utilities.download_and_extract
   utilities.export_recon_hdf5
   utilities.import_recon_hdf5
   utilities.generate_3d_shepp_logan_low_dynamic_range


Preprocessing
-------------

See :ref:`PreprocessDocs` for details on Preprocessing Functions.
These functions various methods to compute and correct the sinogram data as needed.
The following are functions specific to NSI scanners.  See `demo_nsi.py <https://github.com/cabouman/mbirjax_applications/tree/main/nsi>`__ in the
`mbirjax_applications <https://github.com/cabouman/mbirjax_applications>`__ repo.

It also includes functions for processing cone beam and parallel beam data to remove artifacts from metal, detector defects.

It also includes functions for optimal view selection.


