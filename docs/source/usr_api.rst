========
User API
========

Most functions can be accessed by importing mbirjax and creating a model or through mbirjax directly.  Most
commonly used functions are described below.  See :ref:`ExamplesFAQs` for examples.

Geometry Models
---------------

The first step is to create an instance with a specific geometry. This is done by initializing one of the following geometry classes:

.. autosummary::

   mbirjax.ParallelBeamModel
   mbirjax.ConeBeamModel

Reconstruction and Projection
-----------------------------

Each geometry class is derived from :ref:`TomographyModelDocs`, which includes a number of powerful methods listed below for manipulating sinograms and reconstructions.
Detailed documentation for each geometry class is provided in :ref:`ParallelBeamModelDocs` and :ref:`ConeBeamModelDocs`.

.. autosummary::

   mbirjax.TomographyModel.recon
   mbirjax.TomographyModel.scale_recon_shape
   mbirjax.TomographyModel.prox_map
   mbirjax.TomographyModel.forward_project
   mbirjax.TomographyModel.back_project
   mbirjax.TomographyModel.gen_weights
   mbirjax.TomographyModel.gen_weights_mar
   mbirjax.TomographyModel.gen_modified_3d_sl_phantom

Saving and Loading
------------------

Saving and loading are implemented in :ref:`TomographyModelDocs`, with methods overridden in geometry-specific models
as needed.

.. autosummary::

   mbirjax.TomographyModel.to_file
   mbirjax.TomographyModel.from_file

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
   usr_plot_utils
   usr_preprocess

Preprocessing
------------------

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
   preprocess.correct_det_rotation
   preprocess.estimate_background_offset



