========
User API
========

MBIRJAX is designed to give reconstructions using just a few lines of code.

Geometry Models
---------------

The first step is to create an instance with a specific geometry. This is done by initializing one of the following geometry classes:

.. autosummary::

   mbirjax.ParallelBeamModel
   mbirjax.ConeBeamModel

Projection and Reconstruction
-----------------------------

Each geometry class is derived from :ref:`TomographyModelDocs`, which includes a number of powerful methods listed below for manipulating sinograms and reconstructions.
Detailed documentation for each geometry class is provided in :ref:`ParallelBeamModelDocs` and :ref:`ConeBeamModelDocs`.

.. autosummary::

   mbirjax.TomographyModel.recon
   mbirjax.TomographyModel.prox_map
   mbirjax.TomographyModel.forward_project
   mbirjax.TomographyModel.back_project
   mbirjax.TomographyModel.gen_weights
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
Parameter handling is inherited from :ref:`ParameterHandlerDevDocs`, with the following primary methods.

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

.. autosummary::

   mbirjax.preprocess.NSI_load_scans_and_params
   mbirjax.preprocess.transmission_CT_compute_sino
   mbirjax.preprocess.interpolate_defective_pixels
   mbirjax.preprocess.correct_det_rotation
   mbirjax.preprocess.calc_background_offset

