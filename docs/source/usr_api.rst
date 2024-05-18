========
User API
========

MBIRJAX is designed to give reconstructions using just a few lines of code.

The first step is to create an instance with a specific geometry. This is done by initializing one of the following geometry a classes:

.. autosummary::

   mbirjax.ParallelBeamModel
   mbirjax.ConeBeamModel

Each geometry class is derived from :ref:`TomographyModelDocs`, which includes a number of powerful methods listed below for manipulating sinograms and reconstructions.
Detailed documentation for each geometry class is provided in :ref:`ParallelBeamModelDocs` and :ref:`ConeBeamModelDocs`.

.. autosummary::

   mbirjax.TomographyModel.recon
   mbirjax.TomographyModel.prox_map
   mbirjax.TomographyModel.forward_project
   mbirjax.TomographyModel.back_project
   mbirjax.TomographyModel.set_params
   mbirjax.TomographyModel.get_params
   mbirjax.TomographyModel.gen_weights
   mbirjax.TomographyModel.gen_modified_3d_sl_phantom
   mbirjax.TomographyModel.print_params

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
   usr_parallel_beam_model
   usr_cone_beam_model
