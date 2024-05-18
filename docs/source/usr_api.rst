========
User API
========

MBIRJAX is designed to give reconstructions using just a few lines of code.

The first step is to create an instance with a specific geometry. This is done by initializing a class such as

.. autosummary::

   mbirjax.ParallelBeamModel

:ref:`ParallelBeamModelDocs` and classes for other geometries are derived from :ref:`TomographyModelDocs`, which includes several methods
for manipulating sinograms and reconstructions.

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


