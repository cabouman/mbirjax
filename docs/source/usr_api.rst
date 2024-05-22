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

Parameter Handling
------------------

Parameter handling is inherited from :ref:`ParameterHandlerDevDocs`, with the following methods.

.. autosummary::

   mbirjax.ParameterHandler.set_params
   mbirjax.ParameterHandler.get_params
   mbirjax.ParameterHandler.print_params
   mbirjax.ParameterHandler.save_params
   mbirjax.ParameterHandler.load_params

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
