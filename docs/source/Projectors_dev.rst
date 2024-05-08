.. _ProjectorsDevDocs:

=========================
Projectors Developer Docs
=========================

The ``Projectors`` class uses the low-level projection functions
implemented in a specific geometry in order to override

* :meth:`mbirjax.TomographyModel.sparse_forward_project`
* :meth:`mbirjax.TomographyModel.sparse_back_project`
* :meth:`mbirjax.TomographyModel.compute_hessian_diagonal`

The ``Projectors`` class provides JAX-specific code using vmap and scan along with batching along voxels and
views in order to provide code that balances memory-efficiency with time-efficiency.

.. autoclass:: mbirjax.Projectors
   :members:
   :member-order: bysource
   :show-inheritance:

