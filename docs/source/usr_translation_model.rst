.. _TranslationModelDocs:

=================
Translation Model
=================

The ``TranslationModel`` class implements a geometry and reconstruction model for translation computed tomography.
This class inherits all behaviors and attributes of the :ref:`TomographyModelDocs`.

This is an experimental tomography model in alpha testing.
It currently has no implementation of direct reconstructions, so mbirjax uses an initial condition of zer.

See the API docs for the :class:`~mbirjax.TomographyModel` class for details on a wide range
of functions that can be implemented using the ``TranslationModel``.

Constructor
-----------

.. autoclass:: mbirjax.TranslationModel
   :show-inheritance:


