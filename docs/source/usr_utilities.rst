===================
Display and IO
===================

MBIRJAX includes some utilities to view 3D data slice-by-slice and to save and load `.tar` files.
Saving and loading models and reconstructions is handled through :ref:`TomographyModelDocs`.

Here is an example showing views of a modified Shepp-Logan phantom, with changing intensity window and displayed slice:

.. image:: https://www.math.purdue.edu/~buzzard/images/slice_viewer_demo.gif
   :alt: An animated image of the slice viewer.

Display Functions
-----------------

.. automodule:: mbirjax.viewer
   :members: slice_viewer, SliceViewer
   :undoc-members:
   :no-index:
   :show-inheritance:

.. autofunction:: mbirjax.viewer.slice_viewer
.. automethod:: mbirjax.viewer.SliceViewer

IO Functions
------------

.. automodule:: mbirjax.utilities
   :members: download_and_extract_tar
   :undoc-members:
   :no-index:
   :show-inheritance:

.. autofunction:: mbirjax.utilities.download_and_extract_tar