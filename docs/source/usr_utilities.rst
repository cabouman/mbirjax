===============
Plotting and IO
===============

MBIRJAX includes some utilities to view 3D data slice-by-slice and to save and load tar files.
Saving and loading models and recons is done through :ref:`TomographyModelDocs`.

Here is an example showing views of a modified Shepp-Logan phantom, with changing intensity window and displayed slice.

.. image:: https://www.math.purdue.edu/~buzzard/images/slice_viewer_demo.gif
   :alt: An animated image of the slice viewer.

Display Functions
-----------------

.. autofunction::
   mbirjax.viewer.slice_viewer

IO Functions
------------

.. autofunction::
   mbirjax.utilities.download_and_extract_tar
