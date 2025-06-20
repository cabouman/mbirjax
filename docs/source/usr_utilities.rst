==============
Display and IO
==============

MBIRJAX includes utilities to view 3D data slice-by-slice and to save and load `.hdf5` files and to download and extract `.tar` files.

Saving and loading models and reconstructions is handled through TomographyModel: :ref:`SaveLoadDocs`.


Display Functions
-----------------

.. autofunction:: mbirjax.viewer.slice_viewer

Here is an example showing views of a modified Shepp-Logan phantom, with changing intensity window and displayed slice:

.. image:: https://www.math.purdue.edu/~buzzard/images/slice_viewer_demo.gif
   :alt: An animated image of the slice viewer.

IO Functions
------------

As noted above, saving and loading models and reconstructions is handled through TomographyModel: :ref:`SaveLoadDocs`.

The functions here are for direct interactions with files.

.. autofunction:: mbirjax.utilities.download_and_extract_tar
.. autofunction:: mbirjax.utilities.save_data_hdf5
.. autofunction:: mbirjax.utilities.load_data_hdf5
.. autofunction:: mbirjax.utilities.export_recon_hdf5
.. autofunction:: mbirjax.utilities.import_recon_hdf5