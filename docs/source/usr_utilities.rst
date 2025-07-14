.. _Utilities:

=========
Utilities
=========

MBIRJAX contains utilities for viewing, downloading, exporting/importing, and generating synthetic data.

Saving and loading models and reconstructions is handled through TomographyModel: :ref:`SaveLoadDocs`.


3D Data Viewer
--------------

.. autofunction:: mbirjax.viewer.slice_viewer

Here is an example showing views of a modified Shepp-Logan phantom, with changing intensity window and displayed slice:

.. image:: https://www.math.purdue.edu/~buzzard/images/slice_viewer_demo.gif
   :alt: An animated image of the slice viewer.


Weight Generation
-----------------

.. autofunction:: mbirjax.vcd_utils.gen_weights
.. autofunction:: mbirjax.vcd_utils.gen_weights_mar


IO Functions
------------

As noted above, saving and loading models and reconstructions is handled through TomographyModel: :ref:`SaveLoadDocs`.

The functions here are for direct interactions with files.

.. autofunction:: mbirjax.utilities.download_and_extract
.. autofunction:: mbirjax.utilities.save_data_hdf5
.. autofunction:: mbirjax.utilities.load_data_hdf5
.. autofunction:: mbirjax.utilities.export_recon_hdf5
.. autofunction:: mbirjax.utilities.import_recon_hdf5


Synthetic Data Generation
-------------------------

.. autofunction:: mbirjax.utilities.generate_3d_shepp_logan_reference
.. autofunction:: mbirjax.utilities.generate_3d_shepp_logan_low_dynamic_range
.. autofunction:: mbirjax.utilities.gen_translation_phantom
.. autofunction:: mbirjax.utilities.generate_demo_data

