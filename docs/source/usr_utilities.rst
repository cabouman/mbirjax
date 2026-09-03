.. _Utilities:

=========
Utilities
=========

MBIRJAX contains utilities for viewing, downloading, exporting/importing, and generating synthetic data.

Saving and loading models and reconstructions is handled through TomographyModel: :ref:`SaveLoadDocs`.


Viewing Reconstructions
-----------------------

.. autofunction:: mbirjax.viewer.slice_viewer

Here is an example showing views of a modified Shepp-Logan phantom, with changing intensity window and displayed slice:

.. image:: https://www.math.purdue.edu/~buzzard/images/slice_viewer_demo.gif
   :alt: An animated image of the slice viewer.

For a movie saved to a file rather than viewed interactively, use the GIF writer.  It loops a 3D
or 4D volume over an axis of your choosing, laying out the plane shown the same way the viewer
does.  By default a 3D volume loops over its first axis and a 4D volume loops over time at the
middle slice.

.. autofunction:: mbirjax.utilities.save_volume_as_gif


General Purpose
---------------

.. autofunction:: mbirjax.utilities.stitch_arrays
.. autofunction:: mbirjax.utilities.get_ct_model
.. autofunction:: mbirjax.utilities.copy_ct_model
.. autofunction:: mbirjax.utilities.build_model


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


.. _synthetic-data-generation:

Synthetic Data Generation
-------------------------

.. autofunction:: mbirjax.utilities.generate_demo_data
.. autofunction:: mbirjax.utilities.generate_3d_shepp_logan_reference
.. autofunction:: mbirjax.utilities.generate_3d_shepp_logan_low_dynamic_range
.. autofunction:: mbirjax.utilities.gen_translation_phantom

