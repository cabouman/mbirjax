=================
Examples and FAQs
=================

First follow the installation instructions at :ref:`InstallationDocs`.

Then try ``demo_shepp_logan.py`` in `mbirjax/demo <https://github.com/cabouman/mbirjax/tree/main/demo>`__ and adjust
some of the parameters to better understand how the code works.
If you have a GPU, you can increase the problem size by changing ``num_views``, ``num_det_rows``, and ``num_det_channels``.

The separate repo `mbirjax_applications <https://github.com/cabouman/mbirjax_applications>`__ provides a wider variety of examples using real data.


FAQs
----

Q: Why is there a bright ring around my reconstruction?

A: If the object does not project completely inside the detector, then MBIR will produce a bright ring
around the edge of the reconstruction to account for the object that projects to the detector in only some views.
You can improve the reconstruction at the cost of extra time and memory by increasing recon_shape:

.. code-block:: python

        ct_model.scale_recon_shape(row_scale=1.2, col_scale=1.2)

