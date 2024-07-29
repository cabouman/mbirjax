.. _ExamplesFAQs:

=================
Examples and FAQs
=================

Demos
-----

Here are some demos to illustrate the basics of MBIRJAX along with some more advanced features.

1. **Basic Demo:** `Jupyter notebook <https://colab.research.google.com/drive/1zG_H6CDjuQxeMRQHan3XEyX2YVKcSSNC?usp=drive_link>`__ or `Python script <https://github.com/cabouman/mbirjax/blob/main/demo/demo_1_shepp_logan.py>`__
2. **Large Object:**  `Jupyter notebook <https://colab.research.google.com/drive/1-kk_HeR8Y8f6pZ2zjTza8NTEpAgwgVRB?usp=sharing>`__ or `Python script <https://github.com/cabouman/mbirjax/blob/main/demo/demo_2_large_object.py>`__
3. **Cropped Center:**  `Jupyter notebook <https://colab.research.google.com/drive/1WQwIJ_mDcuMMcWseM66aRPvtv6FmMWF-?usp=sharing>`__ or `Python script <https://github.com/cabouman/mbirjax/blob/main/demo/demo_3_cropped_center.py>`__
4. **Wrong Rotation:**  `Jupyter notebook <https://colab.research.google.com/drive/1Gd-fMm3XK1WBsuJUklHdZ-4jjsvdpeIT?usp=sharing>`__ or `Python script <https://github.com/cabouman/mbirjax/blob/main/demo/demo_4_wrong_rotation_direction.py>`__

First browse the notebooks, then copy and run in your own notebook environment,
or follow the installation instructions at :ref:`InstallationDocs` and run the scripts directly.

Then adjust some of the parameters to better understand how the code works.
If you have a GPU, you can increase the problem size by changing ``num_views``, ``num_det_rows``, and ``num_det_channels``.

The separate repo `mbirjax_applications <https://github.com/cabouman/mbirjax_applications>`__ provides a wider variety of examples using real data.


FAQs
----

Q: Why is there a bright ring around my reconstruction?
+++++++++++++++++++++++++++++++++++++++++++++++++++++++

A: If the object does not project completely inside the detector, then MBIR will produce a bright ring
around the edge of the reconstruction to account for the portion of the object that projects to the detector in only some views.
See Demo 2: Large Object for an example of this.
You can improve the reconstruction by increasing recon_shape:

.. code-block:: python

        ct_model.scale_recon_shape(row_scale=1.2, col_scale=1.2)

Note that the scale factor need only be large enough to give some padding around the region of valid projection --
it does not need to match the size of the true object.  Larger scale factors will lead to increased time and memory.

Q: Why is my reconstruction blurry?
+++++++++++++++++++++++++++++++++++

A:  If your reconstruction is blurry, the first thing to try is to increase the sharpness parameter.  Values of
``sharpness=1.0`` or ``sharpness=2.0`` are typical, but larger values can further improve sharpness. If the
reconstruction remains blurry, it is often the case that some geometry parameter is incorrectly set for your data.
Typical problems include an incorrect center of rotation (change ``det_channel_offset``), incorrect rotation direction
(reverse the angles using ``angles[::-1]``), or an incorrect ``source_detector_dist`` or  ``source_iso_dist`` for
cone beam reconstructions.

Q: How can I do larger reconstructions?
+++++++++++++++++++++++++++++++++++++++

A: Note that a 2K x 2K x 2K reconstruction occupies 32GB of memory, not counting the sinogram or memory needed for processing.
If your reconstruction is too large for your GPU memory, MBIRJAX will use CPU memory for some processing and then transfer
to GPU as needed; this reduces memory use but increases reconstruction time.  If you have no GPU or your GPU memory is small relative
to the problem size, then all processing is done on the CPU.

If you have a parallel beam system, you can select a subset of rows of your sinogram, reconstruct them separately, and then
concatenate them at the end.  If you have a cone beam system, you can reconstruct a subset of the central slices.  In either
case, you can do a center cropped reconstruction as in Demo 3: Cropped Center, although as seen in that demo, this can
introduce an intensity shift and other artifacts.

We continue to improve the time and memory efficiency of MBIRJAX and will investigate multi-GPU/multi-CPU solutions

Q: Why is my reconstruction noisy?
++++++++++++++++++++++++++++++++++

A:  This could be due to noise in the data or to incomplete convergence. Decreasing sharpness to -0.5 or -1.0 will
reduce high-frequency artifacts but also lead to some blurring of edges:

.. code-block::

    sharpness=-1.0
    ct_model.set_params(sharpness=sharpness)

If the percent change indicated at the end of your reconstruction (using verbose=1) is more than about 0.1, then
you might also try increasing the number of iterations:

.. code-block::

    ct_model.recon(sinogram, num_iterations=20)

If the percent change is less than about 0.01, then extra iterations will not improve image quality but will take
extra time.  The default number of iterations is typically sufficient for very good image quality.

Q: Why does my reconstruction look distorted?
+++++++++++++++++++++++++++++++++++++++++++++

A: Make sure that the type of reconstruction (parallel or cone) matches your data and that the geometry parameters all match.  If you
are using a cone beam system, make sure the source to detector and source to iso distances are correct, and make
sure the rotation direction is correct.  See Demo 3: Wrong Rotation Direction for an example of what can happen if
the rotation direction is incorrect.

Q: Why are there rings in my reconstruction?
++++++++++++++++++++++++++++++++++++++++++++

A: Some detectors have significant variation in per-pixel sensitivity.  These differences can lead to different
measured energy in adjacent detectors that should have essentially the same measured energy.  When these differences
are incorporated into a reconstruction, they lead to concentric rings.  We are working on preprocessing utilities for
reducing ring artifacts.

Q: How can I shift a cone-beam recon up or down relative to the detector?
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

A: You can shift the region of reconstruction up or down using ``ct_model.set_params(recon_slice_offset=offset)``
before calling recon.
