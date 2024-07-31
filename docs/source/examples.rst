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
``sharpness=1.0`` or ``sharpness=2.0`` are typical, but larger values can further improve sharpness.
You can also increase the assumed SNR by setting the parameter ``snr_db=35`` or ``snr_db=40``. This is similar to increasing sharpness but will also create higher contrast edges in the image.

If the reconstruction remains blurry, it is often the case that some geometry parameter is incorrectly set for your data.
Typical problems include an incorrect center of rotation (change ``det_channel_offset``), incorrect rotation direction
(reverse the angles using ``angles[::-1]``), or an incorrect ``source_detector_dist`` or  ``source_iso_dist`` for
cone beam reconstructions.

Q: How can I do larger reconstructions?
+++++++++++++++++++++++++++++++++++++++

A: MBIRJAX runs on both CPU and GPU computers, but we strongly recommend the use of GPUs for large reconstructions since they are much faster.
On a GPU, the size of the reconstruction is typically limited by the amount of GPU memory.
So you should find a fast GPU with the largest possible memory. These days that is typically 40GB to 80GB of GPU memory.
The GPU will be hosted on a CPU, and it is best if that CPU also has even a larger amount of memory, ideally greater than 200GB.

Note that a 2K x 2K x 2K reconstruction occupies 32GB of memory, not counting the sinogram or memory needed for processing.
If your reconstruction is too large for your GPU memory, MBIRJAX will use CPU memory for some processing and then transfer
to the GPU as needed; this reduces memory use but increases reconstruction time.  If you have no GPU or your GPU memory is small relative
to the problem size, then all processing is done on the CPU.

If you have a parallel beam system, you can select a subset of rows of your sinogram, reconstruct them separately, and then
concatenate them at the end.  If you have a cone beam system, you can reconstruct a subset of the central slices.  In either
case, you can do a center cropped reconstruction as in Demo 3: Cropped Center, although as seen in that demo, this can
introduce an intensity shift and other artifacts.

We continue to improve the time and memory efficiency of MBIRJAX and will investigate multi-GPU/multi-CPU solutions.
So stay tuned for further improvements.


Q: Why does my reconstruction have artifacts?
+++++++++++++++++++++++++++++++++++++++++++

There are many reasons that a reconstruction may have artifacts including noise, blurring, streaks, cupping, etc.

First, make sure you are using the geometry (parallel or cone) that matches your data.
Parallel beam geometry is faster, but it may not be accurate is the source is too close to the object.

For transmission tomography, it is critically important to preprocess the raw photon measurements by normalizing by an air-scan and taking the negative log of the ratio.
We provide simple preprocessing utilities in ``mbirjax.preprocess`` for doing this, and we plan to provide more utilities for specific instruments in the future.

In conebeam scans, it is sometimes the case that the rotation direction is reversed.
This can cause the reconstruction to look blurry or distorted.
You can correct this by simply taking the negative of your view angles.
See Demo 3: Wrong Rotation Direction for an example of what can happen if the rotation direction is incorrect.

A common artifact is rings that are generated when the center-of-rotation is off in the views.
This can be corrected by setting the parameter ``det_channel_offset`` to account for a center-of-rotation that is offset from the center of the view.

If the image is too noisy, you might try reducing the value of the ``sharpness`` or ``snr_db``` parameters.
You can also improve image quality by using the ``weights`` array that can be generated using the ``gen_weights()`` method.
The weights provide information on the reliability of the sinogram values, with larger weights indicating higher reliability.

Streaks are often caused by metal in the object being scanned.
One advantage of MBIR is that it generally has fewer metal artifacts, but some artifacts typically remain.
Using weights will reduce metal artifacts, and the function ``gen_weights_mar()`` can be used to generate weights that further reduce metal artifacts.

Cupping is typically caused by beam hardening with polychromatic X-ray sources.
This can be partially corrected with a low order polynomial correction.
We are working on utilities to do beam hardening correction in the future.

Ring artifacts are typically caused either by an incorrect center of rotation or detector nonuniformity.
Detector nonuniformity results from the variation in detector sensitivity from pixel to pixel.
This variation is taken out to some degree by air scan normalization, but some variation may remain.
These variations will lead to concentric rings in the reconstruction.
We are working on preprocessing utilities for reducing these ring artifacts.


Q: How can I shift region-of-reconstruction up or down for a conebeam reconstruction?
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

A: You can shift the region of reconstruction up or down using ``ct_model.set_params(recon_slice_offset=offset)``
before calling recon.
Positive values of ``offset`` will shift the region down relative to the detector.
This is useful if you would like to reconstruct the top or bottom half of a conebeam reconstruction in order to save memory.


