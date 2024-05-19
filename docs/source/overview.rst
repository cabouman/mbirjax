========
Overview
========

**MBIRJAX** is a Python package for Model Based Iterative Reconstruction (MBIR) of images from tomographic data.

Here are the reasons to use MBIRJAX:

- **Image quality:**
  MBIR offers the best image quality because it uses a forward (sensor) and prior (image) model.

- **Ease of use:**
  MBIRJAX has built-in automatic parameter selection, so it will produce a good reconstruction the first time.

- **Speed:**
  MBIRJAX is fast (for MBIR) because:

  - **GPU power:**
    Uses JAX to harness the power of GPUs, CPUs, and clusters.

  - **VCD algorithm:**
    The vectorized coordinate descent algorithm has fast convergence to both reduce reconstruction time and improve image quality.

- **Flexibility:**

  - **OO Python interface:**
    Based on object-oriented python, so it is modular and easy to use.

  - **Plug-and-Play prior models:**
    Supports proximal map interfaces, so it can be used with PnP deep neural net priors.


We provide simple bash demo scripts located in `[mbirjax/demo] <https://github.com/cabouman/mbirjax/tree/main/demo>`__ that make it easy to get started.
We also have a bash install script at `[mbirjax/demo] <https://github.com/cabouman/mbirjax/tree/main/dev_scripts>`__.
Also, installing JAX is not too difficult on most platforms and is getting easier.


**Geometries**

MBIRJAX supports the *parallel-beam* and *cone-beam* imaging geometries shown below.
However, new geometries can be added by constructing a new class with the associated sparse forward and back projection code.

.. list-table::

    * - .. figure:: figs/geom-parallel.png
           :align: center
           :width: 75%

           Parallel-beam geometry

      - .. figure:: figs/geom-cone-beam.png
           :align: center
           :width: 75%

           Cone-beam geometry
