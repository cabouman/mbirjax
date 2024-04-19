========
Overview
========

**MBIRJAX** is a Python package for Model Based Iterative Reconstruction (MBIR) of images from tomographic data.
Here are the reasons to do MBIR reconstruction with MBIRJAX:

- Image quality:
  MBIR offers the best image quality because it uses a forward (sensor) and prior (image) model.

- Ease of use:
  MBIRJAX has built-in automatic parameter selection, so it will produce a good reconstruction the first time.

- Speed:
  MBIRJAX is fast (for MBIR) because:

  - GPU power:
    It uses JAX to harness the power of GPUs, CPUs, and clusters.

  - VCD algorithm:
    The vectorized coordinate descent algorithm converges fast and robustly.
    This not only reduces reconstruction time but also greatly improves image quality by accurately resolving high spatial frequencies.

- Flexibility:

  - OO Python interfact:
    MBIRJAX is based on object-oriented python, so it is modular and easy to use.

  - Plug-and-Play prior models:
    MBIRJAX supports proximal map interfaces, so it can be used with PnP deep neural net priors (coming soon).


**Getting Started**

We provide simple bash demo scripts located in `[mbirjax/demo] <https://github.com/cabouman/mbirjax/tree/main/demo>`__ that make it easy to get started.
We also have a bash install script at `[mbirjax/demo] <https://github.com/cabouman/mbirjax/tree/main/dev_scripts>`__.
Also, installing JAX is not too difficult on most platforms and is getting easier.

So to get started, you should remember to do the following things:

- **Get your data:**
  Import your data into numpy and to convert it to JAX format using ``jnp.array()``.

- **Reconstruct:**
  Use ``.recon()`` to produce an initial reconstruction that's usually pretty good.

- **Tune image quality:**
  Make your reconstruction sharper or smoother by adjusting the ``sharpness`` parameter.


**Geometry**

Right now MBIRJAX supports *parallel-beam* imaging geometry as shown below, but more geometries are on the way.

.. list-table::

    * - .. figure:: figs/geom-parallel.png
           :align: center
           :width: 50%

           Parallel-beam geometry
