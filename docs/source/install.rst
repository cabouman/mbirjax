.. _InstallationDocs:

============
Installation
============

The ``MBIRJAX`` package is available through PyPI or from source on GitHub.

**Install from PyPI**

In order to install from PyPI, use one of the following commands.

For CPU only::

    pip install --upgrade mbirjax

For CPU with a CUDA12-enabled GPU::

    pip install --upgrade mbirjax[cuda12]


**Installing from source**

1. Download the source code

In order to download the python code, move to a directory of your choice and run the following two commands::

    git clone https://github.com/cabouman/mbirjax.git
    cd mbirjax

2. Install the conda environment and package

Clean install using mbirjax/dev_scripts - We provide bash scripts that will do a clean install of ``MBIRJAX`` in a new conda environment using the following commands::

    cd dev_scripts
    source clean_install_all.sh


**Optional Pixi development environment**

For contributors who use `Pixi <https://pixi.sh>`__, ``MBIRJAX`` also provides an optional reproducible development environment.
This does not replace the conda installation workflow above.

For the default CPU environment on Linux or Apple Silicon macOS::

    pixi run smoke
    pixi run test-fast

For a CUDA-enabled Linux system::

    pixi run -e cuda smoke-jax
    pixi run -e cuda test-fast

Additional useful tasks include::

    pixi run test
    pixi run test-data
    pixi run docs


**Installing on Windows**

In order to use MBIRJAX on a windows computer with an NVIDIA GPU,
we recommend that you use Windows Subsystem for Linux (WSL).
Download `this pdf file <https://www.datadepot.rcac.purdue.edu/bouman/data/MBIRJAX_on_windows.pdf>`_ for details on how to install MBIRJAX with WSL.
