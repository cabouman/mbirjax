.. docs-include-ref

MBIRJAX
=======

Model-Based Iterative Reconstruction (MBIR) for tomographic reconstruction that is based on the `JAX <https://github.com/google/jax>`__ library.
Full documentation is available at https://mbirjax.readthedocs.io .

Installing from PyPI
--------------------

For CPU only:

    .. code-block::

        pip install mbirjax

For CPU with a CUDA12-enabled GPU:

    .. code-block::

        pip install --upgrade mbirjax[cuda12]

Installing from Source
----------------------

1. *Clone the repository:*

    .. code-block::

        git clone git@github.com:cabouman/mbirjax.git

2. Install the conda environment and package

    a. Option 1: Clean install using dev_scripts - We provide bash scripts that will do a clean install of MBIRJAX in a new conda environment using the commands:

        .. code-block::

            cd dev_scripts
            source clean_install_all.sh

    b. Option 2: Manual install - You can also manually install MBIRJAX from the main directory of the repository with the following commands:

        .. code-block::

            conda create --name mbirjax python=3.10
            conda activate mbirjax
            pip install -r requirements.txt
            pip install .

Optional Pixi Development Environment
-------------------------------------

For contributors who use `Pixi <https://pixi.sh>`__, MBIRJAX also provides an optional reproducible development environment.
This does not replace the conda installation workflow above.

For the default CPU environment on Linux or Apple Silicon macOS:

    .. code-block::

        pixi run smoke
        pixi run test-fast

For a CUDA-enabled Linux system:

    .. code-block::

        pixi run -e cuda smoke-jax
        pixi run -e cuda test-fast

Additional useful tasks include:

    .. code-block::

        pixi run test
        pixi run test-data
        pixi run docs

Running Demo(s)
---------------

Run any of the available demo scripts with something like the following:

    .. code-block::

        python demo/<demo_file>.py
