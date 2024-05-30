.. docs-include-ref

MBIRJAX
=======

Model-Based Iterative Reconstruction (MBIR) for tomographic reconstruction that is based on the `JAX <https://github.com/google/jax>`__ library.
Full documentation is available at https://mbirjax.readthedocs.io .

Installing from PyPI
--------------------

    .. code-block::

        pip install mbirjax

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

Running Demo(s)
---------------

Run any of the available demo scripts with something like the following:

    .. code-block::

        python demo/<demo_file>.py

