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


