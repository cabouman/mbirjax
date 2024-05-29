============
Installation 
============
.. _installation_label:

The ``MBIRJAX`` package currently is available either from PyPI or you can download and install the source from GitHub.

**Install from PyPI**

In order to install from PyPI, simply use the following command::

    pip install mbirjax


**Installing from source**

1. Download the source code

In order to download the python code, move to a directory of your choice and run the following two commands::

    git clone https://github.com/cabouman/mbirjax.git
    cd mbirjax

2. Install the conda environment and package

Option 1: Clean install using mbirjax/dev_scripts - We provide bash scripts that will do a clean install of ``MBIRJAX`` in a new conda environment using the following commands::

    cd dev_scripts
    source clean_install_all.sh

Option 2: Manual install - You can also manually install ``MBIRJAX`` from the main directory of the repository with the following commands::

    conda env create --name mbirjax --file environment.yml
    conda activate mbirjax
    pip install .



