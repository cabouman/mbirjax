============
Installation 
============

The ``MBIRJAX`` package currently is only available to download and install from source through GitHub.


Downloading and installing from source
-----------------------------------------

1. Download the source code:

  In order to download the python code, move to a directory of your choice and run the following two commands.

    | ``git clone https://github.com/cabouman/mbirjax.git``
    | ``cd mbirjax``

2. 2. Install the conda environment and package

    a. Option 1: Clean install using mbirjax/dev_scripts

        To do a clean install, use the command:

        .. code-block::

            cd dev_scripts
            source clean_install_all.sh


    b. Option 2: Manual install

        1. *Create conda environment:*

            Create a new conda environment named ``mbirjax`` using the following commands:

            .. code-block::

                conda create --name mbirjax python=3.10
                conda activate mbirjax
                pip install -r requirements.txt

            Anytime you want to use this package, this ``mbirjax`` environment should be activated with the following:

            .. code-block::

                conda activate mbirjax


        2. *Install mbirjax package:*

            Navigate to the main directory ``mbirjax/`` and run the following:

            .. code-block::

                pip install .

            To allow editing of the package source while using the package, use

            .. code-block::

                pip install -e .
