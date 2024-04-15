.. docs-include-ref

mbirjax
=======
Project for creating a JAX version of MBIR

Full documentation is available at https://mbirjax.readthedocs.io .

..
    Include more detailed description here.

Installing
----------
1. *Clone or download the repository:*

    .. code-block::

        git clone git@github.com:bouman/mbirjax

2. Install the conda environment and package

    a. Option 1: Clean install from dev_scripts

        *******You can skip all other steps if you do a clean install.******

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


Running Demo(s)
---------------

Run any of the available demo scripts with something like the following:

    .. code-block::

        python demo/<demo_file>.py

