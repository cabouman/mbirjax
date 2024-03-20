============
Installation 
============

The ``mbirjax_sandbox`` package currently is only available to download and install from source through GitHub.


Downloading and installing from source
-----------------------------------------

1. Download the source code:

  In order to download the python code, move to a directory of your choice and run the following two commands.

    | ``git clone https://github.com/cabouman/mbirjax_sandbox.git``
    | ``cd mbirjax_sandbox``


2. Create a Virtual Environment:

  It is recommended that you install to a virtual environment.
  If you have Anaconda installed, you can run the following:

    | ``conda create --name mbirjax_sandbox python=mbirjax_sandbox``
    | ``conda activate mbirjax_sandbox``

  Install the dependencies using:

    ``pip install -r requirements.txt``

  Install the package using:

    ``pip install .``

  or to edit the source code while using the package, install using

    ``pip install -e .``

  Now to use the package, this ``mbirjax_sandbox`` environment needs to be activated.


3. Install:

You can verify the installation by running ``pip show mbirjax_sandbox``, which should display a brief summary of the packages installed in the ``mbirjax_sandbox`` environment.
Now you will be able to use the ``mbirjax_sandbox`` python commands from any directory by running the python command ``import mbirjax_sandbox``.

