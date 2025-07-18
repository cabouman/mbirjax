Package Maintenance
===================

The following describes procedures for basic package maintenance.

Unit Tests
----------

In order to run unit tests, install MBIRJAX using the provided install scripts, and from the root directory of the repository, activate the conda environment, and then run the following::

    pytest

This should be repeated for each supported platform.

Uploading to PyPI
-----------------

This is only available for registered maintainers.

0. Update the version number in prerelease and accept the PR to main.

1. Run the script clean_install_all.sh and activate mbirjax

2. First, make sure you have installed the newest versions of `setuptools`, `wheel`, `build`, `twine`, and `pkginfo`. Then from the main mbirjax directory, delete any previous build and then build the project::

    pip install --upgrade setuptools build wheel twine pkginfo
    rm -rf dist/ build/ *.egg-info
    python -m build

3. Check the distribution::

    python -m twine check dist/*

4. Upload to PyPI.  You will need an API token from PyPI.  NOTE: You cannot upload the same version more than once::

    python -m twine upload dist/*

   View the package upload here:
   `https://pypi.org/project/mbirjax <https://pypi.org/project/mbirjax>`__

3. Test the uploaded package (NOTE: to test on the GPU, use 'pip install mbirjax[cuda12]')::

    pip install mbirjax    # OR, "mbirjax==0.1.1" e.g. for a specific version number
    python -c "import mbirjax"     # spin the wheel
    pip install pytest
    pytest tests

4. Run one of the demos in `mbirjax/demo <https://github.com/cabouman/mbirjax/tree/main/demo>`__.


5. Verify that the `corresponding build <https://readthedocs.org/projects/mbirjax/builds/>`__ of the MBIRJAX documentation has built correctly.

Uploading to Test PyPI
----------------------

This is only available for registered maintainers.  Typically, you would perform these steps on the prerelease branch before the final commit to main.

Follow steps 0-3 as above.  Then continue from step 4.

4. Upload to Test PyPI. You will need to get an API token from TestPyPI. You will be prompted for this token from the command line. NOTE: You cannot upload the same version more than once::

    python -m twine upload --repository testpypi dist/*

   View the package upload here:
   `https://test.pypi.org/project/mbirjax <https://test.pypi.org/project/mbirjax>`__

5. Test the uploaded package (NOTE: to test on the GPU, use 'pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple mbirjax[cuda12]')::

    pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple mbirjax
    python -c "import mbirjax"     # spin the wheel
    pip install pytest
    pytest tests

6. Run one of the demos in `mbirjax/demo <https://github.com/cabouman/mbirjax/tree/main/demo>`__.

   NOTE: If the install fails and you need to re-test, *temporarily* set
   the version number in `pyproject.toml` from X.X.X to X.X.X.1 (then 2, 3, etc.),
   for further testing. After the test is successful, reset the version number in
   `pyproject.toml`, then merge any required changes into the master branch,
   then delete and re-create the git tag, and proceed to PyPI upload.

7. Verify that the `corresponding build <https://readthedocs.org/projects/mbirjax/builds/>`__ of the MBIRJAX documentation has built correctly.

Installing a specified branch
-----------------------------

Sometimes it's helpful to install a specific branch directly from github.  You can do this with one of the following commands (include [cuda12] to install on a GPU machine)::

    # For CPU installation:
    pip install git+https://github.com/cabouman/mbirjax.git@<branch_name>

    # For GPU installation:
    pip install "mbirjax[cuda12] @ git+https://github.com/cabouman/mbirjax.git@<branch_name>"


Reference
---------

More details can be found in the sources below.

  | [1] Packaging Python projects: `[link] <https://packaging.python.org/tutorials/packaging-projects/>`__
  | [2] Using TestPyPI: `[link] <https://packaging.python.org/guides/using-testpypi/>`__
