Package Maintenance
===================

The following lists out procedures for basic package maintenance.

Unit Tests
----------

In order to run unit tests, install MBIRJAX using the provided install scripts, and from the root directory of the repository, activate the conda environment, and then run the following::

    pytest

This should be repeated for each supported platform.

Uploading to Test PyPI
----------------------

This is only available for registered maintainers.

0. First, make sure you have installed the newest versions of `setuptools`, `wheel`, `build`, and `twine`. Then from the main mbirjax directory, delete any previous build and then build the project::

    rm -r dist
    python -m build

1. Upload to Test PyPI. You will need to get an API token from TestPyPI. You will be prompted for this token from the command line. NOTE: You cannot upload the same version more than once::

    python -m twine upload --repository testpypi dist/*

   View the package upload here:
   `https://test.pypi.org/project/mbirjax <https://test.pypi.org/project/mbirjax>`__

2. Test the uploaded package::

    pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple mbirjax==0.1.1  # change version no.
    python -c "import mbirjax"     # spin the wheel

3. Run one of the `demo scripts <examples.html>`_

   NOTE: If the install fails and you need to re-test, *temporarily* set
   the version number in `pyproject.toml` from X.X.X to X.X.X.1 (then 2, 3, etc.),
   for further testing. After the test is successful, reset the version number in
   `pyproject.toml`, then merge any required changes into the master branch,
   then delete and re-create the git tag, and proceed to PyPI upload.


Upload to PyPI
--------------

This is only available for registered maintainers.

1. Upload to PyPI.  As above, you will need an API token, this time from PyPI.  NOTE: You cannot upload the same version more than once::

    python -m twine upload dist/*

   View the package upload here:
   `https://pypi.org/project/mbirjax <https://pypi.org/project/mbirjax>`__

2. Test the uploaded package::

    pip install mbirjax    # OR, "mbirjax==0.1.1" e.g. for a specific version number
    python -c "import mbirjax"     # spin the wheel

3. Run one of the `demo scripts <examples.html>`_

Reference
---------

More details can be found in the sources below.

  | [1] Packaging Python projects: `[link] <https://packaging.python.org/tutorials/packaging-projects/>`__
  | [2] Using TestPyPI: `[link] <https://packaging.python.org/guides/using-testpypi/>`__
