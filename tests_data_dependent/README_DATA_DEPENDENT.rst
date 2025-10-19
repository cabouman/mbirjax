.. docs-include-ref

Data-Dependent Tests
====================

The data-dependent tests rely on external datasets that are not stored in the
``mbirjax`` repo. To keep regular CI runs fast and resilient, these tests only
run when explicitly requested. This README describes how to run them and what
each file does.

Running tests
-------------

To run non–data-dependent unit tests from the CLI::

   pytest

   ...
   configfile: pytest.ini
   testpaths: tests, tests_data_dependent
   collected 37 items / 18 deselected / 19 selected
   ...

To run **only** the data-dependent tests::

   pytest -m data_dependent

   ...
   configfile: pytest.ini
   testpaths: tests, tests_data_dependent
   collected 37 items / 19 deselected / 18 selected
   ...

To run **all** tests (overriding the default exclude)::

   pytest -m "data_dependent or not data_dependent"

   ...
   configfile: pytest.ini
   testpaths: tests, tests_data_dependent
   collected 37 items
   ...

``pytest.ini``
--------------

This file configures pytest to discover tests in both ``tests`` and
``tests_data_dependent``. It declares the ``data_dependent`` marker and sets the
default behavior to skip those tests unless explicitly enabled.

.. code-block:: ini

   [pytest]
   norecursedirs = build venv nodist
   testpaths = tests tests_data_dependent
   markers =
       data_dependent: requires external data; run with 'pytest -m data_dependent'
   addopts = -m "not data_dependent"


``tests_data_dependent/conftest.py``
------------------------------------

This hook makes right-click ▶ Run in PyCharm execute **all tests in the file**
(even if they’re marked ``data_dependent``) when no explicit ``-m`` is provided,
while leaving CLI defaults unchanged.

.. code-block:: python

   import os, sys

   def _explicit_m_passed(argv):
       return any(a == "-m" or a.startswith("-m=") for a in argv)

   def pytest_configure(config):
       if os.getenv("PYCHARM_HOSTED") == "1" and not _explicit_m_passed(sys.argv):
           config.option.markexpr = "data_dependent or not data_dependent"


Test modules (``test_*.py``)
----------------------------

- Tests that need external data are marked with ``@pytest.mark.data_dependent``.
- Abstract/base classes set ``__test__ = False`` so they are not collected.
  Concrete subclasses set ``__test__ = True`` and will be collected.

Example:

.. code-block:: python

   class ProjectionTestBase(unittest.TestCase):
       ...
       __test__ = False
       ...

   @pytest.mark.data_dependent
   class TestProjectionCone(ProjectionTestBase):
       __test__ = True
       MODEL = mj.ConeBeamModel
       SOURCE_FILEPATH = "https://www.datadepot.rcac.purdue.edu/bouman/data/unit_test_data/cone_32_projection_data.tgz"

   @pytest.mark.data_dependent
   class TestProjectionParallel(ProjectionTestBase):
       __test__ = True
       MODEL = mj.ParallelBeamModel
       SOURCE_FILEPATH = "https://www.datadepot.rcac.purdue.edu/bouman/data/unit_test_data/parallel_32_projection_data.tgz"


GPU behavior
------------

If GPUs are available, the suite exercises GPU modes; otherwise it falls back to
the ``none`` mode and emits a warning that not all tests could run.

.. code-block:: python

   class ProjectionTestBase(unittest.TestCase):
       ...
       HAS_GPU = any(d.platform == "gpu" for d in jax.devices())
       USE_GPU_OPTS = ["automatic", "full", "sinograms", "projections", "none"] if HAS_GPU else ["none"]
       ...


``tests_data_dependent/generate_test_data.py``
----------------------------------------------

Utility script to generate the datasets consumed by the data-dependent tests.
Run it to (re)build local copies of the required fixtures before executing the
suite.
