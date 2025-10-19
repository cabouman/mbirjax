.. docs-include-ref

DATA DEPENDENT TESTS
=======

The data dependent tests are a set of tests that rely on data that is not included in the mbirjax repo. To handle the situation where the data is inaccessible the data dependent tests have been set up so that they only run when explicitly evoked. This readme goes over the each of the files and the important things to note about them.

Running unit tests
--------------------

To run non—data dependent unit tests from CLI::

    pytest

To run data dependent unit tests::

    pytest -m data_dependent

``pytest.ini``
--------------------

This file is used to tell pytest which directories to look in for unit tests. This file tells it to look in tests and tests_data_dependent for unit tests.
It also defines the marker `data_dependent` which will be used to mark a test as one that requires exernal data.
The addopts setting allows the default behavior of not running tests that are marked as data_dependent.

.. code-block:: ini

    [pytest]
    norecursedirs = build venv nodist
    testpaths = tests tests_data_dependent
    markers =
        data_dependent: requires external data -- to run data dependent tests, run 'pytest -m data_dependent'
    addopts = -m "not data_dependent"

``conftest.py``
--------------------
The conftest python script is run any times the unit test scripts in the tests_data_dependednt dir is run. This script detects if the host of the running is pycharm. IF the host is pycharm and there are no explicite -m flags then it overrides the default behavior and includes the data_dependent tests. This allows running the test from a right click on the file in pycharm.

.. code-block:: python

    import os, sys

    def _explicit_m_passed(argv):
        return any(a == "-m" or a.startswith("-m=") for a in argv)

    def pytest_configure(config):
        if os.getenv("PYCHARM_HOSTED") == "1" and not _explicit_m_passed(sys.argv):
            config.option.markexpr = "data_dependent or not data_dependent"

``test_*.py``
--------------------

The test files have multiple

The classes defined in the test file are marked with the `@pytest.mark.data_dependent` decorator. This means these tests can be turned on an off using the `data_dependent` marker option.
The parent classes are marked with the class field __test__ = False so they are not discoverable and will not be run as unit tests. The subclasses are marked with __test__ = True so they are discoverable and will run when data_dependent tests are run.

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

GPU behavior:
The current behavior is to detect if gpus are present and only run the 'none' test if no gpus are present.
A warning will be generated if no gpus are present that not all test are able to run.

.. code-block:: python

    class ProjectionTestBase(unittest.TestCase):
        ...
        HAS_GPU = any(d.platform == "gpu" for d in jax.devices())
        USE_GPU_OPTS = ["automatic", "full", "sinograms", "projections", "none"] if HAS_GPU else ["none"]
        ...

``generate_test_data.py``
--------------------
This python script is used to generate the data that is used by the unit tests.