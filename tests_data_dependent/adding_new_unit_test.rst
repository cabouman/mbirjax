==========================================
Adding a Data-Dependent Unit Test
==========================================

1. CREATING NEW UNIT TEST CLASSES
=================================

Reference the patterns used in previous tests in the ``tests_data_dependent`` directory.

If creating a base class:

- Do NOT start the class name with ``Test`` so the base mixin is not collected by ``pytest``.
- Do NOT inherit from ``unittest.TestCase`` so the base mixin is not collected by ``unittest``.

If creating a concrete subclass:

- DO inherit from ``unittest.TestCase`` and start the class name with ``Test`` so the test is collected.
- DO start every unit test method with ``test`` so pytest collects the method.
- DO decorate the class with ``@pytest.mark.data_dependent`` so it is not run by default.
- The ``setUpClass`` method runs only once before all unit tests, so use it to download the relevant data once for the suite.
- The ``tearDownClass`` method runs only once after all unit tests, so use it to clean up the data.
- The ``setUp`` method runs before each unit test method, so use it to reinitialize test data when needed.
- ``with self.subTest(...)`` lets you iterate through test variations (for example, all ``use_gpu`` options).


1. PREPARE OR UPDATE THE DATASET
================================

If the required archive already exists, just point ``SOURCE_FILEPATH`` at it. To build new fixtures, run:

.. code-block:: bash

   python tests_data_dependent/_test_data_dependent_utils.py

The helper script writes fresh ``.tgz`` datasets under ``/depot/bouman/data/unit_test_data`` mirroring the layout expected by the tests. Upload or publish the generated archive and update ``SOURCE_FILEPATH`` accordingly.

3. GET THE SHA-256 FOR THE DATA ARCHIVE
=======================================

Follow the checksum workflow from ``README_DATA_DEPENDENT.rst``:

1. Put any placeholder string in ``DATA_FILE_SHA256``.
2. Run the test once with ``pytest -m data_dependent tests_data_dependent/test_your_module.py``.
3. The run warns if the checksum is wrong, showing the **actual** hash; copy that value.
4. Replace the placeholder with the reported hash so future runs verify the dataset integrity.

You can regenerate the hash manually by calling ``sha256_file`` from ``tests_data_dependent/_test_data_dependent_utils.py`` if you already have the archive locally.

4. EXECUTE THE SUITE
====================

Run the targeted module to confirm the new tests pass and the checksum matches:

.. code-block:: bash

   pytest -m data_dependent tests_data_dependent/test_your_module.py

When GPUs are available, the mixins iterate through the ``USE_GPU_OPTS`` list; otherwise they fall back to CPU-only execution. Review the logs in ``tests_data_dependent/logs`` if recon tests write additional output.
