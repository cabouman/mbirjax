==========================================
Adding a Data-Dependent Unit Test
==========================================

1. START FROM AN EXISTING TEMPLATE
==================================

Copy the mixin + subclass pattern from ``tests_data_dependent/test_projection.py`` or ``tests_data_dependent/test_recon.py``. Keep reusable logic inside a mixin class whose name does **not** begin with ``Test``. Create concrete subclasses that inherit the mixin **and** ``unittest.TestCase`` so pytest only collects the subclasses.

2. DECLARE REQUIRED CLASS ATTRIBUTES
====================================

Each concrete test class must set:

- ``MODEL``: a class with ``.from_file`` (for example, ``mj.ConeBeamModel``).
- ``SOURCE_FILEPATH``: URL or path to a ``.tgz`` archive containing ``phantom``, ``sinogram``, ``recon``, and ``params`` datasets.
- ``DATA_FILE_SHA256``: checksum for the archive (see Section 4 below).
- ``TOLERANCES``: only for recon tests; a dict defining ``nrmse``, ``max_diff``, and ``pct_95`` thresholds.

Always decorate the concrete subclasses with ``@pytest.mark.data_dependent`` so they stay opt-in during regular CI runs.

3. PREPARE OR UPDATE THE DATASET
================================

If the required archive already exists, just point ``SOURCE_FILEPATH`` at it. To build new fixtures, run:

.. code-block:: bash

   python tests_data_dependent/_test_data_dependent_utils.py

The helper script writes fresh ``.tgz`` datasets under ``/depot/bouman/data/unit_test_data`` mirroring the layout expected by the tests. Upload or publish the generated archive and update ``SOURCE_FILEPATH`` accordingly.

4. GET THE SHA-256 FOR THE DATA ARCHIVE
=======================================

Follow the checksum workflow from ``README_DATA_DEPENDENT.rst``:

1. Put any placeholder string in ``DATA_FILE_SHA256``.
2. Run the test once with ``pytest -m data_dependent tests_data_dependent/test_your_module.py``.
3. The run will warn if the checksum is wrong, showing the **actual** hash; copy that value.
4. Replace the placeholder with the reported hash so future runs verify the dataset integrity.

You can regenerate the hash manually by calling ``sha256_file`` from ``tests_data_dependent/_test_data_dependent_utils.py`` if you already have the archive locally.

5. EXECUTE THE SUITE
====================

Run the targeted module to confirm the new tests pass and the checksum matches:

.. code-block:: bash

   pytest -m data_dependent tests_data_dependent/test_your_module.py

When GPUs are available, the mixins iterate through the ``USE_GPU_OPTS`` list; otherwise they fall back to CPU-only execution. Review the logs in ``tests_data_dependent/logs`` if recon tests write additional output.
