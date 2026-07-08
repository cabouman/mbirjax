"""Shared platform/device skip markers for the test suite.

Phase 4 of the platform-helper refactor: one place for "should this test run on the current
box," built on the library's ``get_platform()`` / ``default_devices()`` so tests never
re-derive the platform.  ``unittest.skipIf`` (not ``pytest.mark.skipif``) because the suite is
``unittest.TestCase``-based -- this works under both pytest and bare unittest, and the
condition is evaluated once at import (the platform/device count is fixed per process).

Importable from ``tests/`` and ``tests/sharding/`` alike: the top-level ``tests/conftest.py`` is
loaded for every run under ``testpaths = tests`` (including an isolated ``pytest tests/sharding/``),
which puts ``tests/`` on ``sys.path``.

Usage::

    from _platform import skip_unless_cpu, skip_unless_multidevice

    @skip_unless_cpu
    def test_cpu_only_behavior(self): ...
"""
import unittest

from mbirjax import get_platform
from mbirjax._device_setup import default_devices

_PLATFORM = get_platform()                 # 'CPU' / 'GPU' / 'TPU' -- the default run's platform
_NUM_DEVICES = len(default_devices())       # GPUs if present, else the (virtual) CPU devices

skip_unless_cpu = unittest.skipIf(
    _PLATFORM != 'CPU', 'CPU-only behavior (a GPU applies platform overrides)')
skip_unless_gpu = unittest.skipIf(
    _PLATFORM != 'GPU', 'needs a real GPU')
skip_unless_multidevice = unittest.skipIf(
    _NUM_DEVICES < 2, 'needs >= 2 devices for the multi-device path '
                      '(a single-GPU allocation has one; CPU CI has several virtual devices)')
