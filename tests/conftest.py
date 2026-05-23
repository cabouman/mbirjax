"""
pytest configuration for the mbirjax test suite.

Sets XLA_FLAGS before JAX is first imported so that sharding tests can run
on CPU with multiple virtual devices.  setdefault is used so that a real-GPU
environment (or CI that already sets XLA_FLAGS) is not overridden.
"""
import os

# 4 virtual CPU devices → sharding tests can use 2 of them.
# This must happen before any test module imports jax.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
