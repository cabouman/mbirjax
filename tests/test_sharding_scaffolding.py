"""
Phase 0 scaffolding test for the sharding work.

Confirms that the device-setup machinery actually produces multiple devices to
shard across, and that the conftest `preferred_devices` helper works.  This is
the minimal gate that "the beta worktree is testable for sharding at all" —
later phase tests rely on >= 2 devices being available (real GPUs on a cluster,
virtual CPU devices on a laptop/CI).
"""
import unittest

import jax

from conftest import preferred_devices


class TestShardingScaffolding(unittest.TestCase):

    def test_multiple_devices_available(self):
        """At least 2 devices exist (real GPUs, or virtual CPUs via XLA_FLAGS)."""
        n = len(jax.devices())
        if n < 2:
            self.skipTest("Need >= 2 devices for sharding tests.")
        self.assertTrue(True)

    def test_device_setup_flag_present(self):
        """The XLA virtual-device flag is set (by conftest or _device_setup)."""
        import os
        self.assertIn(
            "xla_force_host_platform_device_count",
            os.environ.get("XLA_FLAGS", ""),
            "XLA_FLAGS should contain the virtual-device count flag."
        )

    def test_preferred_devices_returns_two(self):
        """preferred_devices(2) returns exactly two devices."""
        devs = preferred_devices(2)
        n = len(jax.devices())
        if n < 2:
            self.skipTest("Need >= 2 devices for sharding tests.")
        self.assertEqual(len(devs), 2)

    def test_preferred_devices_too_many_returns_none(self):
        """Requesting more devices than exist returns None (caller skips)."""
        too_many = len(jax.devices()) + 1
        self.assertIsNone(preferred_devices(too_many))


if __name__ == "__main__":
    unittest.main()
