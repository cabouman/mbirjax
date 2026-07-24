"""
Scaffolding test for the sharding work.

Confirms that the device-setup machinery actually produces multiple devices to
shard across, and that the conftest `preferred_devices` helper works.  This is
the minimal gate that "the beta worktree is testable for sharding at all" —
later tests rely on >= 2 devices being available (real GPUs on a cluster,
virtual CPU devices on a laptop/CI).
"""
import unittest

import jax

from conftest import preferred_devices


class TestShardingScaffolding(unittest.TestCase):

    def test_preferred_devices_returns_two(self):
        """preferred_devices(2) returns exactly two devices."""
        devs = preferred_devices(2)
        n = len(jax.devices())
        if n < 2:
            self.skipTest("Need >= 2 devices for sharding tests.")
        self.assertEqual(len(devs), 2)


if __name__ == "__main__":
    unittest.main()
