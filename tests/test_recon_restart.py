"""Restart reproducibility: a chunked recon must reproduce a continuous run exactly.

Two properties, both enabled by gen_pixel_partition's num_subsets==1 early return (which
stops full-index 'partitions' -- the Hessian diagonal's and the direct-recon init's
gen_full_indices calls -- from consuming global np.random state):

  1. gen_pixel_partition(num_subsets=1) is RNG-NEUTRAL and returns the sorted indices
     (bit-identical to the old permute-then-sort path).
  2. Stepping vcd_recon one iteration at a time (caller-held partitions, seed set ONCE,
     no re-seeding between chunks) matches a single multi-iteration vcd_recon call to
     iterated-fp noise.  Without the RNG-neutrality fix, each restart's Hessian shifted
     the permutation stream and the paths diverged at the percent level (measured
     2026-07-04 on the partition-sequence study harness).
"""
import unittest

import numpy as np
import jax.numpy as jnp

import mbirjax


class TestReconRestart(unittest.TestCase):

    def test_single_subset_partition_is_rng_neutral_and_sorted(self):
        recon_shape = (16, 16, 8)
        np.random.seed(3)
        state_before = np.random.get_state()
        partition = np.asarray(mbirjax.gen_pixel_partition(recon_shape, num_subsets=1))
        state_after = np.random.get_state()
        # RNG-neutral: no global state consumed.
        self.assertTrue(all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
                            for a, b in zip(state_before, state_after)))
        # Sorted single subset, same contents as the RoR mask's indices.
        self.assertEqual(partition.shape[0], 1)
        self.assertTrue(np.all(np.diff(partition[0]) > 0))

    def test_chunked_vcd_recon_matches_continuous(self):
        num_views, n = 24, 32
        angles = jnp.asarray(np.linspace(0, np.pi, num_views, endpoint=False))
        model = mbirjax.ParallelBeamModel((num_views, n, n), angles)
        model.set_params(verbose=0, partition_sequence=[0, 2, 4])
        rng = np.random.default_rng(0)
        recon_shape = tuple(int(x) for x in model.get_params('recon_shape'))
        phantom = jnp.asarray(rng.standard_normal(recon_shape, dtype=np.float32))
        sino = np.asarray(model.forward_project(phantom))

        def setup():
            # Mirror TomographyModel.initialize_recon: seed ONCE, then partitions.
            np.random.seed(0)
            partitions = mbirjax.gen_set_of_pixel_partitions(
                recon_shape, model.get_params('granularity'),
                output_device=model.recon_placement.devices[0],
                use_ror_mask=model.get_params('use_ror_mask'))
            seq = np.asarray(mbirjax.gen_partition_sequence([0, 2, 4], max_iterations=3))
            model._log_run_header(0, '~/.mbirjax/logs/recon.log', print_logs=False)
            model.auto_set_regularization_params(sino)
            return partitions, seq

        partitions, seq = setup()
        mono, _ = model.vcd_recon(sino, partitions, seq, 0)
        mono = np.asarray(mono)

        partitions, seq = setup()
        recon = None
        for k in range(3):
            # NO re-seeding between chunks: the permutation stream must continue.
            recon, _ = model.vcd_recon(sino, partitions, seq[k:k + 1], 0,
                                       init_recon=recon, first_iteration=k)
        rel = float(np.max(np.abs(np.asarray(recon) - mono)) / np.max(np.abs(mono)))
        self.assertLess(rel, 1e-5, msg=f'chunked vs continuous rel_max {rel:.2e}')

        # Checkpoint-threaded chunking (init_error_sinogram + fm_hessian round-tripped via
        # return_checkpoint) must ALSO match -- and it skips the per-restart forward
        # projection and Hessian back projection, so it is the cheap form.
        partitions, seq = setup()
        recon, ckpt = None, {}
        for k in range(3):
            recon, _, ckpt = model.vcd_recon(sino, partitions, seq[k:k + 1], 0,
                                             init_recon=recon, first_iteration=k,
                                             init_error_sinogram=ckpt.get('error_sinogram'),
                                             fm_hessian=ckpt.get('fm_hessian'),
                                             return_checkpoint=True)
        rel = float(np.max(np.abs(np.asarray(recon) - mono)) / np.max(np.abs(mono)))
        self.assertLess(rel, 1e-5, msg=f'checkpointed vs continuous rel_max {rel:.2e}')

        # Guard: an error sinogram without its recon is an inconsistent resume state.
        with self.assertRaises(ValueError):
            model.vcd_recon(sino, partitions, seq[0:1], 0,
                            init_error_sinogram=ckpt['error_sinogram'])

    def test_consumed_checkpoint_raises_informative_error(self):
        """A checkpoint is SINGLE-USE: the resumed loop DONATES the checkpoint's error-sinogram
        buffer for its in-place updates, deleting the caller-visible array.  A second resume
        from the same checkpoint must fail AT ENTRY with guidance -- not with the opaque
        'Array has been deleted' the donated update would raise mid-loop."""
        num_views, n = 24, 32
        angles = jnp.asarray(np.linspace(0, np.pi, num_views, endpoint=False))
        model = mbirjax.ParallelBeamModel((num_views, n, n), angles)
        model.set_params(verbose=0, partition_sequence=[0, 2])
        rng = np.random.default_rng(0)
        recon_shape = tuple(int(x) for x in model.get_params('recon_shape'))
        phantom = jnp.asarray(rng.standard_normal(recon_shape, dtype=np.float32))
        sino = np.asarray(model.forward_project(phantom))
        np.random.seed(0)
        partitions = mbirjax.gen_set_of_pixel_partitions(
            recon_shape, model.get_params('granularity'),
            output_device=model.recon_placement.devices[0],
            use_ror_mask=model.get_params('use_ror_mask'))
        seq = np.asarray(mbirjax.gen_partition_sequence([0, 2], max_iterations=2))
        model._log_run_header(0, '~/.mbirjax/logs/recon.log', print_logs=False)
        model.auto_set_regularization_params(sino)

        recon0, _, ckpt = model.vcd_recon(sino, partitions, seq[0:1], 0, return_checkpoint=True)
        # The first resume works -- and consumes the checkpoint's error sinogram.
        model.vcd_recon(sino, partitions, seq[1:2], 0, init_recon=recon0, first_iteration=1,
                        init_error_sinogram=ckpt['error_sinogram'],
                        fm_hessian=ckpt['fm_hessian'])
        self.assertTrue(ckpt['error_sinogram'].is_deleted())   # the documented consume semantics
        # A second resume from the SAME checkpoint fails at entry with guidance.
        with self.assertRaisesRegex(ValueError, 'SINGLE-USE'):
            model.vcd_recon(sino, partitions, seq[1:2], 0, init_recon=recon0, first_iteration=1,
                            init_error_sinogram=ckpt['error_sinogram'],
                            fm_hessian=ckpt['fm_hessian'])


if __name__ == '__main__':
    unittest.main()
