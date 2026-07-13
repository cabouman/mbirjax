"""Tests for the projector TilePolicy (TomographyModel._select_tile_policy).

The tile policy is the single decision site for the projector batching/banding knobs and
kernel-algorithm flags (see the TilePolicy note in tomography_model.py).  These tests pin:
the base policy's values (the long-standing defaults), the parallel-beam GPU override
(measured constants), the tiny-problem guard on the sorted-reduce flag, re-selection on
configure_devices, the _replace override idiom, and the loud guards on the retired attribute
names.
"""
import unittest

import numpy as np

import mbirjax
from mbirjax.projectors import SORTED_CHANNEL_REDUCE_MIN_COLS

from _platform import skip_unless_cpu, skip_unless_multidevice


def make_model(num_views=64, num_rows=40, num_channels=32):
    angles = np.linspace(0, np.pi, num_views, endpoint=False)
    return mbirjax.ParallelBeamModel((num_views, num_rows, num_channels), angles)


def centers_for(model_cls, idx, view_params, pp):
    """One view's concrete integer channel centers, as the projector wrappers compute them
    (the kernels take the centers as inputs; see compute_channel_coordinate)."""
    import jax.numpy as jnp
    return jnp.round(model_cls.compute_channel_coordinate(idx, view_params, pp)).astype(jnp.int32)


class TestTilePolicy(unittest.TestCase):

    @skip_unless_cpu
    def test_base_policy_on_cpu(self):
        # On CPU, parallel beam inherits the base policy untouched.  (A GPU applies the
        # parallel-beam overrides instead; that path is covered hardware-independently by
        # test_parallel_gpu_override.)
        model = make_model()
        model.configure_devices(1)
        t = model.tiles
        self.assertEqual(t.fwd_view_batch, 64)          # min(views, cap 128)
        self.assertEqual(t.back_view_batch, 64)
        self.assertEqual(t.fwd_pixel_batch, model._PIXEL_BATCH_DEFAULT)
        self.assertEqual(t.back_pixel_batch, model._PIXEL_BATCH_DEFAULT)
        self.assertIsNone(t.fwd_slice_band)
        self.assertIsNone(t.back_slice_band)
        self.assertFalse(t.sort_by_channel)
        self.assertFalse(t.back_stacked_gather)

    @skip_unless_multidevice
    def test_view_batch_caps(self):
        model = make_model(num_views=600)
        model.configure_devices(1)
        self.assertEqual(model.tiles.fwd_view_batch, 128)    # fwd cap
        self.assertEqual(model.tiles.back_view_batch, 128)   # single-device back cap
        model.configure_devices(2)
        self.assertEqual(model.tiles.fwd_view_batch, 128)
        self.assertEqual(model.tiles.back_view_batch, 300)   # per-shard single vmap (< 512 cap)

    def test_parallel_gpu_override(self):
        # Call the selection directly with on_gpu=True (no GPU needed): the measured parallel
        # forward tiling applies, and everything else stays at the base values.
        model = make_model(num_views=513, num_rows=449, num_channels=385)
        base = model._select_tile_policy(False, 513, 385, 1)
        gpu = model._select_tile_policy(True, 513, 385, 1)
        self.assertEqual(gpu.fwd_slice_band, model._FWD_SLICE_BAND_GPU)
        self.assertEqual(gpu.fwd_pixel_batch, model._FWD_PIXEL_BATCH_GPU)
        self.assertTrue(gpu.sort_by_channel)                 # bands >= the sorted-reduce minimum
        self.assertEqual(gpu.back_view_batch, base.back_view_batch)   # back untouched
        self.assertEqual(gpu.back_pixel_batch, base.back_pixel_batch)
        self.assertIsNone(gpu.back_slice_band)
        self.assertTrue(gpu.back_stacked_gather)             # stacked back gather on GPU
        self.assertFalse(base.back_stacked_gather)

    def test_sorted_flag_guard_on_tiny_problems(self):
        # When the balanced band width falls below the sorted-reduce minimum, the flag stays
        # off (the sort's fixed cost loses on narrow bands).
        model = make_model()
        tiny = model._select_tile_policy(True, 64, SORTED_CHANNEL_REDUCE_MIN_COLS - 8, 2)
        self.assertFalse(tiny.sort_by_channel)
        wide = model._select_tile_policy(True, 64, 8 * SORTED_CHANNEL_REDUCE_MIN_COLS, 2)
        self.assertTrue(wide.sort_by_channel)

    @skip_unless_multidevice
    def test_reselected_on_configure_devices_and_replace_override(self):
        model = make_model()
        model.configure_devices(1)
        self.assertEqual(model.tiles.back_view_batch, 64)
        model.configure_devices(2)
        self.assertEqual(model.tiles.back_view_batch, 32)    # per-shard
        # The experiment idiom: _replace persists until the next re-layout and is what the
        # late-bound projector wrappers read.
        model.tiles = model.tiles._replace(fwd_pixel_batch=512)
        self.assertEqual(model.tiles.fwd_pixel_batch, 512)

    def test_retired_attribute_names_raise(self):
        model = make_model()
        for name in ('view_batch_size_for_vmap', 'fwd_view_batch_size_for_vmap',
                     'back_view_batch_size_for_vmap', 'pixel_batch_size_for_vmap',
                     'transfer_pixel_batch_size'):
            with self.assertRaises(AttributeError):
                getattr(model, name)
            with self.assertRaises(AttributeError):
                setattr(model, name, 64)

    def test_kernel_sorted_branch_matches_scatter_branch(self):
        # The forward kernel's sort_by_channel branch (normally GPU-only) must produce the
        # same view as the scatter branch; runnable on CPU since both reductions are portable.
        from mbirjax.projectors import ProjectorParams
        model = make_model()
        model.configure_devices(1)
        pp_args = (tuple(model.get_params('sinogram_shape')),
                   tuple(model.get_params('recon_shape')), model.get_geometry_parameters())
        recon_shape = model.get_params('recon_shape')
        idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=False)
        rng = np.random.default_rng(1)
        vox = rng.random((len(idx), recon_shape[2]), dtype=np.float32)
        kern = mbirjax.ParallelBeamModel.forward_project_pixel_batch_to_one_view
        n_pc = centers_for(mbirjax.ParallelBeamModel, idx, np.float32(0.4), ProjectorParams(*pp_args, 0))
        out_scatter = np.asarray(kern(vox, idx, np.float32(0.4), n_pc, ProjectorParams(*pp_args, 0)))
        out_sorted = np.asarray(kern(vox, idx, np.float32(0.4), n_pc, ProjectorParams(*pp_args, 1)))
        np.testing.assert_allclose(out_sorted, out_scatter, rtol=1e-5,
                                   atol=1e-5 * np.abs(out_scatter).max())

    def test_back_kernel_stacked_branch_matches_loop(self):
        # The back kernel's stacked-gather branch (GPU) must match the per-tap loop, for both
        # coeff_power values (1 = back projection, 2 = the Hessian diagonal) and for a
        # row-sliced view (the sharded band path's shape).
        from mbirjax.projectors import ProjectorParams
        model = make_model()
        model.configure_devices(1)
        pp_args = (tuple(model.get_params('sinogram_shape')),
                   tuple(model.get_params('recon_shape')), model.get_geometry_parameters())
        pp_loop = ProjectorParams(*pp_args, 0, 0)
        pp_stacked = ProjectorParams(*pp_args, 0, 1)
        recon_shape = model.get_params('recon_shape')
        idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=False)
        rng = np.random.default_rng(4)
        num_rows, num_channels = model.get_params('sinogram_shape')[1:]
        view = rng.random((num_rows, num_channels), dtype=np.float32)
        kern = mbirjax.ParallelBeamModel.back_project_one_view_to_pixel_batch
        n_pc = centers_for(mbirjax.ParallelBeamModel, idx, np.float32(0.7), pp_loop)
        for coeff_power in (1, 2):
            for sl in (slice(None), slice(8, 24)):     # full view and a row-sliced band
                ref = np.asarray(kern(view[sl], idx, np.float32(0.7), n_pc, pp_loop, coeff_power))
                out = np.asarray(kern(view[sl], idx, np.float32(0.7), n_pc, pp_stacked, coeff_power))
                np.testing.assert_allclose(out, ref, rtol=1e-5,
                                           atol=1e-5 * np.abs(ref).max(),
                                           err_msg=f'coeff_power={coeff_power} slice={sl}')

    def test_cone_kernel_sorted_branch_matches_scatter_branch(self):
        # Cone's horizontal fan shares the channel reduction; its sorted branch must match the
        # scatter branch (per-pixel W_p_c/footprint arrays exercise the broadcast path that
        # parallel's scalar geometry does not).  Curved detector covers the other n_p formula.
        from mbirjax.projectors import ProjectorParams
        for curved in (False, True):
            angles = np.linspace(0, np.pi, 32, endpoint=False)
            model = mbirjax.ConeBeamModel((32, 40, 36), angles,
                                          source_detector_dist=4 * 36, source_iso_dist=2 * 36)
            if curved:
                model.set_params(use_curved_detector=True)
            model.configure_devices(1)
            pp_args = (tuple(model.get_params('sinogram_shape')),
                       tuple(model.get_params('recon_shape')), model.get_geometry_parameters())
            recon_shape = model.get_params('recon_shape')
            idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=False)
            rng = np.random.default_rng(2)
            vox = rng.random((len(idx), recon_shape[2]), dtype=np.float32)
            view_params = np.asarray(model.projector_functions.view_params_array)[3]
            kern = mbirjax.ConeBeamModel.forward_project_pixel_batch_to_one_view
            n_pc = centers_for(mbirjax.ConeBeamModel, idx, view_params, ProjectorParams(*pp_args, 0))
            out_scatter = np.asarray(kern(vox, idx, view_params, n_pc, ProjectorParams(*pp_args, 0)))
            out_sorted = np.asarray(kern(vox, idx, view_params, n_pc, ProjectorParams(*pp_args, 1)))
            np.testing.assert_allclose(out_sorted, out_scatter, rtol=1e-5,
                                       atol=1e-5 * np.abs(out_scatter).max(),
                                       err_msg=f'curved={curved}')

    def test_cone_gpu_policy_sets_sort_flag(self):
        angles = np.linspace(0, np.pi, 32, endpoint=False)
        model = mbirjax.ConeBeamModel((32, 96, 64), angles,
                                      source_detector_dist=4 * 64, source_iso_dist=2 * 64)
        num_slices = model.get_params('recon_shape')[2]
        gpu = model._select_tile_policy(True, 32, num_slices, 1)
        self.assertTrue(gpu.sort_by_channel)            # 96 detector rows >= the minimum
        cpu = model._select_tile_policy(False, 32, num_slices, 1)
        self.assertFalse(cpu.sort_by_channel)
        # Band inherited; pixel batch: default below the large-problem threshold, the measured
        # larger batch above it; back_stacked_gather deliberately NOT set (measured no-op for
        # cone -- the gather hides behind the vertical fan).
        self.assertIsNone(gpu.fwd_slice_band)
        self.assertEqual(gpu.fwd_pixel_batch, model._PIXEL_BATCH_DEFAULT)
        self.assertFalse(gpu.back_stacked_gather)
        big = model._select_tile_policy(True, 32, model._FWD_PIXEL_BATCH_MIN_SLICES, 1)
        self.assertEqual(big.fwd_pixel_batch, model._FWD_PIXEL_BATCH_GPU_LARGE)

    def test_multiaxis_kernel_sorted_branch_matches_scatter_branch(self):
        # Multiaxis' forward horizontal fan shares the channel reduction; its sorted branch
        # must match the scatter branch.  Nonzero elevations exercise the tilted vertical fan
        # feeding the reduce; the fan's weights are per-view SCALARS here (unlike cone's
        # per-pixel arrays), covering the scalar-broadcast case.
        from mbirjax.projectors import ProjectorParams
        angles = np.stack([np.linspace(0, np.pi, 16, endpoint=False), np.full(16, 0.3)], axis=1)
        model = mbirjax.MultiAxisParallelBeamModel((16, 96, 64), angles)
        model.configure_devices(1)
        pp_args = (tuple(model.get_params('sinogram_shape')),
                   tuple(model.get_params('recon_shape')), model.get_geometry_parameters())
        recon_shape = model.get_params('recon_shape')
        idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=False)
        rng = np.random.default_rng(5)
        vox = rng.random((len(idx), recon_shape[2]), dtype=np.float32)
        view_params = np.asarray(model.projector_functions.view_params_array)[3]
        kern = mbirjax.MultiAxisParallelBeamModel.forward_project_pixel_batch_to_one_view
        n_pc = centers_for(mbirjax.MultiAxisParallelBeamModel, idx, view_params, ProjectorParams(*pp_args, 0))
        out_scatter = np.asarray(kern(vox, idx, view_params, n_pc, ProjectorParams(*pp_args, 0)))
        out_sorted = np.asarray(kern(vox, idx, view_params, n_pc, ProjectorParams(*pp_args, 1)))
        np.testing.assert_allclose(out_sorted, out_scatter, rtol=1e-5,
                                   atol=1e-5 * np.abs(out_scatter).max())

    def test_multiaxis_gpu_policy_sets_sort_flag(self):
        angles = np.stack([np.linspace(0, np.pi, 16, endpoint=False), np.full(16, 0.2)], axis=1)
        model = mbirjax.MultiAxisParallelBeamModel((16, 96, 64), angles)
        num_slices = model.get_params('recon_shape')[2]
        gpu = model._select_tile_policy(True, 16, num_slices, 1)
        self.assertTrue(gpu.sort_by_channel)            # 96 detector rows >= the minimum
        cpu = model._select_tile_policy(False, 16, num_slices, 1)
        self.assertFalse(cpu.sort_by_channel)
        # Tiny detectors stay on the scatter path; back_stacked_gather deliberately NOT set
        # (measured composition no-op behind the vertical fan, as for cone).
        tiny = mbirjax.MultiAxisParallelBeamModel(
            (16, SORTED_CHANNEL_REDUCE_MIN_COLS - 8, 64), angles)
        self.assertFalse(tiny._select_tile_policy(
            True, 16, tiny.get_params('recon_shape')[2], 1).sort_by_channel)
        self.assertFalse(gpu.back_stacked_gather)
        # The channel-collision guard: at a WIDE detector the collision ratio
        # psf_width * pixel_batch / channels drops below the measured cliff threshold and
        # the flag stays off (the translation TCT lesson, applied to multiaxis' policy).
        wide = mbirjax.MultiAxisParallelBeamModel((16, 96, 3064), angles)
        self.assertFalse(wide._select_tile_policy(
            True, 16, wide.get_params('recon_shape')[2], 1).sort_by_channel)

    @staticmethod
    def _make_translation_model(num_det_rows=64, pitch_factor=1.5):
        # High magnification (source close to iso) is the large-cone-angle regime; scaling
        # delta_voxel up from the auto value pushes psf_radius into the wide-psf range
        # translation commonly runs at (the auto geometry here has a large voxel_row_aspect,
        # so magnification grows quickly with the pitch: 1.25 -> radius 2, 1.5 -> radius 3).
        from mbirjax.translation_model import TranslationModel
        rng = np.random.default_rng(3)
        translations = rng.uniform(-2.0, 2.0, (12, 3)).astype(np.float32)
        model = TranslationModel((12, num_det_rows, 36), translations,
                                 source_detector_dist=120.0, source_iso_dist=30.0)
        model.auto_set_recon_geometry()
        model.set_params(delta_voxel=pitch_factor * model.get_params('delta_voxel'))
        return model

    def test_translation_kernel_sorted_branch_matches_scatter_branch(self):
        # Translation's forward horizontal fan: per-pixel W_p_c / cos_theta_p (pixel-dependent
        # magnification) exercise the broadcast path, and translation is the WIDE-psf geometry
        # (large cone angle), so run at psf_radius 2 AND 3 (psf_width 5 / 7) -- the crossover
        # constant and the fused stacked transient were only measured at psf_width 3.
        from mbirjax.projectors import ProjectorParams
        from mbirjax.translation_model import TranslationModel
        for pitch_factor, expected_radius in ((1.25, 2), (1.5, 3)):
            model = self._make_translation_model(pitch_factor=pitch_factor)
            self.assertEqual(model.get_psf_radius(), expected_radius)
            model.configure_devices(1)
            pp_args = (tuple(model.get_params('sinogram_shape')),
                       tuple(model.get_params('recon_shape')), model.get_geometry_parameters())
            recon_shape = model.get_params('recon_shape')
            idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=False)
            rng = np.random.default_rng(6)
            vox = rng.random((len(idx), recon_shape[2]), dtype=np.float32)
            view_params = np.asarray(model.projector_functions.view_params_array)[5]
            kern = TranslationModel.forward_project_pixel_batch_to_one_view
            n_pc = centers_for(TranslationModel, idx, view_params, ProjectorParams(*pp_args, 0))
            out_scatter = np.asarray(kern(vox, idx, view_params, n_pc, ProjectorParams(*pp_args, 0)))
            out_sorted = np.asarray(kern(vox, idx, view_params, n_pc, ProjectorParams(*pp_args, 1)))
            np.testing.assert_allclose(out_sorted, out_scatter, rtol=1e-5,
                                       atol=1e-5 * np.abs(out_scatter).max(),
                                       err_msg=f'psf_radius={expected_radius}')

    def test_mt_back_kernel_stacked_branch_matches_loop(self):
        # The shared horizontal_fan_back helper gives every geometry the stacked-gather
        # branch (parallel is the only policy that ENABLES it, but the branch must be
        # value-correct everywhere -- it is reachable via the tiles override).  Multiaxis
        # covers the scalar-weight case; translation covers per-pixel weights at WIDE psf.
        # Both coeff_powers; tolerance, not exactness (tap-summation order differs).
        from mbirjax.projectors import ProjectorParams
        angles = np.stack([np.linspace(0, np.pi, 12, endpoint=False), np.full(12, 0.25)], axis=1)
        ma = mbirjax.MultiAxisParallelBeamModel((12, 72, 48), angles)
        tr = self._make_translation_model(pitch_factor=1.5)      # psf_radius 3
        for model, view_idx in ((ma, 2), (tr, 5)):
            model.configure_devices(1)
            pp_args = (tuple(model.get_params('sinogram_shape')),
                       tuple(model.get_params('recon_shape')), model.get_geometry_parameters())
            recon_shape = model.get_params('recon_shape')
            idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=False)
            rng = np.random.default_rng(8)
            num_rows, num_channels = model.get_params('sinogram_shape')[1:]
            view = rng.random((num_rows, num_channels), dtype=np.float32)
            vp = np.asarray(model.projector_functions.view_params_array)[view_idx]
            kern = type(model).back_project_one_view_to_pixel_batch
            n_pc = centers_for(type(model), idx, vp, ProjectorParams(*pp_args, 0, 0))
            for coeff_power in (1, 2):
                ref = np.asarray(kern(view, idx, vp, n_pc, ProjectorParams(*pp_args, 0, 0), coeff_power))
                out = np.asarray(kern(view, idx, vp, n_pc, ProjectorParams(*pp_args, 0, 1), coeff_power))
                np.testing.assert_allclose(out, ref, rtol=1e-5,
                                           atol=1e-5 * np.abs(ref).max(),
                                           err_msg=f'{type(model).__name__} cp={coeff_power}')

    def test_translation_gpu_policy_keeps_scatter(self):
        # Translation's policy deliberately does NOT set sort_by_channel: at the REAL TCT
        # detector shapes (1936x3064-class) the sorted reduce measured 4.5-6.5x SLOWER (the
        # channel-collision cliff; see TranslationModel._select_tile_policy).  The kernel
        # branch exists (tested above) but only an explicit experiment override enables it.
        for pitch_factor in (1.25, 1.5):
            model = self._make_translation_model(num_det_rows=64, pitch_factor=pitch_factor)
            num_slices = model.get_params('recon_shape')[2]
            gpu = model._select_tile_policy(True, 12, num_slices, 1)
            self.assertFalse(gpu.sort_by_channel)
            self.assertFalse(gpu.back_stacked_gather)
            # Everything else inherits the base policy untouched.
            base = super(type(model), model)._select_tile_policy(True, 12, num_slices, 1)
            self.assertEqual(gpu, base)

    def test_get_compute_config(self):
        # The introspection contract: all sections present, tiles mirrored verbatim,
        # pallas state consistent with availability, and print_results doesn't raise.
        model = make_model()
        model.configure_devices(1)
        config = model.get_compute_config(print_results=True)
        self.assertEqual(set(config),
                         {'versions', 'devices', 'tiles', 'kernels', 'jit_cache'})
        self.assertEqual(config['tiles'], dict(model.tiles._asdict()))
        self.assertEqual(config['devices']['count'], 1)
        kernels = config['kernels']
        from mbirjax import _pallas_kernels
        self.assertEqual(kernels['pallas_available'], _pallas_kernels.is_available())
        if not kernels['pallas_available']:
            # On CPU CI the reason must be populated and the paths off -- and the
            # device line must NOT carry a pallas token.
            self.assertTrue(len(kernels['pallas_status']) > 0)
            self.assertFalse(kernels['back_pallas'] or kernels['fwd_pallas'])
            self.assertNotIn('pallas', model.device_summary)
        else:
            self.assertIn('pallas', model.device_summary)

    def test_projection_runs_with_replaced_tiles(self):
        # End-to-end smoke: an overridden tile policy still projects correctly (values equal
        # to the default-policy projection up to reordering).
        model = make_model()
        model.configure_devices(1)
        recon_shape = model.get_params('recon_shape')
        idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=False)
        rng = np.random.default_rng(0)
        vox = rng.random((len(idx), recon_shape[2]), dtype=np.float32)
        ref = np.asarray(model.sparse_forward_project(vox, idx))
        model.tiles = model.tiles._replace(fwd_view_batch=16, fwd_pixel_batch=256)
        out = np.asarray(model.sparse_forward_project(vox, idx))
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5 * np.abs(ref).max())


if __name__ == '__main__':
    unittest.main()
