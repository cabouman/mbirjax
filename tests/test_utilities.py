import warnings
import numpy as np
import jax.numpy as jnp
import mbirjax as mj
import unittest


class TestUtilities(unittest.TestCase):
    """
    Test various utilities
    """

    def setUp(self):
        """Set up before each test method."""
        pass

    def tearDown(self):
        """Clean up after each test method."""
        pass

    def test_parameter_names(self):

        print('Testing consistency of parameter names in ParameterHandler')
        consistent = mj._utils.update_param_literal(verify_match_and_exit=True)
        if not consistent:
            warnings.warn('Run mbirjax._utils.update_param_literal() to update ParamNames in ParameterHandler')
        assert consistent

    def test_merge_log_files(self):

        print('Testing merge_log_files')
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            merged_path = os.path.join(tmp_dir, 'merged.log')
            path_a = os.path.join(tmp_dir, 'a.log')
            path_b = os.path.join(tmp_dir, 'b.log')
            with open(path_a, 'w') as f:
                f.write('content a\n')
            with open(path_b, 'w') as f:
                f.write('content b\n')

            # Missing temps are skipped; existing ones appear in order under their headers, then are removed.
            mj.merge_log_files(merged_path, [('first', path_a), ('missing', path_a + '.nope'), ('second', path_b)])
            with open(merged_path, 'r') as f:
                text = f.read()
            self.assertEqual(text, '======== first ========\ncontent a\n'
                                   '======== second ========\ncontent b\n')
            self.assertNotIn('missing', text)
            self.assertFalse(os.path.exists(path_a))
            self.assertFalse(os.path.exists(path_b))

            # No temps at all (including None paths): no output file is written.
            merged_path_2 = os.path.join(tmp_dir, 'merged2.log')
            mj.merge_log_files(merged_path_2, [('first', path_a), ('none', None)])
            self.assertFalse(os.path.exists(merged_path_2))


class TestExportReconHostResidence(unittest.TestCase):
    """The HDF5 export path must stay on the HOST for a host recon, so a large volume is never copied
    back onto a single device (the OOM Charlie hit at downsampling 1, f32[1370,1880,1880] ~ 18 GiB).

    This is a host-RESIDENCE guard (the real failure is placement, deterministic at any size), not an
    actual large run: feed a small host recon and assert nothing is promoted to a jax device, plus a
    small export -> import round-trip.
    """

    def test_apply_cylindrical_mask_host_in_host_out(self):
        import mbirjax.preprocess as mjp
        recon = np.random.RandomState(0).rand(16, 16, 8).astype(np.float32)
        out = mjp.apply_cylindrical_mask(recon, radial_margin=2, top_margin=1, bottom_margin=1)
        self.assertIsInstance(out, np.ndarray)             # host in -> host out (no device promotion)
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_array_equal(out[:, :, 0], 0)     # top margin zeroed
        np.testing.assert_array_equal(out[:, :, -1], 0)    # bottom margin zeroed
        self.assertEqual(float(out[0, 0, 4]), 0.0)         # a corner outside the cylinder is zeroed

    def test_apply_cylindrical_mask_jax_in_jax_out(self):
        import jax
        import mbirjax.preprocess as mjp
        recon = jnp.asarray(np.random.RandomState(1).rand(16, 16, 8).astype(np.float32))
        out = mjp.apply_cylindrical_mask(recon)
        self.assertIsInstance(out, jax.Array)              # jax in -> jax out (on-device preserved)

    def test_export_import_roundtrip_host(self):
        import os, tempfile
        recon = np.random.RandomState(2).rand(12, 10, 6).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'recon.h5')
            mj.export_recon_hdf5(path, recon, recon_dict=None, remove_flash=True)
            self.assertTrue(os.path.exists(path))
            loaded, _ = mj.import_recon_hdf5(path)
            self.assertIsInstance(loaded, np.ndarray)      # host array back
            self.assertEqual(loaded.shape, recon.shape)    # round-trips to (row, col, slice)

    def test_export_import_roundtrip_values(self):
        # No flash mask, so the round-trip must be EXACT: catches any re-introduced
        # axis flip or transpose mismatch between export and import (the import-side
        # slice-axis reversal removed in e80f4d0 would fail this test).
        import os, tempfile
        recon = np.random.RandomState(3).rand(12, 10, 6).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'recon.h5')
            mj.export_recon_hdf5(path, recon, recon_dict=None)
            loaded, _ = mj.import_recon_hdf5(path)
            np.testing.assert_array_equal(loaded, recon)


class TestIsOomClassifier(unittest.TestCase):
    """is_oom scans the FULL traceback, whose file paths can embed a marker as a substring: a
    checkout named "mbirjax_headroom" put "oom" in every traceback and classified every error as
    an OOM.  These tests are path-independent (the sharding-hooks non-OOM test only catches the
    bug when the checkout path itself contains "oom")."""

    def test_real_oom_signatures_classify_true(self):
        for message in ('XLA: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 1073741824 bytes',
                        'OOM when allocating tensor with shape[1024,1024,1024]',
                        'Failed to create cuFFT batched plan with scratch allocator',
                        'Failed to allocate work area',
                        'MemoryError: std::bad_alloc'):
            self.assertTrue(mj._utils.is_oom(message), message)

    def test_marker_inside_path_word_does_not_classify(self):
        fake_traceback = ('Traceback (most recent call last):\n'
                          '  File "/home/user/mbirjax_headroom/mbirjax/tomography_model.py", line 1\n'
                          '  File "/home/user/tests/test_non_oom_error.py", line 2\n'
                          'jax.errors.JaxRuntimeError: XLA computation has a shape mismatch')
        self.assertFalse(mj._utils.is_oom(fake_traceback))


class TestGetAllParamsBuildModel(unittest.TestCase):
    """get_all_params is the single source of truth for reading a model's params back out; build_model
    reconstructs a model from (required, optional, regularization).  The round-trip must reproduce the
    model, and (required, optional) alone must be a self-contained model description (build_model
    resolves the class from required['geometry_type'])."""

    @staticmethod
    def _angles(n=20):
        return np.linspace(0, np.pi, n, endpoint=False)

    def _assert_roundtrip(self, model):
        req, opt, reg = model.get_all_params()
        self.assertEqual(req['geometry_type'], str(type(model)))
        self.assertNotIn('geometry_type', opt)
        rebuilt = mj.build_model(req, opt, reg)
        self.assertIs(type(rebuilt), type(model))
        self.assertEqual(tuple(rebuilt.get_params('recon_shape')), tuple(model.get_params('recon_shape')))
        self.assertAlmostEqual(float(rebuilt.get_params('delta_voxel')), float(model.get_params('delta_voxel')))

    def test_cone_roundtrip_unpacks_view_components(self):
        m = mj.ConeBeamModel((20, 32, 32), self._angles(), source_detector_dist=128, source_iso_dist=64)
        req, opt, reg = m.get_all_params()
        self.assertIn('angles', req)              # view components unpacked into constructor args
        self.assertIn('helical_z_shifts', req)
        self.assertNotIn('view_params_array', opt)   # the packed array is rebuilt from the components
        self._assert_roundtrip(m)

    def test_parallel_and_translation_roundtrip(self):
        self._assert_roundtrip(mj.ParallelBeamModel((20, 32, 32), self._angles()))
        tv = np.random.RandomState(0).randn(20, 3) * 5
        self._assert_roundtrip(mj.TranslationModel((20, 24, 24), tv, source_detector_dist=500, source_iso_dist=250))

    def test_regularization_bucket_membership(self):
        _, opt, reg = mj.ConeBeamModel((20, 32, 32), self._angles(),
                                       source_detector_dist=128, source_iso_dist=64).get_all_params()
        self.assertEqual(set(reg),
                         {'sigma_y', 'sigma_x', 'sigma_prox', 'snr_db', 'sharpness', 'auto_regularize_flag'})
        for k in ('p', 'q', 'T', 'qggmrf_nbr_wts'):   # prior SHAPE params are structural -> optional
            self.assertIn(k, opt)
            self.assertNotIn(k, reg)

    def test_build_model_from_required_only(self):
        req, _, _ = mj.ConeBeamModel((20, 32, 32), self._angles(),
                                     source_detector_dist=128, source_iso_dist=64).get_all_params()
        self.assertIs(type(mj.build_model(req)), mj.ConeBeamModel)   # optional/regularization default None

    def test_pinned_recon_shape_survives_auto(self):
        req, opt, _ = mj.ConeBeamModel((20, 32, 32), self._angles(),
                                       source_detector_dist=128, source_iso_dist=64).get_all_params()
        opt['recon_shape'] = (40, 40, 40)
        self.assertEqual(tuple(mj.build_model(req, opt).get_params('recon_shape')), (40, 40, 40))

    def test_build_model_unknown_geometry_type_raises(self):
        with self.assertRaises(ValueError):
            mj.build_model({'geometry_type': 'not-a-class', 'sinogram_shape': (1, 1, 1), 'angles': [0.0]})


class TestTimeFrames(unittest.TestCase):
    """Splitting a rotation into overlapping time frames."""

    NUM_VIEWS = 24          # 24 views over 360 degrees -> 15 degrees per view
    DET_ROWS = 8
    DET_COLS = 10

    def setUp(self):
        angles = np.radians(360.0 / self.NUM_VIEWS) * np.arange(self.NUM_VIEWS)
        self.model = mj.ConeBeamModel(
            sinogram_shape=(self.NUM_VIEWS, self.DET_ROWS, self.DET_COLS), angles=angles,
            source_detector_dist=100.0, source_iso_dist=50.0)
        self.sinogram = np.arange(
            self.NUM_VIEWS * self.DET_ROWS * self.DET_COLS, dtype=np.float32).reshape(
            self.NUM_VIEWS, self.DET_ROWS, self.DET_COLS)

    def test_frame_count_and_views(self):
        # 120 degree frames advancing 60 degrees: views 0-7, 4-11, 8-15, 12-19, 16-23.
        print('Testing time frame count and view membership')
        sino_frames, model_frames = mj.construct_time_frames(
            self.sinogram, self.model, frames_per_rotation=6, frame_overlap_factor=2.0)
        self.assertEqual(len(sino_frames), 5)
        self.assertEqual(len(model_frames), 5)
        for sino_frame in sino_frames:
            self.assertEqual(sino_frame.shape, (8, self.DET_ROWS, self.DET_COLS))
        self.assertTrue(np.array_equal(sino_frames[1], self.sinogram[4:12]))

    def test_frame_angles_match_views(self):
        print('Testing that each frame model carries the angles of its own views')
        _, model_frames = mj.construct_time_frames(
            self.sinogram, self.model, frames_per_rotation=6, frame_overlap_factor=2.0)
        full_angles = self.model.get_all_params()[0]['angles']
        frame_angles = model_frames[1].get_all_params()[0]['angles']
        self.assertTrue(np.allclose(frame_angles, full_angles[4:12]))

    def test_view_slices_agree_with_sino_frames(self):
        """The wrapper must slice exactly the views the primitive reports."""
        print('Testing agreement between construct_time_frame_models and construct_time_frames')
        model_frames, view_slices = mj.construct_time_frame_models(
            self.model, frames_per_rotation=6, frame_overlap_factor=2.0)
        sino_frames, wrapper_models = mj.construct_time_frames(
            self.sinogram, self.model, frames_per_rotation=6, frame_overlap_factor=2.0)
        self.assertEqual(len(view_slices), len(sino_frames))
        self.assertEqual(len(model_frames), len(wrapper_models))
        for view_slice, sino_frame in zip(view_slices, sino_frames):
            self.assertTrue(np.array_equal(self.sinogram[view_slice], sino_frame))

    def test_sino_frames_are_views_not_copies(self):
        print('Testing that sinogram frames share memory with the full sinogram')
        sino_frames, _ = mj.construct_time_frames(self.sinogram, self.model)
        self.assertTrue(np.shares_memory(sino_frames[0], self.sinogram))

    def test_frames_track_view_subsampling(self):
        """Half the views at twice the angle step must give the same frames, half the size."""
        print('Testing that frame size is derived from the model angle spacing')
        subsampled = mj.copy_ct_model(
            self.model, new_angles=self.model.get_all_params()[0]['angles'][::2])
        model_frames, view_slices = mj.construct_time_frame_models(
            subsampled, frames_per_rotation=6, frame_overlap_factor=2.0)
        self.assertEqual(len(model_frames), 5)
        self.assertEqual([s.stop - s.start for s in view_slices], [4] * 5)

    def test_span_too_large_raises(self):
        print('Testing that a frame longer than the scan raises')
        with self.assertRaises(ValueError):
            mj.construct_time_frames(self.sinogram, self.model, frames_per_rotation=6,
                                     frame_overlap_factor=12.0)

    def test_stride_smaller_than_one_view_raises(self):
        print('Testing that a sub-view stride raises')
        with self.assertRaises(ValueError):
            mj.construct_time_frame_models(self.model, frames_per_rotation=self.NUM_VIEWS * 4)


class TestSaveVolumeAsGif(unittest.TestCase):
    """Titles and frame rate in save_volume_as_gif."""

    def setUp(self):
        self.volume = np.linspace(0, 1, 3 * 5 * 7, dtype=np.float32).reshape(3, 5, 7)

    def test_gif_written_with_and_without_titles(self):
        print('Testing save_volume_as_gif with per-frame titles')
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            plain_path = os.path.join(tmp_dir, 'plain.gif')
            mj.save_volume_as_gif(self.volume, plain_path)
            self.assertTrue(os.path.isfile(plain_path))

            titled_path = os.path.join(tmp_dir, 'titled.gif')
            mj.save_volume_as_gif(self.volume, titled_path, vmax=0.5, titles='t={}', fps=10)
            self.assertTrue(os.path.isfile(titled_path))

            listed_path = os.path.join(tmp_dir, 'listed.gif')
            mj.save_volume_as_gif(self.volume, listed_path, titles=['a', 'b', 'c'])
            self.assertTrue(os.path.isfile(listed_path))

    def test_wrong_number_of_titles_raises(self):
        print('Testing that a title list of the wrong length raises')
        with self.assertRaises(ValueError):
            mj.save_volume_as_gif(self.volume, 'unused.gif', titles=['only', 'two'])


if __name__ == '__main__':
    unittest.main()
