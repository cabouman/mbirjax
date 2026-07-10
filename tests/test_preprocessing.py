import unittest
import numpy as np
import jax.numpy as jnp
import mbirjax as mj

import mbirjax.preprocess.utilities as preprocess


class TestNSIPreprocessing(unittest.TestCase):
    """
    Unit tests for NSI dataset preprocessing functions.
    Tests:
    Sinogram computation from scans using JAX.
    """
    @staticmethod
    def generate_dark_scan(shape, mean=0.0, stddev=1.1, clip_negative=True, seed=None):
        """
        Generate a random dark scan with Gaussian noise.

        Parameters:
        - shape (tuple): Shape of the dark scan (e.g., (height, width)).
        - mean (float or ndarray or jax.array): Mean of the Gaussian noise (default: 0.0).
        - stddev (float or ndarray or jax.array): Standard deviation of the Gaussian noise (default: 0.01).
        - clip_negative (bool): Whether to clip negative values to zero (default: True).

        Returns:
        - dark_scan (numpy.ndarray): Simulated dark scan.
        """
        if seed is not None:
            np.random.seed(seed)
        # Generate Gaussian noise
        dark_scan = np.random.normal(mean, stddev, size=shape)

        # We should get no negative values.  If we do, then we take the absolute value instead of
        # clipping so that the values are still random.
        if clip_negative:
            dark_scan = np.abs(dark_scan)

        return dark_scan

    def setUp(self):
        """Set up parameters and initialize models before each test."""
        # Sinogram parameters
        self.num_views = 40
        self.num_det_rows = 64
        self.num_det_channels = 128
        self.sinogram_shape = (self.num_views, self.num_det_rows, self.num_det_channels)

        # Geometry parameters - FDK
        self.source_detector_dist = 4 * self.num_det_channels
        self.source_iso_dist = self.source_detector_dist / 2
        start_angle = -jnp.pi  # For testing purposes, we use a full 360 degrees
        end_angle = jnp.pi
        self.angles = jnp.linspace(start_angle, end_angle, self.num_views, endpoint=False)
        self.maximum_intensity = 4.0

        # Initialize CT model
        self.cone_model = mj.ConeBeamModel(self.sinogram_shape,
                                                self.angles,
                                                source_detector_dist=self.source_detector_dist,
                                                source_iso_dist=self.source_iso_dist)

        # Generate 3D Shepp-Logan phantom and sinogram
        # Define the crop region to test cropping and defective pixels.  We'll mask the phantom so that there is
        # sufficient border to do background estimation.  We'll include (0, 0) and the opposite corners as defective
        # pixels.  The crop region will exclude (0, 0) but include the opposite pixel.  In the crop region,
        # we have (row_frac0, row_frac1), (col_frac0, col_frac1), so we need the form (row_frac0, 1), (0, 1) to meet these conditions.
        self.crop_pixels_sides = 1
        self.crop_pixels_top = 5
        self.crop_pixels_bottom = 0
        self.edge_width = 4
        row0 = round(self.crop_pixels_top)
        border_width = self.edge_width

        phantom_shape = self.cone_model.get_params('recon_shape')
        self.phantom = mj.generate_3d_shepp_logan_low_dynamic_range(phantom_shape)
        sino_gt = self.cone_model.forward_project(self.phantom)

        # Mask the borders as needed
        sino_gt = np.array(sino_gt)
        sino_gt[:, :border_width + row0, :] = 0.0  # Top
        sino_gt[:, -border_width:, :] = 0.0  # Bottom
        sino_gt[:, :, :border_width] = 0.0  # Left
        sino_gt[:, :, -border_width:] = 0.0  # Right
        self.sino_gt = jnp.array(sino_gt)

        # Normalize the sinogram
        self.sino_gt = self.sino_gt / jnp.percentile(self.sino_gt, 98)
        self.ideal_obj_scan = self.maximum_intensity * jnp.exp(-self.sino_gt)
        # Set the mean and standard deviation for the dark scan
        # These values are estimated empirically from NSI data.
        dark_mean = 0.02
        dark_stddev = 0.001

        # Generate a single dark scan
        self.dark_scan = self.generate_dark_scan((1,) + self.sinogram_shape[1:],
                                                 mean=dark_mean, stddev=dark_stddev, seed=44)

        # Create blank scan using the maximum intensity plus a realization of the dark scan.
        # Then repeat for the object scan, using the noise-free scan plus a new dark scan.
        self.blank_scan = self.maximum_intensity + self.generate_dark_scan((1,) + self.sinogram_shape[1:],
                                                                           mean=dark_mean, stddev=dark_stddev, seed=42)
        self.obj_scan = self.ideal_obj_scan + self.generate_dark_scan(self.sinogram_shape, mean=dark_mean,
                                                                      stddev=dark_stddev, seed=43)

        # Randomly generate defective pixel coordinates.
        np.random.seed(25)
        num_defective_pixels = 15
        defective_pixels = [
            (np.random.randint(0, self.num_det_rows - 1), np.random.randint(0, self.num_det_channels - 1)) for j in
            range(num_defective_pixels)]
        # Include (0, 0) to test the ability to crop when there are defective pixels.
        defective_pixels = [(0, 0), (self.num_det_rows-1, self.num_det_channels-1)] + defective_pixels
        self.defective_pixel_array = np.array(defective_pixels)

        # Randomly set other pixels to nan to test the function's ability to recover
        nan_pixels = [(np.random.randint(0, self.num_views - 1), np.random.randint(0, self.num_det_rows - 1),
                       np.random.randint(0, self.num_det_channels - 1)) for j in range(num_defective_pixels)]

        obj_scan = np.array(self.obj_scan)
        for index in nan_pixels:
            obj_scan[index[0], index[1], index[2]] = np.nan
        self.obj_scan = jnp.array(obj_scan)

        # Set the tolerances for the test.  interpolate_defective_pixels now fills invalid pixels with
        # the neighborhood MEAN (a jittable, fixed-iteration dense fill) rather than the median.  The mean
        # is a slightly looser estimate at the few dead pixels that fall on sinogram gradients (max abs
        # diff ~0.21 vs ~0.14 for the median, at <1% of pixels), so atol/nrmse are relaxed accordingly;
        # the 99th-percentile gate (the bulk of the sinogram) is unchanged.
        self.preprocessing_tolerance = {'atol': 0.25, 'nrmse_tol': 0.0025, 'pct99_tol': 0.0018}

    def test_preprocessing(self):
        """Test if background offset correction is consistent between JAX and GDT implementations."""
        obj_scan, blank_scan, dark_scan, defective_pixel_array = preprocess.crop_view_data(self.obj_scan, self.blank_scan, self.dark_scan,
                                                                                           defective_pixel_array=self.defective_pixel_array,
                                                                                           crop_pixels_sides=self.crop_pixels_sides,
                                                                                           crop_pixels_top=self.crop_pixels_top,
                                                                                           crop_pixels_bottom=self.crop_pixels_bottom)
        sino_computed = preprocess.compute_sino_transmission(obj_scan, blank_scan, dark_scan,
                                                             defective_pixel_array=defective_pixel_array)

        # Compute background offsets
        sino_computed = preprocess.correct_background_offset(sino_computed, edge_width=self.edge_width)

        sino_gt_cropped, _, _, _ = preprocess.crop_view_data(self.sino_gt, self.blank_scan, self.dark_scan,
                                                             crop_pixels_sides=self.crop_pixels_sides,
                                                             crop_pixels_top=self.crop_pixels_top,
                                                             crop_pixels_bottom=self.crop_pixels_bottom)
        abs_sino_diff = np.abs(sino_computed - sino_gt_cropped)
        max_diff = np.max(np.abs(abs_sino_diff))
        nrmse = np.linalg.norm(abs_sino_diff) / np.linalg.norm(sino_gt_cropped)
        pct99 = np.percentile(abs_sino_diff, 99)

        print('Difference between gt sino and estimated sino: max abs = {:.4f}, nrmse = {:.4f}'.format(max_diff, nrmse))
        print('99% of absolute sinogram differences are less than {:.4f}'.format(pct99))

        tolerance = self.preprocessing_tolerance['atol']
        tolerance_mean = self.preprocessing_tolerance['nrmse_tol']
        tolerance_pct99 = self.preprocessing_tolerance['pct99_tol']

        # Check if differences are within tolerance
        self.assertTrue(
            max_diff < tolerance and nrmse < tolerance_mean and pct99 < tolerance_pct99,
            f"Sinograms differ more than the tolerance. "
            f"Max diff={max_diff:.4f} (tolerance: {tolerance}), "
            f"NRMSE={nrmse:.4f} (tolerance: {tolerance_mean}), 99th percentile={pct99:.4f} (tolerance: {tolerance_pct99})"
        )
        self.assertFalse(np.isnan(sino_computed).any(), "Error: sino_computed contains NaN values!")


class TestSavePreprocessing(unittest.TestCase):
    """mj.preprocess.save_preprocessing / load_preprocessing round-trip (the two-stage preprocess->recon
    workflow): the sinogram, geometry-parameter dicts, and optional custom weights survive a disk
    round-trip, with sinogram_shape restored to a tuple and numpy-scalar params coerced cleanly."""

    @staticmethod
    def _example():
        sino = (np.random.RandomState(0).rand(8, 6, 5) * 3).astype(np.float32)
        cone_beam_params = {'sinogram_shape': (8, 6, 5),
                            'angles': np.linspace(0, np.pi, 8, dtype=np.float64),
                            'source_detector_dist': np.float64(8192), 'source_iso_dist': 4096}
        optional_params = {'delta_det_channel': 1.0, 'delta_det_row': np.float32(1.0),
                           'delta_voxel': 0.5, 'det_channel_offset': np.float64(0.0),
                           'det_row_offset': 0.0, 'recon_slice_offset': 0.0,
                           'alu_unit': 'mm', 'alu_value': 1.0}
        return sino, cone_beam_params, optional_params

    def test_roundtrip_no_weights(self):
        import os, tempfile
        sino, cbp, opt = self._example()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'pre.h5')
            mj.preprocess.save_preprocessing(path, sino, cbp, opt)
            s2, cbp2, opt2, w2 = mj.preprocess.load_preprocessing(path)
        np.testing.assert_array_equal(s2, sino)
        self.assertEqual(s2.dtype, np.float32)
        self.assertIsNone(w2)                                            # no weights saved -> None
        self.assertEqual(cbp2['sinogram_shape'], (8, 6, 5))
        self.assertIsInstance(cbp2['sinogram_shape'], tuple)             # JSON list restored to tuple
        np.testing.assert_allclose(cbp2['angles'], cbp['angles'])        # array param round-trips
        self.assertEqual(float(cbp2['source_detector_dist']), 8192.0)    # numpy scalar coerced
        self.assertEqual(opt2['alu_unit'], 'mm')                         # string preserved
        self.assertEqual(set(cbp2), set(cbp))
        self.assertEqual(set(opt2), set(opt))

    def test_roundtrip_with_custom_weights(self):
        import os, tempfile
        sino, cbp, opt = self._example()
        weights = (np.random.RandomState(1).rand(8, 6, 5)).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'pre.h5')
            mj.preprocess.save_preprocessing(path, sino, cbp, opt, weights=weights)
            s2, _, _, w2 = mj.preprocess.load_preprocessing(path)
        np.testing.assert_array_equal(s2, sino)
        np.testing.assert_array_equal(w2, weights)                       # custom weights survive
        self.assertEqual(w2.dtype, np.float32)


class TestZeissSinoShifts(unittest.TestCase):
    """correct_sino_shifts (zeiss module; the unused zeiss_tct copy was removed): padding must
    cover the largest ABSOLUTE
    per-view shift (the translation applies absolute offsets, so range-based padding under-pads
    whenever the shifts share a common offset), and the downsample scaling must not mutate the
    caller's zeiss_params in place."""

    @staticmethod
    def _params(x_shifts, y_shifts):
        return {'x_shifts': np.asarray(x_shifts, dtype=np.float64),
                'y_shifts': np.asarray(y_shifts, dtype=np.float64)}

    def test_common_offset_preserves_boundary(self):
        # A constant sinogram shifted by a COMMON offset must come back (nearly) constant:
        # with edge-mode padding sized to the absolute shift, no zeros can enter the frame.
        # Under range-based padding the pad is 0 here and a zero band scrolls in at the edge.
        from mbirjax.preprocess.zeiss import correct_sino_shifts
        num_views, n = 3, 16
        sino = np.ones((num_views, n, n), dtype=np.float32)
        params = self._params([5.0, 5.0, 5.0], [3.0, 3.0, 3.0])
        out = np.asarray(correct_sino_shifts(sino, params, downsample_factor=(1, 1),
                                             subsample_view_factor=1))
        self.assertEqual(out.shape, sino.shape)
        np.testing.assert_allclose(out, 1.0, atol=1e-5,
                                   err_msg='boundary corrupted: padding did not cover the '
                                           'absolute shift')

    def test_impulse_moves_by_per_view_shift(self):
        # Integer per-view shifts move an impulse exactly (linear interpolation is exact at
        # integers): positive x -> right (channels), positive y -> down (rows).
        from mbirjax.preprocess.zeiss import correct_sino_shifts
        num_views, n, c = 3, 17, 19
        r0, c0 = 8, 9
        sino = np.zeros((num_views, n, c), dtype=np.float32)
        sino[:, r0, c0] = 1.0
        x_shifts, y_shifts = [2.0, -3.0, 0.0], [-1.0, 4.0, 0.0]
        out = np.asarray(correct_sino_shifts(sino, self._params(x_shifts, y_shifts),
                                             downsample_factor=(1, 1), subsample_view_factor=1))
        for v, (dx, dy) in enumerate(zip(x_shifts, y_shifts)):
            peak = np.unravel_index(np.argmax(out[v]), out[v].shape)
            self.assertEqual(peak, (r0 + int(dy), c0 + int(dx)), msg=f'view {v}')
            self.assertAlmostEqual(float(out[v][peak]), 1.0, places=5)

    def test_zeiss_copy_does_not_mutate_params_and_is_idempotent(self):
        from mbirjax.preprocess.zeiss import correct_sino_shifts
        num_views, n = 4, 12
        rng = np.random.default_rng(0)
        sino = rng.random((num_views, n, n)).astype(np.float32)
        params = self._params([4.0, -2.0, 1.0, 3.0], [2.0, 0.0, -3.0, 1.0])
        saved = {k: v.copy() for k, v in params.items()}
        out1 = np.asarray(correct_sino_shifts(sino, params, downsample_factor=(2, 2),
                                              subsample_view_factor=1))
        np.testing.assert_array_equal(params['x_shifts'], saved['x_shifts'])   # no in-place /=
        np.testing.assert_array_equal(params['y_shifts'], saved['y_shifts'])
        out2 = np.asarray(correct_sino_shifts(sino, params, downsample_factor=(2, 2),
                                              subsample_view_factor=1))
        np.testing.assert_array_equal(out1, out2)                              # idempotent


if __name__ == '__main__':
    unittest.main()

