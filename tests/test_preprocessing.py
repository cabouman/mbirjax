import unittest
from unittest import mock
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

        # The phantom is only a source of realistic sinogram values for the preprocessing
        # roundtrip, so build it on the central-ray base slab and embed it (zero-filled) in
        # the auto recon shape: auto_set_recon_geometry extends the slab with
        # visibility-extension end slices, and letting the phantom stretch into that
        # half-sampled wedge would move the sinogram's gradients relative to the fixed
        # defective-pixel seeds, silently changing what the tolerances measure.  Embedding
        # keeps the ground-truth sinogram identical to its pre-extension calibration.
        recon_shape = self.cone_model.get_params('recon_shape')
        delta_det_row, delta_voxel = self.cone_model.get_params(['delta_det_row', 'delta_voxel'])
        magnification = self.cone_model.get_magnification()
        base_slices = int(np.ceil(self.num_det_rows * (delta_det_row / magnification) / delta_voxel))
        phantom_core = mj.generate_3d_shepp_logan_low_dynamic_range(
            (recon_shape[0], recon_shape[1], base_slices))
        slab_start = (recon_shape[2] - base_slices) // 2
        self.phantom = jnp.zeros(recon_shape).at[:, :, slab_start:slab_start + base_slices].set(phantom_core)
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
    """mj.preprocess.save_cone_preprocessing / load_cone_preprocessing round-trip (the two-stage preprocess->recon
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
            mj.preprocess.save_cone_preprocessing(path, sino, cbp, opt)
            s2, cbp2, opt2, w2 = mj.preprocess.load_cone_preprocessing(path)
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
            mj.preprocess.save_cone_preprocessing(path, sino, cbp, opt, weights=weights)
            s2, _, _, w2 = mj.preprocess.load_cone_preprocessing(path)
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


class TestDetectorCrop(unittest.TestCase):
    """apply_detector_crop is the single, detector-plane-only source of crop-to-geometry
    bookkeeping (sinogram_shape + det offsets, never recon_slice_offset); _auto_crop_sino composes
    detect_blank_margins + array slicing + apply_detector_crop.  These check the offset formula, the
    geometry-general handling across cone and parallel models (and a dict missing det_row_offset),
    array<->geometry consistency (shape AND object position), and that the crop leaves
    recon_slice_offset to auto_set_recon_geometry (pinned with a sentinel)."""

    def test_apply_detector_crop_shape_and_offset_formula(self):
        required_in = {'sinogram_shape': (10, 64, 128)}
        optional_in = {'det_row_offset': 1.0, 'det_channel_offset': 2.0,
                       'delta_det_row': 0.5, 'delta_det_channel': 0.25}
        required, optional = preprocess.apply_detector_crop(required_in, optional_in, crop_top=4,
                                                            crop_bottom=10, crop_left=6, crop_right=2)
        self.assertEqual(required['sinogram_shape'], (10, 64 - 14, 128 - 8))
        # det_row_offset     += (crop_bottom - crop_top) / 2 * delta_det_row     = (10-4)/2*0.5 =  1.5
        # det_channel_offset += (crop_right  - crop_left) / 2 * delta_det_channel = (2-6)/2*0.25 = -0.5
        self.assertAlmostEqual(optional['det_row_offset'], 1.0 + 1.5)
        self.assertAlmostEqual(optional['det_channel_offset'], 2.0 - 0.5)
        # pure: the inputs are not mutated (result comes only through the return value)
        self.assertEqual(required_in['sinogram_shape'], (10, 64, 128))
        self.assertAlmostEqual(optional_in['det_row_offset'], 1.0)

    def test_apply_detector_crop_symmetric_leaves_offsets(self):
        required = {'sinogram_shape': (10, 64, 128)}
        optional = {'det_row_offset': 1.0, 'det_channel_offset': 2.0,
                    'delta_det_row': 0.5, 'delta_det_channel': 0.25}
        required, optional = preprocess.apply_detector_crop(required, optional, 8, 8, 3, 3)
        self.assertEqual(required['sinogram_shape'], (10, 48, 122))
        self.assertAlmostEqual(optional['det_row_offset'], 1.0)      # symmetric crop -> no shift
        self.assertAlmostEqual(optional['det_channel_offset'], 2.0)

    def test_apply_detector_crop_missing_offset_key_is_a_noop_guard(self):
        # Defensive guard: if an optional_params dict lacks det_row_offset, apply_detector_crop must
        # neither raise nor invent one -- it updates the shape and the offset that IS present.  (Real
        # models all carry det_row_offset=0.0 in their params; that path is covered by the parallel
        # and cone round-trip tests below.  The parallel projector ignores det_row_offset regardless.)
        required = {'sinogram_shape': (10, 64, 128)}
        optional = {'det_channel_offset': 0.0, 'delta_det_channel': 1.0}   # no det_row_offset key
        required, optional = preprocess.apply_detector_crop(required, optional, 5, 1, 4, 0)
        self.assertEqual(required['sinogram_shape'], (10, 58, 124))
        self.assertNotIn('det_row_offset', optional)                       # not created
        self.assertAlmostEqual(optional['det_channel_offset'], (0 - 4) / 2 * 1.0)

    def test_apply_detector_crop_rejects_overcrop(self):
        # Geometry-path guard: a crop >= the detector dimension, or a negative crop, must raise here
        # (independent of crop_view_data's array-path assert) rather than yield a negative shape.
        required = {'sinogram_shape': (10, 64, 128)}
        with self.assertRaises(AssertionError):
            preprocess.apply_detector_crop(dict(required), {}, 40, 40, 0, 0)     # 40 + 40 >= 64 rows
        with self.assertRaises(AssertionError):
            preprocess.apply_detector_crop(dict(required), {}, 0, 0, 70, 70)     # 70 + 70 >= 128 channels
        with self.assertRaises(AssertionError):
            preprocess.apply_detector_crop(dict(required), {}, 0, 0, 0, -1)      # negative crop

    def test_apply_config_crop_matches_formula(self):
        # apply_config_crop is the scalar adapter around apply_detector_crop used by the convert_* readers.
        nr, nc, dro, dco = preprocess.apply_config_crop(
            64, 128, 1.0, 2.0, 0.5, 0.25, crop_pixels_top=4, crop_pixels_bottom=10, crop_pixels_sides=6)
        self.assertEqual((nr, nc), (64 - 14, 128 - 12))
        self.assertAlmostEqual(dro, 1.0 + (10 - 4) / 2 * 0.5)      # asymmetric top/bottom -> row shift
        self.assertAlmostEqual(dco, 2.0)                          # sides symmetric -> no channel shift

    def test_to_alu(self):
        self.assertAlmostEqual(preprocess.to_alu(1.0, 'mm', 'um'), 1000.0)
        self.assertAlmostEqual(preprocess.to_alu(2.0, 'cm', 'mm'), 20.0)
        self.assertAlmostEqual(preprocess.to_alu(5.0, 'mm', 'mm'), 5.0)   # identity when units match

    def test_detect_blank_margins_exact_and_deterministic(self):
        sino = np.zeros((8, 80, 100), dtype=np.float32)
        sino[:, 25:60, 30:75] = 5.0                                        # object block, off-center
        margins = preprocess.detect_blank_margins(sino, safety_buffer=5)
        # Deterministic and exactly computable: object rows [25:60] of 80, cols [30:75] of 100,
        # buffer 5 -> (top, bottom, left, right) = (25-5, (80-60)-5, 30-5, (100-75)-5).
        self.assertEqual(tuple(int(x) for x in margins), (20, 15, 25, 20))
        self.assertEqual(margins, preprocess.detect_blank_margins(sino, safety_buffer=5))   # repeatable
        ct, cb, cl, cr = margins
        self.assertLessEqual(ct, 25)                                       # object fully preserved
        self.assertGreaterEqual(80 - cb, 60)
        self.assertLessEqual(cl, 30)
        self.assertGreaterEqual(100 - cr, 75)

    def test_parallel_auto_crop_roundtrip(self):
        # A real non-cone geometry through the full path: ParallelBeamModel has no recon_slice_offset,
        # so this exercises apply_detector_crop's shape + det_channel_offset bookkeeping on a real
        # model and confirms it survives build_model.
        angles = np.linspace(0, np.pi, 12, endpoint=False)
        model = mj.ParallelBeamModel((12, 80, 100), angles)
        model.set_params(delta_det_channel=0.5)
        required, optional, regularization = model.get_all_params()
        optional.pop('recon_shape', None)
        channel_offset_before = optional['det_channel_offset']

        sino = np.zeros((12, 80, 100), dtype=np.float32)
        sino[:, 20:60, 28:66] = 5.0                     # asymmetric in channels -> channel offset shifts
        ct, cb, cl, cr = preprocess.detect_blank_margins(sino, safety_buffer=5)
        sino, required, optional = preprocess._auto_crop_sino(sino, required, optional, safety_buffer=5)

        self.assertEqual(tuple(required['sinogram_shape']), sino.shape)
        self.assertAlmostEqual(optional['det_channel_offset'], channel_offset_before + (cr - cl) / 2 * 0.5)
        rebuilt = mj.build_model(required, optional, regularization)
        self.assertEqual(tuple(rebuilt.get_params('sinogram_shape')), sino.shape)        # survives build

    def test_auto_crop_sino_consistent_and_survives_build_model(self):
        # A cone model's params round-trip through _auto_crop_sino + build_model: the cropped
        # sinogram shape, the object's POSITION, and the shifted det_row_offset survive; the crop
        # never touches recon_slice_offset; and auto_set_recon_geometry (via build_model) OVERWRITES
        # recon_slice_offset (proven with a sentinel), matching a reference model run through the
        # same auto pass.  (The cone auto geometry follows the detector: the asymmetric crop shifts
        # det_row_offset, so the derived offset moves off iso rather than staying 0 -- the exact
        # value comes from the per-end visibility extension, so compare to the reference model
        # instead of pinning a constant.)
        angles = np.linspace(0, np.pi, 12, endpoint=False)
        model = mj.ConeBeamModel((12, 80, 100), angles, source_detector_dist=200, source_iso_dist=100)
        model.set_params(delta_det_row=0.5, delta_det_channel=0.5)
        required, optional, regularization = model.get_all_params()
        optional.pop('recon_shape', None)          # reader flow: let auto size the recon from the crop
        row_offset_before = optional['det_row_offset']
        recon_slice_offset_before = optional['recon_slice_offset']

        obj_row0, obj_col0 = 25, 30
        sino = np.zeros((12, 80, 100), dtype=np.float32)
        sino[:, obj_row0:60, obj_col0:70] = 5.0
        ct, cb, cl, cr = preprocess.detect_blank_margins(sino, safety_buffer=5)

        sino, required, optional = preprocess._auto_crop_sino(sino, required, optional, safety_buffer=5)
        self.assertEqual(tuple(required['sinogram_shape']), sino.shape)                  # array <-> geometry
        self.assertAlmostEqual(optional['det_row_offset'], row_offset_before + (cb - ct) / 2 * 0.5)
        self.assertEqual(optional['recon_slice_offset'], recon_slice_offset_before)      # crop leaves it alone
        # Slice direction is tied to the geometry: the object's top row/left col land at
        # (orig - crop), so a top/bottom (or left/right) swap in the slice -- same shape -- moves the
        # object and fails here.
        nz_rows = np.asarray(np.any(sino != 0, axis=(0, 2)))
        nz_cols = np.asarray(np.any(sino != 0, axis=(0, 1)))
        self.assertEqual(int(np.argmax(nz_rows)), obj_row0 - ct)
        self.assertEqual(int(np.argmax(nz_cols)), obj_col0 - cl)

        # Sentinel: a value the crop never wrote must be overwritten by auto_set_recon_geometry,
        # proving build_model re-derives recon_slice_offset (a regression that skipped auto keeps 999.0).
        optional['recon_slice_offset'] = 999.0
        rebuilt = mj.build_model(required, optional, regularization)
        self.assertEqual(tuple(rebuilt.get_params('sinogram_shape')), sino.shape)        # survives build
        # The sentinel must be gone, replaced by the value the auto pass derives from the CROPPED
        # geometry.  The reference runs construct -> set_params -> auto_set_recon_geometry by hand,
        # so this also catches build_model applying the steps in the wrong order (auto before
        # set_params would derive from the uncropped det_row_offset).
        reference = mj.ConeBeamModel(**{k: v for k, v in required.items() if k != 'geometry_type'})
        reference.set_params(**{k: v for k, v in optional.items() if k != 'recon_slice_offset'})
        reference.auto_set_recon_geometry()
        derived = float(rebuilt.get_params('recon_slice_offset'))
        self.assertNotAlmostEqual(derived, 999.0)                                        # sentinel overwritten
        self.assertAlmostEqual(derived, float(reference.get_params('recon_slice_offset')))
        self.assertEqual(tuple(rebuilt.get_params('recon_shape')),
                         tuple(reference.get_params('recon_shape')))


class TestConfigCropUnification(unittest.TestCase):
    """Phase 2: the configuration crop in convert_nsi / convert_zeiss now routes through
    apply_detector_crop.  A symmetric crop is byte-identical (offsets unchanged); an asymmetric
    top/bottom crop shifts det_row_offset by (crop_bottom - crop_top)/2 * RAW delta_det_row -- the
    previously-uncompensated asymmetric-crop bug.  Side crops are symmetric by construction, so
    det_channel_offset never moves.  The offset shift uses the raw pitch and is independent of
    downsampling (crop is applied at raw resolution, before the downsample rescale)."""

    @staticmethod
    def _nsi_params():
        # A clean orthonormal cone geometry (source at -y, detector at +y, rows along x, cols along -z).
        sid, sdd = 100.0, 200.0
        nrows, nchan, dr, dc = 64, 80, 0.2, 0.2
        r_n = np.array([0., 1., 0.]); r_h = np.array([1., 0., 0.]); r_a = np.array([0., 0., -1.])
        r_s = np.array([0., -sid, 0.])
        r_v = np.cross(r_n, r_h)
        r_r = np.array([0., sdd - sid, 0.]) - (nchan / 2.0) * dc * r_h - (nrows / 2.0) * dr * r_v
        return dict(r_a=r_a, r_n=r_n, r_h=r_h, r_s=r_s, r_r=r_r,
                    delta_det_channel=dc, delta_det_row=dr,
                    num_det_channels=nchan, num_det_rows=nrows,
                    angles=np.linspace(0, 2 * np.pi, 20, endpoint=False))

    @staticmethod
    def _zeiss_params():
        return dict(source_iso_dist=50.0, iso_det_dist=150.0, source_iso_dist_unit='mm', iso_det_dist_unit='mm',
                    delta_det_channel=0.15, delta_det_row=0.15, delta_det_channel_unit='mm', delta_det_row_unit='mm',
                    iso_pixel_pitch=0.05, iso_pixel_pitch_unit='mm', opt_mag=None,
                    num_det_rows=64, num_det_channels=80,
                    angles=np.linspace(0, 360, 20, endpoint=False), angle_unit='deg',
                    det_row_offset=3.0, det_channel_offset=2.0, scanner_type='versa')

    def test_nsi_symmetric_crop_is_byte_identical(self):
        from mbirjax.preprocess.nsi import convert_nsi_to_mbirjax_params as conv
        p = self._nsi_params()
        _, base = conv(p, (1, 1), 0, 0, 0)
        cb, op = conv(p, (1, 1), 3, 5, 5)                    # symmetric sides + top == bottom
        self.assertEqual(cb['sinogram_shape'], (20, 64 - 10, 80 - 6))
        for key in ('det_row_offset', 'det_channel_offset', 'recon_slice_offset',
                    'delta_det_row', 'delta_det_channel', 'delta_voxel'):
            self.assertAlmostEqual(op[key], base[key], msg=key)   # crop changes nothing but the shape

    def test_nsi_asymmetric_crop_shifts_row_offset(self):
        from mbirjax.preprocess.nsi import convert_nsi_to_mbirjax_params as conv
        p = self._nsi_params()
        _, base = conv(p, (1, 1), 0, 0, 0)
        cb, op = conv(p, (1, 1), 0, 10, 0)                   # crop_top=10, crop_bottom=0
        self.assertEqual(cb['sinogram_shape'], (20, 54, 80))
        self.assertAlmostEqual(op['det_row_offset'], base['det_row_offset'] + (0 - 10) / 2 * base['delta_det_row'])
        self.assertAlmostEqual(op['det_channel_offset'], base['det_channel_offset'])   # sides symmetric

    def test_nsi_offset_shift_uses_raw_pitch_independent_of_downsample(self):
        from mbirjax.preprocess.nsi import convert_nsi_to_mbirjax_params as conv
        p = self._nsi_params()
        _, base = conv(p, (1, 1), 0, 0, 0)
        raw_pitch = base['delta_det_row']
        expected = base['det_row_offset'] + (0 - 10) / 2 * raw_pitch   # physical shift; independent of downsample
        for ds in [(1, 1), (2, 2)]:
            cb, op = conv(p, ds, 0, 10, 0)
            self.assertAlmostEqual(op['det_row_offset'], expected, msg=str(ds))
        cb2, op2 = conv(p, (2, 2), 0, 10, 0)
        self.assertAlmostEqual(op2['delta_det_row'], raw_pitch * 2)    # downsample still scales the pitch
        self.assertEqual(cb2['sinogram_shape'], (20, 54 // 2, 80 // 2))

    def test_zeiss_symmetric_crop_is_byte_identical(self):
        from mbirjax.preprocess.zeiss import convert_zeiss_to_mbirjax_params as conv
        p = self._zeiss_params()
        _, base, _ = conv(p, (1, 1), 0, 0, 0)
        gp, op, _ = conv(p, (1, 1), 3, 5, 5)
        self.assertEqual(gp['sinogram_shape'], (20, 54, 74))
        self.assertAlmostEqual(op['det_row_offset'], base['det_row_offset'])
        self.assertAlmostEqual(op['det_channel_offset'], base['det_channel_offset'])

    def test_zeiss_asymmetric_crop_shifts_row_offset(self):
        from mbirjax.preprocess.zeiss import convert_zeiss_to_mbirjax_params as conv
        p = self._zeiss_params()
        _, base, _ = conv(p, (1, 1), 0, 0, 0)
        gp, op, _ = conv(p, (1, 1), 0, 10, 0)
        self.assertEqual(gp['sinogram_shape'], (20, 54, 80))
        self.assertAlmostEqual(op['det_row_offset'], base['det_row_offset'] + (0 - 10) / 2 * base['delta_det_row'])
        self.assertAlmostEqual(op['det_channel_offset'], base['det_channel_offset'])

    @staticmethod
    def _tct_params():
        n = 20
        return dict(source_iso_dist=50.0, iso_det_dist=150.0, source_iso_dist_unit='mm', iso_det_dist_unit='mm',
                    delta_det_channel=0.1, delta_det_row=0.1, delta_det_channel_unit='mm', delta_det_row_unit='mm',
                    iso_pixel_pitch=0.05, iso_pixel_pitch_unit='mm', opt_mag=None,
                    num_det_rows=64, num_det_channels=80,
                    object_x_positions=np.linspace(-5, 5, n), object_x_position_unit='mm',
                    object_y_positions=np.zeros(n), object_y_position_unit='mm',
                    object_z_positions=np.linspace(-2, 2, n), object_z_position_unit='mm',
                    det_row_offset=3.0, det_channel_offset=2.0)

    def test_zeiss_tct_symmetric_crop_is_byte_identical(self):
        from mbirjax.preprocess.zeiss_tct import convert_zeiss_to_mbirjax_params as conv
        p = self._tct_params()
        _, base = conv(p, 0, 0, 0)
        tp, op = conv(p, 3, 5, 5)                                # sides=3, top=bottom=5 (symmetric)
        self.assertEqual(tp['sinogram_shape'], (20, 54, 74))
        self.assertAlmostEqual(op['det_row_offset'], base['det_row_offset'])
        self.assertAlmostEqual(op['det_channel_offset'], base['det_channel_offset'])

    def test_zeiss_tct_asymmetric_crop_shifts_row_offset(self):
        from mbirjax.preprocess.zeiss_tct import convert_zeiss_to_mbirjax_params as conv
        p = self._tct_params()
        _, base = conv(p, 0, 0, 0)
        tp, op = conv(p, 0, 10, 0)                               # sides=0, top=10, bottom=0 (asymmetric)
        self.assertEqual(tp['sinogram_shape'], (20, 54, 80))
        self.assertAlmostEqual(op['det_row_offset'], base['det_row_offset'] + (0 - 10) / 2 * base['delta_det_row'])
        self.assertAlmostEqual(op['det_channel_offset'], base['det_channel_offset'])

    def test_zeiss_ultra_parallel_symmetric_crop(self):
        # The 'ultra' (parallel) branch shares the crop code that runs before the branch: a symmetric
        # crop reduces the shape and leaves the offsets unchanged.
        from mbirjax.preprocess.zeiss import convert_zeiss_to_mbirjax_params as conv
        p = self._zeiss_params(); p['scanner_type'] = 'ultra'
        _, base, _ = conv(p, (1, 1), 0, 0, 0)
        gp, op, _ = conv(p, (1, 1), 4, 6, 6)
        self.assertEqual(gp['sinogram_shape'], (20, 52, 72))
        self.assertNotIn('source_detector_dist', gp)           # parallel: no source distances
        self.assertAlmostEqual(op['det_row_offset'], base['det_row_offset'])
        self.assertAlmostEqual(op['det_channel_offset'], base['det_channel_offset'])


class TestGetSinoAndModel(unittest.TestCase):
    """Phase 3: the NSI get_sino_and_model wrapper turns the (sino, required, optional) triple from
    _compute_sino_and_params into a ready-to-reconstruct model via build_model, optionally auto-cropping
    first.  _compute_sino_and_params is mocked so the wrapper is exercised without a real NSI dataset."""

    @staticmethod
    def _fake_compute(with_margins):
        # A real ConeBeamModel's params (required already carries geometry_type via get_all_params),
        # plus a matching synthetic sinogram.
        angles = np.linspace(0, np.pi, 12, endpoint=False)
        model = mj.ConeBeamModel((12, 80, 100), angles, source_detector_dist=200, source_iso_dist=100)
        required, optional, _ = model.get_all_params()
        optional.pop('recon_shape', None)                 # let build_model's auto size the recon
        sino = np.zeros((12, 80, 100), dtype=np.float32)
        if with_margins:
            sino[:, 25:60, 30:70] = 5.0                   # blank margins -> auto_crop shrinks the sino
        else:
            sino[:] = 1.0
        return sino, required, optional

    def test_builds_ready_cone_model(self):
        from mbirjax.preprocess import nsi
        with mock.patch.object(nsi, '_compute_sino_and_params', return_value=self._fake_compute(False)):
            sino, model = nsi.get_sino_and_model('/unused', verbose=0)
        self.assertIsInstance(model, mj.ConeBeamModel)
        self.assertEqual(tuple(model.get_params('sinogram_shape')), sino.shape)        # geometry matches sino
        self.assertGreater(int(np.prod(model.get_params('recon_shape'))), 0)           # recon geometry is set

    def test_auto_crop_shrinks_and_stays_consistent(self):
        from mbirjax.preprocess import nsi
        with mock.patch.object(nsi, '_compute_sino_and_params', return_value=self._fake_compute(True)):
            sino_full, _ = nsi.get_sino_and_model('/unused', auto_crop=False, verbose=0)
        with mock.patch.object(nsi, '_compute_sino_and_params', return_value=self._fake_compute(True)):
            sino_crop, model_crop = nsi.get_sino_and_model('/unused', auto_crop=True, verbose=0)
        self.assertEqual(sino_full.shape, (12, 80, 100))                               # default: no crop
        self.assertLess(sino_crop.shape[1], 80)                                        # auto_crop removed blank rows
        self.assertEqual(tuple(model_crop.get_params('sinogram_shape')), sino_crop.shape)  # model tracks the crop

    def test_pymbir_builds_ready_cone_model(self):
        # The pymbir reader follows the same template; mock its _compute_sino_and_params.
        from mbirjax.preprocess import pymbir
        with mock.patch.object(pymbir, '_compute_sino_and_params', return_value=self._fake_compute(False)):
            sino, model = pymbir.get_sino_and_model('/unused.h5')
        self.assertIsInstance(model, mj.ConeBeamModel)
        self.assertEqual(tuple(model.get_params('sinogram_shape')), sino.shape)
        self.assertGreater(int(np.prod(model.get_params('recon_shape'))), 0)

    @staticmethod
    def _fake_zeiss(parallel):
        # A real model's params (required carries geometry_type via get_all_params, encoding the class
        # the zeiss _compute selects from scanner_type) + a matching sinogram.
        angles = np.linspace(0, np.pi, 12, endpoint=False)
        if parallel:
            model = mj.ParallelBeamModel((12, 80, 100), angles)
        else:
            model = mj.ConeBeamModel((12, 80, 100), angles, source_detector_dist=200, source_iso_dist=100)
        required, optional, _ = model.get_all_params()
        optional.pop('recon_shape', None)
        return np.ones((12, 80, 100), dtype=np.float32), required, optional

    def test_zeiss_versa_builds_cone_model(self):
        from mbirjax.preprocess import zeiss
        with mock.patch.object(zeiss, '_compute_sino_and_params', return_value=self._fake_zeiss(parallel=False)):
            sino, model = zeiss.get_sino_and_model('/unused')
        self.assertIsInstance(model, mj.ConeBeamModel)                       # 'versa' -> cone
        self.assertEqual(tuple(model.get_params('sinogram_shape')), sino.shape)

    def test_zeiss_ultra_builds_parallel_model(self):
        from mbirjax.preprocess import zeiss
        with mock.patch.object(zeiss, '_compute_sino_and_params', return_value=self._fake_zeiss(parallel=True)):
            sino, model = zeiss.get_sino_and_model('/unused')
        self.assertIsInstance(model, mj.ParallelBeamModel)                   # 'ultra' -> parallel
        self.assertEqual(tuple(model.get_params('sinogram_shape')), sino.shape)

    def test_zeiss_tct_builds_translation_model_and_returns_weights(self):
        # zeiss_tct is the special case: get_sino_and_model returns (sino, model, weights).
        from mbirjax.preprocess import zeiss_tct
        tv = np.random.RandomState(0).randn(20, 3) * 5
        model_src = mj.TranslationModel((20, 64, 80), tv, source_detector_dist=500, source_iso_dist=250)
        required, optional, _ = model_src.get_all_params()
        sino = np.ones((20, 64, 80), dtype=np.float32)
        weights = np.ones((20, 64, 80), dtype=np.float32)
        with mock.patch.object(zeiss_tct, '_compute_sino_and_params', return_value=(sino, required, optional, weights)):
            out_sino, model, out_weights = zeiss_tct.get_sino_and_model('/unused')
        self.assertIsInstance(model, mj.TranslationModel)
        self.assertEqual(tuple(model.get_params('sinogram_shape')), out_sino.shape)
        self.assertEqual(out_weights.shape, out_sino.shape)                  # data-specific weights returned


if __name__ == '__main__':
    unittest.main()

