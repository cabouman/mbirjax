"""
The mace4d module implements 4D MACE reconstruction.  The public interface is the
MACE4DModel class.
"""
from __future__ import annotations

import concurrent.futures
import csv
import datetime
import os
import threading
import time
import warnings
from importlib.metadata import version as importlib_version
from typing import Any, Literal, Union, overload

import jax
import jax.numpy as jnp
import numpy as np
from scipy.fft import dct, idct

import mbirjax as mj
from mbirjax import ParameterHandler
from mbirjax._device_setup import cpu_devices, default_devices, gpu_devices

MACE4DParamNames = mj.ParamNames | Literal['mace_prior_weight', 'rho_mann', 'prox_num_iterations',
                                           'prox_stop_threshold', 'dejitter', 'dejitter_verbose']

# Only the model and its parameter-name type are public.  This keeps
# `from .mace4d import *` from adding scipy and typing names to mbirjax.
__all__ = ['MACE4DModel', 'MACE4DParamNames']

# MBIR iterations for the per-frame initialization recon.
_INIT_MBIR_ITERATIONS = 15

# Hyperplane orientations of the three prior agents.  Each permutation moves the
# hyperplane axis first.  The recon axes are (t, x, y, z).
_PRIOR_ORIENTATIONS = [
    ("XY-t", (3, 0, 1, 2)),  # nz hyperplanes of shape (num_frames, nx, ny)
    ("YZ-t", (1, 0, 2, 3)),  # nx hyperplanes of shape (num_frames, ny, nz)
    ("XZ-t", (2, 0, 1, 3)),  # ny hyperplanes of shape (num_frames, nx, nz)
]

_TIMING_FIELDS = [
    "iteration",
    "prox_total_sec",
    "denoise_total_sec",
    "makespan_sec",
    "iteration_total_sec",
    "consensus_change_pct",
]

_TASK_FIELDS = ["iteration", "kind", "index", "device", "start_sec", "end_sec"]

# Estimated cost of denoising one hyperplane, in units of one prox_map task.  The
# value was measured on an H100 GPU on a small test problem.  Only the relative size
# matters, and it is used only for the task assignment.
_DENOISE_COST_PER_PLANE = 0.015


class MACE4DModel(ParameterHandler):
    """
    The MACE4DModel class is used to compute space-time reconstructions from a single continuous CT scan.
    The views are assumed to be collected sequentially in time with uniform temporal sampling.

    The scan is divided into overlapping time frames.  Each frame covers a contiguous angular
    window of views, and the reconstruction produces one 3D volume per frame.  The constructor
    takes the model of the full scan and the two parameters that define the frame decomposition.
    The sinogram is passed in later, to :meth:`recon`.

    The constructor arguments fix the frame decomposition for the lifetime of the object.
    Reconstruction parameters such as ``mace_prior_weight`` and ``rho_mann`` are ordinary
    parameters set with :meth:`~mbirjax.ParameterHandler.set_params`.

    Args:
        ct_model (mbirjax.TomographyModel): ConeBeamModel or ParallelBeamModel for the full scan.
        frames_per_rotation (int, optional): Number of time frames per full 360 degree rotation.
            This also sets the period of the temporal dejitter filter.  Defaults to 6.
        frame_overlap_factor (float, optional): Number of frames that share any given view.
            Each frame spans frame_overlap_factor * (360 / frames_per_rotation) degrees.
            Defaults to 2.0.
        num_frames (int, optional): If given, reconstruct only the first num_frames time frames.
            Defaults to None, which uses every frame.

    Attributes:
        model_list (list of mbirjax.TomographyModel): One model per time frame.
        view_slices (list of slice): The views of the full sinogram belonging to each frame.
        num_frames (int): Number of time frames.

    Example:
        >>> import mbirjax as mj
        >>> mace = mj.MACE4DModel(ct_model, frames_per_rotation=6, frame_overlap_factor=2.0)
        >>> mace.set_params(mace_prior_weight=0.5, rho_mann=0.5)
        >>> weights = mj.gen_weights(sinogram, weight_type='transmission_root')
        >>> recon_4d, recon_dict = mace.recon(sinogram, weights=weights, max_iterations=10)
    """

    def __init__(self, ct_model, frames_per_rotation=6, frame_overlap_factor=2.0, num_frames=None):
        super().__init__()

        self.ct_model = ct_model
        self.frames_per_rotation = frames_per_rotation
        self.frame_overlap_factor = frame_overlap_factor
        self.sinogram_shape = tuple(ct_model.get_params('sinogram_shape'))

        self.model_list, self.view_slices = mj.construct_time_frame_models(
            ct_model, frames_per_rotation=frames_per_rotation,
            frame_overlap_factor=frame_overlap_factor)
        if num_frames is not None and num_frames < 1:
            raise ValueError(f'num_frames must be at least 1; got {num_frames}.')
        if num_frames is not None and num_frames < len(self.model_list):
            self.model_list = self.model_list[:num_frames]
            self.view_slices = self.view_slices[:num_frames]
        self.num_frames = len(self.model_list)
        # The reconstruction shape comes from the frame models, since those are the
        # models that produce the volumes.  It can differ from the recon shape of ct_model.
        self.recon_shape = tuple(self.model_list[0].get_params('recon_shape'))
        try:
            self.version = importlib_version('mbirjax')
        except Exception:
            self.version = 'unknown'

        # Set the default reconstruction parameters.  no_warning=True allows parameter
        # names that are new in this class to be registered.
        self.set_params(no_warning=True, mace_prior_weight=0.5, rho_mann=0.5,
                        prox_num_iterations=3, prox_stop_threshold=0.02, dejitter=True,
                        dejitter_verbose=0, sigma_prox=None)

        # The device pool is None until set_device_pool is called.  It is resolved to a
        # device list on first use.
        self._devices = None
        self._recon_token = 0

    @overload
    def get_params(self, parameter_names: Union[MACE4DParamNames, list[MACE4DParamNames]]) -> Any: ...

    def get_params(self, parameter_names) -> Any:
        return super().get_params(parameter_names)

    def set_params(self, no_warning=False, no_compile=False, **kwargs):
        """
        Update reconstruction parameters of a MACE4DModel using keyword arguments.

        Args:
            no_warning (bool, optional): If True, disables validity checking and warning messages.
                Defaults to False.
            no_compile (bool, optional): If True, suppresses projector recompilation after updates.
                Defaults to False.
            **kwargs: Parameter names and values to update.

        Example:
            >>> mace.set_params(mace_prior_weight=0.5, rho_mann=0.5, dejitter=True)
        """
        if 'mace_prior_weight' in kwargs:
            _normalize_prior_weights(kwargs['mace_prior_weight'])   # Reject an invalid weight now.

        # sigma_prox is forwarded unchanged to each frame's prox_map.  This model performs
        # no reconstruction of its own, so the base class warning about disabled
        # auto-regularization does not apply.  The warning is suppressed for that one name.
        sigma_prox_given = 'sigma_prox' in kwargs
        sigma_prox = kwargs.pop('sigma_prox', None)
        recompile_flag = False
        if sigma_prox_given:
            recompile_flag |= bool(super().set_params(no_warning=True, no_compile=no_compile,
                                                      sigma_prox=sigma_prox))
        if kwargs:
            recompile_flag |= bool(super().set_params(no_warning=no_warning, no_compile=no_compile,
                                                      **kwargs))
        return recompile_flag

    def set_device_pool(self, devices=None):
        """
        Set the devices that :meth:`recon` distributes its work across.

        The reconstruction is a set of independent tasks, and each task runs entirely on one
        device from this pool.  Calling this method only stores the pool.  It takes effect at
        the next :meth:`recon` call.

        Args:
            devices (optional): The devices to use, in one of the following forms.
                Defaults to None.

                * None: all visible GPUs, or the CPU when there is no GPU.  This is also
                  the behavior when the method is never called.
                * 'cpu' or 'gpu': all devices of that platform.
                * int n: the first n devices of the default platform.  A value of 1 runs
                  every task on one device.
                * sequence of ints: the devices with those indices.
                * sequence of jax devices: exactly those devices.

        Raises:
            ValueError: If the platform string is not 'cpu' or 'gpu', a GPU is requested
                when none is available, or more devices are requested than are visible.

        Example:
            >>> mace.set_device_pool(2)   # run on the first two GPUs
        """
        self._devices = _resolve_devices(devices)

    @property
    def devices(self):
        """Return the device pool set by set_device_pool, or the automatic selection if none was set."""
        return self._devices if self._devices is not None else _resolve_devices(None)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def recon(self, sinogram, weights=None, init_recon=None, max_iterations=10,
              stop_threshold_change_pct=0.2, init_dir=None, log_dir=None):
        """
        Compute a 4D MACE reconstruction from the sinogram of the full scan.

        The method returns one 3D volume per time frame.  All visible GPUs are used by
        default.  The device set can be changed with :meth:`set_device_pool`.

        Args:
            sinogram (ndarray): Sinogram of the full scan, with shape
                (num_views, num_det_rows, num_det_channels).  It is sliced into per-frame
                sinograms internally.
            weights (ndarray, optional): Positive weights with the same shape as ``sinogram``.
                Defaults to None, which uses unit weights.  For transmission data,
                ``mj.gen_weights(sinogram, weight_type='transmission_root')`` is recommended.
            init_recon (ndarray, optional): Initial 4D image with shape (num_frames, nx, ny, nz).
                Defaults to None.  In that case the initial image is loaded from ``init_dir``
                if available, and is otherwise computed by reconstructing each frame separately.
            max_iterations (int, optional): Maximum number of MACE iterations.  Defaults to 10.
            stop_threshold_change_pct (float, optional): Stop when the percent change of the
                consensus reconstruction in one iteration falls below this value.  Defaults
                to 0.2.  A value of 0 runs all ``max_iterations`` iterations.
            init_dir (str, optional): Directory used to cache the computed initial image.
                If the directory holds a valid image, that image is used.  Otherwise the
                initialization is computed and saved there.  Defaults to None, which disables
                caching.
            log_dir (str, optional): Directory for the log files run_info.txt, timing_log.csv
                and task_log.csv.  Defaults to None, which writes no log files.

        Returns:
            (recon, recon_dict): The reconstruction and a dictionary describing the run.
                - recon (ndarray): 4D reconstruction with shape (num_frames, nx, ny, nz).
                - recon_dict (dict): Run settings, per-iteration timing, and model parameters.

        Raises:
            ValueError: If ``sinogram``, ``weights`` or ``init_recon`` has the wrong shape.

        Example:
            >>> weights = mj.gen_weights(sinogram, weight_type='transmission_root')
            >>> recon_4d, recon_dict = mace.recon(sinogram, weights=weights, log_dir='./logs')
        """
        num_frames = self.num_frames
        beta = _normalize_prior_weights(self.get_params('mace_prior_weight'))
        verbose = self.get_params('verbose')
        rho_mann = self.get_params('rho_mann')

        sinogram = self._validate_sinogram(sinogram, 'sinogram')
        if weights is not None:
            weights = self._validate_sinogram(weights, 'weights')

        devs = self.devices
        self._assign_and_place(devs, sinogram, weights)
        if verbose:
            counts = [self._frame_device.count(d) for d in range(len(devs))]
            print(f"[MACE] {len(devs)} device(s); prox frames per device: {counts}; "
                  f"denoise on devices {self._orient_device}.")
            print(f"[MACE] Start 4D reconstruction with {num_frames} time frames.")

        # Each device gets one single-thread executor, so a device's tasks always run on
        # the same thread.  This keeps the per-thread denoiser caches valid, and it
        # ensures that each model object is used by only one thread.
        executors = ([concurrent.futures.ThreadPoolExecutor(max_workers=1) for _ in devs]
                     if len(devs) > 1 else None)
        # Incrementing this token makes each denoiser reconfigure once per recon call.
        self._recon_token += 1
        timing_rows = []
        try:
            # -- Initialization ------------------------------------------------
            if init_recon is not None:
                init_recon = self._validate_init_recon(init_recon)
                init_source = "provided by caller"
                if verbose:
                    print("[MACE] Using provided init_recon.")
            else:
                if init_dir is not None:
                    init_recon = self._load_cached_init(init_dir)
                if init_recon is not None:
                    init_source = f"cached ({os.path.join(init_dir, 'init_recon.npy')})"
                else:
                    init_recon = self._compute_init_recon(devs, executors, init_dir)
                    init_source = (f"computed ({self.num_frames} frames, "
                                   f"{_INIT_MBIR_ITERATIONS} MBIR iterations each)")

            # -- Global denoiser sigma (one value for all orientations) --------
            global_sigma = self._estimate_global_sigma(init_recon, devs[0])
            if not np.isfinite(global_sigma) or global_sigma <= 0:
                # Every denoiser is scaled by this sigma, and a zero value would later cause
                # a division by zero deep inside the qGGMRF code.  Report the cause here.
                raise ValueError(
                    f"The denoiser noise level estimated from the initial image is "
                    f"{global_sigma}, which happens when that image is constant (all zeros, "
                    f"for instance).  Supply a non-constant init_recon, or omit init_recon and "
                    f"let the model compute the per-frame initialization.")
            if verbose:
                print(f"[MACE] Global denoiser sigma = {global_sigma:.6g}")

            run_settings = self._run_settings(devs, init_source, global_sigma, weights,
                                              max_iterations, stop_threshold_change_pct)

            # -- MACE state (all on CPU / NumPy) -------------------------------
            W = [np.copy(init_recon) for _ in range(4)]
            X = [np.copy(init_recon) for _ in range(4)]
            # This scratch array is reused by the consensus update in every iteration.
            _consensus_scratch = np.empty_like(init_recon)

            # -- Log files -----------------------------------------------------
            timing_log_path = task_log_path = None
            if log_dir is not None:
                os.makedirs(log_dir, exist_ok=True)
                _write_run_info(os.path.join(log_dir, "run_info.txt"), run_settings)
                timing_log_path = os.path.join(log_dir, "timing_log.csv")
                with open(timing_log_path, "w", newline="") as f:
                    csv.DictWriter(f, fieldnames=_TIMING_FIELDS).writeheader()
                task_log_path = os.path.join(log_dir, "task_log.csv")
                with open(task_log_path, "w", newline="") as f:
                    csv.DictWriter(f, fieldnames=_TASK_FIELDS).writeheader()

            # -- Main MACE loop ------------------------------------------------
            # xbar is the consensus reconstruction sum(beta[k] X[k]).  Its percent change
            # per iteration is the convergence measure and the stopping criterion.
            xbar = init_recon
            for itr in range(max_iterations):
                itr_t0 = time.time()
                if verbose:
                    print(f"\n[MACE] -- Iteration {itr + 1}/{max_iterations} --")

                # The tasks only read W.  W is not written until all tasks have finished,
                # so no copy of W is needed.
                tasks = []
                for t in range(num_frames):
                    d = self._frame_device[t]
                    tasks.append((d, ("prox", t),
                                  lambda tt=t, dd=d: self._run_prox_task(tt, W[0][tt], X[0][tt], devs[dd])))
                for k in range(3):
                    d = self._orient_device[k]
                    perm = _PRIOR_ORIENTATIONS[k][1]
                    tasks.append((d, ("denoise", k),
                                  lambda kk=k, pp=perm, dd=d: self._run_denoise_task(
                                      W[kk + 1], pp, global_sigma, devs[dd])))
                results, task_rows = self._run_task_set(executors, tasks, itr_t0)

                # Gather the prox results in frame order and dejitter the assembled stack.
                # X[0] keeps the dejittered stack, which feeds the next prox calls.
                X[0] = self._dejitter(np.stack([results[("prox", t)] for t in range(num_frames)]))
                for k in range(3):
                    X[k + 1] = results[("denoise", k)]

                # This is the ADMM consensus update, computed in place on the CPU.  The
                # equivalent expression form allocates about 28 full-size arrays per
                # iteration and measured 7.1 times slower on the full-resolution volume.
                # The in-place form computes the same values in the same order.
                scratch = _consensus_scratch
                z = np.zeros_like(X[0])
                for k in range(4):
                    np.multiply(X[k], 2.0, out=scratch)
                    scratch -= W[k]
                    scratch *= beta[k]
                    z += scratch
                for k in range(4):
                    np.subtract(z, X[k], out=scratch)
                    scratch *= (2.0 * rho_mann)
                    W[k] += scratch

                xbar_prev = xbar
                xbar = np.zeros_like(X[0])
                for k in range(4):
                    np.multiply(X[k], beta[k], out=scratch)
                    xbar += scratch
                denom = np.linalg.norm(xbar_prev)
                change_pct = 100.0 * np.linalg.norm(xbar - xbar_prev) / denom if denom > 0 else np.inf

                iteration_sec = time.time() - itr_t0
                prox_total = sum(r[4] - r[3] for r in task_rows if r[0] == "prox")
                denoise_total = sum(r[4] - r[3] for r in task_rows if r[0] == "denoise")
                makespan = max(r[4] for r in task_rows)
                timing_row = dict(zip(_TIMING_FIELDS,
                                      [itr + 1, prox_total, denoise_total, makespan,
                                       iteration_sec, change_pct]))
                timing_rows.append(timing_row)
                if timing_log_path is not None:
                    with open(timing_log_path, "a", newline="") as f:
                        csv.DictWriter(f, fieldnames=_TIMING_FIELDS).writerow(timing_row)
                    with open(task_log_path, "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=_TASK_FIELDS)
                        for kind, index, dev_idx, start, end in sorted(task_rows, key=lambda r: r[3]):
                            w.writerow(dict(zip(_TASK_FIELDS,
                                                [itr + 1, kind, index, dev_idx,
                                                 round(start, 3), round(end, 3)])))
                if verbose:
                    print(f"[MACE] Timing: itr={itr + 1}, prox={prox_total:.2f}s, "
                          f"denoise={denoise_total:.2f}s, makespan={makespan:.2f}s, "
                          f"total={iteration_sec:.2f}s, change={change_pct:.4f}%")

                if change_pct < stop_threshold_change_pct:
                    if verbose:
                        print(f"[MACE] Change threshold stopping condition reached "
                              f"({change_pct:.4f}% < {stop_threshold_change_pct}%).")
                    break
        finally:
            if executors is not None:
                for ex in executors:
                    ex.shutdown(wait=True)

        if verbose:
            print("\n[MACE] Reconstruction complete.")

        # run_info.txt was written before the loop so that the settings of a long run can
        # be read while it runs.  Rewrite it now that the iteration count is known.
        run_settings['iterations completed'] = len(timing_rows)
        if log_dir is not None:
            _write_run_info(os.path.join(log_dir, "run_info.txt"), run_settings)
        recon_dict = {
            'recon_params': run_settings,
            'timing': timing_rows,
            'notes': 'Reconstruction completed: {}\n\n'.format(datetime.datetime.now()),
            'model_params': self.params.copy(),
        }
        return xbar, recon_dict

    # ------------------------------------------------------------------
    # Task execution
    # ------------------------------------------------------------------

    def _assign_and_place(self, devs, sinogram, weights):
        """Fix the task-to-device assignment, pin the models, and place the per-frame data.

        The assignment is computed once and reused for every iteration.  Each frame's
        sinogram and weights are uploaded to the frame's device once and stay there.  The
        frames are slices of the full arrays, so nothing is copied on the host.
        """
        plane_counts = [self.recon_shape[2], self.recon_shape[0], self.recon_shape[1]]  # XY-t, YZ-t, XZ-t
        self._frame_device, self._orient_device = _assign_tasks(self.num_frames, plane_counts, len(devs))
        for t in range(self.num_frames):
            self.model_list[t].configure_devices([devs[self._frame_device[t]]])
        self._sino_dev = [jax.device_put(np.asarray(sinogram[self.view_slices[t]]),
                                         devs[self._frame_device[t]]) for t in range(self.num_frames)]
        if weights is None:
            self._weights_dev = [None] * self.num_frames
        else:
            self._weights_dev = [jax.device_put(np.asarray(weights[self.view_slices[t]]),
                                                devs[self._frame_device[t]]) for t in range(self.num_frames)]

    def _run_task_set(self, executors, tasks, t0):
        """Run a list of tasks and wait for all of them to finish.

        Each task is a tuple (device_index, tag, fn).  When executors is None, the tasks
        run one at a time on the calling thread.  The method returns a dict mapping each
        tag to its result, and a list of rows (kind, index, device_index, start, end)
        with times measured relative to t0.  A failed task raises RuntimeError naming
        the task.
        """
        results = {}
        rows = []

        def run_one(fn):
            start = time.time() - t0
            out = fn()
            return out, start, time.time() - t0

        if executors is None:
            for dev_idx, tag, fn in tasks:
                out, start, end = run_one(fn)
                results[tag] = out
                rows.append((tag[0], tag[1], dev_idx, start, end))
            return results, rows

        futures = {}
        for dev_idx, tag, fn in tasks:
            futures[executors[dev_idx].submit(run_one, fn)] = (tag, dev_idx)
        for fut in concurrent.futures.as_completed(futures):
            tag, dev_idx = futures[fut]
            try:
                out, start, end = fut.result()
            except Exception as err:
                raise RuntimeError(f"task {tag} on device {dev_idx} failed") from err
            results[tag] = out
            rows.append((tag[0], tag[1], dev_idx, start, end))
        return results, rows

    def _run_prox_task(self, t, W0_t, X0_t, device):
        """Run one frame's proximal map on its assigned device."""
        return np.asarray(
            self.model_list[t].prox_map(
                prox_input=jax.device_put(W0_t, device),
                sinogram=self._sino_dev[t],
                sigma_prox=self.get_params('sigma_prox'),
                weights=self._weights_dev[t],
                init_recon=jax.device_put(X0_t, device),
                max_iterations=self.get_params('prox_num_iterations'),
                stop_threshold_change_pct=self.get_params('prox_stop_threshold'),
                logfile_path=None,
                print_logs=False,
            )[0])

    def _run_denoise_task(self, W_k, permute_vector, sigma, device):
        """Run one orientation's batched qGGMRF denoise on its assigned device."""
        return _denoiser_wrapper(self._dejitter(W_k), permute_vector=permute_vector,
                                 sigma=sigma, device=device,
                                 config_token=self._recon_token)

    def _init_frame_task(self, t, device):
        """Run one frame's MBIR initialization recon on its assigned device."""
        return np.asarray(
            self.model_list[t].recon(
                self._sino_dev[t],
                weights=self._weights_dev[t],
                max_iterations=_INIT_MBIR_ITERATIONS,
                stop_threshold_change_pct=self.get_params('prox_stop_threshold'),
                logfile_path=None,
                print_logs=False,
            )[0])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _estimate_global_sigma(self, init_recon, device):
        """Estimate one global noise sigma from the initial image for all denoising."""
        # Merge the first two axes so the estimator sees a 3D array.  The estimator
        # subsamples internally.
        image_3d = init_recon.reshape(-1, init_recon.shape[2], init_recon.shape[3])
        denoiser = mj.QGGMRFDenoiser(image_3d.shape)
        denoiser.configure_devices([device])
        return float(denoiser.estimate_image_noise_std(image_3d))

    def _dejitter(self, x):
        """Apply the temporal dejitter filter when enabled.  Otherwise return x unchanged."""
        if not self.get_params('dejitter'):
            return x
        # Printing is controlled by dejitter_verbose rather than verbose.  This filter runs
        # four times per iteration and would repeat the same output each time.
        return _dejitter_4d_dct(x, period=self.frames_per_rotation, harmonics=True,
                                band_width=1, dtype=np.float32,
                                verbose=bool(self.get_params('dejitter_verbose')))

    def _run_settings(self, devs, init_source, global_sigma, weights, max_iterations,
                      stop_threshold_change_pct):
        """Return the settings that describe this run, for run_info.txt and the recon dict."""
        if len(devs) > 1:
            mode = f"task queue over {len(devs)} devices: " + ", ".join(str(d) for d in devs)
        else:
            mode = f"serial on {devs[0]}"
        beta = _normalize_prior_weights(self.get_params('mace_prior_weight'))
        sigma_prox = self.get_params('sigma_prox')
        return {
            'date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'mbirjax version': self.version,
            'time frames': self.num_frames,
            'frame shape': self.recon_shape,
            'views per frame': self.view_slices[0].stop - self.view_slices[0].start,
            'mode': mode,
            'init source': init_source,
            'weights': 'unit (weights=None)' if weights is None else 'supplied by caller',
            'beta [fwd, xyt, yzt, xzt]': [round(float(b), 4) for b in beta],
            'rho_mann': self.get_params('rho_mann'),
            'max_iterations': max_iterations,
            'stop_threshold_change_pct': stop_threshold_change_pct,
            'prox_num_iterations': self.get_params('prox_num_iterations'),
            'prox_stop_threshold': self.get_params('prox_stop_threshold'),
            'sigma_prox': 'auto' if sigma_prox is None else sigma_prox,
            'denoiser sigma (global)': float(global_sigma),
            'dejitter': self.get_params('dejitter'),
            'frames_per_rotation': self.frames_per_rotation,
            'frame_overlap_factor': self.frame_overlap_factor,
        }

    def _validate_sinogram(self, sinogram, name):
        """Return the array unchanged, or raise ValueError if its shape is not the sinogram shape."""
        sinogram = np.asarray(sinogram)
        if sinogram.shape != self.sinogram_shape:
            raise ValueError(f"{name} shape {sinogram.shape} does not match the model's "
                             f"sinogram shape {self.sinogram_shape}.")
        return sinogram

    def _expected_init_shape(self):
        """Return the required shape of the initial image: (num_frames,) + recon_shape."""
        return (self.num_frames,) + self.recon_shape

    def _validate_init_recon(self, init_recon):
        """Return init_recon as float32, or raise ValueError on a wrong shape."""
        init_recon = np.asarray(init_recon, dtype=np.float32)
        expected = self._expected_init_shape()
        if init_recon.shape != expected:
            raise ValueError(
                f"init_recon shape {init_recon.shape} does not match expected {expected}."
            )
        return init_recon

    def _load_cached_init(self, init_dir):
        """Load init_recon.npy from init_dir, or return None.

        A missing file is normal on a first run and returns None silently.  A file that
        cannot be loaded or has the wrong shape produces a warning and returns None.
        """
        path = os.path.join(init_dir, "init_recon.npy")
        if not os.path.isfile(path):
            return None
        try:
            init_recon = self._validate_init_recon(np.load(path))
        except (ValueError, OSError) as e:
            warnings.warn(f"init_dir has an invalid initialization image ({e}); recomputing.")
            return None
        if self.get_params('verbose'):
            print(f"[MACE] Using cached init from {path}.")
        return init_recon

    def _compute_init_recon(self, devs, executors, init_dir):
        """Compute the initial image by reconstructing each frame separately.

        The computation uses the same workers and the same frame-to-device assignment as
        the MACE loop, so the compiled programs and resident data are reused there.
        """
        verbose = self.get_params('verbose')
        if verbose:
            print(f"[MACE] Computing initial MBIR recon on {len(devs)} device(s)...")
        t0 = time.time()
        tasks = [(self._frame_device[t], ("init", t),
                  lambda tt=t: self._init_frame_task(tt, devs[self._frame_device[tt]]))
                 for t in range(self.num_frames)]
        results, _ = self._run_task_set(executors, tasks, t0)
        init_recon = np.stack([results[("init", t)] for t in range(self.num_frames)])
        if init_dir is not None:
            os.makedirs(init_dir, exist_ok=True)
            np.save(os.path.join(init_dir, "init_recon.npy"), init_recon)
        if verbose:
            print(f"[MACE] Initialization done in {time.time() - t0:.2f} sec.")
        return init_recon


# Thread-local cache of QGGMRFDenoiser objects, keyed by (shape, device).  The cache
# ensures that no denoiser instance is shared across threads.
_THREAD_LOCAL = threading.local()


# ---------------------------------------------------------------------------
# Device selection and task assignment
# ---------------------------------------------------------------------------

def _resolve_devices(devices):
    """Return the list of jax devices to use.  See MACE4DModel.set_device_pool."""
    if devices is None:
        # The automatic choice is every GPU, or one CPU device when there is no GPU.  This
        # model runs one independent task per device, so using the several virtual CPU
        # devices of one machine would oversubscribe the same cores.
        return list(gpu_devices()) or [cpu_devices()[0]]
    if isinstance(devices, str):
        platform = devices.lower()
        if platform == 'cpu':
            return list(cpu_devices())
        if platform == 'gpu':
            pool = list(gpu_devices())
            if not pool:
                raise ValueError("set_device_pool('gpu') was requested but no GPU backend "
                                 "is available.")
            return pool
        raise ValueError("set_device_pool platform string must be 'cpu' or 'gpu'; "
                         "got {!r}.".format(devices))
    if isinstance(devices, (int, np.integer)):
        pool = default_devices()
        if not 1 <= int(devices) <= len(pool):
            raise ValueError(f"devices={devices}, but {len(pool)} device(s) are visible.")
        return pool[:int(devices)]
    devices = list(devices)
    if devices and all(isinstance(d, (int, np.integer)) for d in devices):
        pool = default_devices()
        return [pool[int(i)] for i in devices]
    return devices


def _assign_tasks(num_frames, plane_counts, num_devices):
    """Assign the tasks to devices, always placing the next task on the least-loaded device.

    The three denoise tasks are placed first, largest first, with estimated cost
    proportional to their hyperplane counts.  Each prox task then has unit cost and
    goes to the least-loaded device.  The assignment is fixed for the whole run.

    Returns:
        tuple: (frame_device, orient_device).  frame_device lists the device index of
            each frame's prox task.  orient_device lists the device index of each
            orientation's denoise task.
    """
    loads = [0.0] * num_devices
    orient_device = [0] * len(plane_counts)
    for k in sorted(range(len(plane_counts)), key=lambda k: -plane_counts[k]):
        d = loads.index(min(loads))
        orient_device[k] = d
        loads[d] += _DENOISE_COST_PER_PLANE * plane_counts[k]
    frame_device = [0] * num_frames
    for t in range(num_frames):
        d = loads.index(min(loads))
        frame_device[t] = d
        loads[d] += 1.0
    return frame_device, orient_device


def _write_run_info(path, run_settings):
    """Write the run settings to a human-readable text file."""
    width = max(len(key) for key in run_settings)
    lines = ["# MACE4DModel run settings"]
    lines += ["{:<{width}} = {}".format(key, value, width=width)
              for key, value in run_settings.items()]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# DCT-I temporal dejitter
# ---------------------------------------------------------------------------

def _dejitter_4d_dct(
    recon_4d,
    period,
    harmonics=True,
    band_width=1,
    dtype=np.float32,
    chunk_size=None,
    verbose=False,
):
    """Remove periodic temporal jitter from a 4D reconstruction via DCT-I filtering.

    Gating a continuous scan into overlapping frames imprints a periodic modulation on the
    time axis, one period per rotation.  Zeroing the DCT-I modes at that period and its
    harmonics removes the modulation while leaving the rest of the temporal spectrum intact.

    Args:
        recon_4d (ndarray): 4D volume, shape (time, x, y, z).
        period (float or int): Main jitter period in frames (e.g. 6 for a 6-phase
            gating protocol).
        harmonics (bool or list of int): True removes the main period and all harmonics
            with period/h >= 2. False removes only the main period. A list specifies
            explicit harmonic indices h to remove.
        band_width (int): Number of DCT-I modes to zero on each side of the target
            mode. band_width=1 zeroes [k_center-1, k_center, k_center+1].
        dtype (np.dtype): Working dtype (float32 reduces memory).
        chunk_size (int or None): Process the last spatial axis in chunks of this size
            to reduce peak memory. None processes the whole axis in one pass.
        verbose (bool): Print the modes being zeroed. Defaults to False.

    Returns:
        ndarray: Dejittered volume, same shape as recon_4d.
    """
    recon_4d = np.asarray(recon_4d)
    N = recon_4d.shape[0]
    spatial_shape = recon_4d.shape[1:]

    if harmonics is False:
        harmonic_list = [1]
    elif harmonics is True:
        max_h = int(np.floor(period / 2))
        harmonic_list = list(range(1, max_h + 1))
    else:
        harmonic_list = list(harmonics)

    periods_to_remove = [period / h for h in harmonic_list]

    if verbose:
        print("Input shape:", recon_4d.shape)
        print("Periods to remove:", periods_to_remove)

    Z = spatial_shape[-1]
    if chunk_size is None:
        chunk_size = Z

    recon_dejittered = np.empty((N,) + spatial_shape, dtype=dtype)
    for z0 in range(0, Z, chunk_size):
        z1 = min(z0 + chunk_size, Z)
        block = np.asarray(recon_4d[..., z0:z1], dtype=dtype)
        C = dct(block, type=1, norm="ortho", axis=0)
        for p in periods_to_remove:
            k_center = 2 * (N - 1) / p
            k0 = int(round(k_center))
            lo = max(0, k0 - band_width)
            hi = min(C.shape[0], k0 + band_width + 1)
            if lo < hi:
                C[lo:hi, ...] = 0
            if verbose and z0 == 0:
                actual_period = 2 * (N - 1) / k0 if k0 != 0 else np.inf
                print(
                    f"  Removed period {p:.3g}: "
                    f"k~{k_center:.2f}, rounded k={k0}, "
                    f"actual period~{actual_period:.3g}, "
                    f"zeroed k={lo}:{hi - 1}"
                )
        recon_dejittered[..., z0:z1] = idct(C, type=1, norm="ortho", axis=0).astype(dtype, copy=False)
        del block, C
    return recon_dejittered


# ---------------------------------------------------------------------------
# Agent weights
# ---------------------------------------------------------------------------

def _normalize_prior_weights(prior_weight):
    """
    Convert a scalar or list prior weight into the form [forward_w, xyt_w, yzt_w, xzt_w].

    A scalar w becomes [1-w, w/3, w/3, w/3].  A list [w1, w2, w3] becomes
    [1-(w1+w2+w3), w1, w2, w3].
    """
    if isinstance(prior_weight, (list, tuple, np.ndarray)):
        prior = [float(w) for w in prior_weight]
        if len(prior) != 3:
            raise ValueError("mace_prior_weight list must have 3 entries [xyt, yzt, xzt].")
    else:
        w = float(prior_weight) / 3.0
        prior = [w, w, w]
    if any(w < 0 for w in prior) or sum(prior) > 1.0:
        raise ValueError("mace_prior_weight must be nonnegative and sum to at most 1.")
    return [1.0 - sum(prior)] + prior


# ---------------------------------------------------------------------------
# Device-pinned denoiser helpers
# ---------------------------------------------------------------------------
#
# IMPORTANT: each QGGMRFDenoiser must be pinned to exactly one GPU with
# configure_devices([device]).  Without pinning, mbirjax shards the denoiser across
# every visible GPU.  Several such denoisers running concurrently then deadlock in
# NCCL with the error "Acquire clique ... may be stuck".  The cache key includes the
# device, so each thread gets its own pinned instance.

def _get_qggmrf_denoiser(shape, device):
    """Return a cached QGGMRFDenoiser pinned to one device.  The cache is per thread."""
    cache = getattr(_THREAD_LOCAL, "denoiser_cache", None)
    if cache is None:
        cache = {}
        _THREAD_LOCAL.denoiser_cache = cache
    key = (shape, device)
    if key not in cache:
        denoiser = mj.QGGMRFDenoiser(shape)
        denoiser.configure_devices([device])
        cache[key] = denoiser
    return cache[key]


# Denoiser iteration settings.  These match the defaults of QGGMRFDenoiser.denoise().
_DENOISE_MAX_ITERATIONS = 15
_DENOISE_STOP_THRESHOLD_PCT = 0.2

# Multiplier for the batch-size estimate.  It approximates the bytes used per volume
# during the jitted sweep, as a multiple of the volume size.  The value is a heuristic.
_DENOISE_BUFFER_MULTIPLIER = 16

# The batch size is capped so that a bad memory estimate cannot request an enormous
# compilation.
_DENOISE_BATCH_CAP = 128

# Floor for the auto-estimated qGGMRF regularization scale sigma_x.  A batch dominated
# by background hyperplanes can produce a sigma_x near zero, and the qGGMRF solver
# would then return NaN for the whole batch.
_SIGMA_X_FLOOR = 1e-6


def _configure_denoiser(denoiser, sigma, image_for_stats):
    """Set the shared sigma and the regularization constants on the denoiser.

    This replicates the parameter setup that QGGMRFDenoiser.denoise() performs, so the
    jitted sweep can be called directly with shared constants.

    The standard method auto_set_regularization_params() is not used here.  For a
    hyperplane batch it would compute its statistics from only the first hyperplane.
    The individual auto-set methods are therefore called directly on the full array.
    """
    denoiser.set_params(use_ror_mask=False, sigma_noise=float(sigma))
    verbose = denoiser.get_params('verbose')
    denoiser.set_params(verbose=0)
    image_for_stats = np.asarray(image_for_stats)
    sino_indicator = denoiser._get_sino_indicator(image_for_stats)
    denoiser.auto_set_sigma_y(image_for_stats, sino_indicator)
    recon_std = denoiser._get_estimate_of_recon_std(image_for_stats, sino_indicator)
    if not np.isfinite(recon_std):
        recon_std = 0.0
    denoiser.auto_set_sigma_x(recon_std)
    denoiser.auto_set_sigma_prox(recon_std)
    sigma_x = denoiser.get_params('sigma_x')
    if not np.isfinite(sigma_x) or sigma_x < _SIGMA_X_FLOOR:
        denoiser.set_params(no_warning=True, sigma_x=np.float32(_SIGMA_X_FLOOR))
    denoiser.set_params(verbose=verbose)
    # The sweep's progress callback converts its arguments with int() and float(),
    # which fails on the batched arrays that a vmapped sweep passes it.  Silence it.
    denoiser._log_denoise_progress = lambda *args: None
    # Recompute the sweep constants for this configuration.  The pixel partition is
    # drawn at random each time it is generated, so it is built once here and reused.
    # Otherwise repeated calls would use different VCD subset orders.
    denoiser._mace4d_constants = None
    _denoise_constants(denoiser)
    # New constants invalidate the cached batch size and compiled batch function.
    denoiser._mace4d_batch = None
    denoiser._mace4d_batched_fn = None


def _denoise_constants(denoiser):
    """Return the constant arguments of the denoiser's jitted sweep, cached per configuration."""
    cached = getattr(denoiser, '_mace4d_constants', None)
    if cached is not None:
        return cached
    image_shape, granularity = denoiser.get_params(['recon_shape', 'granularity'])
    # Keep at least 64 pixels per VCD subset.  With very small subsets the qGGMRF line
    # search can reach 0/0 in flat regions.  At real volume sizes this limit leaves the
    # subset count unchanged.
    num_pixels = image_shape[0] * image_shape[1]
    num_subsets = max(1, min(granularity[0], num_pixels // 64))
    partition = mj.gen_set_of_pixel_partitions(image_shape, [num_subsets],
                                               use_ror_mask=False)[0]
    fm_constant = 1.0 / (denoiser.get_params('sigma_y') ** 2.0)
    qggmrf_nbr_wts, sigma_x, p, q, T = denoiser.get_params(
        ['qggmrf_nbr_wts', 'sigma_x', 'p', 'q', 'T'])
    qggmrf_params = (mj.get_b_from_nbr_wts(qggmrf_nbr_wts), sigma_x, p, q, T)
    denoiser._mace4d_constants = (partition, fm_constant, qggmrf_params, image_shape)
    return denoiser._mace4d_constants


def _auto_batch_size(vol_shape, device):
    """Return the largest volume batch that fits in device memory.

    A device without memory statistics, such as the CPU, gets a small fixed batch.
    """
    stats = getattr(device, 'memory_stats', lambda: None)()
    if not stats:
        return 4
    free = stats.get('bytes_limit', 0) - stats.get('bytes_in_use', 0)
    vol_bytes = 4 * int(np.prod(vol_shape))
    return max(1, int(0.5 * free) // (_DENOISE_BUFFER_MULTIPLIER * vol_bytes))


def _batched_hyperplane_denoise(x, denoiser, device):
    """Denoise a stack of 3D volumes of equal shape with shared, preconfigured settings.

    One jax.vmap call runs the denoiser's jitted sweep over a whole batch, so the device
    processes many volumes at once instead of one at a time.  The volumes are
    independent, so the result equals per-volume denoising with the same constants.

    Args:
        x (ndarray): Stack of volumes, shape (num_volumes, d0, d1, d2).
        denoiser (QGGMRFDenoiser): Configured for shape (d0, d1, d2) via
            _configure_denoiser.
        device (jax.Device): Device on which denoising runs.

    Returns:
        ndarray: Denoised stack, same shape as x.
    """
    num_vols, vol_shape = x.shape[0], x.shape[1:]
    partition, fm_constant, qggmrf_params, image_shape = _denoise_constants(denoiser)
    stop_thresh = _DENOISE_STOP_THRESHOLD_PCT / 100.0

    def denoise_one(flat_vol):
        out, _, _, _ = denoiser._denoise_single_device(
            flat_vol, jnp.zeros_like(flat_vol), partition, fm_constant,
            qggmrf_params, image_shape, _DENOISE_MAX_ITERATIONS, stop_thresh, 0)
        return out

    # Each configuration uses one fixed batch size and one compiled batch function.
    # The last block is padded to the fixed size, so every call reuses the same
    # compiled program.
    if getattr(denoiser, "_mace4d_batch", None) is None:
        denoiser._mace4d_batch = min(_DENOISE_BATCH_CAP, num_vols,
                                     _auto_batch_size(vol_shape, device))
        denoiser._mace4d_batched_fn = jax.jit(jax.vmap(denoise_one))

    flat = x.reshape(num_vols, -1, vol_shape[-1])
    y = np.empty_like(x)
    b0 = 0
    with jax.default_device(device):
        while b0 < num_vols:
            batch = denoiser._mace4d_batch
            fn = denoiser._mace4d_batched_fn
            b1 = min(b0 + batch, num_vols)
            block = flat[b0:b1]
            if b1 - b0 < batch:
                pad = np.zeros((batch - (b1 - b0),) + block.shape[1:], dtype=block.dtype)
                block = np.concatenate([block, pad], axis=0)
            try:
                out = np.asarray(fn(jax.device_put(block, device)))
            except Exception as err:
                # The device ran out of memory.  Halve the batch size and recompile.
                if "RESOURCE_EXHAUSTED" in str(err) and denoiser._mace4d_batch > 1:
                    denoiser._mace4d_batch = max(1, denoiser._mace4d_batch // 2)
                    denoiser._mace4d_batched_fn = jax.jit(jax.vmap(denoise_one))
                    continue
                raise
            y[b0:b1] = out[: b1 - b0].reshape((b1 - b0,) + vol_shape)
            b0 = b1
    return y


def _denoiser_wrapper(x, permute_vector, sigma, device, config_token=None):
    """Denoise the hyperplanes of a 4D volume at the shared global sigma.

    The volume is permuted so that the hyperplane axis is first.  The resulting stack
    of 3D volumes is denoised in batches, and the result is permuted back.

    Args:
        x (ndarray): 4D volume, shape (num_frames, nx, ny, nz).
        permute_vector (tuple of int): Permutation that puts the hyperplane axis first.
        sigma (float): Global noise sigma shared by every volume.
        device (jax.Device): Device on which denoising runs.
        config_token (hashable or None): The denoiser is reconfigured only when this
            token changes, which happens once per recon call.  A value of None
            reconfigures on every call.

    Returns:
        ndarray: Denoised volume, same shape as x.
    """
    x_perm = np.ascontiguousarray(np.transpose(x, permute_vector))
    denoiser = _get_qggmrf_denoiser(x_perm.shape[1:], device)
    if config_token is None or getattr(denoiser, "_mace4d_token", None) != config_token:
        # Regularization statistics come from the whole stack (merged to 3D),
        # so every orientation sees the same voxel population.
        _configure_denoiser(denoiser, sigma, x_perm.reshape(-1, *x_perm.shape[2:]))
        denoiser._mace4d_token = config_token
    y_perm = _batched_hyperplane_denoise(x_perm, denoiser, device)
    inv_perm = np.argsort(permute_vector)
    return np.transpose(y_perm, inv_perm)
