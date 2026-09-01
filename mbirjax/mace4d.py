"""
4D MACE reconstruction: the MACE4DModel class and the helpers it uses for DCT-I temporal
dejitter and batched hyperplane denoising.

MACE4DModel reconstructs a time sequence of volumes from a single continuous scan.  The scan
is divided into overlapping time frames (see :func:`mbirjax.construct_time_frame_models`), and
the frames are reconciled by MACE: one cone-beam ``prox_map`` per time frame as the forward
agent, plus three batched qGGMRF denoisers acting on the XY-t, YZ-t and XZ-t hyperplanes as
prior agents.  Each iteration's work is a set of independent tasks executed by one worker
thread per device under a fixed least-loaded assignment; a single device runs the same tasks
inline.
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

# Only the model and its parameter-name type are public; everything else in this module is an
# implementation detail, so `from .mace4d import *` does not add scipy/typing names to mbirjax.
__all__ = ['MACE4DModel', 'MACE4DParamNames']

# MBIR iterations for the per-frame initialization recon.
_INIT_MBIR_ITERATIONS = 15

# Prior-agent hyperplane orientations. The permutation moves the hyperplane
# axis first; recon axes are (t, x, y, z).
_PRIOR_ORIENTATIONS = [
    ("XY-t", (3, 0, 1, 2)),  # nz hyperplanes of shape (nt, nx, ny)
    ("YZ-t", (1, 0, 2, 3)),  # nx hyperplanes of shape (nt, ny, nz)
    ("XZ-t", (2, 0, 1, 3)),  # ny hyperplanes of shape (nt, nx, nz)
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

# Estimated cost of denoising one hyperplane, in units of one prox_map task.
# Measured on an H100 at smoke scale; only the relative size matters, and only
# for the static load-balancing assignment.
_DENOISE_COST_PER_PLANE = 0.015


class MACE4DModel(ParameterHandler):
    """4D MACE CT reconstruction model.

    The model is built from a cone-beam or parallel-beam model of the full scan.  The frame
    decomposition follows from that model's angles alone, so the per-frame models and view
    slices exist immediately after construction, before any data is supplied; the sinogram
    enters at :meth:`recon`, as in every other mbirjax model.

    Args:
        ct_model (mbirjax.TomographyModel): Fully-built ConeBeamModel or ParallelBeamModel for
            the full scan.  Per-frame models and view slices are derived from its angles.
        frames_per_rotation (int, optional): Time frames per full 360 degree rotation.  This is
            also the period of the jitter that gating introduces, so it sets the period of the
            temporal dejitter filter.  Defaults to 6.
        frame_overlap_factor (float, optional): Number of frames that share any given view.  Each
            frame spans frame_overlap_factor * (360 / frames_per_rotation) degrees.  Defaults
            to 2.0.
        num_frames (int, optional): Reconstruct only the first N time frames, for smoke tests and
            partial runs.  Defaults to None, which uses every frame.  A value at or above the
            total frame count uses every frame.

    These are structural arguments, like ``angles`` in a geometry model: they fix the frame
    decomposition for the lifetime of the object and are not settable with
    :meth:`~mbirjax.ParameterHandler.set_params`.  Changing them means building a new model.
    The reconstruction parameters (``mace_prior_weight``, ``rho_mann``, ``prox_num_iterations``,
    ``prox_stop_threshold``, ``sigma_prox``, ``dejitter``, ``dejitter_verbose``, ``verbose``)
    are ordinary parameters set with :meth:`~mbirjax.ParameterHandler.set_params`.

    Attributes:
        model_list (list of mbirjax.TomographyModel): One model per time frame.
        view_slices (list of slice): The views of the full sinogram belonging to each frame.
        nt (int): Number of time frames.

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
        self.nt = len(self.model_list)
        # The reconstruction shape is the FRAME models' shape: those are the models that
        # produce the volumes.  It follows from the detector geometry, so it matches ct_model's
        # own recon shape unless copy_ct_model recomputed it differently for the shorter scan.
        self.recon_shape = tuple(self.model_list[0].get_params('recon_shape'))
        try:
            self.version = importlib_version('mbirjax')
        except Exception:
            self.version = 'unknown'

        # Reconstruction parameters. no_warning=True registers the names that are new here; the
        # ones the base class already knows (sigma_prox, verbose) keep their existing defaults.
        self.set_params(no_warning=True, mace_prior_weight=0.5, rho_mann=0.5,
                        prox_num_iterations=3, prox_stop_threshold=0.02, dejitter=True,
                        dejitter_verbose=0, sigma_prox=None)

        # Device layout: unset until configure_devices is called, resolved on first use.
        self._devices = None
        self._recon_token = 0

    @overload
    def get_params(self, parameter_names: Union[MACE4DParamNames, list[MACE4DParamNames]]) -> Any: ...

    def get_params(self, parameter_names) -> Any:
        return super().get_params(parameter_names)

    def set_params(self, no_warning=False, no_compile=False, **kwargs):
        """
        Update reconstruction parameters using keyword arguments.

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
            _normalize_prior_weights(kwargs['mace_prior_weight'])   # reject a bad weight here

        # sigma_prox is forwarded verbatim to each frame's prox_map; this model performs no
        # reconstruction of its own, so the base class's "you have disabled auto-regularization"
        # warning does not apply and is suppressed for that one name.
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

    def configure_devices(self, devices=None):
        """
        Configure which devices the reconstruction runs on.

        Every task -- one ``prox_map`` per time frame and one batched denoise per prior
        orientation -- is pinned to a single device, and one worker thread per device executes
        the tasks assigned to it.  A single device runs all tasks inline with no threads.

          * ``None`` -- automatic: all visible GPUs, or the CPU when there is no GPU.  Never
            calling this method is equivalent to ``configure_devices(None)``.
          * ``'cpu'`` / ``'gpu'`` -- all devices of that platform.
          * ``int n`` -- the first ``n`` devices of the default platform.  ``configure_devices(1)``
            forces the serial path.
          * ``sequence of ints`` -- those indices into the default device list.
          * ``sequence of jax devices`` -- exactly those devices.

        Args:
            devices (None, str, int, or sequence of ints / jax devices): see above.

        Raises:
            ValueError: If a platform string is unrecognized, a GPU is requested with no GPU
                backend, or the requested device count exceeds the number available.
        """
        self._devices = _resolve_devices(devices)

    @property
    def devices(self):
        """The devices this model runs on: those configured, else the automatic selection."""
        return self._devices if self._devices is not None else _resolve_devices(None)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def recon(self, sinogram, weights=None, init_recon=None, max_iterations=10,
              stop_threshold_change_pct=0.2, init_dir=None, log_dir=None):
        """Run 4D MACE reconstruction.

        Each iteration is a set of independent tasks -- one ``prox_map`` per time frame and one
        batched denoise per prior orientation -- executed by one worker thread per device with a
        fixed least-loaded task assignment.

        Args:
            sinogram (ndarray): Full sinogram, shape (num_views, num_det_rows, num_det_channels),
                matching the sinogram shape of the model this object was built from.  It is
                sliced into per-frame sinograms internally.
            weights (ndarray, optional): Positive weights with the same shape as ``sinogram``,
                sliced per frame internally.  Defaults to None, in which case the weights are
                implicitly all 1, as in :meth:`mbirjax.TomographyModel.recon`.  For 4D
                transmission data, ``mj.gen_weights(sinogram, weight_type='transmission_root')``
                is the validated choice and is strongly preferred over the unweighted default.
            init_recon (ndarray, optional): Initial 4D image, shape (nt, nx, ny, nz).  Defaults
                to None, in which case the initial image comes from ``init_dir`` if it holds one,
                and is otherwise computed by reconstructing each frame separately.
            max_iterations (int, optional): Maximum number of outer MACE iterations.  Defaults to 10.
            stop_threshold_change_pct (float, optional): Stop when the percent change of the
                consensus image from one iteration to the next falls below this value.  Defaults
                to 0.2.  Set to 0 to guarantee exactly ``max_iterations``.
            init_dir (str, optional): Cache directory for the computed initial image
                (``init_recon.npy``).  If it holds an image of the correct shape that image is
                used; otherwise the initialization is recomputed and saved there.  Defaults to None.
            log_dir (str, optional): Directory for ``run_info.txt``, ``timing_log.csv`` and
                ``task_log.csv``.  Created if needed.  Defaults to None, which writes no log files.

        Returns:
            (recon, recon_dict): the 4D reconstruction and a dict describing the run.
                - recon (ndarray): 4D reconstruction, shape (nt, nx, ny, nz).
                - recon_dict (dict): the run settings ('recon_params'), one entry per iteration
                  with its timing and consensus change ('timing'), plus 'notes' and 'model_params'.

        Raises:
            ValueError: If ``sinogram``, ``weights`` or ``init_recon`` has the wrong shape.

        Example:
            >>> weights = mj.gen_weights(sinogram, weight_type='transmission_root')
            >>> recon_4d, recon_dict = mace.recon(sinogram, weights=weights, log_dir='./logs')
        """
        nt = self.nt
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
            print(f"[MACE] Start 4D reconstruction with {nt} time frames.")

        # One single-thread executor per device: each device's tasks always run
        # on the same thread, which keeps the per-thread denoiser caches valid
        # and gives every model object exactly one owning thread.
        executors = ([concurrent.futures.ThreadPoolExecutor(max_workers=1) for _ in devs]
                     if len(devs) > 1 else None)
        # Denoisers reconfigure once per recon (sigma + regularization constants).
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
                    init_source = (f"computed ({self.nt} frames, "
                                   f"{_INIT_MBIR_ITERATIONS} MBIR iterations each)")

            # -- Global denoiser sigma (one value for all orientations) --------
            global_sigma = self._estimate_global_sigma(init_recon, devs[0])
            if not np.isfinite(global_sigma) or global_sigma <= 0:
                # Every denoiser is scaled by this sigma, and a zero divides through to the
                # qGGMRF forward-model constant.  Report the cause here rather than as a
                # division by zero several calls deeper.
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
            # Reused every iteration by the consensus update below, so the
            # temp-heavy expression form (sum() over freshly allocated full-size
            # arrays) never runs -- see the in-place rewrite there.
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
            # xbar is the consensus average sum(beta[k] X[k]); its relative
            # change per iteration is the convergence measure in timing_log.csv.
            xbar = init_recon
            for itr in range(max_iterations):
                itr_t0 = time.time()
                if verbose:
                    print(f"\n[MACE] -- Iteration {itr + 1}/{max_iterations} --")

                # Tasks only read W; W is not written until the consensus update
                # after the barrier, so no snapshot copy is needed.
                tasks = []
                for t in range(nt):
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

                # Gather in frame order, then dejitter the assembled stack.
                # X[0] keeps the dejittered stack -- it feeds the next prox calls.
                X[0] = self._dejitter(np.stack([results[("prox", t)] for t in range(nt)]))
                for k in range(3):
                    X[k + 1] = results[("denoise", k)]

                # ADMM consensus (CPU). In-place: the equivalent expression
                # form (z = sum(beta[k]*(2*X[k]-W[k]) ...), W[k] = W[k] + ...)
                # allocates ~28 fresh full-size arrays per iteration and measured
                # 7.1x slower on the full-resolution volume. Same math, same order
                # of operations, verified to produce identical results.
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

        # run_info.txt is written before the loop so a long run's settings are readable while
        # it is still going; rewrite it now that the iteration count is known.
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
        """Fix the task-to-device assignment, pin models, and place per-frame data.

        The assignment is computed once and reused for every iteration: each model object gets
        one owning thread, and each frame's sinogram and weights are uploaded to its device once
        and stay there.  Frames are slices of the full arrays, so nothing is copied on the host.
        """
        plane_counts = [self.recon_shape[2], self.recon_shape[0], self.recon_shape[1]]  # XY-t, YZ-t, XZ-t
        self._frame_device, self._orient_device = _assign_tasks(self.nt, plane_counts, len(devs))
        for t in range(self.nt):
            self.model_list[t].configure_devices([devs[self._frame_device[t]]])
        self._sino_dev = [jax.device_put(np.asarray(sinogram[self.view_slices[t]]),
                                         devs[self._frame_device[t]]) for t in range(self.nt)]
        if weights is None:
            self._weights_dev = [None] * self.nt
        else:
            self._weights_dev = [jax.device_put(np.asarray(weights[self.view_slices[t]]),
                                                devs[self._frame_device[t]]) for t in range(self.nt)]

    def _run_task_set(self, executors, tasks, t0):
        """Run tasks [(device_index, tag, fn)] and wait for all of them.

        Inline when executors is None (one device). Returns ({tag: result},
        [(kind, index, device_index, start, end)]) with times relative to t0.
        A failed task raises immediately, naming the task.
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
        """One frame's proximal map on its assigned device."""
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
        """One orientation's batched qGGMRF denoise on its assigned device."""
        return _denoiser_wrapper(self._dejitter(W_k), permute_vector=permute_vector,
                                 sigma=sigma, device=device,
                                 config_token=self._recon_token)

    def _init_frame_task(self, t, device):
        """One frame's MBIR initialization recon on its assigned device."""
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
        """One global noise sigma for all denoising, estimated from the initial image."""
        # Merge (nt, nx) so the estimator sees a 3D array; it subsamples internally.
        image_3d = init_recon.reshape(-1, init_recon.shape[2], init_recon.shape[3])
        denoiser = mj.QGGMRFDenoiser(image_3d.shape)
        denoiser.configure_devices([device])
        return float(denoiser.estimate_image_noise_std(image_3d))

    def _dejitter(self, x):
        """Apply the DCT-I temporal dejitter if enabled; otherwise return x unchanged."""
        if not self.get_params('dejitter'):
            return x
        return _dejitter_4d_dct(x, period=self.frames_per_rotation, harmonics=True,
                                band_width=1, dtype=np.float32,
                                verbose=bool(self.get_params('verbose')))

    def _run_settings(self, devs, init_source, global_sigma, weights, max_iterations,
                      stop_threshold_change_pct):
        """The settings that describe this run, for run_info.txt and the returned recon dict."""
        if len(devs) > 1:
            mode = f"task queue over {len(devs)} devices: " + ", ".join(str(d) for d in devs)
        else:
            mode = f"serial on {devs[0]}"
        beta = _normalize_prior_weights(self.get_params('mace_prior_weight'))
        sigma_prox = self.get_params('sigma_prox')
        return {
            'date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'mbirjax version': self.version,
            'time frames (nt)': self.nt,
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
        """Return the array unchanged, or raise ValueError if it is not sinogram-shaped."""
        sinogram = np.asarray(sinogram)
        if sinogram.shape != self.sinogram_shape:
            raise ValueError(f"{name} shape {sinogram.shape} does not match the model's "
                             f"sinogram shape {self.sinogram_shape}.")
        return sinogram

    def _expected_init_shape(self):
        """Shape the initial image must have: (nt,) + per-frame recon shape."""
        return (self.nt,) + self.recon_shape

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
        """Load init_recon.npy from init_dir if present and valid; else return None.

        A missing file is normal (first run) and silent. A file that cannot be
        loaded or has the wrong shape produces a warning.
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
        """Per-frame MBIR recon used as the MACE initial image.

        Uses the same workers and frame-to-device assignment as the MACE loop,
        so compiled programs and resident data carry over.
        """
        verbose = self.get_params('verbose')
        if verbose:
            print(f"[MACE] Computing initial MBIR recon on {len(devs)} device(s)...")
        t0 = time.time()
        tasks = [(self._frame_device[t], ("init", t),
                  lambda tt=t: self._init_frame_task(tt, devs[self._frame_device[tt]]))
                 for t in range(self.nt)]
        results, _ = self._run_task_set(executors, tasks, t0)
        init_recon = np.stack([results[("init", t)] for t in range(self.nt)])
        if init_dir is not None:
            os.makedirs(init_dir, exist_ok=True)
            np.save(os.path.join(init_dir, "init_recon.npy"), init_recon)
        if verbose:
            print(f"[MACE] Initialization done in {time.time() - t0:.2f} sec.")
        return init_recon


# Thread-local denoiser cache: key = (shape, device), value = QGGMRFDenoiser.
# Ensures no denoiser instance is shared across threads (critical for multi-GPU).
_THREAD_LOCAL = threading.local()


# ---------------------------------------------------------------------------
# Device selection and task assignment
# ---------------------------------------------------------------------------

def _resolve_devices(devices):
    """Return the list of jax devices to use; see MACE4DModel.configure_devices."""
    if devices is None:
        # Automatic: every GPU, or a single CPU device when there is no GPU.  (Unlike a sharded
        # recon, this model runs one independent task per device, so spreading over the virtual
        # CPU devices of one machine would oversubscribe the same cores.)
        return list(gpu_devices()) or [cpu_devices()[0]]
    if isinstance(devices, str):
        platform = devices.lower()
        if platform == 'cpu':
            return list(cpu_devices())
        if platform == 'gpu':
            pool = list(gpu_devices())
            if not pool:
                raise ValueError("configure_devices('gpu') was requested but no GPU backend "
                                 "is available.")
            return pool
        raise ValueError("configure_devices platform string must be 'cpu' or 'gpu'; "
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
    """Fixed least-loaded-first assignment of tasks to devices.

    The denoise tasks (estimated cost proportional to their hyperplane count)
    are placed first, largest first; then each unit-cost prox task goes to the
    least-loaded device.

    Returns:
        tuple: (frame_device, orient_device) where frame_device is a list of int
            device indices for each frame's prox task, and orient_device is a list
            of int device indices for each orientation's denoise.
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
    Convert a scalar or list prior weight into [forward_w, xyt_w, yzt_w, xzt_w].

    Scalar w -> [1-w, w/3, w/3, w/3].
    List/tuple [w1, w2, w3] -> [1-(w1+w2+w3), w1, w2, w3].
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
# IMPORTANT: each QGGMRFDenoiser must be pinned to exactly ONE GPU via
# configure_devices([device]). Without this, mbirjax auto-shards the denoiser
# across every visible GPU using a NamedSharding Mesh. Running 4 such denoisers
# concurrently then causes each thread's model to open its own 4-way NCCL
# clique simultaneously -- producing an "Acquire clique ... may be stuck" deadlock.
# Cache key includes the device so each thread gets its own pinned instance.

def _get_qggmrf_denoiser(shape, device):
    """Return a per-thread, per-device cached QGGMRFDenoiser pinned to one GPU."""
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


# Denoiser iteration settings (match the per-volume denoise() defaults).
_DENOISE_MAX_ITERATIONS = 15
_DENOISE_STOP_THRESHOLD_PCT = 0.2

# Working-set multiplier for the batch-size estimate: bytes used per volume
# during the jitted sweep, as a multiple of the volume size. Heuristic; refine
# by measurement on the target GPU.
_DENOISE_BUFFER_MULTIPLIER = 16

# Absolute cap on the denoise batch size, so a bad memory estimate cannot
# request an enormous compile.
_DENOISE_BATCH_CAP = 128

# Floor for the auto-estimated qGGMRF regularization scale (sigma_x). Guards
# against a batch whose statistics happen to come out at or near zero (e.g. a
# batch dominated by background-only hyperplanes), which would otherwise make
# the qGGMRF solver produce NaN for the whole batch.
_SIGMA_X_FLOOR = 1e-6


def _configure_denoiser(denoiser, sigma, image_for_stats):
    """Set the shared sigma and the regularization constants on the denoiser.

    Replicates the parameter setup that QGGMRFDenoiser.denoise() performs, so
    the jitted sweep can be called directly with shared constants.

    QGGMRFDenoiser.auto_set_regularization_params() (inherited from
    TomographyModel) calls subsample_views with num_real_views=sinogram_shape[0],
    which equals nt for a hyperplane batch -- not the batch size -- so it would
    use only the first hyperplane's statistics. The individual auto-set methods
    are called directly on the full array instead.
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
    # The sweep's progress callback converts its arguments with int()/float(),
    # which fails on the batched arrays a vmapped sweep passes it; silence it.
    denoiser._log_denoise_progress = lambda *args: None
    # Recompute the sweep constants for this configuration. The pixel partition
    # is random per generation, so it must be built once here and reused --
    # otherwise repeated calls run different VCD subset orders.
    denoiser._mace4d_constants = None
    _denoise_constants(denoiser)
    # New constants invalidate the cached batch size and compiled batch function.
    denoiser._mace4d_batch = None
    denoiser._mace4d_batched_fn = None


def _denoise_constants(denoiser):
    """The constant arguments of the denoiser's jitted sweep, cached per configuration."""
    cached = getattr(denoiser, '_mace4d_constants', None)
    if cached is not None:
        return cached
    image_shape, granularity = denoiser.get_params(['recon_shape', 'granularity'])
    # Keep at least ~64 pixels per VCD subset: with very small subsets the
    # qGGMRF line search can hit 0/0 in flat regions. At real volume sizes
    # this leaves the subset count unchanged.
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
    """Largest volume batch that fits in device memory; a small fixed batch on CPU."""
    stats = getattr(device, 'memory_stats', lambda: None)()
    if not stats:
        return 4
    free = stats.get('bytes_limit', 0) - stats.get('bytes_in_use', 0)
    vol_bytes = 4 * int(np.prod(vol_shape))
    return max(1, int(0.5 * free) // (_DENOISE_BUFFER_MULTIPLIER * vol_bytes))


def _batched_hyperplane_denoise(x, denoiser, device):
    """Denoise a stack of same-shaped 3D volumes with shared, preconfigured settings.

    One jax.vmap call runs the denoiser's single-device jitted sweep over a
    whole batch, so the GPU is filled instead of processing volumes one at a time.
    The volumes are independent, so the result equals per-volume denoising with
    the same constants.

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

    # One fixed batch size and one compiled batch function per configuration.
    # The last block is padded to the fixed size so every call reuses the same
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
                # Out of device memory: halve the batch and recompile once.
                if "RESOURCE_EXHAUSTED" in str(err) and denoiser._mace4d_batch > 1:
                    denoiser._mace4d_batch = max(1, denoiser._mace4d_batch // 2)
                    denoiser._mace4d_batched_fn = jax.jit(jax.vmap(denoise_one))
                    continue
                raise
            y[b0:b1] = out[: b1 - b0].reshape((b1 - b0,) + vol_shape)
            b0 = b1
    return y


def _denoiser_wrapper(x, permute_vector, sigma, device, config_token=None):
    """Permute a 4D volume so the hyperplane axis is first, batch-denoise the
    resulting stack of 3D volumes at the shared global sigma, then unpermute.

    Args:
        x (ndarray): 4D volume, shape (nt, nx, ny, nz).
        permute_vector (tuple of int): Permutation that puts the hyperplane axis first.
        sigma (float): Global noise sigma shared by every volume.
        device (jax.Device): Device on which denoising runs.
        config_token (hashable or None): Configure the denoiser (sigma, regularization
            constants, partition) only when this token changes -- once per recon. None
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
