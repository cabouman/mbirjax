import functools
import io
import math
import types
import warnings
import inspect
import os
from collections import namedtuple
import traceback
from typing import Literal, Union, overload
import datetime
import time  # Used for debugging/performance tuning

from ruamel.yaml import YAML
import numpy as np
from jax.errors import JaxRuntimeError

# Virtual CPU device setup now lives in mbirjax/_device_setup.py, which runs as
# the first import in mbirjax/__init__.py (before JAX initializes its backends).
# These lines are kept commented as a historical pointer; do not re-enable.
# num_cpus = 3 * os.cpu_count() // 4
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count={}'.format(num_cpus)
import jax
import jax.numpy as jnp

import mbirjax as mj
from mbirjax import ParameterHandler
from mbirjax._utils import is_oom, log_oom_guidance
from mbirjax._device_setup import gpu_devices, cpu_devices, default_devices, get_device_platform
# Internal sharding primitives (see _sharding), accessed with the `mjs` prefix.
# Importing the SUBMODULE directly (not aliasing the top-level `mbirjax` and
# reaching submodules as attributes) is safe even mid-import of mbirjax: it
# forces `mbirjax._sharding` to load, and that subpackage pulls in only
# jax/numpy/warnings, nothing that loops back into the partially-initialized
# mbirjax package.
import mbirjax._sharding as mjs
import mbirjax.tomography_utils as tomography_utils

from importlib.metadata import version, PackageNotFoundError

# The projector TILING POLICY: every batching/banding knob and kernel-algorithm flag the
# projectors consume, selected in ONE place (TomographyModel._select_tile_policy, re-run on
# every device re-layout) and stored on the model as ``self.tiles``.  Immutable by design:
# experiments override a field with ``model.tiles = model.tiles._replace(...)`` rather than
# mutating scattered attributes.  Fields:
#   fwd_view_batch / back_view_batch  -- per-op vmap widths over views.
#   fwd_pixel_batch / back_pixel_batch -- per-op pixel tile sizes (the forward driver's pixel
#       scan; back's pixel concatenation; cone's gather-forward host tiling).
#   fwd_slice_band / back_slice_band  -- slice-band length overrides for the banded projector
#       paths; None = the memory-driven _slice_band_length formula.
#   sort_by_channel -- kernel-algorithm flag: use the sorted segment-sum channel reduction in
#       the forward kernel (GPU layouts with wide-enough bands; see
#       projectors.channel_scatter_reduce and parallel_beam's policy override).
#   back_stacked_gather -- kernel-algorithm flag: the back kernel gathers all psf taps in one
#       stacked (psf_width * num_pixels, num_rows) gather + reshape-sum over taps (GPU: ~95%
#       gather-bound, measured 1.6-1.8x; CPU: worse, keeps the per-tap loop).  See
#       parallel_beam.back_project_one_view_to_pixel_batch.
TilePolicy = namedtuple('TilePolicy', ['fwd_view_batch', 'back_view_batch',
                                       'fwd_pixel_batch', 'back_pixel_batch',
                                       'fwd_slice_band', 'back_slice_band',
                                       'sort_by_channel', 'back_stacked_gather'])

# Persistent jit-compilation cache: repeat runs of the same model shapes load
# compiled executables from disk instead of recompiling (a real win for
# production-size recons, whose XLA compiles take seconds).  The cache lives in
# the per-user mbirjax home (NOT shared /tmp, which risks cross-user permission
# collisions on multi-user machines and is cleared on reboot), and is set ONLY
# if the user has not already configured a cache of their own (via
# JAX_COMPILATION_CACHE_DIR or jax.config) -- never override an explicit choice.
if jax.config.jax_compilation_cache_dir is None:
    jax.config.update("jax_compilation_cache_dir",
                      os.path.expanduser(os.path.join('~', '.mbirjax', 'jax_cache')))
    # Persist only the compiled executables, NOT XLA's GPU per-fusion autotune cache.  jax defaults
    # `jax_persistent_cache_enable_xla_caches` to 'xla_gpu_per_fusion_autotune_cache_dir', which
    # writes temp files under <cache>/xla_gpu_per_fusion_autotune_cache_dir/tmp and renames them --
    # that fails with NOT_FOUND on cluster filesystems (NFS) and on a fresh cache dir.  Disabling it
    # leaves the executable cache (the real recompile-avoidance win) untouched.  Only set here, where
    # WE configured the cache -- never override a user's explicit cache configuration.
    if hasattr(jax.config, "jax_persistent_cache_enable_xla_caches"):
        jax.config.update("jax_persistent_cache_enable_xla_caches", "none")
# Set the GPU memory fraction for JAX -- the share of each GPU that XLA's BFC pool may reserve.
# 0.94 rather than 0.98: everything that allocates OUTSIDE the pool (cuSolver/cuDNN workspaces,
# NCCL/collective vmm buffers for cross-device reductions on sharded arrays) has to fit in the
# remainder, and the pool RETAINS its high-water mark for the life of the process -- at 0.98 those
# out-of-pool allocations starved (CUDA_ERROR_OUT_OF_MEMORY from the vmm allocator, cuSolver handle
# failures) even with the pool mostly idle.  See .claude/lessons.md "Out-of-pool GPU allocations".
# setdefault (not a hard set) so users can tune per-run via the environment before importing mbirjax.
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.94')

recon_param_names = ['num_iterations', 'granularity', 'partition_sequence', 'fm_rmse', 'prior_loss',
                     'regularization_params', 'stop_threshold_change_pct', 'alpha_values']
ReconParams = namedtuple('ReconParams', recon_param_names)

TomographyParamNames = mj.ParamNames | Literal['view_params_name']


class TomographyModel(ParameterHandler):
    """
    Represents a general model for tomographic reconstruction using MBIRJAX. This class encapsulates the parameters and
    methods for the forward and back projection processes required in tomographic imaging.

    Note that this class is a template for specific subclasses.  TomographyModel by itself does not implement
    projectors or recon.  Use self.print_params() to print the parameters of the model after initialization.

    Args:
        sinogram_shape (tuple): The shape of the sinogram array expected (num_views, num_det_rows, num_det_channels).
        recon_shape (tuple): The shape of the reconstruction array (num_rows, num_cols, num_slices).
        **kwargs (dict): Arbitrary keyword arguments for setting model parameters dynamically.
            See the full list of parameters and their descriptions at :ref:`detailed-parameter-docs`.

    Sets up the reconstruction size and parameters.
    """


    def __init__(self, sinogram_shape, **kwargs):

        super().__init__()
        self.set_params(no_compile=True, no_warning=True, sinogram_shape=sinogram_shape, **kwargs)

        self.auto_set_recon_geometry(no_compile=True, no_warning=True)

        self.set_params(geometry_type=str(type(self)))

        self.projector_functions = None
        self.prox_data = None

        # Device layout.  recon_placement / sino_placement are the single source of truth for how
        # recon-like (slice-sharded) and sino-like (view-sharded) arrays are distributed across
        # devices.  Each is a Placement owning a device list, a sharded axis, AND its own 1-D mesh.
        # set_devices() builds them at the end of __init__ (a trivial 1-device layout when there is
        # one device), so the placement path is always on (recon_placement is never None after
        # construction).  configure_devices() pins an explicit device list (_sharding_configured =
        # True); absent that, set_devices() auto-selects.  mesh / shard_devices are derived,
        # read-only views of recon_placement (see the properties below); arrays that must
        # be created on a single device use recon_placement.devices[0] / sino_placement.devices[0].
        self.recon_placement = None
        self.sino_placement = None
        self.dev2dev_safe = True   # set empirically in _set_device_layout
        self._sharding_configured = False  # True only after an explicit configure_devices()
        # AUTOMATIC selection shards across CPU devices too, the same path as multi-GPU (the
        # platform-UNIFORM auto policy -- a platform-dependent policy is how "sharded + X" gaps
        # stayed invisible to the CPU suite).  mbirjax._device_setup exposes a conservative number
        # of virtual CPU devices on CPU-only hosts.  Measured (M3 Max, 2 devices, 8-iter VCD): 1.30x
        # at 256^3 and better at larger sizes, vs 0.83x at 128^3 and 0.64x at 64^3 -- a real win
        # where time matters and a few absolute seconds where it does not (small recons also use
        # LESS peak RSS sharded).  To force single-device, pin it with configure_devices(1).
        # CPU-CLUSTER tuning (real multi-socket hosts) remains an open adjacent task (see the plan).
        # Cached per-device qGGMRF interface masks for a padded slice axis (built lazily
        # by _qggmrf_interface_masks; invalidated by _set_device_layout on every recompile).
        self._qggmrf_interface_masks_cache = None

        # The projector TILING POLICY (batch sizes, band lengths, kernel-algorithm flags) is
        # selected in ONE place -- _select_tile_policy, called from _set_device_layout on every
        # re-layout -- and stored here as an immutable TilePolicy namedtuple.  The projector
        # wrappers read self.tiles at CALL time (late binding), so a configure_devices()
        # re-layout takes effect on the next projection call, and an experiment can override a
        # field with ``model.tiles = model.tiles._replace(...)``.  Geometry classes override
        # _select_tile_policy for measured, geometry-specific choices.
        self.tiles = None                       # set by set_devices() -> _set_device_layout below
        self.set_devices()
        self.create_projectors()
        try:
            __version__ = version("mbirjax")
        except PackageNotFoundError:
            # package is not installed
            __version__ = "unknown"
        self.version = __version__

    @classmethod
    def get_required_param_names(cls):
        """
        Return a list with the names of the required parameters of cls.__init__.

        Args:
            cls : type
                The class whose __init__ we want to inspect.

        Returns:
            list[str]
                A list of parameter names, in the order they appear in the signature.
        """
        sig = inspect.signature(cls.__init__)
        params = sig.parameters.values()

        # Filter out *args and **kwargs to get simple names
        names = [
            p.name
            for p in params
            if p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                              inspect.Parameter.VAR_KEYWORD)
        ]

        if names[0] == "self":
            names = names[1:]

        return names

    @overload
    def get_params(self, parameter_names: Union[TomographyParamNames, list[TomographyParamNames]]): ...

    def get_params(self, parameter_names):
        return super().get_params(parameter_names)

    @property
    def shard_devices(self):
        """The devices the recon and sinogram are sharded over (a list), or ``None`` before the
        device layout is configured.

        A property -- read it as an attribute (``model.shard_devices``), not a call.

        This is the public, read-only view of the device layout: ``len(model.shard_devices)`` is
        the device count and ``model.shard_devices[0].platform`` its platform.  Derived from
        ``recon_placement`` (the recon and sino placements share the same devices).
        """
        return self.recon_placement.devices if self.recon_placement is not None else None

    def configure_devices(self, devices=None):
        """
        Configure which devices the reconstruction runs on (the user-facing control surface).

        Resolves ``devices`` to a concrete device list and PINS it: a later ``set_params`` (which
        re-runs ``set_devices``) keeps these devices rather than re-auto-selecting.

          * ``None``  -- automatic: the same choice ``use_gpu='automatic'`` makes by default --
            shard across all available GPUs (or, with no GPU, all available CPU devices), trimmed
            by ``_auto_device_count`` so the last shard is never entirely padding.  A single
            available device gives a trivial 1-device layout.
          * ``int n`` -- use the first ``n`` devices of the default platform (GPUs if any, else CPUs);
            ``configure_devices(1)`` is the way to force single-device operation.
          * ``sequence of ints`` -- use those indices into the default device list (``jax.devices()``).
          * ``sequence of jax devices`` -- use exactly those devices.

        Sharding scheme (uniform across geometries): sinogram-like arrays are sharded by **view**
        (axis 0), recon-like arrays by **slice**.  NEITHER axis constrains the device count: a
        non-dividing view or slice axis is zero-padded to the next multiple of the count and the
        padding is kept exactly inert (it cannot affect the results), so the result is independent
        of the device count.  The pad metadata is re-derived from the current shapes on every
        recompile, so device selection is order-independent with respect to shape changes.

        Args:
            devices (None, int, or sequence of ints / jax devices): see above.

        Returns:
            Nothing; builds recon_placement / sino_placement and the empirical device-to-device
            safety flag (see ``_set_device_layout``).
        """
        pool = self._auto_device_pool() if devices is None else self._resolve_devices(devices)
        self._set_device_layout(pool, pinned=True)

    def _resolve_devices(self, devices):
        """Resolve a non-None configure_devices() ``devices`` argument to a concrete device list.

        (``None`` is handled by configure_devices via ``_auto_device_pool``.)
        """
        if isinstance(devices, (int, np.integer)):
            return default_devices()[:int(devices)]
        devices = list(devices)
        if devices and all(isinstance(d, (int, np.integer)) for d in devices):
            all_devices = default_devices()
            return [all_devices[int(i)] for i in devices]
        return devices

    def _auto_device_pool(self):
        """The device list for AUTOMATIC selection (no explicit pin).

        Shared by ``set_devices`` (the construction-time default) and ``configure_devices(None)``
        so "automatic" has a single definition.  Shards across all GPUs when a GPU backend is
        present and ``use_gpu`` is not ``'none'``, otherwise across all (possibly virtual) CPU
        devices -- the platform-uniform auto policy.  ``_auto_device_count`` trims a count whose
        last shard would be entirely padding, and returns 1 when only one device is available (a
        trivial 1-device layout).  GPU detection is by backend (``gpu_devices()``), not a device's
        ``.platform`` string, so it is robust to ``'cuda'``/``'rocm'`` platform names.
        """
        gpus = gpu_devices()   # () when there is no GPU backend
        on_gpu = bool(gpus) and self.get_params('use_gpu') != 'none'
        pool = list(gpus) if on_gpu else list(cpu_devices())
        return pool[:self._auto_device_count(len(pool))]

    def _auto_device_count(self, n_available):
        """Device count for automatic selection: ALL available devices, except any count
        whose last shard would be entirely padding.

        Automatic device selection (use_gpu='automatic'/'full' on a multi-GPU box) shards across
        this many devices.  NEITHER sharded axis constrains the count anymore: a non-dividing
        view axis or slice axis is zero-padded to the next multiple of the
        device count and the padding is kept exactly inert (entry zero-fill + the projector
        output masks + the qGGMRF interface mask), so the result is independent of N.  The one
        guard: skip a count whose LAST shard would hold zero real slices (e.g. 5 slices on 4
        devices -> shards of 2 with the last entirely padding) -- correct but a wasted device;
        fall to the next smaller count.  (The broader choose-N-vs-communication policy is a later
        discussion.)  There is NO per-device size floor (decided 2026-06-09 from the GPU
        band/scale sweeps): sharding's value is capacity + near-linear speedup at the sizes
        that matter, and over-sharding a small problem is only a mild overhead.

        Returns 1 for n_available <= 1.
        """
        if n_available <= 1:
            return 1
        recon_shape = self.get_params('recon_shape')
        num_recon = int(recon_shape[self.recon_shard_axis() % len(recon_shape)])
        for n in range(n_available, 0, -1):
            # Shard size at this count; the last shard is fully padded iff the real
            # slices don't reach into it: num_recon <= (n-1) * shard_size.
            shard_size = -(-num_recon // n)   # ceil(num_recon / n)
            if num_recon > (n - 1) * shard_size:
                return n
        return 1

    @staticmethod
    def _platform_label(device):
        """Short uppercase platform name for a jax device ('GPU' / 'CPU' / 'TPU').

        Thin delegator to :func:`mbirjax.get_device_platform` (the single source of truth);
        kept for the existing internal/test call sites.
        """
        return get_device_platform(device)

    def _device_report(self):
        """A 'N x PLATFORM (sharded)' summary of the recon devices, for the recon log.

        ``(sharded)`` always appears (every recon runs the placement path), regardless of device
        count.  When automatic selection left GPUs idle because the device count cannot divide both
        sharded axes, the reason is appended so idle hardware is never silent.
        """
        devices = self.shard_devices
        n = len(devices)
        platform = self._platform_label(devices[0])
        report = '{} x {} (sharded)'.format(n, platform)
        # Padding is invisible in the results (exactly inert), so say so in the log rather
        # than leaving the device-form shapes a surprise.
        if self.sino_placement is not None and self.sino_placement.is_padded:
            report += ' (views padded {}->{})'.format(
                self.sino_placement.real_size, self.sino_placement.padded_size)
        if self.recon_placement is not None and self.recon_placement.is_padded:
            report += ' (slices padded {}->{})'.format(
                self.recon_placement.real_size, self.recon_placement.padded_size)
        # Automatic selection that left devices idle (a count whose last shard would be
        # entirely padding is skipped): explain why hardware is unused.
        if not self._sharding_configured and platform == 'GPU':
            n_available = len(gpu_devices()) or n   # () -> 0 only off-GPU, which can't reach here
            if n_available > n:
                recon_shape = self.get_params('recon_shape')
                num_slices = recon_shape[self.recon_shard_axis() % len(recon_shape)]
                report += (' (using {} of {} GPUs: with num_slices={}, more devices would '
                           'leave some entirely idle (a fully padded shard))'.format(
                               n, n_available, num_slices))
        return report

    @property
    def device_summary(self):
        """str: A read-only summary of the devices the reconstruction will actually use.

        A property -- read it as an attribute (``model.device_summary``), not a call.

        This is the resolved OUTCOME of the device configuration -- the ``use_gpu``
        parameter and ``configure_devices`` are the request; this reports what was
        chosen.  E.g. ``'4 x GPU (sharded)'``, with notes appended when the view
        axis is padded or when automatic selection left GPUs idle.  The same line
        is logged at the start of every reconstruction.
        """
        return self._device_report()

    def _set_device_layout(self, devices, pinned):
        """Set the device layout from a concrete device list -- the single place that does so.

        Builds recon_placement / sino_placement (each owns a 1-D mesh over ``devices``: recon on
        the slice axis, sino on the view axis; a single device is the trivial 1-shard case) and
        sets the empirical device-to-device safety flag.  These placements ARE the device layout;
        mesh / shard_devices are derived views of them.

        Each placement receives the problem-owned (REAL) length of its sharded axis from the
        params, so it knows the device-form (padded) length when that size does not divide the
        device count.  The params always keep the real shapes ("what is the problem?"); the
        placements own the padded device shapes ("what is on the devices?").  This runs on every
        recompile (via set_devices), so the pad metadata tracks shape changes.  Both sharded axes
        pad: views and recon slices (for parallel beam the detector rows pad with the slices --
        see :meth:`_sino_row_padding`).  No divisibility constraint: a non-dividing axis is
        zero-padded (inert), so the result is independent of the device count.

        Args:
            devices (sequence of jax devices): the devices to lay arrays out over (length 1 =
                trivial single-device layout).
            pinned (bool): if True, mark the configuration as user-selected (configure_devices) so
                set_devices will not re-auto-select it; if False, leave it overridable so a later
                set_params (which re-runs set_devices) can re-evaluate the layout.
        """
        devices = list(devices)
        if len(devices) < 1:
            raise ValueError("_set_device_layout requires at least one device.")
        self.dev2dev_safe = mjs.is_dev2dev_safe(devices)
        self._sharding_configured = pinned
        # Layout changed: drop any cached per-device qGGMRF interface masks (they
        # encode the previous padded shard ranges; rebuilt lazily on first use).
        self._qggmrf_interface_masks_cache = None
        recon_axis = self.recon_shard_axis()
        sino_axis = self.sinogram_shard_axis()
        sinogram_shape = self.get_params('sinogram_shape')
        recon_shape = self.get_params('recon_shape')
        num_views = sinogram_shape[sino_axis % len(sinogram_shape)]
        num_slices = recon_shape[recon_axis % len(recon_shape)]
        self.recon_placement = mjs.Placement(devices, axis=recon_axis, real_size=num_slices)
        self.sino_placement = mjs.Placement(devices, axis=sino_axis, real_size=num_views)
        on_gpu = devices[0].platform == 'gpu'
        self.tiles = self._select_tile_policy(on_gpu, num_views, num_slices, len(devices))

    # Base tiling constants (geometry-independent defaults; geometry classes change measured
    # values in their _select_tile_policy overrides, not by shadowing these).
    _FWD_VIEW_CAP = 128                # forward vmap width that stays OOM-safe at the largest sizes
    _BACK_VIEW_CAP_SINGLE = 128        # unsharded back: wider vmaps raise peak memory ~25%
    _BACK_VIEW_CAP_SHARDED = 512       # safe for a per-device view shard
    _PIXEL_BATCH_DEFAULT = 2048        # generic pixel tile (both ops); the long-standing default

    def _select_tile_policy(self, on_gpu, num_views, num_slices, n_devices):
        """Select the projector TILING POLICY for this device layout -- the ONE decision site.

        Returns a TilePolicy (see its definition above) covering every batching/banding knob and
        kernel-algorithm flag the projectors consume.  Geometry classes override this method to
        change ONLY what they have measured (e.g. ParallelBeamModel's GPU forward tiling); the
        base policy preserves the long-standing defaults:

          * VIEW batches -- forward and back want OPPOSITE policies (measured).  Forward's
            vmap transient scales with the batch width PER DEVICE (view-sharding does not
            shrink it), so it keeps a flat OOM-safe width clipped to the shard.  Back
            single-vmaps its whole per-device view shard (a smaller batch drops into the
            accumulating-SCAN path, whose live carry inflates the peak), capped per the
            constants above.
          * PIXEL batches -- the generic default (_PIXEL_BATCH_DEFAULT) for both ops.
          * SLICE bands -- None = the memory-driven ``_slice_band_length`` formula.  A number
            here overrides it (clipped to the slice shard); the per-instance
            ``fwd/back_project_slice_band`` attributes remain the top-priority experiment hook.
          * ``sort_by_channel`` -- False: the scatter-add channel reduction (see
            projectors.channel_scatter_reduce).

        BINDING: consumers read ``self.tiles`` at CALL time (late binding), so a
        ``configure_devices()`` re-layout takes effect on the next projection without recreating
        the projectors.  Exception: ``sort_by_channel`` is baked into the STATIC ProjectorParams
        at create_projectors -- every sanctioned platform change recreates the projectors via
        set_params, and a stale flag can only cost speed, never correctness (the reductions are
        value-equal).  Values that reach jit are STATIC arguments: a changed value retraces,
        never silently misbatches.
        """
        per_shard_views = -(-num_views // max(1, n_devices))   # ceil: views per device
        back_cap = self._BACK_VIEW_CAP_SINGLE if n_devices <= 1 else self._BACK_VIEW_CAP_SHARDED
        return TilePolicy(
            fwd_view_batch=max(1, min(per_shard_views, self._FWD_VIEW_CAP)),
            back_view_batch=max(1, min(per_shard_views, back_cap)),
            fwd_pixel_batch=self._PIXEL_BATCH_DEFAULT,
            back_pixel_batch=self._PIXEL_BATCH_DEFAULT,
            fwd_slice_band=None,
            back_slice_band=None,
            sort_by_channel=False,
            back_stacked_gather=False,
        )

    @property
    def view_batch_size_for_vmap(self):
        raise AttributeError(
            "view_batch_size_for_vmap no longer exists: the batching knobs are consolidated in "
            "the TilePolicy at model.tiles (see _select_tile_policy).  Read model.tiles.fwd_"
            "view_batch / .back_view_batch; override with model.tiles = model.tiles._replace(...).")

    @view_batch_size_for_vmap.setter
    def view_batch_size_for_vmap(self, value):
        raise AttributeError(
            "view_batch_size_for_vmap no longer exists; setting it would silently do nothing.  "
            "Use model.tiles = model.tiles._replace(fwd_view_batch=..., back_view_batch=...).")

    @property
    def fwd_view_batch_size_for_vmap(self):
        raise AttributeError("moved into TilePolicy: read model.tiles.fwd_view_batch")

    @fwd_view_batch_size_for_vmap.setter
    def fwd_view_batch_size_for_vmap(self, value):
        raise AttributeError(
            "moved into TilePolicy: model.tiles = model.tiles._replace(fwd_view_batch=...)")

    @property
    def back_view_batch_size_for_vmap(self):
        raise AttributeError("moved into TilePolicy: read model.tiles.back_view_batch")

    @back_view_batch_size_for_vmap.setter
    def back_view_batch_size_for_vmap(self, value):
        raise AttributeError(
            "moved into TilePolicy: model.tiles = model.tiles._replace(back_view_batch=...)")

    @property
    def pixel_batch_size_for_vmap(self):
        raise AttributeError("moved into TilePolicy: read model.tiles.fwd_pixel_batch / "
                             ".back_pixel_batch")

    @pixel_batch_size_for_vmap.setter
    def pixel_batch_size_for_vmap(self, value):
        raise AttributeError(
            "moved into TilePolicy: model.tiles = model.tiles._replace(fwd_pixel_batch=..., "
            "back_pixel_batch=...)")

    @property
    def transfer_pixel_batch_size(self):
        raise AttributeError(
            "transfer_pixel_batch_size is now derived at its use site as "
            "100 * model.tiles.back_pixel_batch")

    @transfer_pixel_batch_size.setter
    def transfer_pixel_batch_size(self, value):
        raise AttributeError(
            "transfer_pixel_batch_size is now derived at its use site as "
            "100 * model.tiles.back_pixel_batch")

    # ------------------------------------------------------------------
    # Sharding hooks (uniform default scheme; override per geometry only
    # if a geometry needs a different axis or halo strategy)
    # ------------------------------------------------------------------
    # The uniform scheme shards sinogram-like objects by view and recon-like
    # objects by slice.  The two axis-declaration hooks below are the single
    # source of truth for *which* axis is sharded; the placement build in
    # _set_device_layout and the _shard_*/_gather_* helpers all derive from
    # them.  Keeping the choice here (rather than hardcoded throughout) means a
    # geometry can declare a different axis by overriding one small method,
    # without hunting down scattered assumptions.

    @classmethod
    def sinogram_shard_axis(cls):
        """Axis of a sinogram-like array (views, det_rows, channels) to shard.

        Default 0 (views).  Sinogram, weights, and error-sinogram all share
        this layout, so they shard on the same axis.  A classmethod so it is
        available without an instance (e.g. to size a placement for generated
        data) and stays overridable per geometry.
        """
        return 0

    @classmethod
    def recon_shard_axis(cls):
        """Axis of a recon-like array to shard.

        Default -1 (the last axis = slices).  This is correct for both the 3-D
        recon ``(rows, cols, slices)`` and the flat recon
        ``(rows*cols, slices)`` because slices are the last axis in both, so a
        single value works regardless of rank.  A classmethod so it is available
        without an instance (e.g. to size a placement for generated data) and
        stays overridable per geometry.
        """
        return -1

    def _sino_row_padding(self):
        """Detector-row padding spec for the sinogram device form, or None.

        Geometries whose projection kernels tie detector rows to recon slices
        (parallel beam: row r <-> slice r) must present the SAME padded length on
        both axes, so when the recon slice axis is padded for sharding the
        sinogram's (unsharded) row axis pads with it -- zero-filled at entry and
        inert, exactly like the padded views.  Geometries without that tie (cone:
        rows are independent of slices) keep their real rows and return None.

        Returns:
            (row_axis, real_rows, padded_rows) when row padding is active, else
            None.  Base default is None; ParallelBeamModel overrides.
        """
        return None

    def _shard_on_axis(self, x, axis, what='array'):
        """Distribute array ``x`` across the mesh along ``axis``.

        ``x`` is placed in a NamedSharding that partitions ``axis`` across the mesh's
        ``'devices'`` axis (a single device is the trivial 1-shard case).

        If ``x`` already carries exactly that sharding, it is returned as-is with
        no data movement.  Otherwise it is moved via the transfer helper, which
        copies directly when device-to-device transfer is safe on this hardware
        and routes through host memory otherwise (see mbirjax._sharding).

        This is the single chokepoint where entry arrays are sharded, so it is also
        where the equal-shard divisibility requirement is enforced with a clear
        error: the sharded axis must be divisible by the device count.  Real-shape
        inputs on a padded axis never reach this check (the pad-aware entry
        ``_pad_shard_on_axis`` handles them), so it is a BACKSTOP for device-form
        arrays, with a clear message rather than a cryptic XLA shard error.  (A
        no-op for a single device, since any size is divisible by 1.)

        Args:
            x: a JAX (or numpy) array to distribute.
            axis (int): the axis of ``x`` to partition across devices; may be
                negative (counted from the end).
            what (str): human label for ``x`` in the divisibility error message.

        Returns:
            The array in the requested NamedSharding.
        """
        axis = axis % x.ndim
        n_dev = len(self.shard_devices)
        if x.shape[axis] % n_dev != 0:
            raise ValueError(
                'Cannot shard the {} across {} devices: its sharded axis (size {}) is not '
                'divisible by {}.  Change the geometry so that axis is a multiple of {}, or select '
                'a compatible device count with configure_devices().'.format(
                    what, n_dev, x.shape[axis], n_dev, n_dev))
        spec = [None] * x.ndim
        spec[axis] = 'devices'
        sharding = jax.sharding.NamedSharding(
            self.recon_placement.mesh, jax.sharding.PartitionSpec(*spec))
        # Skip any movement if x is already in exactly this sharding.
        if isinstance(getattr(x, 'sharding', None), jax.sharding.NamedSharding):
            if x.sharding == sharding:
                return x
        return mjs.move_shard(x, sharding, dev2dev_safe=self.dev2dev_safe)

    def _gather_to_host(self, x):
        """Gather a (possibly sharded) array to a HOST NumPy array.

        ``np.asarray`` reads each addressable shard to the host and assembles one contiguous host
        buffer -- it never materializes the whole array on a single device.  This is what makes the
        ``output_sharded=False`` exit safe at any size: a large volume sharded across N devices is
        never re-gathered onto one device (which would OOM, e.g. a 32 GiB 2048^3 volume on one GPU).

        Args:
            x: a (possibly sharded) JAX array.

        Returns:
            A host ``numpy.ndarray`` with the same values.
        """
        # np.asarray of a multi-device sharded array assembles on the host shard-by-shard (the read
        # path is always safe, even where device-to-device writes are not); for a single-shard array
        # it copies that one device's data to the host.  Either way nothing large lands on one device.
        return np.asarray(x)

    def _shard_sinogram(self, sinogram):
        """Distribute a sinogram-like array in its native (per-geometry) sharding.

        Pad-aware on BOTH padded axes: when the view axis does not divide the
        device count, and/or the geometry pads detector rows with the recon
        slices (:meth:`_sino_row_padding`), a real-shape input is zero-padded to
        the device form (see :meth:`_pad_shard_on_axis`) and an already-padded
        input passes through unchanged.  Otherwise this is the plain shard
        chokepoint (:meth:`_shard_on_axis`), with a shape check that catches a
        STALE prepared array (padded for a previous device configuration whose
        size happens to divide the new one -- silently wrong if sharded).
        """
        axis = self.sinogram_shard_axis()
        row_pad = self._sino_row_padding()
        if self.sino_placement.is_padded or row_pad is not None:
            return self._pad_shard_on_axis(sinogram, self.sino_placement, axis,
                                           what='sinogram (view axis)', row_pad=row_pad)
        if sinogram.shape[axis % sinogram.ndim] != self.sino_placement.real_size:
            raise ValueError(
                'Cannot place the sinogram: its view axis has size {}, but the model expects '
                '{} views.  If this array was prepared with prepare_sino_for_devices under a '
                'different device configuration, re-run prepare_sino_for_devices (or pass the '
                'plain, unprepared array).'.format(
                    sinogram.shape[axis % sinogram.ndim], self.sino_placement.real_size))
        return self._shard_on_axis(sinogram, axis, what='sinogram (view axis)')

    def _gather_sinogram(self, sinogram):
        """Gather a sharded sinogram back to a single uncommitted array, cropping
        any zero-filled padded views (and padded detector rows) back to the
        problem's real counts."""
        out = self._gather_to_host(sinogram)
        if self.sino_placement.is_padded:
            axis = self.sinogram_shard_axis() % out.ndim
            if out.shape[axis] == self.sino_placement.padded_size:
                idx = [slice(None)] * out.ndim
                idx[axis] = slice(0, self.sino_placement.real_size)
                out = out[tuple(idx)]
        row_pad = self._sino_row_padding()
        if row_pad is not None:
            row_axis, real_rows, padded_rows = row_pad
            row_axis = row_axis % out.ndim
            if out.shape[row_axis] == padded_rows:
                idx = [slice(None)] * out.ndim
                idx[row_axis] = slice(0, real_rows)
                out = out[tuple(idx)]
        return out

    def _pad_shard_on_axis(self, x, placement, axis, what='array', row_pad=None):
        """Distribute ``x`` across ``placement``, zero-padding its sharded axis to
        the device form (``placement.padded_size``).

        ``row_pad`` is used in exactly one case -- the **ParallelBeam sinogram**.
        That sinogram is sharded by view (its sharded axis, padded as usual), but
        ParallelBeam back/forward-projects detector row ``r`` to/from recon slice
        ``r`` (a 1-to-1 row<->slice alignment unique to parallel beam), so the
        detector-row axis must match the recon-slice axis.  The recon slices are
        themselves sharded -- hence padded to the device form -- so the sinogram's
        detector rows must be zero-padded to that SAME length, even though rows are
        NOT the sinogram's sharded axis.  ``row_pad`` carries that second
        (unsharded) axis; for every other geometry, and for recon arrays, it is
        None and only the sharded axis is padded.  See :meth:`_sino_row_padding`.

        The padding never exists on the host: each device receives its own slice
        of the host array directly (``device_put`` per shard), and all zero tails
        (the last shard's view tail, and each shard's row tail) are created ON the
        receiving device -- so the only transient is one shard on one device,
        never a padded copy of the whole array.  The zero-filled tails are what
        keep the padding inert downstream (zero entries contribute nothing to any
        reduction).

        Accepts either the problem's REAL lengths (pads) or the device-form
        PADDED lengths (already prepared -- passes through `_shard_on_axis`,
        which is a no-op when the sharding already matches).  Mixed shapes (one
        axis real, the other padded) are rejected.

        Args:
            x: array (numpy or jax) to distribute.
            placement (mjs.Placement): the target placement; must carry
                ``real_size`` (and hence ``padded_size``) for its sharded axis.
            axis (int): the axis of ``x`` to partition (may be negative).
            what (str): human label for error messages.
            row_pad (tuple or None): the ParallelBeam detector-row case above, as
                ``(row_axis, real_rows, padded_rows)`` -- zero-pad this additional,
                unsharded axis up to ``padded_rows`` (the device-form recon-slice
                count).  None for every other geometry (pad only the sharded axis).

        Returns:
            The zero-padded array in the placement's NamedSharding.
        """
        axis = axis % x.ndim
        # Sharded-axis lengths: real_size is the problem's true length; padded_size is real_size
        # rounded up to a multiple of the device count (the device-form length).
        real_size, padded_size = placement.real_size, placement.padded_size
        if row_pad is not None:
            # A second, UNSHARDED axis (detector rows) that must also be zero-padded to track the
            # recon slices -- (axis, real length, padded length); see :meth:`_sino_row_padding`.
            row_axis, real_rows, padded_rows = row_pad
            row_axis = row_axis % x.ndim
        # Classify the incoming shape on each padded axis: already at its device-form (padded)
        # length, or at its real (problem) length?  (With no row_pad both are trivially True.)
        rows_are_padded = row_pad is None or x.shape[row_axis] == padded_rows
        rows_are_real = row_pad is None or x.shape[row_axis] == real_rows

        if x.shape[axis] == padded_size and rows_are_padded:
            # Already in the device form (e.g. a prepare_sino_for_devices output): nothing to pad,
            # just place it (``_shard_on_axis`` is a no-op when the sharding already matches).
            return self._shard_on_axis(x, axis, what=what)
        if x.shape[axis] != real_size or not rows_are_real:
            # Neither fully-real nor fully-device-form (e.g. a stale prepared array): refuse it.
            expected_real = 'problem size {}'.format(real_size) if row_pad is None else \
                'problem sizes {}/{} (sharded axis / padded row axis)'.format(real_size, real_rows)
            expected_dev = '{}'.format(padded_size) if row_pad is None else \
                '{}/{}'.format(padded_size, padded_rows)
            raise ValueError(
                'Cannot place the {}: got shape {}, but the model expects the {} '
                '(or the prepared device-form size {}).  If the device '
                'configuration changed since prepare_sino_for_devices, re-run it.'.format(
                    what, tuple(x.shape), expected_real, expected_dev))
        zeros_dtype = jax.dtypes.canonicalize_dtype(x.dtype)

        def with_row_tail(piece, dev):
            # Append the unsharded row axis's zero tail (real_rows -> padded_rows), built ON ``dev``,
            # so a placed shard reaches the device-form row length without a host-side padded copy.
            if row_pad is None:
                return piece
            tail_shape = list(piece.shape)
            tail_shape[row_axis] = padded_rows - real_rows
            tail = jnp.zeros(tuple(tail_shape), dtype=zeros_dtype, device=dev)
            return jnp.concatenate([piece, tail], axis=row_axis)

        # Build one shard per device.  ``padded_shard_ranges()`` walks the PADDED sharded axis and
        # yields, per device: the device, its ``[start, end)`` span on that axis, and ``n_real`` =
        # how many of those positions hold real data (any remainder is padding).  Each shard is
        # assembled directly on its own device, so the host never holds a padded copy.
        pieces = []
        for dev, (start, end), n_real in placement.padded_shard_ranges():
            shard_len = end - start          # this shard's span on the (padded) sharded axis
            parts = []
            if n_real > 0:
                # The real slice ``[start, start + n_real)`` of x, placed on ``dev`` (+ its row tail).
                idx = [slice(None)] * x.ndim
                idx[axis] = slice(start, start + n_real)
                parts.append(with_row_tail(jax.device_put(x[tuple(idx)], dev), dev))
            if shard_len - n_real > 0:
                # The remaining ``shard_len - n_real`` positions on the sharded axis are pure
                # padding: a zero block built directly on ``dev`` at the device-form row length.
                tail_shape = list(x.shape)
                tail_shape[axis] = shard_len - n_real
                if row_pad is not None:
                    tail_shape[row_axis] = padded_rows
                parts.append(jnp.zeros(tuple(tail_shape), dtype=zeros_dtype, device=dev))
            # A shard is the real block, the zero block, or -- at the one boundary device that
            # straddles real/padding -- both joined along the sharded axis.
            pieces.append(parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=axis))

        # Wrap the per-device shards as one logical array in the placement's NamedSharding; the
        # global shape is x's shape with the padded length(s) substituted on the padded axis/axes.
        global_shape = list(x.shape)
        global_shape[axis] = padded_size
        if row_pad is not None:
            global_shape[row_axis] = padded_rows
        return mjs.assemble_sharded(pieces, tuple(global_shape),
                                    placement.shard_structure(x.ndim))

    def prepare_sino_for_devices(self, sinogram, weights=None):
        """Place a sinogram (and optionally weights) in the model's device form, once.

        The device form is the view-sharded layout the reconstruction methods use
        internally: the sinogram is distributed across the configured devices,
        and when the view count does not divide the device count it is
        zero-padded to the next multiple (the padding is exactly inert -- it
        cannot affect the results).  The transfer streams shard-by-shard from the
        host, so no padded host copy is ever created; the only transient is one
        shard on one device.

        Calling this is OPTIONAL: every reconstruction method applies the same
        placement automatically to a plain input.  Use it to pay the host-to-
        device transfer once when running several reconstructions on the same
        large sinogram -- a prepared array passes through the entry placement
        untouched.  If the device configuration changes afterwards (e.g. a new
        configure_devices), the prepared array no longer matches and the entry
        placement raises with instructions to re-run this method.

        Args:
            sinogram (numpy or jax array): sinogram in the model's sinogram_shape.
            weights (numpy or jax array, optional): weights of the same shape;
                the zero-filled padding makes padded views weightless as well.

        Returns:
            The prepared sinogram, or a (sinogram, weights) tuple when weights
            were given.
        """
        sino = self._shard_sinogram(sinogram)
        if weights is None:
            return sino
        return sino, self._shard_sinogram(weights)

    def _shard_recon(self, recon):
        """Distribute a recon-like array (3-D or flat) in its native sharding.

        Pad-aware: when the slice axis does not divide the device count, a
        real-shape input is zero-padded to the device form (the forced-zero
        padded slices) and an already-padded input passes through unchanged.
        Otherwise this is the plain shard chokepoint, with a shape check that
        catches a stale device-form array from a previous configuration."""
        axis = self.recon_shard_axis()
        if self.recon_placement.is_padded:
            return self._pad_shard_on_axis(recon, self.recon_placement, axis,
                                           what='reconstruction (slice axis)')
        if recon.shape[axis % recon.ndim] != self.recon_placement.real_size:
            raise ValueError(
                'Cannot place the reconstruction: its slice axis has size {}, but the model '
                'expects {} slices.  If this array came from a previous device configuration '
                '(output_sharded=True), gather it with the old configuration or rebuild it.'.format(
                    recon.shape[axis % recon.ndim], self.recon_placement.real_size))
        return self._shard_on_axis(recon, axis, what='reconstruction (slice axis)')

    def _gather_recon(self, recon):
        """Gather a sharded recon back to a single uncommitted array, cropping any
        zero-filled padded slices back to the problem's real slice count."""
        out = self._gather_to_host(recon)
        if self.recon_placement.is_padded:
            axis = self.recon_shard_axis() % out.ndim
            if out.shape[axis] == self.recon_placement.padded_size:
                idx = [slice(None)] * out.ndim
                idx[axis] = slice(0, self.recon_placement.real_size)
                out = out[tuple(idx)]
        return out

    def _extract_halos(self, flat_recon):
        """Per-shard boundary slices for the qGGMRF inter-slice prior (thin wrapper).

        The substance lives in :func:`mbirjax.qggmrf.extract_halos` (explicit-args,
        model-free); this wrapper supplies the model's shard axis.  A single shard
        (trivial 1-device mesh) yields ``([None], [None])`` -- the reflected BC at both edges.
        """
        return mj.extract_halos(flat_recon, self.recon_shard_axis())

    def _replicate_scalar(self, x, placement):
        """Place scalar ``x`` replicated across ``placement``'s devices (no host round-trip).

        Used in the VCD line search: the forward-model scalars are reduced on the sino
        mesh and the prior scalars on the recon mesh (distinct meshes over the same
        devices), and the resulting ``alpha`` must scale both a recon-sharded and a
        sino-sharded array.  ``device_put`` to a fully-replicated ``NamedSharding`` keeps
        the value on-device (a cheap scalar broadcast/reshard, NVLink not host), so the
        line search never bounces a scalar through the host -- avoiding the per-subset
        device→host syncs that stall the GPU pipeline.
        """
        return jax.device_put(
            x, jax.sharding.NamedSharding(placement.mesh, jax.sharding.PartitionSpec()))

    def _stage_halos(self, flat_recon):
        """Extract + pre-place the qGGMRF boundary halos, once per partition pass
        (thin wrapper around :func:`mbirjax.qggmrf.stage_halos`; see there for the
        ordering contract and the per-pass-vs-per-subset rationale).  A single shard
        (trivial 1-device mesh) yields ``([None], [None])`` -- the reflected BC at both edges.
        """
        return mj.stage_halos(flat_recon, self.recon_shard_axis())

    def _qggmrf_prior_sharded(self, flat_recon, pixel_indices, qggmrf_params,
                              staged_halos=None):
        """qGGMRF prior gradient/Hessian on a slice-sharded recon (thin wrapper).

        The substance lives in :func:`mbirjax.qggmrf.qggmrf_gradient_and_hessian_sharded`
        (explicit-args, model-free); this wrapper supplies the model's placement
        state -- the in-plane shape, the recon placement, and the cached interface
        masks for a padded slice axis.
        """
        num_rows, num_cols = self.get_params('recon_shape')[:2]
        return mj.qggmrf_gradient_and_hessian_sharded(
            flat_recon, pixel_indices, qggmrf_params, num_rows, num_cols,
            self.recon_placement, staged_halos=staged_halos,
            interface_masks=self._qggmrf_interface_masks())

    def _qggmrf_interface_masks(self):
        """Per-device qGGMRF interface masks for a padded slice axis, or None.

        The qGGMRF kernel masks the inter-slice DIFFERENCES: interface j of a shard
        starting at global slice g0 is valid iff its higher-index slice is real
        (``g0 + j < num_real_slices`` -- the one padding predicate).  Masking an
        interface reproduces the reflected boundary condition there, so the prior
        sees the recon as ending at the last REAL slice even mid-shard (see
        :func:`mbirjax.qggmrf_grad_and_hessian_per_cylinder`).

        When the slice axis is padded, EVERY shard gets a mask (all-ones for fully
        real shards) so the jitted kernel keeps one uniform trace; when nothing is
        padded this returns None and the kernel call carries zero new work.

        The masks depend only on the device layout, not on the recon values, so
        they are built once per layout and cached (``_set_device_layout`` invalidates
        the cache on every recompile).  Building them per VCD subset would re-pay a
        host->device transfer thousands of times -- the per-subset host round-trips
        are exactly what capped multi-GPU VCD scaling (see the staged-halos lesson).

        Returns:
            dict (device -> (local_slices+1,) float32 mask on that device), or None.
        """
        if not self.recon_placement.is_padded:
            return None
        if self._qggmrf_interface_masks_cache is None:
            real = self.recon_placement.real_size
            masks = {}
            for dev, (s0, s1), _n_valid in self.recon_placement.padded_shard_ranges():
                mask = ((s0 + np.arange(s1 - s0 + 1)) < real).astype(np.float32)
                masks[dev] = jax.device_put(jnp.asarray(mask), dev)
            self._qggmrf_interface_masks_cache = masks
        return self._qggmrf_interface_masks_cache

    def set_devices(self):
        """
        Determine whether to run the reconstruction entirely on the GPU (when one is available) or
        entirely on the CPU, and set the corresponding devices.

        This determination can be overridden by using ct_model.set_params(use_gpu=string), where string is
        'automatic' (use the gpu when available) or 'none' (cpu only).  ('full' is a deprecated synonym
        of 'automatic'.)

        Returns:
            Nothing, but instance variables are set to appropriate values.
        """
        # Establish the device layout (the placements own it; there is no stored mode string or
        # primary-device scalar to go stale).  We deliberately do NOT estimate whether the problem
        # fits -- an over-large recon surfaces as an OOM at recon time, where _handle_jax_error
        # guides the user.
        use_gpu = self.get_params('use_gpu')
        gpus = gpu_devices()   # () when there is no GPU backend
        if not gpus and use_gpu not in ['automatic', 'none']:
            warnings.warn("'use_gpu' is set to {} but no gpu is available. Proceeding on cpu. "
                          "Use set_params(use_gpu='automatic') to avoid this warning.".format(use_gpu))

        # A pinned configuration (configure_devices) keeps its devices across recompiles; otherwise
        # auto-select (_auto_device_pool: shard across all GPUs or all CPU devices, a single device
        # giving a trivial 1-device layout).  Either way _set_device_layout rebuilds the placements
        # and re-derives the pad metadata from the current shapes, so the layout tracks a
        # 'full' <-> 'none' flip or a sinogram_shape change.
        cur_devices = self.recon_placement.devices if self._sharding_configured else self._auto_device_pool()
        self._set_device_layout(cur_devices, pinned=self._sharding_configured)
        return

    def get_recon_dict(self, recon_params=None, notes=None, save_log=True, save_model=True, str_format=False):
        """
        Encapsulate the recon parameters, logs, notes, and optionally all model parameters to a text-based dict
        with entries 'recon_params', 'recon_log', 'notes', and optionally 'model_params'.  This dict can be used with
        :func:`mbirjax.viewer.slice_viewer` and :meth:`TomographyModel.save_recon_hdf5`.

        This dict from this function is returned by :meth:`TomographyModel.recon`.

        Args:
            recon_params (dict, optional): dict of reconstruction parameters. Defaults to None.
            notes (str, optional): User-supplied notes to attach to the dataset. Defaults to None.
            save_log (bool, optional): If True, saves the internal log buffer (if available). Defaults to True.
            save_model (bool, optional): If True, saves the model parameters as a YAML string. Defaults to True.
            str_format (bool, optional): If True, then each top level entry is a string, which is a yaml string when the entries could be saved as a dict.

        Returns:
            dict: A dict with entries
                 - 'recon_params'
                 - 'notes'
                 - 'recon_log'
                 - 'model_params'.

        Example:
            >>> recon, recon_dict = ct_model.recon(sinogram)
            >>> print(recon_dict['recon_log'])
        """

        # Create the attribute dictionary
        recon_dict = dict()
        if recon_params is None:
            recon_dict['recon_params'] = "# Recon params not saved."
        else:
            recon_dict['recon_params'] = recon_params

        if self.log_buffer is None or not save_log:
            recon_dict['recon_log'] = "# Log info not saved."
        else:
            recon_dict['recon_log'] = self.log_buffer.getvalue()

        if notes is None:
            notes = '# No notes saved'
        recon_dict['notes'] = notes

        # Optionally save model parameters to YAML
        if save_model:
            recon_dict['model_params'] = self.params.copy()
        else:
            recon_dict['model_params'] = '# Model not saved'

        if str_format:
            recon_dict = self.convert_subdicts_to_strings(recon_dict)

        return recon_dict

    @staticmethod
    def convert_subdicts_to_strings(recon_dict):
        """Serialize the entries in the recon_dict to strings"""
        if isinstance(recon_dict, dict):
            string_dict = recon_dict.copy()
            yaml_writer = YAML()
            for key, value in string_dict.items():
                if key.startswith('model_params') and isinstance(string_dict[key], dict):
                    # 'model_params' must be handled separately to guarantee the ability to reload
                    string_dict[key] = ParameterHandler.save_params(string_dict[key])
                elif isinstance(value, dict):
                    # Otherwise convert dicts to yaml strings
                    buf = io.StringIO()
                    yaml_writer.dump(value, buf)
                    string_dict[key] = buf.getvalue()
                else:
                    try:
                        string_dict[key] = str(value)
                    except:
                        raise ValueError('Entries in recon_dict must be strings or dicts that can be converted to strings')
        else:
            string_dict = recon_dict

        return string_dict

    def save_recon_hdf5(self, filepath, recon, recon_dict=None):
        """
        Save the reconstruction array and optionally the recon_dict from :meth:`recon`.

        This method creates a file that contains a single dataset named 'recon', with the entries in recon_dict
        serialized to strings and saved as hdf5 dataset attributes.

        The resulting file can be loaded with :meth:`load_recon_hdf5` or :meth:`mbirjax.viewer.slice_viewer`.

        Args:
            filepath (str or Path): Path to the output HDF5 file. Should typically end with a .h5 extension.
            recon (array-like): The reconstruction volume as a NumPy or JAX array.
            recon_dict (dict or None, optional): The dictionary of recon attributes from :meth:`get_recon_dict`

        Raises:
            Exception: If saving the file or directory creation fails.

        Example:
            >>> recon, recon_dict = ct_model.recon(sinogram)
            >>> recon_dict['notes'] += 'Test scan'
            >>> ct_model.save_recon_hdf5("output/my_recon.h5", recon, recon_dict=recon_dict)
        """
        arr = np.array(recon)
        mj.save_data_hdf5(filepath, arr, 'recon', recon_dict)

        # Log the save
        if self.logger:
            self.logger.info(f"Saved reconstruction and params to '{filepath}'")

    @staticmethod
    def load_recon_hdf5(filepath, recreate_model=False):
        """
        This function loads a numpy array stored in an HDF5 file created by :meth:`save_recon_hdf5`.
        It also loads any associated attribute dict and can use the model parameters in that dict to create a new model.

        Args:
            filepath (str): Path to the HDF5 file containing the reconstructed volume.
            recreate_model (bool, optional): Deprecated.  Will raise a ValueError if set to True.

        Returns:
            (recon, recon_dict)
                - recon (ndarray): The tensor saved by save_data_hdf5()
                - recon_dict (dict): A dict with the attributes for the data array as in :meth:`get_recon_dict`

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If more than one dataset is not found in the file or if recreate_model is set to True.

        Example:
            >>> recon, recon_dict = ct_model.load_recon_hdf5("output/recon_volume.h5")
            >>> recon.shape
            (64, 256, 256)
        """
        recon, recon_dict = mj.load_data_hdf5(filepath)
        if recreate_model:
            raise ValueError('recreate_model has been deprecated.  Remove this option and expect only 2 return values.')

        return recon, recon_dict


    def create_projectors(self):
        """
        Creates an instance of the Projectors class and set the local instance variables needed for forward
        and back projection and compute_hessian_diagonal.  This method requires that the current geometry has
        implementations of :meth:`forward_project_pixel_batch_to_one_view` and :meth:`back_project_one_view_to_pixel_batch`

        Returns:
            Nothing, but creates jit-compiled functions.
        """
        self.projector_functions = mj.Projectors(self)

    @staticmethod
    def compute_channel_coordinate(pixel_indices, single_view_params, projector_params):
        """
        Compute the CONTINUOUS projected detector-channel coordinate n_p for one view.

        This is the float coordinate whose rounded value is the horizontal fans' integer
        channel center.  The projector wrappers round it OUTSIDE the projector programs
        (projectors._jit_compute_scatter_centers) and pass the concrete integer centers
        into the kernels -- the horizontal-fan rounding-bug fix (see
        experiments/bugs_and_artifacts/jax rounding bug/phase_d_design.md).  Implementations
        must reuse the SAME float chain the kernels use for their weights (e.g. return the
        n_p element of compute_proj_data / compute_horizontal_data), so centers and weights
        can never disagree.

        Note:
            This method must be overridden for a specific geometry.

        Args:
            pixel_indices (jax array of int):  1D vector of indices into flattened array of size num_rows x num_cols.
            single_view_params (jax array): A 1D array of view-specific parameters (such as angle) for the current view.
            projector_params (namedtuple):  Tuple containing (sinogram_shape, recon_shape, get_geometry_params())

        Returns:
            jax array of shape (num_pixels,): the continuous channel coordinate per pixel.
        """
        warnings.warn('compute_channel_coordinate not implemented for TomographyModel.')
        return None

    @staticmethod
    def forward_project_pixel_batch_to_one_view(voxel_values, pixel_indices, view_params, n_p_centers,
                                                projector_params):
        """
        Forward project a set of voxels determined by indices into the flattened array of size num_rows x num_cols.

        Note:
            This method must be overridden for a specific geometry.

        Args:
            voxel_values (jax array):  2D array of shape (num_indices, num_slices) of voxel values, where
                voxel_values[i, j] is the value of the voxel in slice j at the location determined by indices[i].
            pixel_indices (jax array of int):  1D vector of indices into flattened array of size num_rows x num_cols.
            view_params (jax array):  A 1D array of view-specific parameters (such as angle) for the current view.
            n_p_centers (jax array of int): 1D vector of shape (num_pixels,): this view's integer
                channel centers, computed OUTSIDE the projector program from
                compute_channel_coordinate (concrete input; do NOT recompute in-kernel --
                see compute_channel_coordinate).
            projector_params (namedtuple):  Tuple containing (sinogram_shape, recon_shape, get_geometry_params())

        Returns:
            jax array of shape (num_det_rows, num_det_channels)
        """
        warnings.warn('Forward projector not implemented for TomographyModel.')
        return None

    @staticmethod
    def back_project_one_view_to_pixel_batch(sinogram_view, pixel_indices, single_view_params, n_p_centers,
                                             projector_params, coeff_power=1):
        """
        Calculate the backprojection value at a specified recon voxel cylinder given a sinogram view and parameters.

        Note:
            This method must be overridden for a specific geometry.

        Args:
            sinogram_view (jax array): one view of the sinogram to be back projected
            pixel_indices (jax array of int):  1D vector of indices into flattened array of size num_rows x num_cols.
            single_view_params (jax array): A 1D array of view-specific parameters (such as angle) for the current view.
            n_p_centers (jax array of int): 1D vector of shape (num_pixels,): this view's integer
                channel centers (concrete input; the SAME centers feed the forward projector --
                see compute_channel_coordinate).
            projector_params (namedtuple):  Tuple containing (sinogram_shape, recon_shape, get_geometry_params())
            coeff_power (int): backproject using the coefficients of (A_ij ** coeff_power).
                Normally 1, but should be 2 for compute_hessian_diagonal.

        Returns:
            The value of the voxel for all slices at the input index (i.e., a voxel cylinder) obtained by backprojecting
            the input sinogram view.
        """
        warnings.warn('Back projector not implemented for TomographyModel.')
        return None

    def forward_project(self, recon, output_sharded=False):
        """
        Perform a full forward projection at all voxels in the field-of-view.

        The projection automatically uses whatever devices the model is configured
        for: on one device it runs there, and on several it is spread across them.
        ``recon`` may be an ordinary array or one already distributed across the
        model's devices (e.g. the output of an earlier on-device step) -- either is
        accepted, and the returned form is set by ``output_sharded`` regardless of
        which you pass.

        Note:
            This method should generally not be used directly for iterative reconstruction.  For iterative
            reconstruction, use :meth:`recon`.

        Args:
            recon (numpy or jax array): The 3D reconstruction array, either ordinary or
                already distributed across the model's devices.
            output_sharded (bool, optional): Choose the form of the returned
                sinogram.  If False (default), return an ordinary host NumPy array
                (assembled on the host -- safe at any size).  If True, leave it
                distributed (view-sharded) across the model's devices, so a following
                on-device step can use it without gathering it back.

        Returns:
            numpy or jax array: The 3D sinogram -- an ordinary host NumPy array by
            default, or view-sharded across the devices if ``output_sharded=True``.
        """
        # Implementation notes (not user-facing):
        # - The input is sharded at entry (_shard_recon is a no-op when it already
        #   is), so the output form does not depend on the input's form.
        # - The compute is the same all-gather forward projection either way
        #   (sparse_forward_project); only the exit differs -- output_sharded keeps
        #   the view-sharded device form, otherwise _gather_sinogram brings it back
        #   to one array and crops any padded views/detector rows.
        recon_shape, use_ror_mask = self.get_params(['recon_shape', 'use_ror_mask'])
        full_indices = mj.gen_full_indices(recon_shape, use_ror_mask=use_ror_mask)

        recon = self._shard_recon(recon)            # no-op if already sharded
        # Extracting cylinders from a slice-sharded volume keeps the slice
        # sharding (the index is on the unsharded row/col axes), so this yields
        # slice-sharded cylinders with no movement.
        voxel_values = self.get_voxels_at_indices(recon, full_indices)
        # All-gather forward projection -> view-sharded sinogram.
        sinogram = self.sparse_forward_project(voxel_values, full_indices)
        if output_sharded:
            return sinogram                          # keep the device form
        return self._gather_sinogram(sinogram)       # default: numpy output

    def back_project(self, sinogram, output_sharded=False):
        """
        Perform a full back projection at all voxels in the field-of-view.

        The back projection automatically uses whatever devices the model is
        configured for: on one device it runs there, and on several it is spread
        across them.  ``sinogram`` may be an ordinary array or one already
        distributed across the model's devices (e.g. a sinogram from
        :meth:`prepare_sino_for_devices`) -- either is accepted, and the returned
        form is set by ``output_sharded`` regardless of which you pass.

        Note:
            This method should generally not be used directly for iterative reconstruction.  For iterative
            reconstruction, use :meth:`recon`.

        Args:
            sinogram (numpy or jax array): The 3D sinogram, either ordinary or already
                distributed across the model's devices.
            output_sharded (bool, optional): Choose the form of the returned recon.
                If False (default), return an ordinary host NumPy array (assembled
                on the host -- safe at any size).  If True, leave it distributed
                (slice-sharded) across the model's
                devices, so a following on-device step (such as a sharded
                initialization for :meth:`recon`) can use it without gathering it
                back; on a single device the two forms are identical.

        Returns:
            numpy or jax array: The reconstructed 3D volume -- an ordinary host NumPy
            array by default, or slice-sharded across the devices if
            ``output_sharded=True``.
        """
        # Implementation notes (not user-facing):
        # - The input is sharded at entry (_shard_sinogram is a no-op when it
        #   already is), so the output form does not depend on the input's form.
        # - The compute is the same reduce-scatter back projection either way
        #   (sparse_back_project); only the exit differs -- output_sharded
        #   assembles a slice-sharded volume, otherwise the cylinder is gathered
        #   and scattered into one ordinary volume.
        recon_shape, use_ror_mask = self.get_params(['recon_shape', 'use_ror_mask'])
        full_indices = mj.gen_full_indices(recon_shape, use_ror_mask=use_ror_mask)
        row_index, col_index = jnp.unravel_index(full_indices, recon_shape[:2])

        sinogram = self._shard_sinogram(sinogram)   # no-op if already sharded
        # Reduce-scatter back projection -> slice-sharded cylinder.
        recon_cylinders = self.sparse_back_project(sinogram, full_indices)
        if output_sharded:
            # Keep the recon sharded (no host round-trip).
            return self._assemble_recon_volume_sharded(
                recon_cylinders, recon_shape, row_index, col_index)
        # Default: gather the cylinder to the host and scatter into a plain HOST volume.  Building the
        # full volume with np.zeros (not jnp.zeros) keeps it off-device -- a jnp.zeros(recon_shape) here
        # would put the whole recon (e.g. 32 GiB at 2048^3) on one device.
        recon_cylinders = self._gather_recon(recon_cylinders)
        recon = np.zeros(recon_shape, dtype=recon_cylinders.dtype)
        recon[row_index, col_index] = recon_cylinders
        return recon

    def _assemble_recon_volume_sharded(self, recon_cylinders, recon_shape,
                                       row_index, col_index):
        """Scatter a slice-sharded cylinder into a slice-sharded 3-D recon volume.

        ``recon_cylinders`` is ``(num_pixels, num_slices)`` sharded along the slice
        axis (each device owns a contiguous band of slices).  The scatter
        ``recon[row_index, col_index, :] = cylinder`` is identical across slices,
        so each device scatters its own band locally into a
        ``(rows, cols, local_slices)`` zeros array — no cross-device movement —
        and the per-device volumes are wrapped as one slice-sharded array.

        Assumes the recon slice axis is the last axis (``recon_shard_axis() ==
        -1``); a geometry whose recon shards on a different axis would override
        this.

        Args:
            recon_cylinders (jax.Array): slice-sharded ``(num_pixels, num_slices)``.
            recon_shape (tuple): the global recon shape ``(rows, cols, num_slices)``.
            row_index, col_index (jax arrays): FOV pixel row/col indices (length
                num_pixels), the same on every device.

        Returns:
            jax.Array: slice-sharded ``(rows, cols, num_slices)`` recon volume.
        """
        rows, cols = int(recon_shape[0]), int(recon_shape[1])
        # Map each device to its local cylinder shard (already resident on it).
        dev_to_shard = {s.device: s.data for s in recon_cylinders.addressable_shards}

        def worker(i, device):
            cyl = dev_to_shard[device]                  # (num_pixels, local_slices)
            local = jnp.zeros((rows, cols, cyl.shape[1]))
            return local.at[row_index, col_index, :].set(cyl)

        results = mjs.run_per_device(self.shard_devices, worker)
        axis = self.recon_shard_axis() % len(recon_shape)
        spec = [None] * len(recon_shape)
        spec[axis] = 'devices'
        recon_sharding = jax.sharding.NamedSharding(
            self.recon_placement.mesh, jax.sharding.PartitionSpec(*spec))
        # Global slice count from the cylinder (the DEVICE form -- padded when the
        # slice axis pads); rows/cols from the params (never padded).
        return mjs.assemble_sharded(
            results, (rows, cols, recon_cylinders.shape[-1]), recon_sharding)

    def sparse_forward_project(self, voxel_values, pixel_indices):
        """
        Forward project the given voxel values to a sinogram.
        The indices are into a flattened 2D array of shape (recon_rows, recon_cols), and the projection is done using
        all voxels with those indices across all the slices.

        Args:
            voxel_values (jax.numpy.DeviceArray): 2D array of voxel values to project, size (len(pixel_indices), num_recon_slices).
            pixel_indices (jax array): Array of indices specifying which voxels to project.

        Returns:
            jnp array: The resulting 3D sinogram after projection, view-sharded across the mesh.
        """
        # Every geometry runs the placement path (a trivial 1-device mesh when single-device), so
        # the sharded projector handles all cases: the recon cylinders are slice-sharded in and the
        # result is returned view-sharded (no gather here; the user-facing forward_project gathers).
        return self._sparse_forward_project_sharded(voxel_values, pixel_indices)

    def _sharded_forward_project_setup(self, voxel_values, pixel_indices):
        """Shared setup for sharded forward projection (adjoint of the back setup).

        Defensively slice-shards the recon cylinders (a no-op when already
        slice-sharded), and builds the per-device data the band streaming needs.

        Returns:
            (devices, n_dev, num_padded_views, num_slices, num_pixels, recon_shard_info,
             view_ranges, local_pixels): ``num_padded_views`` is the DEVICE-FORM view
            count (the params' num_views, or the next multiple of the device count when
            that does not divide); ``recon_shard_info`` maps each slice-owner
            to ``(its cylinder slice-shard, the GLOBAL (start, stop) slice range it
            owns)``; ``view_ranges`` maps each view-owner to the GLOBAL view indices
            it produces (its sinogram view-shard); ``local_pixels`` is
            ``pixel_indices`` placed on each device once.
        """
        # Defensively slice-shard the cylinders (idempotent, no-op when already
        # slice-sharded) so internal callers that pass a plain array still work.
        voxel_values = self._shard_recon(voxel_values)

        devices = self.sino_placement.devices
        n_dev = len(devices)
        # The DEVICE-FORM view count: equals the params' num_views when it divides the
        # device count, the next multiple of it otherwise (the padded views are masked
        # to zero after projection, so they are inert).
        num_padded_views = self.sino_placement.padded_size
        num_slices = voxel_values.shape[1]
        num_pixels = len(pixel_indices)

        # recon_shard_info: slice-owner -> (its (num_pixels, slices_per_dev) shard,
        # the GLOBAL slice range it owns).  index[1] is the slice axis of the 2-D
        # cylinder; .indices() normalizes a possibly-open slice to (start, stop, step).
        recon_shard_info = {}
        for s in voxel_values.addressable_shards:
            start, stop, step = s.index[1].indices(num_slices)
            recon_shard_info[s.device] = (s.data, (start, stop))

        # view_ranges: view-owner -> the GLOBAL views it produces (its sino shard).
        # Padded view indices (>= the real view count) are clamped to the last real
        # view so they index the baked view_params_array safely; their projected
        # values are garbage at that repeated angle, but they are zeroed by the
        # padded-view mask after projection, so the value never matters.
        view_ranges = {dev: jnp.arange(v0, v1)
                       for dev, (v0, v1) in self.sino_placement.shard_ranges(num_padded_views)}
        if self.sino_placement.is_padded:
            last_real = self.sino_placement.real_size - 1
            view_ranges = {dev: jnp.minimum(rng, last_real)
                           for dev, rng in view_ranges.items()}

        local_pixels = [jax.device_put(pixel_indices, dev) for dev in devices]
        return (devices, n_dev, num_padded_views, num_slices, num_pixels,
                recon_shard_info, view_ranges, local_pixels)

    def _sparse_forward_project_sharded(self, voxel_values, pixel_indices):
        """
        Sharded forward projection: slice-sharded recon -> view-sharded sinogram.

        This is the **all-gather adjoint** of :meth:`_sparse_back_project_sharded`'s
        reduce-scatter.  Same two-level slice structure (see that method):

          * The recon is sharded by **slice**: device t (a *slice-owner*) owns a
            *slice-shard* of the cylinders, ``(num_pixels, slices_per_dev)``.
          * The sinogram is sharded by **view**: device d (a *view-owner*) owns a
            *view-shard* and must produce all detector rows for its own views.

        Because a view-owner's detector row r is the forward projection of slice r
        (parallel beam), producing its views needs slices from every slice-owner.
        The work is **streamed in bands**, the mirror of back projection:

          For each slice-band held by a slice-owner, the band is **broadcast** to
          every view-owner (``broadcast_band_to_views`` -- the adjoint of back's
          ``sum_band_to_owner``); each view-owner forward-projects ITS views from
          that band, producing exactly detector rows ``[g0:g1)`` (the kernel sizes
          its output rows from the input slices).  A view-owner concatenates its
          row-bands, in global-slice order, into its full view-shard -- no reduce,
          since each detector row is produced by exactly one view-owner.

        No gather is performed here (contract: the user-facing ``forward_project``
        gathers at exit).

        Args:
            voxel_values (jax array): slice-sharded recon cylinders
                ``(num_pixels, num_slices)``.
            pixel_indices (jax array): 1D indices into the flattened (rows, cols).

        Returns:
            jax array of shape sinogram_shape, view-sharded across the mesh.
        """
        (devices, n_dev, num_padded_views, num_slices, num_pixels,
         recon_shard_info, view_ranges, local_pixels) = \
            self._sharded_forward_project_setup(voxel_values, pixel_indices)

        # Produce each view-owner's forward-projected view-shard.  The default (geometry-neutral)
        # path gathers the full slice cylinder per view-owner and runs the monolithic forward;
        # ParallelBeamModel OVERRIDES _forward_project_to_view_shards with a banded forward that never
        # gathers (it exploits its detector-row r <- slice r identity).  See that hook.
        owned_views = self._forward_project_to_view_shards(
            devices, n_dev, num_slices, num_pixels, recon_shard_info, view_ranges, local_pixels)

        # Zero the padded views (if any) on their owners.  This is THE mask site that
        # keeps padding inert: with the entry zero-fill it establishes the invariant
        # that padded views of every sinogram-domain array are identically zero.
        owned_views = self._mask_padded_views(owned_views)

        # Wrap the per-view-owner shards as one view-sharded sinogram (no movement).  The
        # device-form sino shape carries the padded view count and, for geometries that pad
        # detector rows with slices (parallel beam, row r <- slice r), the padded row count; cone
        # keeps its real detector rows (it does not pad rows).  Channels come from params.
        return mjs.assemble_sharded(
            owned_views, self._sino_device_shape(),
            self.sino_placement.shard_structure(3))

    def _forward_project_to_view_shards(self, devices, n_dev, num_slices, num_pixels,
                                     recon_shard_info, view_ranges, local_pixels):
        """Produce every view-owner's forward-projected view-shard from the slice-sharded recon.

        Default (geometry-neutral) path: each view-owner GATHERS the full slice cylinder and runs
        the MONOLITHIC forward.  A slice can project to a RANGE of detector rows (cone), so every
        view-owner needs ALL slices to produce its own views' rows -- it cannot stream slice-bands.
        To bound memory the gather is done PER PIXEL-BATCH: only one pixel-batch's slices
        (``pixel_batch x num_slices``) are gathered at a time, the monolithic forward is run on
        that batch, and the per-view contributions are SUMMED over pixel-batches (forward
        projection sums over voxels).  All view-owners run in parallel, one thread each.

        ``ParallelBeamModel`` OVERRIDES this with a banded forward: it broadcasts one slice-band at
        a time and projects detector rows [g0:g1) from it (row r <- slice r), so it never gathers
        the full cylinder.

        Returns a list of per-view-owner sinogram shards ``(views_per_dev, num_det_rows,
        num_channels)``, ``owned_views[i]`` resident on ``devices[i]``.
        """
        pixel_batch = self.tiles.fwd_pixel_batch
        # Gather slice-shards in GLOBAL slice order so the assembled cylinder is correctly ordered.
        slice_owners = sorted(devices, key=lambda d: recon_shard_info[d][1][0])
        # The monolithic forward kernel anchors its slice->detector-row geometry on the REAL
        # slice count (recon_shape[2]) and requires the cylinder to have exactly that length.
        # When the slice axis is padded for sharding, the gathered cylinder carries the
        # device-form (padded) slices; crop them off before projecting.  This is EXACT, not an
        # approximation: the padded slices are zero (the forced-zero invariant on the
        # slice-sharded recon), so dropping them changes nothing -- it is what makes the
        # divisibility padding exactly inert for the cone gather-forward.  A no-op when the
        # slice count divides the device count (real_size == device-form length).
        real_slices = self.recon_placement.real_size

        def worker(i, view_owner):
            idx = local_pixels[i]                       # pixel indices, resident on view_owner
            n_local = int(idx.shape[0])
            # Single device: the whole cylinder is already local on this device, so there is
            # NO cross-device gather to bound -- the per-pixel-batch loop below is pure overhead
            # and, by issuing many small rigid dispatches, it defeats the XLA rematerialization
            # that the monolithic single-device forward relies on (measured: 1024^3 single-device
            # peak ~16 GB one-shot vs ~32 GB looped).  Project the full cylinder in ONE call,
            # recovering the one-shot single-device memory profile.
            if n_dev == 1:
                full_cyl = recon_shard_info[slice_owners[0]][0]
                return self.projector_functions.sparse_forward_project(
                    full_cyl, idx, owned_view_indices=view_ranges[view_owner])
            owned = None
            for p0 in range(0, n_local, pixel_batch):
                p1 = min(p0 + pixel_batch, n_local)
                # Gather THIS pixel-batch's slices from every slice-owner onto this view-owner and
                # concatenate along the slice axis -> the full cylinder (p1 - p0, device-form
                # slices), then crop to the real slice count (the inert zero padding; see above).
                full_cyl = jnp.concatenate(
                    [mjs.move_shard(recon_shard_info[so][0][p0:p1], view_owner,
                                    dev2dev_safe=self.dev2dev_safe)
                     for so in slice_owners], axis=1)
                full_cyl = full_cyl[:, :real_slices]
                part = self.projector_functions.sparse_forward_project(
                    full_cyl, idx[p0:p1], owned_view_indices=view_ranges[view_owner])
                owned = part if owned is None else owned + part
            return owned

        with mjs.device_pool(len(devices)) as pool:
            return mjs.run_per_device(devices, worker, executor=pool)

    def _mask_padded_views(self, owned_views):
        """Zero the padded views in per-view-owner sinogram blocks.

        ``owned_views[i]`` is view-owner i's full block of views (local view axis
        0), resident on its device.  Only owners whose global view range extends
        past the real view count have anything to zero -- with end-padding that
        is at most the last owner.  Applied AFTER the per-owner block is
        assembled from its bands (band-structure-agnostic), and on the local
        single-device arrays (the replaced block frees on refcount; no
        sharded-array reference cycle is created).  A no-op when nothing is
        padded.
        """
        if not self.sino_placement.is_padded:
            return owned_views
        masked_views = list(owned_views)
        for i, (dev, (v0, v1), n_valid) in enumerate(self.sino_placement.padded_shard_ranges()):
            block = v1 - v0
            if n_valid == block:
                continue
            if n_valid <= 0:
                masked_views[i] = jnp.zeros_like(masked_views[i])
            else:
                # Eager op on the committed local block: executes on its device.
                masked_views[i] = masked_views[i].at[n_valid:].set(0.0)
        return masked_views

    def _sino_ones_device_form(self, sino_like):
        """All-ones sinogram in the device form, with any padded entries ZERO.

        The constant-weights Hessian/VCD path back-projects a ones sinogram; padded views AND
        padded detector rows must contribute nothing, so their entries are zero (mirroring the
        entry zero-fill).  Thin wrapper over :func:`mbirjax._sharding.sharded_full` (fill 1), which
        builds it per-shard on each owner device (no host array, no data movement); ``sino_like``
        supplies only the dtype.

        Args:
            sino_like (jax array): a device-form sinogram supplying the dtype.

        Returns:
            A device-form all-ones (real entries) / zeros (padded entries) array.
        """
        return mjs.sharded_full(self.sino_placement, tuple(self.get_params('sinogram_shape')),
                                1.0, row_pad=self._sino_row_padding(), dtype=sino_like.dtype)

    def _sino_device_shape(self):
        """The sinogram shape as it exists ON THE DEVICES: the params shape with the
        view axis (and, for geometries that pad rows with slices, the row axis) at
        its device-form (possibly padded) length.  Equals the params shape exactly
        when nothing pads.  Use for validating device-form arrays; the params answer
        "what is the problem?", this answers "what is on the devices?"."""
        shape = list(self.get_params('sinogram_shape'))
        shape[self.sinogram_shard_axis() % len(shape)] = self.sino_placement.padded_size
        row_pad = self._sino_row_padding()
        if row_pad is not None:
            row_axis, _real_rows, padded_rows = row_pad
            shape[row_axis % len(shape)] = padded_rows
        return tuple(shape)

    def _recon_device_shape(self):
        """The recon shape as it exists ON THE DEVICES: the params shape with the
        slice axis at its device-form (possibly padded) length.  The padded slices
        are identically zero (forced-zero invariant)."""
        shape = list(self.get_params('recon_shape'))
        shape[self.recon_shard_axis() % len(shape)] = self.recon_placement.padded_size
        return tuple(shape)

    def _forward_project_all_bands(self, band_bounds, recon_shard_info, view_ranges,
                                   local_pixels, devices):
        """Broadcast every slice-band and forward-project it on every view-owner.

        Slice-owners are visited in GLOBAL slice order so each view-owner's
        row-bands accumulate in detector-row order.  For each band: broadcast it
        from its slice-owner to all view-owners, then every view-owner
        forward-projects ITS views from the band (rows ``[g0:g1)``).  A view-owner
        concatenates its row-bands along the detector-row axis into its view-shard.

        One thread pool spans every band's per-device fan-out.

        Returns:
            list: per-view-owner sinogram shards; ``owned_views[i]`` is
            ``(views_per_dev, num_rows, num_channels)`` resident on ``devices[i]``.
        """
        # view_bands[i] collects view-owner devices[i]'s row-bands in row order.
        view_bands = [[] for _ in devices]
        # Visit slice-owners in global slice order (their shard's start), so the
        # appended row-bands tile [0, num_rows) in order for every view-owner.
        slice_owners = sorted(devices, key=lambda d: recon_shard_info[d][1][0])
        with mjs.device_pool(len(devices)) as pool:
            for slice_owner in slice_owners:
                cyl_shard, _ = recon_shard_info[slice_owner]   # (num_pixels, slices_per_dev)
                for (l0, l1) in band_bounds:
                    band = cyl_shard[:, l0:l1]                 # (num_pixels, L) on slice_owner
                    band_on_views = mjs.broadcast_band_to_views(
                        band, devices, self.dev2dev_safe)
                    row_bands = self._forward_project_band_to_local_views(
                        band_on_views, local_pixels, view_ranges, devices, pool)
                    for i in range(len(devices)):
                        view_bands[i].append(row_bands[i])     # (vpd, L, num_channels)
        return [bands[0] if len(bands) == 1 else jnp.concatenate(bands, axis=1)
                for bands in view_bands]

    def _forward_project_band_to_local_views(self, band_on_views, local_pixels,
                                             view_ranges, devices, pool):
        """Forward-project a broadcast band on every view-owner, its own views only.

        The adjoint of ``_back_project_local_views_to_band``: there each view-owner
        back-projects its views onto a slice band (then the partials are summed);
        here each view-owner forward-projects ITS views FROM the band (no sum, each
        detector row has a single producer).  The kernel sizes its output rows from
        the input slices, so a band of ``L`` slices yields detector rows of length
        ``L``.  Runs one thread per view-owner (reusing ``pool``).

        Returns:
            list: per-view-owner row-bands, each ``(views_per_dev, L, num_channels)``,
            in device order.
        """
        def worker(i, device):
            band = band_on_views[device]                       # (num_pixels, L)
            return self.projector_functions.sparse_forward_project(
                band, local_pixels[i], owned_view_indices=view_ranges[device])
        return mjs.run_per_device(devices, worker, executor=pool)

    def sparse_back_project(self, sinogram, pixel_indices, coeff_power=1):
        """
        Back project the given sinogram to the voxels given by the indices.  The sinogram should be the full sinogram
        associated with all of the angles used to define the ct model.
        The indices are into a flattened 2D array of shape (recon_rows, recon_cols), and the projection is done using
        all voxels with those indices across all the slices.

        Args:
            sinogram (jnp array): 3D jax array containing the full sinogram.
            pixel_indices (jnp array): Array of indices specifying which voxels to back project.
            coeff_power (int, optional): Normally 1, but set to 2 for Hessian diagonal

        Returns:
            A jax array of shape (len(indices), num_slices).  On a sharded model this is the
            slice-sharded DEVICE FORM: its slice axis is the device-form (possibly padded) length,
            not the problem's real slice count, and the padded slices are exactly zero.  The
            user-facing :meth:`back_project` gathers and crops it to the real slice count; callers
            of this internal method that need the real shape must crop ``[:, :num_real_slices]``
            themselves (the padded slices are inert zeros, so the crop is exact).
        """
        # Every geometry runs the placement path, so the sharded projector handles all cases: the
        # sinogram is view-sharded in and the result is slice-sharded out (no gather here; the
        # user-facing back_project gathers).  The single-device back DRIVER below stays live -- the
        # GPU n=1 short-circuit in _sparse_back_project_sharded routes a single-GPU recon to it (the
        # pixel kernel is ~2.25x faster than the band kernel on GPU).
        return self._sparse_back_project_sharded(sinogram, pixel_indices, coeff_power=coeff_power)

    def _sparse_back_project_single_device(self, sinogram, pixel_indices,
                                           coeff_power=1, output_device=None):
        """
        Single-device back projection: project ALL views (view-batched) to voxel cylinders.

        No longer a dispatch target of :meth:`sparse_back_project` (which always takes the
        placement path); it is kept LIVE because the GPU n=1 back short-circuit in
        :meth:`_sparse_back_project_sharded` routes a single-GPU recon here -- the monolithic
        pixel kernel is ~2.25x faster than the band kernel on a single GPU.

        Contract: ``sinogram`` may be plain or view-sharded; it is sharded at entry
        (:meth:`_shard_sinogram`, a no-op when already sharded) so the whole body operates on
        the model's single device (``sino_placement.devices[0]``).
        """
        # Shard at entry so every sinogram slice below is already on the model's single device.
        sinogram = self._shard_sinogram(sinogram)

        # Batch the views and pixels to bound vmap memory.  This is a BACK-projection driver, so
        # the view slices follow the back knob (they feed sparse_back_project below).
        transfer_view_batch_size = self.tiles.back_view_batch
        # Host-transfer granularity: a large multiple of the pixel tile (the transfer is
        # communication, not a vmap width; 100x keeps the historical 204800 default).
        transfer_pixel_batch_size = 100 * self.tiles.back_pixel_batch
        num_views = sinogram.shape[0]
        all_view_indices = jnp.arange(num_views)          # all views (transfer-batched below)
        num_view_batches = jnp.ceil(sinogram.shape[0] / transfer_view_batch_size).astype(int)
        view_indices_batched = jnp.array_split(all_view_indices, num_view_batches)

        # Pin the indices to the MODEL's device -- not necessarily JAX's default device, since a
        # single-GPU recon can run on a non-default GPU of a multi-GPU host.  jnp.array_split below
        # would otherwise place its batches on the default device, mismatching the sharded
        # sinogram's device in the projector call.
        pixel_indices = jax.device_put(pixel_indices, self.sino_placement.devices[0])
        num_pixel_batches = jnp.ceil(pixel_indices.shape[0] / transfer_pixel_batch_size).astype(int)
        pixel_indices_batched = jnp.array_split(pixel_indices, num_pixel_batches)

        recon_shape = self.get_params('recon_shape')
        num_pixels = len(pixel_indices)
        num_slices = recon_shape[2]

        # Get the final recon as a jax array
        recon_at_indices = jnp.zeros((num_pixels, num_slices), device=output_device)
        for view_indices_batch in view_indices_batched:
            # sinogram was sharded at entry, so this slice is already on the model's device.
            view_batch = sinogram[view_indices_batch]

            # Loop over pixel batches
            voxel_batch_list = []
            for pixel_index_batch in pixel_indices_batched:
                # Back project a batch
                voxel_batch = self.projector_functions.sparse_back_project(view_batch, pixel_index_batch,
                                                                           owned_view_indices=view_indices_batch,
                                                                           coeff_power=coeff_power)
                voxel_batch = voxel_batch.block_until_ready()
                voxel_batch_list.append(jax.device_put(voxel_batch, output_device))

            recon_at_indices = recon_at_indices + jnp.concatenate(voxel_batch_list, axis=0)

        return recon_at_indices

    def _sharded_back_project_setup(self, sinogram, pixel_indices):
        """Shared setup for the sharded back-projection paths.

        Defensively shards the sinogram (a no-op when already view-sharded, so internal
        callers that pass a plain array -- e.g. compute_hessian_diagonal with plain
        weights -- still work), and builds the data needed for each view-owner.

        Returns:
            (devices, n_dev, num_slices, num_pixels, shard_info, local_pixels):
            ``shard_info`` maps each view-owner to ``(its local view-shard, the
            GLOBAL view indices it covers)`` so the projector picks the matching
            angles; ``local_pixels`` is ``pixel_indices`` placed on each view-owner
            once.
        """
        # Defensively run the (idempotent, no-op-when-already-sharded) shard so
        # internal callers that pass a plain array still work correctly.
        sinogram = self._shard_sinogram(sinogram)

        # The sino placement's device list (identical to the configured mesh's
        # shard devices) is the per-device fan-out order for the band projections.
        devices = self.sino_placement.devices
        n_dev = len(devices)
        # The DEVICE-FORM view count, read from the (already-placed) input: the params'
        # num_views, or the next multiple of the device count when that does not divide.
        num_padded_views = sinogram.shape[0]
        # The DEVICE-FORM slice count: the recon placement's padded size (equals the
        # params' num_slices when it divides the device count).  The band machinery
        # tiles this padded axis; the padded slices are zeroed post-assembly
        # (_mask_padded_slices), so their content is never seen downstream.
        num_slices = self.recon_placement.padded_size
        num_pixels = len(pixel_indices)

        # Create shard_info: a dict to map a view-owner device to its view-shard info.
        # shard.index[0] may be a full slice(None, None, None) on a single-device
        # shard; .indices() normalizes it to (start, stop, step).
        # On a padded view axis the global indices past the real view count are
        # clamped to the last real view so they index the baked view_params_array
        # safely; the padded views' DATA is identically zero (the entry/forward-mask
        # invariant), so they contribute nothing to the back-projection sum at any
        # angle.
        if self.sino_placement.is_padded:
            last_real = self.sino_placement.real_size - 1
        else:
            last_real = None
        shard_info = {}
        for s in sinogram.addressable_shards:
            start, stop, step = s.index[0].indices(num_padded_views)
            view_idx = jnp.arange(start, stop, step)
            if last_real is not None:
                view_idx = jnp.minimum(view_idx, last_real)
            shard_info[s.device] = (s.data, view_idx)

        # Send the pixel indices to all the view-owners for backprojection
        local_pixels = [jax.device_put(pixel_indices, dev) for dev in devices]
        return devices, n_dev, num_slices, num_pixels, shard_info, local_pixels

    def _sparse_back_project_sharded(self, sinogram, pixel_indices, coeff_power=1):
        """
        Sharded back projection: view-sharded sinogram -> slice-sharded recon.

        This implements the back-projection pipeline for the view/slice
        sharding scheme:

          * The sinogram is sharded by **view**: device d (a *view-owner*) owns a
            *view-shard*, a contiguous range of views (all detector rows, all channels).
          * The recon is sharded by **slice**: device t (a *slice-owner*) owns a
            *slice-shard*, a contiguous range of slices ``S_t``.
          * For communication purposes, each slice-shard on one device
            is further subdivided into *bands* of slices (aka *slice-bands*).
          * Aside from the heterogeneous CPU-(single GPU) case, each device is both
            a view-owner and a slice-owner.

        Since back projection sums each voxel's contribution over all views
        and the views are split across devices, the result is a *reduce-scatter*.

        The key picture for a maintainer:

          Each view-owner holds only a **subset (or shard) of sinogram views**, so it
          can compute only a *partial* back projection (its own views'
          contribution).  The view-owners compute these partials **in parallel**, one
          band at a time; the partials from a single band are then **summed onto the
          slice-owner for that band**. So, the only cross-device communication is
          this sum over all view-owners to a single slice-owner, once per band.
          Looping over bands keeps every device busy as a view-owner/view-projector,
          minimizes cross-device communication, and minimizes memory from
          intermediates in the projection and the sum over partials.

        The work is **streamed in bands**, so no full-cylinder partial is ever
        held.  Mind the two levels:

          * **slice-shard** -- the ``slices_per_dev`` contiguous slices held by a
            slice-owner device (the recon is slice-sharded).
          * **band** -- (aka **slice-band**) a contiguous sub-range of a slice-shard.
            (the streaming unit).  By default a band is *smaller* than the slice-shard
            (~n_dev bands per slice-owner); it equals the whole slice-shard only in the
            degenerate one-band case (see ``_slice_band_length()`` for why, and the
            sizing rationale).

        Bands tile each slice-owner's slice-shard with a balanced, no-overlap, zero-recompute
        split (``_balanced_slice_bounds()``).  The bands for one slice-owner are collected to
        form the slice-shard for that slice-owner.  Together, these shards form the full
        slice-sharded back projection with no further data movement.  No gather is performed
        here (contract: the user-facing ``back_project`` gathers at exit).

        Args:
            sinogram (jax array): view-sharded sinogram (NamedSharding on axis 0).
            pixel_indices (jax array): 1D indices into the flattened (rows, cols).
            coeff_power (int): 1 normally, 2 for the Hessian diagonal.

        Returns:
            jax array of shape (len(pixel_indices), num_slices), slice-sharded
            across the mesh.
        """
        # n=1 GPU short-circuit.  A single-GPU mesh has nothing to reduce-scatter, and the
        # banded reduce-scatter KERNEL (back_project_one_view_to_band) is ~2.25x slower than the
        # monolithic single-device kernel (back_project_one_view_to_pixel_batch) ON GPU -- measured
        # (cone, 512^3/1024^3): the sharded *driver* overhead is ~0 (a driver-less band loop ties
        # the full sharded path to 1.00x), so the cost is the band kernel itself.  So route to the
        # single-device path and wrap its output as a 1-shard slice-sharded array to honor the
        # output contract (metadata only, no copy; padded_size == the real slice count at n=1).
        # GPU ONLY: on CPU the SAME band kernel is ~8x FASTER -- it avoids the single-device
        # back-vertical cache cliff -- so CPU keeps the sharded path.  (The two kernels have
        # OPPOSITE platform rankings; see the platform-divergent back-kernel lesson in lessons.md.)
        if (len(self.recon_placement.devices) == 1
                and self.recon_placement.devices[0].platform == 'gpu'):
            owner = self.recon_placement.devices[0]
            out = self._sparse_back_project_single_device(
                sinogram, pixel_indices, coeff_power=coeff_power, output_device=owner)
            return mjs.assemble_sharded(
                [out], (len(pixel_indices), self.recon_placement.padded_size),
                self.recon_placement.shard_structure(2))

        devices, n_dev, num_slices, num_pixels, shard_info, local_pixels = \
            self._sharded_back_project_setup(sinogram, pixel_indices)

        # Each device owns a slice-shard, a contiguous range of slices_per_dev slices
        # (exact, since the slice axis is padded to a multiple of n_dev).  That
        # slice-shard is streamed in BANDS -- contiguous sub-ranges -- so the
        # slice-owner never gathers all n_dev partials for the whole slice-shard at once
        # (thereby limiting intermediate memory).  By default, a band is
        # ~slices_per_dev/n_dev (so ~n_dev bands per slice-owner); it equals the full
        # slice-shard only in the degenerate one-band case.
        slices_per_dev = num_slices // n_dev
        band_len = self._slice_band_length(
            slices_per_dev, n_dev, num_pixels,
            fixed_band=self._fixed_slice_band('back'))
        band_bounds = self._balanced_slice_bounds(slices_per_dev, band_len)

        # Do the back projection:
        #   a double loop over slice-owners on the outside and bands on the inside
        # owned[t] = slice-owner t's recon slice-shard assembled from its bands.
        owned = self._back_project_all_bands(
            slices_per_dev, band_bounds, shard_info, local_pixels, coeff_power,
            devices)

        # Zero the padded slices (if any) on their owners.  This is the back-projection
        # POSTCONDITION that mirrors the forward projection's _mask_padded_views: each
        # sharded projector guarantees its own output is inert in the padded region, so
        # every consumer (the VCD gradient, the Hessian diagonal, the direct/FBP init)
        # is automatically clean with no further mask sites.  Under parallel beam the
        # slice<->row identity already makes these entries structurally zero (defense);
        # under cone back projection bleeds real rows into padded slices and this
        # single site is what zeroes them (load-bearing).
        owned = self._mask_padded_slices(owned)

        # Wrap the shards from each slice-owner into one slice-sharded array (no data movement).
        return mjs.assemble_sharded(
            owned, (num_pixels, num_slices),
            self.recon_placement.shard_structure(2))

    def _mask_padded_slices(self, owned):
        """Zero the padded slices in per-slice-owner cylinder blocks.

        ``owned[t]`` is slice-owner t's block ``(num_pixels, slices_per_dev)``,
        resident on its device, with the slice axis LOCAL (axis 1).  Only owners
        whose global slice range extends past the real slice count have anything
        to zero -- with end-padding that is the boundary owner and any fully
        padded owners after it.  Applied AFTER the per-owner block is assembled
        from its bands (band-structure-agnostic), on the local single-device
        arrays (the replaced block frees on refcount).  A no-op when nothing is
        padded.
        """
        if not self.recon_placement.is_padded:
            return owned
        masked = list(owned)
        for i, (dev, (s0, s1), n_valid) in enumerate(self.recon_placement.padded_shard_ranges()):
            block = s1 - s0
            if n_valid == block:
                continue
            if n_valid <= 0:
                masked[i] = jnp.zeros_like(masked[i])
            else:
                # Eager op on the committed local block: executes on its device.
                masked[i] = masked[i].at[:, n_valid:].set(0.0)
        return masked

    def _back_project_all_bands(self, slices_per_dev, band_bounds, shard_info,
                                local_pixels, coeff_power, devices):
        """Back-project every band of every slice-owner; return each slice-owner's slice-shard.

        Slice-owner ``t`` owns one slice-shard, the contiguous range of slices
        ``[t*slices_per_dev, (t+1)*slices_per_dev)``; ``band_bounds`` tiles that
        slice-shard into bands (local offsets, the same tiling for every slice-owner).  For
        each band, all view-owners back-project their local views in parallel and the
        partials are summed onto slice-owner ``t``; the slice-owner's bands are concatenated
        along the slice axis into its shard.  A band equals the whole slice-shard only
        when there is one band per slice-owner.

        One thread pool is reused across every band (there are up to
        ~n_dev^2 bands) rather than a fresh pool per call. Each thread
        does the backprojections on a single view-owner.

        Returns:
            list: slice-shards; ``owned[t]`` is a slice-shard of shape ``(num_pixels, slices_per_dev)``
            resident on the slice-owner ``devices[t]``.
        """
        owned = []
        # Get the threads - one per view-owner to do the backprojection.
        with mjs.device_pool(len(devices)) as pool:
            # Loop over the slice-owners, then an inner loop over each slice-band so each view-owner can
            # project its subset of views, then sum over all views and return the slice-band to slice-owner.
            for t, slice_owner in enumerate(devices):
                bands = []
                # Loop over the bands - multiple bands per slice-shard.
                for (l0, l1) in band_bounds:
                    g0, g1 = t * slices_per_dev + l0, t * slices_per_dev + l1

                    # Do the backprojection of this band on each view-owner in parallel.
                    # The slice-band output, `partials`, is spread over the view-owners.
                    # These are 'partial' because they still need to be summed over all views.
                    partials = self._back_project_local_views_to_band(
                        shard_info, local_pixels, g0, g1, coeff_power, devices, pool)

                    # Sum all the partial bands across the view-owners and return to this slice-owner.
                    bands.append(mjs.sum_band_to_owner(
                        partials, slice_owner, self.dev2dev_safe))

                # Convert the separate bands into a single slice-shard.
                # `owned` is a list of slice-shards, one per slice-owner.
                owned.append(bands[0] if len(bands) == 1
                             else jnp.concatenate(bands, axis=1))
        return owned

    # Lower floor on per-band work (num_pixels * band, in elements) for the
    # default slice-band sizing.  From the H100 band sweep (1024^3 vs 256^3):
    # ~50M elements/band scaled fine and even sped up (smaller working set ->
    # better locality on this bandwidth-bound op), while ~0.8M/band added dispatch
    # overhead with no memory benefit (small recons are already tiny).  4M is a
    # safe knee; tunable.
    _BACK_PROJECT_MIN_BAND_WORK = 4_000_000
    # Upper cap on per-band work for the default.  This is what makes a SINGLE
    # device stream: the cross-device reduce-gather bound (slices_per_dev/n_dev)
    # is vacuous at n_dev=1, so without this cap one device would use the full
    # cylinder.  Capping the per-band partial bounds the compute working set
    # (partial + vmap-over-views buffer, both ~ band) so the peak drops toward the
    # sino+recon floor.  From the H100 single-device sweep at 1024^3: ~100M
    # elements/band lands near band ~128 (peak ~0.37x of unstreamed) and time was
    # FLAT across all bands, so this is free.  Tunable.
    _BACK_PROJECT_MAX_BAND_WORK = 100_000_000

    def _back_project_local_views_to_band(self, shard_info, local_pixels, g0, g1,
                                          coeff_power, devices, pool):
        """Back-project every view-owner's local views onto the global slice band [g0:g1).

        Each view-owner holds only a shard of the sinogram's views, so this produces a
        *partial* back projection per view-owner -- the contribution of that view-owner's
        views to slices [g0:g1).  This is why the caller must sum the results
        across view-owners (see ``mbirjax._sharding.sum_band_to_owner()``).

        For each view-owner, the **detector-row axis** of its view-shard is cropped
        to rows [g0:g1) (all views and all channels kept) and the projector is run;
        the kernel sizes its output-slice axis from the number of input rows, so it
        returns exactly slices [g0:g1).  The view-owners run in parallel, one thread
        each (reusing ``pool``).

        NOTE -- this row-crop is **parallel-beam-specific**: it works only because
        detector row r back-projects to slice r alone (the projection mixes
        channels, never rows).  Divergent geometries (cone / translation /
        multiaxis) draw a slice from a *range* of rows and so cannot crop rows.  The
        planned geometry-neutral replacement passes the full view plus a slice band
        ``(g0 dynamic, L static)`` and lets each geometry map the band to the rows
        it needs -- all local, since a view-owner holds every row for its own views.
        See the "geometry-neutral slice-band projector interface" design note in
        sharding_implementation_plan_v2.md.

        Args:
            shard_info (dict): device -> (local view-shard data, its GLOBAL view
                indices), from ``_sharded_back_project_setup``.
            local_pixels (list): ``pixel_indices`` replicated over view-owners;
                ``local_pixels[i]`` lives on ``devices[i]``.
            g0, g1 (int): half-open GLOBAL slice range for this band.
            coeff_power (int): 1 normally, 2 for the Hessian diagonal.
            devices (sequence): the per-device fan-out order.
            pool (ThreadPoolExecutor): reused across all bands.

        Returns:
            list: per-view-owner slice-band partials, each ``(num_pixels, g1 - g0)``, in
            device order (``result[i]`` is the partial on ``devices[i]``).
        """
        def worker(i, device):
            data, global_view_idx = shard_info[device]
            return self._back_project_view_shard_to_band(
                data, local_pixels[i], g0, g1, global_view_idx, coeff_power)
        return mjs.run_per_device(devices, worker, executor=pool)

    def _back_project_view_shard_to_band(self, view_data, pixel_indices, g0, g1,
                                         owned_view_indices, coeff_power):
        """Back-project one view-owner's full view-shard onto the GLOBAL slice band [g0, g1).

        Default (geometry-neutral) behavior: run the geometry's BANDED back kernel on the FULL
        view, producing slices [g0, g1) directly -- correct when a slice is drawn from a RANGE of
        detector rows (so the rows cannot be cropped).  This is the general case; cone uses it as
        is, as will translation/multiaxis once ported.  ``ParallelBeamModel`` OVERRIDES it with a
        cheaper detector-row crop (its row r back-projects to slice r alone), a specialization that
        avoids processing the full detector rows.

        Returns the per-view-owner PARTIAL band ``(num_pixels, g1 - g0)`` -- this view-owner's
        views' contribution to slices [g0, g1), still to be summed over view-owners by the
        reduce-scatter (``sum_band_to_owner``).
        """
        return self.projector_functions.sparse_back_project_band(
            view_data, pixel_indices, g0, g1 - g0,
            owned_view_indices=owned_view_indices, coeff_power=coeff_power)

    def _fixed_slice_band(self, op):
        """Resolve the slice-band override for ``op`` ('fwd' or 'back'), or None for the formula.

        Resolution order: a per-INSTANCE ``fwd_project_slice_band`` / ``back_project_slice_band``
        attribute (the experiment/test hook -- highest priority), else the tile policy's
        ``fwd_slice_band`` / ``back_slice_band`` field.  ``_slice_band_length`` clips the result
        to the slice shard, so an override larger than the shard means one band.
        """
        hook = {'fwd': 'forward_project_slice_band', 'back': 'back_project_slice_band'}[op]
        fixed = getattr(self, hook, None)
        if fixed is not None:
            return fixed
        return self.tiles.fwd_slice_band if op == 'fwd' else self.tiles.back_slice_band

    @staticmethod
    def _slice_band_length(slices_per_dev, n_dev, num_pixels, fixed_band=None):
        """Band length B for streaming the slice axis in sharded back projection.

        A smaller B lowers peak memory (the per-band partial and the
        vmap-over-views buffer both scale with B, and so does each slice-owner's
        reduce-scatter gather) at the cost of more, smaller projector calls -- but
        experiment sweeps over B showed time is essentially flat across B on GPUs,
        so smaller B is close to a free memory win.  (A 2026-06-09 multi-device
        sweep -- H100x4, n_dev=1/2/4, 512^3/1024^3 -- confirmed this in the
        high-n_dev / cross-NUMA regime: time flat across B, while a full-shard band
        cost up to ~2x the peak for identical time, so the n_dev^2 default is kept
        and budget-driven band sizing was rejected.)

        Default, two upper bounds (take the smaller) plus a lower floor:
          * reduce-gather bound ``slices_per_dev / n_dev``.  This is the elegant
            one.  Adding devices buys two compounding factors of 1/n_dev, from two
            different places, and the band length absorbs both:

              - The recon is slice-sharded, so a slice-owner's *output* slice-shard is already
                ``num_slices/n_dev`` -- the memory win we came for: the shard on
                each device shrinks like 1/n_dev.
              - But to build a band, the slice-owner must briefly hold one partial *per
                contributing view-owner* before summing them -- a transient of
                ``n_dev x (num_pixels x band)``.  For a fixed band that transient
                *grows* with n_dev and would cancel the win above.

            Pinning the transient to ~one output slice-shard,
            ``n_dev x band ~ slices_per_dev``, gives ``band ~ slices_per_dev/n_dev
            = num_slices/n_dev^2``.  The second 1/n_dev is exactly what the n_dev-way
            gather needs so it stays within the budget the first 1/n_dev already
            shrank -- so the whole per-device peak tracks the (1/n_dev) output slice-shard
            (the multi-device knee, ~3.3x a slice-shard at 1024^3/4-dev) instead of
            plateauing.  The price -- ~n_dev bands per slice-owner, hence more dispatches
            -- is cheap on GPU (launch throughput hides it; time is flat across B)
            and carries no extra FLOPs (bands tile with no overlap).  Vacuous at
            n_dev=1 (the next bound takes over).
          * compute bound ``_BACK_PROJECT_MAX_BAND_WORK / num_pixels`` --
            bounds the per-band compute working set; this is what makes a SINGLE
            device stream (1024^3 on one GPU: ~28 GB -> ~10 GB, no time cost),
            driving the peak toward the sino+recon floor.
          * lower floor ``_BACK_PROJECT_MIN_BAND_WORK / num_pixels`` -- keep a
            band's work above the dispatch floor so small recons aren't split into
            many tiny projector calls (the 256^3 penalty).

        Pass ``fixed_band`` (from ``self.back_project_slice_band``) to override the
        default with a fixed B (sweepable, like ``tomography_utils.ROW_FILTER_BATCH``).
        Always capped at slices_per_dev, so a band never crosses a slice-owner boundary.
        """
        b = fixed_band
        if not b:
            reduce_band = max(1, slices_per_dev // n_dev)      # cross-device gather
            compute_band = max(1, TomographyModel._BACK_PROJECT_MAX_BAND_WORK //
                               max(1, num_pixels))             # makes n_dev=1 stream
            work_floor_band = max(1, TomographyModel._BACK_PROJECT_MIN_BAND_WORK //
                                  max(1, num_pixels))          # avoid over-split
            b = max(min(reduce_band, compute_band), work_floor_band)
        return min(int(b), slices_per_dev)

    @staticmethod
    def _balanced_slice_bounds(extent, band_len):
        """Tile ``[0, extent)`` into balanced bands no longer than ``band_len``.

        Uses the fewest bands (ceil(extent / band_len)) with lengths as equal as
        possible (differing by at most 1), so bands never overlap and no slice is
        recomputed -- the right tradeoff for back projection, where a band is an
        expensive projector call (unlike the cheap per-row FBP filter, where a
        fixed length with an overlapping tail is fine).  Returns (start, stop)
        pairs covering [0, extent) exactly.
        """
        num_bands = -(-extent // band_len)            # ceil division
        base, rem = divmod(extent, num_bands)
        bounds, start = [], 0
        for k in range(num_bands):
            length = base + (1 if k < rem else 0)
            bounds.append((start, start + length))
            start += length
        return bounds

    def compute_hessian_diagonal(self, weights=None, output_sharded=False):
        """
        Computes the diagonal of the Hessian matrix, which is computed by doing a backprojection of the weight
        matrix except using the square of the coefficients in the backprojection to a given voxel.
        If weights is not None, it must be an array with the same shape as the sinogram to be backprojected.
        If weights is None, then constant weights 1 will be used

        Args:
            weights (numpy or jax array, optional): 3D positive weights with same shape as sinogram.  Defaults to all 1s.
            output_sharded (bool, optional): If False (default), return a numpy array in the
                problem's REAL shape (any padded slices are cropped).  If True, return the
                internal device form (slice-sharded, slice axis possibly padded with
                exactly-zero entries).

        Returns:
            numpy or jax array: Diagonal of the Hessian matrix with same shape as recon.
        """
        sinogram_shape, recon_shape = self.get_params(['sinogram_shape', 'recon_shape'])
        num_views = sinogram_shape[0]
        if weights is None:
            # Unit weights built directly in the view-sharded device form (ones in the real
            # views/rows, zero in the inert padding) -- no full sinogram-of-ones is ever
            # materialized on one device before sharding.
            weights = mjs.sharded_full(self.sino_placement, (num_views,) + tuple(sinogram_shape[1:]),
                                       1.0, row_pad=self._sino_row_padding())
        elif tuple(weights.shape) not in (tuple(sinogram_shape), self._sino_device_shape()):
            # Accept the problem shape (plain weights) or the device-form shape (weights
            # already placed, with a possibly padded view axis).
            error_message = 'Weights must be constant or an array compatible with sinogram'
            error_message += '\nGot weights.shape = {}, but sinogram.shape = {}'.format(weights.shape, sinogram_shape)
            raise ValueError(error_message)

        num_recon_rows, num_recon_cols, num_recon_slices = recon_shape[:3]
        max_index = num_recon_rows * num_recon_cols
        indices = jnp.arange(max_index)
        hessian_diagonal = self.sparse_back_project(weights, indices, coeff_power=2)

        # Slice count from the result: the sharded path returns the device form
        # (padded slice axis, masked to zero), the single-device path the real shape.
        hessian_diagonal = hessian_diagonal.reshape((num_recon_rows, num_recon_cols,
                                                     hessian_diagonal.shape[-1]))
        if output_sharded:
            return hessian_diagonal
        # Default exit: numpy array in the problem's REAL shape.
        return self._gather_recon(hessian_diagonal)

    def set_view_parameters(self, view_params):
        """
        Replace the view-dependent parameters (angles for parallel/cone beam, translation vectors
        for translation, etc.) used by the projectors, WITHOUT rebuilding them.

        The view parameters are a runtime input to the jitted projectors (not a baked constant),
        so this is a cheap value update: it does NOT recompile, provided the number of views is
        unchanged (a change in view count is a geometry change -- use set_params for that).  Use
        it to vary the acquisition geometry on the fly -- e.g. vcls iterating one view at a time,
        or motion correction perturbing per-view angles between iterations.

        Args:
            view_params: array shaped like the model's current view-parameter array (first axis =
                num_views); only the values may differ, not the shape.
        """
        view_params_name = self.get_params('view_params_name')
        current = jnp.asarray(self.get_params(view_params_name))
        view_params = jnp.asarray(view_params)
        if view_params.shape != current.shape:
            raise ValueError(
                'set_view_parameters requires the same shape as the current view-parameter '
                'array {} ({}); got {}.  A change in the number of views is a geometry change: '
                'use set_params({}=...) instead.'.format(
                    view_params_name, tuple(current.shape), tuple(view_params.shape),
                    view_params_name))
        # Update the stored parameter (the single source of truth for save/load and for any
        # later recompile) WITHOUT triggering a recompile -- avoiding that is the point here.
        self.set_params(no_compile=True, no_warning=True, **{view_params_name: view_params})
        # And the projectors' runtime array: the jitted projectors read it per call (late
        # binding), so the new values take effect on the next projection with no recompile.
        self.projector_functions.view_params_array = view_params

    def set_params(self, no_warning=False, no_compile=False, **kwargs):
        """
        Update parameters using keyword arguments.

        This method updates internal model parameters. If any key geometry-related parameters
        are modified, it triggers recompilation of the projector system unless suppressed
        via the `no_compile` flag.

        Args:
            no_warning (bool, optional): If True, disables validity checking and warning messages. Defaults to False.
            no_compile (bool, optional): If True, suppresses projector recompilation after updates. Defaults to False.
            **kwargs: Arbitrary keyword arguments specifying parameter names and values to update.

        Returns:
            bool: True if projector recompilation is required and not suppressed by `no_compile`,
            otherwise False.

        Example:
            >>> import mbirjax as mj
            >>> ct_model = mj.ParallelBeamModel(sinogram_shape, angles)
            >>> ct_model.set_params(recon_shape=(128, 128, 128), sharpness=0.7)
        """
        recompile_flag = super().set_params(no_warning=no_warning, no_compile=no_compile, **kwargs)
        if recompile_flag:
            self.set_devices()
            self.create_projectors()

    def verify_valid_params(self):
        """
        Check that all parameters are compatible for a reconstruction.

        Note:
            Raises ValueError for invalid parameters.
        """
        super().verify_valid_params()
        use_gpu = self.get_params('use_gpu')

        if use_gpu not in ['automatic', 'full', 'none']:
            error_message = "use_gpu must be one of \n"
            error_message += " 'automatic' (use the gpu when one is available, otherwise the cpu),\n"
            error_message += " 'none' (do not use gpu at all).\n"
            error_message += " ('full' is a deprecated synonym of 'automatic'.)"
            raise ValueError(error_message)

    def auto_set_regularization_params(self, sinogram, weights=None):
        """
        Automatically sets the regularization parameters (self.sigma_y, self.sigma_x, and self.sigma_prox) used in MBIR reconstruction based on the provided sinogram and optional weights.

        Args:
            sinogram (ndarray): 3D jax array containing the sinogram with shape (num_views, num_det_rows, num_det_channels).
            weights (ndarray, optional): 3D weights array with the same shape as the sinogram. Defaults to all 1s.

        Returns:
            namedtuple containing the parameters sigma_y, sigma_x, sigma_prox

        Notes:
            The method adjusts the regularization parameters only if `auto_regularize_flag` is set to True within the model's parameters.
            Also, the inputs may be jax arrays, but they are cast to numpy arrays before calculation to avoid
            duplicating large sinograms on the GPU.
        """
        if self.get_params('auto_regularize_flag'):
            # Make sure sinogram and weights are on the cpu to avoid duplication of large sinos on the GPU.
            # Estimate the regularization stats from a view subsample (see subsample_views) -- both cheap
            # and avoids reducing the whole (possibly huge) sinogram on the GPU.  Sample only the REAL
            # views: a device-form input (prepare_sino_for_devices) may zero-pad the view axis, and those
            # inert views must not bias the statistics (e.g. the all-ones indicator fallback would average
            # them in).  subsample_views(num_real_views=...) samples array[:num_real_views] directly.
            num_real_views = self.get_params('sinogram_shape')[0]
            small_sinogram = self.subsample_views(sinogram, num_real_views=num_real_views)
            small_weights = 1 if weights is None else self.subsample_views(weights, num_real_views=num_real_views)

            # Likewise crop padded detector ROWS (a device-form input whose row axis pads
            # with the recon slices) -- the zero rows would bias the indicator/sigma stats.
            num_real_rows = self.get_params('sinogram_shape')[1]
            if small_sinogram.shape[1] != num_real_rows:
                small_sinogram = small_sinogram[:, :num_real_rows]
                if weights is not None:
                    small_weights = small_weights[:, :num_real_rows]
            # Compute indicator function for sinogram support
            sino_indicator = self._get_sino_indicator(small_sinogram, verbose=self.get_params('verbose'))
            self.auto_set_sigma_y(small_sinogram, sino_indicator, small_weights)

            recon_std = self._get_estimate_of_recon_std(small_sinogram, sino_indicator)
            self.auto_set_sigma_x(recon_std)
            self.auto_set_sigma_prox(recon_std)

        regularization_param_names = ['sigma_y', 'sigma_x', 'sigma_prox']
        RegularizationParams = namedtuple('RegularizationParams', regularization_param_names)
        regularization_param_values = [float(val) for val in self.get_params(
            regularization_param_names)]  # These should be floats, but the user may have set them to jnp.float
        regularization_params = RegularizationParams(*tuple(regularization_param_values))._asdict()

        return regularization_params

    def auto_set_sigma_y(self, sinogram, sino_indicator, weights=1):
        """
        Sets the value of the parameter sigma_y used for use in MBIR reconstruction.

        Args:
            sinogram (jax array or ndarray): 3D jax array containing sinogram with shape (num_views, num_det_rows, num_det_channels).
            sino_indicator (jax array or ndarray): a binary mask that indicates the region of sinogram support; same shape as sinogram.
            weights (jax array, optional): 3D positive weights with same shape as sinogram.  Defaults to all 1s.
        """

        # Get parameters
        snr_db = self.get_params('snr_db')
        magnification = self.get_magnification()
        delta_voxel, delta_det_channel = self.get_params(['delta_voxel', 'delta_det_channel'])

        # Compute RMS value of sinogram excluding empty space
        signal_rms = float(np.average(weights * sinogram ** 2, None, sino_indicator) ** 0.5)

        # Convert snr to relative noise standard deviation
        rel_noise_std = 10 ** (-snr_db / 20)

        # This section adjusts the regularization when the reconstruction resolution is greater or less than normal.
        # For normal resolution reconstructions pixel_pitch_relative_to_default=1.0
        # For low resolution reconstructions pixel_pitch_relative_to_default>>1.0
        # And for high resolution reconstructions pixel_pitch_relative_to_default<<1.0
        #
        # compute the default_pixel_pitch = the detector pixel pitch in the recon plane given the magnification
        default_pixel_pitch = delta_det_channel / magnification
        # Compute the recon pixel pitch relative to the default.
        pixel_pitch_relative_to_default = delta_voxel / default_pixel_pitch

        # Compute sigma_y and scale by relative pixel pitch
        sigma_y = np.float32(rel_noise_std * signal_rms * (pixel_pitch_relative_to_default ** 0.5))
        self.set_params(no_warning=True, sigma_y=sigma_y, auto_regularize_flag=True)

    def auto_set_sigma_x(self, recon_std):
        """
        Compute the automatic value of ``sigma_x`` for use in MBIR reconstruction with qGGMRF prior.

        Args:
            recon_std (float): Estimated standard deviation of the reconstruction from _get_estimate_of_recon_std.
        """
        # Get parameters
        sharpness = self.get_params('sharpness')

        # Compute sigma_x as a fraction of the typical recon value
        # 0.2 is an empirically determined constant
        sigma_x = np.float32(0.2 * (2 ** sharpness) * recon_std)
        self.set_params(no_warning=True, sigma_x=sigma_x, auto_regularize_flag=True)

    def auto_set_sigma_prox(self, recon_std):
        """
        Compute the automatic value of ``sigma_prox`` for use in MBIR reconstruction with proximal map prior.

        Args:
            recon_std (float): Estimated standard deviation of the reconstruction from _get_estimate_of_recon_std.
        """
        # Get parameters
        sharpness = self.get_params('sharpness')

        # Compute sigma_x as a fraction of the typical recon value
        # 0.2 is an empirically determined constant
        sigma_prox = np.float32(0.2 * (2 ** sharpness) * recon_std)
        self.set_params(no_warning=True, sigma_prox=sigma_prox, auto_regularize_flag=True)

    def auto_set_recon_geometry(self, no_compile=False, no_warning=False):
        """
        Set the automatic value of the recon shape and voxel pitch using the geometry parameters and sinogram shape.

        Note: This function should be run after changing geometry parameters such as ``delta_det_channel``.
        It will set reconstruction parameters such as ``recon_shape`` and ``delta_voxel`` parameters to reasonable values.

        Args:
            no_compile (bool, optional): If True, do not recompile the JAX projector functions. Defaults to False.
            no_warning (bool, optional): If True, do not issue warnings. Defaults to False.

        Example:
            >>> import mbirjax as mj
            >>> import jax.numpy as jnp
            >>> sinogram = jnp.zeros(shape=(100, 100, 100))
            >>> angles = jnp.linspace(0, jnp.pi, 100)
            >>> ct_model = mj.ParallelBeamModel(sinogram.shape, angles)
            >>> ct_model.set_params(delta_det_channel=100.0)
            >>>
            >>> # Required reset of recon shape and voxel spacing parameters
            >>> ct_model.auto_set_recon_geometry()

        """
        raise NotImplementedError('auto_set_recon_geometry must be implemented by each specific geometry model.')

    def get_voxels_at_indices(self, recon, indices):
        """
        Retrieves voxel values from a reconstruction array at specified indices.
        The indices are into a flattened 2D array of shape (recon_rows, recon_cols), and the values are retrieved
        using all voxels with those indices across all the slices.

        Args:
            recon (ndarray): The 3D reconstruction array.
            indices (ndarray): Array of indices specifying which voxels to project.

        Returns:
            numpy.ndarray or jax.numpy.DeviceArray: Array of voxel values at the specified indices.
        """
        # Flatten the recon along the first two dimensions, then retrieve values of recon
        # at the indices locations.  The slice count comes from the ARRAY (its last axis),
        # not the params: a device-form recon may carry a padded slice axis, and the
        # indices address the (unsharded) pixel axes only.
        voxel_values = recon.reshape((-1, recon.shape[-1]))[indices]

        return voxel_values

    @staticmethod
    def subsample_views(array, max_views_to_use=20, num_real_views=None):
        """Return an evenly-spaced subsample of at most ``max_views_to_use`` views (axis 0) as a host
        NumPy array.

        Statistical sinogram estimates -- the support indicator (:meth:`_get_sino_indicator`), the RMS /
        typical value, the auto-crop width, the auto-regularization stats -- do not need every view, so
        they are computed on such a subsample.  This both keeps the host reductions cheap and, crucially,
        avoids materializing / reducing the whole (possibly ~20 GB) sinogram on the GPU.  Callers that
        need to subsample a companion array (e.g. weights) the same way just call this again with the
        same ``max_views_to_use``/``num_real_views`` (the stride depends only on the view count).

        If ``num_real_views`` is given, only the first ``num_real_views`` views are sampled: a device-form
        input (see ``prepare_sino_for_devices``) may zero-pad the view axis, and those padded views must
        not enter statistical estimates -- so we sample the REAL views directly rather than sampling the
        padded array and dropping padded slots afterward (which would leave fewer real views).

        Args:
            array (ndarray or jax array): array batched along axis 0 (views).
            max_views_to_use (int, optional): cap on the number of views retained. Defaults to 20.
            num_real_views (int or None, optional): if set, sample only ``array[:num_real_views]``.

        Returns:
            numpy.ndarray: the view-subsampled array on the host.
        """
        num_views = array.shape[0] if num_real_views is None else num_real_views
        max_views_to_use = min(max_views_to_use, num_views)
        step_size = max(num_views // max_views_to_use, 1)
        return np.array(array[:num_views][::step_size])

    @staticmethod
    def _get_sino_indicator(sinogram, verbose=1):
        """
        Compute a binary mask that indicates the region of sinogram support.

        Typically called on a view SUBSAMPLE (see :meth:`subsample_views`), not the full sinogram: this
        runs several host-side reductions, so callers estimating a statistical quantity should subsample
        the views first rather than pass the whole (possibly ~20 GB) sinogram.

        Args:
            sinogram (ndarray): 3D jax array containing sinogram with shape (num_views, num_det_rows, num_det_channels).
            verbose (int, optional): Verbosity level. Defaults to 1.

        Returns:
            (ndarray): Weights used in mbircone reconstruction, with the same array shape as ``sinogram``.
        """
        # Sometimes users accidentally create complex sinograms when they take the -log.
        # So we check for complex numbers or NaNs and raise an error.
        if np.iscomplexobj(sinogram):
            raise TypeError("sinogram must be real-valued; got complex dtype.")
        if not np.isfinite(sinogram).all():
            raise ValueError("sinogram contains NaN and/or Inf values.")

        # Compute an initial threshold the results in a non-empty region that contains no background.
        left_cluster_boundary, right_cluster_boundary = mj.utilities.estimate_background_cluster_boundaries(sinogram)
        cluster_width = right_cluster_boundary - left_cluster_boundary
        threshold = right_cluster_boundary + cluster_width      # This give some measure of safety about the estimate background

        # Make sure right_cluster_boundary less than or equal to the maximum sinogram value
        max_sino = np.max(sinogram)
        if max_sino <= 0:
            if verbose > 0:
                warnings.warn("Sinogram contains no positive values. This may lead to a contrast reversed reconstruction.")
            indicator = np.ones_like(sinogram, dtype=np.int8)
            return indicator

        if max_sino < threshold:
            if verbose > 0:
                warnings.warn('\nUnable to determine sinogram background. This may affect regularization.\n')
            indicator = np.ones_like(sinogram, dtype=np.int8)
            return indicator

        # Compute the a final threshold that is a fraction of the median of the object region
        object_level = 0.25
        object_median = np.median(sinogram[sinogram >= threshold])
        object_threshold = object_level * object_median

        # Compute the indicator
        indicator = np.int8(sinogram >= object_threshold)

        return indicator

    def _get_estimate_of_recon_std(self, sinogram, sino_indicator):
        """
        Estimate the standard deviation of the reconstruction from the sinogram.  This is used to scale sigma_prox and
        sigma_x in MBIR reconstruction.

        Args:
            sinogram (ndarray): 3D jax array containing sinogram with shape (num_views, num_det_rows, num_det_channels).
            sino_indicator (ndarray): a binary mask that indicates the region of sinogram support; same shape as sinogram.
        """
        # Get parameters
        delta_det_channel = self.get_params('delta_det_channel')
        delta_voxel = self.get_params('delta_voxel')
        recon_shape = self.get_params('recon_shape')
        magnification = self.get_magnification()
        num_det_channels = sinogram.shape[-1]

        # Compute the typical magnitude of a sinogram value
        typical_sinogram_value = np.average(np.abs(sinogram), weights=sino_indicator)

        # TODO: Can we replace this with some type of approximate operator norm of A? That would make it universal.
        # Compute a typical projection path length based on the soft minimum of the recon width and height
        typical_path_length_space = (2 * recon_shape[0] * recon_shape[1]) / (
                recon_shape[0] + recon_shape[1]) * delta_voxel

        # Compute a typical projection path length based on the detector column width
        typical_path_length_sino = num_det_channels * delta_det_channel / magnification

        # Compute a typical projection path as the minimum of the two estimates
        typical_path_length = np.minimum(typical_path_length_space, typical_path_length_sino)

        # Compute a typical recon value by dividing average sinogram value by a typical projection path length
        recon_std = typical_sinogram_value / typical_path_length

        return recon_std

    def direct_recon(self, sinogram, filter_name=None, output_sharded=False):
        """
        Do a direct (non-iterative) reconstruction, typically using a form of filtered backprojection.  The
        implementation details are geometry specific, and direct_recon may not be available for all geometries.

        Args:
            sinogram (numpy or jax array): 3D sinogram data with shape (num_views, num_det_rows, num_det_channels).
            filter_name (string or None, optional): The name of the filter to use, defaults to None, in which case the geometry specific method chooses a default, typically 'ramp'.
            output_sharded (bool, optional): If False (default), return a numpy array.  If True, return
                the internal device form (slice-sharded on a sharded model; on an unsharded model the
                output is the same single-device array either way).

        Returns:
            recon (numpy or jax array): The reconstructed volume after direct reconstruction.
        """
        warnings.warn('direct_recon not implemented for TomographyModel.')
        recon_shape = self.get_params('recon_shape')
        # Honor the output_sharded contract: device form (slice-sharded zeros, built per-shard)
        # vs a plain host array.  There is no sharded array to gather here, so build the host
        # zeros directly with np.zeros rather than routing a full-volume jnp.zeros through
        # _gather_to_host (which would materialize the whole volume on one device first).
        if output_sharded:
            return mjs.sharded_full(self.recon_placement, tuple(recon_shape), 0.0)
        return np.zeros(recon_shape, dtype=np.float32)

    def direct_filter(self, sinogram, filter_name=None, output_sharded=False):
        """
        Perform filtering on the given sinogram as needed for an FBP/FDK or other direct recon.

        Args:
            sinogram (numpy or jax array): The input sinogram with shape (num_views, num_rows, num_channels).
            filter_name (string, optional): Name of the filter to be used. Defaults to "ramp"
            output_sharded (bool, optional): If False (default), return a numpy array.  If True, return
                the internal device form (view-sharded on a sharded model; on an unsharded model the
                output is the same single-device array either way).

        Returns:
            filtered_sinogram (numpy or jax array): The sinogram after FBP filtering.
        """
        warnings.warn('direct_filter not implemented for TomographyModel.')
        sinogram_shape = self.get_params('sinogram_shape')
        # Honor the output_sharded contract: device form (view-sharded zeros, built per-shard)
        # vs a plain host array.  There is no sharded array to gather here, so build the host
        # zeros directly with np.zeros rather than routing a full-volume jnp.zeros through
        # _gather_to_host (which would materialize the whole sinogram on one device first).
        if output_sharded:
            return mjs.sharded_full(self.sino_placement, tuple(sinogram_shape), 0.0,
                                    row_pad=self._sino_row_padding())
        return np.zeros(sinogram_shape, dtype=np.float32)

    def _apply_direct_recon_filter(self, sinogram, filter_name, filter_scale, output_sharded,
                                   row_weight=None):
        """Shared FBP/FDK row-filter for direct reconstruction (one codebase for both).

        Scales the recon filter by ``filter_scale * pi / num_views`` -- folded into the
        (tiny) filter array, NOT applied as an out-of-place full-sinogram multiply (which
        promotes f32 -> f64 via np.pi and ~doubles peak memory -- the large-size
        single-device OOM this replaces).  ``num_views`` is the REAL count from params
        (padded views excluded).  Optionally pre-weights each detector row by
        ``row_weight`` (the FDK cosine map; None for FBP).  The row-batched kernel
        (tomography_utils.apply_row_filter) keeps the peak at the input+output floor;
        when a mesh is configured each device filters its own view-shard locally (no
        cross-device movement), exactly like ParallelBeamModel.fbp_filter.

        Equally-spaced-angle assumption: the ``pi / num_views`` factor is the angular
        quadrature weight ``d(theta)`` of the backprojection sum that approximates the
        FBP/FDK angular integral, so it assumes the views are EQUALLY SPACED over the
        conventional full angular range (the [0, pi) period for parallel beam, via the
        conjugate-ray symmetry).  For NONUNIFORMLY-spaced angles, LIMITED-ANGLE scans, or
        short scans this scalar is only approximate -- the quadrature-correct weight would be
        a PER-VIEW local angular spacing, and short scans additionally need redundancy
        (Parker-style) weighting.  That is acceptable for the intended use of direct recon as
        a quick analytic image or an MBIR initializer (iterative ``recon()`` absorbs a global
        angular mis-weighting in a few iterations); a STANDALONE direct recon on such data is
        not quantitatively accurate -- prefer ``recon()`` there.  Generalizing this to a
        per-view weight is deliberately not done: the correct weight depends on assumptions
        the model cannot safely infer (full vs partial angular coverage; geometry redundancy),
        so changing it risks regressing the common equispaced case.

        Args:
            sinogram (jax array): (num_views, num_rows, num_channels); plain or view-sharded.
            filter_name (str): filter for generate_direct_recon_filter (e.g. 'ramp').
            filter_scale (float): geometry-specific filter scaling (FBP: 1/(dv*dvr);
                FDK: alpha = delta_det_row/(voxel_volume*M_0)).
            output_sharded (bool): True keeps the view-sharded device form; False gathers.
            row_weight (jax array or None): optional (rows, channels) FDK cosine pre-weight.

        Returns:
            filtered_sinogram: numpy by default, view-sharded device form if output_sharded.
        """
        num_channels = sinogram.shape[2]
        # Real view count from params (the array's view axis may be zero-padded for
        # sharding; padded views contribute nothing and must not be counted here).
        num_views = self.get_params('sinogram_shape')[0]
        recon_filter = tomography_utils.generate_direct_recon_filter(num_channels, filter_name=filter_name)
        # Fold both scalars into the (tiny) filter via a Python-float scale, which keeps
        # the f32 filter f32 (no out-of-place full-sino multiply, no f32->f64 promotion
        # that would double the sinogram's memory).  Matches ParallelBeamModel's fold.
        recon_filter = recon_filter * float(filter_scale * (np.pi / num_views))
        filter_np = np.asarray(recon_filter, dtype=np.float32)
        # The row weight enters the hot per-row multiply, so it MUST be f32 too -- an f64
        # weight would promote the window and cascade the whole sinogram to f64.
        row_weight_np = None if row_weight is None else np.asarray(row_weight, dtype=np.float32)

        # One thread per device, each filtering its own view shard locally (no cross-device data
        # movement).  Shard once at entry (no-op when already view-sharded) so the per-device map
        # sees every mesh device's shard (a single device is the trivial 1-shard case).
        sinogram = self._shard_sinogram(sinogram)
        dev_to_shard = {s.device: s.data for s in sinogram.addressable_shards}

        def worker_process(i, device):
            shard = dev_to_shard[device]
            filter_jax = jnp.array(filter_np)            # tiny upload: 2*channels-1 floats
            rw = None if row_weight_np is None else jnp.array(row_weight_np)
            return tomography_utils.apply_row_filter(shard, filter_jax, row_weight=rw)

        results = mjs.run_per_device(self.shard_devices, worker_process)
        filtered_sinogram = mjs.assemble_sharded(results, sinogram.shape, sinogram.sharding)

        if output_sharded:
            return filtered_sinogram                     # keep the device form
        return self._gather_sinogram(filtered_sinogram)  # default: numpy output

    def initialize_recon(self, sinogram, weights=None, init_recon=None, max_iterations=25, first_iteration=0,
                         compute_prior_loss=False, logfile_path='~/.mbirjax/logs/recon.log', print_logs=True):
        """
        Do the device management and parameter initialization needed for recon and prox_map.

        Args:
            See :meth:`recon` for arguments.

        Returns:
            sinogram, weights, init_recon, partitions, partition_sequence, granularity, regularization_params
        """

        # Initialize logging for this run
        self._log_run_header(first_iteration, logfile_path, print_logs)

        # Generate set of voxel partitions
        recon_shape, granularity, use_ror_mask = self.get_params(['recon_shape', 'granularity', 'use_ror_mask'])
        partitions = mj.gen_set_of_pixel_partitions(recon_shape, granularity, output_device=self.recon_placement.devices[0], use_ror_mask=use_ror_mask)

        # Generate sequence of partitions to use
        partition_sequence = self.get_params('partition_sequence')
        partition_sequence = mj.gen_partition_sequence(partition_sequence, max_iterations=max_iterations)
        partition_sequence = partition_sequence[first_iteration:]
        regularization_params = None

        try:
            # No input pre-placement here: vcd_recon's entry placement (to_sino / to_recon) shards
            # sinogram / weights / init_recon from wherever they reside (a prepared NamedSharding
            # passes through; a plain array is sharded), so a single-device device_put would only
            # add a redundant hop.

            # Test the sinogram contains valid data
            # Sometimes users accidentally create complex sinograms when they take the -log.
            # So we check for complex numbers or NaNs and raise an error.
            if np.iscomplexobj(sinogram):
                raise TypeError("sinogram must be real-valued; got complex dtype.")
            if not np.isfinite(sinogram).all():
                raise ValueError("sinogram contains NaN and/or Inf values.")

            # Test the weights contain valid data
            if weights is not None:
                # Test for NaNs and Inf values
                if not jnp.isfinite(weights).all():
                    raise ValueError("weights contains NaN and/or Inf values.")
                # Test the weights are non-negative
                if weights is not None and (weights < 0).any():
                    raise ValueError("weights contain negative values.")
                # Test the weights are not all zero
                if weights is not None and (weights == 0).all():
                    raise ValueError("all weights are zero.")

            # Run auto regularization. If auto_regularize_flag is False, then this will have no effect
            regularization_params = self.auto_set_regularization_params(sinogram, weights=weights)

            if compute_prior_loss:
                msg = 'Computing the prior loss on every iteration uses significant memory and computing power.\n'
                msg += 'Set compute_prior_loss=False for most applications aside from debugging and demos.'
                self.logger.warning(msg)

        except MemoryError as e:
            self.logger.error('Insufficient CPU memory')
            raise e
        except JaxRuntimeError as e:
            self._handle_jax_error(e)

        return sinogram, weights, init_recon, partitions, partition_sequence, granularity, regularization_params

    def _handle_jax_error(self, e):
        """
        Log a JAX runtime error, adding out-of-memory recovery guidance when applicable, then re-raise.

        Args:
            e (JaxRuntimeError): The error to handle

        Returns:
            Nothing, but re-raises the error.
        """
        self.logger.error(e)
        # Classify from the FULL traceback, not just str(e): an out-of-memory error often surfaces
        # as an unrelated-looking error with the real RESOURCE_EXHAUSTED buried deeper in the stack.
        if is_oom(traceback.format_exc()):
            # Derive on-GPU from the actual recon device platform, not the use_gpu REQUEST param:
            # the request does not say where the recon actually ran (e.g. explicit CPU sharding via
            # configure_devices under use_gpu='automatic'), and a CPU OOM must get CPU guidance.
            recon_devices = self.shard_devices
            on_gpu = bool(recon_devices) and recon_devices[0] is not None \
                and self._platform_label(recon_devices[0]) == 'GPU'
            log_oom_guidance(self.logger, on_gpu=on_gpu)
        raise e

    def recon(self, sinogram, weights=None, init_recon=None, max_iterations=25, stop_threshold_change_pct=0.2, first_iteration=0,
              compute_prior_loss=False, logfile_path='~/.mbirjax/logs/recon.log', print_logs=True, output_sharded=False):
        """
        Perform MBIR reconstruction using the Multi-Granular Vector Coordinate Descent algorithm.
        This function takes care of generating its own partitions and partition sequence.
        TO restart a recon using the same partition sequence, set first_iteration to be the number of iterations
        completed so far, and set init_recon to be the output of the previous recon.  This will continue using
        the same partition sequence from where the previous recon left off.

        Reproducibility note: the pixel partitions are drawn from numpy's global random number
        generator, so reconstructions vary slightly from run to run.  For a reproducible result,
        call ``np.random.seed(seed)`` before calling this method.

        Args:
            sinogram (numpy or jax array): 3D sinogram data with shape (num_views, num_det_rows, num_det_channels).
            weights (numpy or jax array, optional): 3D positive weights with same shape as error_sinogram.  Defaults to None, in which case the weights are implicitly all 1.
            init_recon (numpy or jax array, or None or 0, optional): Initial reconstruction to use in reconstruction. If None, then direct_recon is called with default arguments.  Defaults to None.
            max_iterations (int, optional): maximum number of iterations of the VCD algorithm to perform.
            stop_threshold_change_pct (float, optional): Stop reconstruction when 100 * ||delta_recon||_1 / ||recon||_1 change from one iteration to the next is below stop_threshold_change_pct.  Defaults to 0.2.  Set this to 0 to guarantee exactly max_iterations.
            first_iteration (int, optional): Set this to be the number of iterations previously completed when restarting a recon using init_recon.  This defines the first index in the partition sequence.  Defaults to 0.
            compute_prior_loss (bool, optional):  Set true to calculate and return the prior model loss.  This will lead to slower reconstructions and is meant only for small recons.
            logfile_path (str, optional): Path to the output log file ('~' expands to the user's
                home directory).  Defaults to '~/.mbirjax/logs/recon.log'.  Set to None or '' to
                skip file logging.
            print_logs (bool, optional): If true then print logs to console.  Defaults to True.
            output_sharded (bool, optional): If False (default), return a numpy reconstruction array.
                If True, return the internal device form (slice-sharded across the model's devices,
                no exit gather); on an unsharded model the output is the same either way.

        Returns:
            (recon, recon_dict): reconstruction array and a dict containing the recon parameters.
                - recon (numpy or jax array): the reconstruction volume
                - recon_dict (dict): A dict obtained from :meth:`get_recon_dict` with entries
                    * 'recon_params'
                    * 'notes'
                    * 'recon_logs'
                    * 'model_params'
        """
        sinogram, weights, init_recon, partitions, partition_sequence, granularity, regularization_params = (
            self.initialize_recon(sinogram, weights, init_recon, max_iterations, first_iteration,
                                  compute_prior_loss, logfile_path, print_logs))
        try:
            # Compute reconstruction
            recon, loss_vectors = self.vcd_recon(sinogram, partitions, partition_sequence, weights=weights,
                                                 init_recon=init_recon, compute_prior_loss=compute_prior_loss,
                                                 first_iteration=first_iteration,
                                                 stop_threshold_change_pct=stop_threshold_change_pct)

            # Return num_iterations, granularity, partition_sequence, fm_rmse values, regularization_params
            partition_sequence = [int(val) for val in partition_sequence]
            fm_rmse = [float(val) for val in loss_vectors[0]]
            if compute_prior_loss:
                prior_loss = [float(val) for val in loss_vectors[1]]
            else:
                prior_loss = [0]
            stop_threshold_change_pct = [100 * float(val) for val in loss_vectors[2]]
            alpha_values = [float(val) for val in loss_vectors[3]]
            num_iterations = len(fm_rmse)
            recon_param_values = [num_iterations, granularity, partition_sequence, fm_rmse, prior_loss,
                                  regularization_params, stop_threshold_change_pct, alpha_values]
            recon_params = ReconParams(*tuple(recon_param_values))._asdict()

        except MemoryError as e:
            self.logger.error('Insufficient CPU memory')
            raise e
        except JaxRuntimeError as e:
            self._handle_jax_error(e)

        if logfile_path:
            self.logger.info('Logs written to {}'.format(
                os.path.abspath(os.path.expanduser(logfile_path))))

        for h in list(self.logger.handlers):  # Make sure the log files are up to date
            h.flush()

        notes = 'Reconstruction completed: {}\n\n'.format(datetime.datetime.now())
        recon_dict = self.get_recon_dict(recon_params, notes=notes)
        if not output_sharded:
            # Default exit: gather to a HOST NumPy array in the problem's REAL shape
            # (_gather_recon assembles on the host and crops any padded slices) -- never re-materialized
            # on a single device, so this is safe at any size.
            recon = self._gather_recon(recon)
        return recon, recon_dict

    def vcd_recon(self, sinogram, partitions, partition_sequence, stop_threshold_change_pct, weights=None,
                  init_recon=None, prox_input=None, compute_prior_loss=False, first_iteration=0,
                  init_error_sinogram=None, fm_hessian=None, return_checkpoint=False):
        """
        Perform MBIR reconstruction using the Multi-Granular Vector Coordinate Descent algorithm
        for a given set of partitions and a prescribed partition sequence.

        Args:
            sinogram (jax array): 3D sinogram data with shape (num_views, num_det_rows, num_det_channels).
            partitions (tuple or list): A collection of K partitions, with each partition being an (N_indices) integer index array of voxels to be updated in a flattened recon.
            partition_sequence (jax array): A sequence of integers that specify which partition should be used at each iteration.
            stop_threshold_change_pct (float): Stop reconstruction when NMAE percent change from one iteration to the next is below stop_threshold_change_pct.
            weights (jax array, optional): 3D positive weights with same shape as error_sinogram.  Defaults to all 1s.
            init_recon (jax array or None or 0, optional): Initial reconstruction to use in reconstruction. If None, then direct_recon is called with default arguments.  Defaults to None.
            prox_input (jax array, optional): Reconstruction to be used as input to a proximal map.
            compute_prior_loss (bool, optional):  Set true to calculate and return the prior model loss.
            first_iteration (int, optional): Set this to be the number of iterations previously completed when restarting a recon using init_recon.
            init_error_sinogram (jax array, optional): Precomputed error sinogram to resume from,
                skipping the initializing forward projection.  Must be supplied together with
                init_recon, and the pair is TRUSTED as consistent (init_error_sinogram ==
                sinogram - A @ init_recon for the SAME sinogram and geometry) -- verifying would
                cost the forward projection this argument exists to avoid.  The device-form array
                returned via return_checkpoint satisfies this by construction.  Defaults to None.
            fm_hessian (jax array, optional): Precomputed forward-model Hessian diagonal (as
                returned via return_checkpoint, or compute_hessian_diagonal(weights=weights,
                output_sharded=True) flattened to (num_pixels, num_slices)).  Must correspond to
                the SAME weights and geometry.  When None (default), it is computed internally.
            return_checkpoint (bool, optional): If True, additionally return the resume state --
                a dict {'error_sinogram': <device form>, 'fm_hessian': <device form>} suitable for
                the two arguments above -- so a chunked/checkpointed run can continue with no
                re-initialization cost.  Zero-copy: the dict references the loop's own device
                arrays.  Defaults to False.
        Returns:
            (recon, recon_stats): tuple of 3D reconstruction and a tuple containing arrays of per-iteration stats.
            With return_checkpoint=True: (recon, recon_stats, checkpoint).
            recon_stats = (fm_rmse, pm_rmse, nrms_update), where fm is forward model, pm is prior model, and
            nrms_update is ||recon(i+1) - recon(i)||_2 / ||recon(i+1)||_2.

        Note:
            To maximize GPU memory, each of sinogram, weights, init_recon, and prox_input should be on the CPU for large recons.
        """
        # Ensure that everything has the right shape and is on the main device
        self.verify_valid_params()

        # Placement helpers: recon-like arrays are slice-sharded and sino-like arrays are
        # view-sharded (a single device is the trivial 1-shard case).  Routing every placement
        # through these keeps the rest of the loop placement-agnostic and (crucially) avoids
        # committing a sharded array to a single device, which would silently gather it.
        def to_sino(x):
            return self._shard_sinogram(x)

        def to_recon(x):
            return self._shard_recon(x)

        if weights is None:
            weights = 1
            constant_weights = True
        else:
            weights = to_sino(weights)
            constant_weights = False

        recon_shape = self.get_params('recon_shape')
        num_recon_slices = recon_shape[2]

        # Place the sinogram (view-sharded when sharding is on) so direct_recon yields a matching
        # slice-sharded init and the alpha dot-products below stay aligned with the error sinogram.
        # Decide whether we may free the placed sinogram once consumed (below).  We own FRESH
        # buffers only when to_sino had to transfer the data here: a host/numpy input, or a jax
        # array on OTHER devices, forces a copy to new buffers on the sino devices.  An input
        # already resident on these devices may be returned by to_sino as a no-copy reshard that
        # SHARES the caller's buffers (or unchanged, e.g. prepare_sino_for_devices) -- deleting
        # that would invalidate the caller's array, so leave it.
        if isinstance(sinogram, jax.Array):
            own_sinogram = set(sinogram.devices()).isdisjoint(self.sino_placement.devices)
        else:
            own_sinogram = True   # numpy/host input -> to_sino copies to device buffers we own
        sinogram = to_sino(sinogram)

        scale_recon_to_sinogram = True if init_recon is None else False
        if init_error_sinogram is not None and init_recon is None:
            raise ValueError('init_error_sinogram requires init_recon (the pair must be a '
                             'consistent resume state; see the docstring).')
        if init_recon is None:
            # Initialize VCD recon, and error sinogram.  output_sharded=True keeps the init in
            # the internal device form (slice-sharded when sharding is on; no gather).
            self.logger.info('Starting direct recon for initial reconstruction')
            init_recon = self.direct_recon(sinogram, output_sharded=True)
        elif isinstance(init_recon, int):
            # A constant init recon built directly in the slice-sharded device form (no full
            # volume is materialized on one device before sharding).
            init_recon = mjs.sharded_full(self.recon_placement, tuple(recon_shape), float(init_recon))

        # Make sure that init_recon has the correct shape and type: the problem's REAL
        # shape (a user-supplied init) or the device form (direct_recon output_sharded=True,
        # whose slice axis may be padded for sharding).
        if tuple(init_recon.shape) not in (tuple(recon_shape), self._recon_device_shape()):
            error_message = "init_recon does not have the correct shape. \n"
            error_message += "Expected {}, but got shape {} for init_recon shape.".format(recon_shape,
                                                                                          init_recon.shape)
            self.logger.error(error_message)
            raise ValueError(error_message)

        # Place the init as a recon-like array (slice-sharded when sharding is on); a no-op when
        # direct_recon already returned it sharded.
        init_recon = to_recon(init_recon)

        # Initialize VCD recon and error sinogram using the init_recon
        if init_error_sinogram is not None:
            # RESUME path: the caller supplies error_sinogram = sinogram - A @ init_recon
            # (a trusted, consistent pair -- typically the return_checkpoint output of a
            # previous call), skipping the initializing forward projection.
            self.logger.info('Resuming from a supplied error sinogram')
            error_sinogram = init_error_sinogram
        else:
            # We find the optimal alpha to minimize (1/2)||y - alpha Ax||_weights^2, where y is the sinogram and x is init_recon
            # output_sharded=True keeps the error sinogram in the device form (view-sharded when
            # sharding is on; no gather between the forward projection and the loop).
            self.logger.info('Initializing error sinogram')
            error_sinogram = self.forward_project(init_recon, output_sharded=True)
            if not constant_weights:
                weighted_error_sinogram = weights * error_sinogram  # Note that fm_constant will be included below
            else:
                weighted_error_sinogram = error_sinogram
            wtd_err_sino_norm = jnp.sum(weighted_error_sinogram * error_sinogram)
            if wtd_err_sino_norm > 0 and scale_recon_to_sinogram:
                alpha = jnp.sum(weighted_error_sinogram * sinogram) / wtd_err_sino_norm
                alpha = alpha.item()
            else:
                alpha = 1

            error_sinogram = sinogram - alpha * error_sinogram
            init_recon = alpha * init_recon

        recon = init_recon
        recon = to_recon(recon)  # slice-shard (a single device is the trivial 1-shard case)
        error_sinogram = to_sino(error_sinogram)

        # sinogram's contents are now fully folded into error_sinogram (above); its only remaining
        # use is a dtype read for the constant-weights ones array, which error_sinogram serves.  Free
        # the view-sharded sinogram now -- at 2048^3/8 that reclaims ~4 GiB/shard before the Hessian
        # diagonal and the VCD loop -- but ONLY when we own it (to_sino copied the input).  A caller
        # who pre-sharded via prepare_sino_for_devices still owns their array, so we must not delete.
        # block_until_ready makes the delete race-free (error_sinogram has finished reading sinogram);
        # it is a one-time init sync, not in the hot subset loop.
        if own_sinogram:
            jax.block_until_ready(error_sinogram)
            sinogram.delete()
        sinogram = None

        # Test to make sure the prox_input input is correct
        if prox_input is not None:
            # Accept the problem's REAL shape (the normal user input) or the device form
            # (a prox chain feeding back an output_sharded result, slice axis possibly padded).
            if tuple(prox_input.shape) not in (tuple(recon_shape), self._recon_device_shape()):
                error_message = "prox_input does not have the correct size. \n"
                error_message += "Expected {}, but got shape {} for prox_input shape.".format(recon_shape,
                                                                                              prox_input.shape)
                self.logger.error(error_message)
                raise ValueError(error_message)

            # Flatten (keeping the array's own slice count -- real or device form), then
            # place like every other recon-domain array (slice-sharded; the entry placement
            # zero-pads a real-shape input).
            # flat_recon is slice-sharded in the sharded path and the prox gradient is an
            # elementwise difference of the two, so committing prox_input to a single
            # device would hand the jitted prox gradient arrays on incompatible devices.
            prox_input = to_recon(prox_input.reshape((-1, prox_input.shape[-1])))

        # Get required parameters
        verbose, sigma_y = self.get_params(['verbose', 'sigma_y'])

        # The REAL sinogram element count (from the params, which always hold the problem's
        # shapes).  Equals the device arrays' size except when the view axis and/or the
        # detector-row axis is padded for sharding; normalizing by the real count keeps the
        # reported losses independent of the (inert, identically-zero) padding.
        # math.prod (exact Python ints), NOT np.prod: numpy's product accumulates in the platform
        # default integer -- int32 on Windows/numpy<2 -- and a full-size sinogram (>2^31 elements)
        # would silently wrap.
        real_sino_size = math.prod(self.get_params('sinogram_shape'))
        pad_active = (self.sino_placement.is_padded
                      or self._sino_row_padding() is not None)
        loss_num_real = real_sino_size if pad_active else None

        # Initialize the diagonal of the hessian of the forward model (or accept a
        # precomputed one -- as returned via return_checkpoint -- and skip the back
        # projection that computes it).
        if fm_hessian is None:
            if constant_weights:
                # Ones over the real views, ZEROS over any padded views (device form):
                # padded views must not contribute to the Hessian back projection.
                # _sino_ones_device_form uses only its argument's dtype; error_sinogram (same dtype,
                # same device form) stands in so the original sinogram can be freed above.
                weights = self._sino_ones_device_form(error_sinogram)

            self.logger.info('Computing Hessian diagonal')
            # output_sharded=True keeps the Hessian in the device form (slice-sharded, slice
            # axis possibly padded -- the padded entries are zero, masked by the back projection).
            fm_hessian = self.compute_hessian_diagonal(weights=weights, output_sharded=True)
            # Flatten keeping each array's OWN slice count (the device form may carry a
            # padded slice axis; num_recon_slices is the problem's real count).
            fm_hessian = fm_hessian.reshape((-1, fm_hessian.shape[-1]))
        else:
            self.logger.info('Using precomputed Hessian diagonal')
            # Accept the flat device form (num_pixels, device slices) that return_checkpoint
            # hands back; to_recon places a plain host array (a no-op on a placed one).
            fm_hessian = to_recon(fm_hessian)
        if constant_weights:
            weights = 1
        else:
            weights = to_sino(weights)

        # Initialize the emtpy recon
        flat_recon = recon.reshape((-1, recon.shape[-1]))
        flat_recon = to_recon(flat_recon)

        # Create the finer grained recon update operators
        vcd_subset_updater = self.create_vcd_subset_updater(fm_hessian, weights=weights, prox_input=prox_input)

        self.logger.info('Starting VCD iterations')
        if verbose >= 2:
            output = io.StringIO()
            mj.get_memory_stats(file=output)
            self.logger.debug(output.getvalue())
            self.logger.debug('--------')

        # Do the iterations
        max_iters = partition_sequence.size
        fm_rmse = np.zeros(max_iters)
        pm_loss = np.zeros(max_iters)
        nmae_update = np.zeros(max_iters)
        alpha_values = np.zeros(max_iters)
        num_iters = 0
        for i in range(max_iters):
            # Get the current partition (set of subsets) and shuffle the subsets
            partition = partitions[partition_sequence[i]]

            # Do an iteration
            flat_recon, error_sinogram, ell1_for_partition, alpha = self.vcd_partition_iterator(vcd_subset_updater,
                                                                                                 flat_recon,
                                                                                                 error_sinogram,
                                                                                                 partition)

            # Compute the stats and display as desired -- one fused pass over the error sinogram
            # (see _vcd_iteration_stats).  real_sino_size == error_sinogram.size except under view
            # padding, where the padded entries are identically zero and must not dilute the RMSE.
            fm_loss_i, recon_l1, es_rmse = self._vcd_iteration_stats(
                error_sinogram, flat_recon, sigma_y, weights,
                num_real_elements=loss_num_real, real_sino_size=float(real_sino_size))
            fm_rmse[i] = fm_loss_i
            nmae_update[i] = ell1_for_partition / recon_l1
            alpha_values[i] = alpha

            if verbose >= 1:
                iter_output = '\nAfter iteration {} of a max of {}: Pct change={:.4f}, Forward loss={:.4f}'.format(i + first_iteration, max_iters + first_iteration,
                                                                                                  100 * nmae_update[i],
                                                                                                  fm_rmse[i])
                if compute_prior_loss:
                    qggmrf_nbr_wts, sigma_x, p, q, T = self.get_params(['qggmrf_nbr_wts', 'sigma_x', 'p', 'q', 'T'])
                    b = mj.get_b_from_nbr_wts(qggmrf_nbr_wts)
                    qggmrf_params = (b, sigma_x, p, q, T)
                    # Evaluate the prior loss on the REAL volume: _gather_recon crops any
                    # padded slices (whose zero values would otherwise add spurious
                    # boundary-difference terms to the loss).  Debug/verbose path only.
                    real_recon_size = math.prod(recon_shape)   # exact Python ints (see real_sino_size)
                    loss_recon = self._gather_recon(flat_recon).reshape(recon_shape)
                    pm_loss[i] = mj.qggmrf_loss(loss_recon, qggmrf_params)
                    pm_loss[i] /= real_recon_size
                    # Each loss is scaled by the number of elements, but the optimization uses unscaled values.
                    # To provide an accurate, yet properly scaled total loss, first remove the scaling and add,
                    # then scale by the average number of elements between the two.
                    total_loss = ((fm_rmse[i] * real_sino_size + pm_loss[i] * real_recon_size) /
                                  (0.5 * (real_sino_size + real_recon_size)))
                    iter_output += ', Prior loss={:.4f}, Weighted total loss={:.4f}'.format(pm_loss[i], total_loss)

                self.logger.info(iter_output)
                self.logger.info(f'Relative step size (alpha)={alpha:.2f}, Error sino RMSE={es_rmse:.4f}')
                self.logger.info('Number subsets = {}'.format(partition.shape[0]))
                if verbose >= 2:
                    output = io.StringIO()
                    mj.get_memory_stats(file=output)
                    self.logger.debug(output.getvalue())
                    self.logger.debug('--------')
            num_iters += 1
            if nmae_update[i] < stop_threshold_change_pct / 100:
                self.logger.warning('Change threshold stopping condition reached')
                break

        # Reshape to 3-D keeping the array's OWN slice count (the device form may carry a
        # padded slice axis -- the caller's exit handling gathers + crops to the real shape).
        recon_3d = flat_recon.reshape(tuple(recon_shape[:2]) + (flat_recon.shape[-1],))
        losses = (fm_rmse[0:num_iters], pm_loss[0:num_iters], nmae_update[0:num_iters],
                  alpha_values[0:num_iters])
        if return_checkpoint:
            # Zero-copy resume state: references to the loop's own device-form arrays.  Feed
            # these back as init_error_sinogram / fm_hessian (with init_recon = the returned
            # recon) to continue with no re-initialization cost.
            return recon_3d, losses, {'error_sinogram': error_sinogram, 'fm_hessian': fm_hessian}
        return recon_3d, losses

    def vcd_partition_iterator(self, vcd_subset_updater, flat_recon, error_sinogram, partition):
        """
        Calculate a full iteration of the VCD algorithm by scanning over the subsets of the partition.
        Each iteration of the algorithm should return a better reconstructed recon.
        The error_sinogram should always be:  error_sinogram = measured_sinogram - forward_proj(recon)
        where measured_sinogram is the measured sinogram and recon is the current reconstruction.

        Args:
            vcd_subset_updater (callable): Function to iterate over each subset in the partition.
            flat_recon (jax array): 2D array reconstruction with shape (num_recon_rows x num_recon_cols, num_recon_slices).
            error_sinogram (jax array): 3D error sinogram with shape (num_views, num_det_rows, num_det_channels).
            partition (jax array): 2D array where partition[subset_index] gives a 1D array of pixel indices.

        Returns:
            (flat_recon, error_sinogram, ell1_for_partition, alpha): The first two have the same shape as above, but
            are updated to reduce overall loss function.
            The ell1_for_partition includes the changes from all subsets of this partition.
            alpha is the relative step size in the gradient descent step, averaged over the subsets
            in the partition.
        """

        # Loop over the subsets of the partition, using random subset_indices to order them.
        ell1_for_partition = 0
        alpha_sum = 0
        subset_indices = np.random.permutation(partition.shape[0])

        times = np.zeros(13)
        # np.set_printoptions(precision=1, floatmode='fixed', suppress=True)
        # Stage the qGGMRF boundary halos ONCE for this whole partition pass and reuse them
        # for every subset.  The prior couples a voxel only to its same-pixel cross-shard
        # slice neighbor, and the partition's subsets are (almost) disjoint, so a subset's
        # halo at its own pixels is unchanged until that subset runs -- hence pass-start
        # halos are correct for each subset.  This turns the per-subset host halo read (the
        # main per-subset host round-trip, which caps multi-GPU scaling) into a per-pass one.
        # (Caveat: gen_pixel_partition replicates a few pixels to equalize subset lengths, so
        # this is not strictly bit-exact at those pixels -- quantified by test; negligible.
        # Set self._vcd_halo_per_subset = True to restore per-subset extraction for A/B.)
        stage_per_pass = not getattr(self, '_vcd_halo_per_subset', False)
        staged_halos = self._stage_halos(flat_recon) if stage_per_pass else None
        for index in subset_indices:
            subset = partition[index]
            flat_recon, error_sinogram, ell1_for_subset, alpha_for_subset = vcd_subset_updater(
                flat_recon, error_sinogram, subset, staged_halos)
            ell1_for_partition += ell1_for_subset
            alpha_sum += alpha_for_subset

        return flat_recon, error_sinogram, ell1_for_partition, alpha_sum / partition.shape[0]

    def create_vcd_subset_updater(self, fm_hessian, weights, prox_input=None):
        """
        Create a jit-compiled function to update a subset of pixels in the recon and error sinogram.

        Args:
            fm_hessian (jax array): Array with same shape as recon containing diagonal of hessian for forward model loss.
            weights (jax array, optional): 3D positive weights with same shape as sinogram.  Defaults to all 1s.
            prox_input (jax array): optional input for proximal map with same shape as reconstruction.

        Returns:
            (callable) vcd_subset_updater(error_sinogram, flat_recon, pixel_indices) that updates the recon.
        """

        positivity_flag = self.get_params('positivity_flag')
        fm_constant = 1.0 / (self.get_params('sigma_y') ** 2.0)
        qggmrf_nbr_wts, sigma_x, p, q, T = self.get_params(['qggmrf_nbr_wts', 'sigma_x', 'p', 'q', 'T'])
        b = mj.get_b_from_nbr_wts(qggmrf_nbr_wts)
        qggmrf_params = tuple((b, sigma_x, p, q, T))
        sigma_prox = self.get_params('sigma_prox')
        recon_shape = self.get_params('recon_shape')
        max_alpha = self.get_params('max_overrelaxation')
        sparse_back_project = self.sparse_back_project
        sparse_forward_project = self.sparse_forward_project
        try:
            const_weights = False
            sinogram_shape = self.get_params('sinogram_shape')
            # Accept the problem shape (plain weights) or the device-form shape (weights
            # already placed by vcd_recon, with a possibly padded view axis).
            if tuple(weights.shape) not in (tuple(sinogram_shape), self._sino_device_shape()):
                raise ValueError('weights must be a constant or have the same shape as sinogram.')
        except AttributeError:
            eps = 1e-5
            if np.abs(weights - 1) > eps:
                raise ValueError('Constant weights must have value 1.')
            const_weights = True

        def vcd_subset_updater(flat_recon, error_sinogram, pixel_indices, staged_halos=None):
            """
            Calculate an iteration of the VCD algorithm on a single subset of the partition
            Each iteration of the algorithm should return a better reconstructed recon.
            The combination of (error_sinogram, recon) forms an overcomplete state that makes computation efficient.
            However, it is important that at each application the state should meet the constraint that:
            error_sinogram = measured_sinogram - forward_proj(recon)
            where measured_sinogram forward_proj() is whatever forward projection is being used in reconstruction.

            Args:
                flat_recon (jax array): 2D array reconstruction with shape (num_recon_rows x num_recon_cols, num_recon_slices).
                error_sinogram (jax array): 3D error sinogram with shape (num_views, num_det_rows, num_det_channels).
                pixel_indices (jax array): 1D array of pixel indices.
                staged_halos (tuple or None): ``(staged_left, staged_right)`` qGGMRF boundary
                    halos staged once per partition pass (see :meth:`_stage_halos`); forwarded
                    to the sharded prior so the halos are not re-read every subset.  ``None``
                    in the single-device path or when per-subset extraction is forced.

            Returns:
                flat_recon, error_sinogram, ell1_for_subset, alpha_for_subset:
                The first two have the same shape as above, but are updated to reduce the overall loss function.
                ell1_for_subset is for the change to the recon from this one subset.
                alpha is the relative step size for this subset.
            """

            # Compute the forward model gradient and hessian at each pixel in the index set.
            # Assumes Loss(delta) = 1/(2 sigma_y^2) || error_sinogram - A delta ||_weights^2

            # Recon-domain gathers/scatters below (fm_hessian[...], flat_recon[...], update_recon)
            # index the *unsharded* pixel axis of a slice-sharded array.  For the gather to be valid
            # the index array must live on the same devices as the array, so replicate the indices
            # across the recon mesh (PartitionSpec() == fully replicated; a single device is the
            # trivial 1-shard case).
            recon_indices = jax.device_put(
                pixel_indices,
                jax.sharding.NamedSharding(self.recon_placement.mesh,
                                           jax.sharding.PartitionSpec()))

            # Compute the prior model gradient and hessian (i.e., second derivative) terms
            if prox_input is None:

                # qGGMRF prior - compute the qggmrf gradient and hessian at each pixel in the index set.
                # flat_recon is slice-sharded, so compute the prior per slice-owner with halo
                # exchange for the inter-slice term, keeping the result slice-sharded.  staged_halos
                # (staged once per partition pass) avoids re-reading the halos here.
                prior_grad, prior_hess = self._qggmrf_prior_sharded(
                    flat_recon, pixel_indices, qggmrf_params, staged_halos=staged_halos)
            else:
                # Proximal map prior - compute the prior model gradient at each pixel in the index set.
                # The prox prior is pointwise (no inter-slice coupling, so no halos); recon_indices
                # (replicated on the recon mesh in the sharded path) makes the pixel-axis gather
                # local to each slice-shard, exactly like the fm_hessian/flat_recon gathers below.
                prior_hess = 1 / (sigma_prox ** 2)
                prior_grad = mj.prox_gradient_at_indices(flat_recon, prox_input, recon_indices, sigma_prox)

            if not const_weights:
                weighted_error_sinogram = weights * error_sinogram  # Note that fm_constant will be included below
            else:
                weighted_error_sinogram = error_sinogram

            # Back project to get the gradient
            forward_grad = - fm_constant * sparse_back_project(weighted_error_sinogram, pixel_indices)

            # Get the forward hessian for this subset
            forward_hess = fm_constant * fm_hessian[recon_indices]

            # Compute update vector update direction in recon domain
            delta_recon_at_indices = - ((forward_grad + prior_grad) / (forward_hess + prior_hess))

            # Compute delta^T \nabla Q(x_hat; x'=x_hat) for use in finding alpha
            prior_linear = jnp.sum(prior_grad * delta_recon_at_indices)

            # Estimated upper bound for hessian
            prior_overrelaxation_factor = 1.0
            prior_quadratic_approx = ((1 / prior_overrelaxation_factor) *
                                      jnp.sum(prior_hess * delta_recon_at_indices ** 2))

            # Free the (now-dead) gradient/Hessian buffers BEFORE the memory-heavy forward projection.
            # forward_grad/forward_hess were last read by delta_recon_at_indices; prior_grad/prior_hess by
            # prior_linear/prior_quadratic_approx just above.  In eager mode a queued op pins its inputs, so
            # blocking on those two scalars (which transitively force delta_recon_at_indices, hence the
            # forward_* reads) guarantees all four are consumed; the del then releases them -- ~4 subset-
            # sized arrays (~11.6 GB at the coarse 1024^3 subset) -- so sparse_forward_project's ~5x-volume
            # transient has room.  Compute-free on a compute-bound GPU: the host was running ahead anyway
            # and the projection depends on delta_recon_at_indices regardless.  (A no-op if ever jitted.)
            jax.block_until_ready((prior_linear, prior_quadratic_approx))
            del forward_grad, prior_grad, forward_hess, prior_hess

            # Compute update direction in sinogram domain
            delta_sinogram = sparse_forward_project(delta_recon_at_indices, pixel_indices)

            forward_linear, forward_quadratic = self.get_forward_lin_quad(
                weighted_error_sinogram, delta_sinogram, weights, fm_constant, const_weights)

            # Compute optimal update step.
            # The forward-model line-search scalars are reduced over the view-sharded sinogram (the
            # sino mesh) while the prior scalars are reduced over the slice-sharded recon (the recon
            # mesh); combining device scalars across two distinct meshes is not allowed.  Rather than
            # bounce all four through the host (5 device->host syncs/subset that stall the GPU), keep
            # the line search ON-DEVICE: replicate the forward scalars onto the recon mesh (a cheap
            # scalar reshard over the same devices), do the arithmetic there, and replicate the
            # resulting alpha onto the sino mesh below to scale the sino-sharded delta.
            forward_linear = self._replicate_scalar(forward_linear, self.recon_placement)
            forward_quadratic = self._replicate_scalar(forward_quadratic, self.recon_placement)
            # prior_linear / prior_quadratic_approx are already replicated on the recon mesh.
            alpha_numerator = forward_linear - prior_linear
            alpha_denominator = forward_quadratic + prior_quadratic_approx + jnp.finfo(jnp.float32).eps
            alpha = alpha_numerator / alpha_denominator
            alpha = jnp.clip(alpha, jnp.finfo(jnp.float32).eps, max_alpha)

            # Enforce positivity constraint if desired
            # Greg, this may result in excess compilation. Not sure.
            if positivity_flag is True:
                # Get recon at index_batch
                recon_at_indices = flat_recon[recon_indices]

                # Clip updates to ensure non-negativity
                pos_constant = 1.0 / (alpha + jnp.finfo(jnp.float32).eps)
                # delta_recon_at_indices is already slice-sharded to match flat_recon, so this
                # stays a local elementwise op (no gather).
                delta_recon_at_indices = jnp.maximum(-pos_constant * recon_at_indices, delta_recon_at_indices)

                # Recompute sinogram projection
                delta_sinogram = sparse_forward_project(delta_recon_at_indices, pixel_indices)

            # Perform sparse updates at index locations.  delta_recon_at_indices is already
            # slice-sharded to match flat_recon, so update_recon stays a local scatter.
            delta_recon_at_indices = alpha * delta_recon_at_indices

            flat_recon = update_recon(flat_recon, recon_indices, delta_recon_at_indices)

            # Update sinogram and loss: error_sinogram <- error_sinogram - alpha * delta_sinogram,
            # IN PLACE via a buffer-DONATING fused multiply-add (alpha replicated onto the sino mesh
            # so the scale stays on-device).  In-place reuse is required because an out-of-place
            # per-subset update allocates a fresh view-sharded error sinogram each subset, and the
            # stale ones accumulate in jax's internal sharded-array reference cycles until gc.  For
            # constant weights weighted_error_sinogram IS error_sinogram, so release that alias to
            # make error_sinogram donatable.  (Non-constant weights leave a weighted product
            # transient that is freed in the cleanup section at the end of the subset.)
            if const_weights:
                weighted_error_sinogram = None
            error_sinogram = update_error_sinogram(
                error_sinogram, self._replicate_scalar(alpha, self.sino_placement), delta_sinogram)

            ell1_for_subset = jnp.sum(jnp.abs(delta_recon_at_indices))
            alpha_for_subset = alpha

            # === Release this subset's transient sharded buffers (single cleanup site) ===
            # When weights are non-constant, the weights*error_sinogram product is a view-sharded
            # sinogram that jax holds in an internal reference cycle, so refcounting never frees it
            # and it would pile up one per subset until gc (the multi-device memory blowup).  Free
            # it explicitly here.  This is race-free: it is consumed only upstream of the returned
            # (flat_recon, error_sinogram), so once those are ready it is finished being read.
            # (The alpha*delta scale is now fused into the donated update_error_sinogram jit, so
            # there is no separate scaled_delta transient; forward-projection outputs come from
            # assemble_sharded and DO free on refcount, so neither is freed here.)
            if not const_weights:
                jax.block_until_ready((flat_recon, error_sinogram))
                weighted_error_sinogram.delete()

            return flat_recon, error_sinogram, ell1_for_subset, alpha_for_subset

        return vcd_subset_updater

    def get_forward_lin_quad(self, weighted_error_sinogram, delta_sinogram, weights, fm_constant, const_weights):
        """
        Compute forward model terms used in line-search updates:
        ``forward_linear = fm_constant * jnp.sum(weighted_error_sinogram * delta_sinogram)`` and
        ``forward_quadratic = fm_constant * jnp.sum(delta_sinogram * delta_sinogram * weights)``.
        This supports batching to a worker, with only two floats returned per batch.

        The two scalars are left wherever the reductions produced them (the sino mesh in the
        sharded path); the caller reconciles them onto the recon mesh.

        Args:
            weighted_error_sinogram (jax array):
            delta_sinogram (jax array):
            weights (jax array or constant):
            fm_constant (constant):
            const_weights (bool): True if the weights are constant 1

        Returns:
            tuple: ``(forward_linear, forward_quadratic)``
        """
        forward_linear = fm_constant * jnp.sum(weighted_error_sinogram * delta_sinogram)
        forward_quadratic = fm_constant * jnp.sum(delta_sinogram * delta_sinogram * weights)
        return forward_linear, forward_quadratic

    @staticmethod
    @functools.partial(jax.jit, static_argnums=(3, 4), static_argnames=('normalize', 'num_real_elements'))
    def get_forward_model_loss(error_sinogram, sigma_y, weights=None, normalize=True,
                               num_real_elements=None):
        """
        Calculate the loss function for the forward model from the error_sinogram and weights.
        The error sinogram should be error_sinogram = measured_sinogram - forward_proj(recon)

        This is called once per VCD iteration on the full (possibly sharded) error sinogram, so it is
        jitted: XLA fuses the elementwise products into the reductions, so no sinogram-sized
        temporaries are materialized (eagerly, ``e*e``, ``weights/avg`` and their product each
        allocated a full sinogram per call).  Scalar factors (``avg_weight``, the element count,
        ``sigma_y``) divide the reduced SUM rather than the full-size array -- algebraically identical.

        Args:
            error_sinogram (jax array): 3D error sinogram with shape (num_views, num_det_rows, num_det_channels).
            sigma_y (float): Estimate obtained from auto_set_sigma_y or get_params('sigma_y')
            weights (jax array, optional): 3D positive weights with same shape as sinogram.  Defaults to all 1s.
            normalize (bool, optional, default=True):  If true, then
            num_real_elements (int, optional): the number of REAL sinogram elements, when
                error_sinogram carries extra zero-filled padding (a padded view axis under
                sharding).  The padded entries contribute nothing to the sums, so normalizing
                by the real count gives exactly the unpadded loss.  Default None uses
                error_sinogram.size (the unpadded case).

        Returns:
            float loss.
        """
        # Element counts enter the computation as FLOATS: a Python int operand is converted to int32
        # by jax (x64 disabled), which overflows for sinograms above 2^31 elements (~2.1e9 -- a
        # full-size half sino exceeds this).  float conversion carries the same ~1e-7 rounding that
        # jnp.mean's internal count already had.
        if weights is None:
            weights = 1
            avg_weight = 1
        elif num_real_elements is None:
            avg_weight = jnp.average(weights)
        else:
            avg_weight = jnp.sum(weights) / float(num_real_elements)
        if normalize:
            weighted_sq_sum = jnp.sum(error_sinogram * error_sinogram * weights)
            denom = float(error_sinogram.size) if num_real_elements is None else float(num_real_elements)
            loss = jnp.sqrt(weighted_sq_sum / (avg_weight * denom)) / sigma_y
        else:
            loss = (1.0 / (2 * sigma_y ** 2)) * jnp.sum((error_sinogram * error_sinogram) * weights)
        return loss

    @staticmethod
    @functools.partial(jax.jit, static_argnames=('num_real_elements',))
    def _vcd_iteration_stats(error_sinogram, flat_recon, sigma_y, weights=None, num_real_elements=None,
                             real_sino_size=None):
        """Per-iteration VCD logging stats in ONE fused, jitted pass.

        Returns ``(forward_model_loss, recon_l1, error_sino_rmse)`` as device scalars.  Eagerly, the
        loss, the recon L1, and ``jnp.linalg.norm(error_sinogram)`` each materialized full-size
        temporaries and -- on a SHARDED error sinogram -- each separate eager op also carried its own
        cross-device reduction with its own collective buffers.  One jitted function means one
        executable, one set of collective allocations, and no full-size temporaries.

        ``real_sino_size`` is the REAL element count for the error-sino RMSE (see the caller: padded
        entries are identically zero and must not dilute the RMSE).
        """
        fm_loss = TomographyModel.get_forward_model_loss(error_sinogram, sigma_y, weights,
                                                         num_real_elements=num_real_elements)
        recon_l1 = jnp.sum(jnp.abs(flat_recon))
        es_rmse = jnp.sqrt(jnp.sum(error_sinogram * error_sinogram) / real_sino_size)
        return fm_loss, recon_l1, es_rmse

    def prox_map(self, prox_input, sinogram, sigma_prox=None, weights=None, init_recon=None, do_initialization=True, stop_threshold_change_pct=0.2,
                 max_iterations=3, first_iteration=0, logfile_path='~/.mbirjax/logs/prox.log', print_logs=True, output_sharded=False):
        """
        Proximal Map function for use in Plug-and-Play applications.
        This function is similar to recon, but it essentially uses a prior with a mean of prox_input and a standard deviation of sigma_prox.

        Reproducibility note: the pixel partitions are drawn from numpy's global random number
        generator, so results vary slightly from run to run.  For a reproducible result, call
        ``np.random.seed(seed)`` before calling this method.

        Args:
            prox_input (numpy or jax array): proximal map input with same shape as reconstruction.
            sinogram (numpy or jax array): 3D sinogram data with shape (num_views, num_det_rows, num_det_channels).
            sigma_prox (None or float, optional): The standard deviation of the proximal map prior term.  If None, then set automatically from the sinogram.  Defaults to None.
            weights (numpy or jax array, optional): 3D positive weights with same shape as sinogram.  Defaults to None, in which case the weights are implicitly all 1s.
            init_recon (numpy or jax array, optional): optional reconstruction to be used for initialization.  Defaults to None, in which case the initial recon is determined by vcd_recon.
            do_initialization (bool, optional):  If True, then initialize parameters and place arrays on appropriate devices.  Defaults to True.
                Set to False if initialization (partitions and regularization parameters) has already been performed
                on this sinogram by a previous prox_map call on this model.
            stop_threshold_change_pct (float, optional): Stop reconstruction when NMAE percent change from one iteration to the next is below stop_threshold_change_pct.  Defaults to 0.2.
            max_iterations (int, optional): maximum number of iterations of the VCD algorithm to perform.
            first_iteration (int, optional): Set this to be the number of iterations previously completed when restarting a recon using init_recon.  This defines the first index in the partition sequence.  Defaults to 0.
            logfile_path (str, optional): Path to the output log file ('~' expands to the user's
                home directory).  Defaults to '~/.mbirjax/logs/prox.log'.  Set to None or '' to
                skip file logging.
            print_logs (bool, optional): If true then print logs to console.  Defaults to True.
            output_sharded (bool, optional): If False (default), return a numpy reconstruction array.
                If True, return the internal device form (no exit gather); on an unsharded model the
                output is the same either way.

        Returns:
            (recon, recon_dict): reconstruction array and a dict containing the recon parameters.
                - recon (numpy or jax array): the reconstruction volume
                - recon_dict (dict): A dict obtained from :meth:`get_recon_dict` with entries
                    * 'recon_params'
                    * 'notes'
                    * 'recon_logs'
                    * 'model_params'
        """
        compute_prior_loss = False
        prior_loss = [0]
        if do_initialization or self.prox_data is None:
            # (prox_input is NOT pre-placed here: vcd_recon routes it through to_recon, which
            # slice-shards it; an early device_put to a single device would just commit it to one
            # device and force an immediate reshard.)
            sinogram, weights, init_recon, partitions, partition_sequence, granularity, regularization_params = (
                self.initialize_recon(sinogram, weights, init_recon, max_iterations, first_iteration,
                                      compute_prior_loss, logfile_path, print_logs))
            self.prox_data = (partitions, partition_sequence, granularity, regularization_params)
        else:
            partitions, partition_sequence, granularity, regularization_params = self.prox_data

        self_sigma_prox = self.get_params('sigma_prox')
        if sigma_prox is not None:  # Override the auto sigma_prox if needed
            regularization_params['sigma_prox'] = sigma_prox
            self.set_params(no_warning=True, sigma_prox=sigma_prox, auto_regularize_flag=self.get_params('auto_regularize_flag'))

        # Compute proximal map
        try:

            recon, loss_vectors = self.vcd_recon(sinogram, partitions, partition_sequence, stop_threshold_change_pct,
                                                 weights=weights, init_recon=init_recon, prox_input=prox_input,
                                                 first_iteration=first_iteration)

            # Return num_iterations, granularity, partition_sequence, fm_rmse values, regularization_params
            partition_sequence = [int(val) for val in partition_sequence]
            fm_rmse = [float(val) for val in loss_vectors[0]]
            stop_threshold_change_pct = [100 * float(val) for val in loss_vectors[2]]
            alpha_values = [float(val) for val in loss_vectors[3]]
            num_iterations = len(fm_rmse)
            recon_param_values = [num_iterations, granularity, partition_sequence, fm_rmse, prior_loss,
                                  regularization_params, stop_threshold_change_pct, alpha_values]
            recon_params = ReconParams(*tuple(recon_param_values))._asdict()
            self.set_params(no_warning=True, sigma_prox=self_sigma_prox)

        except MemoryError as e:
            self.logger.error('Insufficient CPU memory')
            raise e
        except JaxRuntimeError as e:
            self._handle_jax_error(e)

        if logfile_path:
            self.logger.info('Logs written to {}'.format(
                os.path.abspath(os.path.expanduser(logfile_path))))

        for h in list(self.logger.handlers):  # Make sure the log files are up to date
            h.flush()

        notes = 'Prox completed: {}\n\n'.format(datetime.datetime.now())
        recon_dict = self.get_recon_dict(recon_params, notes=notes)
        if not output_sharded:
            # Default exit: gather to a HOST NumPy array in the problem's REAL shape
            # (_gather_recon assembles on the host and crops any padded slices) -- never re-materialized
            # on a single device.
            recon = self._gather_recon(recon)
        return recon, recon_dict

    @staticmethod
    def gen_weights(sinogram, weight_type):
        """
        DEPRECATED:  Use :func:`mbirjax.gen_weights` instead.

        Compute the optional weights used in MBIR reconstruction.

        Args:
            sinogram (jax array): 3D jax array containing sinogram with shape (num_views, num_det_rows, num_det_channels).
            weight_type (string): Type of noise model used for data
                    - weight_type = 'unweighted' => return numpy.ones(sinogram.shape).
                    - weight_type = 'transmission' => return numpy.exp(-sinogram).
                    - weight_type = 'transmission_root' => return numpy.exp(-sinogram/2).
                    - weight_type = 'emission' => return 1/(numpy.absolute(sinogram) + 0.1).

        Returns:
            (jax array): Weights used in mbircone reconstruction, with the same array shape as ``sinogram``.

        Raises:
            Exception: Raised if ``weight_type`` is not one of the above options.
        """
        warnings.warn('TomographyModel.gen_weights() is deprecated and will be removed in a future release.  Use mbirjax.gen_weights() instead.')
        return mj.gen_weights(sinogram, weight_type)

    def reshape_recon(self, recon):
        """
        Reshape recon into its 3D form.

        Args:
            recon (numpy or jax array): A 3D array of shape specified by (num_recon_rows, num_recon_cols, num_recon_slices)
        """
        recon_shape = self.get_params('recon_shape')
        return recon.reshape(recon_shape)

    def scale_recon_shape(self, row_scale=1.0, col_scale=1.0, slice_scale=1.0):
        """
        Scale the reconstruction shape by the given scale factors.

        This can be used before starting a reconstruction to improve results when part of the object
        projects outside the detector. The method updates the internal `recon_shape` parameter.

        Args:
            row_scale (float): Scale factor for the number of rows in the reconstruction.
            col_scale (float): Scale factor for the number of columns in the reconstruction.
            slice_scale (float): Scale factor for the number of slices in the reconstruction.

        Returns:
            tuple[int, int, int]: A 3-tuple representing the number of pixels added to the
            (rows, columns, slices) dimensions due to scaling.

        Example:
            >>> old_shape = model.get_params('recon_shape')
            >>> added_padding = model.scale_recon_shape(row_scale=1.2, col_scale=1.1)
            >>> new_shape = model.get_params('recon_shape')
            >>> print(f"Shape increased by: {added_padding}")
        """
        old_rows, old_cols, old_slices = self.get_params('recon_shape')
        new_rows = int(old_rows * row_scale)
        new_cols = int(old_cols * col_scale)
        new_slices = int(old_slices * slice_scale)
        self.set_params(recon_shape=(new_rows, new_cols, new_slices))
        return new_rows - old_rows, new_cols - old_cols, new_slices - old_slices


from functools import partial


@partial(jax.jit, donate_argnames='cur_flat_recon')
def update_recon(cur_flat_recon, cur_indices, cur_delta):
    cur_flat_recon = cur_flat_recon.at[cur_indices].add(cur_delta)
    return cur_flat_recon


@partial(jax.jit, donate_argnames='error_sinogram')
def update_error_sinogram(error_sinogram, alpha, delta_sinogram):
    # Fused update error_sinogram <- error_sinogram - alpha * delta_sinogram, DONATING
    # error_sinogram so XLA updates it in place (the same trick update_recon uses for the recon).
    # Without in-place reuse each VCD subset allocates a fresh view-sharded error sinogram, and the
    # stale ones are held alive in jax's internal sharded-array reference cycles -- so they
    # accumulate (one full sinogram per subset) until garbage collection, which is what made
    # multi-device memory blow up.
    # alpha is folded into the jit (rather than eagerly pre-scaling delta_sinogram) so XLA emits a
    # single fused multiply-add and there is no separate scaled-delta transient to free.  The FMA
    # differs from an eager pre-scale by ~1 ULP; that is within the placement path's accepted
    # tolerance (the trivial-mesh path is no longer required to be bit-exact).
    return error_sinogram - alpha * delta_sinogram


@jax.jit
def sum_product(array0, array1):
    prod = jax.vmap(jnp.multiply)(array0, array1)
    sum_of_prod = jax.vmap(jnp.sum)(prod)
    sum_of_prod = jnp.sum(sum_of_prod)
    return sum_of_prod


def get_transpose(linear_map, input_shape):
    """
    Use jax to determine the transpose of a linear map.

    Args:
        linear_map:  [function] The linear function to be transposed
        input_shape: [ndarray] The shape of the input to the function

    Returns:
        transpose: A function to evaluate the transpose of the given map.  The input to transpose
        must be a jax or ndarray with the same shape as the output of the original linear_map.
        transpose(input) returns an array of shape input_shape.
    """
    # print('Defining transpose map')
    # t0 = time.time()
    input_info = types.SimpleNamespace(shape=input_shape, dtype=jnp.dtype(jnp.float32))
    transpose_list = jax.linear_transpose(linear_map, input_info)

    def transpose(input_array):
        return transpose_list(input_array)[0]

    return transpose
