import io
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
# Internal sharding primitives (see _sharding), accessed with the `mjs` prefix.
# Importing the SUBMODULE directly (not aliasing the top-level `mbirjax` and
# reaching submodules as attributes) is safe even mid-import of mbirjax: it
# forces `mbirjax._sharding` to load, and that subpackage pulls in only
# jax/numpy/warnings, nothing that loops back into the partially-initialized
# mbirjax package.
import mbirjax._sharding as mjs

from importlib.metadata import version, PackageNotFoundError

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
# Set the GPU memory fraction for JAX
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.98'

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

    DIRECT_RECON_VIEW_BATCH_SIZE = 100  # This is set here due to a bug in jax.vmap when the batch size is too large.

    def __init__(self, sinogram_shape, **kwargs):

        super().__init__()
        self.set_params(no_compile=True, no_warning=True, sinogram_shape=sinogram_shape, **kwargs)

        self.auto_set_recon_geometry(no_compile=True, no_warning=True)

        self.set_params(geometry_type=str(type(self)))

        self.main_device, self.sinogram_device= None, None
        self.cpus = jax.devices('cpu')
        self.projector_functions = None
        self.prox_data = None

        # Multi-device sharding.  An explicit configure_sharding() builds a mesh over the
        # chosen devices and sets _sharding_configured = True.  Absent that, geometries whose
        # _supports_sharding() is True (ParallelBeam) auto-default the homogeneous single-device
        # case to a trivial 1-device mesh at the end of set_devices(), so the
        # placement path is always on; other geometries keep mesh = None (legacy single-device).
        self.mesh = None
        self.shard_devices = None
        self.dev2dev_safe = True   # set empirically in configure_sharding
        self._sharding_configured = False  # True only after an explicit configure_sharding()
        # Opt-in: let AUTOMATIC selection shard across CPU devices too (off by default).  Auto
        # sharding is GPU-only by default because a normal CPU host exposes ONE jax device -- a
        # multi-CPU-device setup is usually a virtual/test artifact (XLA_FLAGS), and virtual-CPU
        # sharding is bandwidth-bound.  On a real multi-core/CPU-cluster host it may pay off; set
        # this True (then call set_devices(), or set it before the recon) to auto-shard on CPU.
        # EXPERIMENTAL: CPU-cluster performance is not yet characterized (see the plan's adjacent
        # task on differentiating virtual CPUs from a real CPU cluster).  Explicit CPU sharding via
        # configure_devices(n)/configure_sharding(cpu_devices) does NOT need this flag.
        self._auto_shard_cpu = False
        # recon_placement / sino_placement describe how recon-like and sino-like
        # arrays are distributed across devices.  Each is a Placement, which owns
        # a device list, a sharded axis, AND its own 1-D mesh; _set_placements()
        # builds them from the current device config (a single device gives a
        # trivial 1-shard placement).
        #
        # These are intended to be the single source of truth for device layout,
        # but for now they coexist with the two older representations they will
        # replace: the scalar main_device / sinogram_device, and mesh /
        # shard_devices (where `mesh is None` also serves as the single-vs-multi
        # flag).  In sharded mode each placement's mesh is currently an
        # equal-but-distinct copy of self.mesh.
        #
        # TODO (retire the pre-placement device representations): once the
        # projector and VCD paths consume recon_placement / sino_placement,
        # remove main_device / sinogram_device and mesh / shard_devices; the
        # placements (each owning its mesh) become the single source of device
        # layout and the `mesh is None` branching is unified away.  (self.mesh
        # does not generalize to the hybrid case anyway, where recon and sino
        # live on different meshes.)  Also remove the previous paragraph.
        self.recon_placement = None
        self.sino_placement = None

        # The following may be adjusted based on memory in set_devices()
        self.view_batch_size_for_vmap = 512
        self.pixel_batch_size_for_vmap = 2048
        self.transfer_pixel_batch_size = 100 * self.pixel_batch_size_for_vmap
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
    def is_sharded(self):
        """True when multi-device sharding is configured (the sharded code path is active).

        Single source of truth for "are we sharded?", so call sites read intent rather
        than the current implementation detail (``self.mesh is not None``).  When the
        device representation migrates from ``mesh`` to placements this property's body
        changes in one place; and once every geometry uses the sharded/placement path,
        this (and the branches that test it) is what gets retired.
        """
        return self.mesh is not None

    def _supports_sharding(self):
        """Whether this geometry has the placement/movement projector path implemented.

        Only such geometries auto-default the single-device case to a trivial 1-device mesh
        (the always-on placement path); the rest stay on the legacy single-device path until
        their projectors are ported (P6).  Base default is False; ParallelBeamModel overrides
        to True.  (This does not gate an explicit configure_sharding(), which a caller invokes
        deliberately.)
        """
        return False

    def configure_devices(self, devices=None):
        """
        Configure which devices the reconstruction runs on (the user-facing control surface).

        Resolves ``devices`` to a concrete device list and PINS that configuration (set_devices()
        will not override it on a later set_params):

          * ``None``  -- automatic: shard across all available GPUs whose count divides the sharded
            recon axis (see :meth:`_auto_device_count`); if no GPU is available, use a single CPU
            device.  This is the explicit, pinned form of the choice ``use_gpu='automatic'`` makes
            by default.
          * ``int n`` -- use the first ``n`` devices of the default platform (GPUs if any, else CPUs).
          * ``sequence of ints`` -- use those indices into the default device list (``jax.devices()``).
          * ``sequence of jax devices`` -- use exactly those devices.

        Sharding requires the device count to evenly divide the sharded recon axis (slices).  The
        sinogram view axis does NOT constrain the count: a non-dividing view axis is zero-padded to
        the next multiple of the device count, and the padding is kept exactly inert (it cannot
        affect the results).  A count incompatible with the CURRENT slice count only warns (the
        shapes may still change before the recon); the reconstruction raises if it is still
        incompatible.

        Args:
            devices (None, int, or sequence of ints / jax devices): see above.

        Returns:
            Nothing; sets the device mesh and placements (see :meth:`configure_sharding`).
        """
        self.configure_sharding(self._resolve_devices(devices))

    def _resolve_devices(self, devices):
        """Resolve the configure_devices() ``devices`` argument to a concrete device list."""
        def default_pool():
            try:
                g = list(jax.devices('gpu'))
                if g:
                    return g
            except RuntimeError:
                pass
            return list(jax.devices('cpu'))

        if devices is None:
            pool = default_pool()
            on_gpu = bool(pool) and pool[0].platform == 'gpu'
            n = self._auto_device_count(len(pool)) if on_gpu else 1
            return pool[:n]
        if isinstance(devices, (int, np.integer)):
            return default_pool()[:int(devices)]
        devices = list(devices)
        if devices and all(isinstance(d, (int, np.integer)) for d in devices):
            all_devices = jax.devices()
            return [all_devices[int(i)] for i in devices]
        return devices

    def configure_sharding(self, devices=None):
        """
        Configure multi-device sharding over the given devices (opt-in).

        This is additive: it sets up a device mesh and an empirical
        device-to-device safety flag, but does NOT modify ``main_device`` /
        ``sinogram_device`` or the existing single-device code paths.  The
        projector phases consume this configuration as they are ported to
        sharding; until then a configured mesh has no effect on existing flows.

        Sharding scheme (uniform across geometries): sinogram-like objects are
        sharded by **view** (axis 0) and recon-like objects by **slice**.  The
        device count must evenly divide ``num_slices``; ``num_views`` does NOT
        constrain it (a non-dividing view axis is zero-padded to the next
        multiple of the device count, and the padding is kept exactly inert).  A
        count incompatible with the CURRENT slice count only **warns** here (the
        shapes may still change before the recon); the hard error is raised when
        an array is actually sharded (so selection is order-independent).

        Passing ``devices=None`` (or a single device) sets up a trivial 1-device
        mesh, so single-device operation is just the degenerate sharded case and
        the same code path can be used throughout.

        Args:
            devices (sequence of jax devices, or None): devices to shard across.
                None uses a single device (trivial sharding).

        Returns:
            Nothing; sets ``self.mesh``, ``self.shard_devices``,
            ``self.dev2dev_safe``, and the placements (pinned: a later
            ``set_params`` will not override this configuration).
        """
        if devices is None:
            devices = jax.devices()[:1]
        devices = list(devices)
        n_devices = len(devices)
        if n_devices < 1:
            raise ValueError("configure_sharding requires at least one device.")

        # Divisibility: each array's sharded axis must be evenly divisible by the device count
        # (equal contiguous shards).  The current shapes may still change before reconstruction
        # (e.g. scale_recon_shape), so we only WARN here and defer the hard error to the actual
        # sharding op (_shard_on_axis).  This keeps selection order-independent: a config that does
        # not fit the current shapes but is fixed up before the recon still works.
        msg = self._divisibility_warning(n_devices)
        if msg:
            warnings.warn('configure_sharding: ' + msg)

        self._apply_mesh(devices, user_selected=True)

    def _apply_mesh(self, devices, user_selected):
        """Build the device mesh and placements from ``devices`` (a 1-D mesh over them).

        Shared by the explicit configure_sharding() (user_selected=True) and the automatic default
        in set_devices() (user_selected=False).

        Args:
            devices (sequence of jax devices): devices to shard across (length 1 = trivial mesh).
            user_selected (bool): if True, mark the configuration as user-selected so set_devices
                will not override it; if False, leave it overridable so a later set_params (which
                re-runs set_devices) can re-evaluate the device layout.
        """
        from jax.sharding import Mesh
        devices = list(devices)
        # 1-D mesh; the axis name 'devices' is referenced by the PartitionSpecs used when sharding
        # the sinogram (view axis) and recon (slice axis).
        self.mesh = Mesh(np.array(devices), ('devices',))
        self.shard_devices = devices
        self.dev2dev_safe = mjs.is_dev2dev_safe(devices)
        self._sharding_configured = user_selected
        self._set_placements()

    def _divisibility_warning(self, n_devices):
        """Message (or None) if ``n_devices`` does not evenly divide the sharded RECON axis of the
        CURRENT shapes.

        Only the recon (slice) axis constrains the device count: a non-dividing VIEW axis is
        zero-padded to the next multiple of the device count, with the padding kept exactly inert,
        so it never warns.  Used for an EARLY, non-blocking warning (configure_sharding, or a
        geometry change to a user-selected config); the HARD error is raised later at the actual
        sharding op (_shard_on_axis).  Warning rather than raising here keeps device selection
        order-independent: a config that doesn't fit the current shapes but is fixed up before
        reconstruction (e.g. a later scale_recon_shape) still works, regardless of the order of the
        configure/set_params calls.  Which axis is sharded comes from the axis-declaration hooks
        (single source of truth).
        """
        recon_shape = self.get_params('recon_shape')
        recon_axis = self.recon_shard_axis() % len(recon_shape)
        if recon_shape[recon_axis] % n_devices == 0:
            return None
        return ('{} devices do not evenly divide recon axis {} (size {}).  Resolve before '
                'reconstruction (make the axis a multiple of {}, or choose a compatible device '
                'count); the reconstruction will raise if it is still incompatible.'.format(
                    n_devices, recon_axis, recon_shape[recon_axis], n_devices))

    def _auto_device_count(self, n_available):
        """Largest device count <= n_available that evenly divides the sharded RECON axis (slices).

        Automatic device selection (use_gpu='automatic'/'full' on a multi-GPU box) shards across
        this many devices.  The VIEW axis no longer constrains the count: a non-dividing view axis
        is zero-padded to the next multiple of the device count and the padding is kept exactly
        inert (entry zero-fill + a mask on the forward-projection output), so any N works there.
        The slice axis must still divide so each device owns an equal contiguous block (slice
        padding is a later step -- it couples to the qGGMRF prior).  There is NO per-device size
        floor (decided 2026-06-09 from the GPU band/scale sweeps): sharding's value is capacity +
        near-linear speedup at the sizes that matter, and over-sharding a small problem is only a
        mild overhead.

        Returns 1 for unsupported geometries, n_available <= 1, or when nothing > 1 divides slices.
        """
        if n_available <= 1 or not self._supports_sharding():
            return 1
        recon_shape = self.get_params('recon_shape')
        num_recon = int(recon_shape[self.recon_shard_axis() % len(recon_shape)])
        for n in range(n_available, 0, -1):
            if num_recon % n == 0:
                return n
        return 1

    @staticmethod
    def _platform_label(device):
        """Short uppercase platform name for a jax device ('GPU' / 'CPU' / 'TPU')."""
        return {'cpu': 'CPU', 'tpu': 'TPU'}.get(device.platform, 'GPU')

    def _recon_devices(self):
        """The devices the reconstruction actually runs on.

        The shard devices when sharded (the placement path), else the single main device (legacy
        path).  This is the truth for "where does the recon run" -- use it rather than
        ``main_device``, which is not updated by ``configure_sharding`` and so can be stale (e.g. it
        stays a GPU after sharding explicitly onto CPU devices).
        """
        return self.shard_devices if self.is_sharded else [self.main_device]

    def _device_report(self):
        """A 'N x PLATFORM [(sharded)]' summary of the recon devices, for the recon log.

        ``(sharded)`` marks the placement path (``is_sharded``), regardless of device count, so a
        single-device PLACEMENT recon (e.g. ParallelBeam) is distinguished from a single-device
        LEGACY recon (a geometry not yet ported) -- a real difference until P6.  When automatic
        selection left GPUs idle because the device count cannot divide both sharded axes, the
        reason is appended so idle hardware is never silent.
        """
        devices = self._recon_devices()
        suffix = ' (sharded)' if self.is_sharded else ''
        n = len(devices)
        platform = self._platform_label(devices[0])
        report = '{} x {}{}'.format(n, platform, suffix)
        # A padded view axis is invisible in the results (the padding is exactly inert), so say so
        # in the log rather than leaving the device-form shape a surprise.
        if self.is_sharded and self.sino_placement is not None and self.sino_placement.is_padded:
            report += ' (views padded {}->{})'.format(
                self.sino_placement.real_size, self.sino_placement.padded_size)
        # Automatic GPU selection that left devices idle (slices not divisible by more): explain why.
        if self.is_sharded and not self._sharding_configured and platform == 'GPU':
            try:
                n_available = len(jax.devices('gpu'))
            except RuntimeError:
                n_available = n
            if n_available > n:
                recon_shape = self.get_params('recon_shape')
                num_slices = recon_shape[self.recon_shard_axis() % len(recon_shape)]
                report += (' (using {} of {} GPUs: {} is the largest device count dividing '
                           'num_slices={}; make num_slices a multiple of the device count to use '
                           'more)'.format(n, n_available, n, num_slices))
        return report

    @property
    def device_summary(self):
        """str: A read-only summary of the devices the reconstruction will actually use.

        This is the resolved OUTCOME of the device configuration -- the ``use_gpu``
        parameter and ``configure_devices`` are the request; this reports what was
        chosen.  E.g. ``'4 x GPU (sharded)'``, with notes appended when the view
        axis is padded or when automatic selection left GPUs idle.  The same line
        is logged at the start of every reconstruction.
        """
        return self._device_report()

    def _set_placements(self):
        """Build recon_placement / sino_placement from the current device config.

        A configured mesh gives placements over its devices (recon on the slice
        axis, sino on the view axis); otherwise each is a trivial 1-device
        placement on the configured single device -- which may differ for recon
        and sino (e.g. recon on CPU and sino on GPU for a large recon).

        Each placement also receives the problem-owned (REAL) length of its
        sharded axis from the params, so it knows the device-form (padded)
        length when that size does not divide the device count.  The params
        always keep the real shapes ("what is the problem?"); the placements own
        the padded device shapes ("what is on the devices?").  This runs on
        every recompile (set_devices), so the pad metadata tracks shape changes.
        Only the VIEW axis is actually padded today; the recon (slice) axis must
        still divide (its padding is a later step).
        """
        recon_axis = self.recon_shard_axis()
        sino_axis = self.sinogram_shard_axis()
        sinogram_shape = self.get_params('sinogram_shape')
        recon_shape = self.get_params('recon_shape')
        num_views = sinogram_shape[sino_axis % len(sinogram_shape)]
        num_slices = recon_shape[recon_axis % len(recon_shape)]
        if self.is_sharded:
            devices = self.shard_devices
            self.recon_placement = mjs.Placement(devices, axis=recon_axis, real_size=num_slices)
            self.sino_placement = mjs.Placement(devices, axis=sino_axis, real_size=num_views)
        else:
            # Single-device.  This runs at the end of set_devices
            # (which has just set both devices) or configure_sharding (mesh path
            # above), so the devices are always concrete here.
            assert self.main_device is not None and self.sinogram_device is not None, \
                "main_device/sinogram_device must be set before _set_placements"
            self.recon_placement = mjs.Placement([self.main_device], axis=recon_axis, real_size=num_slices)
            self.sino_placement = mjs.Placement([self.sinogram_device], axis=sino_axis, real_size=num_views)

    # ------------------------------------------------------------------
    # Sharding hooks (uniform default scheme; override per geometry only
    # if a geometry needs a different axis or halo strategy)
    # ------------------------------------------------------------------
    # The uniform scheme shards sinogram-like objects by view and recon-like
    # objects by slice.  The two axis-declaration hooks below are the single
    # source of truth for *which* axis is sharded; the divisibility check in
    # configure_sharding and the _shard_*/_gather_* helpers all derive from
    # them.  Keeping the choice here (rather than hardcoded throughout) means a
    # geometry can declare a different axis by overriding one small method,
    # without hunting down scattered assumptions.

    def sinogram_shard_axis(self):
        """Axis of a sinogram-like array (views, det_rows, channels) to shard.

        Default 0 (views).  Sinogram, weights, and error-sinogram all share
        this layout, so they shard on the same axis.
        """
        return 0

    def recon_shard_axis(self):
        """Axis of a recon-like array to shard.

        Default -1 (the last axis = slices).  This is correct for both the 3-D
        recon ``(rows, cols, slices)`` and the flat recon
        ``(rows*cols, slices)`` because slices are the last axis in both, so a
        single value works regardless of rank.
        """
        return -1

    def _shard_on_axis(self, x, axis, what='array'):
        """Distribute array ``x`` across the mesh along ``axis`` (no-op if no mesh).

        When ``self.mesh`` is None (single-device, not configured) ``x`` is
        returned unchanged.  Otherwise ``x`` is placed in a NamedSharding that
        partitions ``axis`` across the mesh's ``'devices'`` axis.

        If ``x`` already carries exactly that sharding, it is returned as-is with
        no data movement.  Otherwise it is moved via the transfer helper, which
        copies directly when device-to-device transfer is safe on this hardware
        and routes through host memory otherwise (see mbirjax._sharding).

        This is the single chokepoint where entry arrays are sharded, so it is also
        where the equal-shard divisibility requirement is enforced with a clear
        error: the sharded axis must be divisible by the device count.  Selection
        (configure_*/set_params) only WARNS about divisibility; the hard check lands
        here, at the actual op, so it is order-independent and the message is clear
        rather than a cryptic XLA shard error.  (A no-op for a single device, since
        any size is divisible by 1.)

        Args:
            x: a JAX (or numpy) array to distribute.
            axis (int): the axis of ``x`` to partition across devices; may be
                negative (counted from the end).
            what (str): human label for ``x`` in the divisibility error message.

        Returns:
            The array in the requested NamedSharding (or ``x`` unchanged when no
            mesh is configured).
        """
        if not self.is_sharded:
            return x
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
            self.mesh, jax.sharding.PartitionSpec(*spec))
        # Skip any movement if x is already in exactly this sharding.
        if isinstance(getattr(x, 'sharding', None), jax.sharding.NamedSharding):
            if x.sharding == sharding:
                return x
        return mjs.move_shard(x, sharding, dev2dev_safe=self.dev2dev_safe)

    def _gather_to_host(self, x):
        """Gather a sharded array to a single uncommitted JAX array (no-op if no mesh).

        When ``self.mesh`` is None, returns ``x`` unchanged.  Otherwise reads all
        shards to a contiguous host buffer with ``np.asarray`` (the read path is
        always safe, even on hardware where device-to-device writes are not) and
        wraps it as an uncommitted JAX array, leaving JAX free to place it for
        downstream ops.

        Args:
            x: a (possibly sharded) JAX array.

        Returns:
            An uncommitted single-device JAX array with the same values, or ``x``
            unchanged when no mesh is configured.
        """
        if not self.is_sharded:
            return x
        return jnp.array(np.asarray(x))

    def _shard_sinogram(self, sinogram):
        """Distribute a sinogram-like array in its native (per-geometry) sharding.

        Pad-aware: when the view axis does not divide the device count, a
        real-shape input is zero-padded to the device form (see
        :meth:`_pad_shard_on_axis`) and an already-padded input passes through
        unchanged.  When the view axis divides, this is the plain shard
        chokepoint (:meth:`_shard_on_axis`).
        """
        axis = self.sinogram_shard_axis()
        if self.is_sharded and self.sino_placement.is_padded:
            return self._pad_shard_on_axis(sinogram, self.sino_placement, axis,
                                           what='sinogram (view axis)')
        return self._shard_on_axis(sinogram, axis, what='sinogram (view axis)')

    def _gather_sinogram(self, sinogram):
        """Gather a sharded sinogram back to a single uncommitted array, cropping
        any zero-filled padded views back to the problem's real view count."""
        out = self._gather_to_host(sinogram)
        if self.is_sharded and self.sino_placement.is_padded:
            axis = self.sinogram_shard_axis() % out.ndim
            if out.shape[axis] == self.sino_placement.padded_size:
                idx = [slice(None)] * out.ndim
                idx[axis] = slice(0, self.sino_placement.real_size)
                out = out[tuple(idx)]
        return out

    def _pad_shard_on_axis(self, x, placement, axis, what='array'):
        """Distribute ``x`` across ``placement``, zero-padding the sharded axis to
        the device form (``placement.padded_size``).

        The padding never exists on the host: each device receives its own slice
        of the host array directly (``device_put`` per shard), and the last
        shard's tail is zero-filled ON its device -- so the only transient is one
        shard on one device, never a padded copy of the whole array.  The
        zero-filled tail is what keeps the padding inert downstream (zero views
        contribute nothing to any reduction over views).

        Accepts either the problem's REAL axis length (pads) or the device-form
        PADDED length (already prepared -- passes through `_shard_on_axis`, which
        is a no-op when the sharding already matches).

        Args:
            x: array (numpy or jax) to distribute.
            placement (mjs.Placement): the target placement; must carry
                ``real_size`` (and hence ``padded_size``) for its sharded axis.
            axis (int): the axis of ``x`` to partition (may be negative).
            what (str): human label for error messages.

        Returns:
            The zero-padded array in the placement's NamedSharding.
        """
        axis = axis % x.ndim
        real, padded = placement.real_size, placement.padded_size
        if x.shape[axis] == padded:
            # Already in the device form (e.g. prepare_sino_for_devices output).
            return self._shard_on_axis(x, axis, what=what)
        if x.shape[axis] != real:
            raise ValueError(
                'Cannot place the {}: its sharded axis has size {}, but the model expects the '
                'problem size {} (or the prepared device-form size {}).  If the device '
                'configuration changed since prepare_sino_for_devices, re-run it.'.format(
                    what, x.shape[axis], real, padded))
        zeros_dtype = jax.dtypes.canonicalize_dtype(x.dtype)
        pieces = []
        for dev, (start, end), n_valid in placement.padded_shard_ranges():
            block = end - start
            idx = [slice(None)] * x.ndim
            if n_valid == block:
                idx[axis] = slice(start, end)
                pieces.append(jax.device_put(x[tuple(idx)], dev))
                continue
            tail_shape = list(x.shape)
            tail_shape[axis] = block - n_valid
            tail = jnp.zeros(tuple(tail_shape), dtype=zeros_dtype, device=dev)
            if n_valid <= 0:
                pieces.append(tail)
                continue
            idx[axis] = slice(start, start + n_valid)
            valid = jax.device_put(x[tuple(idx)], dev)
            pieces.append(jnp.concatenate([valid, tail], axis=axis))
        global_shape = list(x.shape)
        global_shape[axis] = padded
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
            sinogram (ndarray or jax array): sinogram in the model's sinogram_shape.
            weights (ndarray or jax array, optional): weights of the same shape;
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
        """Distribute a recon-like array (3-D or flat) in its native sharding."""
        return self._shard_on_axis(recon, self.recon_shard_axis(), what='reconstruction (slice axis)')

    def _gather_recon(self, recon):
        """Gather a sharded recon back to a single uncommitted array."""
        return self._gather_to_host(recon)

    def _extract_halos(self, flat_recon):
        """Return per-device boundary slices for the qggmrf inter-slice prior.

        ``flat_recon`` is sharded along the slice axis (the last axis), so each
        device holds ``(num_pixels, local_slices)``.  The qggmrf prior couples a
        slice to its neighbors, so each device needs one boundary slice from each
        adjacent device:

          left_halo[i]  = last slice of device i-1  (None for device 0)
          right_halo[i] = first slice of device i+1 (None for the last device)

        Reads ``2*(n_devices-1)`` slices to host — negligible vs. compute.
        Returns ``([None], [None])`` when no mesh is configured (single device).

        Args:
            flat_recon: a slice-sharded flat recon ``(num_pixels, slices)``.

        Returns:
            (left_halos, right_halos): two lists of length ``n_devices``; each
            entry is a numpy array of shape ``(num_pixels,)`` or None at a
            boundary.
        """
        if not self.is_sharded:
            return [None], [None]
        slice_axis = self.recon_shard_axis() % flat_recon.ndim
        # Order shards by their start index along the sharded (slice) axis so the
        # device sequence is deterministic.
        shards = sorted(flat_recon.addressable_shards,
                        key=lambda s: s.index[slice_axis].start)
        left_halos = [None] + [np.asarray(s.data[..., -1]) for s in shards[:-1]]
        right_halos = [np.asarray(s.data[..., 0]) for s in shards[1:]] + [None]
        return left_halos, right_halos

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
        """Extract the qGGMRF boundary halos once and pre-place each on its shard's device.

        :meth:`_extract_halos` reads the ``2*(n_dev-1)`` boundary slices to host; this
        wrapper then ``device_put``s each onto the device of the shard that will use it,
        so a caller can stage the halos ONCE (e.g. per VCD partition pass) and hand the
        result to :meth:`_qggmrf_prior_sharded` for every subset in that pass — turning
        a per-subset host round-trip into a per-pass one (the host round-trips are what
        cap VCD's multi-GPU scaling).

        The per-shard ordering is by slice-start, matching ``_qggmrf_prior_sharded``'s
        shard sort; the recon's sharding is constant across a pass, so a halo staged on
        ``shards[i].device`` here lines up with shard ``i`` there even though the recon
        array itself is replaced each subset.

        Args:
            flat_recon (jax array): the (slice-sharded) recon to read boundaries from.

        Returns:
            (staged_left, staged_right): per-shard lists (slice-start order); each entry
            is an on-device halo slice ``(num_pixels,)`` or ``None`` at a true recon edge.
            ``([None], [None])`` when no mesh is configured.
        """
        left_halos, right_halos = self._extract_halos(flat_recon)
        if not self.is_sharded:
            return left_halos, right_halos
        slice_axis = self.recon_shard_axis() % flat_recon.ndim
        shards = sorted(flat_recon.addressable_shards,
                        key=lambda s: s.index[slice_axis].start)
        staged_left = [None if h is None else jax.device_put(h, s.device)
                       for h, s in zip(left_halos, shards)]
        staged_right = [None if h is None else jax.device_put(h, s.device)
                        for h, s in zip(right_halos, shards)]
        return staged_left, staged_right

    def _qggmrf_prior_sharded(self, flat_recon, pixel_indices, qggmrf_params,
                              staged_halos=None):
        """Compute the qGGMRF prior gradient and Hessian on a slice-sharded recon.

        This is the recon-domain analogue of the sharded projectors: the recon is
        sharded by slice, so each slice-owner computes the prior on its own
        slice-shard **locally** and the results are assembled (with no data
        movement) into one slice-sharded array.

        The qGGMRF prior couples each slice to its slice-axis neighbors, so an
        interior shard boundary needs one boundary slice from each adjacent shard.
        Those are supplied as *halos* (per-shard boundary slices, pre-placed on each
        shard's device); the halo-aware
        :func:`mbirjax.qggmrf_gradient_and_hessian_at_indices` then runs entirely
        on each shard's device with no cross-device communication inside the kernel.
        At the true recon edges (device 0's left, the last device's right) the halo
        is ``None`` and the boundary slice is mirrored (reflected BC) -- matching the
        single-device result exactly.

        The in-slice (row/col) prior term is fully local and uses only the shard's
        own slices, so passing the *local* slice count in ``recon_shape`` is correct
        (and identical across equal shards, so the jitted prior compiles once).

        No gather is performed: the inputs are slice-sharded and the outputs are
        returned slice-sharded (matching ``recon_placement``).

        Args:
            flat_recon (jax array): slice-sharded recon ``(num_pixels, num_slices)``
                (a 1-device mesh is the trivial 1-shard case).
            pixel_indices (jax array): 1D indices into the flattened (rows, cols)
                identifying the subset of cylinders to evaluate.
            qggmrf_params (tuple): the prior parameters ``(b, sigma_x, p, q, T)``.
            staged_halos (tuple or None): ``(staged_left, staged_right)`` from
                :meth:`_stage_halos`, pre-placed on the shard devices.  Pass these to
                avoid re-reading the halos every subset (the VCD loop stages once per
                pass).  When ``None``, the halos are extracted+staged here from
                ``flat_recon`` (the self-contained path, e.g. for tests/standalone use).

        Returns:
            (gradient, hessian): each a slice-sharded array of shape
            ``(len(pixel_indices), num_slices)``.
        """
        num_rows, num_cols, num_slices = self.get_params('recon_shape')[:3]
        num_indices = len(pixel_indices)

        # Boundary slices each shard needs from its neighbors (None at the true edges),
        # pre-placed on the shard devices.  Stage here if the caller did not.
        if staged_halos is None:
            staged_left, staged_right = self._stage_halos(flat_recon)
        else:
            staged_left, staged_right = staged_halos

        # Order the shards by their start index along the sharded (slice) axis so the
        # device sequence matches the staged-halo order and recon_placement's device
        # order (used to reassemble below).
        slice_axis = self.recon_shard_axis() % flat_recon.ndim
        shards = sorted(flat_recon.addressable_shards,
                        key=lambda s: s.index[slice_axis].start)

        grad_owned, hess_owned = [], []
        for i, shard in enumerate(shards):
            device = shard.device
            local = shard.data                       # (num_pixels, local_slices) on this device
            local_slices = local.shape[slice_axis]
            recon_shape_local = (num_rows, num_cols, local_slices)
            # Indices must be resident on this shard's device; the halos already are.
            local_indices = jax.device_put(pixel_indices, device)
            g, h = mj.qggmrf_gradient_and_hessian_at_indices(
                local, recon_shape_local, local_indices, qggmrf_params,
                left_halo=staged_left[i], right_halo=staged_right[i])
            grad_owned.append(g)
            hess_owned.append(h)

        # Wrap the per-shard pieces into one slice-sharded array (no data movement).
        structure = self.recon_placement.shard_structure(2)
        gradient = mjs.assemble_sharded(grad_owned, (num_indices, num_slices), structure)
        hessian = mjs.assemble_sharded(hess_owned, (num_indices, num_slices), structure)
        return gradient, hessian

    def set_devices(self):
        """
        Determine whether to run the reconstruction entirely on the GPU (when one is available) or
        entirely on the CPU, and set the corresponding devices.

        This determination can be overridden by using ct_model.set_params(use_gpu=string), where string is one of
        'automatic', 'full', 'none'

        Returns:
            Nothing, but instance variables are set to appropriate values.
        """
        # This does two things:
        #   (1) Choose the physical device(s): the whole reconstruction runs on the GPU when one is
        #       available and the user has not forced CPU, otherwise on the CPU.  We deliberately do
        #       NOT estimate whether the problem fits -- an over-large recon surfaces as an OOM at
        #       recon time, where _handle_jax_error guides the user (add GPUs, shrink, split, or CPU).
        #   (2) Establish how arrays are laid out on those device(s) -- the placement block below +
        #       _set_placements (which always runs and builds recon_placement / sino_placement).
        cpus = jax.devices('cpu')
        use_gpu = self.get_params('use_gpu')
        try:
            gpus = jax.devices('gpu')
        except RuntimeError:
            gpus = []
        gpu_available = len(gpus) > 0
        if not gpu_available and use_gpu not in ['automatic', 'none']:
            warnings.warn("'use_gpu' is set to {} but no gpu is available. Proceeding on cpu. "
                          "Use set_params(use_gpu='automatic') to avoid this warning.".format(use_gpu))

        # 'projections' (GPU-only projections, CPU-resident data) is no longer supported.
        if use_gpu == 'projections':
            raise ValueError("use_gpu == 'projections' is no longer supported.")

        # (1) Choose the device: everything on the GPU when one is available and the user has not
        # forced CPU, otherwise everything on the CPU.  on_gpu is the RESOLUTION of the use_gpu
        # REQUEST for this call; it is deliberately a local -- the request stays in the params
        # (re-evaluated on every recompile) and the resolved layout lives in the placements
        # (is_sharded / shard_devices / _recon_devices), so there is no stored mode string to go
        # stale.  (The former hybrid 'sinograms' mode -- recon on CPU, sinograms on GPU -- has been
        # removed.)
        on_gpu = gpu_available and use_gpu != 'none'
        if on_gpu:
            self.main_device, self.sinogram_device = gpus[0], gpus[0]
        else:
            self.main_device, self.sinogram_device = cpus[0], cpus[0]

        # (2) Establish the device layout.  Geometries that implement the placement/movement
        # projector path (_supports_sharding -> ParallelBeam) run an ALWAYS-ON placement path; unless
        # the caller has already pinned a configuration with configure_sharding()/configure_devices(),
        # auto-select here.  Auto SHARDS across all GPUs (the count is capped by slice divisibility,
        # _auto_device_count; a non-dividing VIEW count is padded) -- and across CPU devices too when
        # the _auto_shard_cpu opt-in is set (off by default; see __init__).  One GPU / one CPU /
        # opt-out -> a trivial 1-device mesh.  user_selected=False keeps it overridable, so a later
        # set_params re-evaluates the layout (e.g. a 'full' <-> 'none' mode flip, or a
        # sinogram_shape change that changes the divisible count).  Geometries not yet ported leave
        # self.mesh = None and fall back to trivial single-device placements in _set_placements (the
        # legacy path, retiring at P6).  _set_placements() runs in all branches (via _apply_mesh or
        # directly).
        if not self._sharding_configured and self._supports_sharding():
            if on_gpu and len(gpus) > 1:
                auto_pool = list(gpus)
            elif not on_gpu and self._auto_shard_cpu and len(cpus) > 1:
                auto_pool = list(cpus)
            else:
                auto_pool = None
            if auto_pool is not None:
                n = self._auto_device_count(len(auto_pool))   # largest count dividing num_slices
                self._apply_mesh(auto_pool[:n], user_selected=False)
            else:
                self._apply_mesh([self.main_device], user_selected=False)
        else:
            # User-selected config (or an unported geometry).  If a geometry change has made a
            # user-selected multi-device config no longer divide the sharded axes, warn early; the
            # hard error is still deferred to the sharding op (_shard_on_axis), so order-independence
            # holds (a later compatible shape change rescues it).
            if self._sharding_configured and self.is_sharded and len(self.shard_devices) > 1:
                msg = self._divisibility_warning(len(self.shard_devices))
                if msg:
                    warnings.warn(msg)
            self._set_placements()
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
    def forward_project_pixel_batch_to_one_view(voxel_values, pixel_indices, view_params, projector_params):
        """
        Forward project a set of voxels determined by indices into the flattened array of size num_rows x num_cols.

        Note:
            This method must be overridden for a specific geometry.

        Args:
            voxel_values (jax array):  2D array of shape (num_indices, num_slices) of voxel values, where
                voxel_values[i, j] is the value of the voxel in slice j at the location determined by indices[i].
            pixel_indices (jax array of int):  1D vector of indices into flattened array of size num_rows x num_cols.
            view_params (jax array):  A 1D array of view-specific parameters (such as angle) for the current view.
            projector_params (namedtuple):  Tuple containing (sinogram_shape, recon_shape, get_geometry_params())

        Returns:
            jax array of shape (num_det_rows, num_det_channels)
        """
        warnings.warn('Forward projector not implemented for TomographyModel.')
        return None

    @staticmethod
    def back_project_one_view_to_pixel_batch(sinogram_view, pixel_indices, single_view_params, projector_params,
                                             coeff_power=1):
        """
        Calculate the backprojection value at a specified recon voxel cylinder given a sinogram view and parameters.

        Note:
            This method must be overridden for a specific geometry.

        Args:
            sinogram_view (jax array): one view of the sinogram to be back projected
            pixel_indices (jax array of int):  1D vector of indices into flattened array of size num_rows x num_cols.
            single_view_params (jax array): A 1D array of view-specific parameters (such as angle) for the current view.
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

        This is a **user-facing** method.  The input may be plain or sharded
        (a plain recon is sharded at entry); the OUTPUT form is chosen by
        ``output_sharded``, independent of where the input lives.  By default the
        sinogram is gathered to a plain array; with ``output_sharded=True`` it is
        returned view-sharded so callers composing on-device (e.g. the VCD loop)
        pay no host round-trip.  Internally the work is the same all-gather
        either way (see :meth:`sparse_forward_project`); only the exit differs.

        Note:
            This method should generally not be used directly for iterative reconstruction.  For iterative
            reconstruction, use :meth:`recon`.

        Args:
            recon (jnp array): The 3D reconstruction array.
            output_sharded (bool, optional): If False (default), return a plain
                array.  If True, return the internal device form (view-sharded
                across the model's devices; on an unsharded model this is the
                same single-device array either way).

        Returns:
            jnp array: The resulting 3D sinogram -- plain by default,
            view-sharded if ``output_sharded=True``.
        """
        recon_shape, use_ror_mask = self.get_params(['recon_shape', 'use_ror_mask'])
        full_indices = mj.gen_full_indices(recon_shape, use_ror_mask=use_ror_mask)

        if self.is_sharded:
            recon = self._shard_recon(recon)            # no-op if already sharded
            # Extracting cylinders from a slice-sharded volume keeps the slice
            # sharding (the index is on the unsharded row/col axes), so this yields
            # slice-sharded cylinders with no movement.
            voxel_values = self.get_voxels_at_indices(recon, full_indices)
            # All-gather forward projection -> view-sharded sinogram.
            sinogram = self.sparse_forward_project(voxel_values, full_indices)
            if output_sharded:
                return sinogram                          # keep the device form
            return self._gather_sinogram(sinogram)       # default: plain output

        voxel_values = self.get_voxels_at_indices(recon, full_indices)
        return self.sparse_forward_project(voxel_values, full_indices,
                                           output_device=self.sinogram_device)

    def back_project(self, sinogram, output_sharded=False):
        """
        Perform a full back projection at all voxels in the field-of-view.

        This is a **user-facing** method.  The input may be plain or sharded
        (a plain sinogram is sharded at entry); the OUTPUT form is chosen by
        ``output_sharded``, independent of where the input lives.  By default the
        recon is gathered to a plain array; with ``output_sharded=True`` it is
        returned as a slice-sharded 3-D array so callers composing on-device
        (e.g. a sharded FBP init for the VCD loop) pay no host round-trip.
        Internally the work is the same reduce-scatter either way (see
        :meth:`sparse_back_project`); only the exit handling differs.

        Note:
            This method should generally not be used directly for iterative reconstruction.  For iterative
            reconstruction, use :meth:`recon`.

        Args:
            sinogram (jnp array): 3D jax array containing sinogram.
            output_sharded (bool, optional): If False (default), return a plain
                array.  If True, return the internal device form (slice-sharded
                across the model's devices; on an unsharded model this is the
                same single-device array either way).

        Returns:
            jnp array: The reconstructed 3D volume — plain by default,
            slice-sharded if ``output_sharded=True``.
        """
        recon_shape, use_ror_mask = self.get_params(['recon_shape', 'use_ror_mask'])
        full_indices = mj.gen_full_indices(recon_shape, use_ror_mask=use_ror_mask)
        row_index, col_index = jnp.unravel_index(full_indices, recon_shape[:2])

        if self.is_sharded:
            sinogram = self._shard_sinogram(sinogram)   # no-op if already sharded
            # Reduce-scatter back projection -> slice-sharded cylinder.
            recon_cylinder = self.sparse_back_project(sinogram, full_indices)
            if output_sharded:
                # Keep the recon sharded (no host round-trip).
                return self._assemble_recon_volume_sharded(
                    recon_cylinder, recon_shape, row_index, col_index)
            # Default: gather the cylinder and scatter into a plain volume.
            recon_cylinder = self._gather_recon(recon_cylinder)
            recon = jnp.zeros(recon_shape)
            recon = recon.at[row_index, col_index].set(recon_cylinder)
            return recon

        output_device = self.main_device
        recon_cylinder = self.sparse_back_project(sinogram, full_indices, output_device=output_device)
        with jax.default_device(output_device):
            recon = jnp.zeros(recon_shape, device=output_device)
        recon = recon.at[row_index, col_index].set(recon_cylinder)
        return recon

    def _assemble_recon_volume_sharded(self, recon_cylinder, recon_shape,
                                       row_index, col_index):
        """Scatter a slice-sharded cylinder into a slice-sharded 3-D recon volume.

        ``recon_cylinder`` is ``(num_pixels, num_slices)`` sharded along the slice
        axis (each device owns a contiguous band of slices).  The scatter
        ``recon[row_index, col_index, :] = cylinder`` is identical across slices,
        so each device scatters its own band locally into a
        ``(rows, cols, local_slices)`` zeros array — no cross-device movement —
        and the per-device volumes are wrapped as one slice-sharded array.

        Assumes the recon slice axis is the last axis (``recon_shard_axis() ==
        -1``); a geometry whose recon shards on a different axis would override
        this.

        Args:
            recon_cylinder (jax.Array): slice-sharded ``(num_pixels, num_slices)``.
            recon_shape (tuple): the global recon shape ``(rows, cols, num_slices)``.
            row_index, col_index (jax arrays): FOV pixel row/col indices (length
                num_pixels), the same on every device.

        Returns:
            jax.Array: slice-sharded ``(rows, cols, num_slices)`` recon volume.
        """
        rows, cols = int(recon_shape[0]), int(recon_shape[1])
        # Map each device to its local cylinder shard (already resident on it).
        dev_to_shard = {s.device: s.data for s in recon_cylinder.addressable_shards}

        def worker(i, device):
            cyl = dev_to_shard[device]                  # (num_pixels, local_slices)
            local = jnp.zeros((rows, cols, cyl.shape[1]))
            return local.at[row_index, col_index, :].set(cyl)

        results = mjs.run_per_device(self.shard_devices, worker)
        axis = self.recon_shard_axis() % len(recon_shape)
        spec = [None] * len(recon_shape)
        spec[axis] = 'devices'
        recon_sharding = jax.sharding.NamedSharding(
            self.mesh, jax.sharding.PartitionSpec(*spec))
        return mjs.assemble_sharded(results, tuple(recon_shape), recon_sharding)

    def sparse_forward_project(self, voxel_values, pixel_indices, view_indices=None, output_device=None):
        """
        Forward project the given voxel values to a sinogram.
        The indices are into a flattened 2D array of shape (recon_rows, recon_cols), and the projection is done using
        all voxels with those indices across all the slices.

        Args:
            voxel_values (jax.numpy.DeviceArray): 2D array of voxel values to project, size (len(pixel_indices), num_recon_slices).
            pixel_indices (jax array): Array of indices specifying which voxels to project.
            view_indices (jax array): Array of indices of views to project
            output_device (jax device): Device on which to put the output

        Returns:
            jnp array: The resulting 3D sinogram after projection.
        """
        # When a mesh is configured, take the sharded path: the recon cylinders are
        # expected slice-sharded and the result is returned view-sharded (no gather
        # here; the user-facing forward_project gathers).  Otherwise the
        # single-device path runs unchanged.
        #
        # Partial-view projection (view_indices) is single-device-only: a subset of views
        # breaks the equal view-shard, so the sharded path does not implement it.  On a trivial
        # 1-device mesh (the auto-default placement path) there is nothing to shard, so fall
        # through to the single-device implementation; only a real multi-device mesh rejects
        # view_indices.  (The view-batching callers, e.g. vcls, run single-device.)
        if self.is_sharded and view_indices is None:
            return self._sparse_forward_project_sharded(
                voxel_values, pixel_indices, view_indices=None)
        if self.is_sharded and len(self.shard_devices) > 1:
            raise NotImplementedError(
                "Sharded forward projection with view_indices is not supported on a "
                "multi-device mesh (a subset of views breaks the equal view-shard).")
        return self._sparse_forward_project_single_device(
            voxel_values, pixel_indices, view_indices=view_indices,
            output_device=output_device)

    def _sparse_forward_project_single_device(self, voxel_values, pixel_indices,
                                              view_indices=None, output_device=None):
        """
        Single-device forward projection (the original prerelease implementation).

        See :meth:`sparse_forward_project` for the full argument description.  This
        method carries the unchanged single-device body so the sharded path can be
        added alongside it without disturbing single-device behavior.
        """
        # Batch the views and pixels for possible transfer to the gpu
        transfer_view_batch_size = self.view_batch_size_for_vmap
        transfer_pixel_batch_size = self.transfer_pixel_batch_size
        sinogram_shape = self.get_params('sinogram_shape')
        if view_indices is None:
            view_indices = jnp.arange(sinogram_shape[0])
        num_view_batches = jnp.ceil(sinogram_shape[0] / transfer_view_batch_size).astype(int)
        view_indices_batched = jnp.array_split(view_indices, num_view_batches)
        sinogram_shape = self.get_params('sinogram_shape')

        num_pixels = len(pixel_indices)
        pixel_batch_boundaries = np.arange(start=0, stop=num_pixels, step=transfer_pixel_batch_size)
        pixel_batch_boundaries = np.append(pixel_batch_boundaries, num_pixels)

        sinogram = []
        for view_indices_batch in view_indices_batched:
            sinogram_views = jnp.zeros((len(view_indices_batch), *sinogram_shape[1:]), device=self.sinogram_device)
            # Loop over pixel batches
            for k, pixel_index_start in enumerate(pixel_batch_boundaries[:-1]):
                # Send a batch of pixels to sinogram_device
                pixel_index_end = pixel_batch_boundaries[k + 1]
                voxel_batch, pixel_index_batch = jax.device_put([voxel_values[pixel_index_start:pixel_index_end],
                                                                 pixel_indices[pixel_index_start:pixel_index_end]],
                                                                self.sinogram_device)
                sinogram_views = sinogram_views.block_until_ready()
                sinogram_views = sinogram_views + self.projector_functions.sparse_forward_project(voxel_batch, pixel_index_batch, view_indices=view_indices_batch)

            # Include these views in the sinogram
            sinogram.append(jax.device_put(sinogram_views, output_device))

        sinogram = jnp.concatenate(sinogram)
        return sinogram

    def _sharded_forward_project_setup(self, voxel_values, pixel_indices, view_indices):
        """Shared setup for sharded forward projection (adjoint of the back setup).

        Validates the contract (no view subset), defensively slice-shards the
        recon cylinders (a no-op when already slice-sharded), and builds the
        per-device data the band streaming needs.

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
        if view_indices is not None:
            raise NotImplementedError(
                "Sharded forward projection currently requires view_indices=None "
                "(the full view-sharded sinogram).")
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

    def _sparse_forward_project_sharded(self, voxel_values, pixel_indices, view_indices=None):
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
            view_indices: must be None in sharded mode.

        Returns:
            jax array of shape sinogram_shape, view-sharded across the mesh.
        """
        (devices, n_dev, num_padded_views, num_slices, num_pixels,
         recon_shard_info, view_ranges, local_pixels) = \
            self._sharded_forward_project_setup(voxel_values, pixel_indices, view_indices)

        # Each slice-owner owns a contiguous block of slices_per_dev slices; that
        # block is streamed in BANDS (contiguous sub-ranges) so each view-owner
        # holds only one band at a time, not the whole gathered cylinder.  Band
        # sizing mirrors back projection (see _slice_band_length); forward's
        # transient is even smaller (no n_dev-way gather), so the back sizing is a
        # safe, conservative reuse.
        slices_per_dev = num_slices // n_dev
        band_len = self._slice_band_length(
            slices_per_dev, n_dev, num_pixels,
            fixed_band=getattr(self, 'forward_project_slice_band', None))
        band_bounds = self._balanced_slice_bounds(slices_per_dev, band_len)

        owned_views = self._forward_project_all_bands(
            band_bounds, recon_shard_info, view_ranges, local_pixels, devices)

        # Zero the padded views (if any) on their owners.  This is THE mask site that
        # keeps padding inert: with the entry zero-fill it establishes the invariant
        # that padded views of every sinogram-domain array are identically zero.
        owned_views = self._mask_padded_views(owned_views)

        # Wrap the per-view-owner shards as one view-sharded sinogram (no movement).
        # The global view axis is the DEVICE-FORM count; rows/channels come from the
        # params (unpadded axes).
        sinogram_shape = self.get_params('sinogram_shape')
        return mjs.assemble_sharded(
            owned_views, (num_padded_views, *sinogram_shape[1:]),
            self.sino_placement.shard_structure(3))

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
        """All-ones sinogram in the device form, with any padded views ZERO.

        The constant-weights Hessian computation back-projects a ones sinogram;
        on a padded view axis the padded views must contribute nothing, so their
        entries are zero (mirroring the entry zero-fill).  Built per-shard
        directly on each owner device (no host array, no data movement, no
        sharded-array reference cycle).  Without padding this is just
        ``ones_like``.

        Args:
            sino_like (jax array): a device-form sinogram supplying shape, dtype,
                and (when sharded) the target sharding.

        Returns:
            A device-form all-ones (real views) / zeros (padded views) array.
        """
        if not (self.is_sharded and self.sino_placement.is_padded):
            return jnp.ones_like(sino_like)
        axis = self.sinogram_shard_axis() % sino_like.ndim
        pieces = []
        for dev, (v0, v1), n_valid in self.sino_placement.padded_shard_ranges():
            block = v1 - v0
            parts = []
            if n_valid > 0:
                shape_valid = list(sino_like.shape)
                shape_valid[axis] = n_valid
                parts.append(jnp.ones(tuple(shape_valid), dtype=sino_like.dtype, device=dev))
            if block - n_valid > 0:
                shape_tail = list(sino_like.shape)
                shape_tail[axis] = block - n_valid
                parts.append(jnp.zeros(tuple(shape_tail), dtype=sino_like.dtype, device=dev))
            pieces.append(parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=axis))
        return mjs.assemble_sharded(pieces, sino_like.shape,
                                    self.sino_placement.shard_structure(sino_like.ndim))

    def _sino_device_shape(self):
        """The sinogram shape as it exists ON THE DEVICES: the params shape with the
        view axis at its device-form (possibly padded) length.  Equals the params
        shape exactly when the view axis divides the device count (or when not
        sharded).  Use for validating device-form arrays; the params answer "what
        is the problem?", this answers "what is on the devices?"."""
        shape = list(self.get_params('sinogram_shape'))
        if self.is_sharded:
            shape[self.sinogram_shard_axis() % len(shape)] = self.sino_placement.padded_size
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
                band, local_pixels[i], view_indices=view_ranges[device])
        return mjs.run_per_device(devices, worker, executor=pool)

    def sparse_back_project(self, sinogram, pixel_indices, view_indices=None, coeff_power=1, output_device=None):
        """
        Back project the given sinogram to the voxels given by the indices.  The sinogram should be the full sinogram
        associated with all of the angles used to define the ct model, even if a set of view_indices is provided.
        The indices are into a flattened 2D array of shape (recon_rows, recon_cols), and the projection is done using
        all voxels with those indices across all the slices.  If view_indices is a jax array of ints, then they should
        be the indices into the sinogram that is passed in here.

        Args:
            sinogram (jnp array): 3D jax array containing the full sinogram.
            pixel_indices (jnp array): Array of indices specifying which voxels to back project.
            view_indices (jax array): Array of indices of views to project.  These are indices into the first axis of sinogram.
            coeff_power (int, optional): Normally 1, but set to 2 for Hessian diagonal
            output_device (jax device, optional): Device on which to put the output

        Returns:
            A jax array of shape (len(indices), num_slices)
        """
        # When a mesh is configured, take the sharded path: the
        # sinogram is expected view-sharded and the result is returned
        # slice-sharded (no gather here; the user-facing back_project gathers).
        # Otherwise the single-device path runs unchanged.
        #
        # Partial-view back projection (view_indices) is single-device-only (a subset of views
        # breaks the equal view-shard).  On a trivial 1-device mesh (the auto-default placement
        # path) there is nothing to shard, so fall through to the single-device implementation;
        # only a real multi-device mesh rejects view_indices.  (e.g. vcls runs single-device.)
        if self.is_sharded and view_indices is None:
            return self._sparse_back_project_sharded(
                sinogram, pixel_indices, view_indices=None,
                coeff_power=coeff_power)
        if self.is_sharded and len(self.shard_devices) > 1:
            raise NotImplementedError(
                "Sharded back projection with view_indices is not supported on a "
                "multi-device mesh (a subset of views breaks the equal view-shard).")
        return self._sparse_back_project_single_device(
            sinogram, pixel_indices, view_indices=view_indices,
            coeff_power=coeff_power, output_device=output_device)

    def _sparse_back_project_single_device(self, sinogram, pixel_indices, view_indices=None,
                                           coeff_power=1, output_device=None):
        """
        Single-device back projection (the original prerelease implementation).

        See :meth:`sparse_back_project` for the full argument description.  This
        method carries the unchanged single-device body so the sharded
        path can be added alongside it without disturbing single-device behavior.
        """
        # Batch the views and pixels for possible transfer to the gpu
        transfer_view_batch_size = self.view_batch_size_for_vmap
        transfer_pixel_batch_size = self.transfer_pixel_batch_size
        num_views = sinogram.shape[0]
        if view_indices is None:
            view_indices = jnp.arange(num_views)
        num_view_batches = jnp.ceil(sinogram.shape[0] / transfer_view_batch_size).astype(int)
        view_indices_batched = jnp.array_split(view_indices, num_view_batches)

        pixel_indices = jax.device_put(pixel_indices, self.sinogram_device)
        num_pixel_batches = jnp.ceil(pixel_indices.shape[0] / transfer_pixel_batch_size).astype(int)
        pixel_indices_batched = jnp.array_split(pixel_indices, num_pixel_batches)

        recon_shape = self.get_params('recon_shape')
        num_pixels = len(pixel_indices)
        num_slices = recon_shape[2]

        # Get the final recon as a jax array
        recon_at_indices = jnp.zeros((num_pixels, num_slices), device=output_device)
        for view_indices_batch in view_indices_batched:
            view_batch = sinogram[view_indices_batch]
            view_batch = jax.device_put(view_batch, self.sinogram_device)

            # Loop over pixel batches
            voxel_batch_list = []
            for pixel_index_batch in pixel_indices_batched:
                # Back project a batch
                voxel_batch = self.projector_functions.sparse_back_project(view_batch, pixel_index_batch,
                                                                           view_indices=view_indices_batch,
                                                                           coeff_power=coeff_power)
                voxel_batch = voxel_batch.block_until_ready()
                voxel_batch_list.append(jax.device_put(voxel_batch, output_device))

            recon_at_indices = recon_at_indices + jnp.concatenate(voxel_batch_list, axis=0)

        return recon_at_indices

    def _sharded_back_project_setup(self, sinogram, pixel_indices, view_indices):
        """Shared setup for the sharded back-projection paths.

        Validates the view-sharded contract (no view subset), defensively shards
        the sinogram (a no-op when already view-sharded, so internal callers that
        pass a plain array -- e.g. compute_hessian_diagonal with plain weights --
        still work), and builds the data needed for each view-owner.

        Returns:
            (devices, n_dev, num_slices, num_pixels, shard_info, local_pixels):
            ``shard_info`` maps each view-owner to ``(its local view-shard, the
            GLOBAL view indices it covers)`` so the projector picks the matching
            angles; ``local_pixels`` is ``pixel_indices`` placed on each view-owner
            once.
        """
        if view_indices is not None:
            # A view subset would mean filtering each view-owner's view-shard to the
            # requested views and adjusting the angle lookup per shard; not needed
            # by the consumers (back_project / direct_recon pass all views).
            raise NotImplementedError(
                "Sharded back projection currently requires view_indices=None "
                "(the full view-sharded sinogram).")
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
        num_slices = self.get_params('recon_shape')[2]
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

    def _sparse_back_project_sharded(self, sinogram, pixel_indices, view_indices=None,
                                     coeff_power=1):
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
            view_indices: must be None in sharded mode (the full view-sharded
                sinogram is expected).  A view subset would require per-shard
                filtering or masking and is deferred.
            coeff_power (int): 1 normally, 2 for the Hessian diagonal.

        Returns:
            jax array of shape (len(pixel_indices), num_slices), slice-sharded
            across the mesh.
        """
        devices, n_dev, num_slices, num_pixels, shard_info, local_pixels = \
            self._sharded_back_project_setup(sinogram, pixel_indices, view_indices)

        # Each device owns a slice-shard, a contiguous range of slices_per_dev slices
        # (exact, since configure_sharding requires num_slices % n_dev == 0).  That
        # slice-shard is streamed in BANDS -- contiguous sub-ranges -- so the
        # slice-owner never gathers all n_dev partials for the whole slice-shard at once
        # (thereby limiting intermediate memory).  By default, a band is
        # ~slices_per_dev/n_dev (so ~n_dev bands per slice-owner); it equals the full
        # slice-shard only in the degenerate one-band case.
        slices_per_dev = num_slices // n_dev
        band_len = self._slice_band_length(
            slices_per_dev, n_dev, num_pixels,
            fixed_band=getattr(self, 'back_project_slice_band', None))
        band_bounds = self._balanced_slice_bounds(slices_per_dev, band_len)

        # Do the back projection:
        #   a double loop over slice-owners on the outside and bands on the inside
        # owned[t] = slice-owner t's recon slice-shard assembled from its bands.
        owned = self._back_project_all_bands(
            slices_per_dev, band_bounds, shard_info, local_pixels, coeff_power,
            devices)

        # Wrap the shards from each slice-owner into one slice-sharded array (no data movement).
        return mjs.assemble_sharded(
            owned, (num_pixels, num_slices),
            self.recon_placement.shard_structure(2))

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
            return self.projector_functions.sparse_back_project(
                data[:, g0:g1, :], local_pixels[i],
                view_indices=global_view_idx, coeff_power=coeff_power)
        return mjs.run_per_device(devices, worker, executor=pool)

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

    def compute_hessian_diagonal(self, weights=None, output_device=None):
        """
        Computes the diagonal of the Hessian matrix, which is computed by doing a backprojection of the weight
        matrix except using the square of the coefficients in the backprojection to a given voxel.
        If weights is not None, it must be an array with the same shape as the sinogram to be backprojected.
        If weights is None, then constant weights 1 will be used

        Args:
            weights (jax array, optional): 3D positive weights with same shape as sinogram.  Defaults to all 1s.
            output_device (jax device): Device on which to put the output

        Returns:
            jnp array: Diagonal of the Hessian matrix with same shape as recon.
        """
        sinogram_shape, recon_shape = self.get_params(['sinogram_shape', 'recon_shape'])
        num_views = sinogram_shape[0]
        if weights is None:
            # Plain ones in the REAL shape: the sharded back projection's entry placement
            # zero-pads any padded views, which is exactly the inert weighting they need.
            with jax.default_device(self.main_device):
                weights = jnp.ones((num_views,) + sinogram_shape[1:])
        elif tuple(weights.shape) not in (tuple(sinogram_shape), self._sino_device_shape()):
            # Accept the problem shape (plain weights) or the device-form shape (weights
            # already placed, with a possibly padded view axis).
            error_message = 'Weights must be constant or an array compatible with sinogram'
            error_message += '\nGot weights.shape = {}, but sinogram.shape = {}'.format(weights.shape, sinogram_shape)
            raise ValueError(error_message)

        num_recon_rows, num_recon_cols, num_recon_slices = recon_shape[:3]
        max_index = num_recon_rows * num_recon_cols
        indices = jnp.arange(max_index)
        hessian_diagonal = self.sparse_back_project(weights, indices, coeff_power=2, output_device=output_device)

        return hessian_diagonal.reshape((num_recon_rows, num_recon_cols, num_recon_slices))

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
            error_message += " 'full' (use gpu for all calculations),\n"
            error_message += " 'none' (do not use gpu at all)."
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
            max_views_to_use = np.minimum(20, sinogram.shape[0])
            step_size = sinogram.shape[0] // max_views_to_use

            small_sinogram = np.array(sinogram[::step_size])
            if weights is None:
                small_weights = 1
            else:
                small_weights = np.array(weights[::step_size])

            # Use only the REAL views.  A device-form input (prepare_sino_for_devices) may carry a
            # zero-padded view axis; the padded views are inert in the recon and must not bias the
            # regularization statistics (e.g. the all-ones indicator fallback would average them in).
            # Crop the host-side subsample by global view index -- no device-side work.
            num_real_views = self.get_params('sinogram_shape')[0]
            if sinogram.shape[0] != num_real_views:
                keep = np.arange(0, sinogram.shape[0], step_size) < num_real_views
                small_sinogram = small_sinogram[keep]
                if weights is not None:
                    small_weights = small_weights[keep]
            # Compute indicator function for sinogram support
            sino_indicator = self._get_sino_indicator(small_sinogram)
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
        recon_shape = self.get_params('recon_shape')

        # Flatten the recon along the first two dimensions, then retrieve values of recon at the indices locations
        voxel_values = recon.reshape((-1,) + recon_shape[2:])[indices]

        return voxel_values

    @staticmethod
    def _get_sino_indicator(sinogram):
        """
        Compute a binary mask that indicates the region of sinogram support.

        Args:
            sinogram (ndarray): 3D jax array containing sinogram with shape (num_views, num_det_rows, num_det_channels).

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
            warnings.warn("Sinogram contains no positive values. This may lead to a contrast reversed reconstruction.")
            indicator = np.ones_like(sinogram, dtype=np.int8)
            return indicator

        if max_sino < threshold:
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

    def direct_recon(self, sinogram, filter_name=None, view_batch_size=DIRECT_RECON_VIEW_BATCH_SIZE,
                     output_sharded=False):
        """
        Do a direct (non-iterative) reconstruction, typically using a form of filtered backprojection.  The
        implementation details are geometry specific, and direct_recon may not be available for all geometries.

        Args:
            sinogram (ndarray or jax array): 3D sinogram data with shape (num_views, num_det_rows, num_det_channels).
            filter_name (string or None, optional): The name of the filter to use, defaults to None, in which case the geometry specific method chooses a default, typically 'ramp'.
            view_batch_size (int, optional): An integer specifying the size of a view batch to limit memory use.  Defaults to 100.
            output_sharded (bool, optional): If False (default), return a plain array.  If True, return
                the internal device form (slice-sharded on a sharded model; on an unsharded model the
                output is the same single-device array either way).

        Returns:
            recon (jax array): The reconstructed volume after direct reconstruction.
        """
        warnings.warn('direct_recon not implemented for TomographyModel.')
        recon_shape = self.get_params('recon_shape')
        return jnp.zeros(recon_shape, device=self.main_device)

    def direct_filter(self, sinogram, filter_name=None, view_batch_size=DIRECT_RECON_VIEW_BATCH_SIZE,
                      output_sharded=False):
        """
        Perform filtering on the given sinogram as needed for an FBP/FDK or other direct recon.

        Args:
            sinogram (jax array): The input sinogram with shape (num_views, num_rows, num_channels).
            filter_name (string, optional): Name of the filter to be used. Defaults to "ramp"
            view_batch_size (int, optional):  Size of view batches (used to limit memory use)
            output_sharded (bool, optional): If False (default), return a plain array.  If True, return
                the internal device form (view-sharded on a sharded model; on an unsharded model the
                output is the same single-device array either way).

        Returns:
            filtered_sinogram (jax array): The sinogram after FBP filtering.
        """
        warnings.warn('direct_filter not implemented for TomographyModel.')
        return jnp.zeros_like(sinogram, device=self.sinogram_device)

    def initialize_recon(self, sinogram, weights=None, init_recon=None, max_iterations=15, first_iteration=0,
                         compute_prior_loss=False, logfile_path='./logs/recon.log', print_logs=True):
        """
        Do the device management and parameter initialization needed for recon and prox_map.

        Args:
            See :meth:`recon` for arguments.

        Returns:
            sinogram, weights, init_recon, partitions, partition_sequence, granularity, regularization_params
        """

        # Initialize logging for this run
        if first_iteration == 0 or self.logger is None:
            self.setup_logger(logfile_path=logfile_path, print_logs=print_logs)
        self.logger.info('MBIRJAX Version = {}'.format(self.version))
        self.logger.info('Reconstruction devices: {}'.format(self._device_report()))

        # Generate set of voxel partitions
        recon_shape, granularity, use_ror_mask = self.get_params(['recon_shape', 'granularity', 'use_ror_mask'])
        partitions = mj.gen_set_of_pixel_partitions(recon_shape, granularity, output_device=self.main_device, use_ror_mask=use_ror_mask)
        partitions = [jax.device_put(partition, self.main_device) for partition in partitions]

        # Generate sequence of partitions to use
        partition_sequence = self.get_params('partition_sequence')
        partition_sequence = mj.gen_partition_sequence(partition_sequence, max_iterations=max_iterations)
        partition_sequence = partition_sequence[first_iteration:]
        regularization_params = None

        try:
            # Check that sinogram and weights are not taking up GPU space.  Arrays already in a
            # multi-device NamedSharding (e.g. prepared by prepare_sino_for_devices) are left in
            # place: a device_put to a single device would silently GATHER them, defeating the
            # prepared placement; vcd_recon's entry placement handles them directly.
            def _committed_elsewhere(x, target_device):
                return (isinstance(x, type(jnp.zeros(1)))
                        and not isinstance(getattr(x, 'sharding', None), jax.sharding.NamedSharding)
                        and list(x.devices())[0] != target_device)

            if _committed_elsewhere(sinogram, self.sinogram_device):
                sinogram = jax.device_put(sinogram, self.sinogram_device)
            if weights is not None and _committed_elsewhere(weights, self.sinogram_device):
                weights = jax.device_put(weights, self.sinogram_device)
            if init_recon is not None and _committed_elsewhere(init_recon, self.main_device):
                init_recon = jax.device_put(init_recon, self.main_device)

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
            # configure_sharding under use_gpu='automatic'), and a CPU OOM must get CPU guidance.
            recon_devices = self._recon_devices()
            on_gpu = bool(recon_devices) and recon_devices[0] is not None \
                and self._platform_label(recon_devices[0]) == 'GPU'
            log_oom_guidance(self.logger, on_gpu=on_gpu)
        raise e

    def recon(self, sinogram, weights=None, init_recon=None, max_iterations=15, stop_threshold_change_pct=0.2, first_iteration=0,
              compute_prior_loss=False, logfile_path='./logs/recon.log', print_logs=True, output_sharded=False):
        """
        Perform MBIR reconstruction using the Multi-Granular Vector Coordinate Descent algorithm.
        This function takes care of generating its own partitions and partition sequence.
        TO restart a recon using the same partition sequence, set first_iteration to be the number of iterations
        completed so far, and set init_recon to be the output of the previous recon.  This will continue using
        the same partition sequence from where the previous recon left off.

        Args:
            sinogram (ndarray or jax array): 3D sinogram data with shape (num_views, num_det_rows, num_det_channels).
            weights (ndarray or jax array, optional): 3D positive weights with same shape as error_sinogram.  Defaults to None, in which case the weights are implicitly all 1.
            init_recon (jax array or None or 0, optional): Initial reconstruction to use in reconstruction. If None, then direct_recon is called with default arguments.  Defaults to None.
            max_iterations (int, optional): maximum number of iterations of the VCD algorithm to perform.
            stop_threshold_change_pct (float, optional): Stop reconstruction when 100 * ||delta_recon||_1 / ||recon||_1 change from one iteration to the next is below stop_threshold_change_pct.  Defaults to 0.2.  Set this to 0 to guarantee exactly max_iterations.
            first_iteration (int, optional): Set this to be the number of iterations previously completed when restarting a recon using init_recon.  This defines the first index in the partition sequence.  Defaults to 0.
            compute_prior_loss (bool, optional):  Set true to calculate and return the prior model loss.  This will lead to slower reconstructions and is meant only for small recons.
            logfile_path (str, optional): Path to the output log file.  Defaults to './logs/recon.log'.
            print_logs (bool, optional): If true then print logs to console.  Defaults to True.
            output_sharded (bool, optional): If False (default), return a plain reconstruction array.
                If True, return the internal device form (slice-sharded across the model's devices,
                no exit gather); on an unsharded model the output is the same either way.

        Returns:
            (recon, recon_dict): reconstruction array and a dict containing the recon parameters.
                - recon (jax array): the reconstruction volume
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
            self.logger.info('Logs written to {}'.format(os.path.abspath(logfile_path)))

        for h in list(self.logger.handlers):  # Make sure the log files are up to date
            h.flush()

        notes = 'Reconstruction completed: {}\n\n'.format(datetime.datetime.now())
        recon_dict = self.get_recon_dict(recon_params, notes=notes)
        if not output_sharded:
            # Default exit: gather to a plain single-device array (a no-op placement
            # on an unsharded model; the exit gather on a sharded one).
            recon = jax.device_put(recon, device=self.main_device)
        return recon, recon_dict

    def vcd_recon(self, sinogram, partitions, partition_sequence, stop_threshold_change_pct, weights=None,
                  init_recon=None, prox_input=None, compute_prior_loss=False, first_iteration=0):
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
        Returns:
            (recon, recon_stats): tuple of 3D reconstruction and a tuple containing arrays of per-iteration stats.
            recon_stats = (fm_rmse, pm_rmse, nrms_update), where fm is forward model, pm is prior model, and
            nrms_update is ||recon(i+1) - recon(i)||_2 / ||recon(i+1)||_2.

        Note:
            To maximize GPU memory, each of sinogram, weights, init_recon, and prox_input should be on the CPU for large recons.
        """
        # Ensure that everything has the right shape and is on the main device
        self.verify_valid_params()

        # Placement helpers.  When sharding is on, recon-like arrays are slice-sharded and sino-like
        # arrays are view-sharded; otherwise each is committed to main/sinogram_device.  Routing every
        # placement through these keeps the rest of the loop placement-agnostic and (crucially) avoids
        # committing a sharded array to a single device, which would silently gather it.
        def to_sino(x):
            return self._shard_sinogram(x) if self.is_sharded else jax.device_put(x, self.sinogram_device)

        def to_recon(x):
            return self._shard_recon(x) if self.is_sharded else jax.device_put(x, self.main_device)

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
        sinogram = to_sino(sinogram)

        scale_recon_to_sinogram = True if init_recon is None else False
        if init_recon is None:
            # Initialize VCD recon, and error sinogram.  output_sharded=True keeps the init in
            # the internal device form (slice-sharded when sharding is on; no gather).
            self.logger.info('Starting direct recon for initial reconstruction')
            init_recon = self.direct_recon(sinogram, output_sharded=True)
        elif isinstance(init_recon, int):
            init_recon = init_recon * jnp.ones(recon_shape, device=self.main_device)

        # Make sure that init_recon has the correct shape and type
        if init_recon.shape != recon_shape:
            error_message = "init_recon does not have the correct shape. \n"
            error_message += "Expected {}, but got shape {} for init_recon shape.".format(recon_shape,
                                                                                          init_recon.shape)
            self.logger.error(error_message)
            raise ValueError(error_message)

        # Place the init as a recon-like array (slice-sharded when sharding is on); a no-op when
        # direct_recon already returned it sharded.
        init_recon = to_recon(init_recon)

        # Initialize VCD recon and error sinogram using the init_recon
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
        recon = to_recon(recon)  # commit to main_device (single device) or keep slice-sharded (sharded path)
        error_sinogram = to_sino(error_sinogram)

        # Test to make sure the prox_input input is correct
        if prox_input is not None:
            # Make sure that prox_input has the correct size
            if prox_input.shape != recon.shape:
                error_message = "prox_input does not have the correct size. \n"
                error_message += "Expected {}, but got shape {} for prox_input shape.".format(recon.shape,
                                                                                              prox_input.shape)
                self.logger.error(error_message)
                raise ValueError(error_message)

            with jax.default_device(self.main_device):
                prox_input = jnp.array(prox_input.reshape((-1, num_recon_slices)))
            prox_input = jax.device_put(prox_input, self.main_device)

        # Get required parameters
        verbose, sigma_y = self.get_params(['verbose', 'sigma_y'])

        # The REAL sinogram element count (from the params, which always hold the problem's
        # shapes).  Equals the device arrays' size except when the view axis is padded for
        # sharding; normalizing by the real count keeps the reported losses independent of the
        # (inert, identically-zero) padding.
        real_sino_size = int(np.prod(self.get_params('sinogram_shape')))
        pad_active = self.is_sharded and self.sino_placement.is_padded
        loss_num_real = real_sino_size if pad_active else None

        # Initialize the diagonal of the hessian of the forward model
        if constant_weights:
            # Ones over the real views, ZEROS over any padded views (device form):
            # padded views must not contribute to the Hessian back projection.
            weights = self._sino_ones_device_form(sinogram)

        self.logger.info('Computing Hessian diagonal')
        fm_hessian = self.compute_hessian_diagonal(weights=weights, output_device=self.main_device)
        fm_hessian = fm_hessian.reshape((-1, num_recon_slices))
        if constant_weights:
            weights = 1
        else:
            weights = to_sino(weights)

        # Initialize the emtpy recon
        flat_recon = recon.reshape((-1, num_recon_slices))
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

            # Compute the stats and display as desired
            fm_rmse[i] = self.get_forward_model_loss(error_sinogram, sigma_y, weights,
                                                     num_real_elements=loss_num_real)
            nmae_update[i] = ell1_for_partition / jnp.sum(jnp.abs(flat_recon))
            # real_sino_size == error_sinogram.size except under view padding, where the
            # padded entries are identically zero and must not dilute the RMSE.
            es_rmse = jnp.linalg.norm(error_sinogram) / jnp.sqrt(float(real_sino_size))
            alpha_values[i] = alpha

            if verbose >= 1:
                iter_output = '\nAfter iteration {} of a max of {}: Pct change={:.4f}, Forward loss={:.4f}'.format(i + first_iteration, max_iters + first_iteration,
                                                                                                  100 * nmae_update[i],
                                                                                                  fm_rmse[i])
                if compute_prior_loss:
                    qggmrf_nbr_wts, sigma_x, p, q, T = self.get_params(['qggmrf_nbr_wts', 'sigma_x', 'p', 'q', 'T'])
                    b = mj.get_b_from_nbr_wts(qggmrf_nbr_wts)
                    qggmrf_params = (b, sigma_x, p, q, T)
                    pm_loss[i] = mj.qggmrf_loss(flat_recon.reshape(recon.shape), qggmrf_params)
                    pm_loss[i] /= flat_recon.size
                    # Each loss is scaled by the number of elements, but the optimization uses unscaled values.
                    # To provide an accurate, yet properly scaled total loss, first remove the scaling and add,
                    # then scale by the average number of elements between the two.
                    total_loss = ((fm_rmse[i] * real_sino_size + pm_loss[i] * flat_recon.size) /
                                  (0.5 * (real_sino_size + flat_recon.size)))
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

        return self.reshape_recon(flat_recon), (fm_rmse[0:num_iters], pm_loss[0:num_iters], nmae_update[0:num_iters],
                                                alpha_values[0:num_iters])

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
        partition_worker = jax.device_put(partition, self.sinogram_device)
        # Stage the qGGMRF boundary halos ONCE for this whole partition pass and reuse them
        # for every subset.  The prior couples a voxel only to its same-pixel cross-shard
        # slice neighbor, and the partition's subsets are (almost) disjoint, so a subset's
        # halo at its own pixels is unchanged until that subset runs -- hence pass-start
        # halos are correct for each subset.  This turns the per-subset host halo read (the
        # main per-subset host round-trip, which caps multi-GPU scaling) into a per-pass one.
        # (Caveat: gen_pixel_partition replicates a few pixels to equalize subset lengths, so
        # this is not strictly bit-exact at those pixels -- quantified by test; negligible.
        # Set self._vcd_halo_per_subset = True to restore per-subset extraction for A/B.)
        stage_per_pass = (self.is_sharded
                          and not getattr(self, '_vcd_halo_per_subset', False))
        staged_halos = self._stage_halos(flat_recon) if stage_per_pass else None
        for index in subset_indices:
            subset = partition[index]
            subset_worker = partition_worker[index]
            flat_recon, error_sinogram, ell1_for_subset, alpha_for_subset = vcd_subset_updater(
                flat_recon, error_sinogram, subset, subset_worker, staged_halos)
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

        def vcd_subset_updater(flat_recon, error_sinogram, pixel_indices, pixel_indices_worker,
                               staged_halos=None):
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
                pixel_indices_worker (jax array): Same as pixel_indices, but copied onto the worker device.
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
            # the index array must live on the same devices as the array, so in the sharded path
            # replicate the indices across the recon mesh (PartitionSpec() == fully replicated); the
            # single-device path leaves them on main_device unchanged.
            if self.is_sharded:
                recon_indices = jax.device_put(
                    pixel_indices,
                    jax.sharding.NamedSharding(self.recon_placement.mesh,
                                               jax.sharding.PartitionSpec()))
            else:
                recon_indices = pixel_indices

            # Compute the prior model gradient and hessian (i.e., second derivative) terms
            if prox_input is None:

                # qGGMRF prior - compute the qggmrf gradient and hessian at each pixel in the index set.
                if self.is_sharded:
                    # Sharded path: flat_recon is slice-sharded, so compute the prior per slice-owner
                    # with halo exchange for the inter-slice term, keeping the result slice-sharded.
                    # staged_halos (staged once per partition pass) avoids re-reading the halos here.
                    prior_grad, prior_hess = self._qggmrf_prior_sharded(
                        flat_recon, pixel_indices, qggmrf_params, staged_halos=staged_halos)
                else:
                    with jax.default_device(self.main_device):
                        prior_grad, prior_hess = (
                            mj.qggmrf_gradient_and_hessian_at_indices(flat_recon, recon_shape, pixel_indices,
                                                                      qggmrf_params))
            else:
                # Proximal map prior - compute the prior model gradient at each pixel in the index set.
                prior_hess = 1 / (sigma_prox ** 2)
                prior_grad = mj.prox_gradient_at_indices(flat_recon, prox_input, pixel_indices, sigma_prox)

            if not const_weights:
                weighted_error_sinogram = weights * error_sinogram  # Note that fm_constant will be included below
            else:
                weighted_error_sinogram = error_sinogram

            # Back project to get the gradient
            forward_grad = - fm_constant * sparse_back_project(weighted_error_sinogram, pixel_indices_worker,
                                                               output_device=self.main_device)

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

            # Compute update direction in sinogram domain
            delta_sinogram = sparse_forward_project(delta_recon_at_indices, pixel_indices_worker,
                                                    output_device=self.sinogram_device)

            forward_linear, forward_quadratic = self.get_forward_lin_quad(
                weighted_error_sinogram, delta_sinogram, weights, fm_constant, const_weights,
                # Sharded: leave the forward scalars where the reduction produced them (the sino
                # mesh) and reconcile meshes on-device below; single-device: commit to main_device.
                output_device=(None if self.is_sharded else self.main_device))

            # Compute optimal update step.
            # The forward-model line-search scalars are reduced over the view-sharded sinogram (the
            # sino mesh) while the prior scalars are reduced over the slice-sharded recon (the recon
            # mesh); combining device scalars across two distinct meshes is not allowed.  Rather than
            # bounce all four through the host (5 device->host syncs/subset that stall the GPU), keep
            # the line search ON-DEVICE: replicate the forward scalars onto the recon mesh (a cheap
            # scalar reshard over the same devices), do the arithmetic there, and replicate the
            # resulting alpha onto the sino mesh below to scale the sino-sharded delta.
            if self.is_sharded:
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
                # In the sharded path delta_recon_at_indices is already slice-sharded to match
                # flat_recon; committing it to a single device here would gather it, so only place
                # it on main_device in the single-device path.
                if not self.is_sharded:
                    delta_recon_at_indices = jax.device_put(delta_recon_at_indices, self.main_device)
                delta_recon_at_indices = jnp.maximum(-pos_constant * recon_at_indices, delta_recon_at_indices)

                # Recompute sinogram projection
                delta_sinogram = sparse_forward_project(delta_recon_at_indices, pixel_indices, output_device=self.sinogram_device)

            # Perform sparse updates at index locations.  In the sharded path delta_recon_at_indices
            # is already slice-sharded to match flat_recon (so update_recon stays a local scatter);
            # committing it to a single device would gather it, so only do that single-device.
            if not self.is_sharded:
                delta_recon_at_indices = jax.device_put(delta_recon_at_indices, self.main_device)
            delta_recon_at_indices = alpha * delta_recon_at_indices

            flat_recon = update_recon(flat_recon, recon_indices, delta_recon_at_indices)

            # Update sinogram and loss
            # Update the error sinogram: error_sinogram <- error_sinogram - alpha * delta_sinogram.
            if self.is_sharded:
                # Update the error sinogram IN PLACE via a buffer-DONATING fused multiply-add
                # (alpha replicated onto the sino mesh so the scale stays on-device).  In-place
                # reuse is required because an out-of-place per-subset update allocates a fresh
                # view-sharded error sinogram each subset, and the stale ones accumulate in jax's
                # internal sharded-array reference cycles until gc.  For constant weights
                # weighted_error_sinogram IS error_sinogram, so release that alias to make
                # error_sinogram donatable.  (Non-constant weights leave a weighted product
                # transient that is freed in the cleanup section at the end of the subset.)
                if const_weights:
                    weighted_error_sinogram = None
                error_sinogram = update_error_sinogram(
                    error_sinogram, self._replicate_scalar(alpha, self.sino_placement), delta_sinogram)
            else:
                # Single-device path unchanged (SingleDeviceSharding arrays free on refcount).
                delta_sinogram = float(alpha) * delta_sinogram
                error_sinogram = error_sinogram - delta_sinogram

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
            if self.is_sharded and not const_weights:
                jax.block_until_ready((flat_recon, error_sinogram))
                weighted_error_sinogram.delete()

            return flat_recon, error_sinogram, ell1_for_subset, alpha_for_subset

        return vcd_subset_updater

    def get_forward_lin_quad(self, weighted_error_sinogram, delta_sinogram, weights, fm_constant, const_weights,
                             output_device=None):
        """
        Compute forward model terms used in line-search updates:
        ``forward_linear = fm_constant * jnp.sum(weighted_error_sinogram * delta_sinogram)`` and
        ``forward_quadratic = fm_constant * jnp.sum(delta_sinogram * delta_sinogram * weights)``.
        This supports batching to a worker, with only two floats returned per batch.

        Args:
            weighted_error_sinogram (jax array):
            delta_sinogram (jax array):
            weights (jax array or constant):
            fm_constant (constant):
            const_weights (bool): True if the weights are constant 1
            output_device (jax device): device on which the output will be placed

        Returns:
            tuple: ``(forward_linear, forward_quadratic)``
        """
        forward_linear = fm_constant * jnp.sum(weighted_error_sinogram * delta_sinogram)
        forward_quadratic = fm_constant * jnp.sum(delta_sinogram * delta_sinogram * weights)

        forward_linear = jax.device_put(forward_linear, output_device)
        forward_quadratic = jax.device_put(forward_quadratic, output_device)
        return forward_linear, forward_quadratic

    @staticmethod
    def get_forward_model_loss(error_sinogram, sigma_y, weights=None, normalize=True,
                               num_real_elements=None):
        """
        Calculate the loss function for the forward model from the error_sinogram and weights.
        The error sinogram should be error_sinogram = measured_sinogram - forward_proj(recon)

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
        if weights is None:
            weights = 1
            avg_weight = 1
        elif num_real_elements is None:
            avg_weight = jnp.average(weights)
        else:
            avg_weight = jnp.sum(weights) / num_real_elements
        if normalize:
            weighted_sq = (error_sinogram * error_sinogram) * (weights / avg_weight)
            if num_real_elements is None:
                loss = jnp.sqrt((1.0 / (sigma_y ** 2)) * jnp.mean(weighted_sq))
            else:
                loss = jnp.sqrt((1.0 / (sigma_y ** 2)) * (jnp.sum(weighted_sq) / num_real_elements))
        else:
            loss = (1.0 / (2 * sigma_y ** 2)) * jnp.sum((error_sinogram * error_sinogram) * weights)
        return loss

    def prox_map(self, prox_input, sinogram, sigma_prox=None, weights=None, init_recon=None, do_initialization=True, stop_threshold_change_pct=0.2,
                 max_iterations=3, first_iteration=0, logfile_path='./logs/prox.log', print_logs=True, output_sharded=False):
        """
        Proximal Map function for use in Plug-and-Play applications.
        This function is similar to recon, but it essentially uses a prior with a mean of prox_input and a standard deviation of sigma_prox.

        Args:
            prox_input (jax array): proximal map input with same shape as reconstruction.
            sinogram (jax array): 3D sinogram data with shape (num_views, num_det_rows, num_det_channels).
            sigma_prox (None or float, optional): The standard deviation of the proximal map prior term.  If None, then set automatically from the sinogram.  Defaults to None.
            weights (jax array, optional): 3D positive weights with same shape as sinogram.  Defaults to None, in which case the weights are implicitly all 1s.
            init_recon (jax array, optional): optional reconstruction to be used for initialization.  Defaults to None, in which case the initial recon is determined by vcd_recon.
            do_initialization (bool, optional):  If True, then initialize parameters and place arrays on appropriate devices.  Defaults to True.
                Set to False if initialization has already been performed on this sinogram, and prox_input and init_recon are on main_device and sinogram and weights are on sinogram_device.
            stop_threshold_change_pct (float, optional): Stop reconstruction when NMAE percent change from one iteration to the next is below stop_threshold_change_pct.  Defaults to 0.2.
            max_iterations (int, optional): maximum number of iterations of the VCD algorithm to perform.
            first_iteration (int, optional): Set this to be the number of iterations previously completed when restarting a recon using init_recon.  This defines the first index in the partition sequence.  Defaults to 0.
            logfile_path (str, optional): Path to the output log file.  Defaults to './logs/recon.log'.
            print_logs (bool, optional): If true then print logs to console.  Defaults to True.
            output_sharded (bool, optional): If False (default), return a plain reconstruction array.
                If True, return the internal device form (no exit gather); on an unsharded model the
                output is the same either way.

        Returns:
            (recon, recon_dict): reconstruction array and a dict containing the recon parameters.
                - recon (jax array): the reconstruction volume
                - recon_dict (dict): A dict obtained from :meth:`get_recon_dict` with entries
                    * 'recon_params'
                    * 'notes'
                    * 'recon_logs'
                    * 'model_params'
        """
        compute_prior_loss = False
        prior_loss = [0]
        if do_initialization or self.prox_data is None:
            if isinstance(prox_input, type(jnp.zeros(1))) and list(prox_input.devices())[0] != self.main_device:
                prox_input = jax.device_put(prox_input, self.main_device)

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
            self.logger.info('Logs written to {}'.format(os.path.abspath(logfile_path)))

        for h in list(self.logger.handlers):  # Make sure the log files are up to date
            h.flush()

        notes = 'Prox completed: {}\n\n'.format(datetime.datetime.now())
        recon_dict = self.get_recon_dict(recon_params, notes=notes)
        if not output_sharded:
            # Default exit: gather to a plain single-device array (a no-op placement
            # on an unsharded model, where vcd_recon already left it on main_device).
            recon = jax.device_put(recon, device=self.main_device)
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

    def gen_modified_3d_sl_phantom(self):
        """
        DEPRECATED:  This method has been deprecated and will be removed in a future release.
        Instead, use :func:`mbirjax.generate_3d_shepp_logan_low_dynamic_range`

        Generates a simplified, low-dynamic range version of the 3D Shepp-Logan phantom.

        Returns:
            ndarray: A 3D numpy array of shape specified by TomographyModel class parameters.
        """
        warnings.warn('This method has been deprecated and will be removed in a future release.  Instead, use mbirjax.generate_3d_shepp_logan_low_dynamic_range()')
        recon_shape = self.get_params('recon_shape')
        phantom = mj.generate_3d_shepp_logan_low_dynamic_range(recon_shape, device=self.main_device)
        return phantom

    def reshape_recon(self, recon):
        """
        Reshape recon into its 3D form.

        Args:
            recon (ndarray or jax array): A 3D array of shape specified by (num_recon_rows, num_recon_cols, num_recon_slices)
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
    # tolerance (the trivial-mesh path is no longer required to be bit-exact -- see v2 plan P6).
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
