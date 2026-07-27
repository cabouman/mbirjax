"""Segmented VCD driver for the sharpness/snr_db schedule study.

Runs TomographyModel.vcd_recon ONE ITERATION AT A TIME through the checkpoint API
(return_checkpoint / init_error_sinogram / fm_hessian), so per-iteration recon states,
subset-order permutations, and per-granularity sigma overrides are all observable with
NO library changes.  All study runs -- including fixed-parameter baselines -- go
through this driver so every comparison shares one code path (plan: "Driver and RNG
discipline" in plans/sharpness_schedule/sharpness_schedule_plan.md).

RNG discipline
--------------
np.random is seeded ONCE per variant, partitions are generated ONCE (with the same
arguments initialize_recon uses), and each vcd_recon segment then consumes exactly one
subset-order permutation -- so the global np.random stream advances exactly as in a
single continuous vcd_recon call.  Never reseed between segments: that would hand
every iteration the SAME subset order.  equivalence_gate.py verifies the discipline
end-to-end against a continuous run; rerun it after driver or library changes.

The driver also saves the np.random state around each segment and replays it
afterward, which (a) recovers the segment's subset-order permutation exactly (the
footprint probe's input) and (b) VERIFIES the one-permutation-per-segment assumption:
drawing one permutation from the saved state must land exactly on the post-segment
state.

Sigma discipline
----------------
Regularization targets are computed ONCE via auto_set_regularization_params; the
scheduled per-granularity values are then set with set_params(no_warning=True, ...)
before each segment.  vcd_recon reads sigma_x/sigma_y when it builds its subset
updater, so each segment runs at exactly its scheduled values.  Because the driver
never calls recon()/initialize_recon after setup, auto-regularization cannot silently
overwrite the schedule mid-variant.
"""

import math
import time

import numpy as np

import mbirjax as mj  # mbirjax must be imported before jax (sets XLA env vars)
import jax.numpy as jnp


def compute_targets(model, sinogram, weights=None):
    """Run auto-regularization once and return the target (sigma_x, sigma_y).

    Call BEFORE seeding np.random for a variant, so any RNG use inside parameter
    estimation (none known today) could never desynchronize the partition stream.
    Respects manual sigmas: if auto_regularize_flag is False this is a no-op and the
    manual values are returned as the targets.
    """
    model.auto_set_regularization_params(sinogram, weights=weights)
    sigma_x, sigma_y = model.get_params(['sigma_x', 'sigma_y'])
    return float(sigma_x), float(sigma_y)


def scheduled_sigmas(entry, targets, offsets_by_entry):
    """Map a partition-sequence entry to its scheduled (sigma_x, sigma_y).

    offsets_by_entry: {partition-sequence entry -> (d_sharpness, d_snr_db)}, offsets
    RELATIVE to the targets; entries not in the dict (the fine-granularity tail) get
    (0, 0).  The closed-form multipliers mirror auto_set_sigma_x / auto_set_sigma_y
    (sigma_x scales as 2**sharpness, sigma_y as 10**(-snr_db/20)), so offsets apply
    without re-estimating the data-dependent factors -- and compose with manually set
    sigmas the same way.
    """
    d_sharp, d_db = (offsets_by_entry or {}).get(int(entry), (0.0, 0.0))
    sigma_x_target, sigma_y_target = targets
    return (sigma_x_target * 2.0 ** d_sharp,
            sigma_y_target * 10.0 ** (-d_db / 20.0))


def _states_equal(s1, s2):
    """Compare two np.random.get_state() tuples (MT19937: name, key array, pos,
    has_gauss, cached_gaussian)."""
    return (s1[0] == s2[0] and np.array_equal(s1[1], s2[1]) and s1[2] == s2[2]
            and s1[3] == s2[3] and s1[4] == s2[4])


def _replay_permutation(state_before, num_subsets):
    """Recover the subset-order permutation a segment drew, and verify it was the
    segment's ONLY np.random consumption.  Leaves the stream where the segment
    left it (so later segments are unaffected)."""
    state_after = np.random.get_state()
    np.random.set_state(state_before)
    perm = np.random.permutation(num_subsets)
    verified = _states_equal(np.random.get_state(), state_after)
    np.random.set_state(state_after)
    return perm, verified


def _setup_partitions(model, max_iterations):
    """Generate partitions + sequence exactly as initialize_recon does (same args,
    same np.random consumption).  Call AFTER np.random.seed(seed)."""
    recon_shape, granularity, use_ror_mask = model.get_params(
        ['recon_shape', 'granularity', 'use_ror_mask'])
    partitions = mj.gen_set_of_pixel_partitions(
        recon_shape, granularity, output_device=model.recon_placement.devices[0],
        use_ror_mask=use_ror_mask)
    seq = np.asarray(mj.gen_partition_sequence(model.get_params('partition_sequence'),
                                               max_iterations=max_iterations))
    return partitions, seq


def run_segmented(model, sinogram, weights=None, max_iterations=15, seed=0,
                  offsets_by_entry=None, snapshot_iterations=(),
                  per_iteration_hook=None, init_recon=None):
    """Run a segmented VCD recon; return a records dict.

    Args:
        model: a constructed TomographyModel (geometry and non-sigma params set by
            the caller).
        sinogram: host or device sinogram (device arrays on the model's devices are
            left in place and NOT freed -- safe to reuse across runs).
        weights: optional weights array, or None for constant 1s.
        max_iterations: iteration count.  The driver passes
            stop_threshold_change_pct=0 so segment boundaries are the only control
            (no early stop mid-schedule).
        seed: np.random seed for this variant (partitions + subset orders).
        offsets_by_entry: per-granularity schedule (see scheduled_sigmas); None/{}
            gives a fixed-target baseline.
        snapshot_iterations: iterable of iteration indices to snapshot as host recon
            volumes, or 'all'.  Snapshots gather + crop to the real recon shape (safe
            under sharding).
        per_iteration_hook: optional callable(i, recon_device, checkpoint, seg_record)
            whose return value is appended to records['hook'].  Runs BEFORE the next
            segment consumes the checkpoint, so checkpoint['error_sinogram'] is still
            readable (e.g. for in-stream metrics at sizes too large to snapshot).
        init_recon: optional initial reconstruction for the FIRST segment.  None
            (default) keeps the production path: vcd_recon runs direct_recon (FDK)
            internally and applies the optimal error-sinogram scaling.  A supplied
            array is used AS GIVEN -- vcd_recon skips that scaling for an explicit
            init, so pre-scale it if the FDK-equivalent scaling is wanted.

    Returns:
        dict with per-iteration lists 'entry', 'num_subsets', 'sigma_x', 'sigma_y',
        'fm_rmse_raw' (scheduled-sigma_y ruler), 'fm_rmse' (rescaled to the TARGET
        sigma_y ruler; the loss is exactly proportional to 1/sigma_y), 'es_rmse'
        (sigma-free residual RMSE), 'nmae', 'alpha', 'perm', 'perm_verified',
        'wall_s'; plus 'snapshots' {i: host volume}, 'hook', 'final_recon' (host),
        'targets', 'seq', 'seed', 'partitions_host'.
    """
    # vcd_recon requires model.logger (normally created by initialize_recon, which
    # the driver bypasses).  Quiet by default: the driver records everything itself.
    model.setup_logger(logfile_path=None, print_logs=False)
    targets = compute_targets(model, sinogram, weights)
    sigma_y_target = targets[1]

    np.random.seed(seed)
    partitions, seq = _setup_partitions(model, max_iterations)

    real_sino_size = float(math.prod(model.get_params('sinogram_shape')))
    snap_all = (snapshot_iterations == 'all')
    snap_set = set() if snap_all else {int(k) for k in snapshot_iterations}

    keys = ('entry', 'num_subsets', 'sigma_x', 'sigma_y', 'fm_rmse_raw', 'fm_rmse',
            'es_rmse', 'nmae', 'alpha', 'perm', 'perm_verified', 'wall_s')
    records = {k: [] for k in keys}
    records['snapshots'] = {}
    records['hook'] = []

    recon = None
    ckpt = None
    for i in range(int(max_iterations)):
        entry = int(seq[i])
        sigma_x_i, sigma_y_i = scheduled_sigmas(entry, targets, offsets_by_entry)
        # vcd_recon reads sigma_x/sigma_y when it builds the subset updater, so this
        # scopes the scheduled values to exactly this segment.  no_warning suppresses
        # the manual-regularization warning; sigma params carry no recompile flag, so
        # no projector state is rebuilt.
        model.set_params(no_warning=True, sigma_x=sigma_x_i, sigma_y=sigma_y_i)

        state_before = np.random.get_state()
        t0 = time.time()
        if recon is None:
            # First segment: with init_recon=None vcd_recon runs direct_recon
            # internally and scales it to the sinogram, exactly as a continuous run
            # would; a supplied init_recon is used as given (no internal scaling).
            recon, losses, ckpt = model.vcd_recon(
                sinogram, partitions, seq[i:i + 1], 0.0, weights=weights,
                init_recon=init_recon, first_iteration=i, return_checkpoint=True)
        else:
            # Resume: (recon, error sinogram, Hessian) carry the whole loop state, so
            # there is no re-initialization cost.  The checkpoint is SINGLE-USE (the
            # resume donates the error-sinogram buffer), which is why the fresh ckpt
            # replaces the old reference every segment.
            recon, losses, ckpt = model.vcd_recon(
                sinogram, partitions, seq[i:i + 1], 0.0, weights=weights,
                init_recon=recon, init_error_sinogram=ckpt['error_sinogram'],
                fm_hessian=ckpt['fm_hessian'], first_iteration=i,
                return_checkpoint=True)
        wall = time.time() - t0

        num_subsets = int(partitions[entry].shape[0])
        perm, verified = _replay_permutation(state_before, num_subsets)

        # Sigma-free residual RMSE straight from the fresh checkpoint (device
        # reduction + one scalar read) -- must happen before the next segment's
        # resume consumes the checkpoint.
        es = ckpt['error_sinogram']
        es_rmse = float(jnp.sqrt(jnp.sum(es * es) / real_sino_size))

        fm_raw = float(losses[0][0])
        seg_record = dict(iteration=i, entry=entry, num_subsets=num_subsets,
                          sigma_x=sigma_x_i, sigma_y=sigma_y_i, fm_rmse_raw=fm_raw,
                          fm_rmse=fm_raw * sigma_y_i / sigma_y_target,
                          es_rmse=es_rmse, nmae=float(losses[2][0]),
                          alpha=float(losses[3][0]), perm=perm,
                          perm_verified=bool(verified), wall_s=wall)
        for k in keys:
            records[k].append(seg_record[k])

        if snap_all or i in snap_set:
            # _gather_recon assembles on the host and crops any sharding padding, so
            # snapshots always have the problem's real shape.
            records['snapshots'][i] = np.asarray(model._gather_recon(recon))
        if per_iteration_hook is not None:
            records['hook'].append(per_iteration_hook(i, recon, ckpt, seg_record))

    records['final_recon'] = np.asarray(model._gather_recon(recon))
    records['targets'] = targets
    records['seq'] = seq
    records['seed'] = seed
    # Host copies of the partitions (flat indices into the (rows, cols) grid) for
    # the footprint probe; small (num_subsets x subset_size ints per granularity).
    records['partitions_host'] = [np.asarray(p) for p in partitions]
    return records


def run_continuous(model, sinogram, weights=None, max_iterations=15, seed=0):
    """Reference continuous run under the SAME RNG and sigma discipline (targets set
    explicitly, partitions generated identically) -- the equivalence gate's baseline.
    """
    model.setup_logger(logfile_path=None, print_logs=False)  # see run_segmented
    targets = compute_targets(model, sinogram, weights)
    np.random.seed(seed)
    partitions, seq = _setup_partitions(model, max_iterations)
    model.set_params(no_warning=True, sigma_x=targets[0], sigma_y=targets[1])
    recon, losses = model.vcd_recon(sinogram, partitions, seq, 0.0, weights=weights)
    return dict(final_recon=np.asarray(model._gather_recon(recon)),
                fm_rmse=[float(v) for v in losses[0]],
                nmae=[float(v) for v in losses[2]],
                alpha=[float(v) for v in losses[3]],
                targets=targets, seq=seq, seed=seed)


def rel_max_err(a, b):
    """Scale-invariant relative max error, max|a-b| / max|b| -- the project's float
    comparison gate (never exact equality for computed floats)."""
    a = np.asarray(a)
    b = np.asarray(b)
    denom = max(float(np.max(np.abs(b))), float(np.finfo(np.float32).tiny))
    return float(np.max(np.abs(a - b)) / denom)
