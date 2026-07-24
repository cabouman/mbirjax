"""Step 2 of the A100 tuning campaign: the CONE Pallas-constant sweep, reported as a
(time, peak-memory) PARETO FRONTIER.

Why these axes and not the obvious ones
---------------------------------------
Only four module constants in mbirjax/_pallas_kernels.py can move a CONE workload:

    CONE_LC, CONE_NUM_WARPS          -- the fused-vfan back kernel
    CONE_FWD_NUM_WARPS               -- the fused cone forward kernel
    FWD_SEGMENT_CAP                  -- shared: parallel AND cone forward

Deliberately NOT swept:
  * ROW_CHUNK / NUM_WARPS / FWD_NUM_WARPS are read only inside _make_back_call and
    _make_fwd_phase, i.e. the PARALLEL-beam builders.  Sweeping them at cone shapes
    measures run-to-run drift and nothing else.
  * BACK_VIEW_CHUNK_CAP is a CLAMP, not a knob, at n=1: the driver takes
    min(tiles.back_view_batch, BACK_VIEW_CHUNK_CAP, num_views) and the base policy
    already pins back_view_batch = _BACK_VIEW_CAP_SINGLE = 128, so every cell with
    CAP >= 128 is bit-identical to shipped.  The EFFECTIVE view chunk is swept instead,
    via tiles._replace(back_view_batch=...), and the cap itself is a memory guard to
    revisit at n >= 2.

Where it is measured
--------------------
At a LARGE cone cell whose full VCD still fits one A100-40GB with real headroom (Greg
2026-07-20: "tune on a largish recon that fits in one GPU ... you don't have to go right
to the edge", axis lengths all different to avoid symmetries).  Motivation: the
projector share of a VCD recon grows steeply with size -- from the nightly GPU records,
cone n=1 is ~10% at 200x208x160, ~51% at 512x448x384 and ~62% at 1024x1008x992 -- so
tuning at a small cell optimizes something that barely matters.  The 1024 class is
excluded because its full VCD needs 47.6 GB and does not fit a 40 GB A100 at n=1.

Two operating points, because they are different kernels in practice:
  * SUBSET  -- one VCD fine-tail subset (granularity 128).  The default partition
    sequence [2,4,6,7] indexes granularity [1,2,4,8,16,32,64,128,256] and is extended by
    REPEATING ITS LAST ENTRY, so 12 of 15 iterations run at 128 subsets.  This is the
    shape the recon actually spends its time in, and it is the PRIMARY time objective.
  * FULL    -- the whole ROR grid: the memory-dominant shape (and what coeff_power=2
    runs once per recon).

Measurement rules (from the adversarial methodology review; each fixes a defect that
would otherwise produce a confident wrong answer)
--------------------------------------------------------------------------------------
1. SILENT NO-OP.  The kernel builders are @functools.cache'd on SHAPE tuples that do not
   include these constants, and the two-level builders (_make_cone_fwd_chunk_fn ->
   _make_cone_fwd_phase) cache a jax.jit closure holding already-built pallas_call
   objects.  Clearing the inner builder but not the wrapper returns the OLD binary, so a
   warps sweep would be perfectly flat.  Here EVERY cache in the module is cleared by
   introspection (not a hand-maintained list), and each cell ASSERTS that the relevant
   builder actually missed -- a cell that did not rebuild is recorded as invalid.
2. WRONG THING RUNNING.  availability() is process-cached and falls back to XLA
   silently.  Each cell records it, and a deliberate MBIRJAX_DISABLE_PALLAS=1 positive
   control must come out slower -- if it does not, the sweep is void.
3. MEMORY IS NOT ATTRIBUTABLE.  peak_bytes_in_use is a process-cumulative high-water
   mark with no reset, so the value gate runs in its OWN subprocess and each cell
   reports peak measured around the timed region only; a cell whose peak did not move is
   reported as censored (None), never as a number.
4. DRIFT AND ORDER.  Cell order is RANDOMIZED within each of PASSES independent passes,
   and a fixed ANCHOR cell (shipped constants) is re-run every ANCHOR_EVERY cells so
   thermal/clock drift is visible rather than absorbed into whatever was swept in order.
   Timing uses MIN of TRIALS (one-sided noise), then min across passes.
5. DERIVED-CONFIG CONFOUNDS.  lc is clamped (lc = min(CONE_LC, next_pow2(slices))) and
   l_padded = ceil(S/lc)*lc, so several nominal CONE_LC values can collapse to the SAME
   effective config.  Every cell logs its derived config; clamped duplicates are a free
   within-sweep null control (if two identical configs differ by more than the claimed
   effect, the ruler is broken).

Run:  sbatch a100_cone_sweep.slurm     -- all parameters are in the Config block below.
      Set ROLE=probe first (PROBE_ONLY) to size the cell before committing GPU hours.
"""
import json
import os
import random
import subprocess
import sys
import time

# ── Config ────────────────────────────────────────────────────────────────────
# Sinogram (n_views, n_rows, n_channels); all three differ (axis-swap / symmetry guard).
# Cone recon auto-derives to about (n_channels, n_channels, 1.125*n_rows).
SINO_SHAPE = (752, 720, 688)
CONE_SDD_OVER_CHANNELS = 4.0        # magnification 2 -- the nightly/test-suite convention
# 'cone' or 'parallel'.  The two geometries use DISJOINT constants (cone: CONE_LC /
# CONE_NUM_WARPS / CONE_FWD_NUM_WARPS; parallel: ROW_CHUNK / NUM_WARPS / FWD_NUM_WARPS),
# so the stage axes and the drivers under test both switch on this.
GEOMETRY = os.environ.get('A100_SWEEP_GEOMETRY', 'cone')
NUM_SUBSETS = 128                   # granularity index 7: the VCD fine tail
PROBE_ONLY = False                  # True: just size the cell (memory + per-op time), no sweep

PASSES = 3                          # independent randomized passes
TRIALS_SUBSET = 20                  # timed trials per cell at the subset point
TRIALS_FULL = 5                     # timed trials per cell at the full grid
WARMUP_SECONDS = 0.5                # warm the clocks before timing (not a fixed call count)
ANCHOR_EVERY = 6                    # re-run the shipped-constants anchor this often

# Sweep stages.  Stage 1 is 2-D on purpose: CONE_LC and the effective view chunk jointly
# set the L2 working set (~view_chunk * channels * (lc + 2*bp) * 4 B), which is the one
# mechanism predicted to differ between H100 (50 MB L2) and A100 (40 MB) -- a 1-D lc
# sweep at a pinned view chunk cannot see it.
STAGE1_CONE_LC = [32, 64, 128, 256, 512]
STAGE1_ROW_CHUNK = [64, 128, 256, 512]      # 512 EXTENDS past the H100 range {64,128,256}
STAGE2_NUM_WARPS = [1, 2, 4]
STAGE3_FWD_NUM_WARPS = [1, 2, 4]
STAGE1_VIEW_CHUNK = [32, 64, 128]
STAGE2_CONE_NUM_WARPS = [1, 2, 4]           # over the best stage-1 points
STAGE2_TOP_K = 3
STAGE3_FWD_SEGMENT_CAP = [32, 64, 128, 256]
STAGE3_CONE_FWD_NUM_WARPS = [1, 2, 4]

# Shipped values -- the anchor cell and the incumbent in the decision rule.
SHIPPED = ({'ROW_CHUNK': 256, 'NUM_WARPS': 2, 'FWD_NUM_WARPS': 1,
            'FWD_SEGMENT_CAP': 64, 'VIEW_CHUNK': 128}
           if GEOMETRY == 'parallel' else
           {'CONE_LC': 128, 'CONE_NUM_WARPS': 1, 'CONE_FWD_NUM_WARPS': 2,
            'FWD_SEGMENT_CAP': 64, 'VIEW_CHUNK': 128})
# A challenger must beat the incumbent by more than this AND more than the measured
# noise floor, in a majority of passes.  Pre-registered so the analysis cannot drift.
MIN_REL_GAIN = 0.03
MIN_PASS_WINS = 2

# Value gates (the shipped contracts from _pallas_kernels.py's own docstrings).
GATE_BACK_GRAD_REL = 1e-5
GATE_BACK_HESS_REL = 1e-4
GATE_FWD_NRMSE = 2e-5
GATE_FWD_MAXREL = 3e-4

# Overridable so the SAME harness runs on gautschi's H100s (the boundary
# re-run) without editing: the constants under test ship for both arches.
OUT_DIR = os.environ.get('A100_OUT_DIR',
                         '/scratch/gilbreth/buzzard/a100_tuning')
RESULTS = os.path.join(OUT_DIR, 'results', 'cone_sweep.json')

_ROLE = os.environ.get('A100_SWEEP_ROLE', 'orchestrate')


# ── Shared worker helpers (import jax only inside worker roles) ───────────────
def _clear_all_kernel_caches():
    """Clear EVERY functools.cache in _pallas_kernels by introspection.

    A hand-maintained list is the failure mode this exists to avoid: the cone forward
    has TWO cache levels (_make_cone_fwd_chunk_fn wraps _make_cone_fwd_phase), and
    clearing only the inner one leaves a cached jax.jit closure holding the OLD
    pallas_call -- so the patched constant is silently ignored.  Returns the names
    cleared, for the record.
    """
    from mbirjax import _pallas_kernels as pk
    cleared = []
    for name in dir(pk):
        obj = getattr(pk, name, None)
        if hasattr(obj, 'cache_clear') and hasattr(obj, 'cache_info'):
            obj.cache_clear()
            cleared.append(name)
    return cleared


def _apply_constants(consts):
    """Patch the module globals, then clear caches.  ORDER MATTERS: patch first, clear
    second, and do both before any model/driver call in this process."""
    from mbirjax import _pallas_kernels as pk
    for k, v in consts.items():
        if k == 'VIEW_CHUNK':
            continue                     # applied via the TilePolicy, not a module global
        if not hasattr(pk, k):
            raise KeyError('no such constant in _pallas_kernels: %r' % k)
        setattr(pk, k, v)
    return _clear_all_kernel_caches()


def _build_model(n_devices=1):
    import numpy as np
    import mbirjax
    n_views, n_rows, n_channels = SINO_SHAPE
    angles = np.linspace(0, np.pi, n_views, endpoint=False)
    sdd = CONE_SDD_OVER_CHANNELS * n_channels
    if GEOMETRY == 'parallel':
        model = mbirjax.ParallelBeamModel(SINO_SHAPE, angles)
    else:
        model = mbirjax.ConeBeamModel(SINO_SHAPE, angles, source_detector_dist=sdd,
                                      source_iso_dist=sdd / 2.0)
    model.configure_devices(n_devices)
    return model


PIXEL_CACHE = os.path.join(OUT_DIR, 'cache', 'cone_sweep_pixels_%dx%dx%d.npz'
                           % SINO_SHAPE)


def _pixel_sets(model):
    """(subset_indices, full_indices): the VCD fine-tail subset and the whole ROR grid.

    The subset comes from the LIBRARY partitioner, not rng.choice -- a random pixel draw
    has the wrong spatial structure, and an unrepresentative harness shape is exactly
    what produced the retracted TRANSLATION sort policy (lessons.md).

    CACHED to scratch: building the 128-subset partition over a ~370k-pixel ROR costs
    real seconds, and this runs in EVERY one of ~126 subprocess cells.  The partition is
    deterministic given the seed, so caching changes nothing measured -- it just keeps
    the sweep's wall clock dominated by the kernels rather than by setup.
    """
    import numpy as np
    if os.path.exists(PIXEL_CACHE):
        d = np.load(PIXEL_CACHE)
        return d['subset'], d['full']
    import mbirjax
    from mbirjax.vcd_utils import gen_set_of_pixel_partitions
    recon_shape = model.get_params('recon_shape')
    full = np.asarray(mbirjax.gen_full_indices(recon_shape, use_ror_mask=True))
    np.random.seed(0)                    # the partitioner draws from the global RNG
    partitions = gen_set_of_pixel_partitions(recon_shape, [NUM_SUBSETS])
    subset = np.asarray(partitions[0][0])   # subset 0 of the 128-subset partition
    os.makedirs(os.path.dirname(PIXEL_CACHE), exist_ok=True)
    # The temp name MUST end in .npz: np.savez silently APPENDS '.npz' to any path that
    # does not, so a '<name>.tmp<pid>' temp is written as '<name>.tmp<pid>.npz' and the
    # rename below then fails on a missing file.
    tmp = PIXEL_CACHE + '.tmp%d.npz' % os.getpid()
    np.savez(tmp, subset=subset, full=full)
    os.replace(tmp, PIXEL_CACHE)         # atomic: concurrent cells cannot see a partial
    return subset, full


def _mem():
    import jax
    st = jax.devices()[0].memory_stats() or {}
    return int(st.get('peak_bytes_in_use', 0)), int(st.get('bytes_in_use', 0))


def _time_min(fn, trials):
    """min of `trials` timed calls after a wall-clock warmup.  MIN, not median: GPU
    timing noise is one-sided (contention, clock ramp, DVFS), so the minimum is the
    least-contaminated estimator of the achievable time."""
    import jax
    t_end = time.perf_counter() + WARMUP_SECONDS
    while time.perf_counter() < t_end:
        jax.block_until_ready(fn())
    best = float('inf')
    for _ in range(trials):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        best = min(best, time.perf_counter() - t0)
    return best


# ── Role: probe -- size the cell before spending GPU hours on it ─────────────
def probe():
    import jax
    import numpy as np
    model = _build_model()
    recon_shape = model.get_params('recon_shape')
    subset, full = _pixel_sets(model)
    num_slices = recon_shape[2]
    rng = np.random.default_rng(0)
    sino = jax.device_put(rng.random(SINO_SHAPE, dtype=np.float32))
    out = {
        'sino_shape': list(SINO_SHAPE), 'recon_shape': list(recon_shape),
        'num_pixels_full': int(len(full)), 'num_pixels_subset': int(len(subset)),
        'num_slices': int(num_slices),
        'sino_GB': float(np.prod(SINO_SHAPE) * 4 / 1e9),
        'recon_GB': float(np.prod(recon_shape) * 4 / 1e9),
    }
    from mbirjax import _pallas_kernels as pk
    idx = jax.device_put(full, model.sino_placement.devices[0])
    p0, _ = _mem()
    b = (pk.back_project_single_device(model, sino, idx) if GEOMETRY == 'parallel'
         else pk.cone_back_project_band(model, sino, idx, 0, num_slices))
    jax.block_until_ready(b)
    p1, _ = _mem()
    out['back_full_peak_GB'] = p1 / 1e9
    out['back_full_delta_GB'] = (p1 - p0) / 1e9
    print('PROBE ' + json.dumps(out, indent=2))
    print('\nDoes the isolated back call fit one A100-40GB?  peak = %.1f GB' % (p1 / 1e9))
    print('Full-VCD peak is NOT measured here (it needs a real recon); extrapolate from')
    print('the nightly records or run a100_baseline_ab.py at this shape before trusting it.')


# ── Role: cell -- ONE sweep cell, isolated ───────────────────────────────────
def run_cell():
    import jax
    import numpy as np
    from mbirjax import _pallas_kernels as pk

    spec = json.loads(os.environ['A100_SWEEP_CELL'])
    consts = spec['consts']
    op = spec['op']                       # 'back' | 'fwd'
    point = spec['point']                 # 'subset' | 'full'

    cleared = _apply_constants(consts)
    model = _build_model()
    vc = consts.get('VIEW_CHUNK')
    if vc is not None:
        model.tiles = model.tiles._replace(back_view_batch=vc, fwd_view_batch=vc)
    recon_shape = model.get_params('recon_shape')
    num_slices = recon_shape[2]
    subset, full = _pixel_sets(model)
    idx_host = subset if point == 'subset' else full
    dev = model.sino_placement.devices[0]
    idx = jax.device_put(np.asarray(idx_host), dev)
    num_pixels = int(idx.shape[0])
    rng = np.random.default_rng(0)

    avail, why = pk.availability()
    use_xla = bool(spec.get('use_xla'))
    if use_xla:
        # The XLA reference path, for the positive control.  MBIRJAX_DISABLE_PALLAS is
        # NOT a usable control here: it only flips the TilePolicy dispatch flags, while
        # these cells call the pallas driver FUNCTION directly (deliberately, to time
        # the kernel in isolation) -- so the env var leaves the timing untouched and a
        # control built on it always "fails".  Clearing the flags and going through
        # model.sparse_* is the real comparison.
        model.tiles = model.tiles._replace(back_pallas=False, fwd_pallas=False,
                                           back_pallas_band=False,
                                           fwd_pallas_band=False)
    # Inputs PRE-PLACED and pre-committed: the drivers' own _shard_sinogram/device_put
    # would otherwise sit inside the timed region as a constant additive term,
    # compressing every ratio toward 1.0 (lessons.md records this turning a 2.25x gap
    # into a bogus 1.45x).
    if op == 'back':
        sino = jax.device_put(rng.random(SINO_SHAPE, dtype=np.float32), dev)
        jax.block_until_ready(sino)
        if GEOMETRY == 'parallel':
            # The parallel back driver has no band entry point, so its own
            # _shard_sinogram/device_put sit inside the timed region.  At n=1 with an
            # already-placed array those are near-free, but the absolute times are
            # driver-inclusive and must not be quoted as pure kernel times.
            builder = pk._make_back_call
            fn = ((lambda: model.sparse_back_project(sino, idx)) if use_xla else
                  (lambda: pk.back_project_single_device(model, sino, idx)))
        else:
            builder = pk._make_cone_back_call
            fn = ((lambda: model.sparse_back_project(sino, idx)) if use_xla else
                  (lambda: pk.cone_back_project_band(model, sino, idx, 0, num_slices)))
    else:
        vals = jax.device_put(
            rng.random((num_pixels, num_slices), dtype=np.float32), dev)
        jax.block_until_ready(vals)
        if GEOMETRY == 'parallel':
            builder = pk._make_fwd_phase
            fn = lambda: pk.forward_project_subset(model, vals, idx)
        else:
            builder = pk._make_cone_fwd_phase
            fn = lambda: pk.cone_forward_project(model, vals, idx)

    misses_before = builder.cache_info().misses
    peak_before, _ = _mem()
    jax.block_until_ready(fn())           # first call builds + compiles
    # The XLA reference never touches a pallas builder, so its miss count legitimately
    # stays zero -- only pallas cells must prove they rebuilt.
    misses_after = builder.cache_info().misses + (1 if use_xla else 0)

    trials = TRIALS_SUBSET if point == 'subset' else TRIALS_FULL
    t = _time_min(fn, trials)
    peak_after, _ = _mem()

    # Derived config -- so clamped duplicates are visible rather than mistaken for a
    # measured difference (e.g. CONE_LC 512 and 1024 both clamp to next_pow2(slices)).
    lc = (min(consts.get('ROW_CHUNK', pk.ROW_CHUNK), pk.next_pow2(num_slices))
          if GEOMETRY == 'parallel' else
          min(consts.get('CONE_LC', pk.CONE_LC), pk.next_pow2(num_slices)))
    l_padded = -(-num_slices // lc) * lc
    eff_vc = min(model.tiles.back_view_batch, pk.BACK_VIEW_CHUNK_CAP, SINO_SHAPE[0])

    print('CELL_RESULT ' + json.dumps({
        'consts': consts, 'op': op, 'point': point,
        'time_s': t, 'num_pixels': num_pixels,
        'peak_GB': peak_after / 1e9,
        # Censored, never a fabricated number, when the timed region did not move the
        # process high-water mark (peak_bytes_in_use cannot be reset).
        'peak_delta_GB': ((peak_after - peak_before) / 1e9
                          if peak_after > peak_before else None),
        'rebuilt': bool(misses_after > misses_before),
        'caches_cleared': len(cleared),
        'pallas_available': bool(avail), 'pallas_why': why,
        'derived': {'lc': int(lc), 'l_padded': int(l_padded),
                    'pad_frac': float(l_padded / num_slices),
                    'view_chunk': int(eff_vc),
                    'n_view_chunks': int(-(-SINO_SHAPE[0] // eff_vc)),
                    'num_slices': int(num_slices)},
        'pass': spec.get('pass'), 'anchor': spec.get('anchor', False),
    }))


# ── Role: gate -- value-check one constant setting, in its own process ───────
def run_gate():
    import jax
    import jax.numpy as jnp
    import numpy as np
    from mbirjax import _pallas_kernels as pk

    consts = json.loads(os.environ['A100_SWEEP_CELL'])['consts']
    _apply_constants(consts)
    model = _build_model()
    vc = consts.get('VIEW_CHUNK')
    if vc is not None:
        model.tiles = model.tiles._replace(back_view_batch=vc, fwd_view_batch=vc)
    recon_shape = model.get_params('recon_shape')
    num_slices = recon_shape[2]
    subset, _full = _pixel_sets(model)
    dev = model.sino_placement.devices[0]
    idx = jax.device_put(np.asarray(subset), dev)
    # UNIFORM POSITIVE inputs (rng.random), NOT standard_normal.  This is not cosmetic:
    # the shipped tolerances below were calibrated against positive data (every
    # calibrating experiment uses rng.random -- tests/test_pallas_kernels.py, the E5
    # cone-back spike, the E6 cone-forward spike).  Signed zero-mean data makes the view
    # sum cancel toward zero while individual terms stay O(1), inflating every
    # relative-to-max metric; measured here, standard_normal reported back_grad_rel
    # 4.9e-5 and fwd_nrmse 4.6e-5 against tolerances of 1e-5 and 2e-5 -- a failure of
    # the RULER, not the kernel.  Real sinograms are non-negative anyway.
    rng = np.random.default_rng(1)
    sino = jax.device_put(rng.random(SINO_SHAPE, dtype=np.float32), dev)
    vals = jax.device_put(
        rng.random((int(idx.shape[0]), num_slices), dtype=np.float32), dev)

    def rel_max(a, b):
        return float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b)))

    def nrmse(a, b):
        return float(jnp.sqrt(jnp.mean((a - b) ** 2)) / jnp.sqrt(jnp.mean(b ** 2)))

    # The XLA reference MUST have the pallas flags cleared, or sparse_* routes straight
    # back into the kernels under test and the gate compares a thing with itself.
    model.tiles = model.tiles._replace(back_pallas=False, fwd_pallas=False,
                                       back_pallas_band=False, fwd_pallas_band=False)
    out = {'consts': consts}
    for cp, key in ((1, 'back_grad_rel'), (2, 'back_hess_rel')):
        ref = jnp.asarray(model.sparse_back_project(sino, idx, coeff_power=cp))
        got = (pk.back_project_single_device(model, sino, idx, coeff_power=cp)
               if GEOMETRY == 'parallel' else
               pk.cone_back_project_band(model, sino, idx, 0, num_slices,
                                         coeff_power=cp))
        out[key] = rel_max(got, ref)
    ref_f = jnp.asarray(model.sparse_forward_project(vals, idx))
    got_f = (pk.forward_project_subset(model, vals, idx) if GEOMETRY == 'parallel'
             else pk.cone_forward_project(model, vals, idx))
    out['fwd_nrmse'] = nrmse(got_f, ref_f)
    out['fwd_maxrel'] = rel_max(got_f, ref_f)
    out['pass'] = bool(out['back_grad_rel'] <= GATE_BACK_GRAD_REL
                       and out['back_hess_rel'] <= GATE_BACK_HESS_REL
                       and out['fwd_nrmse'] <= GATE_FWD_NRMSE
                       and out['fwd_maxrel'] <= GATE_FWD_MAXREL)
    print('GATE_RESULT ' + json.dumps(out))


# ── Role: orchestrate (JAX-FREE: must hold no device memory) ─────────────────
def _spawn(role, spec):
    env = dict(os.environ, A100_SWEEP_ROLE=role, A100_SWEEP_CELL=json.dumps(spec))
    if spec.get('disable_pallas'):
        env['MBIRJAX_DISABLE_PALLAS'] = '1'
    else:
        env.pop('MBIRJAX_DISABLE_PALLAS', None)
    r = subprocess.run([sys.executable, '-u', __file__], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tag = 'CELL_RESULT ' if role == 'cell' else 'GATE_RESULT '
    for ln in r.stdout.splitlines():
        if ln.startswith(tag):
            return json.loads(ln[len(tag):])
    print('   cell failed (rc=%d); tail:' % r.returncode)
    for ln in r.stdout.splitlines()[-12:]:
        print('     | ' + ln)
    return None


def _cells_stage1():
    if GEOMETRY == 'parallel':
        return [{'consts': dict(SHIPPED, ROW_CHUNK=rc, VIEW_CHUNK=vc), 'op': 'back'}
                for rc in STAGE1_ROW_CHUNK for vc in STAGE1_VIEW_CHUNK]
    return [{'consts': dict(SHIPPED, CONE_LC=lc, VIEW_CHUNK=vc), 'op': 'back'}
            for lc in STAGE1_CONE_LC for vc in STAGE1_VIEW_CHUNK]


def _cells_stage2(top):
    key = 'NUM_WARPS' if GEOMETRY == 'parallel' else 'CONE_NUM_WARPS'
    vals = STAGE2_NUM_WARPS if GEOMETRY == 'parallel' else STAGE2_CONE_NUM_WARPS
    return [{'consts': dict(c, **{key: w}), 'op': 'back'} for c in top for w in vals]


def _cells_stage3():
    key = 'FWD_NUM_WARPS' if GEOMETRY == 'parallel' else 'CONE_FWD_NUM_WARPS'
    vals = STAGE3_FWD_NUM_WARPS if GEOMETRY == 'parallel' else STAGE3_CONE_FWD_NUM_WARPS
    return [{'consts': dict(SHIPPED, FWD_SEGMENT_CAP=c, **{key: w}), 'op': 'fwd'}
            for c in STAGE3_FWD_SEGMENT_CAP for w in vals]


def _run_stage(name, cells, point):
    """Run `cells` at `point` over PASSES randomized passes with anchor cells."""
    print('\n===== stage %s (%d cells x %d passes, point=%s) ====='
          % (name, len(cells), PASSES, point), flush=True)
    results = []
    for p in range(PASSES):
        order = list(range(len(cells)))
        random.Random(1000 + p).shuffle(order)     # order randomized, seed recorded
        for i, ci in enumerate(order):
            if i % ANCHOR_EVERY == 0:
                a = _spawn('cell', {'consts': dict(SHIPPED), 'op': cells[ci]['op'],
                                    'point': point, 'pass': p, 'anchor': True})
                if a:
                    results.append(a)
                    print('   [anchor] %.4f s' % a['time_s'], flush=True)
            spec = dict(cells[ci], point=point, **{'pass': p})
            r = _spawn('cell', spec)
            if r is None:
                continue
            results.append(r)
            flag = '' if r['rebuilt'] else '  !! NOT REBUILT (constant ignored)'
            print('   %-52s %.4f s  peak %.2f GB%s'
                  % (json.dumps({k: v for k, v in r['consts'].items()
                                 if SHIPPED.get(k) != v} or {'shipped': True}),
                     r['time_s'], r['peak_GB'], flag), flush=True)
    return results


def _pareto(points):
    """Non-dominated set over (time, peak) -- both minimized."""
    out = []
    for a in points:
        if not any((b['t'] <= a['t'] and b['m'] <= a['m']) and
                   (b['t'] < a['t'] or b['m'] < a['m']) for b in points):
            out.append(a)
    return sorted(out, key=lambda x: x['t'])


def _summarize(name, results, op=None):
    """Summarize ONE op's cells.  The op split is load-bearing: a stage may mix back and
    forward cells (stage 4 does), and back/forward differ by ~3x in absolute time here --
    pooling them made the anchor spread read 194.8% when each op's own spread is 0.0%,
    and merged unrelated back/forward rows that happened to share a constants dict.
    Always summarize per op."""
    if op is not None:
        results = [r for r in results if r.get('op') == op]
        name = '%s [op=%s]' % (name, op)
    ops = {r.get('op') for r in results}
    if len(ops) > 1:
        for o in sorted(x for x in ops if x):
            _summarize(name, results, op=o)
        return []
    anchors = [r['time_s'] for r in results if r.get('anchor')]
    noise = (max(anchors) - min(anchors)) / min(anchors) if len(anchors) > 1 else None
    print('\n--- %s ---' % name)
    if noise is not None:
        print('anchor (shipped constants) re-runs: n=%d  min %.4f s  max %.4f s'
              '  spread %.1f%%  <- the NOISE FLOOR; smaller effects are not real'
              % (len(anchors), min(anchors), max(anchors), 100 * noise))
    by = {}
    for r in results:
        if r.get('anchor'):
            continue
        # Key includes the op: identical constants under different ops are DIFFERENT
        # measurements and must never be merged by the min() below.
        key = json.dumps({'op': r.get('op'), 'c': r['consts']}, sort_keys=True)
        by.setdefault(key, []).append(r)
    rows = []
    for key, rs in by.items():
        t = min(x['time_s'] for x in rs)
        m = min(x['peak_GB'] for x in rs)
        rows.append({'key': json.dumps(json.loads(key)['c'], sort_keys=True),
                     'op': json.loads(key)['op'], 't': t, 'm': m, 'n': len(rs),
                     'rebuilt': all(x['rebuilt'] for x in rs),
                     'derived': rs[0]['derived']})
    rows.sort(key=lambda x: x['t'])
    print('%-56s %9s %9s %5s' % ('constants (vs shipped)', 'time_s', 'peak_GB', 'ok'))
    for r in rows:
        c = json.loads(r['key'])
        diff = {k: v for k, v in c.items() if SHIPPED.get(k) != v} or {'shipped': True}
        print('%-56s %9.4f %9.2f %5s'
              % (json.dumps(diff), r['t'], r['m'], 'ok' if r['rebuilt'] else 'STALE'))
    front = _pareto(rows)
    print('\nPARETO FRONTIER (non-dominated in time AND peak memory):')
    for r in front:
        c = json.loads(r['key'])
        diff = {k: v for k, v in c.items() if SHIPPED.get(k) != v} or {'shipped': True}
        print('   %-52s %9.4f s  %6.2f GB   [lc=%s l_pad=%s pad=%.2fx vc=%s]'
              % (json.dumps(diff), r['t'], r['m'], r['derived']['lc'],
                 r['derived']['l_padded'], r['derived']['pad_frac'],
                 r['derived']['view_chunk']))
    ship = [r for r in rows if json.loads(r['key']) == SHIPPED]
    if ship and noise:
        best = rows[0]
        gain = (ship[0]['t'] - best['t']) / ship[0]['t']
        print('\nbest vs shipped: %+.1f%% time (threshold: > max(%.0f%%, noise %.1f%%))'
              % (100 * gain, 100 * MIN_REL_GAIN, 100 * noise))
        if gain <= max(MIN_REL_GAIN, noise):
            print('   -> NOT a real improvement by the pre-registered rule; '
                  'the incumbent stands.')
    return rows


def orchestrate():
    os.makedirs(os.path.join(OUT_DIR, 'results'), exist_ok=True)
    all_results = {}

    # Positive control FIRST: the pallas driver must differ measurably from the XLA path.
    # NOTE the control is pallas-driver vs XLA-PATH, not MBIRJAX_DISABLE_PALLAS: that env
    # var only flips the TilePolicy DISPATCH flags, and these cells call the pallas driver
    # function directly (by design, to time the kernel in isolation), so the env var
    # leaves the timing identical and an env-var-based control aborts on every run.
    # Measured 2026-07-20: ON 0.0252 s vs "OFF" 0.0253 s with availability correctly
    # False -- a false alarm from a misconceived control, not a dead kernel.
    print('===== positive control: pallas driver vs XLA path =====', flush=True)
    on = _spawn('cell', {'consts': dict(SHIPPED), 'op': 'back', 'point': 'subset'})
    ref = _spawn('cell', {'consts': dict(SHIPPED), 'op': 'back', 'point': 'subset',
                          'use_xla': True})
    if on and ref:
        print('   pallas %.4f s   xla %.4f s   ratio %.2fx'
              % (on['time_s'], ref['time_s'], ref['time_s'] / max(on['time_s'], 1e-9)))
        if abs(on['time_s'] - ref['time_s']) / max(ref['time_s'], 1e-9) < 0.05:
            print('   !! FATAL: the pallas driver and the XLA path time the SAME -- the')
            print('      kernels under test are not the code being timed.  Aborting.')
            return 2
        print('   control OK: the pallas kernels are live and distinguishable.')

    # Gate the shipped configuration once, up front.
    g = _spawn('gate', {'consts': dict(SHIPPED)})
    if g:
        print('\nvalue gate @ shipped: back_grad %.2e  back_hess %.2e  fwd_nrmse %.2e'
              '  fwd_maxrel %.2e  -> %s'
              % (g['back_grad_rel'], g['back_hess_rel'], g['fwd_nrmse'],
                 g['fwd_maxrel'], 'PASS' if g['pass'] else 'FAIL'))

    s1 = _run_stage('1: %s x view_chunk (back)'
                    % ('ROW_CHUNK' if GEOMETRY == 'parallel' else 'CONE_LC'),
                    _cells_stage1(), 'subset')
    all_results['stage1'] = s1
    rows1 = _summarize('stage 1 (back, subset)', s1)

    # Stage 2: warps over the best stage-1 points only.
    top = [json.loads(r['key']) for r in rows1[:STAGE2_TOP_K]]
    cells2 = _cells_stage2(top)
    s2 = _run_stage('2: CONE_NUM_WARPS over top-%d (back)' % STAGE2_TOP_K,
                    cells2, 'subset')
    all_results['stage2'] = s2
    _summarize('stage 2 (back, subset)', s2)

    s3 = _run_stage('3: FWD_SEGMENT_CAP x CONE_FWD_NUM_WARPS (fwd)',
                    _cells_stage3(), 'subset')
    all_results['stage3'] = s3
    rows3 = _summarize('stage 3 (fwd, subset)', s3)

    # The subset point ranks TIME; memory is dominated by the FULL grid, so the
    # survivors are re-measured there before anything is called Pareto-optimal.  Both
    # ops, since a config must not win on one and regress the other.
    def _surv(rows, op, k=STAGE2_TOP_K):
        seen, out = set(), []
        for r in _pareto(rows)[:k] + rows[:k]:
            key = r['key']
            if key not in seen:
                seen.add(key)
                out.append({'consts': json.loads(key), 'op': op})
        return out

    cells4 = _surv(rows1, 'back') + _surv(rows3, 'fwd')
    # Always carry the shipped config into the full-grid stage as the comparison point.
    cells4 += [{'consts': dict(SHIPPED), 'op': 'back'},
               {'consts': dict(SHIPPED), 'op': 'fwd'}]
    print('\n===== full-grid confirmation (the memory-dominant point) =====')
    s4 = _run_stage('4: full-grid re-measure of the survivors', cells4, 'full')
    all_results['stage4'] = s4
    _summarize('stage 4 (full grid, time AND memory)', s4)

    with open(RESULTS, 'w') as f:
        json.dump(all_results, f, indent=2)
    print('\nwrote %s' % RESULTS)
    return 0


if __name__ == '__main__':
    if _ROLE == 'probe' or (PROBE_ONLY and _ROLE == 'orchestrate'):
        probe()
    elif _ROLE == 'cell':
        run_cell()
    elif _ROLE == 'gate':
        run_gate()
    else:
        sys.exit(orchestrate())
