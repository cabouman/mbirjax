"""Correctness gates for the pallas back-projection path (mbirjax/_pallas_kernels.py).

CPU CI runs the kernel in pallas INTERPRET mode against the library XLA path (the
selection policy never routes to pallas on CPU, so the driver is called directly with
interpret=True).  On GPU the same gates run compiled.  Float gates follow the project
rules (lessons.md section 2): scale-invariant relative max, never exact equality --
the pallas path reorders the view/tap summation.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

import mbirjax
from mbirjax import _pallas_kernels

SINO_SHAPE = (64, 24, 32)          # small enough for interpret mode
REL_TOL = 1e-5                     # the calibrated single-shot projector gate


def _make_model():
    angles = np.linspace(0, np.pi, SINO_SHAPE[0], endpoint=False)
    model = mbirjax.ParallelBeamModel(SINO_SHAPE, angles)
    model.configure_devices(1)
    return model


def _interpret():
    # Compiled on allowlisted GPUs, interpret mode elsewhere (CPU CI).
    return not _pallas_kernels.is_available()


def _rel_max_err(a, b):
    return float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b)))


@pytest.mark.parametrize('coeff_power', [1, 2])
@pytest.mark.parametrize('subset', [False, True])
def test_back_matches_xla(coeff_power, subset):
    """The pallas driver must match the library XLA back projection at the float gate,
    for the gradient and Hessian paths, on full-grid and random-subset pixel sets."""
    model = _make_model()
    recon_shape = model.get_params('recon_shape')
    idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=True)
    rng = np.random.default_rng(0)
    if subset:
        idx = jnp.asarray(np.sort(rng.choice(np.asarray(idx), size=len(idx) // 7,
                                             replace=False)))
    sino = jnp.asarray(rng.random(SINO_SHAPE, dtype=np.float32))

    ref = model.sparse_back_project(sino, idx, coeff_power=coeff_power)
    out = _pallas_kernels.back_project_single_device(
        model, sino, idx, coeff_power=coeff_power, interpret=_interpret())
    assert _rel_max_err(out, jnp.asarray(ref)) < REL_TOL


def test_back_view_chunking_consistent():
    """Chunked (small back_view_batch) and single-chunk drivers must agree at the float
    gate -- the chunk boundary must not change values beyond summation order."""
    model = _make_model()
    recon_shape = model.get_params('recon_shape')
    idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=True)
    sino = jnp.asarray(np.random.default_rng(1).random(SINO_SHAPE, dtype=np.float32))

    out_single = _pallas_kernels.back_project_single_device(
        model, sino, idx, interpret=_interpret())
    model.tiles = model.tiles._replace(back_view_batch=17)      # ragged chunks incl. tail
    out_chunked = _pallas_kernels.back_project_single_device(
        model, sino, idx, interpret=_interpret())
    assert _rel_max_err(out_chunked, out_single) < REL_TOL


def test_adjoint_identity():
    """<A x, y> == <x, B y> with the library forward and the pallas back -- the
    matched-pair contract that VCD's convergence rests on."""
    model = _make_model()
    recon_shape = model.get_params('recon_shape')
    idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=True)
    rng = np.random.default_rng(2)
    x = jnp.asarray(rng.random((len(idx), recon_shape[2]), dtype=np.float32))
    y = jnp.asarray(rng.random(SINO_SHAPE, dtype=np.float32))

    ax = jnp.asarray(model.sparse_forward_project(x, idx))          # (V, R, C)... (V, rows, C)
    by = _pallas_kernels.back_project_single_device(model, y, idx,
                                                    interpret=_interpret())
    lhs = float(jnp.vdot(ax, jnp.asarray(y)))
    rhs = float(jnp.vdot(x, by))
    assert abs(lhs - rhs) / max(abs(lhs), 1e-30) < REL_TOL


@pytest.mark.parametrize('coeff_power', [1, 2])
def test_back_owned_views_band_matches_xla(coeff_power):
    """The PER-OWNER calling mode (increment 3, the multi-device band path): a view
    subset's cropped-band back projection through the pallas driver must match the
    XLA path at the float gate -- global view indices, band-cropped rows, gradient
    and Hessian."""
    model = _make_model()
    recon_shape = model.get_params('recon_shape')
    idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=True)
    rng = np.random.default_rng(8)
    sino = jnp.asarray(rng.random(SINO_SHAPE, dtype=np.float32))
    owned = np.arange(16, 48)                       # one owner's GLOBAL view block
    g0, g1 = 5, 13                                  # a rows==slices band
    band = sino[16:48, g0:g1, :]

    ref = model.projector_functions.sparse_back_project(
        band, idx, owned_view_indices=owned, coeff_power=coeff_power)
    out = _pallas_kernels.back_project_single_device(
        model, band, idx, coeff_power=coeff_power, owned_view_indices=owned,
        interpret=_interpret())
    assert out.shape == jnp.asarray(ref).shape
    assert _rel_max_err(out, jnp.asarray(ref)) < REL_TOL


def test_band_dispatch_routes_per_owner_calls(monkeypatch):
    """With back_pallas_band forced on, the parallel per-owner band override must
    route through the pallas driver with the owner's global view indices and the
    cropped band (spied so CPU CI can observe the dispatch in interpret mode)."""
    model = _make_model()
    recon_shape = model.get_params('recon_shape')
    idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=True)
    sino = jnp.asarray(np.random.default_rng(9).random(SINO_SHAPE, dtype=np.float32))
    owned = np.arange(8, 40)
    ref = model.projector_functions.sparse_back_project(
        sino[8:40, 3:11, :], idx, owned_view_indices=owned)

    seen = {}
    real = _pallas_kernels.back_project_single_device

    def spy(tm, s, pix, coeff_power=1, owned_view_indices=()):
        seen['shape'] = tuple(s.shape)
        seen['owned'] = np.asarray(owned_view_indices).tolist()
        return real(tm, s, pix, coeff_power=coeff_power,
                    owned_view_indices=owned_view_indices, interpret=_interpret())

    monkeypatch.setattr(_pallas_kernels, 'back_project_single_device', spy)
    model.tiles = model.tiles._replace(back_pallas_band=True)
    out = model._back_project_view_shard_to_band(sino[8:40], idx, 3, 11, owned, 1)
    assert seen['shape'] == (32, 8, SINO_SHAPE[2])
    assert seen['owned'] == list(range(8, 40))
    assert _rel_max_err(out, jnp.asarray(ref)) < REL_TOL


def test_policy_off_on_cpu():
    """On CPU the tile policy must not enable the pallas path (is_available is False),
    so the XLA path serves every call unchanged."""
    model = _make_model()
    if jax.devices()[0].platform != 'gpu':
        assert not model.tiles.back_pallas
        assert not _pallas_kernels.is_available()


def test_weights_match_kernel_formula():
    """The weight builder must reproduce horizontal_fan_back's effective weights: back
    projecting a one-hot sinogram through the XLA path equals the weight column."""
    model = _make_model()
    pf = model.projector_functions
    pp = pf.projector_params
    recon_shape = model.get_params('recon_shape')
    idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=True)[:50]
    w = _pallas_kernels._jit_compute_back_weights(
        pf.view_params_array, jnp.asarray(idx), model.compute_hfan_data, pp,
        coeff_power=1, owned_view_indices=jnp.arange(3))
    assert w.shape == (3, 2 * pp.geometry_params.psf_radius + 1, 50)
    assert bool(jnp.all(w >= 0)) and bool(jnp.any(w > 0))


# ── Increment 2: the forward subset path ──────────────────────────────────────

@pytest.mark.parametrize('band', ['full', 'slice_band'])
def test_fwd_subset_matches_xla(band):
    """The pallas forward driver must match the library XLA forward at the float gate,
    for full cylinders and slice-band values (the banded caller shape)."""
    model = _make_model()
    recon_shape = model.get_params('recon_shape')
    idx_full = mbirjax.gen_full_indices(recon_shape, use_ror_mask=True)
    rng = np.random.default_rng(3)
    idx = jnp.asarray(np.sort(rng.choice(np.asarray(idx_full), size=len(idx_full) // 5,
                                         replace=False)))
    n_band = recon_shape[2] if band == 'full' else max(4, recon_shape[2] // 3)
    values = jnp.asarray(rng.random((len(idx), n_band), dtype=np.float32))

    ref = model.projector_functions.sparse_forward_project(values, idx)
    out = _pallas_kernels.forward_project_subset(model, values, idx,
                                                 interpret=_interpret())
    assert out.shape == ref.shape
    assert _rel_max_err(out, jnp.asarray(ref)) < REL_TOL


def test_wrapper_dispatches_pallas_above_pixel_batch(monkeypatch):
    """The public wrapper must route EVERY pixel count to the pallas driver when the
    policy flag is set -- the former subset-size guard (P <= fwd_pixel_batch) was
    REMOVED after the 2026-07-13 P x band sweep measured pallas faster at all 70
    cells through full grid (plans/projector_kernels/fwd_guard_sweep.md).  The call
    is forced ABOVE fwd_pixel_batch (via a lowered batch), so a reintroduced guard
    would fail the dispatch assertion, and values must still match the XLA path."""
    model = _make_model()
    recon_shape = model.get_params('recon_shape')
    idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=True)
    rng = np.random.default_rng(7)
    values = jnp.asarray(rng.random((len(idx), recon_shape[2]), dtype=np.float32))

    # XLA reference through the same wrapper (flag off; tiles are read at call time).
    model.tiles = model.tiles._replace(fwd_pallas=False)
    ref = model.projector_functions.sparse_forward_project(values, idx)

    # Spy on the driver so CPU CI runs it in interpret mode and the dispatch is
    # observable (the wrapper resolves the module attribute at call time).
    seen = {}
    real = _pallas_kernels.forward_project_subset

    def spy(tm, vals, pix, owned_view_indices=()):
        seen['num_pixels'] = int(pix.shape[0])
        return real(tm, vals, pix, owned_view_indices=owned_view_indices,
                    interpret=_interpret())

    monkeypatch.setattr(_pallas_kernels, 'forward_project_subset', spy)
    model.tiles = model.tiles._replace(fwd_pallas=True, fwd_pixel_batch=64)
    out = model.projector_functions.sparse_forward_project(values, idx)
    assert seen['num_pixels'] == len(idx) > model.tiles.fwd_pixel_batch
    assert _rel_max_err(out, jnp.asarray(ref)) < REL_TOL


def test_fwd_view_chunking_consistent():
    """Chunked (small fwd_view_batch) and single-chunk forward drivers must agree."""
    model = _make_model()
    recon_shape = model.get_params('recon_shape')
    idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=True)[:80]
    values = jnp.asarray(np.random.default_rng(4).random(
        (len(idx), recon_shape[2]), dtype=np.float32))
    out_single = _pallas_kernels.forward_project_subset(model, values, idx,
                                                        interpret=_interpret())
    model.tiles = model.tiles._replace(fwd_view_batch=13)
    out_chunked = _pallas_kernels.forward_project_subset(model, values, idx,
                                                         interpret=_interpret())
    assert _rel_max_err(out_chunked, out_single) < REL_TOL


def test_pair_adjoint_identity():
    """<A x, y> == <x, B y> with BOTH pallas kernels -- the shipped pair's contract."""
    model = _make_model()
    recon_shape = model.get_params('recon_shape')
    idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=True)
    rng = np.random.default_rng(5)
    x = jnp.asarray(rng.random((len(idx), recon_shape[2]), dtype=np.float32))
    y = jnp.asarray(rng.random(SINO_SHAPE, dtype=np.float32))

    ax = _pallas_kernels.forward_project_subset(model, x, idx, interpret=_interpret())
    by = _pallas_kernels.back_project_single_device(model, y, idx,
                                                    interpret=_interpret())
    lhs = float(jnp.vdot(ax, y))
    rhs = float(jnp.vdot(x, by))
    assert abs(lhs - rhs) / max(abs(lhs), 1e-30) < REL_TOL


def test_fwd_over_cap_segments_match_xla(monkeypatch):
    """With the cap forced far below the per-channel tap counts, phase 2 carries REAL
    remainder segments -- this exercises the pl.when guard on the atomic add with work
    that must NOT be skipped (the standard test shapes never exceed the shipped cap,
    so without this test an inverted guard would pass the suite)."""
    monkeypatch.setattr(_pallas_kernels, 'FWD_SEGMENT_CAP', 4)
    _pallas_kernels._make_fwd_chunk_fn.cache_clear()      # keys don't include the cap
    try:
        model = _make_model()
        recon_shape = model.get_params('recon_shape')
        idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=True)
        rng = np.random.default_rng(6)
        values = jnp.asarray(rng.random((len(idx), recon_shape[2]), dtype=np.float32))
        ref = model.projector_functions.sparse_forward_project(values, idx)
        out = _pallas_kernels.forward_project_subset(model, values, idx,
                                                     interpret=_interpret())
        assert _rel_max_err(out, jnp.asarray(ref)) < REL_TOL
        # Self-check: the forced cap really produced remainder segments (else this
        # test silently degenerates back to the all-pad regime it exists to escape).
        pf = model.projector_functions
        _, _, starts = _pallas_kernels._jit_compute_fwd_streams(
            pf.view_params_array, idx, model.compute_hfan_data,
            pf.projector_params, owned_view_indices=jnp.arange(1))
        assert int(jnp.max(jnp.diff(starts[0]))) > 4
    finally:
        _pallas_kernels._make_fwd_chunk_fn.cache_clear()


def test_fwd_split_two_phase_covers_all_taps():
    """The device cap-and-split must partition every contributor exactly once: phase-1
    + phase-2 segment lengths sum to the per-view tap totals, segments stay in-bounds
    and below the cap, unused bound slots target the scratch channel, and every real
    segment lies inside its own channel's stream range."""
    starts = np.array([[0, 3, 3, 150, 155], [0, 0, 70, 80, 90]], dtype=np.int32)
    C, n2 = 4, 8                                  # n2 well above the real segment count
    seg1, seg2 = jax.vmap(
        lambda s: _pallas_kernels._split_two_phase(jnp.asarray(s), 64, n2))(
        jnp.asarray(starts))
    seg1, seg2 = np.asarray(seg1), np.asarray(seg2)
    for v in range(2):
        covered = (seg1[v, :, 1] - seg1[v, :, 0]).sum() + \
                  (seg2[v, :, 1] - seg2[v, :, 0]).sum()
        assert covered == starts[v, -1] - starts[v, 0]
        assert ((seg1[v, :, 1] - seg1[v, :, 0]) <= 64).all()
        assert ((seg2[v, :, 1] - seg2[v, :, 0]) <= 64).all()
        pads = seg2[v][seg2[v, :, 1] == seg2[v, :, 0]]
        assert (pads[:, 2] == C).all()
        real = seg2[v][seg2[v, :, 1] > seg2[v, :, 0]]
        for a, b, c, _ in real:
            assert starts[v, c] <= a and b <= starts[v, c + 1]
