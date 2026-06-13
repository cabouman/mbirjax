"""
experiments/sharding/features/settable_view_parameters_demo.py
──────────────────────────────────────────────────────────────
Demonstration of TomographyModel.set_view_parameters (settable view parameters).

The view-dependent parameters (projection angles for parallel/cone beam) used to be
baked into the jitted projectors as a compile-time constant, so changing an angle
meant set_params(angles=...) and a full projector recompile.  They are now a RUNTIME
input to the jitted projectors: set_view_parameters changes the values with no
recompile (only the shape is static -- a view-COUNT change is still a geometry change
and goes through set_params).

Three demonstrations, each printed with its own banner:

  1. CORRECTNESS -- a model whose angles are CHANGED to B projects identically
     (bit-exact) to a model BUILT with B.  The values really are runtime inputs;
     nothing stale is baked anywhere.

  2. NO RECOMPILE -- changing angles via set_params re-creates the projectors and
     pays jit compilation again on the next projection; set_view_parameters pays
     only the projection itself.  The timing gap is the feature.

  3. PROJECT AT ONE ANGLE (the vcls pattern) -- a ONE-view model swept over
     candidate angles answers "which angle produced this measured view?" by
     re-projecting at each candidate with set_view_parameters: one jit compile
     total, then pure value updates.  This is the natural form of per-view work
     that view_indices used to approximate by indexing the full model's baked
     angle table (mechanism retired at P6), and it is what lets vcls run with no
     per-view recompiles.

Sizes are small so the demo runs in well under a minute on CPU.  No CLI args; edit
the configuration below.

    python experiments/sharding/features/settable_view_parameters_demo.py
"""
import time

import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
NUM_VIEWS = 64
NUM_ROWS = 32          # detector rows = recon slices (parallel beam)
NUM_CHANNELS = 64
NUM_ANGLE_CHANGES = 5  # how many times each path in demo 2 changes the angles
SEED = 0


def banner(title):
    print('\n' + '=' * 72)
    print(title)
    print('=' * 72)


def main():
    import mbirjax as mj
    import jax.numpy as jnp

    rng = np.random.default_rng(SEED)
    sino_shape = (NUM_VIEWS, NUM_ROWS, NUM_CHANNELS)
    angles_a = jnp.asarray(np.sort(rng.uniform(0, np.pi, NUM_VIEWS)).astype(np.float32))
    angles_b = jnp.asarray(np.sort(rng.uniform(0, np.pi, NUM_VIEWS)).astype(np.float32))

    model = mj.ParallelBeamModel(sino_shape, angles_a)
    recon_shape = model.get_params('recon_shape')
    phantom = mj.generate_3d_shepp_logan_low_dynamic_range(recon_shape)

    # ── 1. Correctness ────────────────────────────────────────────────────────
    banner('1. CORRECTNESS: set_view_parameters(B) == a model built with B')
    sino_at_a = model.forward_project(phantom)          # also primes the jit cache
    model.set_view_parameters(angles_b)
    sino_changed = model.forward_project(phantom)

    fresh_b = mj.ParallelBeamModel(sino_shape, angles_b)
    sino_fresh = fresh_b.forward_project(phantom)

    # The two models compile SEPARATE executables for the same program, and on GPU
    # two compilations of identical code can differ by ~1 ULP (autotuning / reduction
    # order) -- so the right check is tight float closeness, not bit equality.  A
    # stale baked angle would fail at O(1), as the sino_B-vs-sino_A line shows.
    max_diff = float(np.max(np.abs(np.asarray(sino_changed) - np.asarray(sino_fresh))))
    moved = float(np.max(np.abs(np.asarray(sino_changed) - np.asarray(sino_at_a))))
    print(f'changed-model vs fresh-model: max |difference| = {max_diff:.2e}  (float noise)')
    print(f'and the values really changed (max |sino_B - sino_A| = {moved:.3f})')
    np.testing.assert_allclose(np.asarray(sino_changed), np.asarray(sino_fresh),
                               rtol=1e-5, atol=1e-5)

    # ── 2. No recompile ───────────────────────────────────────────────────────
    banner('2. NO RECOMPILE: set_params(angles=...) vs set_view_parameters(...)')
    new_angle_sets = [jnp.asarray(np.sort(rng.uniform(0, np.pi, NUM_VIEWS)).astype(np.float32))
                      for _ in range(NUM_ANGLE_CHANGES)]

    # Old path: set_params marks the geometry changed and re-creates the projectors,
    # so the next projection pays jit compilation again, for the same shapes.
    model_old_path = mj.ParallelBeamModel(sino_shape, angles_a)
    _ = model_old_path.forward_project(phantom)         # initial compile (not timed)
    t0 = time.time()
    for angles in new_angle_sets:
        model_old_path.set_params(angles=angles)
        _ = np.asarray(model_old_path.forward_project(phantom))
    t_recompile = (time.time() - t0) / NUM_ANGLE_CHANGES

    # New path: pure value update; the jitted projectors are reused as-is.
    model_new_path = mj.ParallelBeamModel(sino_shape, angles_a)
    _ = model_new_path.forward_project(phantom)         # initial compile (not timed)
    t0 = time.time()
    for angles in new_angle_sets:
        model_new_path.set_view_parameters(angles)
        _ = np.asarray(model_new_path.forward_project(phantom))
    t_no_recompile = (time.time() - t0) / NUM_ANGLE_CHANGES

    print(f'per angle-change + projection, averaged over {NUM_ANGLE_CHANGES} changes:')
    print(f'  set_params(angles=...)      : {t_recompile * 1000:8.1f} ms   (recompiles projectors)')
    print(f'  set_view_parameters(...)    : {t_no_recompile * 1000:8.1f} ms   (value update only)')
    print(f'  speedup                     : {t_recompile / t_no_recompile:8.1f} x')

    # ── 3. Project at one angle (the vcls pattern) ────────────────────────────
    banner('3. PROJECT AT ONE ANGLE: find which angle produced a measured view')
    # "Measure" one view at a secret angle by building a 1-view model there.
    secret_angle = float(angles_b[NUM_VIEWS // 3])
    one_view_shape = (1, NUM_ROWS, NUM_CHANNELS)
    measured_model = mj.ParallelBeamModel(one_view_shape, jnp.asarray([secret_angle]))
    measured_model.set_params(no_warning=True, recon_shape=recon_shape)
    measured_view = np.asarray(measured_model.forward_project(phantom))

    # Sweep candidates on ONE 1-view model: set_view_parameters per candidate --
    # one compile total, then |candidates| cheap value updates.  (vcls does exactly
    # this for its per-view basis functions.)
    candidates = np.linspace(0, np.pi, 256, endpoint=False).astype(np.float32)
    sweep_model = mj.ParallelBeamModel(one_view_shape, jnp.asarray([candidates[0]]))
    sweep_model.set_params(no_warning=True, recon_shape=recon_shape)
    _ = sweep_model.forward_project(phantom)            # the one compile
    t0 = time.time()
    errors = np.empty(len(candidates))
    for k, theta in enumerate(candidates):
        sweep_model.set_view_parameters(jnp.asarray([theta]))
        view = np.asarray(sweep_model.forward_project(phantom))
        errors[k] = np.linalg.norm(view - measured_view)
    dt = time.time() - t0
    best = candidates[int(np.argmin(errors))]

    print(f'swept {len(candidates)} candidate angles in {dt:.2f} s '
          f'({dt / len(candidates) * 1000:.1f} ms per angle, zero recompiles)')
    print(f'secret angle: {secret_angle:.6f}   recovered: {best:.6f}   '
          f'(grid spacing {np.pi / len(candidates):.6f})')
    assert abs(best - secret_angle) <= np.pi / len(candidates)
    print('\nDone.')


if __name__ == '__main__':
    main()
