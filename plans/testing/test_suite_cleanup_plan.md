# mbirjax test-suite cleanup plan (RNG seeding + keep/simplify/delete)

## EXECUTION STATUS (Greg-approved: RNG Tier1+3, deletions+bug-fix, P1+P2 merges)

DONE + verified (nothing committed by me; staged in the working tree for Greg):
- **RNG Tier 1+3**: fixed seed in test_projectors.py (adjoint+Hessian PRNGKey(0), translation vectors
  default_rng(0)); seeded test_qggmrf::test_alpha_derivative; added the conftest autouse
  `_seed_global_numpy_rng` safety net. (test_preprocessing needed nothing -- already seeded via seed=.)
  Phase-A full suite green: 355 passed, fixture safe across all 40 files.
- **Bug fix**: test_vcd_sharded.py `_recon` now forwards `weights=` (the sharded non-const-weights path
  works -- the fix just made the test actually exercise it).
- **Deletions** (test methods removed via edit): test_scaffolding (2 tautologies; kept preferred_devices),
  test_fbp_recon direct_recon alias, test_placement (2), test_vcd_sharded no_halo, test_utilities (2
  batching) AFTER migrating their unique multi-output coverage into test_batching_helpers (verified 5/5).
- **P1 merge**: NEW tests/geometries/test_banded_projectors.py (3 banded -> 1 parametrized; 12 cells pass;
  every original cell mapped 1:1; model_class stored per config).
- **P2 merge**: NEW tests/sharding/test_geometry_sharded.py (3 sharded geometry -> 1; 26 subtests, 0
  skipped) + ADDITIVE tests/sharding/conftest.py (jnp import + _random_sino/_random_recon/
  _usable_device_counts lifted); multiaxis divisibility guard GENERALIZED to all 3 geometries (superset);
  cone VCD gate kept cone-only.

### FILE DELETIONS FOR GREG (superseded by the merges; file deletion is Greg's action)
- tests/geometries/test_cone_banded.py, test_multiaxis_banded.py, test_translation_banded.py
- tests/sharding/test_cone_sharded.py, test_multiaxis_sharded.py, test_translation_sharded.py
- (optional) tests/sharding/test_scaffolding.py wholesale, if folding preferred_devices into conftest.
Final gate: full suite run with the 6 originals --ignore'd (post-deletion state) -- validates the merged
files replace them. GREG COMMITTED + PUSHED all of the above, including the 6 file deletions.

### ROUND 2 (Greg-approved after a cost-benefit review): the 2 coverage FIXES + P4 -- DONE
- **Strengthen test_pallas_kernels::test_weights_match_kernel_formula**: was shape+non-negativity only
  (didn't test the one-hot claim in its docstring). Now verifies the actual weight VALUES against an
  INDEPENDENT XLA oracle -- back-projects a sinogram nonzero only in the 3 owned views and reconstructs
  the identical result from the weights + per-view rounded centers (probe-confirmed exact to ~5e-7).
- **Strengthen test_hsnt::test_estimate_subspace_dimension_sanity**: was loose bounds ([1, wavelengths]).
  The clean data is EXACTLY rank subspace_dim, so now asserts `est == self.subspace_dim` (estimator
  recovers the exact rank -- verified).
- **P4**: extracted the HDF5 save/load round-trip out of test_vcd.py::verify_vcd (which re-ran it
  identically for all 7 geometries) into one dedicated `test_save_load_recon_hdf5_roundtrip` (small
  parallel recon, 2 iters). All 3 pass; full suite re-run.

### LOW PRIORITY (deferred -- weak cost-benefit; do opportunistically, not as a dedicated pass)
- **P3** (collapse the sharding output-form matrices + lift setUp/_divisible to conftest + merge n=2 into
  the sweeps): MEDIUM benefit / MED-HIGH cost -- biggest remaining line reduction but delicate sharding
  refactor. Do only if the duplication actively bothers.
- **P5 / P6** (parametrize pallas adjoint/chunking + preprocessing crop/wrapper tests): LOW benefit
  (maintainability only, no coverage/wall-time change) / LOW cost -- fold in whenever next editing those
  files.
- **RNG Tier 2** (13 files global->local default_rng): LOWEST ratio -- the conftest autouse fixture already
  removed the order-dependence risk, and the migration flips the RNG algorithm (MT19937->PCG64), risking
  spurious re-tuning of value-based tests. SKIP unless a specific need arises.



Analysis only -- NO code changes yet. Produced from a 40-file fan-out evaluation (1 agent/file) +
synthesis, plus a full RNG audit. Full suite currently ~259 passed on CPU (313 counting the sharding
virtual-device cells). 40 test files: `tests/` + `tests/geometries/` (11) + `tests/sharding/` (15).

## HEADLINE

The suite is HEALTHY. Per-test rollup: **definite-keep 224, likely-keep 88, borderline 9, delete-low-loss
6, delete-no-loss 1**. Almost nothing is obsolete -- the "many tests may not serve a purpose" hypothesis
is largely NOT borne out. The real wins are STRUCTURAL CONSOLIDATION (clone files the per-file agents
flagged but couldn't merge), not deletion. Plus one genuine test BUG and the RNG-seeding fixes.

---

# ITEM 1 -- RNG seeding

Audit: all `RandomState(...)`/`default_rng(...)` calls are seeded. The problems are with the GLOBAL
`np.random`.

**Tier 1 -- non-reproducible tests (fix now; small, high value):**
- `tests/geometries/test_projectors.py` -- `seed_value = np.random.randint(1000000)` -> `PRNGKey` (163,
  301) + `np.random.uniform` translation vectors (98,100), NO seed -> random every run. **Greg decided:
  FIXED seed** (`PRNGKey(0)` + `default_rng(0)`); the adjoint identity holds for any input so determinism
  costs nothing.
- `tests/test_qggmrf.py::test_alpha_derivative` -- `np.random.rand` for p/q/T/sigma_x + recon/delta, no
  seed (its siblings seed or hardcode). Seed it.
- `tests/test_preprocessing.py::setUp` -- `generate_dark_scan` builds dark/blank/obj scans (110-117)
  BEFORE `np.random.seed(25)` (121) and callers don't pass `seed=` -> synthetic scans effectively
  unseeded/order-dependent. Seed before scan generation (or pass `seed=`).

**Tier 2 -- isolation (follow-up; mechanical, 13 files):** migrate global `np.random.seed(S)` +
`np.random.<fn>` to per-test `rng = np.random.default_rng(S)` + `rng.<fn>` in
`test_{denoiser,hsnt,preprocessing,prox,qggmrf,recon_restart}`, `geometries/{auto_geometry,split_overlap,
vcd}`, `sharding/{cone_sharded,mar,padding,vcd_sharded}`. Removes cross-test order-dependence.

**Tier 3 -- safety net (now; cheap):** `conftest.py` autouse fixture `np.random.seed(<fixed>)` before each
test, so any stray global use is deterministic regardless of Tier 2.

---

# ITEM 2 -- keep / simplify / delete

## A. Cross-file redundancy (the main finding the per-file pass could not see)

- **3 `*_banded.py` files byte-structurally identical** (cone/multiaxis/translation): same helpers,
  RTOL/ATOL, band-size + coeff_power loops; differ only by model class + geometry factory (~490 lines,
  ~2/3 duplicated harness). NOT redundant coverage -- redundant scaffolding.
- **3 `*_sharded.py` geometry files byte-identical helpers** (`_random_sino/_random_recon/_usable_device_
  counts/_sweep`) + identical `test_back/forward/hessian` bodies; only model factory + label differ.
- **Output-form matrix** (plain/sharded input x `output_sharded`) re-implemented in ~5 sharding files
  (back/forward/fbp/fbp_recon); each needs ITS op's check, but within-file the 3-4 siblings collapse to
  one parametrized matrix; `setUp`/`_divisible` are identical between forward & back -> lift to conftest.
- **`direct_recon`/`direct_filter` alias** (one-line delegates) checked 3x; the single-device
  `test_fbp_recon.py::test_direct_recon_matches_fbp_recon` is redundant (already covered sharded).
- **n=2 double-covered**: `test_sharded_matches_single_device` + `test_device_count_sweep` both assert
  n=2 in back/forward/fbp_recon -> merge (n=2 becomes the sweep's plain-out leg).
- **Batching helpers** duplicated: `test_batching_helpers.py` (dedicated) vs `test_utilities.py::test_
  {sum,concatenate}_function_in_batches` -- migrate the unique tuple/diff-size cases into the former,
  delete the two from utilities.
- **hdf5 round-trip run 7x**: `test_vcd.py::verify_vcd` re-runs `save/load_recon_hdf5` in all 7 geometry
  tests -> extract one dedicated round-trip test, drop from verify_vcd.

## B. DELETE -- no loss (tautological / obsolete / fully redundant)

- `sharding/test_scaffolding.py::test_multiple_devices_available` -- `assertTrue(True)` tautology.
- `sharding/test_scaffolding.py::test_device_setup_flag_present` -- asserts conftest's own env-var plumbing.
- `sharding/test_fbp_recon.py::test_direct_recon_matches_fbp_recon` -- redundant single-device alias.
- `sharding/test_vcd_sharded.py::TestHaloMath::test_no_halo_matches_legacy_reflected_bc` -- subsumed by
  `test_trivial_bit_exact` + `test_boundary_self_consistency`.
- `sharding/test_placement.py::test_move_shard_same_device_preserves_values` -- covered by test_primitives.
- `test_utilities.py::test_sum_function_in_batches`, `::test_concatenate_function_in_batches` (after
  migrating their unique cases to test_batching_helpers.py).
- `sharding/test_placement.py::test_multidevice_target_attenuation_matches_single` -- combination covered
  by the dividing test + attenuation-scale test.
- Likely wholesale: `sharding/test_scaffolding.py` (fold `test_preferred_devices_returns_two` into the
  sharding conftest as a one-time guard, delete the file). NEEDS GREG CONFIRM.

## C. ONE GENUINE BUG (Greg decision: fix or delete)

- `sharding/test_vcd_sharded.py::test_sharded_recon_matches_single_device_nonconst_weights` -- the `_recon`
  helper accepts `weights=` but NEVER forwards it to `model.recon`, so ref and sharded both run with
  DEFAULT weights -> silently duplicates the sweep test; the sharded non-const-weights path on a
  non-padded layout is currently UNtested. **Recommend FIX** (one line: forward `weights=weights`); the
  padded path is correctly covered by `test_padding.py::test_recon_matches_nonconst_weights`.

## D. BORDERLINE / strengthen-or-drop (Greg decisions)

- `test_pallas_kernels.py::test_weights_match_kernel_formula` -- asserts only shape + non-negativity, NOT
  the one-hot equivalence its docstring claims; `*_matches_xla` covers weight correctness. Strengthen
  (compare vs one-hot XLA back-projection) OR drop.
- `test_preprocessing.py::TestGetSinoAndModel::test_zeiss_ultra_builds_parallel_model` -- because `_compute`
  is mocked, does NOT exercise versa/ultra scanner dispatch; only re-verifies build_model on a parallel
  model. Fold into the reader-wrapper parametrization (or accept as a thin class-selection smoke test).
- `sharding/test_fbp.py::test_direct_filter_is_fbp_filter_alias`, `sharding/test_primitives.py::test_is_
  dev2dev_safe_two_devices` -- near-tautologies; fold assertion / strengthen rather than standalone.
- `test_hsnt.py::test_estimate_subspace_dimension_sanity` -- sole coverage of the auto-estimate path but
  assertions too loose (would pass for a wrong value); strengthen rather than delete.
- `geometries/test_vcd.py::test_vcd_anisotropic_translation` -- near-dup of test_vcd_translation; keep if
  the aniso translation VCD path is worth a full recon, else drop (aniso is gated elsewhere).

## E. SIMPLIFY / CONSOLIDATE (ordered by payoff -- all preserve every coverage cell)

- **P1** merge the 3 `*_banded.py` -> one `test_banded_projectors.py` parametrized on (model, config).
- **P2** merge the 3 `*_sharded.py` geometry files; lift `_random_sino/_random_recon/_usable_device_counts/
  _sweep` to `tests/sharding/conftest.py`; fold in the multiaxis divisibility guard. Keep the cone VCD gate.
- **P3** collapse the per-file output-form matrices (back/forward/fbp/fbp_recon) + lift identical
  `setUp`/`_divisible` from forward&back to conftest; merge the n=2 leg into the sweeps. (Also fills the
  currently-missing sharded-in/sharded-out cell in forward.)
- **P4** extract the hdf5 round-trip out of `test_vcd.py::verify_vcd` (removes 6 redundant runs).
- **P5** parametrize `test_pallas_kernels.py` adjoint quartet + chunking trio.
- **P6** parametrize `test_preprocessing.py` per-reader crop tests (TestConfigCropUnification 6) + wrapper
  tests (TestGetSinoAndModel), keeping the distinct cells (raw-pitch/downsample, zeiss-ultra no-source,
  zeiss_tct 3-tuple, auto_crop).
- Minor: parametrize the 3 `TestSupportRadius` tests + the 2 `test_view_params.py` setter tests
  (parallel/cone); drop `test_pad_fraction_zero_helical` (composition of two others).

## F. DEFINITE-KEEP -- load-bearing (do NOT touch), by area

Projector adjoint/Hessian across 8 geometries + the 2 absolute-scaling anchors; all `test_fbp_fdk.py`
(sole ground-truth); `test_vcd.py` full-convergence gates + split_sino; nearly all `test_auto_geometry.py`
incl. the NaN regression guard + zero/tightness pair + axial_pad_fraction facets; sharding invariants
(forward/back adjoint+sweep+output-form, all of `test_padding.py`, `test_primitives.py`, placement banded
adjoint); sharded prior/VCD (`test_vcd_sharded.py` halo->prior->recon->audits, `test_denoise_sharded.py`);
pallas `*_matches_xla` + adjoint + dispatch spies; and the explicit fixed-bug regression guards --
`test_scatter_centers.py` (rounding), `test_channel_reduce.py` (1280-col cap), `test_recon_restart.py`,
`test_device_config.py` (use_gpu), `test_mar.py` (2^31), `test_utilities.py` OOM + axis-flip roundtrip,
`test_projector_cache_sharing.py`, `test_prox.py`, `test_tile_policy.py` memory caps. The 2 fan-out
dropouts self-evaluated: `test_qggmrf.py` (all 3 core qGGMRF; alpha_derivative needs the Tier-1 seed) and
`test_view_params.py` (all 4 core; the 2 setter tests parametrize) -- all keep.

## G. Suite-level notes

- Heaviest (compile-bound): test_vcd, test_vcd_sharded, test_fbp_fdk, test_padding; tile_policy ~11s. The
  biggest WASTED wall time is the n-sweep boilerplate + hdf5-run-7x.
- n=8 sweep legs never run under the default 4-virtual-device CI (only >=8-GPU clusters) -- note, not fix.
- Change-detector oracles (mirror production logic): `test_hooks.py::test_auto_device_count_uses_all_
  devices`, `test_placement.py::test_target_applies_analytic_scale` -- lean on their hard-coded cases.
- Stale module docstrings claiming bit-exact 1-device tests: `test_fbp_recon.py`, `test_forward_projection.py`.

## H. Suggested order of operations

1. GREG DECISIONS: (a) fix-or-delete the broken non-const-weights test; (b) strengthen-or-drop
   `test_weights_match_kernel_formula`; (c) confirm deleting `test_scaffolding.py` wholesale; (d) RNG Tier
   scope (Tier 1+3 now vs also Tier 2).
2. RNG Tier 1 + Tier 3 (small, isolated).
3. Pure deletions (§B) -- zero risk, shrink surface first.
4. Non-test cleanups: dead prints/commented blocks (test_projectors), stale docstrings, dead setUp.
5. P1 + P2 (clone-file merges; P2 builds the conftest helper home).
6. P3 + P4.
7. P5-P6 opportunistically.
Run the full CPU suite after each step to confirm parity (every merge preserves all coverage cells).
