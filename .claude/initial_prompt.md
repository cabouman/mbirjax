We're continuing the projector-kernel performance campaign in `mbirjax`, working branch
`greg/kernel_investigation`.  This session has two focuses: (1) roll the kernel-algorithm
work out to the MULTIAXIS and TRANSLATION geometries, and (2) design (and possibly implement)
Phase D, the `n_p_center` precompute that fixes the known JAX rounding bug.

Read for orientation (verify any code claim against the actual code; docs may lag):
1. `.claude/claude_prompt.md` — collaboration style + workflow (stage only, no commits;
   GPU work runs on gautschi via sbatch; discuss before library edits; sweep, don't guess).
2. `plans/projector_kernels/fwd_back_findings.md` — THE campaign document: developer
   overview of everything changed relative to prerelease, per-phase results, methodology,
   and the numerics notes.  Start here.
3. `.claude/lessons.md` — the engineering playbook.
4. Auto-memory `parallel-fwd-kernel-slowness` has the compressed campaign history.

Companion repos parallel to mbirjax: `mbirjax_metrics` (performance tracking; the
measure_one_cell harness lives there), `mbirjax_applications`.

## State (as of 2026-07-08)

Parallel and cone are DONE.  The machinery to reuse:
- **TilePolicy** (`model.tiles`): every projector batching/banding knob + the two
  kernel-algorithm flags, selected in ONE method per class (`_select_tile_policy`, re-run on
  each device re-layout), read late-bound; experiment override idiom
  `model.tiles = model.tiles._replace(...)`.
- **Kernel-algorithm flags** in `ProjectorParams` (static, int leaves): `sort_by_channel`
  (forward channel reduction: GPU sorted segment-sum via `projectors.channel_scatter_reduce`,
  CPU scatter loop verbatim) and `back_stacked_gather` (back: one stacked
  (psf_width * num_pixels, num_rows) gather; GPU, parallel-beam only).
- 1024³ n=1 H100 scoreboard: parallel fwd 35→8.2 s, parallel back 18.2→10.9 s, cone fwd
  41.5→19.4 s, cone back measured-no-safe-win.
- Verification pattern per change: unit kernel-equality tests (both branches; coeff_power 1
  and 2 where relevant) → full CPU suite + 4-device sharding suite → metadata-stripped
  compiled-HLO diff proving the CPU program bit-identical → isolated harness cells on
  gautschi (ONE measure_one_cell process per device count), old-vs-new library snapshots.

## Focus 1: multiaxis + translation rollouts

Both geometries have HORIZONTAL fans with the same channel-scatter forward structure
(`multiaxis_parallel.py` ~L303 forward / ~L365 back; `translation_model.py` ~L605
compute_horizontal_data) and VERTICAL fans like cone.  Apply the playbook:

- Forward horizontal fans: wire `sort_by_channel` (stacked taps + `channel_scatter_reduce`,
  original loop verbatim in the else branch), per-geometry `_select_tile_policy` override,
  kernel-equality tests.  Expect cone-like end-to-end gains (1.4–1.6×) diluted by the
  vertical fans.
- Back kernels: **measure the COMPOSITION first** — cone's lesson: the stacked back gather
  won 1.75× in isolation but was a 1.00× no-op in the full kernel because XLA overlaps the
  horizontal-fan gather with the vertical-fan band work.  Multiaxis/translation back also
  have vertical fans, so expect the same; a cone_back_kernel_ab.py-style bench decides in
  one GPU job.  Do NOT wire `back_stacked_gather` without that evidence.
- **Translation is the psf_radius 2–3 case** (psf_width 5–7 taps).  Two flagged checks
  before trusting the parallel-derived constants: (a) re-run the band-width crossover sweep
  (`SORTED_CHANNEL_REDUCE_MIN_COLS = 48` was measured at psf_width 3; the scaling analysis
  says roughly stable, but verify); (b) watch the (psf_width * num_pixels, band) stacked
  transient in peak memory — at psf_width 7 it is 2.3× the width XLA fused fine at 3.
- Several multiaxis/translation kernel sites are also in the rounding-bug inventory (see
  Focus 2) — coordinate kernel-signature changes if both land in this session.
- Tiling sweeps for these geometries are OPTIONAL (their harness cells top out at
  512-class); the sweep scripts in `plans/experiments/projector_kernels/` generalize easily.
- Harness cells exist for both geometries (multiaxis GPU sizes 129/256/512/513-class;
  translation 15x65/15x256/15x257) — verify with the usual isolated-cell A/B.

## Focus 2: Phase D — the n_p_center precompute (rounding-bug fix)

Read `plans/bugs_and_artifacts/jax rounding bug/jax_rounding_bug.md` (and its
`lax_map_scatter_bug/` companions).  Summary: `jnp.round` of an in-jit continuous projection
coordinate inside a vmap→map→scatter chain can mis-optimize (antisymmetric ±1-channel
errors, ~0.5/pixel; we re-observed the signature during this campaign as compilation-order
value differences).  The verified fix (T15j) is to precompute the integer center indices and
pass them into the projector's program as CONCRETE arrays.

**The 2026-05 implementation plan in that doc is STALE in one load-bearing respect**: its
§4.1 assumed per-tile host loops, but the June module-jit refactor moved the pixel/view
batching INSIDE the jitted drivers.  Materializing full-grid (views × pixels) int32 indices
is ~4 GB at 1024³ (int16: 2 GB; one 128-view chunk: 0.5 GB; VCD subsets: MBs).  The design
decision: host-side view tiling in the public wrappers vs int16 vs computing n_pc in a
separate small jit per view chunk (CONCRETENESS is what breaks the bug precondition, not
host-origin) — likely a hybrid that chunks only when views × pixels is large.
- The reduction primitive is provenance-agnostic (takes `n` however produced), and the
  sorted path uses lax.sort_key_val keys-AS-ids, so Phase D does not touch the reduction.
  Once n_pc is concrete, cached per-(view, pixel-batch) sort permutations become an optional
  further GPU refinement.
- §4.5 of the bug doc inventories every round-in-jit site (parallel, cone ×3, multiaxis ×4,
  translation ×3).  Uniform treatment via a per-geometry container + kernel-signature change
  (its §4.2/§4.7 sketches).  This is a DESIGN-DISCUSSION-FIRST item: propose the revised
  plan and get approval before implementing.

## Practical notes (hard-won this campaign)

- Cluster: standing infra in gautschi `~/viewbatch_fix_verify/` (slurm templates, bench +
  sweep scripts) and `~/kernel_ab_old` / `~/kernel_ab_new` snapshot dirs (ship OLD via
  `git archive HEAD mbirjax | ssh ... tar -x`, NEW as a working-tree tar; PYTHONPATH
  selects; ALWAYS assert `mbirjax.__file__` — `python -c` puts cwd ahead of PYTHONPATH).
  The cluster conda-module upgrade orphaned `mbirjax_regression`: batch jobs use
  `PYBIN=$HOME/.conda/envs/mbirjax/bin/python` directly (jax 0.10.1 + cuda12 plugin).
  Partition `ai`, account `bouman`, 14 CPUs per GPU (`-n 56` for 4 GPUs), `#!/bin/bash -l`.
- Measurement: one measure_one_cell process per device count (peak_bytes_in_use is
  process-cumulative); min-of-trials; the cone vcd 200³ single-trial cell is NOISY (±6%
  band across identical code) — don't read it as signal.  Value gates are NEVER max-error:
  use fraction-deviating or scale-relative sums (isolated rounding-tie flips are the known
  bug, not a variant error).
- CPU-parity proof: lower the model-level op via `P._jit_sparse_*.lower(...)`, compile,
  strip `metadata={...}` + the FileNames/source-location tables, diff old vs new.
- Nightly watch items still open: memory-gate acks needed for parallel fwd small cells
  (+27–58% rel, ≤0.65 GB abs) and cone fwd 1024³ (+9.6/+29/+46% at n=1/2/4); forward
  fingerprint samples can legally flip at rounding ties; confirm `greg/kernel_investigation`
  is nightly-tracked.

Working reminders (unchanged): stage only / draft commit messages (Greg commits from
PyCharm); exact equality is never the gate for computed floats; any new script must set env
vars / `import mbirjax` before anything touches jax.
