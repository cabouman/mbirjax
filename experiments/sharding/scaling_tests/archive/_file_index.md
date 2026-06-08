# scaling_tests/archive — file index

**Resolved one-off diagnostics.**  Each script here answered a specific question
that is now settled; its conclusion is recorded in the plan/status/lessons docs (or
in the script's own docstring).  They are kept verbatim for reproducibility and as
templates if a similar investigation recurs — *not* part of the live measurement
harness.

**Run caveat (why they may not run as-is from here).**  These were written to sit
next to `scaling_common.py`; several `import scaling_common` (or a sibling driver),
and a couple count `..` from `__file__` to the beta worktree root.  They were moved
**without edits**, so to reproduce one, put the parent dir on the path — from
`scaling_tests/`:

    PYTHONPATH=.. python archive/<script>.py

(or copy it back up one level).  This is intentional: an archive should preserve
exactly what was run.

## Files

- `sparse_back_project_phase_ablation.py` — split sharded back projection into
  Phase 1 (per-device compute) and Phase 2 (reduce-scatter) and timed each across
  device counts.  **Finding:** the cap is Phase 1 (bandwidth-bound); Phase 2 is ≤9%.
  Recorded in `plans/sharding_status.md`.
- `sparse_back_project_memory_attribution.py` — per-device buffer breakdown +
  slice-band sweep (the band-vs-pixel memory/throughput knee).  **Decision: KEEP
  BAND** (pixel floored at ~1.7–2.7× band peak for only ~11–16% less time).
  Recorded in `plans/sharding_implementation_plan_v2.md` §Decisions (band-vs-pixel,
  RESOLVED 2026-06-03).
- `vcd_mesh_mem_attribution.py` — attributed the 1-device-*mesh* VCD memory overhead
  (hypothesis: async dispatch with no backpressure).  **Superseded:** the real root
  cause was jax sharded-array reference cycles; **fixed** via buffer donation +
  end-of-subset `.delete()` cleanup.  Recorded in `plans/sharding_status.md` and
  `.claude/lessons.md` ("Sharded VCD memory").
- `vcd_mesh_sweep.py` — subprocess orchestrator that drove `vcd_mesh_mem_attribution`
  one config per process.  Same fixed-leak context; archived with its worker.
- `fbp_filter_row_batch_sweep.py` — swept `ROW_FILTER_BATCH` for `apply_row_filter`
  (the F1 throughput-knee study).  **Picked B=1024 on GPU.**  (Imports the live
  `fbp_filter_scaling` for its builders.)
- `projector_sino_size_scan.py` — single-device forward/back projection time vs each
  sinogram axis, mixing powers-of-two and not.  **Verified** the channel-major layout
  fix removed the power-of-2 `num_det_channels` cache-aliasing spike (CPU + GPU);
  conclusion in the script docstring.
- `plot_projector_sino_size_scan.py` — plotter for the scan above (2×2 forward/back
  vs swept axis, power-of-two points starred).
- `vcd_shard_vs_noshard.py` — user-facing `recon` head-to-head, no-shard vs shard,
  reporting time / peak memory / NRMSE.  Superseded by `vcd_recon_scaling.py` (pure
  sharded-loop peak) plus the multi-GPU scaling numbers recorded in the docs.
