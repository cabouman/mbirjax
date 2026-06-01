# experiments/sharding/parallel_performance — file index

- `device_put_check.py` — standalone (no mbirjax) probe for the multi-GPU
  `device_put` corruption bug; tests 4 cross-device ops. Verified 2026-05-29:
  PASS on 2× H100, FAIL (A,C) on 2× L40S. Drives the transfer-helper d2d-vs-
  host-bounce decision.
- `fbp_parallel_options.md` — comparison of parallelism strategies (threading
  / shard_map / pmap) for fbp_filter; rationale for choosing Path-G threading.
  (Migrated from research `.claude/parallel_options.md`.)
- `fbp_filter_parallelism_comparison.py` — runnable comparison of every
  multi-device parallelism strategy tried for fbp_filter on GPU and virtual
  CPU; isolates each approach's bottleneck (SPMD overhead, PCIe traffic).
- `fbp_filter_parallelism_comparison.md` — write-up of the results/conclusions
  from the comparison script.
- `forward_vs_back_discussion.md` — analysis of why forward projection
  (scatter-add) is much slower than back projection (gather) on CPU, and ideas
  to improve forward-projection CPU performance.
- `sharding_scaling.py` — measures multi-device speedup (wall-clock + speedup +
  correctness vs single-device) for key operations as each sharding step lands.
  NOTE: written against the research-branch slice-axis API; will need updating
  for the beta view-sharded design before its numbers are meaningful again.
