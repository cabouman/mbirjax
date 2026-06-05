# experiments/sharding — file index

Top-level index of the sharding experiment area.  Each subdirectory has its own
`_file_index.md` with one line per file.

## Subdirectories

- `plans/` — the implementation plan and living status for the beta sharding work.
- `parallel_performance/` — benchmarks, parallelism-strategy comparisons, and the
  multi-GPU `device_put` correctness probe.

## Convention

- Each subdirectory contains a `_file_index.md` giving a one-line description of
  every file in it.  Update it when adding/removing files.
- Create new purpose-named subdirectories when existing labels don't fit (e.g.
  `temporary_tests/` for throwaway dev-time tests that need not be kept).
