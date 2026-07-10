#!/bin/bash

echo "Running pytest with multiple workers on all tests."

# xdist worker count.  Every worker initialises the GPU backend at `import mbirjax`
# (_device_setup runs jax.devices() at import); too many concurrent CUDA inits on a few GPUs abort
# with "Fatal Python error: Aborted" and take the whole run down.  So cap at 4 on GPU machines
# (~one init per GPU) and use 8 on CPU, where there is no such contention.  Detect a GPU cheaply via
# nvidia-smi (no JAX init).  PYTEST_NPROC overrides (e.g. the regression harness pins it per platform).
if [ -n "${PYTEST_NPROC:-}" ]; then
  NPROC="$PYTEST_NPROC"
elif command -v nvidia-smi >/dev/null 2>&1; then
  NPROC=4
else
  NPROC=8
fi

# Use `python -m pytest`, not bare `pytest`: the latter resolves to whatever pytest console
# script is first on PATH, which on some clusters is a stale ~/.local/bin/pytest whose shebang
# Python can't import pytest.  `python -m pytest` runs pytest via the active environment's
# interpreter (the one `which python` points to), so it uses the env's pytest regardless of
# PATH ordering.
# -ra: print the short test-summary block (incl. `FAILED <nodeid>` lines) so the regression
# harness can capture WHICH tests failed for the dashboard, not just the count.
python -m pytest -ra -n "$NPROC" ../tests


