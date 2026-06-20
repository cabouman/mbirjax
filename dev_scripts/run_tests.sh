#!/bin/bash

echo "Running pytest with multiple workers on all tests."

# Use `python -m pytest`, not bare `pytest`: the latter resolves to whatever pytest console
# script is first on PATH, which on some clusters is a stale ~/.local/bin/pytest whose shebang
# Python can't import pytest.  `python -m pytest` runs pytest via the active environment's
# interpreter (the one `which python` points to), so it uses the env's pytest regardless of
# PATH ordering.
# -ra: print the short test-summary block (incl. `FAILED <nodeid>` lines) so the regression
# harness can capture WHICH tests failed for the dashboard, not just the count.
python -m pytest -ra -n 10 ../tests


