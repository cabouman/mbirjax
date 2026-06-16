#!/usr/bin/env bash
# deploy_to_metrics.sh — copy the regression HARNESS (engine + wrapper) from this mbirjax checkout
# into a local mbirjax_metrics clone's tooling/.  The harness is authored here but RUNS FROM the
# metrics repo (the nightly fresh-clones metrics and execs tooling/regression/run_regression.sh), so
# after editing the harness here you deploy + commit + push the metrics clone to make it live.
#
# Copies ONLY the engine closure + wrapper.  Does NOT touch golden/ results/ state/ (data, managed
# separately) and does NOT commit/push (you review the diff, then commit from the metrics clone).
#
# Usage:   bash dev_scripts/regression/deploy_to_metrics.sh [path-to-metrics-clone]
#   default metrics clone = a sibling of the mbirjax repo:  <mbirjax>/../mbirjax_metrics
set -euo pipefail
# Keep an interactive terminal open on a nonzero exit so the error stays visible.
if [ -t 0 ]; then
  trap '_ec=$?; [ "$_ec" -ne 0 ] && { echo; echo ">>> $(basename "$0") exited with status $_ec — press Enter to close."; read -r _ || true; }' EXIT
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MBIRJAX_ROOT="$(cd "$HERE/../.." && pwd)"                 # dev_scripts/regression -> repo root
METRICS="${1:-$(cd "$MBIRJAX_ROOT/.." && pwd)/mbirjax_metrics}"

[ -d "$METRICS/.git" ] || { echo "ERROR: '$METRICS' is not a git clone (pass the path as arg 1)."; exit 1; }

SCALING_SRC="$MBIRJAX_ROOT/experiments/sharding/scaling_tests"
REG_SRC="$MBIRJAX_ROOT/dev_scripts/regression"
SCALING_DST="$METRICS/tooling/scaling_tests"
REG_DST="$METRICS/tooling/regression"
mkdir -p "$SCALING_DST" "$REG_DST"

# Engine closure (the only sibling imports are scaling_common + performance_tracking; matplotlib/
# ruamel are pip deps installed via HARNESS_DEPS).  Explicit list = no stray files / no results/.
ENGINE_FILES=(
  scaling_common.py
  performance_tracking.py
  run_nightly.py
  capture_golden.py
  capture_main_baseline.py
  run_performance_local.py
)
for f in "${ENGINE_FILES[@]}"; do
  cp -v "$SCALING_SRC/$f" "$SCALING_DST/$f"
done

# Wrapper (the *.example is a template; copy it too so the cluster has a reference).
WRAPPER_FILES=(
  run_regression.sh
  regression.env
  cluster_preamble.sh.example
  enable_nightly.sh
  disable_nightly.sh
  com.mbirjax.regression.plist
  README.md
)
for f in "${WRAPPER_FILES[@]}"; do
  [ -f "$REG_SRC/$f" ] && cp -v "$REG_SRC/$f" "$REG_DST/$f" || true
done
chmod +x "$REG_DST"/*.sh 2>/dev/null || true

echo
echo "Deployed harness -> $METRICS/tooling/"
echo "Next: review, then from the metrics clone:"
echo "    git -C '$METRICS' add tooling && git -C '$METRICS' commit -m 'deploy harness' && git -C '$METRICS' push"
echo "(The nightly fresh-clones metrics from its REMOTE, so it only sees pushed changes.)"
