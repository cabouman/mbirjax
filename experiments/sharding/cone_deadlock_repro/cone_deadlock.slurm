#!/bin/bash
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -n 112
#SBATCH -t 04:00:00
#SBATCH -J cone-deadlock
#SBATCH --mail-user=buzzard@purdue.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -o slurm-cone-deadlock-%j.log
#
# cone_deadlock.slurm -- submit the cone-vs-parallel multi-GPU deadlock sweep as a batch job.
#
#   cd <this directory's checkout>
#   sbatch experiments/sharding/cone_deadlock_repro/cone_deadlock.slurm
#
# Prereqs (one-time, mirroring the regression harness):
#   * the node preamble at $HOME/load_conda_cuda.sh (module load conda/cuda/cudnn + RCAC proxy) --
#     see mbirjax_metrics/tooling/regression/cluster_preamble.sh.example;
#   * a conda env (default "mbirjax") that has THIS branch of mbirjax importable.  If it isn't
#     installed there yet, set REPRO_MBIRJAX_DIR to the repo root and this script will `pip install -e`
#     it before running.
#
# Overridable knobs (see run_sweep.sh): GEOMETRIES, SIZES, DEVICE_COUNTS, ITERATIONS, PER_TIMEOUT,
# DUMP_DEVICE_COUNTS.  Example to also probe the size that actually failed:
#   SIZES="256 512 1024" sbatch experiments/sharding/cone_deadlock_repro/cone_deadlock.slurm
set -uo pipefail

# 1) node preamble: modules (conda/cuda/cudnn) + proxy.
PREAMBLE="${PREAMBLE_FILE:-$HOME/load_conda_cuda.sh}"
if [ -f "$PREAMBLE" ]; then
  # shellcheck disable=SC1090
  source "$PREAMBLE"
else
  echo "FATAL: preamble not found at $PREAMBLE (set PREAMBLE_FILE)"; exit 1
fi

# 2) activate the conda env that has mbirjax (this branch) installed.
CONDA_ENV="${REPRO_CONDA_ENV:-mbirjax}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || { echo "FATAL: conda activate '$CONDA_ENV' failed"; exit 1; }

# 3) optional editable install of the branch under test.
if [ -n "${REPRO_MBIRJAX_DIR:-}" ]; then
  echo "pip install -e $REPRO_MBIRJAX_DIR[cuda12]"
  pip install -e "${REPRO_MBIRJAX_DIR}[cuda12]" >/dev/null || { echo "FATAL: pip install failed"; exit 1; }
fi

# 4) identify the EXACT build under test.  mbirjax has no module __version__; model.version comes
#    from importlib.metadata, which for an editable install is only the DECLARED version -- the real
#    build id is the module file path + the git SHA of that checkout.
echo "host=$(hostname)  env=$CONDA_ENV"
REPO_ROOT="$(python -c 'import os,mbirjax; print(os.path.dirname(os.path.dirname(os.path.abspath(mbirjax.__file__))))' 2>/dev/null)"
python - <<'PY'
import mbirjax, jax
try:
    from importlib.metadata import version
    ver = version("mbirjax")
except Exception as e:                       # noqa: BLE001
    ver = "(unavailable: %s)" % e
print("mbirjax module :", mbirjax.__file__)
print("mbirjax version:", ver, "(declared/installed metadata -- not the git commit)")
print("jax            :", jax.__version__)
print("devices        :", jax.devices())
PY
if [ -n "$REPO_ROOT" ] && git -C "$REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
  echo "mbirjax git    : $(git -C "$REPO_ROOT" rev-parse --short HEAD) on $(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)  (repo: $REPO_ROOT)"
  [ -n "$(git -C "$REPO_ROOT" status --porcelain -uno)" ] && echo "                 (NOTE: working tree has uncommitted changes)"
else
  echo "mbirjax git    : (no git repo at '$REPO_ROOT' -- non-editable install?)"
fi
nvidia-smi -L || true

# 5) locate the repro scripts.  SLURM copies the batch script into a spool dir, so BASH_SOURCE is
#    useless here -- derive the repro dir from the mbirjax checkout (so it always matches the lib
#    under test), with a REPRO_DIR override and a submit-dir fallback.
HERE="${REPRO_DIR:-${REPO_ROOT:+$REPO_ROOT/experiments/sharding/cone_deadlock_repro}}"
{ [ -n "$HERE" ] && [ -d "$HERE" ]; } || HERE="${SLURM_SUBMIT_DIR:-$PWD}/experiments/sharding/cone_deadlock_repro"
[ -f "$HERE/run_sweep.sh" ] || { echo "FATAL: run_sweep.sh not found under '$HERE' (set REPRO_DIR to the cone_deadlock_repro dir)"; exit 1; }
echo "repro scripts  : $HERE"

# 6) run the sweep; output lands next to where you submitted from.
export OUTDIR="${OUTDIR:-${SLURM_SUBMIT_DIR:-$PWD}/cone_deadlock_out_${SLURM_JOB_ID:-manual}}"
exec bash "$HERE/run_sweep.sh"
