#!/bin/bash
# A1 sweep on gautschi (1x H100).  Runs the equivalence gate FIRST (on-GPU
# validation of the segmented driver; job aborts if it fails), then the sweep.
# Stage this directory tree to ~/sharpness_schedule on gautschi and submit from
# the repro/ subdirectory:  sbatch run_a1_gautschi.sh
# Outputs go to /scratch/gautschi/buzzard/sharpness_schedule/a1 (auto-detected by
# a1_sweep.py); job logs land in this directory.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 04:00:00
#SBATCH -J a1_sweep
#SBATCH -o a1_sweep_%j.log

# Library: the ~/venvs/sharpness_main venv = mbirjax_regression's jax/CUDA stack
# (venv excludes the broken ~/.local user site) + an editable install of the
# origin/main worktree at ~/mbirjax_main_wt -- NOT the greg/center_slice checkout
# in ~/PycharmProjects/mbirjax, which has diverged library code.
source ~/load_conda_cuda.sh
PY=~/venvs/sharpness_main/bin/python
cd "$SLURM_SUBMIT_DIR"

echo "=== platform check (must be GPU; the nightly once measured on CPU silently) ==="
$PY -c "import mbirjax, jax; d = jax.devices(); print(d); assert d[0].platform == 'gpu', 'NOT ON GPU'" \
    || { echo "PLATFORM CHECK FAILED - aborting"; exit 1; }

echo "=== equivalence gate (GPU) ==="
$PY -u ../driver/equivalence_gate.py || { echo "GATE FAILED - aborting"; exit 1; }

echo "=== A1 sweep ==="
$PY -u a1_sweep.py
