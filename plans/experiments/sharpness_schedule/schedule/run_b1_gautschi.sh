#!/bin/bash
# b1: Phase B downsampled search (6 schedule variants + baseline; stage 2 extras
# when WINNER is set in b1_sweep.py).  Submit from schedule/:
#   sbatch run_b1_gautschi.sh
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 04:00:00
#SBATCH -J sharp_b1
#SBATCH -o /scratch/gautschi/buzzard/sharpness_schedule/logs/%x_%j.log

# Library discipline: sharpness_main venv (regression jax/CUDA + editable
# origin/main worktree); on-GPU assert per the silent-CPU lesson.
source ~/load_conda_cuda.sh
PY=~/venvs/sharpness_main/bin/python
cd "$SLURM_SUBMIT_DIR"

echo "=== platform check ==="
$PY -c "import mbirjax, jax; d = jax.devices(); print(d); assert d[0].platform == 'gpu', 'NOT ON GPU'" \
    || { echo "PLATFORM CHECK FAILED - aborting"; exit 1; }

echo "=== equivalence gate (incl. zero-offset identity) ==="
$PY -u ../driver/equivalence_gate.py || { echo "GATE FAILED - aborting"; exit 1; }

echo "=== b1 sweep ==="
$PY -u b1_sweep.py
