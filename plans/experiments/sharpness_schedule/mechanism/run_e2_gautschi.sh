#!/bin/bash
# E2 long tail (fdk_init vs truth_init, 60 iterations) on gautschi (1x H100).
# Submit from mechanism/:  sbatch run_e2_gautschi.sh
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 04:00:00
#SBATCH -J sharp_e2
#SBATCH -o /scratch/gautschi/buzzard/sharpness_schedule/logs/sharp_e2_%j.log

# Same library discipline as the other sharpness jobs (sharpness_main venv =
# regression jax/CUDA stack + editable origin/main worktree).
source ~/load_conda_cuda.sh
PY=~/venvs/sharpness_main/bin/python
cd "$SLURM_SUBMIT_DIR"

echo "=== platform check ==="
$PY -c "import mbirjax, jax; d = jax.devices(); print(d); print('library under test:', mbirjax.__file__); assert d[0].platform == 'gpu', 'NOT ON GPU'" \
    || { echo "PLATFORM CHECK FAILED - aborting"; exit 1; }

echo "=== e2 longtail ==="
$PY -u e2_longtail.py
