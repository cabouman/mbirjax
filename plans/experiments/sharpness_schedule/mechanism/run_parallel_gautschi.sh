#!/bin/bash
# Parallel-beam hardening probe (contained + truncated, 17 iterations).
# Submit from mechanism/:  sbatch run_parallel_gautschi.sh
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 02:00:00
#SBATCH -J sharp_par
#SBATCH -o /scratch/gautschi/buzzard/sharpness_schedule/logs/sharp_par_%j.log

source ~/load_conda_cuda.sh
PY=~/venvs/sharpness_main/bin/python
cd "$SLURM_SUBMIT_DIR"

echo "=== platform check ==="
$PY -c "import mbirjax, jax; d = jax.devices(); print(d); print('library under test:', mbirjax.__file__); assert d[0].platform == 'gpu', 'NOT ON GPU'" \
    || { echo "PLATFORM CHECK FAILED - aborting"; exit 1; }

echo "=== parallel hardening ==="
$PY -u parallel_hardening.py
