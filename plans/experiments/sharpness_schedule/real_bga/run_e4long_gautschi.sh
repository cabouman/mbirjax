#!/bin/bash
# Padded long pair on the real scan (even-delta scale, 60 iterations).
# Submit from real_bga/:  sbatch run_e4long_gautschi.sh
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 04:00:00
#SBATCH -J sharp_e4l
#SBATCH -o /scratch/gautschi/buzzard/sharpness_schedule/logs/sharp_e4l_%j.log

source ~/load_conda_cuda.sh
PY=~/venvs/sharpness_main/bin/python
cd "$SLURM_SUBMIT_DIR"

echo "=== platform check ==="
$PY -c "import mbirjax, jax; d = jax.devices(); print(d); print('library under test:', mbirjax.__file__); assert d[0].platform == 'gpu', 'NOT ON GPU'" \
    || { echo "PLATFORM CHECK FAILED - aborting"; exit 1; }

echo "=== e4 padded long pair (60 iterations) ==="
$PY -u e4_pad_long_bga.py
