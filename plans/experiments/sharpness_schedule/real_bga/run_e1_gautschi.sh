#!/bin/bash
# E1: downsampled BGA baseline pair to 60 iterations on gautschi (1x H100).
# Submit from real_bga/:  sbatch run_e1_gautschi.sh
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 04:00:00
#SBATCH -J sharp_e1
#SBATCH -o /scratch/gautschi/buzzard/sharpness_schedule/logs/sharp_e1_%j.log

source ~/load_conda_cuda.sh
PY=~/venvs/sharpness_main/bin/python
cd "$SLURM_SUBMIT_DIR"

echo "=== platform check ==="
$PY -c "import mbirjax, jax; d = jax.devices(); print(d); print('library under test:', mbirjax.__file__); assert d[0].platform == 'gpu', 'NOT ON GPU'" \
    || { echo "PLATFORM CHECK FAILED - aborting"; exit 1; }

echo "=== e1 longtail (bga baseline pair, 60 iterations) ==="
$PY -u e1_longtail_bga.py
