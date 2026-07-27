#!/bin/bash
# E4: laterally padded downsampled BGA (scale 1.5, 17 iterations) on gautschi.
# Submit from real_bga/:  sbatch run_e4_bga_gautschi.sh
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 03:00:00
#SBATCH -J sharp_e4
#SBATCH -o /scratch/gautschi/buzzard/sharpness_schedule/logs/sharp_e4_%j.log

source ~/load_conda_cuda.sh
PY=~/venvs/sharpness_main/bin/python
cd "$SLURM_SUBMIT_DIR"

echo "=== platform check ==="
$PY -c "import mbirjax, jax; d = jax.devices(); print(d); print('library under test:', mbirjax.__file__); assert d[0].platform == 'gpu', 'NOT ON GPU'" \
    || { echo "PLATFORM CHECK FAILED - aborting"; exit 1; }

echo "=== e4 padded bga (scale 1.5, 17 iterations) ==="
$PY -u e4_pad_bga.py
