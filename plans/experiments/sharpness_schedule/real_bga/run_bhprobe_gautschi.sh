#!/bin/bash
# Beam-hardening transfer-curve probe (padded + unpadded finals).
# Submit from real_bga/:  sbatch run_bhprobe_gautschi.sh
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 01:00:00
#SBATCH -J sharp_bhp
#SBATCH -o /scratch/gautschi/buzzard/sharpness_schedule/logs/sharp_bhp_%j.log

source ~/load_conda_cuda.sh
PY=~/venvs/sharpness_main/bin/python
cd "$SLURM_SUBMIT_DIR"

echo "=== platform check ==="
$PY -c "import mbirjax, jax; d = jax.devices(); print(d); print('library under test:', mbirjax.__file__); assert d[0].platform == 'gpu', 'NOT ON GPU'" \
    || { echo "PLATFORM CHECK FAILED - aborting"; exit 1; }

echo "=== bh transfer probe ==="
$PY -u bh_transfer_probe.py
