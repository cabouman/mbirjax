#!/bin/bash
# B0 noise calibration (from existing Phase A volumes) + pedagogy-figure regen
# (adds the two-seed damping-comparison figure).  Submit from schedule/:
#   sbatch run_b0_gautschi.sh
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 01:00:00
#SBATCH -J sharp_b0
#SBATCH -o /scratch/gautschi/buzzard/sharpness_schedule/logs/%x_%j.log

source ~/load_conda_cuda.sh
PY=~/venvs/sharpness_main/bin/python
cd "$SLURM_SUBMIT_DIR"
$PY -u ../findings/fig_pedagogy.py
$PY -u b0_calibration.py
