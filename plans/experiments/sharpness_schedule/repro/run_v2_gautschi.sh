#!/bin/bash
# Metric v2 pass over existing A1/A2 volumes (CPU-bound; compute node per the
# no-login-compute rule).  Submit from repro/:  sbatch run_v2_gautschi.sh
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 01:00:00
#SBATCH -J sharp_v2
#SBATCH -o /scratch/gautschi/buzzard/sharpness_schedule/logs/%x_%j.log

source ~/load_conda_cuda.sh
cd "$SLURM_SUBMIT_DIR"
~/venvs/sharpness_main/bin/python -u v2_zspectrum.py
