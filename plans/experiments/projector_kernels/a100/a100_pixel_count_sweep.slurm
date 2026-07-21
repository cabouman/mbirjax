#!/bin/bash
# Pixel-count sweep: the five operating points a real recon actually visits.
# Design in a100_pixel_count_sweep.py; parameters in its Config block.
#SBATCH -A bouman
#SBATCH -p a100-40gb
#SBATCH --constraint=N
#SBATCH -N 1
#SBATCH -n 20
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G
#SBATCH -t 04:00:00
#SBATCH -J a100_px
#SBATCH -o /scratch/gilbreth/buzzard/a100_tuning/logs/a100_px_%j.out
#SBATCH -e /scratch/gilbreth/buzzard/a100_tuning/logs/a100_px_%j.err
set -u
PY=$HOME/.conda/envs/mbirjax/bin/python
hostname; nvidia-smi --query-gpu=name --format=csv,noheader
A100_PX_GEOMETRY=parallel $PY -u /scratch/gilbreth/buzzard/a100_tuning/scripts/a100_pixel_count_sweep.py
echo "exit: $?"
