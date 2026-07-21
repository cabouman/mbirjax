#!/bin/bash
# Step 3: does the view-chunk memory saving compose to a full recon?
# Design and rationale in a100_viewchunk_recon_ab.py; parameters in its Config block.
#SBATCH -A bouman
#SBATCH -p a100-40gb
#SBATCH --constraint=N
#SBATCH -N 1
#SBATCH -n 20
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G
#SBATCH -t 04:00:00
#SBATCH -J a100_vcpar
#SBATCH -o /scratch/gilbreth/buzzard/a100_tuning/logs/a100_vcpar_%j.out
#SBATCH -e /scratch/gilbreth/buzzard/a100_tuning/logs/a100_vcpar_%j.err
set -u
PY=$HOME/.conda/envs/mbirjax/bin/python
hostname; nvidia-smi --query-gpu=name --format=csv,noheader
A100_VC_GEOMETRY=parallel $PY -u /scratch/gilbreth/buzzard/a100_tuning/scripts/a100_viewchunk_recon_ab.py
echo "exit: $?"
