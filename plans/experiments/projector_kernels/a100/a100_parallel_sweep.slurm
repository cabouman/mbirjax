#!/bin/bash
# Step 2 of the A100 tuning campaign: the cone Pallas-constant sweep.
# Design, axes, and the anti-confound rules are documented in a100_cone_sweep.py;
# all parameters live in that file's Config block (no command-line arguments).
# Run the PROBE first (a100_cone_probe.slurm) -- it settles the cell shape and proves
# the machinery before these GPU hours are spent.
#SBATCH -A bouman
#SBATCH -p a100-40gb
#SBATCH --constraint=N
#SBATCH -N 1
#SBATCH -n 20
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G
#SBATCH -t 08:00:00
#SBATCH -J a100_par
#SBATCH -o /scratch/gilbreth/buzzard/a100_tuning/logs/a100_par_%j.out
#SBATCH -e /scratch/gilbreth/buzzard/a100_tuning/logs/a100_par_%j.err

set -u
PY=$HOME/.conda/envs/mbirjax/bin/python
HERE=/scratch/gilbreth/buzzard/a100_tuning/scripts

hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
git -C "$HOME/PycharmProjects/mbirjax" log --oneline -1

# python -u: block buffering makes long sweep logs non-chronological (lessons.md).
A100_SWEEP_GEOMETRY=parallel $PY -u "$HERE/a100_cone_sweep.py"
echo "sweep exit: $?"
