#!/bin/bash
# Step 1 of the A100 tuning campaign: the controlled pallas-on/off A/B of the BGA cone
# recon (time AND peak memory).  See a100_baseline_ab.py for the design and the
# measurement rules; all run parameters live in that file's Config block.
#
# Run:  sbatch a100_baseline_ab.slurm
# --constraint=N pins the A100-SXM4 nodes.  The a100-40gb partition MIXES two hardware
# classes -- SXM4 (features N/n + NVLINK, 4 GPU/node, 400 W) and PCIe (features G/g,
# 2 GPU/node, 250 W) -- whose sustained clocks differ, so wall times are NOT comparable
# across them.  Every timing job in this campaign pins the SAME class (Greg chose SXM4:
# higher power limit, NVLink, and 4 GPUs/node for the later n=2 band spot-check).
#SBATCH -A bouman
#SBATCH -p a100-40gb
#SBATCH --constraint=N
#SBATCH -N 1
#SBATCH -n 20
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G
#SBATCH -t 03:00:00
#SBATCH -J a100_ab
#SBATCH -o /scratch/gilbreth/buzzard/a100_tuning/logs/a100_ab_%j.out
#SBATCH -e /scratch/gilbreth/buzzard/a100_tuning/logs/a100_ab_%j.err

set -u
PY=$HOME/.conda/envs/mbirjax/bin/python
HERE=/scratch/gilbreth/buzzard/a100_tuning/scripts

echo "=== node / GPU ==="
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo
echo "=== checkout under test ==="
git -C "$HOME/PycharmProjects/mbirjax" log --oneline -1

echo
# python -u: block buffering makes sweep logs non-chronological (lessons.md).
$PY -u "$HERE/a100_baseline_ab.py"
echo "exit: $?"
