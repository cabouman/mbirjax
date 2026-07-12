#!/bin/bash -l
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -n 14
#SBATCH -t 04:00:00
#SBATCH -J mbirjax-p3c
#SBATCH --mail-user=buzzard@purdue.edu
#SBATCH --mail-type=FAIL
#SBATCH -o slurm-mbirjax-p3c-%j.log
#
# p3c_lilly_split.slurm -- step-B validation of the shipped split_sino_recon on the
# Lilly scan at the P2c regimes (p3c_lilly_split_check.py in this directory; parameters
# at the top of that file).  Submit from the staging directory:
#
#     sbatch p3c_lilly_split.slurm
set -uo pipefail

PREAMBLE_FILE="${PREAMBLE_FILE:-$HOME/load_conda_cuda.sh}"
if [ -f "$PREAMBLE_FILE" ]; then
  # shellcheck disable=SC1090
  source "$PREAMBLE_FILE"
fi
conda activate mbirjax

cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"
echo "=== p3c step-B split-seam validation: Lilly 4x/8x ==="
echo "node: $(hostname); date: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -u p3c_lilly_split_check.py
