#!/bin/bash -l
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -n 14
#SBATCH -t 06:00:00
#SBATCH -J mbirjax-p3e
#SBATCH --mail-user=buzzard@purdue.edu
#SBATCH --mail-type=FAIL
#SBATCH -o slurm-mbirjax-p3e-%j.log
#
# p3e_lateral_pad.slurm -- lateral+axial padding comparison on BGA and Lilly
# (p3e_lateral_pad_check.py in this directory; parameters at the top of that file).
# Submit from the staging directory:  sbatch p3e_lateral_pad.slurm
set -uo pipefail

PREAMBLE_FILE="${PREAMBLE_FILE:-$HOME/load_conda_cuda.sh}"
if [ -f "$PREAMBLE_FILE" ]; then
  # shellcheck disable=SC1090
  source "$PREAMBLE_FILE"
fi
conda activate mbirjax

cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"
echo "=== p3e lateral+axial padding: Lilly s=1.25, BGA s=1.5/2.0 ==="
echo "node: $(hostname); date: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -u p3e_lateral_pad_check.py
