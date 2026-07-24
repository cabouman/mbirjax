#!/bin/bash -l
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -n 14
#SBATCH -t 04:00:00
#SBATCH -J mbirjax-p3b
#SBATCH --mail-user=buzzard@purdue.edu
#SBATCH --mail-type=FAIL
#SBATCH -o slurm-mbirjax-p3b-%j.log
#
# p3b_bga_axial.slurm -- step-A axial-extension validation on the Purdue BGA Normal scan
# (p3b_bga_axial_check.py in this directory; parameters at the top of that file, taken
# from mbirjax_applications/zeiss/experiment_zeiss.py 'Purdue BGA Normal scan').
# Submit from the staging directory:  sbatch p3b_bga_axial.slurm
set -uo pipefail

PREAMBLE_FILE="${PREAMBLE_FILE:-$HOME/load_conda_cuda.sh}"
if [ -f "$PREAMBLE_FILE" ]; then
  # shellcheck disable=SC1090
  source "$PREAMBLE_FILE"
fi
conda activate mbirjax

cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"
echo "=== p3b step-A axial-extension validation: Purdue BGA Normal ==="
echo "node: $(hostname); date: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -u p3b_bga_axial_check.py
