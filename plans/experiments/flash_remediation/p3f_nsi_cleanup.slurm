#!/bin/bash -l
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -n 14
#SBATCH -t 02:00:00
#SBATCH -J mbirjax-p3f
#SBATCH --mail-user=buzzard@purdue.edu
#SBATCH --mail-type=FAIL
#SBATCH -o slurm-mbirjax-p3f-%j.log
#
# p3f_nsi_cleanup.slurm -- NSI cleanup validation on Lilly (shape, ds4 recon, ds8 split)
# (p3f_nsi_cleanup_check.py in this directory; parameters at the top of that file).
# Submit from the staging directory:  sbatch p3f_nsi_cleanup.slurm
set -uo pipefail

PREAMBLE_FILE="${PREAMBLE_FILE:-$HOME/load_conda_cuda.sh}"
if [ -f "$PREAMBLE_FILE" ]; then
  # shellcheck disable=SC1090
  source "$PREAMBLE_FILE"
fi
conda activate mbirjax

cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"
echo "=== p3f NSI auto-geometry cleanup validation: Lilly ds4+ds8 ==="
echo "node: $(hostname); date: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -u p3f_nsi_cleanup_check.py
