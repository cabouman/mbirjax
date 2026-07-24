#!/bin/bash -l
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -n 14
#SBATCH -t 04:00:00
#SBATCH -J mbirjax-p3a
#SBATCH --mail-user=buzzard@purdue.edu
#SBATCH --mail-type=FAIL
#SBATCH -o slurm-mbirjax-p3a-%j.log
#
# p3a_sic_axial.slurm -- flash-remediation step-A validation on the real SiC scan
# (p3a_sic_axial_check.py in this directory; run parameters at the top of that file).
# ONE GPU: the subsampled sinograms fit a single device, and a fixed device count keeps
# the two variants' wall times comparable.  Submit from the staging directory:
#
#     sbatch p3a_sic_axial.slurm
#
# The worker asserts the live mbirjax commit (EXPECTED_COMMIT in the script), so a stale
# checkout fails loudly instead of measuring the wrong library.
set -uo pipefail

PREAMBLE_FILE="${PREAMBLE_FILE:-$HOME/load_conda_cuda.sh}"
if [ -f "$PREAMBLE_FILE" ]; then
  # shellcheck disable=SC1090
  source "$PREAMBLE_FILE"
fi
conda activate mbirjax

cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"
echo "=== p3a step-A axial-extension validation ==="
echo "node: $(hostname); date: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -u p3a_sic_axial_check.py
