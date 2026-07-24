#!/bin/bash
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 06:00:00
#SBATCH -J headroom-e1
#SBATCH -o /home/buzzard/headroom/logs/e1-%j.log
#
# E1 Amdahl gate (gpu_headroom_plan.md section 6): pixel-count sweep + the
# production-granularity VCD device-share traces (THE gating number for the (a) track).
# Runs from the mbirjax_headroom worktree with the DEDICATED env.
#
#     sbatch plans/experiments/projector_kernels/e1_amdahl.slurm

set -u
source ~/load_conda_cuda.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mbirjax_headroom
cd ~/PycharmProjects/mbirjax_headroom

echo "=== provenance ==="
git log --oneline -1
python -c "import mbirjax, jax; print('mbirjax:', mbirjax.__file__); print('jax:', jax.__version__)"
nvidia-smi -L

echo "=== E1a: pixel-count sweep (fine-granularity per-call costs) ==="
python -u plans/experiments/projector_kernels/e1_pixel_sweep.py

echo "=== E1b: VCD device-share traces (window difference, granularity 128) ==="
python -u plans/experiments/projector_kernels/e1_vcd_trace.py

echo "=== E1 done ==="
