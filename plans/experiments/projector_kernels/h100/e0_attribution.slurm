#!/bin/bash
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 02:00:00
#SBATCH -J headroom-e0
#SBATCH -o /home/buzzard/headroom/logs/e0-%j.log
#
# E0 attribution (gpu_headroom_plan.md section 6): HLO structure dump + nsys sort share.
# Runs from the mbirjax_headroom worktree with the DEDICATED env (editable install points
# at the worktree -- the code-under-test selection mechanism; lessons.md section 5).
#
#     sbatch plans/experiments/projector_kernels/e0_attribution.slurm

set -u
source ~/load_conda_cuda.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mbirjax_headroom
cd ~/PycharmProjects/mbirjax_headroom
RES=~/headroom/results
mkdir -p "$RES"

echo "=== provenance ==="
git log --oneline -1
python -c "import mbirjax, jax; print('mbirjax:', mbirjax.__file__); print('jax:', jax.__version__)"
nvidia-smi -L

echo "=== E0a: HLO structure dump ==="
python -u plans/experiments/projector_kernels/e0_hlo_dump.py

echo "=== E0b: nsys sort share (forward) ==="
if command -v nsys >/dev/null 2>&1; then
    rm -f "$RES/e0_fwd_nsys.nsys-rep"
    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
        -o "$RES/e0_fwd_nsys" \
        python -u plans/experiments/projector_kernels/e0_sort_share.py
    echo "--- kernel summary (radix/cub rows = the sort) ---"
    nsys stats --report cuda_gpu_kern_sum "$RES/e0_fwd_nsys.nsys-rep" | head -80
else
    echo "nsys not on PATH -- sort share deferred (check module avail nsight-systems)"
fi
echo "=== E0 done ==="
