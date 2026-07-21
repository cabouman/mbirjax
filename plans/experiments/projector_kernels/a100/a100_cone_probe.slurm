#!/bin/bash
# Sizing probe for the A100 cone sweep: does the chosen cell fit one A100-40GB, what
# are its derived shapes, and does the harness machinery work on GPU?  Cheap (minutes)
# and deliberately separate from the long sweep, so the shape is settled before any
# GPU hours are committed to it.
#SBATCH -A bouman
#SBATCH -p a100-40gb
#SBATCH --constraint=N
#SBATCH -N 1
#SBATCH -n 20
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G
#SBATCH -t 00:40:00
#SBATCH -J a100_probe
#SBATCH -o /scratch/gilbreth/buzzard/a100_tuning/logs/a100_probe_%j.out
#SBATCH -e /scratch/gilbreth/buzzard/a100_tuning/logs/a100_probe_%j.err

set -u
PY=$HOME/.conda/envs/mbirjax/bin/python
HERE=/scratch/gilbreth/buzzard/a100_tuning/scripts

hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo
echo "=== probe: shapes + isolated back-projection peak ==="
A100_SWEEP_ROLE=probe $PY -u "$HERE/a100_cone_sweep.py"
echo "probe exit: $?"

echo
echo "=== machinery check: one cell + the value gate at shipped constants ==="
A100_SWEEP_ROLE=cell \
  A100_SWEEP_CELL='{"consts":{"CONE_LC":128,"CONE_NUM_WARPS":1,"CONE_FWD_NUM_WARPS":2,"FWD_SEGMENT_CAP":64,"VIEW_CHUNK":128},"op":"back","point":"subset"}' \
  $PY -u "$HERE/a100_cone_sweep.py"

A100_SWEEP_ROLE=cell \
  A100_SWEEP_CELL='{"consts":{"CONE_LC":256,"CONE_NUM_WARPS":1,"CONE_FWD_NUM_WARPS":2,"FWD_SEGMENT_CAP":64,"VIEW_CHUNK":128},"op":"back","point":"subset"}' \
  $PY -u "$HERE/a100_cone_sweep.py"

A100_SWEEP_ROLE=gate \
  A100_SWEEP_CELL='{"consts":{"CONE_LC":128,"CONE_NUM_WARPS":1,"CONE_FWD_NUM_WARPS":2,"FWD_SEGMENT_CAP":64,"VIEW_CHUNK":128}}' \
  $PY -u "$HERE/a100_cone_sweep.py"

echo
echo "Read the two CELL_RESULT lines: 'rebuilt' must be true on BOTH, and their"
echo "times must differ -- if CONE_LC 128 and 256 give identical times AND identical"
echo "derived lc, the constant was clamped and the cells are duplicates by construction."
