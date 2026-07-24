#!/bin/bash
# Step 0 of the A100 tuning campaign: does the shipped pallas path actually COMPILE
# and stay VALUE-CORRECT on sm80?  The A100 allowlist entry (_ARCH_ALLOWLIST in
# mbirjax/_pallas_kernels.py) currently ships with no sm80 gate evidence -- every
# measured number in plans/projector_kernels/ is H100.
#
# THE TRAP THIS JOB EXISTS TO AVOID: tests/test_pallas_kernels.py picks its mode with
#   _interpret() = not _pallas_kernels.is_available()
# so on a node where availability() says NO the whole suite silently runs in pallas
# INTERPRET mode (a CPU-semantics reference implementation) and passes -- proving
# nothing about the compiled Triton kernels.  A green suite is therefore only
# meaningful together with availability() == True, which is asserted BEFORE pytest and
# aborts the job if it fails.
#
# Run:  sbatch a100_gate.slurm     (results in $OUT/logs)
# --constraint=N pins the A100-SXM4 class (see a100_baseline_ab.slurm for why the
# partition's two classes must not be mixed).  Correctness is class-independent, but
# pinning keeps every job in this campaign on identical hardware.
#SBATCH -A bouman
#SBATCH -p a100-40gb
#SBATCH --constraint=N
#SBATCH -N 1
#SBATCH -n 20
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G
#SBATCH -t 00:45:00
#SBATCH -J a100_gate
#SBATCH -o /scratch/gilbreth/buzzard/a100_tuning/logs/a100_gate_%j.out
#SBATCH -e /scratch/gilbreth/buzzard/a100_tuning/logs/a100_gate_%j.err

set -u
PY=$HOME/.conda/envs/mbirjax/bin/python
REPO=$HOME/PycharmProjects/mbirjax

echo "=== node / GPU ==="
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv

echo
echo "=== checkout under test (expected: unmodified, 2 commits behind origin) ==="
git -C "$REPO" log --oneline -1
git -C "$REPO" status --short | head -20

echo
echo "=== jax / device / pallas availability + the shipped tuning constants ==="
$PY -u -c "
import mbirjax, jax
from mbirjax import _pallas_kernels as pk
d = jax.devices()[0]
print('jax          ', jax.__version__)
print('device_kind  ', repr(d.device_kind), '| platform', d.platform, '| n_devices', jax.device_count())
ok, why = pk.availability()
print('availability ', ok, '|', why)
print('constants    : ROW_CHUNK=%s NUM_WARPS=%s BACK_VIEW_CHUNK_CAP=%s' % (
    pk.ROW_CHUNK, pk.NUM_WARPS, pk.BACK_VIEW_CHUNK_CAP))
print('             : FWD_SEGMENT_CAP=%s FWD_NUM_WARPS=%s' % (
    pk.FWD_SEGMENT_CAP, pk.FWD_NUM_WARPS))
print('             : CONE_LC=%s CONE_NUM_WARPS=%s CONE_FWD_NUM_WARPS=%s' % (
    pk.CONE_LC, pk.CONE_NUM_WARPS, pk.CONE_FWD_NUM_WARPS))
raise SystemExit(0 if ok else 3)
"
if [ $? -ne 0 ]; then
    echo "FATAL: pallas is NOT available on this node."
    echo "       The gates below would silently run in INTERPRET mode and prove nothing"
    echo "       about the compiled sm80 kernels.  Aborting instead of reporting a"
    echo "       meaningless green suite."
    exit 3
fi

echo
echo "=== step 0a: the pallas correctness gates, COMPILED on A100 ==="
echo "    (25 tests: parallel back/fwd, cone back/fwd, n=1 + per-owner band drivers,"
echo "     both coeff_powers, and the adjoint identities)"
cd "$REPO" || exit 1
$PY -u -m pytest tests/test_pallas_kernels.py -v -p no:cacheprovider
GATE_A=$?
echo "pallas gate exit: $GATE_A"

echo
echo "=== step 0b: the projector suites that exercise the same paths end to end ==="
$PY -u -m pytest tests/geometries/test_projectors.py tests/geometries/test_banded_projectors.py \
    tests/test_tile_policy.py -q -p no:cacheprovider
GATE_B=$?
echo "projector suite exit: $GATE_B"

echo
echo "=== SUMMARY ==="
echo "pallas gates    : $GATE_A  (0 = pass)"
echo "projector suites: $GATE_B  (0 = pass)"
exit $(( GATE_A != 0 || GATE_B != 0 ))
