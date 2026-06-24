#!/bin/bash
# run_sweep.sh -- drive the cone-vs-parallel multi-GPU deadlock repro.
#
# Runs each (geometry x size x device-count) configuration in its OWN process, wrapped in
# `timeout`, so a deadlocked config is killed and recorded while the rest keep going.  Device
# count is set by restricting CUDA_VISIBLE_DEVICES (MBIRJAX then auto-shards across exactly
# those GPUs -- the same path as a real run).  HLO is dumped for the device counts in
# DUMP_DEVICE_COUNTS so the compiled collectives can be inspected (all-reduce / collective-
# permute / all-gather).
#
# Run it after sourcing the cluster preamble + activating the mbirjax env (cone_deadlock.slurm
# does this), or directly in an interactive GPU session.
#
# Knobs (override via environment):
#   GEOMETRIES        geometries to test           (default: "parallel cone")
#   SIZES             cubic problem sizes           (default: "256 512")
#   DEVICE_COUNTS     device counts to sweep        (default: "1 2 4 8", capped to GPUs present)
#   ITERATIONS        VCD max_iterations            (default: 3)
#   PER_TIMEOUT       seconds per config before kill (default: 420)
#   DUMP_DEVICE_COUNTS device counts to dump HLO for (default: "8")
#   OUTDIR            output root                   (default: ./cone_deadlock_out_<timestamp>)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO="$HERE/repro_recon.py"

GEOMETRIES="${GEOMETRIES:-parallel cone}"
SIZES="${SIZES:-256 512}"
DEVICE_COUNTS="${DEVICE_COUNTS:-1 2 4 8}"
ITERATIONS="${ITERATIONS:-3}"
PER_TIMEOUT="${PER_TIMEOUT:-420}"
DUMP_DEVICE_COUNTS="${DUMP_DEVICE_COUNTS:-8}"
OUTDIR="${OUTDIR:-$PWD/cone_deadlock_out_$(date +%Y%m%d_%H%M%S)}"

# How many GPUs are actually visible to this job?  Cap DEVICE_COUNTS to that.
if command -v nvidia-smi >/dev/null 2>&1; then
  NGPU="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
else
  NGPU=0
fi
[ "$NGPU" -ge 1 ] 2>/dev/null || NGPU=1

# Per-config kill on hang (GNU coreutils `timeout`; present on the cluster).  Without it a
# deadlocked config would hang until the SLURM walltime instead of being isolated.
if command -v timeout >/dev/null 2>&1; then
  TIMEOUT_CMD=(timeout -k 20 "${PER_TIMEOUT}s")
else
  echo "WARNING: 'timeout' not found -- a hung config will block until the job walltime."
  TIMEOUT_CMD=()
fi

mkdir -p "$OUTDIR/logs" "$OUTDIR/hlo"
SUMMARY="$OUTDIR/summary.txt"
: > "$SUMMARY"

echo "=== cone/parallel multi-GPU deadlock sweep ===" | tee -a "$SUMMARY"
echo "host=$(hostname)  visible_gpus=$NGPU  per_config_timeout=${PER_TIMEOUT}s  iters=$ITERATIONS" | tee -a "$SUMMARY"
echo "geometries=[$GEOMETRIES]  sizes=[$SIZES]  device_counts=[$DEVICE_COUNTS]  hlo_dump_for=[$DUMP_DEVICE_COUNTS]" | tee -a "$SUMMARY"
echo "output -> $OUTDIR" | tee -a "$SUMMARY"
echo | tee -a "$SUMMARY"
printf "%-9s %-6s %-7s %-9s %-9s %s\n" geometry size ndev result seconds logfile | tee -a "$SUMMARY"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

for g in $GEOMETRIES; do
  for s in $SIZES; do
    for n in $DEVICE_COUNTS; do
      [ "$n" -le "$NGPU" ] || continue                      # skip counts we don't have GPUs for
      tag="${g}_s${s}_n${n}"
      logf="$OUTDIR/logs/${tag}.log"
      vis="$(seq -s, 0 $((n - 1)))"                          # 0,1,...,n-1

      # HLO dump only for selected device counts (the files are large/numerous).
      dump=""
      for dn in $DUMP_DEVICE_COUNTS; do [ "$n" = "$dn" ] && dump="yes"; done
      if [ -n "$dump" ]; then
        hlodir="$OUTDIR/hlo/${tag}"
        mkdir -p "$hlodir"
        export XLA_FLAGS="--xla_dump_to=$hlodir --xla_dump_hlo_as_text"
      else
        unset XLA_FLAGS
      fi

      start=$(date +%s)
      CUDA_VISIBLE_DEVICES="$vis" "${TIMEOUT_CMD[@]}" \
        python "$REPRO" --geometry "$g" --size "$s" --iterations "$ITERATIONS" \
        > "$logf" 2>&1
      rc=$?
      dur=$(( $(date +%s) - start ))

      case "$rc" in
        0)        result="OK" ;;
        124|137)  result="TIMEOUT" ;;                        # killed by `timeout` -> likely deadlock/hang
        *)        result="ERR($rc)" ;;
      esac
      printf "%-9s %-6s %-7s %-9s %-9s %s\n" "$g" "$s" "$n" "$result" "$dur" "$logf" | tee -a "$SUMMARY"
    done
  done
done

echo | tee -a "$SUMMARY"
echo "Done.  Inspect:" | tee -a "$SUMMARY"
echo "  logs:    $OUTDIR/logs/   (last marker before a TIMEOUT shows the phase it hung in)" | tee -a "$SUMMARY"
echo "  HLO:     grep -lE 'all-reduce|collective-permute|all-gather|reduce-scatter' $OUTDIR/hlo/*/*.txt" | tee -a "$SUMMARY"
echo "  compare cone-vs-parallel HLO collectives to see what the cone path compiles that parallel does not." | tee -a "$SUMMARY"
