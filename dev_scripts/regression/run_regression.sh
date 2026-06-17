#!/usr/bin/env bash
# run_regression.sh — fire-on-change nightly regression driver (FRESH-CLONE model).
#
# Phase 1 (bootstrap, runs from wherever cron/launchd/scrontab invokes it): source the node preamble
#   (proxy + modules), update (or clone) the PERSISTENT metrics clone at $WORK_DIR/metrics, and re-exec
#   ITS copy of this script — so harness/engine/wrapper updates on the remote are always used.
# Phase 2 (work, runs from the persistent metrics clone): for each tracked branch whose remote head moved
#   since last measured (git ls-remote vs metrics state/), make a SHALLOW single-branch clone of the
#   library tip, pip install -e (+platform extras), run that branch's tests + the perf engine, record
#   results + the measured SHA into the metrics clone, and push (non-fatal).
#
# No per-node paths are baked in — only URLs + $WORK_DIR (under $HOME or scratch).  Config: regression.env.
# Exits non-zero ONLY on a hard-gate perf regression (so the cron/slurm mail is a real alert); setup/
# transport hiccups are WARNs.
set -uo pipefail
# Keep an INTERACTIVE terminal open on a nonzero exit so the error stays visible (some terminals
# close the window when a command exits nonzero).  A tty-less run (cron/launchd/scrontab/slurm) has
# no stdin tty, so it skips the pause and exits with the real code — the alert path is preserved.
if [ -t 0 ]; then
  trap '_ec=$?; [ "$_ec" -ne 0 ] && { echo; echo ">>> $(basename "$0") exited with status $_ec — press Enter to close."; read -r _ || true; }' EXIT
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/regression.env"
log() { echo "[$(date '+%F %T')] $*"; }

[ "${ENABLED:-0}" = "1" ] || { log "ENABLED=0 — nothing to do."; exit 0; }

# Node preamble (cluster: `module load` conda/cuda + export HTTPS_PROXY; empty on the Mac).  SOURCED
# in BOTH phases: phase 1 needs the proxy to git-clone from github, phase 2 needs conda/cuda — and
# conda's shell function does NOT survive the phase-1->2 exec, so phase 2 must re-source, not inherit.
# It is SOURCED, so PREAMBLE_FILE must be source-safe: module loads + exports only, NO `exit`.
if [ -n "${PREAMBLE_FILE:-}" ] && [ -f "$PREAMBLE_FILE" ]; then
  # shellcheck disable=SC1090
  source "$PREAMBLE_FILE"
  set -uo pipefail   # re-assert in case the preamble changed shell options (e.g. turned on set -e)
fi

# ── Phase 1: bootstrap — lock, update/clone the persistent metrics clone, re-exec ─────────────
if [ -z "${REG_FRESH:-}" ]; then
  mkdir -p "$WORK_DIR"
  # Portable single-instance lock (macOS has no flock): mkdir is atomic everywhere.  Held as a dir
  # on disk across the exec; phase 2 sets the EXIT trap that removes it.
  if ! mkdir "$WORK_DIR/.lock.d" 2>/dev/null; then
    log "another run holds the lock ($WORK_DIR/.lock.d) — exiting."; exit 0
  fi
  # PERSISTENT metrics clone (NOT throwaway): reuse it across runs so a FAILED push never loses a
  # run's results — they stay committed locally and the next run's rebase-retry pushes them (and you
  # can `git -C "$WORK_DIR/metrics" push` by hand).  Reuse = fetch + rebase (gets remote changes while
  # keeping any unpushed local commits, public repo so no creds needed for read); re-clone only if the
  # existing clone is missing or unusable.  NOTE: this entrypoint must be a SEPARATE checkout from
  # $WORK_DIR/metrics (the cron/standing wrapper), so updating it here never modifies the running script.
  MC="$WORK_DIR/metrics"
  if [ -d "$MC/.git" ]; then
    log "updating metrics clone (fetch + rebase) -> $MC"
    if ! { git -C "$MC" fetch -q origin && git -C "$MC" pull -q --rebase --autostash; }; then
      log "metrics clone unusable — re-cloning fresh."
      rm -rf "$MC"
      git clone --quiet "$METRICS_URL" "$MC" || { log "FATAL: clone metrics failed."; rmdir "$WORK_DIR/.lock.d" 2>/dev/null; exit 2; }
    fi
  else
    log "cloning metrics -> $MC"
    git clone --quiet "$METRICS_URL" "$MC" || { log "FATAL: clone metrics failed."; rmdir "$WORK_DIR/.lock.d" 2>/dev/null; exit 2; }
  fi
  # Re-exec the updated copy (picks up remote wrapper/engine changes).  The env (proxy/modules from the
  # preamble, the lock dir on disk) carries through exec.
  exec env REG_FRESH=1 "$MC/tooling/regression/run_regression.sh"
fi

# ── Phase 2: work — running from the FRESH metrics clone ──────────────────────────────────────
trap 'rm -rf "$WORK_DIR/.lock.d"' EXIT
METRICS_REPO="$(cd "$HERE/../.." && pwd)"        # = $WORK_DIR/metrics
HARNESS_DIR="$METRICS_REPO/tooling"

# Conda env (DEDICATED — the per-branch editable installs churn it).  conda must be on PATH here
# (Mac: your shell PATH; cluster: from PREAMBLE_FILE's `module load conda`).  We (re)source conda.sh
# to define the `conda activate` shell function, which does not survive the phase-1->2 exec.
if ! command -v conda >/dev/null 2>&1; then
  log "FATAL: 'conda' not found.  On the cluster, set PREAMBLE_FILE in regression.env to a script"
  log "       that loads conda+cuda (e.g. PREAMBLE_FILE=\"\$HOME/load_conda_cuda.sh\")."
  exit 2
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
# Auto-create the DEDICATED env if missing, so a fresh node self-bootstraps (the wrapper installs all
# deps per-run, so a bare python env is enough).  One-time per node; subsequent runs just activate.
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  log "conda env '$CONDA_ENV' not found — creating it (one-time on this node)."
  conda create -y -q -n "$CONDA_ENV" "python=${CONDA_PYTHON:-3.11}" \
    || { log "FATAL: could not create conda env '$CONDA_ENV'."; exit 2; }
fi
conda activate "$CONDA_ENV" || { log "FATAL: conda activate '$CONDA_ENV' failed."; exit 2; }

# Harness's own deps (scaling_common imports matplotlib/ruamel at module level) — idempotent.
if [ -n "${HARNESS_DEPS:-}" ]; then
  # shellcheck disable=SC2086
  pip install --quiet $HARNESS_DEPS || log "WARN: harness deps install failed (engine may not import)."
fi

# Platform (no jax import in the wrapper — just the GPU presence signal).
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  PLAT="gpu"; EXTRAS="$INSTALL_EXTRAS_gpu"
else
  PLAT="cpu"; EXTRAS="$INSTALL_EXTRAS_cpu"
fi
RES="$METRICS_REPO/results/$PLAT"; STATE="$METRICS_REPO/state/$PLAT"
mkdir -p "$RES" "$STATE"
log "platform=$PLAT extras=[$EXTRAS] env=$CONDA_ENV metrics=$METRICS_REPO"

# Credential for unattended push (cluster), scoped to this repo only.  TOKEN_FILE must be a git
# credential-STORE file (chmod 600), i.e. ONE line:  https://<user>:<PAT>@github.com  (not the bare
# token).  Never prompt interactively: GIT_TERMINAL_PROMPT=0 makes a missing/invalid credential fail
# fast (a WARN on push) instead of hanging an unattended job on a password prompt.
export GIT_TERMINAL_PROMPT=0
if [ -n "${TOKEN_FILE:-}" ] && [ -f "$TOKEN_FILE" ]; then
  git -C "$METRICS_REPO" config credential.helper "store --file=$TOKEN_FILE"
fi

# ── Change detection via ls-remote (don't clone mbirjax unless something moved) ────────────────
CHANGED_BR=(); CHANGED_SHA=()
for BR in "${TRACKED_BRANCHES[@]}"; do
  SHA="$(git ls-remote "$MBIRJAX_URL" "refs/heads/$BR" 2>/dev/null | awk '{print $1}')"
  [ -n "$SHA" ] || { log "skip $BR: not found on remote."; continue; }
  SLUG="${BR//\//_}"
  LAST="$(cat "$STATE/$SLUG" 2>/dev/null || true)"
  if [ "$SHA" = "$LAST" ]; then
    log "$BR @ ${SHA:0:8}: unchanged — skip."
  else
    [ -n "$LAST" ] && WAS="${LAST:0:8}" || WAS="none"
    log "$BR @ ${SHA:0:8}: CHANGED (was $WAS)."
    CHANGED_BR+=("$BR"); CHANGED_SHA+=("$SHA")
  fi
done
[ "${#CHANGED_BR[@]}" -gt 0 ] || { log "no tracked branch changed — done."; exit 0; }

# Per changed branch: a SHALLOW, SINGLE-BRANCH clone of just the branch tip (no history, no other
# branches) straight into the work dir — small + fast, and (unlike a pip-install-from-git, which
# clones internally anyway) it still carries tests/ + dev_scripts/ for the test step and a .git for
# provenance.  The big experiments/ tree is gitignored, so it is never cloned.
DATE="$(date '+%Y%m%d')"
GATE_FAIL=0
for i in "${!CHANGED_BR[@]}"; do
  BR="${CHANGED_BR[$i]}"; SLUG="${BR//\//_}"
  WT="$WORK_DIR/lib_$SLUG"; rm -rf "$WT"
  log "$BR: shallow-cloning the library tip -> $WT"
  if ! git clone --quiet --depth 1 --branch "$BR" --single-branch "$MBIRJAX_URL" "$WT"; then
    log "ERROR $BR: shallow clone failed — skip."; continue
  fi
  SHA="$(git -C "$WT" rev-parse HEAD)"   # the tip we actually got; recorded as state below

  log "$BR: installing library [$EXTRAS] into $CONDA_ENV (first time pulls jax — can be slow)..."
  if ! pip install -e "$WT[$EXTRAS]" >"$WT/.install.log" 2>&1; then
    log "ERROR $BR: pip install -e '$WT[$EXTRAS]' failed (see $WT/.install.log) — skip."
    rm -rf "$WT"; continue
  fi

  # REG_SMOKE = isolated plumbing test: exercise the whole flow with a TOY 1-cell engine into a TEMP
  # dir, and skip tests / commit / push / state below — so it never touches real data.  Use it to
  # verify the wrapper (persistence, clone, install, engine) in ~1-2 min without a full measurement.
  if [ "${REG_SMOKE:-0}" = "1" ]; then
    OUT="$(mktemp -d)"; log "$BR: REG_SMOKE — toy output to $OUT (skipping tests / commit / push / state)."
  else
    OUT="$RES/$SLUG"
  fi
  mkdir -p "$OUT"

  # Tests: reuse the branch's OWN runner (your -n 10 tuning + conftest knobs); NON-FATAL (logged,
  # not gated — per-branch test diffing is a later increment; the perf engine is the alert path).
  # `tee` shows progress live (interactive) AND records to tests_*.txt; on an unattended run the
  # stdout copy just lands in the cron/slurm log.
  if [ "${RUN_TESTS:-0}" = "1" ] && [ "${REG_SMOKE:-0}" != "1" ]; then
    TLOG="$OUT/tests_${PLAT}_${DATE}.txt"
    log "$BR: running tests (MBIRJAX_NUM_CPU_DEVICES=$TEST_CPU_DEVICES) -> $(basename "$TLOG") ..."
    if [ -f "$WT/dev_scripts/run_tests.sh" ]; then
      # run_tests.sh uses a path RELATIVE to dev_scripts/ (`python -m pytest -n 10 ../tests`), so it
      # MUST be invoked from there or it collects 0 tests (../tests would resolve outside the clone).
      ( cd "$WT/dev_scripts" && MBIRJAX_NUM_CPU_DEVICES="$TEST_CPU_DEVICES" bash run_tests.sh ) 2>&1 | tee "$TLOG"
    else
      ( cd "$WT" && MBIRJAX_NUM_CPU_DEVICES="$TEST_CPU_DEVICES" python -m pytest tests -q -n 10 ) 2>&1 | tee "$TLOG"
    fi
    [ "${PIPESTATUS[0]}" -eq 0 ] || log "$BR: tests reported failures (non-fatal; see $(basename "$TLOG"))."
    log "$BR: tests done."
  fi

  # Perf engine (fixed harness; lib_root=$WT selects the library + provenance; golden + vs-main
  # baseline come from the metrics repo's golden/).  run_nightly.py prints to stdout, so the engine
  # output shows live on a manual run (and lands in the cron/slurm log unattended).
  log "$BR: running perf engine (output follows)..."
  if REG_LIB_ROOT="$WT" REG_OUT_DIR="$OUT" REG_DATE="$DATE" REG_GATE=1 REG_RUN_TAG="$BR" \
       REG_GOLDEN_DIR="$METRICS_REPO/golden" \
       python "$HARNESS_DIR/scaling_tests/run_nightly.py"; then
    log "$BR: engine ok."
  else
    GATE_FAIL=1; log "$BR: GATE FAIL (perf regression) — see $OUT."
  fi

  # Record the measured commit LAST (a crash mid-run re-measures next time).  Skipped in smoke so a
  # test run never marks the branch as measured.
  [ "${REG_SMOKE:-0}" = "1" ] || echo "$SHA" >"$STATE/$SLUG"
  rm -rf "$WT"                  # drop the throwaway library clone
done

# ── Publish to the metrics repo (conflict-safe; NON-FATAL) ────────────────────────────────────
# CPU (Mac) and GPU (cluster) write DISJOINT paths (results/<plat>/, state/<plat>/, *_<plat>.yaml),
# so concurrent runs never conflict on CONTENT — only at the git level (a non-fast-forward if the
# other platform pushed between our clone and our push).  So: rebase onto the latest (always clean,
# the paths don't overlap) and retry.  If the push ultimately fails (auth/network), this run's
# results+state simply aren't persisted — which SELF-HEALS: next run sees no new state and re-measures.
# Skipped under REG_SMOKE — the isolated plumbing test must not touch the real metrics repo.
if [ "${REG_SMOKE:-0}" = "1" ]; then
  log "REG_SMOKE — skipping commit/push."
else
git -C "$METRICS_REPO" add results state >/dev/null 2>&1 || true
# Size cap (backstop): unstage any staged file larger than MAX_PUSH_FILE_MB so a stray large
# artifact can never be pushed.  Normal outputs (YAML/.txt/.npy) are well under this.
git -C "$METRICS_REPO" diff --cached --name-only 2>/dev/null | while IFS= read -r f; do
  [ -f "$METRICS_REPO/$f" ] || continue
  mb=$(( $(wc -c <"$METRICS_REPO/$f") / 1048576 ))
  if [ "$mb" -gt "${MAX_PUSH_FILE_MB:-25}" ]; then
    git -C "$METRICS_REPO" reset -q -- "$f"
    log "WARN: not pushing oversized file ($mb MB > ${MAX_PUSH_FILE_MB:-25} MB): $f"
  fi
done
CHANGED_SUMMARY="$(IFS=,; echo "${CHANGED_BR[*]}")"
if git -C "$METRICS_REPO" commit -q -m "regression $PLAT $DATE [$CHANGED_SUMMARY]" >/dev/null 2>&1; then
  pushed=0
  for attempt in 1 2 3; do
    git -C "$METRICS_REPO" pull --rebase --autostash -q >/dev/null 2>&1 || true
    if git -C "$METRICS_REPO" push -q >/dev/null 2>&1; then pushed=1; break; fi
    log "push attempt $attempt failed (concurrent update?); rebasing + retrying."
  done
  [ "$pushed" = "1" ] && log "pushed results to metrics." \
    || log "WARN: push failed after 3 attempts; results not persisted (re-measures next run)."
else
  log "nothing new to commit."
fi
fi   # end REG_SMOKE publish guard

[ "$GATE_FAIL" = "0" ] || { log "REGRESSION DETECTED — exit 1 (alert)."; exit 1; }
log "done — no regressions."
exit 0
