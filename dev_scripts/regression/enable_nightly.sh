#!/usr/bin/env bash
# enable_nightly.sh (macOS / launchd) — install + load the nightly regression agent.
#
# Run it from the regression/ dir (in the metrics clone for the real install, or here during dev).
# It fills com.mbirjax.regression.plist from regression.env (schedule + the wrapper path + a PATH
# that includes conda) and loads it.  launchd runs at the scheduled time and at the next wake if the
# laptop was asleep.  Re-run after editing regression.env to apply changes.
#
# (Cluster uses scrontab + nightly_regression.slurm instead — see README; this script is Mac-only.)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/regression.env"

LABEL="com.mbirjax.regression"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
WRAPPER="$HERE/run_regression.sh"
TEMPLATE="$HERE/com.mbirjax.regression.plist"
LOGDIR="$HOME/.mbirjax/regression"

[ -f "$WRAPPER" ] || { echo "ERROR: wrapper not found at $WRAPPER"; exit 1; }
command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not on PATH (run from a shell where conda works)."; exit 1; }

# Daily HH:MM from POLL_SCHEDULE ("M H * * *"); only daily is supported by this installer.
MIN="$(echo "$POLL_SCHEDULE" | awk '{print $1}')"; [ "$MIN" = "*" ] && MIN=0
HR="$(echo "$POLL_SCHEDULE"  | awk '{print $2}')"; [ "$HR"  = "*" ] && HR=2
CONDA_BIN="$(dirname "$(command -v conda)")"

mkdir -p "$LOGDIR" "$HOME/Library/LaunchAgents"
sed -e "s|@LABEL@|$LABEL|g" \
    -e "s|@WRAPPER@|$WRAPPER|g" \
    -e "s|@HOUR@|$HR|g" -e "s|@MINUTE@|$MIN|g" \
    -e "s|@PATH@|$CONDA_BIN:/usr/bin:/bin:/usr/sbin:/sbin|g" \
    -e "s|@LOGOUT@|$LOGDIR/launchd.out.log|g" \
    -e "s|@LOGERR@|$LOGDIR/launchd.err.log|g" \
    "$TEMPLATE" > "$PLIST"

launchctl unload -w "$PLIST" 2>/dev/null || true
launchctl load -w "$PLIST"
printf 'Loaded %s — runs daily at %02d:%02d\n' "$LABEL" "$HR" "$MIN"
echo "  wrapper: $WRAPPER"
echo "  logs:    $LOGDIR/launchd.{out,err}.log"
echo "  (ENABLED=$ENABLED in regression.env is the in-wrapper kill-switch; this controls the schedule.)"
