#!/usr/bin/env bash
# disable_nightly.sh (macOS / launchd) — unload + remove the nightly regression agent.
# Stops the SCHEDULE only; it does not touch regression.env or any results.  (For a quick pause
# without uninstalling, set ENABLED=0 in regression.env — the wrapper then exits immediately.)
set -euo pipefail
# Keep an interactive terminal open on a nonzero exit so the error stays visible.
if [ -t 0 ]; then
  trap '_ec=$?; [ "$_ec" -ne 0 ] && { echo; echo ">>> $(basename "$0") exited with status $_ec — press Enter to close."; read -r _ || true; }' EXIT
fi
LABEL="com.mbirjax.regression"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
launchctl unload -w "$PLIST" 2>/dev/null || true
rm -f "$PLIST"
echo "Disabled $LABEL (unloaded + removed $PLIST)."
