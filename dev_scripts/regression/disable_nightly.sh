#!/usr/bin/env bash
# disable_nightly.sh (macOS / launchd) — unload + remove the nightly regression agent.
# Stops the SCHEDULE only; it does not touch regression.env or any results.  (For a quick pause
# without uninstalling, set ENABLED=0 in regression.env — the wrapper then exits immediately.)
set -euo pipefail
LABEL="com.mbirjax.regression"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
launchctl unload -w "$PLIST" 2>/dev/null || true
rm -f "$PLIST"
echo "Disabled $LABEL (unloaded + removed $PLIST)."
