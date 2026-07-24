#!/bin/bash
# Publish the flash-remediation pages to the depot www directory for live viewing.
#
# The HTML pages are fully self-contained (figures are base64-embedded by
# embed_report_figures.py), so the html files alone are the complete site; nothing else
# is transferred.  The destination is PUBLICLY SERVED — internal material (the plan doc,
# the README, scripts) must stay out of it.
#
# All knobs at the top of publish_pages_main below (no CLI args).  Rerun after every
# embed refresh — it is idempotent.
#
# Portability/safety notes.  macOS ships openrsync, which lacks --chmod, so web-readable
# permissions are applied on the remote side after the transfer.  The body runs in a
# SUBSHELL function with strict mode confined to it, so this file is safe to run
# directly OR to `source`: an error aborts the publish and reports failure without
# killing the invoking shell, and no shell options leak into an interactive session.
# The function is deliberately called PLAINLY (not `if fn`/`fn ||` — those are condition
# contexts, in which bash silently disables the function's internal set -e).

publish_pages_main() (
  set -euo pipefail

  REMOTE=buzzard@gautschi.rcac.purdue.edu
  DEST=/depot/bouman/www/mbirjax/flash_remediation
  DELETE=false   # true -> mirror exactly (removes files at DEST not present here)

  HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
  if [ "$DELETE" = true ]; then DELETE_FLAG=--delete; else DELETE_FLAG=; fi

  ssh "$REMOTE" "mkdir -p $DEST"
  rsync -av $DELETE_FLAG "$HERE"/*.html "$REMOTE:$DEST/"
  ssh "$REMOTE" "chmod 755 $DEST && chmod 644 $DEST/*.html"
  echo "published to $DEST (index.html is the entry point)"
)

publish_pages_main
publish_rc=$?
unset -f publish_pages_main
if [ "$publish_rc" -ne 0 ]; then
  echo "publish_pages.sh: FAILED — see the error above; the site may be unchanged." >&2
fi
return $publish_rc 2>/dev/null || exit $publish_rc
