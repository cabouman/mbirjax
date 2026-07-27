#!/bin/bash
# Publish the sharpness/snr_db streak-study pages to the depot www directory.
#
# The HTML pages are fully self-contained (figures base64-embedded by the
# findings/build_*.py scripts), so the html files alone are the complete site;
# nothing else is transferred.  The destination is PUBLICLY SERVED — internal
# material (the plan docs, scripts) must stay out of it.
#
# All knobs at the top of publish_pages_main below (no CLI args).  Rerun after
# every page rebuild — it is idempotent.  Same portability/safety pattern as
# plans/flash_remediation/publish_pages.sh (subshell strict mode; remote-side
# chmod because macOS openrsync lacks --chmod).

publish_pages_main() (
  set -euo pipefail

  REMOTE=buzzard@gautschi.rcac.purdue.edu
  DEST=/depot/bouman/www/mbirjax/sharpness_schedule
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
