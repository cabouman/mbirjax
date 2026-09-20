#!/usr/bin/env bash
# Launch the Dehydration/Hydration Correction GUI, rebuilding first if the
# sources changed.
#
# Usage: ./launch_dehydration_hydration.sh [dehydration_hydration arguments...]
#   e.g. ./launch_dehydration_hydration.sh /SNS/VENUS/IPTS-XXXX/.../Run_YYYY
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="$REPO_DIR/target/release/dehydration_hydration"

# GUI apps need a display (e.g. a ThinLinc session).
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    echo "Error: no display found (DISPLAY/WAYLAND_DISPLAY unset)." >&2
    echo "Run this from a graphical session such as ThinLinc." >&2
    exit 1
fi

# Rebuild if the binary is missing or any source/manifest file is newer.
needs_build=false
if [[ ! -x "$BINARY" ]]; then
    needs_build=true
elif [[ -n "$(find "$REPO_DIR/src" "$REPO_DIR/Cargo.toml" -newer "$BINARY" -print -quit 2>/dev/null)" ]]; then
    needs_build=true
fi

if $needs_build; then
    CARGO="$(command -v cargo || true)"
    [[ -z "$CARGO" && -x "$HOME/.cargo/bin/cargo" ]] && CARGO="$HOME/.cargo/bin/cargo"
    if [[ -z "$CARGO" ]]; then
        echo "Error: binary is out of date and cargo was not found to rebuild it." >&2
        exit 1
    fi
    echo "Building dehydration_hydration (release)..."
    (cd "$REPO_DIR" && "$CARGO" build --release)
fi

exec "$BINARY" "$@"
