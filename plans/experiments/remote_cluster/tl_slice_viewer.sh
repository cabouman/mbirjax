#!/bin/bash
# Launch mbirjax slice_viewer on a GPU COMPUTE NODE, rendering into the ThinLinc desktop
# running on THIS login node.
#
# Why this shape:
#  * ThinLinc's Xvnc runs with `-nolisten tcp -localhost`, so only processes on the SAME
#    login node can draw into it -- hence this script must run on the session's node.
#  * A compute node therefore cannot reach the display directly; `srun --x11` bridges it
#    (gautschi has PrologFlags = Alloc,Contain,X11), tunnelling from the compute node back
#    to this host's display.
#  * The display number is NOT stable (a restarted session moves :1 -> :2 -> ...), so it is
#    discovered from the live Xvnc process rather than hardcoded.
#
# Run ON the login node hosting the ThinLinc session:
#     nohup bash tl_slice_viewer.sh > /tmp/tl_viewer.log 2>&1 &
# The nohup+& matter: the window then survives the ssh that started it.
set -u

PY=$HOME/.conda/envs/mbirjax/bin/python
SCRIPT=/scratch/gautschi/buzzard/h100_tuning/scripts/x11_slice_viewer_demo.py

echo "host: $(hostname)"

# ── discover the live ThinLinc session on this node ───────────────────────────
XVNC_ARGS=$(ps -u "$USER" -o args= 2>/dev/null | grep "[X]vnc :" | head -1)
if [ -z "$XVNC_ARGS" ]; then
    echo "FATAL: no live Xvnc for $USER on $(hostname)."
    echo "       ThinLinc sessions are per-login-node; check you are on the right one."
    exit 2
fi
TL_DISPLAY=$(printf '%s\n' "$XVNC_ARGS" | grep -oE 'Xvnc :[0-9]+' | grep -oE ':[0-9]+')
TL_AUTH=$(printf '%s\n' "$XVNC_ARGS" | sed -n 's/.*-auth \([^ ]*\).*/\1/p')

echo "ThinLinc DISPLAY    : $TL_DISPLAY"
echo "ThinLinc XAUTHORITY : $TL_AUTH"
[ -r "$TL_AUTH" ] || { echo "FATAL: cannot read $TL_AUTH"; exit 3; }

export DISPLAY="$TL_DISPLAY"
export XAUTHORITY="$TL_AUTH"

# Sanity: can something on THIS node draw to the session before we burn a GPU alloc?
if command -v xdpyinfo >/dev/null 2>&1; then
    xdpyinfo >/dev/null 2>&1 \
        && echo "local check: can talk to $TL_DISPLAY  OK" \
        || { echo "FATAL: cannot open $TL_DISPLAY from $(hostname)"; exit 4; }
fi

# ── run on a GPU node, forwarding X back to the ThinLinc display ──────────────
echo "submitting srun --x11 ..."
exec srun --x11 -A bouman -p ai -N1 --gpus-per-node=1 --cpus-per-task=14 \
     -t 01:00:00 -J tl_viewer "$PY" -u "$SCRIPT"
