#!/bin/bash
# Open an xterm INSIDE the ThinLinc desktop that holds an interactive GPU allocation.
#
# This reproduces Greg's own workflow, started by Claude:
#     terminal on the login node -> sinteractive -> shell on a GPU node -> run GUI apps
# and it has the lifetime he expects: the allocation is held by the SHELL in the xterm,
# so closing a viewer window changes nothing.  It ends only when he types `exit` (or
# closes the xterm, or the walltime expires).
#
# Contrast with `srun --x11 <app>`, which allocates the node to run ONE command: when the
# app exits, the allocation dies with it.
#
# `sinteractive --x11` forwards X from the compute node back to this login node's display,
# which is the ThinLinc session -- so GUI apps launched in that shell appear in ThinLinc.
#
# Claude can run further work in the SAME allocation without re-queuing:
#     srun --overlap --jobid=<id> <cmd>
# `--overlap` is required: since Slurm 20.11 job steps are exclusive, and sinteractive's
# own `srun --pty $SHELL` step already holds the resources (see `sinteractive --help`).
#
# Run ON the login node hosting the ThinLinc session:
#     nohup bash tl_gpu_session.sh > /tmp/tl_gpu_session.log 2>&1 &
set -u

# ── allocation request (edit here; no command-line arguments by project convention) ──
ACCOUNT=bouman
PARTITION=ai
GPUS=1
CPUS=14                 # gautschi requires 14 cores per GPU
WALLTIME=04:00:00
TITLE="GPU interactive session (exit here to release the node)"
# Use the SAME terminal as the desktop (ThinLinc here runs XFCE), not bare xterm:
# xfce4-terminal has the File/Edit/View/Terminal/Tabs/Help menu bar -- Edit > Copy/Paste
# is the one Greg actually needs -- and it inherits his saved profile, so the font matches
# his default.  Plain xterm has neither.  --disable-server forces a private process rather
# than attaching to an existing terminal server, which the nohup/detach pattern needs.
TERM_CMD=xfce4-terminal

# ── discover the live ThinLinc session on this node ───────────────────────────
XVNC_ARGS=$(ps -u "$USER" -o args= 2>/dev/null | grep "[X]vnc :" | head -1)
if [ -z "$XVNC_ARGS" ]; then
    echo "FATAL: no live Xvnc for $USER on $(hostname) -- wrong login node?"
    exit 2
fi
DISPLAY=$(printf '%s\n' "$XVNC_ARGS" | grep -oE 'Xvnc :[0-9]+' | grep -oE ':[0-9]+')
XAUTHORITY=$(printf '%s\n' "$XVNC_ARGS" | sed -n 's/.*-auth \([^ ]*\).*/\1/p')
export DISPLAY XAUTHORITY
echo "host=$(hostname)  DISPLAY=$DISPLAY  XAUTHORITY=$XAUTHORITY"

# ── the TERMINAL holds the allocation: sinteractive is its foreground process ─
# --hold keeps the window up after the shell exits so any error stays readable.
# `env SHELL=...` (not a bare VAR=val prefix): xfce4-terminal --command parses argv
# directly rather than via a shell, so a bare prefix would be taken as the program name.
# SHELL= makes sinteractive (which runs `$SHELL -l`) start our wrapper instead, so the
# shell comes up with conda `mbirjax` active -- giving Greg's usual
#     (mbirjax) buzzard@host:[dir] $
# prompt -- and prints a banner with the job id, node and TIME REMAINING.
ALLOC="env SHELL=/scratch/gautschi/buzzard/h100_tuning/scripts/claude_shell sinteractive --x11 -A $ACCOUNT -p $PARTITION -N1 \
       --gpus-per-node=$GPUS --cpus-per-task=$CPUS -t $WALLTIME"

if command -v "$TERM_CMD" >/dev/null 2>&1; then
    exec "$TERM_CMD" --disable-server --hold --geometry=110x32+60+60 \
         --title="$TITLE" --command="$ALLOC"
fi

echo "WARNING: $TERM_CMD not found; falling back to xterm (no menu bar, no copy/paste)."
exec xterm -geometry 100x30+60+60 -title "$TITLE" -bg black -fg lightgray -hold \
     -e $ALLOC
