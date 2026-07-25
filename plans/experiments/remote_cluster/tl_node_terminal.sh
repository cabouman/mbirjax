#!/bin/bash
# Open a terminal ON THE COMPUTE NODE of an existing interactive allocation, with X
# forwarded to the ThinLinc desktop, and (optionally) run a command in it.
#
# This is the "second window" half of the workflow: tl_gpu_session.sh gets the node and
# holds it from a login-node terminal; this puts a shell ON that node so work runs where
# the GPU is, and GUI apps land in ThinLinc.
#
# Run ON the login node hosting the ThinLinc session.  Edit JOBID/RUN_CMD below.
set -u

JOBID=${JOBID:-14201524}          # the allocation to attach to (squeue -u $USER)
# Command run in the new terminal before it drops to an interactive shell.  `exec bash`
# afterwards keeps the window alive when the command (or its viewer window) exits.
RUN_CMD=${RUN_CMD:-"\$HOME/.conda/envs/mbirjax/bin/python -u /scratch/gautschi/buzzard/h100_tuning/scripts/x11_slice_viewer_demo.py"}

XVNC_ARGS=$(ps -u "$USER" -o args= 2>/dev/null | grep "[X]vnc :" | head -1)
[ -n "$XVNC_ARGS" ] || { echo "FATAL: no live Xvnc for $USER on $(hostname)"; exit 2; }
DISPLAY=$(printf '%s\n' "$XVNC_ARGS" | grep -oE 'Xvnc :[0-9]+' | grep -oE ':[0-9]+')
XAUTHORITY=$(printf '%s\n' "$XVNC_ARGS" | sed -n 's/.*-auth \([^ ]*\).*/\1/p')
export DISPLAY XAUTHORITY
echo "login node=$(hostname) DISPLAY=$DISPLAY jobid=$JOBID"

# --overlap: share the allocation with the sinteractive shell already holding it
#            (job steps are exclusive since Slurm 20.11 -- without this it would hang).
# --x11    : forwarding is PER STEP; the sinteractive shell's forwarding is not inherited.
exec srun --overlap --jobid="$JOBID" --x11 \
     xfce4-terminal --disable-server --geometry=110x34+120+120 \
       --title="h-node shell (job $JOBID) -- GPU work runs here" \
       --command="bash --rcfile /scratch/gautschi/buzzard/h100_tuning/scripts/claude_bashrc -i -c '$RUN_CMD; echo; echo \"[command finished -- shell follows]\"; exec bash --rcfile /scratch/gautschi/buzzard/h100_tuning/scripts/claude_bashrc -i'"
