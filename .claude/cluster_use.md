# Using the Purdue RCAC clusters (for Claude sessions)

Two clusters, one shared group storage.  **gautschi = H100, gilbreth = A100.**  Same
SSH key for both; slurm account `bouman` on both.  Coordinate with Greg before heavy or
long batch use — his interactive sessions share the same allocation/queue.

## Contents

- [The workflows that matter](#the-workflows-that-matter)
- [SSH access](#ssh-access)
- [Watching a job](#watching-a-job)
- [Remote GUI windows (slice_viewer, PyCharm) — VERIFIED 2026-07-25](#remote-gui-windows-slice_viewer-pycharm-verified-2026-07-25)
- [gautschi — H100 (the nightly-regression cluster)](#gautschi-h100-the-nightly-regression-cluster)
  - [The nightly regression system (runs on gautschi)](#the-nightly-regression-system-runs-on-gautschi)
- [gilbreth — A100 (lightly used by our group)](#gilbreth-a100-lightly-used-by-our-group)
- [Scratch — fast, big, PURGE-ELIGIBLE](#scratch-fast-big-purge-eligible)
- [Home — a 25 GB quota that fails SILENTLY](#home-a-25-gb-quota-that-fails-silently)
- [Depot — durable, group-shared (mounted on BOTH clusters)](#depot-durable-group-shared-mounted-on-both-clusters)
- [Moving data on and off](#moving-data-on-and-off)
- [Nested ssh — run on a worker node without waiting in the queue](#nested-ssh-run-on-a-worker-node-without-waiting-in-the-queue)
- [Other repos, data, and resources](#other-repos-data-and-resources)
- [Running a specific library state](#running-a-specific-library-state)
- [Job preflight — two lines that catch the two worst failures](#job-preflight-two-lines-that-catch-the-two-worst-failures)
- [Failure signatures → what they actually mean](#failure-signatures-what-they-actually-mean)
- [Don't](#dont)

## The workflows that matter

1. **Shared interactive session (the usual one).**  Either of us starts a terminal in
   Greg's ThinLinc desktop holding a GPU allocation; we both work on that node — he can
   type in it and start PyCharm, Claude can run in it with
   `srun --overlap --jobid=<id>`.  Ends only on `exit`.
   → `remote_cluster/tl_gpu_session.sh`, then `remote_cluster/tl_node_terminal.sh`.
2. **Batch jobs for data collection (Claude, unattended).**  `sbatch` a self-contained
   script, results to scratch, poll the log.  The default for sweeps, benchmarks and
   anything long — no GUI, no held allocation.
3. **A dedicated GUI session Claude starts and Greg watches** — e.g. `slice_viewer` on a
   GPU node rendering into ThinLinc.  → `remote_cluster/tl_slice_viewer.sh`.
4. **Batch on the cluster, look at it locally.**  Compute remotely, write the volume or
   PNGs to scratch, copy down, view on the Mac.  Best when the result fits on a laptop
   (a 128³ float32 recon is ~8 MB) — no network in the interaction loop.

Rules of thumb: **prefer 2** unless a human needs to see something live; **prefer 4 over
3** when the data can travel; use **1** when Greg wants to drive.

## SSH access

```bash
ssh -o BatchMode=yes buzzard@gautschi.rcac.purdue.edu   '<cmd>'
ssh -o BatchMode=yes buzzard@gilbreth.rcac.purdue.edu   '<cmd>'
```

- Key auth with the **default** `~/.ssh/id_rsa` (passphrase-less).  Pass **NO** `-i` —
  SSH auto-selects it.  Do **not** force `-i ~/.ssh/id_rsa_gau`; that key is rejected.
- `-o BatchMode=yes` so a failure errors instead of hanging on a prompt.
- gilbreth's handshake is **slow (~90 s)** — give ssh/scp to it a generous timeout and
  prefer `run_in_background: true`.  gautschi is fast.
- Login nodes have **no GPU**.  GPU work goes through `sbatch`/`srun`, or directly on the
  worker node of an existing interactive session (see "Nested ssh" below).
- To check the job queue: `squeue -u buzzard`.  Verify a run by reading its **log file**,
  not by `pgrep` (a `pgrep -f` self-matches the ssh command that launched it).

## Watching a job

```bash
squeue -u buzzard                      # queued + running
squeue -j <id> -o "%.10i %.8T %.6M %.6L %N"   # state, elapsed, LEFT, node
squeue -h -j <id> -o "%L"              # just the time remaining
sacct -j <id> --format=JobID,State,ExitCode,Elapsed   # after it ends
scancel <id>                           # stop it
```

- **Read the LOG FILE, not the process list.**  `pgrep -f <pattern>` run over ssh matches
  its own ssh command line and reports a false positive.
- `sbatch -o`/`-e` decide where output lands — always point them at scratch.
- Watch a long run with a `tail -f` filtered to the lines that matter, and make the filter
  cover FAILURE signatures too (`Traceback|Error|FAILED|OOM`), not just progress: a filter
  that only matches success is silent during a crash, and silence looks like "still running".
- VCD prints `Error sino RMSE` every iteration — grepping bare `Error` for failures gives
  a hit on every healthy run.

## Remote GUI windows (slice_viewer, PyCharm) — VERIFIED 2026-07-25

Claude *can* put a live GUI window (mbirjax `slice_viewer`, any matplotlib figure) on
Greg's screen from a GPU compute node.  Two routes, both tested end to end.
**Prefer route B (ThinLinc): route A is noticeably laggy on sliders, route B has no
perceptible lag.**

For headless checks neither is needed — save PNG/HTML to scratch and send the file.
For anything that fits on the Mac, copying the volume down and viewing locally beats
both (a 128³ float32 recon is ~8 MB).

**Route A — `ssh -Y` to the Mac's X server.**  Needs **XQuartz 2.8.6+** on the Mac
(universal/arm64-native), installed from <https://www.xquartz.org/>, followed by a
**log out and back in** so the launchd `DISPLAY` socket registers.  Verify with
`xdpyinfo` before blaming anything downstream.  If XQuartz refuses to start with
"Cannot establish any listening sockets", delete the stale `/tmp/.X0-lock` and retry.

```bash
ssh -Y buzzard@gautschi... 'srun --x11 -A bouman -p ai -N1 --gpus-per-node=1 \
    --cpus-per-task=14 -t 01:00:00 python my_viewer_script.py'
```
The window dies if that ssh drops.

**Route B — into Greg's ThinLinc desktop (preferred).**  His session is persistent on
one login node (login01 as of 2026-07-25) and survives disconnects.  Constraints that
shape the recipe:

* ThinLinc's Xvnc runs `-nolisten tcp -localhost`, so **only processes on that same login
  node** can draw into it.  A compute node cannot reach it directly — `srun --x11` bridges
  it (verified: login01 `:2` → h002 `localhost:42.0`).
* **The display number is NOT stable** — a restarted session moves `:1` → `:2` → …
  Discover it from the live Xvnc process; never hardcode.
* `login01.gautschi.rcac.purdue.edu` fails host-key verification directly; hop via the
  round-robin address instead.

```bash
# discover (run ON the session's login node):
XVNC=$(ps -u $USER -o args= | grep "[X]vnc :" | head -1)
export DISPLAY=$(printf '%s' "$XVNC" | grep -oE 'Xvnc :[0-9]+' | grep -oE ':[0-9]+')
export XAUTHORITY=$(printf '%s' "$XVNC" | sed -n 's/.*-auth \([^ ]*\).*/\1/p')
# then launch; nohup + & so it outlives the ssh that started it:
nohup srun --x11 -A bouman -p ai -N1 --gpus-per-node=1 --cpus-per-task=14 \
      -t 04:00:00 python my_viewer_script.py > /tmp/viewer.log 2>&1 &
```
Working example: `plans/experiments/remote_cluster/tl_slice_viewer.sh`
(+ `x11_slice_viewer_demo.py`, the small GPU recon + viewer it runs) (auto-discovers
the display, sanity-checks it, then submits).

**Use the DESKTOP'S terminal, not bare `xterm`.**  ThinLinc here runs **XFCE**, and
`xfce4-terminal` is installed — that is the terminal Greg actually uses.  It has the
File/Edit/View/Terminal/Tabs/Help menu bar (**Edit → Copy/Paste is the one he needs**) and
inherits his saved profile, so the font matches.  Bare `xterm` has neither: small fixed
font, no menus, no copy/paste.  Launch with `--disable-server` so it is a private process
rather than attaching to a terminal server (the nohup/detach pattern needs that), and
`--hold` so errors stay readable.  A harmless `Failed to connect to session manager`
warning appears when launching outside the desktop session manager.

**Allocation lifetime — the gotcha.**  `srun <cmd>` allocates the node to run *that one
command*: when the viewer window closes, the command exits and **the whole allocation
ends**.  That surprised Greg, whose own workflow is `sinteractive` → a login *shell* holds
the allocation → he starts PyCharm from it → closing GUI windows changes nothing → the
session ends only when he types `exit`.  So:

* want the node to persist across closing windows → hold it with a **shell** (`sinteractive`
  / `salloc`), not with the app;
* want to run more work in an allocation that already exists →
  **`srun --overlap --jobid=<id> <cmd>`** (verified working — this is how to inspect a node
  Claude already holds, without re-queuing);
* `salloc --no-shell` (allocate with no controlling process, then ssh in) did not show up in
  this slurm's `--help` — **unverified, needs testing** if a detached persistent allocation
  is wanted.

Working example of the shell-held form: `plans/experiments/remote_cluster/tl_gpu_session.sh`
(discovers the ThinLinc display, opens `xfce4-terminal`, runs `sinteractive --x11` in it).
Verified 2026-07-25: terminal on login01 → shell on h008, released only on `exit`.

Two things that bite when sharing that allocation:

* **`--x11` is set at ALLOCATION time, not per step** — and steps inherit it.  Passing
  `--x11` to a step inside an existing job is refused:
  `srun: error: Ignoring --x11 option for a job step within an existing job.  Set x11
  options at job allocation time.`  Harmless (the step still runs), but the lesson is that
  the allocation must have been created with X11 — `sinteractive --x11` — after which every
  `srun --overlap --jobid=<id>` step gets `DISPLAY` for free (verified: `localhost:75.0`
  with its own `/tmp/.Xauthority-*` on the compute node).  If the allocation was created
  WITHOUT `--x11`, no step can display and it cannot be retrofitted — start a new one.
* **`--overlap` SHARES the GPU, it does not partition it.**  Fine for inspection
  (`nvidia-smi`, `ps`).  For real compute, two JAX processes on one card will collide —
  JAX preallocates ~75% of the GPU, so the second one typically fails outright.  Take turns,
  set `XLA_PYTHON_CLIENT_PREALLOCATE=false`, or use separate allocations.

**Match Greg's shell.**  His prompt is
`(mbirjax) buzzard@login01.gautschi:[mbirjax_applications] $` -- the `user@host:[dir]` part
comes from the SYSTEM profile (there is no PS1 in his ~/.bashrc), and `(mbirjax)` comes from
conda activation, which his .bashrc does NOT do (it installs the hook only).  So a fresh
shell is missing the prefix.  `remote_cluster/claude_bashrc` fixes it by doing what a LOGIN shell does:
source **`/etc/profile`** (which runs all of `/etc/profile.d/*.sh`), then `~/.bashrc`, then
`conda activate mbirjax`, then print a session banner (job, node, walltime, **time
remaining**, end time).  Sourcing `/etc/profile` — rather than cherry-picking the prompt
file — is essential: that directory also defines the **`module`** command
(`modules.sh`, `00-modulepath.sh`, `z01_default_module.sh`), so a shell without it cannot
`module load conda/cuda` at all.  Terminals get it with `bash --rcfile <path> -i`; `sinteractive`
needs `env SHELL=remote_cluster/claude_shell` instead, because it runs `$SHELL -l` and a
LOGIN shell ignores `--rcfile` (the wrapper drops the `-l`).  Note `xfce4-terminal --command`
parses argv directly, so use `env VAR=val cmd`, never a bare `VAR=val cmd` prefix.

**A terminal ON the compute node** (so work runs where the GPU is, and PyCharm/viewers can
be started from it): `plans/experiments/remote_cluster/tl_node_terminal.sh` — run it on the
login node with `JOBID=<id>`; it opens `xfce4-terminal` on the allocated node via
`srun --overlap`, optionally running a command first and then dropping to a shell.  Verified
2026-07-25: terminal on h008 inside job 14201524, viewer displayed in ThinLinc.  Each such
terminal is just a job STEP — closing it leaves the allocation alone, so open as many as
needed.

**Backend note.**  `mbirjax/viewer.py` does `matplotlib.use('TkAgg')` at import.  With no
`DISPLAY` (any batch job) that warns "TkAgg not available" and the backend stays **Agg**,
which draws nothing — this is why every nightly log carries that warning, and it is NOT a
missing dependency.  With `DISPLAY` set, matplotlib picks `tkagg` on its own and the viewer
works.

**Cost.**  JAX preallocates ~75% of the GPU at startup, so an idle viewer window squats on
~77 GB of an H100 at 0% utilization.  Fine for a look; wasteful if left open.
`XLA_PYTHON_CLIENT_PREALLOCATE=false` avoids it, at some performance cost.

## gautschi — H100 (the nightly-regression cluster)

- GPUs: **H100 80GB HBM3**, `gpu:h100:8` per node.  Partition `ai`, account `bouman`,
  QoS `normal`.  **`--cpus-per-task=14` per GPU is required.**
- **The `ai` partition REFUSES `--mem`** — do not pass it.  Host memory is strictly
  proportional to CPUs, and CPUs to GPUs:

  | slurm setting (`scontrol show partition ai`) | value |
  |---|---|
  | `DefCpuPerGPU` | 14 |
  | `DefMemPerCPU` = `MaxMemPerCPU` | 9200 MB |

  Def == Max is exactly why a `--mem` request has nowhere to land.  So **host RAM per GPU is
  fixed at 9200 MB x 14 ≈ 126 GB**, and the only way to get more host memory is to **request
  more GPUs** (`--gpus-per-node=2` → ~252 GB, and so on).  The node is provisioned to match:
  h-nodes have 112 CPUs / 8 GPUs / 1,031,500 MB, and 9200 x 112 = 1,030,400 MB.
  (gilbreth is different — its `sinteractive` line below does take `--mem`.)
- Repos: `~/PycharmProjects/{mbirjax, mbirjax_applications, mbirjax_metrics}`.
- Conda envs: `mbirjax` (interactive) and `mbirjax_regression` (the nightly).
- Node preamble (sourced first — puts conda on PATH, loads cuda, sets the squid proxy so
  git can reach github from a compute node): `source ~/load_conda_cuda.sh`.
- The metrics nightly runs from a scrontab entry (02:00 daily); logs at
  `/home/buzzard/.mbirjax/regression/nightly-<jobid>.log`.

Example batch header (1 GPU):

```bash
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 04:00:00
```

### The nightly regression system (runs on gautschi)

Automated performance + correctness tracking for `mbirjax`, living in the **separate**
`mbirjax_metrics` repo and published at <https://gbuzzard.github.io/mbirjax_metrics/>.

- **Schedule:** a `scrontab` entry, `0 2 * * *` (02:00 daily), job name `mbirjax-nightly`,
  4 GPUs / 56 cores / 4 h.  Inspect with `scrontab -l`.
- **It FRESH-CLONES both repos from origin each run.**  So **uncommitted or unpushed local
  changes have no effect on it** — a fix must be committed and pushed to be picked up.
  This is the single most surprising thing about it.
- **Fire-on-change:** each branch in `TRACKED_BRANCHES` (`run_configs.env`) is measured only
  when its remote tip moves; unchanged branches log `unchanged — skip`.  A gap in a branch's
  data usually means it simply did not move, not that anything failed.
- **Logs:** `/home/buzzard/.mbirjax/regression/nightly-<jobid>.log`.  **scron reuses the job
  id, so this file is OVERWRITTEN every night** — there is no history; capture anything you
  need before the next 02:00.
- **Alerting:** `--mail-type=FAIL` plus an explicit notify email; the run exits 1 when a gate
  trips, e.g. `main: GATE FAIL (perf regression) — REGRESSION DETECTED`.  That is the system
  working.  What it could NOT catch before 2026-07-25 was a run that silently measured on the
  wrong platform — see the platform-mismatch entry in the failure table below.
- **Env:** `mbirjax_regression` (NOT the `mbirjax` env used interactively), built with
  `INSTALL_EXTRAS_gpu` from `mbirjax_metrics/action_scripts/run_configs.env`.  That setting is
  independent of the library's own `dev_scripts/clean_install_all.sh`: changing the library's
  CUDA extra does **not** change the nightly's.  Keep them in sync by hand.

## gilbreth — A100 (lightly used by our group)

- Account `bouman` shows **4× A100-40GB total** (`slist`).  Greg believes there may be a
  cap of ~2 concurrent GPUs but has not found the command that expresses it —
  **treat the concurrent-GPU limit as unconfirmed; needs investigation.**
- Partition `a100-40gb` **silently mixes two hardware classes** — A100-**SXM4** (features
  `N`/`nvlink`, 4 GPU/node, 400 W) and A100-**PCIe** (`G`, 2 GPU/node, 250 W).  Their
  clocks differ, so **wall times are not comparable across them** — pin one class with
  `--constraint=N` (SXM4) or `--constraint=G` (PCIe) on any timing job.
- Repo `~/PycharmProjects/mbirjax`, conda env `mbirjax`.
- Interactive session (Greg's usual invocation):

```bash
sinteractive -N1 -n20 --gpus-per-node=1 --account=bouman \
             --partition=a100-40gb --mem=40G --time=04:00:00
```

- Batch header mirrors gautschi but with `-p a100-40gb --constraint=N`.  **Whether
  gilbreth enforces a per-GPU `--cpus-per-task` rule (gautschi requires 14) is
  unconfirmed — needs investigation;** the `sinteractive` above uses `-n20` for 1 GPU and
  works, so a plain core request is at least accepted.

## Scratch — fast, big, PURGE-ELIGIBLE

- gautschi: `/scratch/gautschi/buzzard/`   gilbreth: `/scratch/gilbreth/buzzard/`
- Multi-TB, fast, but **purge-eligible**: use for job outputs, staging, and any large
  intermediate (`.npy`/`.npz`, traces, compile caches).  Not for anything that must
  persist.

## Home — a 25 GB quota that fails SILENTLY

- **Never write large artifacts under `~`.**  A job that fills home dies mid-write with
  `sacct` `FAILED 1:0` and **no traceback, no shell echo**, leaving a truncated file.
  Quota accounting can also lag ~one retry after freeing space.
- Diagnose with `myquota`.  If a job fails with exit 1 and an empty-looking log, check
  `myquota` **before** debugging the code.  A home path that must hold big data can be a
  symlink into scratch.

## Depot — durable, group-shared (mounted on BOTH clusters)

`/depot/bouman/` (≈7 TB of 10 TB used).  Greg's policy: **long-term = depot, temporary =
scratch.**

- **Primary data:** `/depot/bouman/data/` — scan datasets (`.txrm`, etc.), converged
  reference recons, durable results.  Group-shared, so mind others' files.
- **Public web pages:** `/depot/bouman/www/` is the **web root**, served publicly at
  **`https://www.datadepot.rcac.purdue.edu/bouman/`** (the `www` is dropped in the URL).
  The landing page is mostly a data repository; this project's pages live in the
  **`mbirjax/`** subdirectory:

  | filesystem | public URL |
  |---|---|
  | `/depot/bouman/www/mbirjax/<area>/` | `https://www.datadepot.rcac.purdue.edu/bouman/mbirjax/<area>/` |

  Publish only finished, shareable **HTML** here (no source, no data) — the destination
  is on the open internet.  Files need `chmod 644`, directories `chmod 755`.  The publish
  idiom is an `rsync` of `*.html` to the depot www dir; see
  `plans/flash_remediation/publish_pages.sh` for a working example.

## Moving data on and off

```bash
# to the cluster (scp uses the same default key; no -i)
scp -o BatchMode=yes myscript.py buzzard@gautschi.rcac.purdue.edu:/scratch/gautschi/buzzard/<dir>/
# back to the Mac
scp -o BatchMode=yes buzzard@gautschi...:/scratch/gautschi/buzzard/<dir>/result.npz .
```

- **Staging code:** put run scripts in a scratch `scripts/` dir and `scp` updates as you
  iterate.  Do NOT edit inside Greg's `~/PycharmProjects` checkouts — those are his working
  trees (and on gautschi the `mbirjax` env is an EDITABLE install pointing at one, so a
  change there changes what runs).
- **Which store:** scratch for job output and anything regenerable; **depot for anything that
  must survive** (scratch is purge-eligible).  `/depot/bouman/` is mounted on **both**
  clusters, so it is the natural way to move results between gautschi and gilbreth without
  going through the Mac.
- **Bringing results home for local viewing** (workflow 4): a 128³ float32 recon is ~8 MB —
  copy it down and use `slice_viewer` on the Mac at full speed rather than over X11.
- Big transfers: prefer `rsync -av` (resumable, skips unchanged) over repeated `scp`.

## Nested ssh — run on a worker node without waiting in the queue

`pam_slurm_adopt` admits ssh to a node currently running one of your jobs (the session
dies with that allocation).  From the login node, hop to the worker:

```bash
# find the node running Greg's interactive session (job name 'interact'):
ssh -o BatchMode=yes buzzard@gautschi... 'squeue -u buzzard'
# then hop to it (double-hop through the login node):
ssh -o BatchMode=yes buzzard@gautschi... 'ssh -o BatchMode=yes h001 "<cmd>"'
```

Keep nested commands **simple** (`;`-joined) — complex quoting through the double hop
silently mangles.  Coordinate before using Greg's allocation this way.

## Other repos, data, and resources

- **Repos on the clusters:** `~/PycharmProjects/{mbirjax, mbirjax_applications,
  mbirjax_metrics}` on gautschi; `~/PycharmProjects/mbirjax` on gilbreth.
  - `mbirjax` — the library.
  - `mbirjax_applications` — demos, application/workflow scripts, and larger worked
    examples built on the library.
  - `mbirjax_metrics` — the performance/correctness nightly + dashboard (see below).
- **Lilly flash-remediation data** (real scans, recon volumes, analysis scripts):
  `/scratch/gautschi/buzzard/flash_lilly/` (gautschi scratch — purge-eligible, so the
  durable copies live under depot).
- **Metrics dashboard** (auto-rebuilt performance & correctness time series, CPU + GPU):
  <https://gbuzzard.github.io/mbirjax_metrics/>.  Built from the YAML in
  `mbirjax_metrics/results/`; the GPU series comes from the gautschi nightly.
- **User docs** (readthedocs, always current): <https://mbirjax.readthedocs.io>.  Built
  from `docs/source/` in the library repo.
- **Project web pages** (internal findings, published HTML): under
  `https://www.datadepot.rcac.purdue.edu/bouman/mbirjax/` — see the Depot section.

## Running a specific library state

To measure or debug a particular commit WITHOUT disturbing Greg's checkouts or envs:

```bash
# 1. a worktree of the commit -- never touches the working checkout
git -C ~/PycharmProjects/mbirjax worktree add --detach ~/mbirjax_main_wt <commit>
# 2. a venv layered over an existing env (no copy of jax/cuda deps)
python -m venv --system-site-packages ~/venvs/sharpness_main
# 3. the worktree, deps already satisfied by the parent env
~/venvs/sharpness_main/bin/pip install -e ~/mbirjax_main_wt --no-deps
```

Three properties worth knowing:

* it **wins over any editable install** already in the parent env, so it is a reliable
  override rather than a hope;
* it **never modifies your envs** — the parent conda env is untouched, so the nightly and
  interactive work carry on unaffected;
* **venvs ignore `~/.local` user-site packages**, which is a real advantage: a stray
  `pip install --user` shadowing the library is a distinct failure mode from the mount flap,
  and this layout is immune to it.

Working instance on gautschi: `~/venvs/sharpness_main` over `~/mbirjax_main_wt`.
Remove a worktree with `git worktree remove <path>` (the branch/commit is untouched).

## Job preflight — two lines that catch the two worst failures

Put this at the top of every sbatch/srun python entry point:

```python
import jax, mbirjax
assert jax.devices()[0].platform == 'gpu', f"NOT ON GPU: {jax.devices()}"
print("library under test:", mbirjax.__file__, flush=True)
```

The assert catches a **silent CPU fallback** (a broken CUDA plugin does not raise — jax just
uses the CPU, and the run looks fine while measuring the wrong thing; this cost three nights
of GPU data in July 2026).  The print catches the **wrong library state** — editable installs,
worktrees, venv layering and `~/.local` shadowing all fail the same way, by running code you
did not think you were running.

## Failure signatures → what they actually mean

| symptom | cause / fix |
|---|---|
| `ls: Cannot send after transport endpoint shutdown`, or an intermittent `ModuleNotFoundError` for numpy/stdlib internals that **differs run to run and hits every env** | the LOGIN NODE's home mount is flapping.  The files are fine and compute nodes are unaffected — retry, or move the work to a node.  Do not go hunting for a broken install.  (Bit three times on 2026-07-25.) |
| sbatch/srun on gautschi `ai` rejected for a memory request | that partition refuses `--mem` (`DefMemPerCPU == MaxMemPerCPU == 9200`).  Drop `--mem`; ask for more GPUs if you need more host RAM. |
| job exits 1, log looks empty or truncated mid-write | **home quota full** (25 GB, fails SILENTLY).  `myquota`; write to scratch instead. |
| `Access denied by pam_slurm_adopt: you have no active jobs on this node` | ssh to a compute node is only allowed while you hold a job there.  Get an allocation first, or `srun --overlap --jobid=<id>` instead. |
| `Host key verification failed` on `loginNN.gautschi…` | the per-node name is not in known_hosts.  Hop through the round-robin address: `ssh gautschi 'ssh login01 "…"'`. |
| GPU charts stop updating, results land as `regression_cpu_*` under `results/gpu/` | jax fell back to CPU (usually the CUDA plugin extra not matching the node's `module load cuda`).  Since 2026-07-25 `performance_tracking` hard-aborts instead; look above the abort for `Jax plugin configuration error`. |
| `srun: error: Ignoring --x11 option for a job step within an existing job` | harmless.  X11 is set at ALLOCATION time; steps inherit it.  If the allocation lacks `--x11` it cannot be retrofitted — start a new one. |
| prompt shows `bash-5.1$`, **or `module: command not found`** | non-login shell: `/etc/profile` (hence all of `/etc/profile.d/*.sh`) was not sourced.  That directory supplies BOTH the prompt and the `module` function.  Use `remote_cluster/claude_bashrc`. |
| XQuartz: "Cannot establish any listening sockets" | stale `/tmp/.X0-lock` from a failed start — delete it and retry. |
| a GUI window vanished when its app closed | the allocation was `srun <cmd>`, which ends with the command.  Hold it with a shell instead. |
| tests pass but prove nothing about the GPU kernels | `tests/test_pallas_kernels.py` silently runs in interpret mode when `_pallas_kernels.availability()` is False.  Assert availability first. |

## Don't

- **No compute on login nodes at all** — not even a short python analysis script.  They are
  shared, have no GPU, and their home mount flaps (see the failure table).  There is a
  one-liner substitute with no excuse not to use it:
  ```bash
  sbatch -A bouman -p ai -N1 --gpus-per-node=1 --cpus-per-task=14 -t 0:20:00 \
         --wrap "python -u script.py"
  ```
- **Never write large artifacts under `~`** — the 25 GB quota kills jobs with no traceback.
- **Nothing but finished HTML to `/depot/bouman/www/`** — it is served on the open internet.
  No data, no source, no drafts.
- **Don't edit Greg's cluster checkouts** (`~/PycharmProjects/*`) — stage your own scripts in
  scratch.
- **Coordinate before heavy gautschi use** — his interactive sessions and the 02:00 nightly
  share the same account and queue.  gilbreth is lightly used by the group; submit freely there.
- **Don't assume an uncommitted fix reaches the nightly** — it fresh-clones from origin.
