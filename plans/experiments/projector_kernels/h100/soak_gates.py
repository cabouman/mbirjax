"""Soak validation (2026-07-13): run the three shipped gate harnesses back-to-back
THREE times, then the occasional convergence-equivalence gate once, all in one sbatch
(sequential, ~2.5 h).  The individual gates are already green; this checks REPEATED-RUN
stability -- every cell must PASS in all three repetitions and per-cell walls must be
stable (max/min < 1.1).  The inc5 ``cone_vcd`` cell is the DOCUMENTED expected-fail
(~8.5e-3, intrinsic edge conditioning -- see gpu_headroom_findings.md "inc5 VCD
divergence"); it is reported, not chased.

Each ``w2_incN_ab.py`` is itself an orchestrator that spawns one isolated subprocess per
cell (CUDA_VISIBLE_DEVICES pins n; MBIRJAX_DISABLE_PALLAS=1 is the off config) and prints
a per-cell summary; this wrapper just runs them in order, tagging each run with a
``SOAK-BEGIN/END`` marker so the stability table can be parsed from the slurm log.  The
convergence gate (``w2_inc5_convergence.py``) is Greg-designated occasional-only, so it
runs ONCE, not three times.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
GATES = ['w2_inc3_ab.py', 'w2_inc4_ab.py', 'w2_inc5_ab.py']
REPS = 3


def run(tag, script):
    print(f'\n########## SOAK-BEGIN {tag} ({script}) ##########', flush=True)
    t0 = time.perf_counter()
    rc = subprocess.run([sys.executable, '-u', os.path.join(HERE, script)]).returncode
    print(f'########## SOAK-END {tag} rc={rc} elapsed={time.perf_counter() - t0:.1f}s '
          f'##########', flush=True)


def main():
    for rep in range(1, REPS + 1):
        for g in GATES:
            run(f'rep{rep}/{g[:-3]}', g)
    run('convergence/once', 'w2_inc5_convergence.py')
    print('\n########## SOAK COMPLETE ##########', flush=True)


if __name__ == '__main__':
    main()
