"""E0 attribution (structure): dump + analyze the OPTIMIZED HLO of the production
forward/back projector programs at the 1024^3 n=1 cell.

Answers the four structure questions from gpu_headroom_plan.md section 6 E0 with NO timing
(the traffic model in appendix_roofline_traffic_model.md rests on these):

  1. MATERIALIZATION -- does a multi-GB instruction output exist per step (forward's sorted
     (V, T*P, B) `updates` stream; back's gathered/weighted intermediates)?  Reported as the
     largest instruction outputs per module.
  2. FUSION STRUCTURE -- is back's gather -> multiply -> tap-sum -> view-sum chain one fusion
     or several (each boundary is an HBM round-trip)?
  3. TRANSPOSE HOISTING -- does the back path's channel-major transpose sit inside a loop
     body (paid per step) or at top level (hoisted, paid once)?
  4. FAST PATHS -- is the sort a CUB custom-call, and does the scatter carry
     indices_are_sorted=true (the ScatterWithDistributedIndices precondition)?

Run on ONE GPU.  XLA_FLAGS and the compilation-cache override must be set before jax
initializes, so this script edits os.environ FIRST.  The persistent compilation cache is
DISABLED for this process: a cache hit skips compilation and produces NO dump files.

Output: a REPORT to stdout; the analyzed *after_optimizations* HLO files are copied to
RESULTS_DIR for later inspection (ncu/nsys kernel names join against these).

Run:  python plans/experiments/projector_kernels/e0_hlo_dump.py    (edit constants below)
"""
import os

# ── Config ────────────────────────────────────────────────────────────────────
SINO_SHAPE = (1024, 1008, 992)          # the 1024^3-class benchmark cell
DUMP_DIR = os.path.expanduser('~/headroom/results/e0_hlo_dump')
RESULTS_DIR = os.path.expanduser('~/headroom/results')
TOP_N = 20                              # largest instruction outputs to report per module
MIN_LOOP_REPORT_BYTES = 64 * 2**20      # call out loop-carried/loop-internal outputs >= this

# Must precede jax initialization (import happens below, after os.environ is set).
os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '')
                           + f' --xla_dump_to={DUMP_DIR} --xla_dump_hlo_as_text').strip()
os.environ['JAX_ENABLE_COMPILATION_CACHE'] = 'false'   # a cache hit would skip the dump
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.94')

import glob
import math
import re
import shutil
import sys

_DTYPE_BYTES = {'f64': 8, 'f32': 4, 'bf16': 2, 'f16': 2, 'f8': 1,
                's64': 8, 's32': 4, 's16': 2, 's8': 1,
                'u64': 8, 'u32': 4, 'u16': 2, 'u8': 1, 'pred': 1}

# One HLO instruction: "%name = f32[128,24576,252]{2,1,0} opcode(...)" (possibly indented).
_INSTR_RE = re.compile(r'^\s*(?:ROOT\s+)?%?([\w.\-]+)\s*=\s*\(?(\w+)\[([\d,]*)\]')
# A computation header: "%computation_name (args) -> type {" or "ENTRY %name ...".
_COMP_RE = re.compile(r'^(?:ENTRY\s+)?%?([\w.\-]+)\s+\([^)]*\)\s*->')


def shape_bytes(dtype, dims_str):
    if dtype not in _DTYPE_BYTES:
        return None
    dims = [int(d) for d in dims_str.split(',') if d] or [1]
    return math.prod(dims) * _DTYPE_BYTES[dtype]


def analyze_module(path):
    """Scan one optimized-HLO text file; return the report dict described in the docstring."""
    largest = []            # (bytes, computation, opcode-ish line fragment)
    custom_calls = []       # full lines (sort/cub targets show here)
    scatters = []           # full lines (indices_are_sorted / unique_indices attrs show here)
    transposes = []         # (computation, line fragment)
    fusion_count = 0
    comp = '<header>'
    with open(path) as f:
        for line in f:
            m = _COMP_RE.match(line)
            if m and '=' not in line.split('->')[0]:
                comp = m.group(1)
                continue
            m = _INSTR_RE.match(line)
            if not m:
                continue
            name, dtype, dims = m.groups()
            nbytes = shape_bytes(dtype, dims)
            frag = line.strip()[:160]
            if ' custom-call(' in line or ' custom-call<' in line or 'custom_call_target' in line:
                custom_calls.append(frag)
            if re.search(r'\bscatter\(', line):
                attrs = re.findall(r'indices_are_sorted=\w+|unique_indices=\w+', line)
                scatters.append(frag[:110] + ('  [' + ', '.join(attrs) + ']' if attrs else ''))
            if re.search(r'\btranspose\(', line):
                transposes.append((comp, frag[:120]))
            if ' fusion(' in line:
                fusion_count += 1
            # Materialization ranking: only instructions OUTSIDE fused computations produce
            # HBM buffers (fusion interiors are virtual; the fusion op itself appears in its
            # caller with the output shape), and parameters alias existing buffers.  GPU
            # fusion interiors are named fused_*/wrapped_* (CPU: fused_computation*); the
            # entry (main*) and while bodies/conds (region_*) are the real buffer sites.
            in_fusion_interior = (comp.startswith('fused_') or comp.startswith('wrapped_')
                                  or 'fused_computation' in comp)
            if (nbytes is not None and not in_fusion_interior
                    and ' parameter(' not in line):
                largest.append((nbytes, comp, frag[:120]))
    largest.sort(key=lambda t: -t[0])
    return dict(largest=largest[:TOP_N], custom_calls=custom_calls, scatters=scatters,
                transposes=transposes, fusion_count=fusion_count)


def report_module(tag, path):
    print(f'\n{"=" * 78}\nMODULE [{tag}]: {os.path.basename(path)}\n{"=" * 78}')
    r = analyze_module(path)
    print(f'fusions: {r["fusion_count"]}')
    print(f'\n-- top {TOP_N} instruction outputs (materialized-buffer candidates) --')
    for nbytes, comp, frag in r['largest']:
        in_loop = 'while' in comp or 'body' in comp or 'cond' in comp
        loop_tag = '  [IN LOOP BODY]' if in_loop and nbytes >= MIN_LOOP_REPORT_BYTES else ''
        print(f'  {nbytes / 2**30:8.3f} GiB  ({comp}){loop_tag}\n             {frag}')
    print('\n-- custom-calls (CUB sort shows here) --')
    for c in r['custom_calls'][:12]:
        print(f'  {c}')
    print('\n-- scatters (check indices_are_sorted / unique_indices) --')
    for s in r['scatters'][:8]:
        print(f'  {s}')
    print('\n-- transposes by computation (loop body => paid per step) --')
    for comp, frag in r['transposes'][:12]:
        print(f'  ({comp})  {frag}')
    dest = os.path.join(RESULTS_DIR, f'e0_hlo_{tag}.txt')
    shutil.copyfile(path, dest)
    print(f'\n[saved HLO text to {dest}]')


def main():
    os.makedirs(DUMP_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    import numpy as np
    import mbirjax
    import jax
    print(f'jax {jax.__version__}  devices={jax.devices()}  sino={SINO_SHAPE}')

    angles = np.linspace(0, np.pi, SINO_SHAPE[0], endpoint=False)
    model = mbirjax.ParallelBeamModel(SINO_SHAPE, angles)
    model.configure_devices(1)
    recon_shape = model.get_params('recon_shape')
    idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=model.get_params('use_ror_mask'))
    rng = np.random.default_rng(0)
    cylinders = model._shard_recon(rng.random((len(idx), recon_shape[2]), dtype=np.float32))
    jax.block_until_ready(cylinders)

    # ONE call each is enough -- we want the compiled programs, not timing.
    sino = jax.block_until_ready(model.sparse_forward_project(cylinders, idx))
    jax.block_until_ready(model.sparse_back_project(sino, idx))
    del cylinders, sino

    # The dump contains every compiled module (centers jits, shard helpers, ...).  The
    # projector driver programs are identified by their jitted-function names.
    def newest_matching(substr):
        # Exactly '<module>.<backend>_after_optimizations.txt' -- NOT the sibling
        # '...-memory-usage-report.txt' / '...-buffer-assignment.txt' variants.
        cands = [p for p in glob.glob(os.path.join(DUMP_DIR, '*after_optimizations.txt'))
                 if substr in os.path.basename(p)]
        return max(cands, key=os.path.getmtime) if cands else None

    fwd = newest_matching('sparse_forward_project')
    back = newest_matching('sparse_back_project')
    if not fwd or not back:
        print('\nDUMP FILES PRESENT:', file=sys.stderr)
        for p in sorted(glob.glob(os.path.join(DUMP_DIR, '*.txt')))[:40]:
            print('  ' + os.path.basename(p), file=sys.stderr)
        raise SystemExit('could not locate fwd/back after_optimizations dumps (names above)')
    report_module('fwd', fwd)
    report_module('back', back)


if __name__ == '__main__':
    main()
