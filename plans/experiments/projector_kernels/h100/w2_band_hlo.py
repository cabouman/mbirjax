"""Wave-2 A5 probe: identify the band kernel's two transpose fusions (2026-07-13).

The n=2 trace (w2_band_trace.py, job 13497819) shows `input_transpose_fusion` and
`input_transpose_fusion_2` dominating cone banded back projection, executed once per
(view, band).  This probe dumps the compiled HLO of ONE banded back call at the n=2
shapes and prints each transpose fusion's parameters/shapes, deciding between the A5
forms: if both fusions are layouts of the SINOGRAM VIEW, pre-transposing the view
shard once per call kills both cheaply; if one is the horizontal fan's OUTPUT layout,
the fan itself must be hoisted out of the band loop.

Run on a 2-GPU node; output = the fusion bodies grepped from the XLA dump.
"""
import glob
import os
import re
import subprocess
import sys

SINO_SHAPE = (1024, 1024, 1024)
DUMP_DIR = '/scratch/gautschi/buzzard/w2_band_hlo'


def worker():
    import numpy as np
    import jax
    import jax.numpy as jnp
    import mbirjax

    views, rows, channels = SINO_SHAPE
    angles = np.linspace(-np.pi / 2, np.pi / 2, views, endpoint=False)
    model = mbirjax.ConeBeamModel(SINO_SHAPE, angles,
                                  source_detector_dist=4 * channels,
                                  source_iso_dist=4 * channels)
    print(f'devices={jax.devices()} summary={model.device_summary}', flush=True)
    recon_shape = model.get_params('recon_shape')
    idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=True)
    sino = jnp.asarray(np.random.default_rng(0).random(SINO_SHAPE, dtype=np.float32))
    # One full banded back projection: compiles (and dumps) every jit in the path.
    jax.block_until_ready(model.sparse_back_project(sino, idx))
    print('back_project done; dump written', flush=True)


def orchestrator():
    os.makedirs(DUMP_DIR, exist_ok=True)
    for f in glob.glob(os.path.join(DUMP_DIR, '*')):
        os.remove(f)
    env = dict(os.environ, W2H_CELL='1', CUDA_VISIBLE_DEVICES='0,1',
               MBIRJAX_DISABLE_PALLAS='1',
               XLA_FLAGS=(os.environ.get('XLA_FLAGS', '') +
                          f' --xla_dump_to={DUMP_DIR}'
                          ' --xla_dump_hlo_pass_re=.*fusion.*').strip())
    rc = subprocess.run([sys.executable, os.path.abspath(__file__)], env=env).returncode
    print(f'worker rc={rc}', flush=True)

    # Find the module containing the band kernel and print its transpose fusions.
    for path in sorted(glob.glob(os.path.join(DUMP_DIR, '*after_optimizations.txt'))):
        with open(path) as f:
            text = f.read()
        if 'input_transpose_fusion' not in text:
            continue
        print(f'--- {os.path.basename(path)} ---', flush=True)
        # Print each transpose fusion's computation: name line + its body lines.
        for m in re.finditer(r'^%?(\S*input_transpose_fusion\S*)\s.*?{(.*?)^}',
                             text, re.S | re.M):
            name, body = m.group(1), m.group(2)
            lines = [ln.strip() for ln in body.strip().splitlines()]
            print(f'FUSION {name}: {len(lines)} ops', flush=True)
            for ln in lines[:12]:
                print(f'    {ln[:160]}', flush=True)
        # Also show the call sites (which top-level instruction feeds them).
        for ln in text.splitlines():
            if 'input_transpose_fusion' in ln and 'fusion(' in ln:
                print(f'CALL  {ln.strip()[:200]}', flush=True)
    print('=== w2_band_hlo done ===', flush=True)


if __name__ == '__main__':
    worker() if os.environ.get('W2H_CELL') else orchestrator()
