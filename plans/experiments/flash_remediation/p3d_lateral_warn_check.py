"""Phase 3 / step C validation: the lateral truncation detect-and-warn on real scans.

Host-side only (auto_set_regularization_params runs statistical checks on a view
subsample -- no recon, no GPU): for each cached dataset, build the model exactly as its
cache sidecar prescribes, run auto_set_regularization_params, capture whether the
truncation warning fires, and independently recompute the indicator + edge fraction for
the quantitative table (so the margin is reported for the silent cases too).

Expectations:
  bga_normal_v2x_d2x  -- severely truncated laterally -> MUST fire, large edge fraction.
  z62_v4x_d4x         -- channel-flash-dominated (partition study) -> expect fire.
  sic_v4x_d4x         -- in-FoV in channels (axial-only case) -> expect SILENT.
  lilly_v4x_d4x       -- auto-cropped autoinjector -> expect silent (crop leaves margin).

Run on gautschi (login node is fine):  python -u p3d_lateral_warn_check.py
Writes p3d_lateral_warn_summary.json to OUT_DIR.
"""
import json
import os
import subprocess
import warnings

import numpy as np
import mbirjax as mj                      # must precede jax (env binding)
import mbirjax.preprocess as mjp

# ---------------- run parameters (edit here; no CLI args) ----------------
PARTITION_CACHE = '/depot/bouman/data/mbirjax_metrics/partition_sequence/cache'
PADDING_DIR = '/depot/bouman/data/mbirjax_metrics/padding'
OUT_DIR = PADDING_DIR
DATASETS = {
    'bga_normal_v2x_d2x': os.path.join(PADDING_DIR, 'bga_normal_v2x_d2x_cache'),
    'z62_v4x_d4x_nv201_nch512': os.path.join(PARTITION_CACHE, 'z62_v4x_d4x_nv201_nch512'),
    'sic_v4x_d4x_nv401_nch512': os.path.join(PARTITION_CACHE, 'sic_v4x_d4x_nv401_nch512'),
    'lilly_v4x_d4x_nv450_nch470': os.path.join(PARTITION_CACHE, 'lilly_v4x_d4x_nv450_nch470'),
}
EXPECTED_COMMIT = '41ecbc2'               # step-C head ("Update plans and code for lateral padding.")

TRUNCATION_MATCH = 'Lateral FoV truncation'


def _lib_provenance():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(mj.__file__)))
    try:
        commit = subprocess.run(['git', '-C', repo, 'rev-parse', '--short', 'HEAD'],
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = 'unknown'
    return mj.__file__, commit


def check_dataset(tag, base_path):
    sino, geometry_params, optional_params, _ = mjp.load_cone_preprocessing(base_path + '.h5')
    with open(base_path + '.json') as f:
        sidecar = json.load(f)
    model = getattr(mj, sidecar['model_class'])(**geometry_params)
    if optional_params:
        model.set_params(**optional_params)
    if sidecar['auto_set_recon_geometry']:
        model.auto_set_recon_geometry()
    # verbose stays at its default (1): the warning must fire exactly as a user would see it.

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.auto_set_regularization_params(sino)
    fired = [c for c in caught if TRUNCATION_MATCH in str(c.message)]

    # Independent quantitative view: the same subsample + indicator the check used.
    small = model.subsample_views(sino, num_real_views=model.get_params('sinogram_shape')[0])
    indicator = model._get_sino_indicator(small, verbose=0)
    all_ones = bool(np.all(indicator))
    edge_frac = float(np.mean(np.logical_or(indicator[:, :, 0], indicator[:, :, -1])))

    result = {'tag': tag, 'sino_shape': [int(x) for x in sino.shape],
              'warning_fired': bool(fired), 'edge_frac': edge_frac,
              'indicator_all_ones': all_ones,
              'message': str(fired[0].message) if fired else None}
    status = 'FIRED' if fired else 'silent'
    print(f'{tag}: {status}  edge_frac={edge_frac:.3f}  all_ones={all_ones}', flush=True)
    if fired:
        print(f'  message: {fired[0].message}', flush=True)
    return result


if __name__ == '__main__':
    lib_file, commit = _lib_provenance()
    print(f'mbirjax={lib_file} commit={commit}', flush=True)
    if EXPECTED_COMMIT is not None:
        assert commit == EXPECTED_COMMIT, \
            f'live mbirjax commit {commit} != expected {EXPECTED_COMMIT} -- wrong checkout?'
    results = [check_dataset(tag, path) for tag, path in DATASETS.items()]
    with open(os.path.join(OUT_DIR, 'p3d_lateral_warn_summary.json'), 'w') as f:
        json.dump({'commit': commit, 'results': results}, f, indent=1)
    print('done: p3d_lateral_warn_check', flush=True)
