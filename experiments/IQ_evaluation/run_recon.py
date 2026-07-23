### Run baseline or variation reconstructions for the IQ evaluation test cases.
#
# Cases are defined in test_cases.py; per-case downsampling there is the low-res
# setting, and --full-res overrides downsampling and view subsampling to 1. Each
# run writes FDK + MBIR recons and a params.json snapshot to output/<case>/<tag>/.
# run_low_res.sh and run_full_res.sh run all cases under the 'low_res' and
# 'full_res' tags; use --tag/--set for other comparison runs.
#
# Examples:
#   python run_recon.py --list
#   python run_recon.py --case bga_no_hart
#   python run_recon.py --case bga_no_hart --full-res
#   python run_recon.py --case bga_no_hart --tag sharp2.0 --set sharpness=2.0
#   python run_recon.py --all

import argparse
import ast
import json
import os
import sys
import time
import numpy as np
import mbirjax as mj  # mbirjax must be imported before jax (sets XLA env vars)
import jax.numpy as jnp
import mbirjax.preprocess as mjp

from test_cases import TEST_CASES, DEFAULTS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')

# Kwargs accepted by each loader; all other case settings are recon settings.
LOADER_KEYS = {
    'zeiss': ('downsample_factor', 'subsample_view_factor', 'crop_pixels_sides',
              'crop_pixels_top', 'crop_pixels_bottom', 'bg_option', 'zinger_correction', 'auto_crop'),
    'nsi': ('downsample_factor', 'subsample_view_factor', 'crop_pixels_sides',
            'crop_pixels_top', 'crop_pixels_bottom', 'auto_crop', 'offset_correction'),
    'pymbir': ('bh_correction', 'auto_crop'),
}
RECON_KEYS = ('sharpness', 'snr_db', 'weight_type', 'max_iterations')


def load_sino_and_model(case_type, path, loader_kwargs):
    # Scalar downsample_factor means same factor for rows and channels
    kwargs = dict(loader_kwargs)
    if np.isscalar(kwargs.get('downsample_factor')):
        d = kwargs['downsample_factor']
        kwargs['downsample_factor'] = (d, d)

    if case_type == 'zeiss':
        return mjp.zeiss.get_sino_and_model(path, **kwargs)
    elif case_type == 'nsi':
        return mjp.nsi.get_sino_and_model(path, **kwargs)
    elif case_type == 'pymbir':
        return mjp.pymbir.get_sino_and_model(path, **kwargs)
    else:
        raise ValueError(f'Unknown case type: {case_type}')


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.ndarray, jnp.ndarray)):
        return np.asarray(obj).tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def parse_overrides(pairs):
    overrides = {}
    for pair in pairs or []:
        if '=' not in pair:
            sys.exit(f'Bad --set argument (expected key=value): {pair}')
        key, value = pair.split('=', 1)
        try:
            overrides[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            overrides[key] = value  # keep as string
    return overrides


def run_case(name, tag, overrides, full_res=False, view=False, overwrite=False):
    settings = dict(DEFAULTS)
    settings.update(TEST_CASES[name])
    case_type = settings.pop('type')
    path = settings.pop('path')
    if full_res:
        # Native resolution; cases whose loader has no downsampling options are unchanged.
        for key in ('downsample_factor', 'subsample_view_factor'):
            if key in LOADER_KEYS[case_type]:
                settings[key] = 1
    settings.update(overrides)

    loader_kwargs = {k: settings[k] for k in LOADER_KEYS[case_type] if k in settings}
    recon_kwargs = {k: settings[k] for k in RECON_KEYS if k in settings}
    unknown = set(settings) - set(LOADER_KEYS[case_type]) - set(RECON_KEYS)
    if unknown:
        sys.exit(f"Unknown settings for case '{name}' (type {case_type}): {sorted(unknown)}")

    out_dir = os.path.join(OUTPUT_DIR, name, tag)
    if os.path.exists(out_dir) and not overwrite:
        sys.exit(f'{out_dir} exists; use --overwrite to redo this run.')

    print(f"\n========== Case '{name}', tag '{tag}' ==========")
    print(f'Dataset: {path}')
    print(f'Loader settings: {loader_kwargs}')
    print(f'Recon settings:  {recon_kwargs}')

    sinogram, ct_model = load_sino_and_model(case_type, path, loader_kwargs)
    ct_model.set_params(sharpness=recon_kwargs['sharpness'], snr_db=recon_kwargs['snr_db'])
    weights = mj.gen_weights(sinogram, weight_type=recon_kwargs['weight_type'])

    print('\n********** FDK reconstruction **********')
    fdk_recon = ct_model.direct_recon(sinogram)

    print('\n********** MBIR reconstruction **********')
    time0 = time.time()
    mbir_recon, recon_dict = ct_model.recon(sinogram, init_recon=fdk_recon, weights=weights,
                                            max_iterations=recon_kwargs['max_iterations'])
    elapsed = time.time() - time0
    print(f'Elapsed time: {elapsed:.1f} s')

    os.makedirs(out_dir, exist_ok=True)
    mj.export_recon_hdf5(os.path.join(out_dir, 'fdk_recon.h5'), fdk_recon)
    mj.export_recon_hdf5(os.path.join(out_dir, 'mbir_recon.h5'), mbir_recon)
    with open(os.path.join(out_dir, 'recon_dict.json'), 'w') as f:
        json.dump(json_safe(recon_dict), f, indent=2)

    try:
        from importlib.metadata import version
        mbirjax_version = version('mbirjax')
    except Exception:
        mbirjax_version = 'unknown'
    params = dict(
        case=name, tag=tag, type=case_type, dataset_path=path,
        loader_settings=loader_kwargs, recon_settings=recon_kwargs, overrides=overrides,
        sinogram_shape=list(sinogram.shape),
        recon_shape=list(ct_model.get_params('recon_shape')),
        mbirjax_version=mbirjax_version,
        date=time.strftime('%Y-%m-%d %H:%M:%S'),
        elapsed_seconds=round(elapsed, 1),
    )
    with open(os.path.join(out_dir, 'params.json'), 'w') as f:
        json.dump(json_safe(params), f, indent=2)
    print(f'Saved recons and params to {out_dir}')

    if view:
        mj.slice_viewer(jnp.swapaxes(fdk_recon, 0, 2), jnp.swapaxes(mbir_recon, 0, 2), slice_axis=1,
                        slice_label=['FDK', 'MBIR'], title=f'{name} / {tag}: FDK vs MBIR')


def main():
    parser = argparse.ArgumentParser(description='Run IQ evaluation reconstructions.')
    parser.add_argument('--case', choices=sorted(TEST_CASES), help='Test case to run')
    parser.add_argument('--all', action='store_true', help='Run all test cases')
    parser.add_argument('--list', action='store_true', help='List test cases and exit')
    parser.add_argument('--tag', default='baseline', help='Output subdirectory name (default: baseline)')
    parser.add_argument('--set', dest='overrides', action='append', metavar='KEY=VALUE',
                        help='Override a setting, e.g. --set sharpness=2.0 (repeatable)')
    parser.add_argument('--full-res', action='store_true',
                        help='Set downsample_factor and subsample_view_factor to 1')
    parser.add_argument('--view', action='store_true', help='Open slice viewer when done')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite an existing output tag')
    args = parser.parse_args()

    if args.list:
        for name, case in TEST_CASES.items():
            settings = {**DEFAULTS, **{k: v for k, v in case.items() if k not in ('type', 'path')}}
            print(f"{name}  [{case['type']}]  {case['path']}")
            print(f'    {settings}')
        return
    if args.overrides and args.all:
        sys.exit('--set applies to a single case; use --case with --set.')
    if not args.case and not args.all:
        parser.error('Specify --case NAME, --all, or --list.')

    names = sorted(TEST_CASES) if args.all else [args.case]
    for name in names:
        run_case(name, args.tag, parse_overrides(args.overrides),
                 full_res=args.full_res, view=args.view, overwrite=args.overwrite)


if __name__ == '__main__':
    main()
