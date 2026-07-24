"""R1: real-data schedule protocol (slice_parity_plan.md, R1 section).

Runs the candidate (S, P) schedules at two sharpness settings on real cone datasets,
against 150-iteration references, with the per-slice/cropped-NRMSE diagnostics and the
cone cost accounting from the protocol.  Wave 1: Lilly D01788 ds8 (all four candidates).
Wave 2: z62 (radial character, cached partition-study case) + Lilly ds4 confirmation
(C0/C1/C2).  R2: memory-driven schedules D0-D4 at sharpness {1.0, 2.0} on ds8 + z62
(Greg go 2026-07-12).  Select via ROUND / RUN_CASES / per-case schedules below; earlier
rounds' configs are in git history.

Structure: EVERY GPU step (preprocess, references, arms) runs in its own SUBPROCESS so
the orchestrator stays JAX-free — wave-1 lesson: in-process reference generation held
the XLA pool and every arm worker's cuBLAS init then failed with "Unable to get Blas
support".  Case data + references are cached per case in stage_dir (fail-fast
ordering); parity is mask-form via the P1 ParityMixin (bitwise-verified copy of the
library updater).  Restart-per-iteration capture, seeded per call.

Run (gautschi, 1 GPU):  sbatch plans/experiments/slice_parity/parity_realdata.slurm
"""
import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
NSI_LILLY_DIR = '/scratch/gautschi/buzzard/flash_lilly/D01788'
Z62_H5 = ('/depot/bouman/data/mbirjax_metrics/partition_sequence/cache/'
          'z62_v4x_d4x_nv201_nch512.h5')
ROUND = 'r2'                    # tags the per-case summary file: <ROUND>_summary.json
SHARPNESS_LIST = [1.0, 2.0]     # R2: Greg's realistic band (R1 waves used [1.0, 2.5])
NUM_ITERATIONS = 30
REF_ITERATIONS = 150
SEED = 1000
INTERIOR_RADIUS_FRAC = 0.85     # cropped-metric disk (flash-analysis pattern)
AXIAL_CROP_FRAC = 0.10          # exclude this fraction of slices at each end

# (name, partition_sequence, phases_spec) — phases_spec int or per-iteration list.
SCHED_ALL = [
    ('C0_default',   [0, 2, 4, 6, 7], 1),
    ('C1_parityall', [0, 2, 4, 6, 7], 2),
    ('C2_composite', [1, 3, 5, 7],    [2, 2, 2, 1]),
    ('C3_flat128',   [7],             1),
]
SCHED_CONFIRM = SCHED_ALL[:3]   # ds4 confirmation: C3's read was already clear at ds8

# R2 (memory-driven schedules, plan doc "R1 synthesis" item 3): does dropping the
# granularity-0 full-volume iteration cost anything, and is a g1x2 coarse start better
# than g2x1 on real data?  NOTE: mask-form g1x2 gives NO memory win today (full-height
# buffers) — D2/D3 answer only the convergence-shape question; D1/D4 are the schedules
# deployable now.
R2_SCHEDULES = [
    ('D0_default',   [0, 2, 4, 6, 7], 1),
    ('D1_g2start',   [2, 4, 6, 7],    1),
    ('D2_g1x2once',  [1, 4, 6, 7],    [2, 1, 1, 1]),
    ('D3_g1x2twice', [1, 1, 4, 6, 7], [2, 2, 1, 1, 1]),
    ('D4_g2twice',   [2, 2, 4, 6, 7], 1),
]

# Per-case dataset loader, staging dir, and schedule list.  RUN_CASES picks the round.
CASES = {
    'lilly_ds8': dict(loader='nsi', dataset_dir=NSI_LILLY_DIR, downsample=(8, 8),
                      subsample_views=8, stage_dir='~/parity_lilly',
                      schedules=R2_SCHEDULES),     # wave 1 ran SCHED_ALL (git history)
    'z62': dict(loader='h5', h5_path=Z62_H5, auto_set_recon_geometry=True,
                stage_dir='/scratch/gautschi/buzzard/parity_z62',
                schedules=R2_SCHEDULES),           # wave 2 ran SCHED_ALL (git history)
    'lilly_ds4': dict(loader='nsi', dataset_dir=NSI_LILLY_DIR, downsample=(4, 4),
                      subsample_views=4,
                      stage_dir='/scratch/gautschi/buzzard/parity_lilly_ds4',
                      schedules=SCHED_CONFIRM),    # wave 2 (complete)
}
RUN_CASES = ['lilly_ds8', 'z62']


def stage_dir(case_name):
    return os.path.expanduser(CASES[case_name]['stage_dir'])


def load_case(case_name):
    """Load (sino, weights, geom_params, optional_params); cache slow NSI preprocessing.

    h5 cases (partition-study caches) load directly via load_cone_preprocessing — the h5 IS
    the cache; weights regenerate per call (one cheap op) when none were saved.
    """
    c, stage = CASES[case_name], stage_dir(case_name)
    import mbirjax as mj
    import mbirjax.preprocess as mjp
    if c['loader'] == 'h5':
        sino, geom_params, optional_params, weights = mjp.load_cone_preprocessing(c['h5_path'])
        sino = np.asarray(sino)
        if weights is None:
            weights = np.asarray(mj.gen_weights(sino, 'transmission_root'))
        return sino, weights, geom_params, optional_params

    cache = os.path.join(stage, f'{case_name}_case.npz')
    pkl = os.path.join(stage, f'{case_name}_params.pkl')
    if case_name == 'lilly_ds8':                    # wave-1 cache predates this layout
        cache = os.path.join(stage, 'lilly_ds8_case.npz')
        pkl = os.path.join(stage, 'lilly_ds8_params.pkl')
    if os.path.exists(cache) and os.path.exists(pkl):
        d = np.load(cache)
        with open(pkl, 'rb') as f:
            geom_params, optional_params = pickle.load(f)
        return d['sino'], d['weights'], geom_params, optional_params
    sino, geom_params, optional_params = mjp.nsi.compute_sino_and_params(
        c['dataset_dir'], downsample_factor=c['downsample'],
        subsample_view_factor=c['subsample_views'])
    sino = np.asarray(sino)
    weights = np.asarray(mj.gen_weights(sino, 'transmission_root'))
    os.makedirs(stage, exist_ok=True)
    np.savez_compressed(cache, sino=sino, weights=weights)
    with open(pkl, 'wb') as f:
        pickle.dump((geom_params, optional_params), f)
    return sino, weights, geom_params, optional_params


def build_model(case_name, geom_params, optional_params, sharpness, sino,
                parity_cls=True):
    import mbirjax as mj
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from parity_convergence_ab import ParityMixin

    class ParityCone(ParityMixin, mj.ConeBeamModel):
        pass

    cls = ParityCone if parity_cls else mj.ConeBeamModel
    model = cls(**geom_params)
    if optional_params:
        model.set_params(**optional_params)
    if CASES[case_name].get('auto_set_recon_geometry'):
        model.auto_set_recon_geometry()
    model.set_params(sharpness=sharpness, verbose=0)
    model.auto_set_regularization_params(sino)
    model.set_params(auto_regularize_flag=False)
    return model


def crop_masks(recon_shape):
    nz = recon_shape[2]
    z0, z1 = int(np.ceil(nz * AXIAL_CROP_FRAC)), nz - int(np.ceil(nz * AXIAL_CROP_FRAC))
    i = np.arange(recon_shape[0], dtype=np.float32)[:, None] - (recon_shape[0] - 1) / 2.0
    j = np.arange(recon_shape[1], dtype=np.float32)[None, :] - (recon_shape[1] - 1) / 2.0
    disk = np.sqrt(i ** 2 + j ** 2) < INTERIOR_RADIUS_FRAC * (min(recon_shape[:2]) / 2.0)
    return disk, z0, z1


def worker(cfg):
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.94')
    import mbirjax as mj  # noqa: F401  (device init before heavy work)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from parity_convergence_ab import phase_masks

    out = dict(cfg)
    stage = stage_dir(cfg['case'])
    try:
        sino, weights, geom_params, optional_params = load_case(cfg['case'])
        model = build_model(cfg['case'], geom_params, optional_params,
                            cfg['sharpness'], sino)
        reference = np.load(os.path.join(
            stage, f'ref_sharp{cfg["sharpness"]}.npy'))
        recon_shape = model.get_params('recon_shape')
        disk, z0, z1 = crop_masks(recon_shape)
        ref_crop_norm = np.linalg.norm(reference[disk][:, z0:z1])
        num_slices = recon_shape[2]

        pseq, phases_spec = cfg['pseq'], cfg['phases']
        errs, cropped_lognrmse = [], []
        init_recon = None
        t0 = time.perf_counter()
        for j in range(NUM_ITERATIONS):
            phases_j = (phases_spec if isinstance(phases_spec, int)
                        else phases_spec[min(j, len(phases_spec) - 1)])
            model.parity_masks = phase_masks(num_slices, phases_j)
            model.set_params(partition_sequence=pseq)
            np.random.seed(SEED)
            recon, _ = model.recon(sino, weights=weights, init_recon=init_recon,
                                   first_iteration=j, max_iterations=j + 1,
                                   stop_threshold_change_pct=0, print_logs=False)
            recon = np.asarray(recon)
            diff = recon - reference
            errs.append(np.linalg.norm(diff, axis=(0, 1)))          # per-slice, uncropped
            cropped_lognrmse.append(float(np.log10(
                np.linalg.norm(diff[disk][:, z0:z1]) / ref_crop_norm + 1e-30)))
            init_recon = recon
        out['wall_s'] = round(time.perf_counter() - t0, 1)

        # Idealized cost units per iteration (protocol: back 1, fwd P => (P+1)/2 on cone).
        cost_units = [((phases_spec if isinstance(phases_spec, int)
                        else phases_spec[min(j, len(phases_spec) - 1)]) + 1) / 2.0
                      for j in range(NUM_ITERATIONS)]
        np.savez_compressed(
            os.path.join(stage, f'{cfg["name"]}_sharp{cfg["sharpness"]}.npz'),
            errs=np.stack(errs), cropped_lognrmse=np.array(cropped_lognrmse),
            cost_units=np.array(cost_units), final_recon=recon)
        out['cropped_lognrmse'] = [round(v, 4) for v in cropped_lognrmse]
        out['cum_cost'] = [round(v, 1) for v in np.cumsum(cost_units)]
        out['status'] = 'ok'
    except Exception:
        import traceback
        out['status'] = 'error'
        out['traceback'] = traceback.format_exc()
    print('RESULT ' + json.dumps(out), flush=True)


def prep_worker(case_name):
    """Subprocess: preprocess and cache the case (imports jax when cache is cold)."""
    sino, _, _, _ = load_case(case_name)
    print(f'[{case_name} preprocess] sino {sino.shape}', flush=True)


def ref_worker(case_name, sharpness):
    """Subprocess: one 150-iteration library-solver reference on the GPU."""
    sino, weights, geom_params, optional_params = load_case(case_name)
    model = build_model(case_name, geom_params, optional_params, sharpness, sino,
                        parity_cls=False)
    np.random.seed(SEED)
    ref, _ = model.recon(sino, weights=weights, max_iterations=REF_ITERATIONS,
                         stop_threshold_change_pct=0, print_logs=False)
    np.save(os.path.join(stage_dir(case_name), f'ref_sharp{sharpness}.npy'),
            np.asarray(ref))


def run_step(argv, label):
    """Run one GPU step in a subprocess; the orchestrator must stay JAX-free
    (a resident XLA pool starves later workers' cuBLAS init)."""
    proc = subprocess.run([sys.executable, os.path.abspath(__file__)] + argv,
                          capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        print(f'[{label} FAILED rc={proc.returncode}]\n{proc.stderr[-2000:]}',
              flush=True)
        sys.exit(1)


def main():
    for case_name in RUN_CASES:
        stage = stage_dir(case_name)
        os.makedirs(stage, exist_ok=True)
        print(f'\n===== case {case_name} (stage {stage}) =====', flush=True)
        run_step(['--prep-worker', case_name], f'{case_name} preprocess')

        # References first (fail-fast; library solver, default sequence).
        for sharpness in SHARPNESS_LIST:
            if os.path.exists(os.path.join(stage, f'ref_sharp{sharpness}.npy')):
                continue
            print(f'[{case_name} reference] sharpness {sharpness}: '
                  f'{REF_ITERATIONS} iterations...', flush=True)
            run_step(['--ref-worker', case_name, str(sharpness)],
                     f'{case_name} reference s{sharpness}')

        results = []
        for sharpness in SHARPNESS_LIST:
            for name, pseq, phases in CASES[case_name]['schedules']:
                cfg = dict(case=case_name, name=name, pseq=pseq, phases=phases,
                           sharpness=sharpness)
                proc = subprocess.run([sys.executable, os.path.abspath(__file__),
                                       '--worker', json.dumps(cfg)],
                                      capture_output=True, text=True)
                got = False
                for line in proc.stdout.splitlines():
                    if line.startswith('RESULT '):
                        r = json.loads(line[len('RESULT '):])
                        results.append(r)
                        got = True
                        tail = (r['cropped_lognrmse'][-1] if r['status'] == 'ok'
                                else 'ERR')
                        print(f'[{case_name} {name} s{sharpness}] final cropped '
                              f'log10 NRMSE: {tail} ({r.get("wall_s", "?")}s)',
                              flush=True)
                if not got:
                    print(f'[no RESULT rc={proc.returncode}] {cfg}\n'
                          f'{proc.stderr[-1500:]}', flush=True)

        with open(os.path.join(stage, f'{ROUND}_summary.json'), 'w') as f:
            json.dump(results, f, indent=2)
        print(f'[{case_name} summary + arrays in {stage}]', flush=True)


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == '--worker':
        worker(json.loads(sys.argv[2]))
    elif len(sys.argv) >= 3 and sys.argv[1] == '--prep-worker':
        prep_worker(sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[1] == '--ref-worker':
        ref_worker(sys.argv[2], float(sys.argv[3]))
    else:
        main()
