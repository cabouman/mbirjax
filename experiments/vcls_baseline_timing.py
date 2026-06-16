"""
experiments/vcls_baseline_timing.py
───────────────────────────────────
Apples-to-apples harness for the vcls view_indices -> set_view_parameters conversion.

Runs get_opt_views on a fixed, seeded problem and reports wall time plus the
outputs (selected view indices + VCL value + a gamma checksum).  Run it ONCE on
the pre-conversion code (BASELINE) and once after the conversion; the selected
indices must match exactly, gamma/VCL to float noise, and the time should be
comparable or better (the conversion replaces per-view view_indices slicing of
the full model with a 1-view model + set_view_parameters, which avoids any
recompile per view and removes the only nontrivial view_indices user).

Single-variable discipline: same seed, same sizes, same machine, nothing else
running.  No CLI args; edit the config below.

    python experiments/vcls_baseline_timing.py
"""
import time

import numpy as np

# ── Run configuration ─────────────────────────────────────────────────────────
NUM_VIEWS = 180
NUM_ROWS = 64          # detector rows = recon slices (parallel beam)
NUM_CHANNELS = 64
NUM_SELECTED = 8
SEED = 42


def main():
    import mbirjax as mj
    import mbirjax.vcls as vcls

    # NOTE: pre-conversion vcls hardcodes get_params('view_params_array'), which only
    # exists on ConeBeamModel (parallel beam names its view params 'angles'), so the
    # baseline uses cone; the conversion generalizes the lookup via view_params_name.
    angles = np.linspace(0, np.pi, NUM_VIEWS, endpoint=False)
    model = mj.ConeBeamModel((NUM_VIEWS, NUM_ROWS, NUM_CHANNELS), angles,
                             source_detector_dist=4 * NUM_CHANNELS,
                             source_iso_dist=2 * NUM_CHANNELS)
    recon_shape = model.get_params('recon_shape')
    rng = np.random.default_rng(SEED)
    reference_object = rng.random(recon_shape).astype(np.float32)

    t0 = time.time()
    selected, vcl_value = vcls.get_opt_views(model, reference_object,
                                             num_selected_views=NUM_SELECTED, seed=SEED)
    dt = time.time() - t0

    print(f'\nproblem: {NUM_VIEWS}x{NUM_ROWS}x{NUM_CHANNELS}, select {NUM_SELECTED}, seed {SEED}')
    print(f'wall time: {dt:.2f} s')
    print(f'selected view indices: {sorted(int(i) for i in selected)}')
    print(f'vcl value: {float(vcl_value):.8e}')


if __name__ == '__main__':
    main()
