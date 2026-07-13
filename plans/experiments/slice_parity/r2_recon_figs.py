"""Build the D0-vs-D1 head-to-head slice figures + HTML page from the volumes that
r2_recon_capture.py staged (run this on gautschi AFTER the capture job; CPU-only).

For each (case, sharpness, iteration): a figure with two view rows (axial mid-slice,
coronal mid-cut) and columns [150-it reference (if present) | D0 | D1 | D1 - D0].
Grayscale window shared across columns from the reference's (or D0@30's) percentiles;
the difference panel is symmetric with its own limit printed in the title.

Outputs: PNGs in OUT_DIR + an HTML page (OUT_HTML) with <img> tags pointing at
HTML_IMG_PREFIX<name>.png.  NOTE: the generated page was superseded 2026-07-12 by the
hand-authored narrative `plans/slice_parity/r2_recon_compare.html` (also published at
/depot/bouman/www/mbirjax/skip_0_results/) — rerun this script for the FIGURES only;
do not overwrite the committed/published page with OUT_HTML.

Run:  python -u plans/experiments/slice_parity/r2_recon_figs.py   (login node is fine)
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
STAGE = '/scratch/gautschi/buzzard/parity_recons'
REF_DIRS = {'lilly_ds8': '/home/buzzard/parity_lilly',
            'z62': '/scratch/gautschi/buzzard/parity_z62',
            'lilly_ds4': '/scratch/gautschi/buzzard/parity_lilly_ds4'}
OUT_DIR = os.path.join(STAGE, 'figs')
OUT_HTML = os.path.join(STAGE, 'r2_recon_compare.html')
HTML_IMG_PREFIX = 'r2_recon_figs/'
CASES = ['lilly_ds8', 'z62', 'lilly_ds4']
SHARPNESS_LIST = [1.0, 2.0]
FIG_ITERS = [15, 20]
ARMS = ['D0_default', 'D1_g2start']
ARM_LABELS = {'D0_default': 'D0 [0,2,4,6,7] (default)',
              'D1_g2start': 'D1 [2,4,6,7] (drop 0)'}


def mid_slices(vol):
    """(axial mid-slice, coronal mid-cut) as 2-D arrays with z vertical for coronal."""
    axial = vol[:, :, vol.shape[2] // 2]
    coronal = vol[:, vol.shape[1] // 2, :].T      # (slices, rows): z runs vertically
    return {'axial (z mid)': axial, 'coronal (y mid)': coronal}


def build_fig(case, s, it, out_png):
    vols = {a: np.load(os.path.join(STAGE, f'{case}_{a}_s{s}_it{it}.npy'))
            for a in ARMS}
    ref_path = os.path.join(REF_DIRS[case], f'ref_sharp{s}.npy')
    ref = np.load(ref_path) if os.path.exists(ref_path) else None
    base = ref if ref is not None else np.load(
        os.path.join(STAGE, f'{case}_{ARMS[0]}_s{s}_it30.npy'))
    vmin, vmax = np.percentile(base, [0.5, 99.5])

    cols = ([('reference (150 it)', ref)] if ref is not None else []) + \
           [(ARM_LABELS[a], vols[a]) for a in ARMS] + [('diff', None)]
    views = mid_slices(vols[ARMS[0]]).keys()
    fig, axes = plt.subplots(2, len(cols), figsize=(3.2 * len(cols), 6.4))
    for i, view in enumerate(views):
        d0v = mid_slices(vols[ARMS[0]])[view]
        d1v = mid_slices(vols[ARMS[1]])[view]
        for j, (label, vol) in enumerate(cols):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            if label == 'diff':
                diff = d1v - d0v
                lim = np.percentile(np.abs(diff), 99.9) or 1e-6
                ax.imshow(diff, cmap='coolwarm', vmin=-lim, vmax=lim)
                ax.set_title(f'D1 − D0 (±{lim:.2g})', fontsize=9)
            else:
                ax.imshow(mid_slices(vol)[view], cmap='gray', vmin=vmin, vmax=vmax)
                ax.set_title(label if i == 0 else '', fontsize=9)
            if j == 0:
                ax.set_ylabel(view, fontsize=9)
    fig.suptitle(f'{case}  sharpness {s}  —  {it} iterations '
                 f'(window [{vmin:.3g}, {vmax:.3g}])', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    metrics = {}
    summary_path = os.path.join(STAGE, 'capture_summary.json')
    if os.path.exists(summary_path):
        for r in json.load(open(summary_path)):
            metrics[(r['case'], r['name'], r['sharpness'])] = r.get('cropped_lognrmse')

    html = ['<html><head><title>D0 vs D1 recon comparison</title>',
            '<style>body{font-family:sans-serif;max-width:1400px;margin:auto} '
            'img{max-width:100%} h2{margin-top:2em}</style></head><body>',
            '<h1>D0 [0,2,4,6,7] vs D1 [2,4,6,7] — head-to-head recons</h1>',
            '<p>Same data, same sharpness, same iteration count; the only difference '
            'is whether the schedule includes the granularity-0 (single-subset, '
            'full-volume) first iteration.  Grayscale window is shared within each '
            'figure; the rightmost column shows the signed difference D1 − D0 with '
            'its own symmetric limit.  Volumes for slice_viewer are on gautschi in '
            f'<code>{STAGE}</code>.</p>']
    for case in CASES:
        html.append(f'<h2>{case}</h2>')
        for s in SHARPNESS_LIST:
            m0 = metrics.get((case, 'D0_default', s))
            m1 = metrics.get((case, 'D1_g2start', s))
            if m0 and m1:
                html.append('<p>cropped log10 NRMSE (lower = better): ' +
                            ', '.join(f'{k}: D0 {m0[k]} / D1 {m1[k]}'
                                      for k in sorted(m0)) + '</p>')
            for it in FIG_ITERS:
                name = f'{case}_s{s}_it{it}.png'
                build_fig(case, s, it, os.path.join(OUT_DIR, name))
                html.append(f'<img src="{HTML_IMG_PREFIX}{name}">')
                print(f'built {name}', flush=True)
    html.append('</body></html>')
    with open(OUT_HTML, 'w') as f:
        f.write('\n'.join(html))
    print(f'wrote {OUT_HTML} + {OUT_DIR}', flush=True)


if __name__ == '__main__':
    main()
