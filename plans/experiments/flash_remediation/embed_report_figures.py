"""Refresh the figures embedded in the flash-remediation HTML reports.

The reports in plans/flash_remediation/ are self-contained: every figure is embedded as a
base64 data URI (the repo gitignores PNGs, so linked images would not survive a checkout).
After regenerating any figures, run this script to swap the new PNGs into the reports
in place.  Each <img> is identified by its unique alt text, so the script works on the
already-embedded files -- no placeholder tokens needed.

Run inside the mbirjax conda env from anywhere (paths are resolved relative to this file);
no CLI args.  It is idempotent: rerunning with unchanged figures rewrites identical bytes.
"""

import base64
import os
import re

# Map: report file -> { <img> alt text -> figure filename in figures/ }.
# When a script adds a NEW figure to a report, add its alt -> png entry here.
FIG_MAP = {
    'phase_3_results.html': {
        'P3 SiC x-z: old default vs extended, iter 50': 'p3a_sic_v4x_d4x_nv401_nch512_xz_iter50.png',
        'P3 SiC convergence: old vs extended slab': 'p3a_sic_v4x_d4x_nv401_nch512_convergence.png',
        'P3 BGA per-slice noise: axial extension leaves the center-slice spike': 'p3b_bga_normal_v2x_d2x_noise_profile.png',
        'P3 BGA center slices: no pad vs axial vs axial+lateral': 'p3e_bga_xy_center.png',
        'P3 BGA radial profiles: ring at each grid boundary': 'p3e_bga_radial.png',
        'P3 BGA convergence: no pad vs axial vs axial+lateral': 'p3e_bga_convergence.png',
    },
    'phase_1_results.html': {
        'Lateral truncation: center slice montage': 'lateral_center_slice.png',
        'Lateral truncation: radial profile': 'lateral_radial_profile.png',
        'Lateral truncation: ring buildup over iterations': 'lateral_ring_buildup.png',
        'Lateral truncation: convergence by region': 'lateral_convergence.png',
        'Axial truncation: x-z section montage': 'z_xz_section.png',
        'Axial truncation: z profile': 'z_profile.png',
        'Axial truncation: artifact buildup over iterations': 'z_buildup.png',
        'Axial truncation: convergence by region': 'z_convergence.png',
    },
    'phase_2a_axial_results.html': {
        'P2a x-z recons: no remediation vs row taper': 'p2a_xz_taper.png',
        'P2a x-z recons: padding levels': 'p2a_xz_padding.png',
        'P2a z profile: one representative per family': 'p2a_z_profile.png',
        'P2a z profile: padded family, zoomed to the truncated end': 'p2a_z_profile_controls.png',
        'P2a convergence by region': 'p2a_convergence.png',
        'P2a-R wide-cone x-z recons': 'p2ar_R1_widecone_xz.png',
        'P2a-R sharp-regularization x-z recons': 'p2ar_R2_sharp_xz.png',
        'R2 pad_full at 40, 80, and 160 iterations': 'p2ar_r2probe_xz.png',
        'R2 vs default regularization: pad_full convergence': 'p2ar_r2probe_convergence.png',
        'P2a-R noisy-regime x-z recons': 'p2ar_R3_noise_xz.png',
    },
    'phase_2b_radial_results.html': {
        'P2b core center-slice recons': 'p2b_core_center.png',
        'P2b knee curves by overshoot': 'p2b_knee_overshoot.png',
        'P2b extreme overshoot center-slice recons': 'p2b_extreme_center.png',
        'P2b knee curves by regime': 'p2b_knee_regimes.png',
    },
    'phase_2c_split_results.html': {
        'Lilly D01788 x-z sections near the seam': 'p2c_lilly_stripes_xz.png',
        'Lilly D01788 seam profiles': 'p2c_lilly_seam_profiles.png',
        'Lilly D01788 split variants at 4x': 'p2c_lilly_variants.png',
        'Lilly D01788 seam ablations per-slice RMS': 'p2c_lilly_ablation_rms.png',
        'Lilly D01788 split variants at 8x': 'p2c_lilly_variants_ds8.png',
        'P2c structured x-z sections vs unsplit reference': 'p2c_structured_xz.png',
        'Synthetic reproduction of the seam stripes': 'p2c8_offset_reproduction.png',
    },
}

if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(here, 'figures')
    report_dir = os.path.normpath(os.path.join(here, '..', '..', 'flash_remediation'))

    for report_name, alt_to_png in FIG_MAP.items():
        report_path = os.path.join(report_dir, report_name)
        with open(report_path, encoding='utf-8') as f:
            text = f.read()
        for alt, png_name in alt_to_png.items():
            with open(os.path.join(fig_dir, png_name), 'rb') as f:
                uri = 'data:image/png;base64,' + base64.b64encode(f.read()).decode()
            # Replace the src of the (unique) img tag carrying this alt text.  src comes
            # before alt in the reports' markup; both attribute values are quote-free of
            # embedded quotes, so the non-greedy character classes are safe.
            pattern = r'(<img src=")[^"]*(" alt="' + re.escape(alt) + r'")'
            text, count = re.subn(pattern, lambda m: m.group(1) + uri + m.group(2), text)
            if count != 1:
                raise RuntimeError(f'{report_name}: expected exactly one <img> with '
                                   f'alt="{alt}", found {count}')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f'{report_name}: refreshed {len(alt_to_png)} figures '
              f'({os.path.getsize(report_path) / 1e6:.2f} MB)')
