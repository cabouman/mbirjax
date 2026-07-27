"""Build the beam-hardening page: hardening_template.html + figures/ ->
plans/sharpness_schedule/hardening_streaks.html (self-contained; images embedded).

Refuses to build while [[SECTION]] placeholders remain in the template.

Run:  python build_hardening.py
"""

import base64
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(_HERE, 'hardening_template.html')
FIGURES_DIR = os.path.join(_HERE, 'figures')
OUTPUT = os.path.normpath(os.path.join(
    _HERE, '..', '..', '..', 'sharpness_schedule', 'hardening_streaks.html'))

FIGURES = {
    'bh_transfer_real': 'bh_transfer_real.png',
    'bh_error_panels': 'bh_error_panels.png',
    'bh_severity': 'bh_severity.png',
    'bh_calibrated': 'bh_calibrated.png',
}

CALIBRATED_SECTION = """
<div class="fig">
<div class="figlabel">The calibrated case beside the real padded scan</div>
{{FIG:bh_calibrated}}
<p class="cap">
<b>How to read it:</b> the calibrated truncated-padded case (severity 0.2, dense lattice) at
iteration 16.  <b>Left column:</b> its error against the reference-energy ground truth &mdash;
grid-aligned interball bands in the axial ball-layer slice (top) and z-localized face bands in
(x,&nbsp;z) (bottom).  <b>Right column:</b> the reconstruction itself at the ball layer (top),
and the real padded BGA's (x,&nbsp;z) mid-plane (bottom) for comparison &mdash; the family the
testbed is built to reproduce.</p>
</div>

<p>
Measured calibration, stated honestly: the dense-grid case at s&nbsp;=&nbsp;0.2 produces a
residual top-metal dip of &minus;0.030 against the real scan's &minus;0.014 &mdash; within a
factor ~2, with the residual channel's saturation and the different metal-thickness
distributions (a 256-scale synthetic lattice vs the real package) setting the limit of how
sharp this comparison can be.  The dial's mapping is lattice-dependent: denser grids raise
per-ray metal thickness and deepen the dip at fixed s.  For the bench's purpose &mdash; a
controlled testbed whose artifact family, sign, and rough magnitude match the real scan
&mdash; this is adequate; matching the dip exactly would tune s&nbsp;&asymp;&nbsp;0.1 or a
softer metal w<sub>pe</sub>, one knob either way.</p>
"""


def img_tag(path):
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('ascii')
    return f'<img src="data:image/png;base64,{b64}" alt="{os.path.basename(path)}">'


def main():
    with open(TEMPLATE) as f:
        html = f.read()
    html = html.replace('{{CALIBRATED_SECTION}}', CALIBRATED_SECTION)
    placeholders = re.findall(r'\[\[[A-Z0-9_]+\]\]', html)
    if placeholders:
        raise SystemExit(f'template still has placeholders: {sorted(set(placeholders))}')
    for token, fname in FIGURES.items():
        path = os.path.join(FIGURES_DIR, fname)
        if os.path.exists(path):
            html = html.replace('{{FIG:%s}}' % token, img_tag(path))
        else:
            raise SystemExit(f'missing figure {fname}')
    leftover = re.findall(r'\{\{[A-Z_]+\}\}|\{\{FIG:[^}]*\}\}', html)
    if leftover:
        raise SystemExit(f'unreplaced tokens: {leftover}')
    with open(OUTPUT, 'w') as f:
        f.write(html)
    print(f'wrote {OUTPUT} ({os.path.getsize(OUTPUT) / 1e6:.2f} MB)')


if __name__ == '__main__':
    main()
