"""Build the Phase B results page: phase_b_template.html + figures/ ->
plans/sharpness_schedule/phase_b_results.html (self-contained; images embedded).

Refuses to build while [[SECTION]] placeholders remain in the template, so a
skeleton can never be published by accident.

Run:  python build_phase_b.py
"""

import base64
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(_HERE, 'phase_b_template.html')
FIGURES_DIR = os.path.join(_HERE, 'figures')
OUTPUT = os.path.normpath(os.path.join(
    _HERE, '..', '..', '..', 'sharpness_schedule', 'phase_b_results.html'))

FIGURES = {
    'b1_trajectories': 'b1_trajectories.png',
    'b1_D4_compare': 'b1_D4_compare.png',
    'b1_S4_compare': 'b1_S4_compare.png',
}


def img_tag(path):
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('ascii')
    return f'<img src="data:image/png;base64,{b64}" alt="{os.path.basename(path)}">'


def main():
    with open(TEMPLATE) as f:
        html = f.read()
    placeholders = re.findall(r'\[\[[A-Z0-9_]+\]\]', html)
    if placeholders:
        raise SystemExit(f'template still has placeholders: {sorted(set(placeholders))}')
    for token, fname in FIGURES.items():
        path = os.path.join(FIGURES_DIR, fname)
        if os.path.exists(path):
            html = html.replace('{{FIG:%s}}' % token, img_tag(path))
        else:
            raise SystemExit(f'missing figure {fname}')
    leftover = re.findall(r'\{\{FIG:[^}]*\}\}', html)
    if leftover:
        raise SystemExit(f'unreplaced figure tokens: {leftover}')
    with open(OUTPUT, 'w') as f:
        f.write(html)
    print(f'wrote {OUTPUT} ({os.path.getsize(OUTPUT) / 1e6:.2f} MB)')


if __name__ == '__main__':
    main()
