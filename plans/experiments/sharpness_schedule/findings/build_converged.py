"""Build the converged-streaks page: converged_template.html + figures/ ->
plans/sharpness_schedule/converged_streaks.html (self-contained; images embedded).

Refuses to build while [[SECTION]] placeholders remain in the template, so a
skeleton can never be published by accident.

Run:  python build_converged.py
"""

import base64
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(_HERE, 'converged_template.html')
FIGURES_DIR = os.path.join(_HERE, 'figures')
OUTPUT = os.path.normpath(os.path.join(
    _HERE, '..', '..', '..', 'sharpness_schedule', 'converged_streaks.html'))

FIGURES = {
    'converged_convergence': 'converged_convergence.png',
    'converged_error_panels': 'converged_error_panels.png',
    'sweep_partition': 'sweep_partition.png',
    'pad_vs_unpad_bga': 'pad_vs_unpad_bga.png',
    'e1_longpair': 'e1_longpair.png',
    'e1_increment': 'e1_increment.png',
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
