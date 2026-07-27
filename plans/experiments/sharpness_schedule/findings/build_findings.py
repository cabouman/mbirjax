"""Build the self-contained findings page: replace {{FIG:name}} tokens in
findings_template.html with base64-embedded PNGs from figures/, and write the
result to plans/sharpness_schedule/findings.html (the committed, publishable page --
the depot publish idiom rsyncs *.html only, so images must be embedded).

Run:  python build_findings.py
"""

import base64
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(_HERE, 'findings_template.html')
FIGURES_DIR = os.path.join(_HERE, 'figures')
MATHJAX = os.path.join(_HERE, 'vendor', 'mathjax-tex-svg.js')
OUTPUT = os.path.normpath(os.path.join(
    _HERE, '..', '..', '..', 'sharpness_schedule', 'findings.html'))

# MathJax is EMBEDDED (not CDN-linked) so the published page stays fully
# self-contained; the SVG output renderer needs no font files.  Math is authored
# as \( ... \) inline and $$ ... $$ display in the template.
MATHJAX_CONFIG = ('<script>window.MathJax={tex:{inlineMath:[["\\\\(","\\\\)"]],'
                  'displayMath:[["$$","$$"]]},svg:{fontCache:"none"}};</script>')

# Token name -> figure file in figures/.
FIGURES = {
    'a1_severity': 'a1_severity_axes.png',
    'a2_formation': 'a2_formation_curves.png',
    'a2_panels_center': 'a2_panels_center.png',
    'a2_footprint': 'a2_footprint.png',
    'damp_compare_illustration': 'damp_compare_illustration.png',
    'a2_v2_spectra': 'a2_v2_spectra.png',
    'a2_v2_formation': 'a2_v2_formation.png',
    'a3_fullres_formation': 'a3_fullres_formation.png',
    'two_seed_illustration': 'two_seed_illustration.png',
    'pfz_illustration': 'pfz_illustration.png',
    'footprint_illustration': 'footprint_illustration.png',
    'synth_vs_real': 'synth_vs_real.png',
    'two_seed_damp_compare': 'two_seed_damp_compare.png',
    'hardened_err_xz': 'hardened_err_xz.png',
}


def img_tag(path):
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('ascii')
    return f'<img src="data:image/png;base64,{b64}" alt="{os.path.basename(path)}">'


def main():
    with open(TEMPLATE) as f:
        html = f.read()
    for token, fname in FIGURES.items():
        path = os.path.join(FIGURES_DIR, fname)
        if os.path.exists(path):
            html = html.replace('{{FIG:%s}}' % token, img_tag(path))
        else:
            print(f'WARNING: missing figure {fname}; token left as placeholder box')
            html = html.replace(
                '{{FIG:%s}}' % token,
                f'<div class="pending">figure pending: {fname}</div>')
    leftover = re.findall(r'\{\{FIG:[^}]*\}\}', html)
    if leftover:
        raise SystemExit(f'unreplaced tokens: {leftover}')
    with open(MATHJAX) as f:
        mathjax_src = f.read()
    assert '</script' not in mathjax_src.lower(), 'MathJax bundle would break inline embedding'
    html = html.replace('</body>',
                        MATHJAX_CONFIG + '<script>' + mathjax_src + '</script>\n</body>')
    with open(OUTPUT, 'w') as f:
        f.write(html)
    print(f'wrote {OUTPUT} ({os.path.getsize(OUTPUT) / 1e6:.2f} MB)')


if __name__ == '__main__':
    main()
