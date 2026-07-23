### Display saved IQ evaluation reconstructions with the mbirjax slice viewer.
#
# Loads fdk_recon.h5 and mbir_recon.h5 from output/<case>/<tag>/ and shows them
# side by side, one case at a time (close the viewer to advance). Volumes are
# oriented with the rotation axis vertical and large z at the bottom.
#
# Examples:
#   python view_recon.py --all
#   python view_recon.py --all --tag full_res
#   python view_recon.py --case bga_no_hart --tag low_res

import argparse
import os
import numpy as np
import mbirjax as mj

from test_cases import TEST_CASES

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')


def orient(recon):
    # (row, col, slice) -> (slice, col, row): z runs down the screen.
    return np.swapaxes(recon, 0, 2)


def view_case(name, tag):
    out_dir = os.path.join(OUTPUT_DIR, name, tag)
    fdk_path = os.path.join(out_dir, 'fdk_recon.h5')
    mbir_path = os.path.join(out_dir, 'mbir_recon.h5')
    if not (os.path.exists(fdk_path) and os.path.exists(mbir_path)):
        print(f"Skipping '{name}/{tag}': no recons in {out_dir}")
        return
    fdk_recon, _ = mj.import_recon_hdf5(fdk_path)
    mbir_recon, _ = mj.import_recon_hdf5(mbir_path)
    mj.slice_viewer(orient(fdk_recon), orient(mbir_recon), slice_axis=1,
                    slice_label=['FDK', 'MBIR'], title=f'{name} / {tag}: FDK vs MBIR')


def main():
    parser = argparse.ArgumentParser(description='View IQ evaluation reconstructions.')
    parser.add_argument('--case', choices=sorted(TEST_CASES), help='Test case to view')
    parser.add_argument('--all', action='store_true', help='View all test cases in turn')
    parser.add_argument('--tag', default='low_res', help='Output tag to view (default: low_res)')
    args = parser.parse_args()

    if not args.case and not args.all:
        parser.error('Specify --case NAME or --all.')
    names = sorted(TEST_CASES) if args.all else [args.case]
    for name in names:
        view_case(name, args.tag)


if __name__ == '__main__':
    main()
