import mbirjax as mj
import mbirjax.preprocess as mjp
import numpy as np
import os
import pprint
from xrm_utils import *

def main():
    # Download and extract data

    # Load and preprocess data
    obj_scan_path = '/home/yang1581/purdue_p_xrm/25-08-26 Raw Projections'
    blank_scan_path = '/home/yang1581/purdue_p_xrm/25-08-26 Reference Image'
    dark_scan_path = '/home/yang1581/purdue_p_xrm/25-08-26 Black'
    sino, translation_params = compute_sino_and_params(obj_scan_path, blank_scan_path, dark_scan_path)

    pprint.pprint(translation_params)

    # Initialize model for reconstruction.
    tct_model = mj.TranslationModel(**translation_params)
    tct_model.set_params(sharpness=0.0)
    tct_model.set_params(det_row_offset=-26.5)
    # tct_model.set_params(snr_db=36)
    recon_shape = tct_model.get_params('recon_shape')

    # Change row pitch
    # tct_model.set_params(delta_recon_row=object_thickness_ALU / 3)
    tct_model.set_params(qggmrf_nbr_wts=[1.0, 1.0, 0.1])

    # Print model parameters and display translation array
    translation_vectors = translation_params['translation_vectors']
    tct_model.print_params()
    mj.display_translation_vectors(translation_vectors, recon_shape)

    # View sinogram
    mj.slice_viewer(sino, slice_axis=0, title='Original sinogram', slice_label='View')

    # Generate weights array - for an initial reconstruction, use weights = None, then modify if needed.
    weights = None

    # Perform MBIR reconstruction
    recon, recon_params = tct_model.recon(sino, init_recon=0, weights=weights, max_iterations=15)

    # Display Results
    mj.slice_viewer(recon.transpose(0, 2, 1), vmin=0, vmax=0.003, title='MBIR reconstruction', slice_axis=0)

    # Save as animated gifs
    mj.save_volume_as_gif(recon, "mbir_recon.gif", vmin=0, vmax=1)


if __name__ == '__main__':
    main()
