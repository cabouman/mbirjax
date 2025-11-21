### This script is for testing the offset correction flag of the nsi preprocessing
### Two reconstructions would be generated with and without the offset correction

import numpy as np
import time
import pprint
import os
import jax
import jax.numpy as jnp
import mbirjax as mj
import mbirjax.preprocess as mjp

pp = pprint.PrettyPrinter(indent=4)

if __name__ == "__main__":
    print('Offset correction experiment.')

    # Dataset path
    dataset_dir = '/depot/bouman/data/Lilly/NSI_sample_1'

    # Preprocessing parameters
    downsample_rate = [4, 4]
    subsample_view_factor = 8

    # Recon parameters
    sharpness = 1.0
    verbose = 2

    print("\n************** NSI dataset preprocessing without offset correction **************")
    sino, cone_beam_params, optional_params = \
        mjp.nsi.compute_sino_and_params(dataset_dir, downsample_factor=downsample_rate,
                                        subsample_view_factor=subsample_view_factor, offset_correction=True)

    print("\n***************** Set up MBIRJAX model ****************")
    ct_model = mj.ConeBeamModel(**cone_beam_params)
    ct_model.set_params(**optional_params)
    ct_model.set_params(sharpness=sharpness, verbose=verbose, positivity_flag=True)
    weights_trans = ct_model.gen_weights(sino, weight_type='transmission_root')
    ct_model.print_params()

    recon_with_correction, _ = ct_model.recon(sinogram=sino, weights=weights_trans)

    print("\n************** NSI dataset preprocessing with data correction **************")
    sino, cone_beam_params, optional_params = \
        mjp.nsi.compute_sino_and_params(dataset_dir, downsample_factor=downsample_rate,
                                        subsample_view_factor=subsample_view_factor, offset_correction=False)

    print("\n***************** Set up MBIRJAX model ****************")
    ct_model = mj.ConeBeamModel(**cone_beam_params)
    ct_model.set_params(**optional_params)
    ct_model.set_params(sharpness=sharpness, verbose=verbose, positivity_flag=True)
    weights_trans = ct_model.gen_weights(sino, weight_type='transmission_root')
    ct_model.print_params()

    recon_without_correction, _ = ct_model.recon(sinogram=sino, weights=weights_trans)

    mj.slice_viewer(recon_with_correction, recon_without_correction,
                    title='Comparison on reconstructions with and without offset correction',
                    slice_label=['recon with correction', 'recon without correction'])
