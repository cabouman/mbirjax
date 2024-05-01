import types
import copy

# The order and content of these dictionaries must match the signatures of the corresponding tests below
# The second entry in each case indicates if changing that parameter should trigger a recompile
_forward_model_defaults_dict = {

    'sinogram_shape': {'val': None, 'recompile_flag': True},
    'delta_det_channel': {'val': 1.0, 'recompile_flag': True},
    'delta_det_row': {'val': 1.0, 'recompile_flag': True},
    'det_channel_offset': {'val': 0.0, 'recompile_flag': False},
    'sigma_y': {'val': 1.0, 'recompile_flag': False},
}

_recon_model_defaults_dict = {
    'prox_recon': {'val': None, 'recompile_flag': False},
    'num_recon_rows': {'val': None, 'recompile_flag': True},
    'num_recon_cols': {'val': None, 'recompile_flag': True},
    'num_recon_slices': {'val': None, 'recompile_flag': True},
    'delta_pixel_recon': {'val': 1.0, 'recompile_flag': False},
    'sigma_x': {'val': 1.0, 'recompile_flag': False},
    'sigma_p': {'val': 1.0, 'recompile_flag': False},
    'magnification': {'val': 1.0, 'recompile_flag': False},
    'p': {'val': 2.0, 'recompile_flag': False},
    'q': {'val': 1.2, 'recompile_flag': False},
    'T': {'val': 1.0, 'recompile_flag': False},
    'b': {'val': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'recompile_flag': False},
}

_reconstruction_defaults_dict = {
    'auto_regularize_flag': {'val': True, 'recompile_flag': False},
    'positivity_flag': {'val': False, 'recompile_flag': False},
    'snr_db': {'val': 30.0, 'recompile_flag': False},
    'sharpness': {'val': 0.0, 'recompile_flag': False},
    'granularity': {'val': [1, 8, 64, 256], 'recompile_flag': False},
    'partition_sequence': {'val': [0, 1, 2, 3, 2, 3, 2, 3], 'recompile_flag': False},
    'verbose': {'val': 0, 'recompile_flag': False},
    'voxel_batch_size': {'val': None, 'recompile_flag': True},
    'view_batch_size': {'val': None, 'recompile_flag': True}
}

headings = ['Geometry params', 'Recon params', 'Init params', 'Noise params', 'QGGMRF params', 'Sys params',
            'Misc params']

dicts = [_forward_model_defaults_dict,
         _recon_model_defaults_dict,
         _reconstruction_defaults_dict]

recon_defaults_dict = dict()
for d in dicts:
    recon_defaults_dict = {**recon_defaults_dict, **d}


def get_default_params():
    return copy.deepcopy(recon_defaults_dict)

