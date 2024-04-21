import types

# The order and content of these dictionaries must match the signatures of the corresponding tests below
_forward_model_defaults_dict = {
    'angles': None,
    'sinogram_shape': None,
    'delta_det_channel': 1.0,
    'delta_det_row': 1.0,
    'det_channel_offset': 0.0,
    'sigma_y': 1.0
}

_recon_model_defaults_dict = {
    'prox_recon': None,
    'num_recon_rows': None,
    'num_recon_cols': None,
    'num_recon_slices': None,
    'delta_pixel_recon': 1.0,
    'sigma_x': 1.0,
    'sigma_p': 1.0,
    'p': 2.0,
    'q': 1.2,
    'T': 1.0,
    'b': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
}

_reconstruction_defaults_dict = {
    'auto_regularize_flag': True,
    'proximal_map_flag': False,
    'positivity_flag': False,
    'initialization': 'zero',
    'snr_db': 30.0,
    'sharpness': 0.0,
    'max_resolutions': None,
    'num_iterations': 10,
    'granularity': [1, 8, 64, 256],
    'partition_sequence': [0, 1, 2, 3, 1, 2, 3, 2, 3, 3],
    'verbose': 0
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
    return types.SimpleNamespace(**recon_defaults_dict)

