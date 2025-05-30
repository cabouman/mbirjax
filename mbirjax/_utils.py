import types
import copy

FILE_FORMAT_NUMBER = 1.0  # The format number should be changed if the file format changes.

# Update to include new geometries that should be included in the tests suite
_geometry_types_for_tests = ['parallel', 'cone']

# The order and content of these dictionaries must match the headings and list of dicts below
# The second entry in each case indicates if changing that parameter should trigger a recompile
_forward_model_defaults_dict = {

    'geometry_type': {'val': None,  'recompile_flag': False},  # The geometry type should never change during a recon.
    'file_format': {'val': FILE_FORMAT_NUMBER, 'recompile_flag': False},
    'sinogram_shape': {'val': None, 'recompile_flag': True},
    'delta_det_channel': {'val': 1.0, 'recompile_flag': True},
    'delta_det_row': {'val': 1.0, 'recompile_flag': True},
    'det_row_offset': {'val': 0.0, 'recompile_flag': True},
    'det_channel_offset': {'val': 0.0, 'recompile_flag': True},
    'sigma_y': {'val': 1.0, 'recompile_flag': False},
}

_recon_model_defaults_dict = {
    'recon_shape': {'val': None, 'recompile_flag': True},
    'delta_voxel': {'val': None, 'recompile_flag': True},
    'sigma_x': {'val': 1.0, 'recompile_flag': False},
    'sigma_prox': {'val': 1.0, 'recompile_flag': False},
    'p': {'val': 2.0, 'recompile_flag': False},
    'q': {'val': 1.2, 'recompile_flag': False},
    'T': {'val': 1.0, 'recompile_flag': False},
    'qggmrf_nbr_wts': {'val': [1.0, 1.0, 1.0], 'recompile_flag': False},  # Order is row_nbr_wt, col_nbr_wt, slice_nbr_wt
}

_reconstruction_defaults_dict = {
    'auto_regularize_flag': {'val': True, 'recompile_flag': False},
    'positivity_flag': {'val': False, 'recompile_flag': False},
    'snr_db': {'val': 30.0, 'recompile_flag': False},
    'sharpness': {'val': 1.0, 'recompile_flag': False},
    'granularity': {'val': [1, 2, 4, 8, 16, 32, 64, 128, 256], 'recompile_flag': False},
    'partition_sequence': {'val': [0, 2, 4, 6, 7], 'recompile_flag': False},
    'verbose': {'val': 1, 'recompile_flag': False},
    'use_gpu': {'val': 'automatic', 'recompile_flag': True}  # Possible values are 'automatic', 'full', 'sinograms', 'projections', 'none'
}

# These headings should match the dictionaries
headings = ['Forward model parameters', 'Recon parameters', 'Reconstruction parameters']

dicts = [_forward_model_defaults_dict,
         _recon_model_defaults_dict,
         _reconstruction_defaults_dict]

recon_defaults_dict = dict()
for d in dicts:
    recon_defaults_dict = {**recon_defaults_dict, **d}


def get_default_params():
    return copy.deepcopy(recon_defaults_dict)

