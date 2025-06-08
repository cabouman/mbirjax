from dataclasses import dataclass
from typing import Any
import types
import copy

FILE_FORMAT_NUMBER = 1.0  # The format number should be changed if the file format changes.

# Update to include new geometries that should be included in the tests suite
_geometry_types_for_tests = ['parallel', 'cone']

# The order and content of these dictionaries must match the headings and list of dicts below
# The second entry in each case indicates if changing that parameter should trigger a recompile


# Dataclass for parameter storage
@dataclass
class Param:
    val: Any
    recompile_flag: bool = True


# The order and content of these dictionaries must match the headings and list of dicts below
_forward_model_defaults_dict = {
    'geometry_type': Param(None, False),  # The geometry type should never change during a recon.
    'file_format': Param(FILE_FORMAT_NUMBER, False),
    'sinogram_shape': Param(None, True),
    'delta_det_channel': Param(1.0, True),
    'delta_det_row': Param(1.0, True),
    'det_row_offset': Param(0.0, True),
    'det_channel_offset': Param(0.0, True),
    'sigma_y': Param(1.0, False),
}

_recon_model_defaults_dict = {
    'recon_shape': Param(None, True),
    'delta_voxel': Param(None, True),
    'sigma_x': Param(1.0, False),
    'sigma_prox': Param(1.0, False),
    'p': Param(2.0, False),
    'q': Param(1.2, False),
    'T': Param(1.0, False),
    'qggmrf_nbr_wts': Param([1.0, 1.0, 1.0], False),  # Order is row_nbr_wt, col_nbr_wt, slice_nbr_wt
}

_reconstruction_defaults_dict = {
    'auto_regularize_flag': Param(True, False),
    'positivity_flag': Param(False, False),
    'snr_db': Param(30.0, False),
    'sharpness': Param(1.0, False),
    'granularity': Param([1, 2, 4, 8, 16, 32, 64, 128, 256], False),
    'partition_sequence': Param([0, 2, 4, 6, 7], False),
    'verbose': Param(1, False),
    'use_gpu': Param('automatic', True),  # Possible values are 'automatic', 'full', 'sinograms', 'projections', 'none'
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

