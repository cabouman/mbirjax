from dataclasses import dataclass
from typing import Any, Literal
import re
from pathlib import Path
import copy

FILE_FORMAT_NUMBER = 1.0  # The format number should be changed if the file format changes.

# Update to include new geometries that should be included in the tests suite
_geometry_types_for_tests = ['parallel', 'cone', 'translation']

# The order and content of these dictionaries must match the headings and list of dicts below
# The second entry in each case indicates if changing that parameter should trigger a recompile


# Dataclass for parameter storage
@dataclass
class Param:
    val: Any
    recompile_flag: bool = True

    def __repr__(self):
        return f"Param(val={self.val}, recompile_flag={self.recompile_flag})"


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

all_param_keys = list(recon_defaults_dict.keys())


# Utility function to update ParamNames = Literal[...] in parameter_handler.py
def update_param_literal(verify_match_and_exit=False):
    """
    Update the ParamNames = Literal[...] line in parameter_handler.py with the current all_param_keys list.

    - Supports multi-line definitions.
    - If the new and existing definitions match, exits without changes.
    - Otherwise, displays both versions and prompts for confirmation.

    Args:
        verify_match_and_exit (bool): If False, then make the update if needed.  If True, then return status only.

    Returns:
        True if the existing Literal matches the new, otherwise False.
    """
    param_file = Path(__file__).parent / "parameter_handler.py"

    print('Checking for updates to parameter names in ParameterHandler...')

    try:
        content = param_file.read_text()
    except FileNotFoundError:
        print(f"❌ Could not find {param_file}")
        return False

    key_groups = []
    group_size = 4
    for i in range(0, len(all_param_keys), group_size):
        key_groups.append(all_param_keys[i:i+group_size])
    new_param_name = ""
    for group in key_groups:
        new_param_name += "    "
        new_param_name += ", ".join(f"'{key}'" for key in group)
        new_param_name += ",\n"
    new_param_name = f"ParamNames = Literal[\n{new_param_name}]"

    param_name_pattern = re.compile(r"ParamNames\s*=\s*Literal\[[^\]]*?\]", re.DOTALL)
    match_for_name = param_name_pattern.search(content)

    if not match_for_name:
        print("❌ ParamNames = Literal[...] not found in file.")
        return False

    old_param_name = match_for_name.group(0)

    param_names_match = old_param_name.strip() == new_param_name.strip()

    if param_names_match:
        print("✅ ParamNames are already up to date.")
        return True

    if verify_match_and_exit:
        return False

    print('Proposed changes shown below:  additions in green, subtractions in red.')
    if not param_names_match:
        highlight_differences(old_param_name, new_param_name)

    confirm = input("Replace old entries with new ones? (y/n): ").strip().lower()
    if confirm == 'y':
        content = content[:match_for_name.start()] + new_param_name + content[match_for_name.end():]
        param_file.write_text(content)
        print(f"✅ Updated ParamNames in {param_file}")
        return True
    else:
        print("❌ Update cancelled.")
        return False


def get_default_params():
    return copy.deepcopy(recon_defaults_dict)

import difflib
def highlight_differences(string1, string2):
    differ = difflib.Differ()
    diff = differ.compare(string1, string2)
    highlighted_diff = []
    for line in diff:
        if line.startswith('- '): # Highlight string1 differences
            highlighted_diff.append('\033[91m' + line[-1] + '\033[0m')
        elif line.startswith('+ '): # Highlight string2 differences
            highlighted_diff.append('\033[92m' + line[-1] + '\033[0m')
        else:
            highlighted_diff.append(line[-1])
    print(''.join(highlighted_diff))
    # Example usage string1 = "Hello world" string2 = "Hello Python" highlighted = highlight_differences(string1, string2) print(highlighted)



# Script utility to update ParamNames Literal in parameter_handler.py
if __name__ == "__main__":
    update_param_literal()
