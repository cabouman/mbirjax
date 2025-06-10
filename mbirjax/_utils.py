from dataclasses import dataclass
from typing import Any, Literal
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
    type: str
    recompile_flag: bool = True

    def __repr__(self):
        return f"Param(val={self.val}, type={self.type}, recompile_flag={self.recompile_flag})"


# The order and content of these dictionaries must match the headings and list of dicts below
_forward_model_defaults_dict = {
    'geometry_type': Param(None, 'str', False),  # The geometry type should never change during a recon.
    'file_format': Param(FILE_FORMAT_NUMBER, 'float', False),
    'sinogram_shape': Param(None, 'tuple', True),
    'delta_det_channel': Param(1.0, 'float', True),
    'delta_det_row': Param(1.0, 'float', True),
    'det_row_offset': Param(0.0, 'float', True),
    'det_channel_offset': Param(0.0, 'float', True),
    'sigma_y': Param(1.0, 'float', False),
}

_recon_model_defaults_dict = {
    'recon_shape': Param(None, 'tuple', True),
    'delta_voxel': Param(None, 'float', True),
    'sigma_x': Param(1.0, 'float', False),
    'sigma_prox': Param(1.0, 'float', False),
    'p': Param(2.0, 'float', False),
    'q': Param(1.2, 'float', False),
    'T': Param(1.0, 'float', False),
    'qggmrf_nbr_wts': Param([1.0, 1.0, 1.0], 'list | tuple', False),  # Order is row_nbr_wt, col_nbr_wt, slice_nbr_wt
}

_reconstruction_defaults_dict = {
    'auto_regularize_flag': Param(True, 'bool', False),
    'positivity_flag': Param(False, 'bool', False),
    'snr_db': Param(30.0, 'float', False),
    'sharpness': Param(1.0, 'float', False),
    'granularity': Param([1, 2, 4, 8, 16, 32, 64, 128, 256], 'list | tuple', False),
    'partition_sequence': Param([0, 2, 4, 6, 7], 'list | tuple', False),
    'verbose': Param(1, 'int', False),
    'use_gpu': Param('automatic', 'str', True),  # Possible values are 'automatic', 'full', 'sinograms', 'projections', 'none'
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


# Utility function to update ParamName = Literal[...] in parameter_handler.py
import re
from pathlib import Path


def update_param_literal(verify_match_and_exit=False):
    """
    Update the ParamName = Literal[...] line in parameter_handler.py with the current all_param_keys list.

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

    pattern = re.compile(r"ParamName\s*=\s*Literal\[[^\]]*?\]", re.DOTALL)
    match = pattern.search(content)

    if not match:
        print("❌ ParamName = Literal[...] not found in file.")
        return False

    old_literal = match.group(0)
    key_groups = []
    group_size = 4
    for i in range(0, len(all_param_keys), group_size):
        key_groups.append(all_param_keys[i:i+group_size])
    formatted_items = ""
    formated_typed_dict = 'class _ParamValues(TypedDict, total=False):\n'
    for group in key_groups:
        formatted_items += "    "
        formatted_items += ", ".join(f"'{k}'" for k in group)
        formatted_items += ",\n"
        for k in group:
            param = recon_defaults_dict[k]
            type = param.type
            formated_typed_dict += "    " + k + ": " + type + "\n"
    formated_typed_dict += "    # end _ParamValues"
    new_literal = f"ParamName = Literal[\n{formatted_items}]"

    if old_literal.strip() == new_literal.strip():
        print("✅ ParamName Literal is already up to date.")
        return True

    if verify_match_and_exit:
        return False

    print("Old Literal:")
    print(old_literal)
    print("New Literal:")
    print(new_literal)
    confirm = input("Replace old ParamName Literal with new one? (y/n): ").strip().lower()
    if confirm == 'y':
        new_content = content[:match.start()] + new_literal + content[match.end():]
        param_file.write_text(new_content)
        print(f"✅ Updated ParamName Literal in {param_file}")
        return True
    else:
        print("❌ Update cancelled.")
        return False


def get_default_params():
    return copy.deepcopy(recon_defaults_dict)


# Script utility to update ParamName Literal in parameter_handler.py
if __name__ == "__main__":
    update_param_literal()
