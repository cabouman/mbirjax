import warnings
import copy
import logging
import io
from collections.abc import Iterable, Sized
from typing import Literal, Union, Any, TextIO, overload
import os

import jax.numpy as jnp
import numpy as np
from ruamel.yaml import YAML

from mbirjax._utils import Param
import mbirjax as mj


ParamNames = Literal[
    'geometry_type', 'file_format', 'sinogram_shape', 'delta_det_channel',
    'delta_det_row', 'det_row_offset', 'det_channel_offset', 'sigma_y',
    'recon_shape', 'delta_voxel', 'sigma_x', 'sigma_prox',
    'p', 'q', 'T', 'qggmrf_nbr_wts',
    'auto_regularize_flag', 'positivity_flag', 'snr_db', 'sharpness',
    'granularity', 'partition_sequence', 'verbose', 'use_gpu',
]


class ParameterHandler:
    array_prefix = ':ARRAY:'

    def __init__(self):

        self.params = mj._utils.get_default_params()
        self.logger = None
        self.log_buffer = None

    def setup_logger(self, *, logfile_path: str = "./logs/recon.log", print_logs: bool = True):
        """
        Initialize self.logger and self.log_buffer.

        Args:
            logfile_path: Path to the log file. If None or empty, file logging is skipped.
            verbosity: 0 -> WARNING, 1 -> INFO, 2+ -> DEBUG
            print_logs: If True, emit logs to console.

        Raises:
            Exception: If logfile_path directory cannot be created.
        """
        # Map verbosity to logging level
        verbose = self.get_params('verbose')
        if verbose < 1:
            level = logging.WARNING
        elif verbose < 2:
            level = logging.INFO
        else:
            level = logging.DEBUG

        # Configure logger
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(level)
        # Close and remove any existing handlers to prevent leaked file descriptors
        for h in list(logger.handlers):
            try:
                h.flush()
            finally:
                h.close()
                logger.removeHandler(h)

        # In-memory buffer handler (always enabled)
        self.log_buffer = io.StringIO()
        buffer_handler = logging.StreamHandler(self.log_buffer)
        buffer_handler.setLevel(level)
        buffer_formatter = logging.Formatter('%(message)s')
        buffer_handler.setFormatter(buffer_formatter)
        logger.addHandler(buffer_handler)

        # Console handler
        if print_logs:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_formatter = logging.Formatter('%(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        # File handler (optional)
        if logfile_path:
            mj.makedirs(logfile_path)
            file_handler = logging.FileHandler(logfile_path, mode='w')
            file_handler.setLevel(level)
            file_formatter = logging.Formatter('%(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        self.logger = logger

    def print_params(self):
        """
        Print the current parameter values in the model.

        This method prints all parameters stored in the model's internal dictionary. If the model's
        verbosity level is less than 3, it prints only the parameter names and values. If verbosity
        is 3 or higher, it also includes the `recompile_flag` status for each parameter.

        Example:
            >>> ct_model = mj.ParallelBeamModel(sinogram_shape, angles)
            >>> ct_model.set_params(sharpness=0.7, recon_shape=(128, 128, 128))
            >>> ct_model.print_params()
        """
        verbose, view_params_name = self.get_params(['verbose', 'view_params_name'])
        print("----")
        for key, entry in self.params.items():
            if verbose < 3 and key == view_params_name:
                continue
            param_val = entry.val
            if verbose < 3:
                print("{} = {}".format(key, param_val))
            else:
                recompile_flag = entry.recompile_flag
                print("{} = {}, recompile_flag = {}".format(key, param_val, recompile_flag))
        print("----")
        self.set_params(use_gpu=self.get_params('use_gpu'))

    @staticmethod
    def convert_arrays_to_strings(cur_params):
        """
        Replaces any jax or numpy arrays in cur_params with a flattened string representation and the array shape.
        Args:
            cur_params (dict): Parameter dictionary

        Returns:
            dict: The same dictionary with arrays replaced by strings.
        """
        for key, entry in cur_params.items():
            param_val = entry.val
            if isinstance(param_val, (jnp.ndarray, np.ndarray)):
                # Get the array values, then flatten them and put them in a string.
                cur_array = np.asarray(param_val)
                formatted_string = " ".join(f"{x:.7f}" for x in cur_array.flatten())
                # Include a prefix for identification upon reading
                new_val = ParameterHandler.array_prefix + formatted_string
                cur_params[key].val = new_val
                cur_params[key].shape = param_val.shape

            cur_params[key].val = ParameterHandler.normalize_scalar(param_val)

        return cur_params

    @staticmethod
    def convert_strings_to_arrays(cur_params):
        """
        Convert the string representation of an array back to an array.
        Args:
            cur_params (dict): Parameter dictionary

        Returns:
            dict: The same dictionary with array strings replaced by arrays.
        """
        array_prefix = ParameterHandler.array_prefix
        for key, entry in cur_params.items():
            param_val = entry.val
            # CHeck for a string with the array marker as prefix.
            if type(param_val) is str and param_val[0:len(array_prefix)] == array_prefix:
                # Strip the prefix, then remove the delimiters
                param_str = param_val[len(array_prefix):]
                clean_str = param_str.replace('[', '').replace(']', '').strip()
                # Read to a flat array, then reshape
                new_val = jnp.array(np.fromstring(clean_str + ' ', sep=' '))
                new_shape = cur_params[key].shape
                # Save the value and remove the 'shape' key, which is needed only for the yaml file.
                cur_params[key].val = new_val.reshape(new_shape)
                del cur_params[key].shape

        return cur_params

    @staticmethod
    def normalize_scalar(val):
        """
        Convert numpy/jax scalar types to Python native types.
        Also recursively normalize lists or tuples of scalars.
        Leave strings, bools, None, and arrays untouched.

        Args:
            val: Any parameter value.

        Returns:
            Cleaned version suitable for serialization and comparison.
        """
        if isinstance(val, (np.generic, jnp.generic)):
            return val.item()
        elif isinstance(val, (list, tuple)):
            return type(val)(ParameterHandler.normalize_scalar(v) for v in val)
        return val

    @staticmethod
    def serialize_parameter(param_obj):
        """
        Convert a Param object to a YAML-safe dictionary by serializing arrays and normalizing scalars.

        Args:
            param_obj (Param): A parameter object with fields `val` and `recompile_flag`.

        Returns:
            dict: A dictionary with serialized and normalized data safe for YAML dumping.
        """
        val = param_obj.val

        if isinstance(val, (jnp.ndarray, np.ndarray)):
            cur_array = np.asarray(val)
            formatted_string = " ".join(f"{x:.7f}" for x in cur_array.flatten())
            serialized_val = ParameterHandler.array_prefix + formatted_string
            return {
                'val': serialized_val,
                'shape': cur_array.shape,
                'recompile_flag': param_obj.recompile_flag
            }

        val = ParameterHandler.normalize_scalar(val)

        # Convert lists to tuples for consistency
        if isinstance(val, list):
            val = tuple(val)

        return {
            'val': val,
            'recompile_flag': param_obj.recompile_flag
        }

    @staticmethod
    def deserialize_parameter(entry):
        """
        Convert a dictionary loaded from YAML into a Param object.

        Args:
            entry (dict): A dictionary with 'val' and 'recompile_flag' (and optionally 'shape').

        Returns:
            Param: A reconstructed Param object with normalized and typed values.
        """
        val = entry['val']
        if isinstance(val, str) and val.startswith(ParameterHandler.array_prefix):
            # Keep val as-is for convert_strings_to_arrays
            param = Param(val=val, recompile_flag=entry.get('recompile_flag', True))
            if 'shape' in entry:
                param.shape = tuple(entry['shape'])
            return param
        else:
            val = ParameterHandler.normalize_scalar(val)
            if isinstance(val, list):
                val = tuple(val)
            return Param(val=val, recompile_flag=entry.get('recompile_flag', True))

    @staticmethod
    def is_flat_iterable(x):
        return isinstance(x, Iterable) and isinstance(x, Sized) and not isinstance(x, (str, bytes))

    @staticmethod
    def compare_flat_iterables(v1, v2, atol=1e-6):
        """
        Verify that 2 iterables (tuples or lists, etc) have the same length and entries, up to atol.
        Args:
            v1 (Sized Iterable):  First iterable
            v2 (Sized Iterable):  Second iterable
            atol (float, optional): Absolute floating point tolerance for equality.  Defaults to 1e-6.

        Returns:

        """
        if len(v1) != len(v2):
            return False
        for a, b in zip(v1, v2):
            if (isinstance(a, (int, float, np.generic, jnp.generic)) and
                    isinstance(b, (int, float, np.generic, jnp.generic))):
                if not abs(float(a) - float(b)) <= atol:
                    return False
            else:
                if a != b:
                    return False
        return True

    @staticmethod
    def compare_parameter_handlers(ph1, ph2, atol=1e-6, verbose=False):
        """
        Compare the parameters of two ParameterHandler instances for equality.

        Args:
            ph1 (ParameterHandler): First instance.
            ph2 (ParameterHandler): Second instance.
            atol (float): Absolute tolerance for float/array comparison.
            verbose (bool): If True, print mismatch details.

        Returns:
            bool: True if all parameters match within tolerances, False otherwise.
        """
        keys1 = set(ph1.params.keys())
        keys2 = set(ph2.params.keys())
        if keys1 != keys2:
            if verbose:
                print("Parameter key mismatch:")
                print("Only in ph1:", keys1 - keys2)
                print("Only in ph2:", keys2 - keys1)
            return False

        for key in keys1:
            val1 = ph1.params[key].val
            val2 = ph2.params[key].val

            if isinstance(val1, (np.ndarray, jnp.ndarray)) and isinstance(val2, (np.ndarray, jnp.ndarray)):
                equal = np.allclose(np.asarray(val1), np.asarray(val2), atol=atol)
            elif (isinstance(val1, (int, float, np.generic, jnp.generic)) and
                  isinstance(val2, (int, float, np.generic, jnp.generic))):
                equal = abs(float(val1) - float(val2)) <= atol
            elif ParameterHandler.is_flat_iterable(val1) and ParameterHandler.is_flat_iterable(val2):
                equal = ParameterHandler.compare_flat_iterables(val1, val2, atol)
            else:
                equal = val1 == val2

            if not equal:
                if verbose:
                    print(f"Mismatch in key '{key}': {val1} != {val2}")
                return False

        return True

    @staticmethod
    def save_params(params, filename=None):
        """
        Serialize parameters to YAML. If filename is provided, write to file; otherwise return YAML text.

        Args:
            params (dict): The parameters dict from a TomographyModel
            filename (str or None): Path to save YAML file (must end in .yml/.yaml). If None, return YAML string.

        Returns:
            None if filename was provided; otherwise, YAML-formatted string of parameters.

        Raises:
            ValueError: If filename is invalid.
        """
        # Prepare parameter dict
        output_params = copy.deepcopy(params)
        for key in output_params:
            output_params[key] = ParameterHandler.serialize_parameter(output_params[key])

        yaml_writer = YAML()
        yaml_writer.default_flow_style = False

        if filename:
            if not filename.lower().endswith(('.yml', '.yaml')):
                raise ValueError(f"Filename must end in .yaml or .yml: {filename}")
            # Ensure output directory exists
            mj.makedirs(filename)
            with open(filename, 'w') as file:
                yaml_writer.dump(output_params, file)
            return None
        else:
            stream = io.StringIO()
            yaml_writer.dump(output_params, stream)
            return stream.getvalue()


    @staticmethod
    @overload
    def load_param_dict(source: str) -> dict: ...

    @staticmethod
    @overload
    def load_param_dict(source: TextIO, required_param_names=None, values_only=True) -> tuple: ...

    @staticmethod
    @overload
    def load_param_dict(source: io.StringIO, required_param_names=None, values_only=True) -> tuple: ...

    @staticmethod
    def load_param_dict(source: Union[str, TextIO, io.StringIO], required_param_names=None, values_only=True) -> tuple:
        """
        Load parameters from a YAML file, a YAML string, or a file-like object.

        Args:
            source: A filename (str), a YAML string (str), or a file-like object.  Filename must end in .yml or .yaml
            required_param_names (list of strings): List of parameter names that are required for a class.
            values_only (bool):  If True, then extract and return the values of each entry only.

        Returns:
            required_params (dict): Dictionary of required parameter entries.
            params (dict): Dictionary of all other parameters.
        """
        # Determine file type
        yaml_stream: TextIO

        if isinstance(source, str):
            if os.path.exists(source):
                if not source.lower().endswith(('.yml', '.yaml')):
                    raise ValueError("Filename must end in .yml or .yaml")
                with open(source, 'r') as f:
                    contents = f.read()
                    yaml_stream = io.StringIO(contents)
            else:
                yaml_stream = io.StringIO(source)
        elif hasattr(source, 'read'):
            yaml_stream = source
        else:
            raise TypeError("Invalid source type for load_param_dict. Must be str or file-like object.")
        yaml_reader = YAML(typ="safe")
        param_dict = yaml_reader.load(yaml_stream)

        param_dict = {key: ParameterHandler.deserialize_parameter(val) for key, val in param_dict.items()}
        param_dict = ParameterHandler.convert_strings_to_arrays(param_dict)

        # Convert any lists to tuples for consistency with save
        for key in param_dict.keys():
            if isinstance(param_dict[key].val, list):
                param_dict[key].val = tuple(param_dict[key].val)
        for key in param_dict:
            param_dict[key].val = ParameterHandler.normalize_scalar(param_dict[key].val)
        return ParameterHandler.get_required_params_from_dict(param_dict, required_param_names=required_param_names,
                                                              values_only=values_only)

    @staticmethod
    def get_required_params_from_dict(param_dict, required_param_names=None, values_only=True):
        # Separate the required parameters into a new dict and delete those entries from the original
        required_params = dict()
        param_dict = param_dict.copy()
        for name in required_param_names:
            required_params[name] = param_dict[name]
            del param_dict[name]

        if values_only:
            for key in required_params.keys():
                required_params[key] = required_params[key].val
            for key in param_dict.keys():
                param_dict[key] = param_dict[key].val
        return required_params, param_dict

    def set_params(self, no_warning=False, no_compile=False, **kwargs):
        """
        Update parameters using keyword arguments.

        This method updates internal model parameters. If any key geometry-related parameters
        are modified, it triggers recompilation of the projector system unless suppressed
        via the `no_compile` flag.

        Args:
            no_warning (bool, optional): If True, disables validity checking and warning messages. Defaults to False.
            no_compile (bool, optional): If True, suppresses projector recompilation after updates. Defaults to False.
            **kwargs: Arbitrary keyword arguments specifying parameter names and values to update.

        Returns:
            bool: True if projector recompilation is required and not suppressed by `no_compile`,
            otherwise False.

        Example:
            >>> import mbirjax as mj
            >>> ct_model = mj.ParallelBeamModel(sinogram_shape, angles)
            >>> ct_model.set_params(recon_shape=(128, 128, 128), sharpness=0.7)
        """
        # Get initial geometry parameters
        recompile = False
        regularization_parameter_change = False
        meta_parameter_change = False

        # Set all the given parameters
        for key, val in kwargs.items():
            # Default to forcing a recompile for new parameters
            recompile_flag = True

            if key in self.params.keys():
                recompile_flag = self.params[key].recompile_flag
            elif not no_warning:  # Check if this is a valid parameter.  This is disabled for initialization.
                error_message = '{} is not a recognized parameter'.format(key)
                error_message += '\nValid parameters are: \n'
                for valid_key in self.params.keys():
                    error_message += '   {}\n'.format(valid_key)
                raise ValueError(error_message)

            clean_val = ParameterHandler.normalize_scalar(val)
            new_entry = Param(clean_val, recompile_flag)
            self.params[key] = new_entry

            # Handle special cases
            if recompile_flag:
                recompile = True
            elif key in ["sigma_y", "sigma_x", "sigma_prox"]:
                regularization_parameter_change = True
            elif key in ["sharpness", "snr_db"]:
                meta_parameter_change = True

        # Check for valid parameters
        if not no_warning:
            self.verify_valid_params()

        # Handle case if any regularization parameter changed
        if regularization_parameter_change:
            if not no_warning:
                self.set_params(auto_regularize_flag=False)
                warnings.warn('You are directly setting regularization parameters, sigma_x, sigma_y or sigma_prox. '
                              'This is an advanced feature that will disable auto-regularization.')

        # Handle case if any meta regularization parameter changed
        if meta_parameter_change:
            if self.get_params('auto_regularize_flag') is False:
                self.set_params(auto_regularize_flag=True)
                if not no_warning:
                    warnings.warn('You have re-enabled auto-regularization by setting sharpness or snr_db. '
                                  'It was previously disabled')

        # Return a flag to signify recompiling
        recompile_flag = False
        if recompile and not no_compile:
            recompile_flag = True

        return recompile_flag

    @staticmethod
    def get_params_from_dict(param_dict, parameter_names: Union[str, list[str]]):
        """
        Get the values of the listed parameter names from the supplied dict.
        Raises an exception if a parameter name is not defined in parameters.

        Args:
            param_dict (dict): The dictionary of parameters
            parameter_names (str or list of str): String or list of strings

        Returns:
            Single value or list of values
        """
        if isinstance(parameter_names, str):
            if parameter_names in param_dict.keys():
                value = param_dict[parameter_names].val
            else:
                raise NameError('"{}" is not a recognized argument'.format(parameter_names))
            return value
        values = []
        for name in parameter_names:
            if name in param_dict.keys():
                values.append(param_dict[name].val)
            else:
                raise NameError('"{}" is not a recognized argument'.format(name))
        return values

    def get_params(self, parameter_names: Union[ParamNames, list[ParamNames]]) -> Any:
        """
        Get the values of the listed parameter names from the internal parameter dictionary.

        This method retrieves the current values of one or more parameters managed by the model.

        Args:
            parameter_names (str or list of str): Name of a parameter, or a list of parameter names.

        Returns:
            Any or list: Single parameter value if a string is passed, or a list of values if a list is passed.

        Raises:
            NameError: If any of the provided parameter names are not recognized.

        Example:
            >>> sharpness = model.get_params('sharpness')
            >>> recon_shape, sharpness = model.get_params(['recon_shape', 'sharpness'])
        """
        param_values = ParameterHandler.get_params_from_dict(self.params, parameter_names)
        return param_values

    def get_magnification(self):
        """
        Compute the scale factor from a voxel at iso (at the origin on the center of rotation) to
        its projection on the detector.  For parallel beam, this is 1, but it may be parameter-dependent
        for other geometries.

        Returns:
            (float): magnification
        """
        raise NotImplementedError('get_magnification is not implemented.')

    def verify_valid_params(self):
        """
        Verify any conditions that must be satisfied among parameters for correct projections.

        Subclasses of TomographyModel should call super().verify_valid_params() before checking any
        subclass-specific conditions.

        Note:
            Raises ValueError for invalid parameters.
        """

        sinogram_shape = self.get_params('sinogram_shape')
        if len(sinogram_shape) != 3:
            error_message = "sinogram_shape must be (views, rows, channels). \n"
            error_message += "Got {} for sinogram shape.".format(sinogram_shape)
            raise ValueError(error_message)

        geometry_type = self.get_params('geometry_type')
        if geometry_type != str(type(self)):
            raise ValueError('Parameters are associated with {}, but the current model is {}'.format(geometry_type, str(type(self))))
