import jax.numpy as jnp
import numpy as np
from ruamel.yaml import YAML
import mbirjax._utils as utils
import warnings


class ParameterHandler():
    array_prefix = ':ARRAY:'

    def __init__(self):

        self.params = utils.get_default_params()

    def print_params(self):
        """
        Prints out the parameters of the model.
        """
        verbose = self.get_params('verbose')
        print("----")
        for key, entry in self.params.items():
            if verbose < 2 and key == 'view_params_array':
                continue
            param_val = entry.get('val')
            recompile_flag = entry.get('recompile_flag')
            print("{} = {}, recompile_flag = {}".format(key, param_val, recompile_flag))
        print("----")

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
            param_val = entry.get('val')
            if type(param_val) in [type(jnp.ones(1)), type(np.ones(1))]:
                # Get the array values, then flatten them and put them in a string.
                cur_array = np.array(param_val)
                formatted_string = " ".join(f"{x:.7f}" for x in cur_array.flatten())
                # Include a prefix for identification upon reading
                new_val = ParameterHandler.array_prefix + formatted_string
                cur_params[key]['val'] = new_val
                cur_params[key]['shape'] = param_val.shape

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
            param_val = entry.get('val')
            # CHeck for a string with the array marker as prefix.
            if type(param_val) is str and param_val[0:len(array_prefix)] == array_prefix:
                # Strip the prefix, then remove the delimiters
                param_str = param_val[len(array_prefix):]
                clean_str = param_str.replace('[', '').replace(']', '').strip()
                # Read to a flat array, then reshape
                new_val = jnp.array(np.fromstring(clean_str + ' ', sep=' '))
                new_shape = cur_params[key]['shape']
                # Save the value and remove the 'shape' key, which is needed only for the yaml file.
                cur_params[key]['val'] = new_val.reshape(new_shape)
                del cur_params[key]['shape']

        return cur_params

    def save_params(self, filename):
        """
        Save parameters to yaml file.

        Args:
            filename (str): Path to file to store the parameter dictionary.  Must end in .yml or .yaml

        Returns:
            Nothing but creates or overwrites the specified file.
        """
        output_params = ParameterHandler.convert_arrays_to_strings(self.params.copy())

        # Determine file type
        if filename[-4:] == '.yml' or filename[-5:] == '.yaml':
            # Save the full parameter dictionary
            with open(filename, 'w') as file:
                yaml = YAML()
                yaml.default_flow_style = False
                yaml.dump(output_params, file)
        else:
            raise ValueError('Filename must end in .yaml or .yml: ' + filename)

    @staticmethod
    def load_param_dict(filename, values_only=True):
        """
        Load parameter dictionary from yaml file.

        Args:
            filename (str): Path to load to store the parameter dictionary.  Must end in .yml or .yaml
            values_only (bool):  If True, then extract and return the values of each entry only.

        Returns:
            dict: The dictionary of paramters.
        """
        # Determine file type
        if filename[-4:] == '.yml' or filename[-5:] == '.yaml':
            # Save the full parameter dictionary
            with open(filename, 'r') as file:
                yaml = YAML(typ="safe")
                params = yaml.load(file)
                params = ParameterHandler.convert_strings_to_arrays(params)

        keys = params.keys()
        if 'recon_shape' in keys:
            params['recon_shape']['val'] = tuple(params['recon_shape']['val'])
        if values_only:
            for key in keys:
                params[key] = params[key]['val']
        return params

    def load_params(self, filename):
        """
        Load parameter dictionary from yaml file.
        Args:
            filename (str): Path to load to store the parameter dictionary.  Must end in .yml or .yaml

        Returns:
            Nothing, but the parameters are set from the file.
        """
        # Determine file type
        self.params = ParameterHandler.load_param_dict(filename)

    def set_params(self, no_warning=False, no_compile=False, **kwargs):
        """
        Updates parameters using keyword arguments.
        After setting parameters, it checks if key geometry-related parameters have changed and, if so, recompiles the projectors.

        Args:
            no_warning (bool, optional, default=False): This is used internally to allow for some initial parameter setting.
            no_compile (bool, optional, default=False): Prevent (re)compiling the projectors.  Used for initialization.
            **kwargs: Arbitrary keyword arguments where keys are parameter names and values are the new parameter values.
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
                recompile_flag = self.params[key]['recompile_flag']
            elif not no_warning:
                error_message = '{} is not a recognized parameter'.format(key)
                error_message += '\nValid parameters are: \n'
                for valid_key in self.params.keys():
                    error_message += '   {}\n'.format(valid_key)
                raise ValueError(error_message)

            new_entry = {'val': val, 'recompile_flag': recompile_flag}
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
            self.set_params(auto_regularize_flag=False)
            if not no_warning:
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
    def get_params_from_dict(param_dict, parameter_names):
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
                value = param_dict[parameter_names]['val']
            else:
                raise NameError('"{}" not a recognized argument'.format(parameter_names))
            return value
        values = []
        for name in parameter_names:
            if name in param_dict.keys():
                values.append(param_dict[name]['val'])
            else:
                raise NameError('"{}" not a recognized argument'.format(name))
        return values

    def get_params(self, parameter_names):
        """
        Get the values of the listed parameter names from the internal parameter dict.
        Raises an exception if a parameter name is not defined in parameters.

        Args:
            parameter_names (str or list of str): String or list of strings

        Returns:
            Single value or list of values
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
