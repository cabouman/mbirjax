import jax.numpy as jnp
import yaml
import mbirjax._utils as utils

class ParameterHandler():
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

    def save_params(self, fname='param_dict.npy', binaries=False):
        """Save parameter dict to numpy/pickle file/yaml file"""
        output_params = self.params.copy()
        if binaries is False:
            # Wipe the binaries before saving.
            output_params['weights'] = None
            output_params['prox_recon'] = None
            output_params['init_proj'] = None
            if not jnp.isscalar(output_params['init_recon']):
                output_params['init_recon'] = None
        # Determine file type
        if fname[-4:] == '.npy':
            jnp.save(fname, output_params)
        elif fname[-4:] == '.yml' or fname[-5:] == '.yaml':
            # Work through all the parameters by group, with a heading for each group
            with open(fname, 'w') as file:
                for heading, dic in zip(utils.headings, utils.dicts):
                    file.write('# ' + heading + '\n')
                    for key in dic.keys():
                        val = self.params[key]
                        file.write(key + ': ' + str(val) + '\n')
        else:
            raise ValueError('Invalid file type for saving parameters: ' + fname)

    def load_params(self, fname):
        """Load parameter dict from numpy/pickle file/yaml file, and merge into instance params"""
        # Determine file type
        if fname[-4:] == '.npy':
            read_dict = jnp.load(fname, allow_pickle=True).item()
            self.params = utils.get_default_params()
            self.set_params(**read_dict)
        elif fname[-4:] == '.yml' or fname[-5:] == '.yaml':
            with open(fname) as file:
                try:
                    params = yaml.safe_load(file)
                    self.params = utils.get_default_params()
                    self.set_params(**params)
                except yaml.YAMLError as exc:
                    print(exc)

    def set_params(self, no_warning=False, no_compile=False, **kwargs):
        """
        Updates parameters using keyword arguments.
        After setting parameters, it checks if key geometry-related parameters have changed and, if so, recompiles the projectors.

        Args:
            no_warning (bool, optional, default=False): This is used internally to allow for some initial parameter setting.
            no_compile (bool, optional, default=False): Prevent (re)compiling the projectors.  Used for initialization.
            **kwargs: Arbitrary keyword arguments where keys are parameter names and values are the new parameter values.

        Raises:
            NameError: If any key provided in kwargs is not a recognized parameter.
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
            new_entry = {'val': val, 'recompile_flag': recompile_flag}
            self.params[key] = new_entry

            # Handle special cases
            if recompile_flag:
                recompile = True
            elif key in ["sigma_y", "sigma_x", "sigma_p"]:
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
                warnings.warn('You are directly setting regularization parameters, sigma_x, sigma_y or sigma_p. '
                              'This is an advanced feature that will disable auto-regularization.')

        # Handle case if any meta regularization parameter changed
        if meta_parameter_change:
            if self.get_params('auto_regularize_flag') is False:
                self.set_params(auto_regularize_flag=True)
                if not no_warning:
                    warnings.warn('You have re-enabled auto-regularization by setting sharpness or snr_db. '
                                  'It was previously disabled')

        # Compare the two outputs
        if recompile and not no_compile:
            self.create_projectors()

    def get_params(self, parameter_names):
        """
        Get the values of the listed parameter names.
        Raises an exception if a parameter name is not defined in parameters.

        Args:
            parameter_names: String or list of strings

        Returns:
            Single value or list of values
        """
        if isinstance(parameter_names, str):
            if parameter_names in self.params.keys():
                value = self.params[parameter_names]['val']
            else:
                raise NameError('"{}" not a recognized argument'.format(parameter_names))
            return value
        values = []
        for name in parameter_names:
            if name in self.params.keys():
                values.append(self.params[name]['val'])
            else:
                raise NameError('"{}" not a recognized argument'.format(name))
        return values

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
