import jax.numpy as jnp


def generate_direct_recon_filter(num_channels, filter_name="ramp"):
    """
    Creates the specified space domain filter of size (2*num_channels - 1).

    Currently supported filters include: \"ramp\", which corresponds to a ramp in frequency domain.

    Args:
        num_channels (int): Number of detector channels in the sinogram.
        filter_name (string, optional): Name of the filter to be generated. Defaults to "ramp."

    Returns:
        filter (jnp): The computed filter (filter.size = 2*num_channels + 1).
    """

    # If you want to add a new filter, place its name into supported_filters, and ...
    # ... create a new if statement with the filter math.
    # TODO:  Anyone who adds a second filter will need to address how to document the set of available filters
    # in a way that is easy to maintain and appears correctly on readthedocs.  Also, any new filters will need
    # to have the proper scaling.
    supported_filters = ["ramp"]

    # Raise error if filter is not supported.
    if filter_name not in supported_filters:
        raise ValueError(f"Unsupported filter. Supported filters are: {', '.join(supported_filters)}.")

    n = jnp.arange(-num_channels + 1, num_channels)  # ex: num_channels = 3, -> n = [-2, -1, 0, 1, 2]

    recon_filter = 0
    if filter_name == "ramp":
        recon_filter = (1 / 2) * jnp.sinc(n) - (1 / 4) * (jnp.sinc(n / 2)) ** 2

    return recon_filter


