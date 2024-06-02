import mbirjax
import numpy as np
import os

if __name__ == "__main__":
    # Change to the directory containing the data.
    directory = '/scratch/gilbreth/buzzard/BLS-00637_dyparkinson/reconstructions/'

    filename = 'recon.npz'
    full_path = os.path.join(directory, filename)

    with np.load(full_path) as data:
        recon = data['arr_0']

    mbirjax.slice_viewer(recon, vmin=-5, vmax=10)

