import numpy as np
import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mbirjax.plot_utils as pu
import mbirjax.parallel_beam

import jax

import os
import dxchange
import h5py
import tomopy

# for install of h5py, dxchange and tomopy
# conda install h5py
# conda install -c conda-forge dxchange (or pip install dxchange)
# conda install -c conda-forge tomopy

if __name__ == "__main__":

    username = 'buzzard'  # Adjust as needed
    inputSubFolderName = "BLS-00637_dyparkinson"  # this should be the name of the folder in /depot/bouman/data/ that has data

    outputSubfolderName = "reconstructions"  # this can be anything you want, I usually choose the current date

    # Output files will be in /scratch/gilbreth/<username>/<inputSubFolderName>/<outputSubfolderName>

    inputPath = os.path.join("/depot/bouman/data/", inputSubFolderName)
    wheretosave = "/scratch/gilbreth"

    outputPath = os.path.join(wheretosave, username, inputSubFolderName, outputSubfolderName)

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    pickledparamsfile = f'{outputSubfolderName}.pkl'
    filenamelist = os.listdir(inputPath)
    filenamelist.sort()
    for i in range(len(filenamelist) - 1, np.maximum(len(filenamelist) - 1000, -1), -1):
        print(f'{i}: {filenamelist[i]}')

    filename = filenamelist[0]  # update this number with the index of the file you want to process from the directory listing generated in the previous cell

    outputFilename = os.path.join(outputPath, 'recon.npz')  # + os.path.splitext(filename)[0])

    numslices = int(
        dxchange.read_hdf5(os.path.join(inputPath, filename), "/measurement/instrument/detector/dimension_y")[0])
    numrays = int(
        dxchange.read_hdf5(os.path.join(inputPath, filename), "/measurement/instrument/detector/dimension_x")[0])
    pxsize = dxchange.read_hdf5(os.path.join(inputPath, filename), "/measurement/instrument/detector/pixel_size")[
                 0] / 10.0  # /10 to convert units from mm to cm
    numangles = int(
        dxchange.read_hdf5(os.path.join(inputPath, filename), "/process/acquisition/rotation/num_angles")[0])
    propagation_dist = dxchange.read_hdf5(os.path.join(inputPath, filename),
                                          "/measurement/instrument/camera_motor_stack/setup/camera_distance")[1]
    kev = dxchange.read_hdf5(os.path.join(inputPath, filename), "/measurement/instrument/monochromator/energy")[
              0] / 1000
    angularrange = dxchange.read_hdf5(os.path.join(inputPath, filename), "/process/acquisition/rotation/range")[0]

    print(f'{filename}: \nslices: {numslices}, rays: {numrays}, angles: {numangles}, angularrange: {angularrange}, \npxsize: {pxsize * 10000:.3f} um, distance: {propagation_dist:.3f} mm. energy: {kev} keV')
    if kev > 100:
        print('white light mode detected; energy is set to 30 kev for the phase retrieval function')

    sinoused = (-1, 8, 1)  # using the whole numslices will make it run out of memory
    if sinoused[0] < 0:
        sinoused = (int(np.floor(numslices / 2.0) - np.ceil(sinoused[1] / 2.0)),
                    int(np.floor(numslices / 2.0) + np.floor(sinoused[1] / 2.0)), 1)

    tomo, flat, dark, anglelist = dxchange.exchange.read_aps_tomoscan_hdf5(os.path.join(inputPath, filename),
                                                                           sino=(sinoused[0], sinoused[1], sinoused[2]))
    anglelist = -anglelist

    tomo = tomo.astype(np.float32, copy=False)
    flat = flat.astype(np.float32, copy=False)
    dark = dark.astype(np.float32, copy=False)
    tomopy.normalize(tomo, flat, dark, out=tomo, ncore=64)
    tomopy.minus_log(tomo, out=tomo, ncore=64)

    cor = 1265.5
    theshift = int((cor - numrays / 2))
    sinogram = jax.device_put(tomo)

    # View sinogram
    pu.slice_viewer(tomo.transpose((0, 2, 1)), title='Original sinogram')
    sharpness = -1.0

    parallel_model = mbirjax.ParallelBeamModel(sinogram.shape, anglelist)

    # Generate weights array
    weights = parallel_model.gen_weights(sinogram / sinogram.max(), weight_type='transmission_root')

    # Set reconstruction parameter values
    parallel_model.set_params(sharpness=sharpness, det_channel_offset=theshift, verbose=1, positity_flag=True)

    # Print out model parameters
    parallel_model.print_params()

    # ##########################
    # Perform VCD reconstruction
    time0 = time.time()
    recon, recon_params = parallel_model.recon(sinogram, weights=weights, num_iterations=50)

    # scale by pixel size to units of 1/cm
    recon /= pxsize

    recon.block_until_ready()
    elapsed = time.time() - time0
    print('Elapsed time for recon is {:.3f} seconds'.format(elapsed))
    # ##########################

    np.savez_compressed(outputFilename, recon)
    # Display results
    pu.slice_viewer(recon, vmin=-5, vmax=10, title='VCD Recon (right)')

    a = 0

