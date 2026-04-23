"""
Hyperspectral Dehydration & Rehydration
---------------------------------------

Examples 3(a) + 3(b) demonstrate the use of dehydration and rehydration for fast hyperspectral reconstruction.
A simulated hyperspectral neutron dataset containing three materials (Ni, Cu, and Al) is used for the purpose.
This script - example 3(a) - performs dehydration followed by MBIR and then exports the dehydrated reconstructions.
"""

import os
import numpy as np
import time
import mbirjax as mj

from mbirjax.hsnt import generate_hyper_data, dehydrate, create_hsnt_metadata, export_hsnt_data_hdf5


def main():
    start_time = time.time()
    
    # Set output path and dataset name
    output_path = './processed_data/'  # Path to export data
    dataset_name = "hsnt_dehydrated_recons"  # Name of the output dataset
    os.makedirs(output_path, exist_ok=True)

    # Simulation parameters
    num_angles = 16  # Number of view angles
    detector_rows = 64  # Number of rows in the detector
    detector_columns = 64  # Number of columns in the detector
    dosage_rate = 300  # Neutron dosage rate
    material_density = {"Ni": 0.25, "Cu": 0.25, "Al": 0.75}  # Define material density (vol. fraction)
    dataset_type = 'attenuation'  # Choose between 'attenuation' or 'transmission'

    # Fast hyperspectral reconstruction parameters
    num_materials = 3  # Number of materials
    recon_snr_db = 20  # Assumed SNR for the reconstruction
    verbose = 2  # Verbosity level

    # Fix seed for random number generation
    np.random.seed(129)

    # Load theoretical linear attenuation coefficients for Ni, Cu, and Al
    material_basis_path = './binaries/'
    filename = os.path.join(material_basis_path, 'material_basis.npy')
    material_basis = np.load(filename)

    # Generate simulated noisy hyperspectral projection data
    [hsnt_data, angles, _] = generate_hyper_data(material_basis,
                                                 num_angles=num_angles,
                                                 detector_rows=detector_rows,
                                                 detector_columns=detector_columns,
                                                 dosage_rate=dosage_rate,
                                                 material_density=material_density,
                                                 verbose=verbose)

    # MBIR model setup
    mj_model = mj.ParallelBeamModel((num_angles, detector_rows, detector_columns), angles)
    mj_model.set_params(snr_db=recon_snr_db, verbose=0)

    # Perform dehydration
    hsnt_dehydrated = dehydrate(hsnt_data, dataset_type=dataset_type, num_materials=num_materials, verbose=verbose)

    # Unpack dehydrated data
    [subspace_data, subspace_basis, dataset_type] = hsnt_dehydrated
    
    # Perform MBIR
    subspace_dimension = subspace_data.shape[-1]
    subspace_recons = []
    for idx in range(subspace_dimension):
        print("Reconstructing data for subspace index: " + str(idx))
        subspace_recon, _ = mj_model.recon(subspace_data[:, :, :, idx])
        subspace_recons.append(subspace_recon)
    subspace_recons = np.moveaxis(np.array(subspace_recons), 0, -1)
    
    # Repack dehydrated data
    hsnt_dehydrated_recons = [subspace_recons, subspace_basis, dataset_type]
    
    # Export dehydrated reconstructions into HDF5 file
    metadata = create_hsnt_metadata(dataset_name=dataset_name)
    filename = os.path.join(output_path, dataset_name + ".h5")
    export_hsnt_data_hdf5(filename, hsnt_dehydrated_recons, metadata)

    print('Total time elapsed: ', time.time() - start_time, ' seconds')


if __name__ == "__main__":
    main()
