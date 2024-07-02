import numpy as np
import os
import time
import pprint
import jax.numpy as jnp
import mbirjax
import mbirjax.plot_utils as pu
import demo_utils
import pprint
pp = pprint.PrettyPrinter(indent=4)

if __name__ == "__main__":
    print('This script is a demonstration of the preprocessing module of NSI dataset. Demo functionality includes:\
    \n\t * downloading NSI dataset from specified urls;\
    \n\t * Loading object scans, blank scan, dark scan, view angles, and MBIRJAX geometry parameters;\
    \n\t * Computing sinogram from object scan, blank scan, and dark scan images;\
    \n\t * Computing a 3D reconstruction from the sinogram using MBIRJAX;\
    \n\t * Displaying the results.\n')

    # ###################### User defined params. Change the parameters below for your own use case.
    output_path = './output/nsi_demo/' # path to store output recon images
    os.makedirs(output_path, exist_ok=True) # mkdir if directory does not exist

    # ##### params for dataset downloading. User may change these parameters for their own datasets.
    # An example NSI dataset (tarball) will be downloaded from `dataset_url`, and saved to `download_dir`.
    # url to NSI dataset.
    dataset_url = 'https://engineering.purdue.edu/~bouman/data_repository/data/demo_data_nsi.tgz'
    # destination path to download and extract the NSI data and metadata.
    download_dir = './demo_data/'
    # Path to NSI scan directory.
    _, dataset_dir = demo_utils.download_and_extract_tar(dataset_url, download_dir)
    # for testing user prompt in NSI preprocessing function
    # dataset_dir = "/depot/bouman/data/share_conebeam_data/Autoinjection-Full-LowRes/Vertical-0.5mmTin"
  
    # #### preprocessing parameters
    downsample_factor = [4, 4] # downsample factor of scan images along detector rows and detector columns.
    subsample_view_factor = 1 # view subsample factor.
    
    # #### recon parameters
    sharpness=0.0
    # ###################### End of parameters

    # ###########################################################################
    # NSI preprocess: obtain sinogram, sino weights, angles, and geometry params
    # ###########################################################################
    print("\n********************************************************************************",
          "\n** Load scan images, angles, geometry params, and defective pixel information **",
          "\n********************************************************************************")
    obj_scan, blank_scan, dark_scan, angles, geo_params_jax, defective_pixel_list = \
            mbirjax.preprocess_NSI.load_scans_and_params(dataset_dir,
                                                         downsample_factor=downsample_factor, 
                                                         subsample_view_factor=subsample_view_factor)
    

    print("MBIRJAX geometry paramemters:")
    pp.pprint(geo_params_jax)
    print('obj_scan shape = ', obj_scan.shape)
    print('blank_scan shape = ', blank_scan.shape)
    print('dark_scan shape = ', dark_scan.shape)

    print("\n*******************************************************",
          "\n********** Compute sinogram from scan images **********",
          "\n*******************************************************")
    sino, defective_pixel_list = \
            mbirjax.preprocess.transmission_CT_compute_sino(obj_scan, blank_scan, dark_scan,
                                                            defective_pixel_list
                                                           )
    
    # delete scan images to optimize memory usage
    del obj_scan, blank_scan, dark_scan

    print("\n*******************************************************",
          "\n********* Interpolate defective sino entries **********",
          "\n*******************************************************")
    sino, defective_pixel_list = mbirjax.preprocess.interpolate_defective_pixels(sino, defective_pixel_list)

    print("\n*******************************************************",
          "\n************** Correct background offset **************",
          "\n*******************************************************")
    background_offset = mbirjax.preprocess.calc_background_offset(sino)
    print("background_offset = ", background_offset)
    sino = sino - background_offset

    print("\n*******************************************************",
          "\n**** Rotate sino images w.r.t. rotation axis tilt *****",
          "\n*******************************************************")
    sino = mbirjax.preprocess.correct_det_rotation(sino, det_rotation=geo_params_jax["det_rotation"])
   
    print("\n*******************************************************",
          "\n***************** Set up MBIRJAX model ****************",
          "\n*******************************************************")
    # ConeBeamModel constructor
    ct_model = mbirjax.ConeBeamModel(sinogram_shape=geo_params_jax["sinogram_shape"], 
                                     angles=angles, 
                                     source_detector_dist=geo_params_jax["source_detector_dist"], 
                                     source_iso_dist=geo_params_jax["source_iso_dist"],
                                    )
    
    # Set additonal geometry arguments
    ct_model.set_params(det_row_offset=geo_params_jax["det_row_offset"],
                        det_channel_offset=geo_params_jax["det_channel_offset"],
                        delta_det_channel=geo_params_jax["delta_det_channel"],
                        delta_det_row=geo_params_jax["delta_det_row"])

    # Set reconstruction parameter values
    ct_model.set_params(sharpness=sharpness, verbose=1)
    
    # Print out model parameters
    ct_model.print_params()
     
    print("\n*******************************************************",
          "\n************** Calculate sinogram weights *************",
          "\n*******************************************************")
    weights = ct_model.gen_weights(sino, weight_type='transmission_root')
     
    print("\n*******************************************************",
          "\n************** Perform VCD reconstruction *************",
          "\n*******************************************************")
    # ##########################
    # Perform VCD reconstruction
    time0 = time.time()

    recon, recon_params = ct_model.recon(sino, weights=weights)

    recon.block_until_ready()
    elapsed = time.time() - time0
    print('Elapsed time for recon is {:.3f} seconds'.format(elapsed))
    # ##########################

    # Print out parameters used in recon
    pprint.pprint(recon_params._asdict())
    
    np.save(os.path.join(output_path, "recon.npy"), recon)

    # change the image data shape to (slices, rows, cols), so that the rotation axis points up when viewing the coronal/sagittal slices with slice_viewer
    recon = np.transpose(recon, (2,1,0))
    recon = recon[:,:,::-1]
   
    vmin = 0
    vmax = downsample_factor[0]*0.008
    # Display results
    pu.slice_viewer(recon, vmin=0, vmax=vmax, slice_axis=0, slice_label='Axial Slice', title='MBIRJAX recon')
    pu.slice_viewer(recon, vmin=0, vmax=vmax, slice_axis=1, slice_label='Coronal Slice', title='MBIRJAX recon')
    pu.slice_viewer(recon, vmin=0, vmax=vmax, slice_axis=2, slice_label='Sagittal Slice', title='MBIRJAX recon')
