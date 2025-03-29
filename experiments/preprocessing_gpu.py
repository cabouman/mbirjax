import time
import mbirjax


if __name__ == "__main__":

    # ###################### User defined params. Change the parameters below for your own use case.

    # ##### params for dataset downloading. User may change these parameters for their own datasets.
    # An example NSI dataset (tarball) will be downloaded from `dataset_url`, and saved to `download_dir`.
    # url to NSI dataset.
    dataset_dir = '/depot/bouman/data/nsi_demo_data/demo_nsi_vert_no_metal_all_views'
    # dataset_dir = '/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirjax_applications/nsi/demo_data/demo_data_nsi'

    downsample_factor = [1, 1]  # downsample factor of scan images along detector rows and detector columns.
    subsample_view_factor = 1  # view subsample factor.

    # #### recon parameters
    sharpness = 1.0
    # ###################### End of parameters

    print("\n*******************************************************",
          "\n************** NSI dataset preprocessing **************",
          "\n*******************************************************")
    time0 = time.time()
    crop_pixels_sides = 32
    crop_pixels_top = 250
    crop_pixels_bottom = 20
    sino, cone_beam_params, optional_params = \
        mbirjax.preprocess.nsi.compute_sino_and_params(dataset_dir,
                                                       downsample_factor=downsample_factor,
                                                       subsample_view_factor=subsample_view_factor,
                                                       crop_pixels_sides=crop_pixels_sides,
                                                       crop_pixels_top=crop_pixels_top,
                                                       crop_pixels_bottom=crop_pixels_bottom)

    mbirjax.slice_viewer(sino, slice_axis=0)

    print("\n*******************************************************",
          "\n***************** Set up MBIRJAX model ****************",
          "\n*******************************************************")
    # ConeBeamModel constructor
    ct_model = mbirjax.ConeBeamModel(**cone_beam_params)

    # Set additional geometry arguments
    ct_model.set_params(**optional_params)

    # Set reconstruction parameter values
    ct_model.set_params(sharpness=sharpness, verbose=1)

    # Print out model parameters
    print(f'Preprocessing finished, model initialized. Total time: {time.time() - time0:.2f} seconds.')