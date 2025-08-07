import mbirjax as mj
import mbirjax.preprocess as mjp
import numpy as np
import os

def main():
    # Define geometry
    source_det_dist_mm = 190
    source_iso_dist_mm = 70
    det_pixel_pitch_mm = 75 / 1000
    x_space_mm = 11.4
    z_space_mm = 14

    # Define detector size
    num_det_rows = 1936
    num_det_channels = 3064

    # Define object parameters
    object_width_mm = 22
    object_thickness_mm = 2.15

    # Set recon parameters
    sharpness = 1.0

    # Calculate physical parameters in ALU
    # Note: 1 ALU = detector pixel pitch at iso
    ALU_per_mm = source_det_dist_mm / (source_iso_dist_mm * det_pixel_pitch_mm)
    source_iso_dist_ALU = source_iso_dist_mm * ALU_per_mm
    source_det_dist_ALU = source_iso_dist_ALU
    x_spacing_ALU = x_space_mm * ALU_per_mm
    z_spacing_ALU = z_space_mm * ALU_per_mm
    half_angle_rad = np.arctan2(max(num_det_rows, num_det_channels) / 2.0, source_iso_dist_ALU)
    object_width_ALU = object_width_mm * ALU_per_mm
    object_thickness_ALU = object_thickness_mm * ALU_per_mm

    # Print out important values in ALU
    print("ALU per mm:", ALU_per_mm)
    print("Detector height and width (ALU):", num_det_rows, num_det_channels)
    print("Source to iso distance (ALU):", source_iso_dist_ALU)
    print("x,z spacing (ALU):", x_spacing_ALU, z_spacing_ALU)
    print("Half cone angle (deg):", np.rad2deg(half_angle_rad))
    print("Object width (ALU):", object_width_ALU)
    print("Object thickness (ALU):", object_thickness_ALU)

    # Generate translation vectors
    translation_vectors = np.array([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, -z_spacing_ALU],
                                    [x_spacing_ALU, 0.0, -z_spacing_ALU],
                                    [2*x_spacing_ALU, 0.0, -z_spacing_ALU],
                                    [-x_spacing_ALU, 0.0, -z_spacing_ALU],
                                    [-2*x_spacing_ALU, 0.0, -z_spacing_ALU],
                                    [x_spacing_ALU, 0.0, 0.0],
                                    [2*x_spacing_ALU, 0.0, 0.0],
                                    [-x_spacing_ALU, 0.0, 0.0],
                                    [-2*x_spacing_ALU, 0.0, 0.0],
                                    [0.0, 0.0, z_spacing_ALU],
                                    [x_spacing_ALU, 0.0, z_spacing_ALU],
                                    [2*x_spacing_ALU, 0.0, z_spacing_ALU],
                                    [-x_spacing_ALU, 0.0, z_spacing_ALU],
                                    [-2*x_spacing_ALU, 0.0, z_spacing_ALU]])

    # Download data
    dataset_url = '/depot/bouman/data/Translation/purdue_p.tgz'
    download_dir = './projection_data/'
    dataset_dir = mj.download_and_extract(dataset_url, download_dir)

    # Load obj scan, blank scan, and dark scan
    obj_scan_path = os.path.join(dataset_dir, 'purdue_p_object_scan.tif')
    blank_scan_path = os.path.join(dataset_dir, 'purdue_p_blank_scan.tif')
    dark_scan_path = os.path.join(dataset_dir, 'purdue_p_dark_scan.tif')

    obj_scan = mjp.read_scan_img(obj_scan_path)
    blank_scan = mjp.read_scan_img(blank_scan_path)
    dark_scan = mjp.read_scan_img(dark_scan_path)

    blank_scan = blank_scan[None, :, :]
    dark_scan = dark_scan[None, :, :]

    # Crop out defective rows
    crop_pixel_bottom = 86
    obj_scan, blank_scan, dark_scan, _ = mjp.crop_view_data(
        obj_scan, blank_scan, dark_scan,
        crop_pixels_sides=0, crop_pixels_top=0, crop_pixels_bottom=crop_pixel_bottom,
        defective_pixel_array=())

    # Compute sinogram
    sino = mjp.compute_sino_transmission(obj_scan, blank_scan, dark_scan)

    # Compute sinogram shape
    sino_shape = sino.shape

    # Initialize model for reconstruction.
    tct_model = mj.TranslationModel(sino_shape, translation_vectors, source_detector_dist=source_det_dist_ALU, source_iso_dist=source_iso_dist_ALU)
    tct_model.set_params(sharpness=sharpness)
    recon_shape = tct_model.get_params('recon_shape')

    # Change row pitch
    # tct_model.set_params(delta_recon_row=object_thickness_ALU / 3)
    tct_model.set_params(qggmrf_nbr_wts=[1.0, 1.0, 0.1])

    # Print model parameters and display translation array
    tct_model.print_params()
    mj.display_translation_vectors(translation_vectors, recon_shape)

    # View sinogram
    mj.slice_viewer(sino, slice_axis=0, title='Original sinogram', slice_label='View')

    # Generate weights array - for an initial reconstruction, use weights = None, then modify if needed.
    weights = None

    # Perform MBIR reconstruction
    recon, recon_params = tct_model.recon(sino, init_recon=0, weights=weights, max_iterations=15)

    # Display Results
    mj.slice_viewer(recon.transpose(0, 2, 1), vmin=0, vmax=0.1, title='MBIR reconstruction', slice_axis=0)

    # Save as animated gifs
    mj.save_volume_as_gif(recon, "mbir_recon.gif", vmin=0, vmax=1)


if __name__ == '__main__':
    main()
