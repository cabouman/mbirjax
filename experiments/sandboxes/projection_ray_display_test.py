import numpy as np
import mbirjax as mj
import mbirjax.preprocess as mjp


if __name__ == '__main__':

    ##############################################
    # Sets user selectable parameters
    ##############################################

    # Set geometry parameters
    geometry_type = 'cone'  # 'cone' or 'parallel'
    num_object_rows = 128
    num_object_slices = 64
    magnification = 2.0
    cone_angle = (15/180)*np.pi     # cone angle in radians:  50/180 corresponds to 50 degrees

    angle_candidates = np.array([0, 10, 45, 90]) / 180 * np.pi # jnp.linspace(start_angle, end_angle, num_views, endpoint=False)
    num_candidate_views = len(angle_candidates)

    ####################################################
    # Calculate function parameters from user parameters
    ####################################################

    # Create reference object
    print('Creating phantom')
    reference_object = np.zeros((num_object_rows, num_object_rows, num_object_slices))
    row_quarter = reference_object.shape[0] // 4
    col_quarter = reference_object.shape[1] // 4
    reference_object[row_quarter:-row_quarter, col_quarter] = 1
    reference_object[row_quarter, col_quarter:2*col_quarter] = 1
    mjp.show_image_with_projection_rays(reference_object[:, :, 0], rotation_angles_rad=angle_candidates, title='Reference Object with Selected Projection Angles')

    # Set parameters for the problem size - you can vary these, but if you make num_det_rows very small relative to
    # channels, then the generated phantom may not have an interior.
    num_views = num_candidate_views
    num_det_rows = reference_object.shape[2]
    num_det_channels = reference_object.shape[0]
    sinogram_shape = (num_views, num_det_rows, num_det_channels)

    # For cone beam geometry, we need to describe the distances source to detector and source to rotation axis.
    # np.Inf is an allowable value, in which case this is essentially parallel beam
    source_detector_dist = (1.0/np.tan(cone_angle/2.0)) * (num_det_channels/2)
    source_iso_dist = source_detector_dist / magnification

    # Compute view angles

    # Create the model to contain all the geometry information
    ct_model = mjp.get_ct_model(geometry_type, sinogram_shape, angle_candidates, source_detector_dist, source_iso_dist)

    sino = ct_model.forward_project(reference_object)
    mj.slice_viewer(sino, vmax=2, slice_axis=0, title='Reference object sinogram')