import numpy as np
import mbirjax as mj
import matplotlib.pyplot as plt


def cos_angle(diff1, diff2):
    inner_prod = np.sum(diff1 * diff2, axis=(0, 1))
    cos_theta = inner_prod / (np.linalg.norm(diff1, axis=(0, 1)) * np.linalg.norm(diff2, axis=(0, 1)))
    return cos_theta


if __name__ == "__main__":

    # #### recon parameters
    sharpness = 3.0
    num_iterations = 8

    # Choose the geometry type
    model_type = 'cone'
    object_type = 'cube'

    # Set parameters for the problem size - you can vary these, but if you make num_det_rows very small relative to
    # channels, then the generated phantom may not have an interior.
    num_views = 128
    num_det_rows = 40
    num_det_channels = 128

    source_detector_dist = 12 * num_det_channels
    source_iso_dist = 4 * num_det_channels
    sinogram_shape = (num_views, num_det_rows, num_det_channels)
    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
    ct_model = mj.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist,
                                source_iso_dist=source_iso_dist)

    ct_model.set_params(sharpness=sharpness, verbose=1)
    ct_model.set_params(delta_voxel=0.28, det_row_offset=0.4)
    phantom = mj.gen_cube_phantom(ct_model.get_params('recon_shape'))
    print('Creating sinogram')
    sinogram = ct_model.forward_project(phantom)
    sinogram = np.asarray(sinogram)

    # Get the fully converged version
    filename = 'fully_converged_sharpness_{}.npy'.format(sharpness)
    try:
        converged_result = np.load(filename)
    except FileNotFoundError:
        converged_result, _ = ct_model.recon(sinogram, max_iterations=100, stop_threshold_change_pct=1e-6)
        np.save(filename, converged_result)
        exit(0)

    # ct_model.set_params(partition_sequence=[1,], max_overrelaxation=1)

    recons = [np.zeros_like(phantom)]
    init_recon = None
    for j in range(num_iterations):
        recon0, recon_dict0 = ct_model.recon(sinogram, init_recon=init_recon, first_iteration=j, max_iterations=j+1, stop_threshold_change_pct=1e-6)
        # mj.slice_viewer(recon0, slice_axis=2)
        recons.append(recon0)
        init_recon = recon0

    # Determine convergence to baseline for select slices
    plt.figure(0)
    slice_indices = [3, 10, 13, 16, 19]
    for index in slice_indices:
        norms = [np.linalg.norm(recons[j+1][:, :, index] - converged_result[:, :, index]) for j in range(num_iterations-1)]
        log_norms = np.log10(norms)
        print('log_10 || recon_j - recon_infty || for slice {}:'.format(index))
        print(log_norms)
        plt.plot(log_norms)

    plt.legend(['norms for slice {}'.format(slice_indices[s]) for s in range(len(slice_indices))])
    plt.show()

    plt.figure(1)
    plt.title('norms of consecutive delta_recons by slice')
    for j in range(num_iterations-1):
        diff1 = recons[j+1] - recons[j]

        delta_norm = np.linalg.norm(diff1, axis=(0, 1))
        plt.figure(1)
        plt.plot(delta_norm)

        for k in [2, 3, 4, 5]:
            if j + k < len(recons):
                diff = recons[j + k] - recons[j + k - 1]
                cos_theta = cos_angle(diff, diff1)
                plt.figure(k)
                plt.plot(cos_theta)
    plt.figure(1)
    plt.legend(['norm of delta_recon{}'.format(j) for j in range(num_iterations)])

    for k in [2, 3, 4, 5]:
        plt.figure(k)
        plt.ylim(-1, 1)
        plt.title('cos of angle between diff (j + {}) and diff j'.format(k - 1, j))

    error_sino0 = ct_model.forward_project(recon0) - sinogram


    hessian = ct_model.compute_hessian_diagonal()

    # Use the power method to determine the slice dependent average intensity of D^{-1} A^T A
    # This gives an estimate of which slices overshoot and by how much.
    # This approach assumes the intensity is constant on each slice, so uses the same
    # 1D profile for each voxel cylinder on each iteration.  A random set of voxel cylinders is fixed.
    # In practice, the weight matrix would have to be included also.
    eigenmode = np.random.random(ct_model.get_params('recon_shape'))
    eigenmode *= eigenmode < 0.01
    for j in range(10):
        sinogram = ct_model.forward_project(eigenmode)
        eigenmode = ct_model.back_project(sinogram)
        eigenmode /= hessian
        eigenmode /= np.linalg.norm(eigenmode)
    eigenmode /= np.linalg.norm(eigenmode)
    grad_by_slice = np.linalg.norm(eigenmode, axis=(0, 1))

    plt.figure(10)
    plt.plot(grad_by_slice)
    plt.title('Norm of top eigenmode by slice')
    vrange = 0.001
    eigenmode = eigenmode / np.amax(np.abs(eigenmode)) * vrange
    title = r'Left: Recon, Right: Top eigenmode of $D^{-1} A^T A$'
    title += '\nSharpness = {}, num_iterations = {}'.format(sharpness, num_iterations)
    title += '\nNarrow intensity window to highlight artifacts'
    mj.slice_viewer(recon0.transpose((2, 0, 1)), eigenmode.transpose((2, 0, 1)), slice_axis=0,
                    vmin=-vrange, vmax=vrange, title=title)

