import os
import numpy as np
import jax.numpy as jnp
import jax
import time
import matplotlib.pyplot as plt
import gc
import mbirjax.parallel_beam


import io
import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont


def render_string_from_ttfont(
    height: int,
    width: int,
    pad: int,
    text: str,
    ttfont: TTFont,
) -> np.ndarray:
    """
    Render each character in `text` into a (height×width) greyscale image,
    with `pad`-pixel border, using the Times New Roman font you loaded as a
    fontTools.TTFont instance.

    Args:
        height, width:  output image size
        pad:            border thickness (pixels)
        text:           the string to render
        ttfont:         a fontTools.ttLib.TTFont already loaded with your TNR data

    Returns:
        A uint8 numpy array of shape (height, width, len(text)) where
        [:,:,j] is the j-th character’s greyscale bitmap.
    """
    # serialize your TTFont to raw bytes just once
    buf = io.BytesIO()
    ttfont.save(buf)
    font_bytes = buf.getvalue()

    num_chars = len(text)
    tensor = np.zeros((height, width, num_chars), dtype=np.uint8)

    inner_w = width  - 2 * pad
    inner_h = height - 2 * pad
    if inner_w <= 0 or inner_h <= 0:
        raise ValueError("pad too large for given height/width")

    for j, ch in enumerate(text):
        # binary-search for largest size that fits the inner box
        lo, hi, best = 1, max(inner_w, inner_h), 1
        while lo <= hi:
            mid = (lo + hi) // 2
            # load from bytes at this size
            f = ImageFont.truetype(io.BytesIO(font_bytes), mid)
            x0, y0, x1, y1 = f.getbbox(ch)
            cw, chh = x1 - x0, y1 - y0
            if cw <= inner_w and chh <= inner_h:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1

        # render at best size
        font = ImageFont.truetype(io.BytesIO(font_bytes), best)
        x0, y0, x1, y1 = font.getbbox(ch)
        cw, chh = x1 - x0, y1 - y0

        img = Image.new("L", (width, height), color=0)
        draw = ImageDraw.Draw(img)
        xpos = pad + (inner_w - cw) / 2 - x0
        ypos = pad + (inner_h - chh) / 2 - y0
        draw.text((xpos, ypos), ch, fill=255, font=font)

        tensor[:, :, j] = np.array(img)

    tensor = tensor.transpose((2, 0, 1))
    return tensor


if __name__ == "__main__":

    num_det_channels = 100
    skip = 5
    phantom = np.zeros((num_det_channels, 10, num_det_channels))

    source_detector_dist = 1.1 * phantom.shape[1]
    source_iso_dist = source_detector_dist / 2

    # For cone beam reconstruction, we need a little more than 180 degrees for full coverage.
    start_angle = -np.pi / 2  # - np.pi / 2
    end_angle = -np.pi / 2  # + np.pi / 2
    num_det_rows = num_det_channels
    num_views = 1
    sinogram_shape = (num_views, num_det_rows, num_det_channels)
    angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

    ct_model_for_generation = mbirjax.ConeBeamModel(sinogram_shape, angles,
                                                    source_detector_dist=source_detector_dist,
                                                    source_iso_dist=source_iso_dist)

    ct_model_for_generation.set_params(recon_shape=phantom.shape)
    print('Creating sinogram')
    phantom = 0 * phantom
    # phantom[8:45, 5, 12:60] = 1
    phantom[12:18, 5, 12:20] = 1
    phantom[35:41, 5, 12:20] = 1
    sinogram = ct_model_for_generation.forward_project(phantom)
    mbirjax.slice_viewer(sinogram, slice_axis=0)
    # load your TNR file once
    tt = TTFont("/System/Library/Fonts/Supplemental/Times New Roman.ttf")

    # now render without ever mentioning the .ttf path again
    num_det_channels = 400
    pad = num_det_channels // 5
    imgs = render_string_from_ttfont(
        height=num_det_channels,
        width=num_det_channels,
        pad=pad,
        text="2F",
        ttfont=tt
    )
    skip = 5
    imgs = (imgs / 255.0).astype(np.float32)[::-1]
    phantom = np.zeros((num_det_channels, skip * imgs.shape[0], num_det_channels))
    start_ind = 0
    end_ind = start_ind + skip * imgs.shape[0]

    source_detector_dist = 1.5 * phantom.shape[1]
    source_iso_dist = source_detector_dist / 2

    # For cone beam reconstruction, we need a little more than 180 degrees for full coverage.
    start_angle = -np.pi / 2  # - np.pi / 2
    end_angle = -np.pi / 2  # + np.pi / 2
    num_det_rows = num_det_channels
    num_views = 1
    sinogram_shape = (num_views, num_det_rows, num_det_channels)
    angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

    ct_model_for_generation = mbirjax.ConeBeamModel(sinogram_shape, angles,
                                                    source_detector_dist=source_detector_dist,
                                                    source_iso_dist=source_iso_dist)

    ct_model_for_generation.set_params(recon_shape=phantom.shape)
    print('Creating sinogram')
    sinograms = []
    width = 130
    height = 180
    for start in np.arange(start=0, stop=num_det_channels // 2, step=10):
        phantom = 0 * phantom
        # phantom[200:210, 0, 190:200] = 1
        phantom[0:height, start_ind:end_ind:skip, start:start+width] = imgs.transpose((1, 0, 2))[100:100+height, :, 130:130+width]
        # mbirjax.slice_viewer(phantom, slice_axis=1)
        sinogram = ct_model_for_generation.forward_project(phantom)
        sinograms.append(np.asarray(sinogram))
        # mbirjax.slice_viewer(sinogram, slice_axis=0)
        print(start)
        start_x = start
    for start in np.arange(start=0, stop=num_det_channels // 2, step=10):
        phantom = 0 * phantom
        # phantom[200:210, 0, 190:200] = 1
        phantom[start:start+height, start_ind:end_ind:skip, start_x:start_x+width] = imgs.transpose((1, 0, 2))[100:100+height, :, 130:130+width]
        # mbirjax.slice_viewer(phantom, slice_axis=1)
        sinogram = ct_model_for_generation.forward_project(phantom)
        sinograms.append(np.asarray(sinogram))
        # mbirjax.slice_viewer(sinogram, slice_axis=0)
        print(start)
    sinograms = np.concatenate(sinograms, axis=0).transpose((0, 2, 1))
    sinograms = np.concatenate([sinograms, sinograms[::-1]], axis=0)
    mbirjax.slice_viewer(sinograms, slice_axis=0)

    exit(0)
    """
    This is a script to develop, debug, and tune the translation model projector
    """
    # ##########################
    # Initialize sinogram
    num_x_offsets = 17
    num_z_offsets = 17
    num_views = num_x_offsets * num_z_offsets
    num_det_rows = 128
    num_det_channels = 128
    source_detector_distance = 1.1 * num_det_channels
    source_iso_distance = source_detector_distance // 2
    delta_voxel = 1

    sinogram = jnp.zeros((num_views, num_det_rows, num_det_channels))
    offsets = [(x - num_x_offsets // 2, y - num_z_offsets // 2) for x in range(num_x_offsets) for y in range(num_z_offsets)]
    translations_vectors = np.stack(offsets, axis=0)
    # Initialize a random key
    seed_value = np.random.randint(1000000)
    key = jax.random.PRNGKey(seed_value)

    # Set up parallel beam model
    # translation_model = mbirjax.ParallelBeamModel.from_file('params_parallel.yaml')
    translation_model = mbirjax.TranslationModel(sinogram.shape, translations_vectors, source_detector_distance, source_iso_distance)
    translation_model.set_params(delta_voxel=delta_voxel)

    # Generate phantom
    recon_shape = translation_model.get_params('recon_shape')
    num_recon_rows, num_recon_cols, num_recon_slices = recon_shape[:3]
    phantom = np.zeros(recon_shape)
    phantom[10:60, 10:60, 40:80 ] = 1
    phantom[20:50] = 0
    mbirjax.slice_viewer(phantom)

    # Generate indices of pixels
    num_subsets = 1
    indices = mbirjax.gen_pixel_partition(recon_shape, num_subsets)[0]

    # Generate sinogram data
    voxel_values = phantom.reshape((-1,) + recon_shape[2:])[indices]


    print('Starting forward projection')
    sinogram = translation_model.sparse_forward_project(voxel_values[0], indices)

    # Determine resulting number of views, slices, and channels and image size
    print('Sinogram shape: {}'.format(sinogram.shape))
    mbirjax.slice_viewer(sinogram, slice_axis=0)

    # Run once to finish compiling
    print('Starting back projection')
    bp = translation_model.sparse_back_project(sinogram, indices[0])
    print('Recon shape: ({}, {}, {})'.format(num_recon_rows, num_recon_cols, num_recon_slices))
    print('Memory stats after back projection')
    mbirjax.get_memory_stats(print_results=True)
    # ##########################
    # Test the adjoint property
    # Get a random 3D phantom to test the adjoint property
    key, subkey = jax.random.split(key)
    x = jax.random.uniform(subkey, shape=bp.shape)
    key, subkey = jax.random.split(key)
    y = jax.random.uniform(subkey, shape=sinogram.shape)

    # Do a forward projection, then a backprojection
    voxel_values = x.reshape((-1, num_recon_slices))[indices[0]]
    Ax = translation_model.sparse_forward_project(voxel_values, indices[0])
    Aty = translation_model.sparse_back_project(y, indices[0])

    # Calculate <Aty, x> and <y, Ax>
    Aty_x = jnp.sum(Aty * x)
    y_Ax = jnp.sum(y * Ax)

    print("Adjoint property holds for random x, y <y, Ax> = <Aty, x>: {}".format(np.allclose(Aty_x, y_Ax)))

    # Clean up before further projections
    del Ax, Aty, bp
    del phantom
    del x, y
    gc.collect()

    # ##########################
    # ## Test the hessian against a finite difference approximation ## #
    hessian = translation_model.compute_hessian_diagonal()

    x = jnp.zeros(recon_shape)
    key, subkey = jax.random.split(key)
    i, j = jax.random.randint(subkey, shape=(2,), minval=0, maxval=num_recon_rows)
    key, subkey = jax.random.split(key)
    k = jax.random.randint(subkey, shape=(), minval=0, maxval=num_recon_slices)

    eps = 0.01
    x = x.at[i, j, k].set(eps)
    voxel_values = x.reshape((-1, num_recon_slices))[indices[0]]
    Ax = translation_model.sparse_forward_project(voxel_values, indices[0])
    AtAx = translation_model.sparse_back_project(Ax, indices[0]).reshape(x.shape)
    finite_diff_hessian = AtAx[i, j, k] / eps
    print('Hessian matches finite difference: {}'.format(jnp.allclose(hessian.reshape(x.shape)[i, j, k], finite_diff_hessian)))

    # ##########################
    # Check the time taken per forward projection
    #  NOTE: recompiling happens whenever sparse_forward_project is called with a new *length* of input indices
    time_taken = 0

    print('\nStarting multiple forward projections...')
    for j in range(num_trials):
        voxel_values = x.reshape((-1, num_recon_slices))[indices[j]]
        t0 = time.time()
        fp = translation_model.sparse_forward_project(voxel_values, indices[j])
        time_taken += time.time() - t0
        del fp
        gc.collect()

    print('Mean time per call = {}'.format(time_taken / num_trials))
    print('Done')

    # ##########################
    # Check the time taken per backprojection
    #  NOTE: recompiling happens whenever sparse_back_project is called with a new *length* of input indices
    time_taken = 0

    print('\nStarting multiple backprojections...')
    for j in range(num_trials):
        t0 = time.time()
        bp = translation_model.sparse_back_project(sinogram, indices[j])
        time_taken += time.time() - t0
        del bp
        gc.collect()

    print('Mean time per call = {}'.format(time_taken / num_trials))
    print('Done')

    # ##########################
    # Show the forward and back projection from a single pixel
    i, j = num_recon_rows // 4, num_recon_cols // 3
    x = jnp.zeros(recon_shape)
    x = x.at[i, j, :].set(1)
    voxel_values = x.reshape((-1, num_recon_slices))[indices[0]]

    Ax = translation_model.sparse_forward_project(voxel_values, indices[0])
    Aty = translation_model.sparse_back_project(Ax, indices[0])
    Aty = translation_model.reshape_recon(Aty)

    y = jnp.zeros_like(sinogram)
    view_index = 30
    y = y.at[view_index].set(sinogram[view_index])
    index = jnp.ravel_multi_index((60, 60), (num_recon_rows, num_recon_cols))
    a1 = translation_model.sparse_back_project(y, indices[0])

    slice_index = 0
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    ax[0].imshow(x[:, :, slice_index])
    ax[0].set_title('x = phantom')
    ax[1].imshow(Ax[:, slice_index, :])
    ax[1].set_title('y = Ax')
    ax[2].imshow(Aty[:, :, slice_index])
    ax[2].set_title('Aty = AtAx')
    plt.pause(2)

    print('Final memory stats:')
    mbirjax.get_memory_stats(print_results=True)
    input('Press return to exit')
    a = 0
