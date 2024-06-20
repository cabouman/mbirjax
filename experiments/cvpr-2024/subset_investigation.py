import numpy as np
import mbirjax


if __name__ == "__main__":
    """
    This is a script to develop subset selection for VCD.
    """
    image_shape = (1024, 1024)
    small_tile_side = 16
    tile_type = 'repeat'  # 'repeat', 'permute', 'random', 'select'

    num_subsets = 1

    ror_mask = mbirjax.get_2d_ror_mask(image_shape)

    # bn_size = 256
    # filename = 'HDR_L_{}'.format(bn_size)
    # Code to convert png to npy
    # from PIL import Image
    # img = Image.open(filename + '.png')
    # img_array = np.array(img)
    # np.save(filename + '.npy', img_array)
    # pattern = np.load(filename + '.npy').astype(np.uint16)
    # np.savetxt('bn256.py', pattern, fmt='%d,', header='import numpy as np\nbn256 = np.array([',
    #            footer='], dtype=float).reshape((256, 256))', delimiter='', comments='')
    pattern = mbirjax.bn256.bn256

    bin_boundaries = np.linspace(0, 2**16, num_subsets + 1, endpoint=True)

    subsets = []
    subsets_fft = []
    num_tiles = [np.ceil(image_shape[k] / pattern.shape[k]).astype(int) for k in [0, 1]]

    single_subset_inds = np.floor(pattern / (2**16 / num_subsets)).astype(int)
    full_mask = np.zeros((num_subsets,) + image_shape, dtype=np.float32)

    if tile_type == 'repeat':
        # Repeat each bn subset to do the tiling
        subset_inds = np.tile(single_subset_inds, num_tiles)
        subset_inds = subset_inds[:image_shape[0], :image_shape[1]]
        subset_inds = (subset_inds + 1) * ror_mask - 1  # Get a 0 at each location outside the mask, subset_ind + 1 at other points
        subset_inds = subset_inds.flatten()
        flat_inds = []
        max_points = 0
        min_points = subset_inds.size
        for k in range(num_subsets):
            cur_inds = np.where(subset_inds == k)[0]
            flat_inds.append(cur_inds)  # Get all the indices for each subset
            max_points = max(max_points, cur_inds.size)
            min_points = min(min_points, cur_inds.size)

        extra_point_inds = np.random.randint(min_points, size=(max_points - min_points + 1,))
        for k in range(num_subsets):
            cur_inds = flat_inds[k]
            num_extra_points = max_points - cur_inds.size
            if num_extra_points > 0:
                extra_subset_inds = (k + 1 + np.arange(num_extra_points, dtype=int)) % num_subsets
                new_point_inds = [flat_inds[extra_subset_inds[j]][extra_point_inds[j]] for j in range(num_extra_points)]
                flat_inds[k] = np.concatenate((cur_inds, new_point_inds))
        flat_inds = np.array(flat_inds)
        full_mask = full_mask.reshape((num_subsets, np.prod(image_shape)))
        for j in range(num_subsets):
            full_mask[j][flat_inds[j]] = 1

        full_mask = full_mask.reshape((num_subsets,) + image_shape)

    elif tile_type == 'permute':
        # TODO:  work with indices rather than masks
        # Using a permutation of the bn subsets in each tile location
        single_subsets = [(pattern >= bin_boundaries[j]) * (pattern < bin_boundaries[j + 1]) for j in range(num_subsets)]
        perms = [np.random.permutation(num_subsets) for j in np.arange(np.prod(num_tiles))]
        for k in range(num_subsets):
            subset_indices = [perms[j][k] for j in range(np.prod(num_tiles))]
            cur_mask = [[single_subsets[j]] for j in subset_indices]
            cur_mask = np.array(cur_mask).astype(np.float32)
            cur_mask = cur_mask.reshape((num_tiles[0] * pattern.shape[0], num_tiles[1] * pattern.shape[1]))
            full_mask[k] = cur_mask[:image_shape[0], :image_shape[1]].astype(np.float32)

        full_mask = full_mask * ror_mask.reshape((1,) + image_shape)

    elif tile_type == 'select':
        # For each subset, select one element from each small tile.  Use a different permutation of the subsets
        # for each tile to determine which subset gets which element from each small tile.
        num_subsets = small_tile_side ** 2
        num_small_tiles = [np.ceil(image_shape[k] / small_tile_side).astype(int) for k in [0, 1]]
        perms = [np.random.permutation(num_subsets) for j in np.arange(np.prod(num_small_tiles))]
        perms = np.array(perms).T
        small_tile_corners = np.meshgrid(np.arange(num_small_tiles[0]), np.arange(num_small_tiles[1]))
        small_tile_corners[0] *= small_tile_side
        small_tile_corners[1] *= small_tile_side
        tile_inds = np.unravel_index(perms, (small_tile_side, small_tile_side))
        subset_inds = [small_tile_corners[j].reshape((1, -1)) + tile_inds[j] for j in [0, 1]]
        good_inds = (subset_inds[0] < image_shape[0]) * (subset_inds[1] < image_shape[1])
        flat_inds = []
        for k in range(num_subsets):
            flat_inds.append(
                np.ravel_multi_index((subset_inds[0][k][good_inds[k]], subset_inds[1][k][good_inds[k]]),
                                     image_shape))

        full_mask = full_mask.reshape((num_subsets, np.prod(image_shape)))
        for j in range(num_subsets):
            full_mask[j][flat_inds[j]] = 1

        full_mask = full_mask.reshape((num_subsets,) + image_shape)
        full_mask = full_mask * ror_mask.reshape((1,) + image_shape)

    else:  # 'random'
        # Random sampling - THIS DOES NOT GIVE A PARTITION!
        for k in range(num_subsets):
            cur_mask = np.random.rand(*image_shape)
            full_mask[k] = cur_mask < 1 / num_subsets
        full_mask = full_mask * ror_mask.reshape((1,) + image_shape)

    full_mask_fft = np.fft.fft2(full_mask)
    full_mask_fft = np.fft.fftshift(full_mask_fft, axes=(1, 2))
    full_mask_fft = 20 * np.log10(np.abs(full_mask_fft) + 1e-12)

    # print('Number of points = {}'.format(np.sum(subsets, axis=(1, 2))))
    mbirjax.slice_viewer(40 * full_mask, full_mask_fft, slice_axis=0, slice_label='Subset',
                         title='Subset mask and FFT in dB', vmin=0, vmax=60)

