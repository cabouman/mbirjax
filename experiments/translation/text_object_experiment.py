import numpy as np
import mbirjax as mj
from PIL import Image, ImageDraw, ImageFont

### Set sinogram shape
num_views = 128
num_det_rows = 128
num_det_channels = 128
sinogram_shape = (num_views, num_det_rows, num_det_channels)

### Generate test sample
np.random.seed(42)
test_sample = np.zeros((32, 256, 256), dtype=np.float32)

# Define Central rows
central_start = 8
central_end = 24


def create_text_array(text, array_size=256, vertical_offset=0, fontsize=20):
    img = Image.new('L', (array_size, array_size), 0)  # 'L' for grayscale
    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype("DejaVuSans.ttf", fontsize)

    text_box = draw.textbbox((0, 0), text, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]

    x = (array_size - text_width) // 2

    y = (array_size - text_height) // 2 + vertical_offset

    y = max(0, min(array_size - text_height, y))

    draw.text((x, y), text, fill=1, font=font)

    return np.array(img)


# Fill central rows with random 1s
words = ["Purdue", "Hooray!", "Translation", "Tomography", "Geometry", "University", "MBIRJAX", "Reconstruction",
         "Sinogram", "Presents", "Imaging", "Parallel Beam", "Cone beam", "Computed", "Projection", "Voxel"]
for slice_idx in range(central_start, central_end):
    text = words[slice_idx - central_start]
    vertical_offset = np.random.randint(-100, 100)
    fontsize = np.random.randint(15, 50)

    test_sample[slice_idx] = create_text_array(text, array_size=256, vertical_offset=vertical_offset, fontsize=fontsize)


### Set the source to detector distance and source to iso distance
source_detector_dist = 64
source_iso_dist = 32

### Set the translation vectors
# Define the starting point
center_x, center_z = 128, 128

# Generate grid positions for 128 points
# Using 16 columns and 8 rows (16 * 8 = 128)
cols = 16
rows = 8

# Calculate spacing between positions
spacing_x = 256 / (cols + 1)
spacing_z = 256 / (rows + 1)

# Generate all 128 translation vectors
translation_vectors = np.zeros((num_views, 3))

idx = 0
for row in range(1, rows + 1):
    for col in range(1, cols + 1):
        x = col * spacing_x
        z = row * spacing_z

        dx = x - center_x
        dz = z - center_z
        dy = np.random.choice([-5.0, 0.0, 5.0])

        translation_vectors[idx] = [dx, dy, dz]
        idx += 1


# Define the model for sinogram generation
ct_model_for_generation = mj.TranslationModel(sinogram_shape, translation_vectors, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)

# Check the reconstruction size that is set by auto_set_recon_size
auto_recon_shape = ct_model_for_generation.get_params('recon_shape')
print("Auto set recon size = ", auto_recon_shape)

# Set sinogram generation parameter values
ct_model_for_generation.set_params(recon_shape=test_sample.shape)

# Check the translation of the object along y-direction
source_iso_dist, delta_recon_row = ct_model_for_generation.get_params(['source_iso_dist', 'delta_recon_row'])
source_to_closest_pixel = source_iso_dist - (0.5 * test_sample.shape[0] * delta_recon_row)
source_to_farthest_pixel = source_iso_dist + (0.5 * test_sample.shape[0] * delta_recon_row)
print("Source to closest pixel without translation = ", source_to_closest_pixel)
print("Source to farthest pixel without translation = ", source_to_farthest_pixel)
max_translation = np.amax(translation_vectors, axis=0)
min_translation = np.amin(translation_vectors, axis=0)
source_to_closest_pixel = source_iso_dist - (0.5 * test_sample.shape[0] * delta_recon_row) - max_translation[1]
source_to_farthest_pixel = source_iso_dist + (0.5 * test_sample.shape[0] * delta_recon_row) - min_translation[1]
print("Maximum translation in y = ", max_translation[1])
print("Minimum translation in y = ", min_translation[1])
print("Source to closest pixel distance with translation = ", source_to_closest_pixel)
print("Source to farthest pixel distance with translation = ", source_to_farthest_pixel)

# Generate synthetic sinogram data
sinogram = ct_model_for_generation.forward_project(test_sample)
sinogram = np.asarray(sinogram)

# View test sample
mj.slice_viewer(test_sample, title='Generated Test Sample', slice_label='View')

# View sinogram
mj.slice_viewer(sinogram, slice_axis=0, vmin=0, vmax=1, title='Original sinogram', slice_label='View')

# Initialize model for reconstruction.
ct_model_for_recon = mj.TranslationModel(sinogram_shape, translation_vectors, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)

# Generate weights array - for an initial reconstruction, use weights = None, then modify if needed.
weights = None

# Set reconstruction parameter values
sharpness = 0.0
ct_model_for_recon.set_params(sharpness=sharpness, recon_shape=test_sample.shape)

# Print model parameters
ct_model_for_recon.print_params()

# Perform MBIR reconstruction
recon, recon_params = ct_model_for_recon.recon(sinogram, init_recon=0, weights=weights, stop_threshold_change_pct=0)

# Display Results
mj.slice_viewer(test_sample, recon, vmin=0, vmax=1, title='Object (left) vs. MBIR reconstruction (right)')