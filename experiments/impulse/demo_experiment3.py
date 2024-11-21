import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mbirjax

# **Initialize the CT Model Function**
def initialize_ct_model(geometry_type, num_views, num_det_rows, num_det_channels):
    angles = jnp.linspace(-np.pi / 2, np.pi / 2, num_views, endpoint=False)
    if geometry_type == 'parallel':
        return mbirjax.ParallelBeamModel(
            (num_views, num_det_rows, num_det_channels), angles
        ), angles
    elif geometry_type == 'cone':
        source_detector_dist = 4 * num_det_channels
        source_iso_dist = source_detector_dist
        return mbirjax.ConeBeamModel(
            (num_views, num_det_rows, num_det_channels),
            angles,
            source_detector_dist=source_detector_dist,
            source_iso_dist=source_iso_dist,
        ), angles
    else:
        raise ValueError("Invalid geometry type.")

# **Create Impulse Image**
def create_impulse_image(shape, position):
    image = np.zeros(shape, dtype=np.float32)
    image[position] = 1
    return image

# **Run Experiment**
def run_experiment(num_views, num_det_rows, num_det_channels, impulse_position):
    geometry_type = 'cone'  # Choose between 'parallel' or 'cone'
    voxel_grid_shape = (64, 64, 60)

    # Initialize CT model
    ct_model, angles = initialize_ct_model(geometry_type, num_views, num_det_rows, num_det_channels)

    # Create impulse image
    epsilon_i = create_impulse_image(voxel_grid_shape, impulse_position)

    # Forward projection
    y_v = ct_model.forward_project(jnp.array(epsilon_i))

    # Compute sinogram sums
    sinogram_sums = [jnp.sum(y_v[v, :, :]) for v in range(num_views)]

    return angles, sinogram_sums

# **Combined Visualization for Each Experiment**
def combined_plot(angles_sums_list, param_values, param_name):
    plt.figure(figsize=(10, 6))
    for (angles, sinogram_sums), param_value in zip(angles_sums_list, param_values):
        plt.plot(angles, sinogram_sums, marker='o', label=f'{param_name}={param_value}')
    plt.xlabel("View Angle (radians)")
    plt.ylabel("Sum of Sinogram Entries")
    plt.title(f"Sum of Sinogram Entries vs View Angle ({param_name.capitalize()})")
    plt.grid(True)
    plt.legend()
    plt.show()

# **Experiment 1: Vary num_views**
angles_sums_list = []
param_values = [32, 64, 128]
for num_views in param_values:
    angles, sinogram_sums = run_experiment(num_views, 40, 128, (32, 32, 30))
    angles_sums_list.append((angles, sinogram_sums))
combined_plot(angles_sums_list, param_values, 'num_views')

# **Experiment 2: Vary num_det_channels**
angles_sums_list = []
param_values = [16, 32, 64, 128, 256]
averages = []
for num_det_channels in param_values:
    angles, sinogram_sums = run_experiment(64, 40, num_det_channels, (32, 32, 30))
    average_sum = np.mean(sinogram_sums)  # Calculate the average
    angles_sums_list.append((angles, sinogram_sums))
    averages.append((num_det_channels, average_sum))
combined_plot(angles_sums_list, param_values, 'num_det_channels')

# Print the averages
print("Average Sum of Sinogram Entries for Each num_det_rows:")
for num_det_channels, average_sum in averages:
    print(f"num_det_channels={num_det_channels}: Average Sum = {average_sum:.2f}")


# **Experiment 3: Vary num_det_rows and Calculate Averages**
param_values = [10, 20, 30, 40, 80]
averages = []

for num_det_rows in param_values:
    angles, sinogram_sums = run_experiment(64, num_det_rows, 128, (32, 32, 30))
    average_sum = np.mean(sinogram_sums)  # Calculate the average
    averages.append((num_det_rows, average_sum))
    angles_sums_list.append((angles, sinogram_sums))

combined_plot(angles_sums_list, param_values, 'num_det_rows')

# Print the averages
print("Average Sum of Sinogram Entries for Each num_det_rows:")
for num_det_rows, average_sum in averages:
    print(f"num_det_rows={num_det_rows}: Average Sum = {average_sum:.2f}")


# **Experiment 4: Vary impulse_position**
angles_sums_list = []
param_values = [(32, 32, 30), (10, 10, 10), (50, 50, 40)]
for impulse_position in param_values:
    angles, sinogram_sums = run_experiment(64, 40, 128, impulse_position)
    angles_sums_list.append((angles, sinogram_sums))
combined_plot(angles_sums_list, param_values, 'impulse_position')


