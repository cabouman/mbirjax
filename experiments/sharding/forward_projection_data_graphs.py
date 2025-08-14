import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def graph_sparse_forward_projection_time_vs_num_gpus_loglog(filepath, output_directory):
    # Load the CSV file
    df = pd.read_csv(filepath)

    # Group data by size
    grouped_by_size = {gpu: group.reset_index(drop=True) for gpu, group in df.groupby('size')}

    plt.figure(figsize=(10, 6))

    for sino_size, data in grouped_by_size.items():
        print(data['elapsed'])
        data = data[data['model_type'] != 'cone']
        print(data['elapsed'])
        first_time_elapse = data['elapsed'].iloc[0]
        print(first_time_elapse)
        time_elapsed = [time / first_time_elapse for time in data['elapsed']]
        plt.loglog(data['num_gpus'], time_elapsed, marker='o', label=f'{sino_size}')

    for sino_size, data in grouped_by_size.items():
        data = data[data['model_type'] == 'cone']
        first_time_elapse = data['elapsed'].iloc[0]
        time_elapsed = [time / first_time_elapse for time in data['elapsed']]
        plt.loglog(data['num_gpus'], time_elapsed, marker='x', label=f'{sino_size}')

    num_gpus = 1 + np.arange(8)
    plt.loglog(num_gpus, 1 / num_gpus, marker='*', label=f'Reference 1 / x')

    plt.xlabel('Number of GPUs')
    plt.ylabel('Time relative to 1 GPU')
    plt.title('Sparse Forward Projection Relative Time vs Number of GPUs (loglog Scale)')

    plt.legend(title='Sinogram Size', loc='lower left')
    plt.grid(True, which='both', axis='x')
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], ['1', '2', '3', '4', '5', '6', '7', '8'])
    plt.yticks([0.1, 0.5, 1], ['0.1', '0.5', '1.0'])

    plt.savefig(output_directory + '/graph_sparse_forward_projection_time_vs_num_gpus_loglog.png', dpi=300, bbox_inches='tight')

    plt.show()


def graph_sparse_forward_projection_time_vs_size_of_sinogram(filepath, output_directory):

    # Load the CSV file
    df = pd.read_csv(filepath)

    # Group data by num_gpus
    grouped_by_num_gpus = {gpu: group.reset_index(drop=True) for gpu, group in df.groupby('num_gpus')}

    xtick_vals = [128, 256, 512, 1024, 2048]

    plt.figure(figsize=(10, 6))

    for gpu_count, data in grouped_by_num_gpus.items():
        data = data[data['model_type'] != 'cone']
        plt.loglog(data['size'], data['elapsed'], marker='o', label=f'{gpu_count}')

    for gpu_count, data in grouped_by_num_gpus.items():
        data = data[data['model_type'] == 'cone']
        plt.loglog(data['size'], data['elapsed'], marker='x', label=f'{gpu_count}')

    plt.loglog(xtick_vals, 0.01 * (np.array(xtick_vals) / xtick_vals[0]) ** 3, marker='*', label=f'Reference 0.1 * x**3')

    # Clearer, more descriptive axis labels and title
    plt.xlabel('Size of Sinogram Axes (float32, log scale)')
    plt.ylabel('Elapsed Time (s, log scale)')
    plt.title('Sparse Forward Projection Elapsed Time vs Sinogram Size (Log-Log Scale, with extrapolation)')


    plt.legend(title='Number of GPUs', loc='upper left')

    ax = plt.gca()
    ax.set_xticks(xtick_vals)              # Set major ticks
    ax.set_xticks([], minor=True)          # Remove minor ticks
    ax.set_xticklabels([str(x) for x in xtick_vals])

    plt.yticks([0.01, 0.1, 1.0, 10.0, 20.0, 60.0], ['0.01', '0.1', '1.0', '10', '20', '60']),

    plt.grid(True, which='both', axis='x')

    plt.savefig(output_directory + '/graph_sparse_forward_projection_time_vs_size_of_sinogram.png', dpi=300, bbox_inches='tight')

    plt.show()


def graph_sparse_forward_projection_peak_memory_vs_num_gpus(filepath, output_directory):

    # Load the CSV file
    df = pd.read_csv(filepath)

    # Group data by size
    grouped_by_size = {gpu: group.reset_index(drop=True) for gpu, group in df.groupby('size')}

    plt.figure(figsize=(10, 6))
    for sino_size, data in grouped_by_size.items():
        peak_memory = []

        data = data[data['model_type'] != 'cone']

        for _, row in data.iterrows():
            num_gpus = int(row['num_gpus'])
            peak_memory_values = [
                row[f'gpu{i}_peak_bytes'] for i in range(num_gpus)
                if f'gpu{i}_peak_bytes' in row and not pd.isna(row[f'gpu{i}_peak_bytes'])
            ]

            if peak_memory_values:
                total_peak = max(peak_memory_values)
                normalized = total_peak
            else:
                normalized = np.nan

            peak_memory.append(normalized)

        first_mem = peak_memory[0]
        peak_memory = [mem / first_mem for mem in peak_memory]

        plt.loglog(
            data['num_gpus'],
            peak_memory ,
            marker='o',
            label=f'{sino_size}'
        )

    for sino_size, data in grouped_by_size.items():
        peak_memory = []

        data = data[data['model_type'] == 'cone']

        for _, row in data.iterrows():
            num_gpus = int(row['num_gpus'])
            peak_memory_values = [
                row[f'gpu{i}_peak_bytes'] for i in range(num_gpus)
                if f'gpu{i}_peak_bytes' in row and not pd.isna(row[f'gpu{i}_peak_bytes'])
            ]

            if peak_memory_values:
                total_peak = max(peak_memory_values)
                normalized = total_peak
            else:
                normalized = np.nan

            peak_memory.append(normalized)

        first_mem = peak_memory[0]
        peak_memory = [mem / first_mem for mem in peak_memory]

        plt.loglog(
            data['num_gpus'],
            peak_memory,
            marker='x',
            label=f'{sino_size}'
        )

    num_gpus = 1 + np.arange(8)
    plt.loglog(num_gpus, 1 / num_gpus, marker='*', label=f'Reference 1 / x')
    plt.xlabel('Number of GPUs')
    plt.ylabel('Max Peak Memory Relative to Peak for 1 GPU')
    plt.title('Sparse Forward Projection Max Peak Memory vs Number of GPUs (loglog)')
    plt.legend(title='Sinogram Size', loc='upper right')
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], ['1', '2', '3', '4', '5', '6', '7', '8'])
    plt.yticks([0.1, 0.5, 1], ['0.1', '0.5', '1.0'])
    plt.grid(True, which='both', axis='x')

    plt.savefig(output_directory + '/graph_sparse_forward_projection_peak_memory_vs_num_gpus.png', dpi=300, bbox_inches='tight')

    plt.show()


def graph_sparse_forward_projection_peak_memory_vs_size_of_sinogram(filepath, output_directory):

    # Load the CSV file
    df = pd.read_csv(filepath)

    # Group data by num_gpus
    grouped_by_num_gpus = {gpu: group.reset_index(drop=True) for gpu, group in df.groupby('num_gpus')}

    xtick_vals = [128, 256, 512, 1024, 2048]

    plt.figure(figsize=(10, 6))
    for gpu_count, data in grouped_by_num_gpus.items():
        peak_memory = []

        data = data[data['model_type'] != 'cone']

        for _, row in data.iterrows():
            num_gpus = int(row['num_gpus'])
            peak_memory_values = [
                row[f'gpu{i}_peak_bytes'] for i in range(num_gpus)
                if f'gpu{i}_peak_bytes' in row and not pd.isna(row[f'gpu{i}_peak_bytes'])
            ]

            if peak_memory_values:
                total_peak = max(peak_memory_values)
                normalized = total_peak
            else:
                normalized = np.nan

            peak_memory.append(normalized)

        peak_memory_GB = np.array(peak_memory) / (1024**3)
        plt.loglog(data['size'], peak_memory_GB, marker='o', label=f'{gpu_count}')


    for gpu_count, data in grouped_by_num_gpus.items():
        peak_memory = []

        data = data[data['model_type'] == 'cone']

        for _, row in data.iterrows():
            num_gpus = int(row['num_gpus'])
            peak_memory_values = [
                row[f'gpu{i}_peak_bytes'] for i in range(num_gpus)
                if f'gpu{i}_peak_bytes' in row and not pd.isna(row[f'gpu{i}_peak_bytes'])
            ]

            if peak_memory_values:
                total_peak = max(peak_memory_values)
                normalized = total_peak
            else:
                normalized = np.nan

            peak_memory.append(normalized)

        peak_memory_GB = np.array(peak_memory) / (1024**3)
        plt.loglog(data['size'], peak_memory_GB, marker='x', label=f'{gpu_count}')

    plt.loglog(xtick_vals, 0.05 * (np.array(xtick_vals) / xtick_vals[0]) ** 2, marker='*', label=f'Reference 0.05 * x**2')

    # Clearer, more descriptive axis labels and title
    plt.xlabel('Size of Sinogram Axes (float32, log scale)')
    plt.ylabel('Max Peak Memory (GB)')
    plt.title('Sparse Forward Projection Max Peak Memory vs Sinogram Size (Log-Log Scale, with extrapolation)')

    plt.legend(title='Number of GPUs', loc='upper left')

    ax = plt.gca()
    ax.set_xticks(xtick_vals)              # Set major ticks
    ax.set_xticks([], minor=True)          # Remove minor ticks
    ax.set_xticklabels([str(x) for x in xtick_vals])

    plt.yticks([0.05, 0.1, 1.0, 10.0], ['0.05', '0.1', '1.0', '10']),

    plt.grid(True, which='both', axis='x')

    plt.savefig(output_directory + '/graph_sparse_forward_projection_peak_memory_vs_size_of_sinogram.png', dpi=300, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":

    input_filepath = "../output/forward_project.csv"
    output_directory_path = "../output"

    # slide 5
    graph_sparse_forward_projection_time_vs_num_gpus_loglog(input_filepath, output_directory_path)

    # slide 4
    graph_sparse_forward_projection_time_vs_size_of_sinogram(input_filepath, output_directory_path)

    # slide 7
    graph_sparse_forward_projection_peak_memory_vs_num_gpus(input_filepath, output_directory_path)

    # slide 6
    graph_sparse_forward_projection_peak_memory_vs_size_of_sinogram(input_filepath, output_directory_path)
