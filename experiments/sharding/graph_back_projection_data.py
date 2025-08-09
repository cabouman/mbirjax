import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def graph_sparse_back_projection_time_vs_size_of_sinogram(filepath, output_directory):

    # Load the CSV file
    df = pd.read_csv(filepath)

    # Group data by num_gpus
    grouped_by_num_gpus = {gpu: group.reset_index(drop=True) for gpu, group in df.groupby('num_gpus')}

    plt.figure(figsize=(10, 6))
    for gpu_count, data in grouped_by_num_gpus.items():
        plt.loglog(data['size'], data['elapsed'], marker='o', label=f'{gpu_count} GPU(s)')

    # Clearer, more descriptive axis labels and title
    plt.xlabel('Size of Sinogram Axes (float32, log scale)')
    plt.ylabel('Elapsed Time (s, log scale)')
    plt.title('Sparse Back Projection Elapsed Time vs Sinogram Size (Log-Log Scale)')

    plt.legend(title='Number of GPUs', loc='upper left')


    xtick_vals = [128, 256, 512, 1024]
    ax = plt.gca()
    ax.set_xticks(xtick_vals)              # Set major ticks
    ax.set_xticks([], minor=True)          # Remove minor ticks
    ax.set_xticklabels([str(x) for x in xtick_vals])

    plt.grid(True, which='both', axis='x')

    plt.savefig(output_directory + '/graph_sparse_back_projection_time_vs_size_of_sinogram.png', dpi=300, bbox_inches='tight')

    plt.show()

def graph_sparse_back_projection_time_vs_size_of_sinogram_with_2k3_projection(filepath, output_directory):

    # Load the CSV file
    df = pd.read_csv(filepath)

    # Group data by num_gpus
    grouped_by_num_gpus = {gpu: group.reset_index(drop=True) for gpu, group in df.groupby('num_gpus')}

    plt.figure(figsize=(10, 6))
    for gpu_count, data in grouped_by_num_gpus.items():
        plt.loglog(data['size'], data['elapsed'], marker='o', label=f'{gpu_count}')

    # Clearer, more descriptive axis labels and title
    plt.xlabel('Size of Sinogram Axes (float32, log scale)')
    plt.ylabel('Elapsed Time (s, log scale)')
    plt.title('Sparse Back Projection Elapsed Time vs Sinogram Size (Log-Log Scale)')

    plt.loglog(2000, 55, marker='*', color='#d62728', label=f'8 - 2K')

    plt.legend(title='Number of GPUs', loc='upper left')

    xtick_vals = [128, 256, 512, 1024, 2048]
    ax = plt.gca()
    ax.set_xticks(xtick_vals)              # Set major ticks
    ax.set_xticks([], minor=True)          # Remove minor ticks
    ax.set_xticklabels([str(x) for x in xtick_vals])

    plt.grid(True, which='both', axis='x')

    plt.savefig(output_directory + '/graph_sparse_back_projection_time_vs_size_of_sinogram_with_2k3_projection.png', dpi=300, bbox_inches='tight')

    plt.show()

def graph_sparse_back_projection_time_vs_num_gpus_loglog(filepath, output_directory):
    # Load the CSV file
    df = pd.read_csv(filepath)

    # Group data by size
    grouped_by_size = {gpu: group.reset_index(drop=True) for gpu, group in df.groupby('size')}

    plt.figure(figsize=(10, 6))
    for sino_size, data in grouped_by_size.items():
        plt.loglog(data['num_gpus'], data['elapsed'], marker='o', label=f'{sino_size}')

    plt.xlabel('Number of GPUs (log scale)')
    plt.ylabel('Elapsed Time (s, log scale)')
    plt.title('Post-JIT Sparse Back Projection Time vs Number of GPUs (Log-Log Scale)')

    plt.legend(title='Sinogram Size', loc='upper right')
    plt.grid(True, which='both', axis='x')
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], ['1', '2', '3', '4', '5', '6', '7', '8'])

    plt.savefig(output_directory + '/graph_sparse_back_projection_time_vs_num_gpus_loglog.png', dpi=300, bbox_inches='tight')

    plt.show()

def graph_sparse_back_projection_median_peak_memory_normalized_by_num_gpus_vs_num_gpus_semilog(filepath, output_directory):

    # Load the CSV file
    df = pd.read_csv(filepath)

    # Group data by size
    grouped_by_size = {gpu: group.reset_index(drop=True) for gpu, group in df.groupby('size')}

    plt.figure(figsize=(10, 6))
    for sino_size, data in grouped_by_size.items():
        median_peak_memory_normalized_by_num_gpus = []

        for _, row in data.iterrows():
            num_gpus = int(row['num_gpus'])
            # Collect all non-null GPU peak memory columns
            peak_memory_values = [
                row[f'gpu{i}_peak_bytes'] for i in range(num_gpus)
                if f'gpu{i}_peak_bytes' in row and not pd.isna(row[f'gpu{i}_peak_bytes'])
            ]

            if peak_memory_values:
                median_peak = np.median(peak_memory_values)
                normalized = median_peak / num_gpus
            else:
                normalized = np.nan  # In case of missing data

            median_peak_memory_normalized_by_num_gpus.append(normalized)

        plt.semilogx(data['num_gpus'], median_peak_memory_normalized_by_num_gpus, marker='o', label=f'{sino_size}')

    plt.xlabel('Number of GPUs (log scale)')
    plt.ylabel('Median Peak Memory Normalized by Number of GPUs (Bytes)')
    plt.title('Sparse Back Projection Median Peak Memory Normalized by Number of GPUs vs Number of GPUs (Semilog-X)')

    plt.legend(title='Sinogram Size', loc='upper right')
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], ['1', '2', '3', '4', '5', '6', '7', '8'])
    plt.grid(True, which='both', axis='x')

    plt.savefig(output_directory + '/graph_sparse_back_projection_median_peak_memory_normalized_by_num_gpus_vs_num_gpus_semilog.png', dpi=300, bbox_inches='tight')

    plt.show()

def graph_sparse_back_projection_total_peak_memory_normalized_by_num_gpus_vs_num_gpus_semilog(filepath, output_directory):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Load the CSV file
    df = pd.read_csv(filepath)

    # Group data by size
    grouped_by_size = {gpu: group.reset_index(drop=True) for gpu, group in df.groupby('size')}

    plt.figure(figsize=(10, 6))
    for sino_size, data in grouped_by_size.items():
        total_peak_memory_normalized_by_num_gpus = []

        for _, row in data.iterrows():
            num_gpus = int(row['num_gpus'])
            peak_memory_values = [
                row[f'gpu{i}_peak_bytes'] for i in range(num_gpus)
                if f'gpu{i}_peak_bytes' in row and not pd.isna(row[f'gpu{i}_peak_bytes'])
            ]

            if peak_memory_values:
                total_peak = sum(peak_memory_values)
                normalized = total_peak / num_gpus
            else:
                normalized = np.nan

            total_peak_memory_normalized_by_num_gpus.append(normalized)

        plt.semilogx(
            data['num_gpus'],
            total_peak_memory_normalized_by_num_gpus,
            marker='o',
            label=f'{sino_size}'
        )

    plt.xlabel('Number of GPUs (log scale)')
    plt.ylabel('Total Peak Memory Normalized by Number of GPUs (Bytes)')
    plt.title('Sparse Back Projection Total Peak Memory Normalized by GPUs vs Number of GPUs (Semilog-X)')
    plt.legend(title='Sinogram Size', loc='upper right')
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], ['1', '2', '3', '4', '5', '6', '7', '8'])
    plt.grid(True, which='both', axis='x')

    plt.savefig(output_directory + '/graph_sparse_back_projection_total_peak_memory_normalized_by_num_gpus_vs_num_gpus_semilog.png', dpi=300, bbox_inches='tight')

    plt.show()

if __name__ == "__main__":

    input_filepath = "../output/back_project.csv"
    output_directory_path = "../output"

    # slide 3
    graph_sparse_back_projection_time_vs_size_of_sinogram(input_filepath, output_directory_path)

    # slide 4
    graph_sparse_back_projection_time_vs_size_of_sinogram_with_2k3_projection(input_filepath, output_directory_path)

    # slide 5
    graph_sparse_back_projection_time_vs_num_gpus_loglog(input_filepath, output_directory_path)

    # slide 6
    graph_sparse_back_projection_median_peak_memory_normalized_by_num_gpus_vs_num_gpus_semilog(input_filepath, output_directory_path)

    # slide 7
    graph_sparse_back_projection_total_peak_memory_normalized_by_num_gpus_vs_num_gpus_semilog(input_filepath, output_directory_path)
