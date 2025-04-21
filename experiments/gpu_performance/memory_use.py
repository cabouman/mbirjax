#!/usr/bin/env python
import argparse
import csv
import itertools
import subprocess
import sys
import numpy as np
import time
import os
import mbirjax

target_gb = 40

max_gb = 79
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(target_gb / max_gb)

"""
This is a Python program to run a grid search over parameters and record the peak GPU memory usage from a JAX computation.
The code to run is either a direct recon, or a projection, or a recon with specified use of the GPU. 
The parameters to be scanned are 

    v: number of views
    r: number of rows
    c: number channels
    b: batch size
    
Additionally, the memory fraction for jax should be varied since memory use depends on available memory.  This 
can be done manually using e.g.

target_gb = 10
max_gb = 79
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(target_gb / max_gb)

How This Code Works

For each fixed combination of (v, r, c), the program starts a new process to evaluate the function and record the 
memory and time in a .csv file.  Within this process, the batch size is increased so that memory use is monotonic, 
with termination in case of OOM.  

    Worker Process Mode:

        When run with the --worker flag and with parameters --v, --r, and --c, the script enters worker_main().
        For each value of b (using the same grid as for the other parameters), it runs the JAX function and obtains the peak memory usage via device.memory_stats().
        The worker prints one CSV row (with comma-separated values for v, r, c, b, and peak_memory) per iteration. Any errors are printed to stderr.

    Main Process Mode:

        When no --worker flag is provided, the script runs in main mode.
        It opens (or creates) a CSV file (memory_stats.csv) and writes a header row.
        It then loops over all combinations of v, r, and c (using itertools.product), spawns a worker process for each combination (using subprocess.run), and captures the worker’s stdout.
        The main process parses each CSV row output by the worker and writes it to the CSV file.
        If any worker prints error messages to stderr, these are forwarded to the main process’s stderr.
"""

# Define the set of parameter values to test.
PARAM_VALUES = [50, 100, 250, 500, 800, 1250]
BATCH_SIZES = [100, 250, 500, 800]
OUTPUT_SYMBOL = "#"


def worker_main(v, r, c):
    """
    Worker process:
    For a fixed combination of v, r, c, loop over b values,
    run the JAX computation, and print a CSV row for each b.
    """
    import jax
    import jax.numpy as jnp

    # Define a JAX function that depends on the parameters.
    @jax.jit
    def sample_function(v, r, c, b, x):
        A = jnp.ones((v, r))
        B = jnp.ones((c, b))
        # For a valid dot product, x is chosen with shape (r, c)
        result = jnp.dot(A, jnp.dot(x, B))
        return result

    def run_experiment(v, r, c, b):

        # Create an input array x with shape (v, r, c)
        sino = np.zeros((v, r, c))
        sino[:, r//3:r-r//3, c//3:c-c//3] = 1
        angles = np.linspace(0, 2*np.pi, v, endpoint=False)
        source_detector_dist = 4 * c
        source_iso_dist = source_detector_dist

        ct_model = mbirjax.ConeBeamModel(sino.shape, angles,
                                         source_detector_dist=source_detector_dist,
                                         source_iso_dist=source_iso_dist)
        ct_model.set_params(use_gpu='sinograms', verbose=0)
        ct_model.view_batch_size_for_vmap = b
        weights = ct_model.gen_weights(sino, weight_type='unweighted')

        # Run the computation and block until complete.
        time0 = time.time()
        recon, recon_params = ct_model.recon(sino, weights=weights, max_iterations=1)
        _ = recon.block_until_ready()
        elapsed_time0 = time.time() - time0

        # time0 = time.time()
        # recon, recon_params = ct_model.recon(sino, weights=weights, max_iterations=1)
        # _ = recon.block_until_ready()
        # elapsed_time1 = time.time() - time0
        elapsed_time1 = elapsed_time0

        # Retrieve memory stats.
        stats = mbirjax.get_memory_stats(print_results=False)
        peak_memory = stats[0]['peak_bytes_in_use'] / (1024 ** 3)
        avail_memory = stats[0]['bytes_limit'] / (1024 ** 3)
        return peak_memory, avail_memory, elapsed_time0, elapsed_time1

    # Loop over b values
    summary_output = ""
    num_output_lines = 0
    for b in BATCH_SIZES:
        try:
            peak, avail, elapsed0, elapsed1 = run_experiment(v, r, c, b)
            # Print CSV row: v, r, c, b, peak_memory, elapsed_time.
            # These values will be captured by the main process.
            summary_output += OUTPUT_SYMBOL + f",{v},{r},{c},{b},{peak:.3f},{avail:.3f},{elapsed0:.1f},{elapsed1:.1f}\n"
            num_output_lines += 1
        except Exception as e:
            # If an error occurs, print to stderr.
            print(f"{v},{r},{c},{b},Error: {e}", file=sys.stderr)

    print(summary_output)


def main():
    """
    Main process:
    Iterate over all combinations of (v, r, c), spawn a worker process
    for each, and write the output to a CSV file.
    """
    # Open the output CSV file and write the header.
    output_file_name = "memory_stats.csv"
    file_exists = os.path.isfile(output_file_name)
    with open(output_file_name, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["num views", "num rows", "num channels", "batch size", "peak memory (GB)", "avail memory (GB)", "elapsed time0 (sec)", "elapsed time1 (sec)"])
        # Iterate over all combinations of v, r, c.
        for v, r, c in itertools.product(PARAM_VALUES, repeat=3):
            print('\nStarting multiple batches with v={}, r={}, c={}'.format(v, r, c))
            print('Batch sizes = {}'.format(BATCH_SIZES))
            # Spawn a worker process for the given (v, r, c) combination.
            result = subprocess.run(
                [sys.executable, __file__, "--worker", "--v", str(v), "--r", str(r), "--c", str(c)],
                capture_output=True,
                text=True
            )
            # Process the worker's stdout which contains CSV rows.
            for line in result.stdout.strip().splitlines():
                # Expect each line to have the format: v,r,c,b,peak_memory
                row = line.strip().split(",")
                if row[0] == OUTPUT_SYMBOL:
                    writer.writerow(row[1:])
            if result.stderr:
                print(f"Error in worker for v={v}, r={r}, c={c}:", result.stderr, file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JAX GPU memory usage experiment.")
    parser.add_argument("--worker", action="store_true", help="Run as worker process")
    parser.add_argument("--v", type=int, help="Parameter v (required for worker)")
    parser.add_argument("--r", type=int, help="Parameter r (required for worker)")
    parser.add_argument("--c", type=int, help="Parameter c (required for worker)")
    args = parser.parse_args()

    if args.worker:
        # Worker mode: require all three parameters.
        if args.v is None or args.r is None or args.c is None:
            raise ValueError("Worker process requires --v, --r, and --c arguments.")
        worker_main(args.v, args.r, args.c)
    else:
        # Main mode: spawn worker processes and aggregate CSV output.
        main()
