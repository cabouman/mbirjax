#!/bin/bash

# Define the evaluation type
# 0=forward projection, 1=backprojection
evaluation_type_index=1

# Define the lists for each independent variable (each should be increasing, no spaces after =, spaces between ints)
num_views=(32 64 128)
num_indices=(2000 8000 32000)
num_channels=(1024 2048)
num_det_rows=(1024 2048)

# Convert array to string 
views_str="[${num_views[*]}]"         # Join array elements with spaces
views_str="${views_str// /, }"    # Replace spaces with comma-space
channels_str="[${num_channels[*]}]"         # Join array elements with spaces
channels_str="${channels_str// /, }"    # Replace spaces with comma-space
det_rows_str="[${num_det_rows[*]}]"         # Join array elements with spaces
det_rows_str="${det_rows_str// /, }"    # Replace spaces with comma-space
indices_str="[${num_indices[*]}]"         # Join array elements with spaces
indices_str="${indices_str// /, }"    # Replace spaces with comma-space

# Set up the evaluation files
echo "$views_str"
echo "Initializing" "$views_str" "$channels_str" "$det_rows_str" "$indices_str"
filename=$(python initialize_evaluation.py $evaluation_type_index "$views_str" "$channels_str" "$det_rows_str" "$indices_str")
echo "$filename"
# Nested loop over the lists
for nv in ${num_views[@]}; do
    for nc in ${num_channels[@]}; do
        for nr in ${num_det_rows[@]}; do
            # Loop over the indices for these nv, nc, nr
            echo "Starting ${nv}, ${nc}, ${nr}"
            output=$(python evaluate_over_indices.py "$filename" $nv $nc $nr)
            echo "$output"
        done
    done
done

output=$(python display_results.py "$filename")
