import pandas as pd
import os
import re

def load_spike_data(filename):
    spike_data = pd.read_csv(filename, sep=' ', names=['spike_time', 'neuron_id'])
    spike_data['spike_time'] = pd.to_numeric(spike_data['spike_time'], errors='coerce')
    spike_data = spike_data.dropna()
    spike_data = spike_data[['neuron_id', 'spike_time']]
    spike_data['neuron_id'] = spike_data['neuron_id'].astype(int)
    return spike_data


def load_population_data(filename):
    return pd.read_csv(filename, dtype={
        '#n_cell': int,
        'n_comp': int,
        'name': str,
        'swc_file': str,
        'ion_file': str
    })



def merge_dat_files(directory_path):
    # Get all .dat files in the directory
    dat_files = [f for f in os.listdir(directory_path) if f.endswith('.dat')]

    # Sort by the number extracted from the filename
    sorted_files = sorted(dat_files, key=lambda x: int(re.findall(r'\d+', x)[0]))

    # Output file for merged data
    output_file = os.path.join('merged_output.dat')

    # Read and merge files in order
    with open(output_file, 'w') as outfile:
        for filename in sorted_files:
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as infile:
                outfile.write(infile.read())
            outfile.write('\n')

    print(f"File merge completed. Output file: {output_file}")
