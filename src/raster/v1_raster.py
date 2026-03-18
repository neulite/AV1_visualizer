"""V1 Raster Plot Generator

This module provides functions to create raster plots for V1 neurons.
"""

from src.utils import file_io, process_string
import matplotlib.pyplot as plt
import numpy as np


def get_v1_neuron_types():
    """V1 neuron type definitions and color scheme

    Returns:
        tuple: (neuron_types, color_scheme)
    """
    neuron_types = [
        'i1Htr3a',
        'e23Cux2', 'i23Htr3a', 'i23Pvalb', 'i23Sst',
        'e4Nr5a1', 'e4Rorb', 'e4Scnn1a', 'e4other',
        'i4Htr3a', 'i4Pvalb', 'i4Sst',
        'e5Rbp4', 'e5noRbp4', 'i5Htr3a', 'i5Pvalb', 'i5Sst',
        'e6Ntsr1', 'i6Htr3a', 'i6Pvalb', 'i6Sst'
    ]

    color_scheme = {
        'i1Htr3a': '#FF9999', 'e23Cux2': '#66B3FF', 'i23Htr3a': '#FF99CC', 'i23Pvalb': '#9999FF',
        'i23Sst': '#CC99FF', 'e4Nr5a1': '#99CCFF', 'e4Rorb': '#66CCFF', 'e4Scnn1a': '#33CCFF',
        'e4other': '#00CCFF', 'i4Htr3a': '#FF66CC', 'i4Pvalb': '#6666FF', 'i4Sst': '#9966FF',
        'e5Rbp4': '#99FF99', 'e5noRbp4': '#66FF66', 'i5Htr3a': '#FF3399', 'i5Pvalb': '#3333FF',
        'i5Sst': '#6633FF', 'e6Ntsr1': '#33FF33', 'i6Htr3a': '#FF0066', 'i6Pvalb': '#0000FF',
        'i6Sst': '#3300FF'
    }

    return neuron_types, color_scheme


def create_v1_raster(spike_file, population_csv, output_path=None, time_max=None, time_min=0):
    """Generate a V1 raster plot

    Args:
        spike_file: Path to the spike data file
        population_csv: Path to the population data CSV file
        output_path: Output file path (None = display only)
        time_max: Maximum time (ms, None = auto-detect)
        time_min: Minimum time (ms, default = 0)
    """
    # Load data
    pop = file_io.load_population_data(population_csv)
    pop["name_layer"] = pop["name"].apply(process_string.split_name)
    pop["#n_cell"] = pop["#n_cell"].apply(process_string.int_to_list)
    pop_explode = pop.explode("#n_cell")

    # Get neuron types and color scheme
    neuron_types, color_scheme = get_v1_neuron_types()

    n_neurons = pop_explode.shape[0]

    # Load spike data
    spike_data = np.loadtxt(spike_file)

    # Determine time range
    if time_max is None:
        time_max = int(spike_data[:, 0].max())

    n_timepoints = int(time_max) + 1

    # Generate spike mask (vectorized)
    spike_mask = np.zeros([n_neurons, n_timepoints], dtype=bool)

    time_indices = spike_data[:, 0].astype(int)
    neuron_indices = spike_data[:, 1].astype(int)
    valid = ((neuron_indices >= 0) & (neuron_indices < n_neurons) &
             (time_indices >= 0) & (time_indices < n_timepoints))
    spike_mask[neuron_indices[valid], time_indices[valid]] = True

    # Assign type to each neuron
    neuron_type_mapping = np.array(pop_explode["name_layer"])

    # Create plot
    fig, ax = plt.subplots(figsize=(15, 15))
    plt.rcParams.update({'font.size': 14})

    # Plot raster for each neuron type
    current_index = 0
    for ntype in neuron_types:
        # Get neurons of the current type
        mask = neuron_type_mapping == ntype
        n_current = np.sum(mask)

        if n_current > 0:
            # Set background color
            ax.axhspan(current_index, current_index + n_current, facecolor=color_scheme[ntype], alpha=0.3)

            # Plot spikes (batch drawing for all neurons of this type)
            type_spike_mask = spike_mask[mask]
            neuron_offsets, times = np.where(type_spike_mask)
            ax.scatter(times, neuron_offsets + current_index, color='black', s=1)

            # Add neuron type label
            ax.text(n_timepoints + 5, current_index + n_current/2, ntype, va='center', ha='left', fontsize=15)

            current_index += n_current

    # Configure plot
    ax.set_ylim(0, n_neurons)
    ax.set_xlim(time_min, time_max if time_max else n_timepoints)
    ax.set_xlabel('Time [ms]', fontsize=24)
    ax.set_ylabel('Neuron Index', fontsize=24)
    ax.tick_params(labelsize=24)
    ax.invert_yaxis()

    if output_path:
        plt.savefig(output_path, transparent=True, dpi=300)
        plt.close()
        print(f"Raster plot generated: {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='V1 Raster Plot Generator')
    parser.add_argument('--spike-file', required=True, help='Spike data file (.dat)')
    parser.add_argument('--population-csv', required=True, help='Population data CSV file')
    parser.add_argument('--output', default=None, help='Output PNG file path')
    parser.add_argument('--time-max', type=int, default=None, help='Maximum time (ms)')
    parser.add_argument('--time-min', type=int, default=0, help='Minimum time (ms)')

    args = parser.parse_args()

    create_v1_raster(
        spike_file=args.spike_file,
        population_csv=args.population_csv,
        output_path=args.output,
        time_max=args.time_max,
        time_min=args.time_min
    )
