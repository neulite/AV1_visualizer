import matplotlib.pyplot as plt
import os
import numpy as np
from src.core.swc_reader import read_swc

def transform_coordinates(coords, translation, rotation_y, rotation_z):
    rot_y = np.array([
        [np.cos(rotation_y), 0, np.sin(rotation_y)],
        [0, 1, 0],
        [-np.sin(rotation_y), 0, np.cos(rotation_y)]
    ])
    rot_z = np.array([
        [np.cos(rotation_z), -np.sin(rotation_z), 0],
        [np.sin(rotation_z), np.cos(rotation_z), 0],
        [0, 0, 1]
    ])

    transformed = np.dot(np.dot(coords, rot_z), rot_y) + translation
    return transformed


def plot_neuron(ax, swc_data, cell_body_pos, color='blue', alpha_value=0.3):
    """Plot a single neuron in 3D space (optimized for 3D rendering)"""
    keys = list(swc_data.keys())
    points = np.array([[swc_data[k]['x'], swc_data[k]['y'], swc_data[k]['z']] for k in keys])

    translation = np.array([cell_body_pos["x"], cell_body_pos["y"], cell_body_pos["z"]])

    transformed_points = transform_coordinates(
        points,
        translation,
        cell_body_pos["y_rotate"],
        cell_body_pos["z_rotate"]
    )

    # Map SWC keys to array indices
    key_to_idx = {k: i for i, k in enumerate(keys)}

    # Collect line segments
    lines = []
    for i, k in enumerate(keys):
        parent_key = swc_data[k]['parent']
        if parent_key != -1 and parent_key in key_to_idx:
            lines.extend([transformed_points[i], transformed_points[key_to_idx[parent_key]], [np.nan, np.nan, np.nan]])

    # Draw all lines at once
    if lines:
        lines = np.array(lines)
        ax.plot(lines[:, 2], lines[:, 0], lines[:, 1], color=color, linewidth=0.8, alpha=alpha_value)


def build_swc_cache(swc_paths):
    """Pre-load all SWC files into a cache"""
    cache = {}
    for path in swc_paths:
        if path not in cache:
            cache[path] = read_swc(path)
    return cache


COLOR_SCHEME = {
    'i1Htr3a': '#D55E00',
    'e23Cux2': '#56B4E9', 'i23Htr3a': '#56B4E9', 'i23Pvalb': '#56B4E9', 'i23Sst': '#56B4E9',
    'e4Nr5a1': '#0072B2', 'e4Rorb': '#0072B2', 'e4Scnn1a': '#0072B2', 'e4other': '#0072B2',
    'i4Htr3a': '#0072B2', 'i4Pvalb': '#0072B2', 'i4Sst': '#0072B2',
    'e5Rbp4': '#CC79A7', 'e5noRbp4': '#CC79A7', 'i5Htr3a': '#CC79A7', 'i5Pvalb': '#CC79A7', 'i5Sst': '#CC79A7',
    'e6Ntsr1': '#332288', 'i6Htr3a': '#332288', 'i6Pvalb': '#332288', 'i6Sst': '#332288',
}

LAYER_ALPHA = {
    'i1Htr3a': 0.3,
    'e23Cux2': 0.3, 'i23Htr3a': 0.3, 'i23Pvalb': 0.3, 'i23Sst': 0.3,
    'e4Nr5a1': 0.3, 'e4Rorb': 0.3, 'e4Scnn1a': 0.3, 'e4other': 0.3,
    'i4Htr3a': 0.3, 'i4Pvalb': 0.3, 'i4Sst': 0.3,
    'e5Rbp4': 0.3, 'e5noRbp4': 0.3, 'i5Htr3a': 0.3, 'i5Pvalb': 0.3, 'i5Sst': 0.3,
    'e6Ntsr1': 0.3, 'i6Htr3a': 0.3, 'i6Pvalb': 0.3, 'i6Sst': 0.3,
}


def create_plot(res, spike_data, current_time, spike_duration, swc_cache, elev=10, azim=0):
    fig = plt.figure(figsize=(18, 18))
    ax = fig.add_subplot(111, projection='3d')
    current_spikes = spike_data[(spike_data['spike_time'] >= current_time - spike_duration) &
                                (spike_data['spike_time'] < current_time)]

    spiking_ids = set(current_spikes['neuron_id'].values)

    for key, group in res.groupby("split_name"):
        for neuron in group.iterrows():
            swc_path = neuron[1]['swc_file']
            node_id = neuron[1]['node_id']

            swc_data = swc_cache[swc_path]

            if node_id in spiking_ids:
                plot_neuron(ax, swc_data, neuron[1], color='red', alpha_value=0.8)
            else:
                plot_neuron(ax, swc_data, neuron[1], color=COLOR_SCHEME[key], alpha_value=LAYER_ALPHA.get(key, 0.3))

    ax.set_facecolor('white')
    ax.axis("off")
    fig.patch.set_facecolor('white')
    ax.autoscale()
    ax.view_init(elev=elev, azim=azim)

    return fig, ax

def create_plot_percent(res, spike_data, current_time, spike_duration, swc_cache, start_time=0, sample_rate=1):
    fig = plt.figure(figsize=(18, 18))
    ax = fig.add_subplot(111, projection='3d')
    filtered_spike_data = spike_data[spike_data['spike_time'] >= start_time]

    current_spikes = filtered_spike_data[
        (filtered_spike_data['spike_time'] >= current_time - spike_duration) &
        (filtered_spike_data['spike_time'] < current_time)
    ]

    spiking_ids = set(current_spikes['neuron_id'].values)

    active_neuron_ids = set(filtered_spike_data['neuron_id'].unique())
    filtered_res = res[res['node_id'].isin(active_neuron_ids)]

    total_cells = filtered_res.shape[0]
    selected_cells = np.random.choice(total_cells, size=int(total_cells * sample_rate), replace=False)
    selected_cell_ids = set(filtered_res.iloc[selected_cells]['node_id'].values)

    for key, group in filtered_res.groupby("split_name"):
        for neuron in group.iterrows():
            swc_path = neuron[1]['swc_file']
            node_id = neuron[1]['node_id']
            if node_id not in selected_cell_ids:
                continue

            swc_data = swc_cache[swc_path]

            if node_id in spiking_ids:
                plot_neuron(ax, swc_data, neuron[1], color='red', alpha_value=0.8)
            else:
                plot_neuron(ax, swc_data, neuron[1], color=COLOR_SCHEME[key])

    ax.set_facecolor('white')
    ax.axis("off")
    fig.patch.set_facecolor('white')
    ax.autoscale()

    return fig, ax


def save_frame(frame, fig, output_dir):
    plt.savefig(os.path.join(output_dir, f'frame_{frame:04d}.png'), dpi=300)
    plt.close(fig)
