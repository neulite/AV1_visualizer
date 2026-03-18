"""3D Column Video Generator (Single Process Version)

This module provides functions to generate 3D column videos using single process.
Useful for debugging and simple execution.
"""

import numpy as np
import h5py
import pandas as pd
import os
import time
from src.utils import process_string, file_io
from src.core import plotting
from src.core.plotting import build_swc_cache
from src.video.movie_creator import create_video_from_frames


def create_column_video_single(h5_file, node_types_csv, population_csv, spike_file,
                                output_dir='output/frames', output_video='output/videos/column_video.mp4',
                                total_frames=500, duration_ms=1000, spike_duration_ms=10,
                                ffmpeg_path='ffmpeg', framerate=10,
                                sample_rate=1.0, start_time=0):
    """Generate a 3D neuron video (single process version)

    Args:
        h5_file: Path to the H5 file (neuron coordinate data)
        node_types_csv: Path to the node types CSV file
        population_csv: Path to the population data CSV file
        spike_file: Path to the spike data file
        output_dir: Frame output directory
        output_video: Video output file path
        total_frames: Total number of frames
        duration_ms: Video duration (milliseconds)
        spike_duration_ms: Spike display duration (milliseconds)
        ffmpeg_path: Path to ffmpeg executable
        framerate: Frame rate
        sample_rate: Sampling rate (1.0 = all data, 0.1 = 10%)
        start_time: Start time (milliseconds)
    """
    print("Loading data...")

    # Load data from H5 file
    res = pd.DataFrame()
    with h5py.File(h5_file) as f:
        res["node_id"] = f["nodes/v1/node_id"][:]
        res["node_type"] = f["nodes/v1/node_type_id"][:]
        res["x"] = f["nodes/v1/0/x"][:]
        res["y"] = f["nodes/v1/0/y"][:]
        res["z"] = f["nodes/v1/0/z"][:]
        res["y_rotate"] = f["nodes/v1/0/rotation_angle_yaxis"][:]
        res["z_rotate"] = f["nodes/v1/0/rotation_angle_zaxis"][:]

    # Load node types
    node_types = pd.read_csv(node_types_csv, delimiter=" ")

    res["name"] = res.apply(process_string.combine_name, df_node_type=node_types, axis=1)
    res["split_name"] = res["name"].apply(process_string.split_name)

    # Invert y-axis
    res["y"] = res["y"] * (-1)
    res["y"] = res["y"] - res["y"].mean()  # Center at 0 (for correct elev behavior)

    # Load population data
    pop_csv_df = file_io.load_population_data(population_csv)
    pop_csv_df["name_layer"] = pop_csv_df["name"].apply(process_string.split_name)
    pop_csv_df["#n_cell"] = pop_csv_df["#n_cell"].apply(process_string.int_to_list)
    pop_explode = pop_csv_df.explode("#n_cell")

    res = pd.concat([res.reset_index(drop=True), pop_explode["swc_file"].reset_index(drop=True)], axis=1)

    # Load spike data
    spike_data = file_io.load_spike_data(spike_file)

    # Sampling (optional)
    if sample_rate < 1.0:
        step = int(1 / sample_rate)
        res = res[res["node_id"] % step == 0].reset_index(drop=True)
        filtered_and_shifted_spike_data = spike_data[
            (spike_data["neuron_id"] % step == 0) &
            (spike_data["spike_time"] > start_time)
        ].copy()
        filtered_and_shifted_spike_data["spike_time"] -= start_time
    else:
        filtered_and_shifted_spike_data = spike_data[spike_data["spike_time"] > start_time].copy()
        filtered_and_shifted_spike_data["spike_time"] -= start_time

    print(f"Max neuron ID: {filtered_and_shifted_spike_data['neuron_id'].max()}")
    print(f"Max spike time: {filtered_and_shifted_spike_data['spike_time'].max()}")

    # Pre-cache SWC data
    print("Building SWC data cache...")
    swc_cache = build_swc_cache(res["swc_file"].unique())
    print(f"  Cached SWC files: {len(swc_cache)}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set parameters
    time_window = duration_ms / total_frames

    print("Starting frame generation...")
    start_time_proc = time.time()

    # Frame generation (single process)
    for frame in range(total_frames):
        current_time = frame * time_window
        fig, ax = plotting.create_plot(res, filtered_and_shifted_spike_data, current_time, spike_duration_ms, swc_cache)

        # Rotate viewpoint
        ax.view_init(elev=360, azim=frame % 360)

        plotting.save_frame(frame, fig, output_dir)

        if frame % 10 == 0:
            print(f"Processing frame {frame}/{total_frames}")

    end_time_proc = time.time()
    print(f"Frame generation completed. Saved to {output_dir}. Elapsed time: {end_time_proc - start_time_proc:.2f} sec")

    # Convert frame sequence to video
    print("Starting video generation...")
    input_pattern = os.path.join(output_dir, "frame_%04d.png")
    os.makedirs(os.path.dirname(output_video), exist_ok=True)

    create_video_from_frames(input_pattern, output_video, framerate, ffmpeg_path)
    print(f"Video generation completed: {output_video}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='3D Neuron Video Generator (Single Process)')
    parser.add_argument('--h5-file', required=True, help='H5 file path')
    parser.add_argument('--node-types-csv', required=True, help='Node types CSV file path')
    parser.add_argument('--population-csv', required=True, help='Population data CSV file path')
    parser.add_argument('--spike-file', required=True, help='Spike data file path')
    parser.add_argument('--output-dir', default='output/frames', help='Frame output directory')
    parser.add_argument('--output-video', default='output/videos/column_video.mp4', help='Video output file path')
    parser.add_argument('--frames', type=int, default=500, help='Total number of frames')
    parser.add_argument('--duration', type=int, default=1000, help='Video duration (ms)')
    parser.add_argument('--spike-duration', type=int, default=10, help='Spike display duration (ms)')
    parser.add_argument('--ffmpeg-path', default='ffmpeg', help='ffmpeg executable path')
    parser.add_argument('--framerate', type=int, default=10, help='Frame rate')
    parser.add_argument('--sample-rate', type=float, default=1.0, help='Sampling rate (1.0 = all data)')
    parser.add_argument('--start-time', type=int, default=0, help='Start time (ms)')

    args = parser.parse_args()

    create_column_video_single(
        h5_file=args.h5_file,
        node_types_csv=args.node_types_csv,
        population_csv=args.population_csv,
        spike_file=args.spike_file,
        output_dir=args.output_dir,
        output_video=args.output_video,
        total_frames=args.frames,
        duration_ms=args.duration,
        spike_duration_ms=args.spike_duration,
        ffmpeg_path=args.ffmpeg_path,
        framerate=args.framerate,
        sample_rate=args.sample_rate,
        start_time=args.start_time
    )
