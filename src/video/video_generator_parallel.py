"""3D Column Video Generator (Parallel Processing Version)

This module provides functions to generate 3D column videos using parallel processing.
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
import multiprocessing as mp

# Global variables for worker processes
_worker_res = None
_worker_spike_data = None
_worker_swc_cache = None


def _init_worker(res, spike_data, swc_cache):
    """Initialize worker process: store shared data in global variables"""
    global _worker_res, _worker_spike_data, _worker_swc_cache
    _worker_res = res
    _worker_spike_data = spike_data
    _worker_swc_cache = swc_cache


def process_frame(args):
    """Process a single frame (for parallel processing)

    Args:
        args: (frame, time_window, spike_duration, output_dir, total_frames)
    """
    frame, time_window, spike_duration, output_dir, total_frames = args
    current_time = frame * time_window
    elev = 360
    azim = frame * 360 / total_frames
    fig, ax = plotting.create_plot(_worker_res, _worker_spike_data, current_time, spike_duration, _worker_swc_cache, elev=elev, azim=azim)

    plotting.save_frame(frame, fig, output_dir)

    if frame % 10 == 0:
        print(f"Processing frame {frame}/{total_frames}")


def create_column_video_parallel(h5_file, node_types_csv, population_csv, spike_file,
                                  output_dir='output/frames', output_video='output/videos/column_video.mp4',
                                  total_frames=500, duration_ms=1000, spike_duration_ms=10,
                                  n_processes=None, ffmpeg_path='ffmpeg', framerate=10,
                                  sample_rate=1.0, start_time=0):
    """Generate a 3D neuron video (parallel processing version)

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
        n_processes: Number of parallel processes (None = auto: CPU count - 1)
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

    # Determine number of processes
    if n_processes is None:
        n_processes = max(1, os.cpu_count() - 1)
    print(f"Using {n_processes} processes")

    print("Starting frame generation...")
    start_time_proc = time.time()

    # Set up multiprocessing (pass shared data via initializer)
    pool = mp.Pool(
        processes=n_processes,
        initializer=_init_worker,
        initargs=(res, filtered_and_shifted_spike_data, swc_cache),
    )

    # Argument list for each frame (lightweight parameters only)
    args_list = [
        (frame, time_window, spike_duration_ms, output_dir, total_frames)
        for frame in range(total_frames)
    ]

    # Execute parallel processing
    pool.map(process_frame, args_list)

    # Close pool
    pool.close()
    pool.join()

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

    parser = argparse.ArgumentParser(description='3D Neuron Video Generator (Parallel Processing)')
    parser.add_argument('--h5-file', required=True, help='H5 file path')
    parser.add_argument('--node-types-csv', required=True, help='Node types CSV file path')
    parser.add_argument('--population-csv', required=True, help='Population data CSV file path')
    parser.add_argument('--spike-file', required=True, help='Spike data file path')
    parser.add_argument('--output-dir', default='output/frames', help='Frame output directory')
    parser.add_argument('--output-video', default='output/videos/column_video.mp4', help='Video output file path')
    parser.add_argument('--frames', type=int, default=500, help='Total number of frames')
    parser.add_argument('--duration', type=int, default=1000, help='Video duration (ms)')
    parser.add_argument('--spike-duration', type=int, default=10, help='Spike display duration (ms)')
    parser.add_argument('--processes', type=int, default=None,
                        help=f'Number of parallel processes (default: CPU count - 1 = {max(1, os.cpu_count()-1)})')
    parser.add_argument('--ffmpeg-path', default='ffmpeg', help='ffmpeg executable path')
    parser.add_argument('--framerate', type=int, default=10, help='Frame rate')
    parser.add_argument('--sample-rate', type=float, default=1.0, help='Sampling rate (1.0 = all data)')
    parser.add_argument('--start-time', type=int, default=0, help='Start time (ms)')

    args = parser.parse_args()

    create_column_video_parallel(
        h5_file=args.h5_file,
        node_types_csv=args.node_types_csv,
        population_csv=args.population_csv,
        spike_file=args.spike_file,
        output_dir=args.output_dir,
        output_video=args.output_video,
        total_frames=args.frames,
        duration_ms=args.duration,
        spike_duration_ms=args.spike_duration,
        n_processes=args.processes,
        ffmpeg_path=args.ffmpeg_path,
        framerate=args.framerate,
        sample_rate=args.sample_rate,
        start_time=args.start_time
    )
