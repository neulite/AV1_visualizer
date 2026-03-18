"""
Visualization script for Neulite spike data

Visualizes Neulite parallel execution output (s0.dat to sN.dat)
as raster plots and 3D videos using AV1_visualizer.

Usage:
    python visualize_neulite.py \
        --neulite-dir path/to/neulite_output \
        --h5-file path/to/v1_nodes.h5 \
        --node-types-csv path/to/v1_node_types.csv \
        --population-template path/to/v1_population.csv \
        --output-dir output
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# Add AV1_visualizer to sys.path
AV1_VISUALIZER_DIR = Path(__file__).parent
sys.path.insert(0, str(AV1_VISUALIZER_DIR))

from src.raster.v1_raster import create_v1_raster
from src.video.video_generator_parallel import create_column_video_parallel


# =============================================================================
# Data Preparation
# =============================================================================

def merge_neulite_spikes(neulite_dir: Path) -> np.ndarray:
    """Load and merge all Neulite parallel output files (s0.dat to sN.dat)

    Returns:
        np.ndarray: shape=(N, 2), columns are [time, neuron_id]
    """
    dat_files = sorted(neulite_dir.glob("s*.dat"),
                       key=lambda p: int(p.stem[1:]))
    print(f"  Neulite parallel output files: {len(dat_files)}")

    all_data = []
    for dat_file in dat_files:
        if dat_file.stat().st_size == 0:
            continue
        data = np.loadtxt(dat_file)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        all_data.append(data)

    if not all_data:
        raise ValueError("No spike data found")

    return np.vstack(all_data)


def generate_population_csv(h5_file: Path, template_csv: Path,
                            output_path: Path) -> Path:
    """Generate a population CSV adjusted to the actual node count in the H5 file

    Uses n_comp, swc_file, and ion_file from the template CSV,
    replacing #n_cell with the actual count from H5.
    """
    template = pd.read_csv(template_csv)

    with h5py.File(h5_file) as f:
        type_ids = f["nodes/v1/node_type_id"][:]

    # Count actual nodes per node_type_id
    type_id_counts = pd.Series(type_ids).value_counts()

    # Extract node_type_id from the template (suffix of name column is _<id>)
    rows = []
    for _, row in template.iterrows():
        name = row["name"]
        tid = int(name.split("_")[-1])
        actual_count = int(type_id_counts.get(tid, 0))
        rows.append({
            "#n_cell": actual_count,
            "n_comp": row["n_comp"],
            "name": name,
            "swc_file": row["swc_file"],
            "ion_file": row["ion_file"],
        })

    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    total = result["#n_cell"].sum()
    print(f"  Population CSV generated: {output_path} (total cells: {total})")
    return output_path


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize Neulite spike data as raster plots and 3D videos"
    )
    parser.add_argument("--neulite-dir", required=True,
                        help="Directory containing Neulite parallel output files (s0.dat to sN.dat)")
    parser.add_argument("--h5-file", required=True,
                        help="Path to v1_nodes.h5 (neuron coordinate data)")
    parser.add_argument("--node-types-csv", required=True,
                        help="Path to v1_node_types.csv")
    parser.add_argument("--population-template", required=True,
                        help="Path to population CSV template (v1_population.csv)")
    parser.add_argument("--output-dir", default="output",
                        help="Output directory (default: output)")
    parser.add_argument("--skip-video", action="store_true",
                        help="Skip 3D video generation (raster plot only)")
    parser.add_argument("--total-frames", type=int, default=20,
                        help="Total number of video frames (default: 20)")
    parser.add_argument("--duration-ms", type=int, default=1500,
                        help="Video duration in milliseconds (default: 1500)")
    parser.add_argument("--sample-rate", type=float, default=0.05,
                        help="Neuron sampling rate for video (default: 0.05)")
    parser.add_argument("--start-time", type=int, default=500,
                        help="Start time in milliseconds (default: 500)")
    parser.add_argument("--processes", type=int, default=2,
                        help="Number of parallel processes for video (default: 2)")

    args = parser.parse_args()

    neulite_dir = Path(args.neulite_dir)
    h5_file = Path(args.h5_file)
    node_types_csv = Path(args.node_types_csv)
    population_template = Path(args.population_template)
    output_dir = Path(args.output_dir)

    print("=" * 50)
    print("Neulite Spike Data Visualization (AV1_visualizer)")
    print("=" * 50)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate population CSV (adjusted to actual node count in H5)
    print("\nGenerating population CSV...")
    population_csv = output_dir / "v1_population.csv"
    generate_population_csv(h5_file, population_template, population_csv)

    # Load and merge spike data
    print("\nMerging spike data...")
    spikes = merge_neulite_spikes(neulite_dir)
    print(f"  Merged spike count: {len(spikes)}")
    print(f"  Time range: {spikes[:, 0].min():.1f} - {spikes[:, 0].max():.1f} ms")
    print(f"  Neuron count: {len(np.unique(spikes[:, 1]))}")

    # Write merged data to a temporary file
    merged_spike_file = output_dir / "neulite_spikes_merged.dat"
    np.savetxt(str(merged_spike_file), spikes, fmt="%.6f %d")
    print(f"\n  Merged file created: {merged_spike_file}")

    # Generate raster plot
    print("\nGenerating raster plot...")
    raster_path = output_dir / "neulite_raster.png"
    create_v1_raster(
        spike_file=str(merged_spike_file),
        population_csv=str(population_csv),
        output_path=str(raster_path),
    )

    # Generate 3D video
    if not args.skip_video:
        print("\nGenerating 3D video...")
        frames_dir = str(output_dir / "frames")
        video_path = str(output_dir / "videos" / "neulite_column.mp4")
        create_column_video_parallel(
            h5_file=str(h5_file),
            node_types_csv=str(node_types_csv),
            population_csv=str(population_csv),
            spike_file=str(merged_spike_file),
            output_dir=frames_dir,
            output_video=video_path,
            total_frames=args.total_frames,
            duration_ms=args.duration_ms,
            spike_duration_ms=10,
            sample_rate=args.sample_rate,
            start_time=args.start_time,
            n_processes=args.processes,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
