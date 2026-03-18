# AV1 Visualizer

Neural activity visualization tool for V1 cortical column (Allen V1 Column Visualizer)

## Features

- **V1 Raster Plot Generation**: Spike firing pattern visualization for 21 neuron types
- **3D Neuron Video Generation**: 3D animation of spike activity over time
  - Parallel processing version: High-speed parallel processing (recommended)
  - Single process version: For debugging and simple execution

## Installation

### Requirements

- Python 3.7+
- ffmpeg (required for video generation)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install ffmpeg

**Windows:**
```bash
# Using Chocolatey
choco install ffmpeg

# Or manual install
# Download from https://ffmpeg.org/download.html
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt-get install ffmpeg
```

## Usage

### V1 Raster Plot Generation

```bash
python -m src.raster.v1_raster \
  --spike-file data/spike/v1_spike.dat \
  --population-csv data/v1_population.csv \
  --output output/rasters/v1_raster.png
```

**Options:**
- `--time-max`: Maximum time to display (ms)
- `--time-min`: Minimum time to display (ms)

### 3D Video Generation

#### Parallel Processing Version (Recommended)

```bash
python -m src.video.video_generator_parallel \
  --h5-file data/v1_nodes.h5 \
  --node-types-csv data/v1_node_types.csv \
  --population-csv data/v1_population.csv \
  --spike-file data/spike/v1_spike.dat \
  --output-video output/videos/column_video.mp4 \
  --processes 8
```

**Options:**
- `--frames`: Number of frames (default: 500)
- `--duration`: Video duration in milliseconds (default: 1000)
- `--spike-duration`: Spike display duration in milliseconds (default: 10)
- `--processes`: Number of parallel processes (default: CPU count - 1)
- `--framerate`: Frame rate (default: 10)
- `--ffmpeg-path`: Path to ffmpeg executable (default: 'ffmpeg')

#### Single Process Version (For Debugging)

```bash
python -m src.video.video_generator_single \
  --h5-file data/v1_nodes.h5 \
  --node-types-csv data/v1_node_types.csv \
  --population-csv data/v1_population.csv \
  --spike-file data/spike/v1_spike.dat \
  --output-video output/videos/column_video.mp4
```

**Options:**
- `--frames`: Number of frames (default: 500)
- `--framerate`: Frame rate (default: 10)

## Sample Code

### Using from Python Script

```python
from src.raster.v1_raster import create_v1_raster
from src.video.video_generator_parallel import create_column_video_parallel

# Generate raster plot
create_v1_raster(
    spike_file='data/spike/v1_spike.dat',
    population_csv='data/v1_population.csv',
    output_path='output/rasters/v1_raster.png'
)

# Generate video (parallel processing version)
create_column_video_parallel(
    h5_file='data/v1_nodes.h5',
    node_types_csv='data/v1_node_types.csv',
    population_csv='data/v1_population.csv',
    spike_file='data/spike/v1_spike.dat',
    output_video='output/videos/column_video.mp4',
    n_processes=8
)
```

## Directory Structure

```
AV1_visualizer/
├── src/               # Source code
│   ├── raster/        # Raster plot
│   ├── video/         # Video generation
│   ├── core/          # Core functionality
│   └── utils/         # Utilities
├── config/            # Configuration files
├── data/              # Data directory
└── output/            # Output directory
```

## Troubleshooting

### Import errors occur

Run from the project root directory:

```bash
cd /path/to/AV1_visualizer
python -m src.raster.v1_raster --help
```

### ffmpeg not found

Verify that ffmpeg is included in the system PATH:

```bash
ffmpeg -version
```

If not installed, refer to the installation instructions above.

### Memory errors with parallel processing

Reduce the number of processes using the `--processes` option:

```bash
--processes 4  # Limit to 4 processes
```

## Data Attribution

The SWC morphology reconstruction files included in the `data/` directory are provided by the [Allen Institute for Brain Science](https://alleninstitute.org/) via the [Cell Types Database](https://celltypes.brain-map.org/). These data are used under the [Allen Institute Terms of Use](https://alleninstitute.org/terms-of-use/) for noncommercial research purposes.

If you use these data, please cite them in accordance with the [Allen Institute Citation Policy](https://alleninstitute.org/citation-policy/).

## License

The source code in this repository is licensed under the [GNU General Public License v2.0](LICENSE).

Note: The SWC morphology data files in the `data/` directory are **not** covered by GPL-2.0. They are subject to the [Allen Institute Terms of Use](https://alleninstitute.org/terms-of-use/) (noncommercial use only). See [Data Attribution](#data-attribution) for details.

## References

- [Neulite](https://neulite.github.io/en/)
- [BMTK (Brain Modeling Toolkit)](https://alleninstitute.github.io/bmtk/)
- [Allen Institute Cell Types Database](https://celltypes.brain-map.org/)
