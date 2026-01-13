# LightGlue Structure-from-Motion Test

This repository contains a multi-cell tiepoint matching and Structure-from-Motion (SfM) pipeline using LightGlue for feature matching. The system processes drone imagery organized by H3 geospatial cells, computing intra-cell and inter-cell matches, linking tracks across cells, and reconstructing 3D point clouds.

## Features

- **Multi-Cell Processing**: Handles both intra-cell (within a cell) and inter-cell (between cells) feature matching
- **LightGlue Feature Matching**: Uses ALIKED or SuperPoint feature extractors with LightGlue matcher
- **Epipolar Filtering**: Robust RANSAC-based filtering of matches using epipolar geometry
- **Track Linking**: Automatically links tracks across cells to form longer, unified tracks
- **Tiepoint Graph**: Maintains a graph structure for managing features, matches, and tracks across cells
- **Point Cloud Reconstruction**: Aerial triangulation and bundle adjustment for 3D reconstruction
- **Oblique Imagery Support**: Handles both nadir and oblique camera angles (30° and 60°)
- **Footprint Overlap Filtering**: Only matches image pairs with sufficient ground footprint overlap

## Requirements

- Python 3.14+
- PyTorch (with MPS support for Apple Silicon)
- lightglue
- numpy
- shapely
- PIL/Pillow
- tqdm

## Installation

```bash
pip install torch lightglue numpy shapely pillow tqdm
```

## Usage

### Single Cell Processing

Process a single H3 cell:

```bash
python matcher.py --mode ortho --resolution 1.0 --min-overlap 50.0
```

### Multi-Cell Processing

Process multiple cells with intra-cell and inter-cell matching:

1. Create a JSON file with cell IDs:
```json
["cell_id_1", "cell_id_2", "cell_id_3"]
```

2. Run the matcher:
```bash
python matcher.py \
  --cell-ids-file inputs/input_cells.json \
  --mode ortho \
  --resolution 1.0 \
  --output-dir outputs_multi_cell \
  --image-dir /path/to/images \
  --min-overlap 50.0
```

### Parameters

- `--mode`: Processing mode - `ortho` (nadir-only) or `hybrid` (nadir+oblique)
- `--resolution`: Image resolution scale factor (1.0 = full resolution, 0.5 = half, 0.25 = quarter)
- `--min-overlap`: Minimum footprint overlap percentage (default: 50.0)
- `--cell-ids-file`: Path to JSON file with list of cell IDs for multi-cell processing
- `--output-dir`: Override output directory (default: `outputs/` for ortho, `outputs_hybrid/` for hybrid)
- `--image-dir`: Override image directory
- `--no-epipolar-filtering`: Disable epipolar geometry filtering (filtering enabled by default)

## Output Structure

```
outputs/
├── cell_analysis.json              # Analysis of all H3 cells found
├── central_cell_info.json          # Information about the central cell
├── camera_poses.json               # Camera poses extracted from EXIF
├── camera_poses_3d.ply             # 3D visualization of camera poses
├── footprint_overlaps.json        # Footprint overlap percentages
├── features_cache_cell_*.json     # Cached features per cell
├── matches_unfiltered_cell_*.json # Unfiltered matches per cell
├── matches_filtered_cell_*.json   # Epipolar-filtered matches per cell
├── tracks_cell_*.json             # Tracks per cell
├── tracks_filtered_cell_*.json    # Filtered tracks per cell
├── point_cloud_cell_*.ply         # Reconstructed 3D point cloud
├── visualization/                 # Track and feature visualizations
├── visualize_tracks_on_images/    # Track visualizations on image triplets
└── run_log.txt                    # Full run log

# Multi-cell outputs:
├── cell_<cell_id>/                # Per-cell outputs
├── inter_<cell1>_<cell2>/         # Inter-cell outputs
├── linked_tracks.json            # Linked tracks across cells
└── tiepoint_graph.json           # Complete tiepoint graph
```

## Architecture

### Multi-Cell Processing

The system supports processing multiple H3 cells with the following workflow:

1. **Intra-Cell Processing**: For each cell, compute features, matches, and tracks within that cell
2. **Inter-Cell Processing**: For each pair of cells, compute matches between images in different cells
3. **Track Linking**: Link tracks that span multiple cells to form longer, unified tracks
4. **Tiepoint Graph**: Maintain a graph structure that tracks all features, matches, and tracks with their cell associations

### Feature Extractors

The system automatically selects the best available feature extractor:
- **ALIKED** (preferred): High-quality feature detection
- **SuperPoint**: Fallback if ALIKED not available
- **DISK**: Alternative extractor
- **SIFT**: Traditional feature detector

### Epipolar Filtering

Matches are filtered using robust RANSAC-based epipolar geometry validation:
- RANSAC threshold: 0.5 pixels
- Minimum inlier ratio: 50%
- Confidence: 0.999

## File Structure

```
research-lightglue_sfm_test/
├── matcher.py                     # Main entry point
├── matcher/
│   ├── __init__.py
│   ├── lightglue_matcher.py      # LightGlue feature extraction and matching
│   ├── multi_cell_processor.py   # Multi-cell processing logic
│   ├── tiepoint_graph.py         # Tiepoint graph data structure
│   ├── epipolar_validation.py    # Epipolar geometry filtering
│   ├── track_analysis.py         # Track computation and analysis
│   ├── cameras.py                # Camera pose extraction
│   ├── utils.py                  # Utility functions (footprints, etc.)
│   ├── visualization.py          # Visualization functions
│   ├── robust_reconstruction.py  # 3D reconstruction with bundle adjustment
│   ├── triangulate_from_matches.py # Direct triangulation from matches
│   └── ...
├── inputs/                       # Input files (cell IDs, etc.)
└── outputs/                       # Output directory
```

## License

[Add your license here]
