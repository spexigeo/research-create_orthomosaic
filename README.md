# research-create_orthomosaic

Tiepoint matcher for orthomosaic creation from drone imagery organized by H3 cells.

## Overview

This module provides a tiepoint matcher for creating orthomosaics from drone imagery. The system is designed to work with images organized by H3 cells, where each cell contains multiple images captured at different angles (typically 30°, 60°, and nadir views).

The tiepoint matcher uses LightGlue for feature detection and matching, and exports results in MetaShape-compatible formats for integration with Agisoft Metashape workflows.

## Features

- **INTRA-cell matching**: Match features within a single H3 cell
- **LightGlue-based matching**: State-of-the-art feature matching using LightGlue
- **Footprint overlap filtering**: Only match image pairs with sufficient ground coverage overlap (default: 50%)
- **Memory-efficient**: Uses top 200 features per image (configurable)
- **Parallel processing**: Multi-threaded matching for faster processing
- **MetaShape export**: Export tiepoints in MetaShape-compatible JSON format
- **Track computation**: Generate feature tracks across multiple images
- **Point cloud reconstruction**: 3D reconstruction from feature tracks
- **Visualization**: Generate visualizations of features, matches, tracks, and overlap statistics
- **H3 cell utilities**: Find central cells and organize images by H3 cells
- **Caching**: Automatic caching of features and matches for faster re-runs

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Usage

### Running the Test Script

```bash
python matcher.py [--min-overlap MIN_OVERLAP]
```

**Options:**
- `--min-overlap`: Minimum footprint overlap percentage to consider for matching (default: 50.0)

**Example:**
```bash
# Use default 50% overlap threshold
python matcher.py

# Use 30% overlap threshold
python matcher.py --min-overlap 30.0
```

### What the Script Does

1. **Parse images**: Extracts H3 cell information from image filenames
2. **Find central cell**: Identifies the central H3 cell (completely surrounded by 6 neighbors)
3. **Create quarter-resolution images**: Generates downsampled images for faster processing
4. **Extract features**: Detects features using SuperPoint (top 200 per image)
5. **Match features**: Matches features between image pairs with sufficient footprint overlap
6. **Compute tracks**: Links matched features across multiple images to form tracks
7. **Reconstruct point cloud**: Performs 3D reconstruction from feature tracks
8. **Export results**: Saves matches, tracks, and point clouds in various formats
9. **Generate visualizations**: Creates visualizations of features, matches, tracks, and statistics

### Output Files

The script generates the following output files in the `outputs/` directory:

#### Core Data Files
- `features_cache_cell_{cell}.json`: Full feature cache with descriptors (for reloading)
- `features_cell_{cell}.json`: Feature summary with keypoints, scores, and descriptors
- `matches_cache_cell_{cell}.json`: Full match cache with confidences
- `matches_unfiltered_cell_{cell}.json`: Unfiltered matches (original matches from matcher)
- `tracks_cell_{cell}.json`: Feature tracks linking features across images
- `point_cloud_cell_{cell}.json`: 3D point cloud reconstruction
- `point_cloud_cell_{cell}.ply`: PLY file for 3D visualization

#### Export Files
- `tiepoints_cell_{cell}_metashape.json`: MetaShape-compatible tiepoint export
- `statistics_cell_{cell}.json`: Summary statistics

#### Visualization Files
- `feature_distribution_cell_{cell}.png`: Feature and match statistics
- `overlap_counts_per_image.png`: Number of overlapping images per image
- `track_length_histogram_cell_{cell}.png`: Distribution of track lengths
- `matches_visualization_cell_{cell}.png`: Visual match examples
- `visualization/camera_poses.png`: Camera position and orientation visualization
- `visualization/footprints_2d.png`: 2D footprint visualization
- `visualization/track_{id}_first_{n}_images.png`: Track visualizations showing matched features
- `visualize_single_track/track_{id}_first_{n}_images.png`: Individual track visualizations
- `visualize_tracks_on_images/tracks_{img1}_{img2}_{img3}.png`: Track visualizations on image triplets

#### Analysis Files
- `cell_analysis.json`: H3 cell organization analysis
- `central_cell_info.json`: Central cell identification
- `performance_summary_cell_{cell}.json`: Processing time statistics

### Programmatic Usage

```python
from matcher import (
    find_central_cell,
    get_cell_images,
    LightGlueMatcher,
    export_to_metashape,
    visualize_features_and_matches
)

# Get images organized by H3 cells
image_dir = "path/to/images"
cell_to_images = get_cell_images(image_dir)

# Find central cell (completely surrounded by 6 neighbors)
cell_ids = list(cell_to_images.keys())
central_cell = find_central_cell(cell_ids)

# Get images for central cell
cell_images = cell_to_images[central_cell]

# Initialize matcher (200 features per image, uses GPU if available)
matcher = LightGlueMatcher(
    extractor_type='superpoint',
    device=None,  # Auto-detect: CUDA, MPS (Apple Silicon), or CPU
    max_features_per_image=200
)

# Match features within cell
match_results = matcher.match_cell_images(cell_images)

# Export to MetaShape format
export_to_metashape(match_results, "output/tiepoints.json", image_base_path=image_dir)

# Visualize results
visualize_features_and_matches(match_results, cell_images, "output/visualization.png")
```

## Key Features and Optimizations

### Memory Optimization
- **Top 200 features per image**: Only the highest-scoring features are kept to reduce memory usage
- **Quarter-resolution matching**: Images are downsampled to 25% resolution for faster processing
- **Efficient caching**: Features and matches are cached to disk for faster re-runs

### Footprint Overlap Filtering
- **Smart pair selection**: Only matches image pairs with sufficient ground coverage overlap
- **Default threshold**: 50% overlap (configurable via `--min-overlap`)
- **Reduces computation**: Significantly fewer pairs to process (e.g., 1,161 vs 4,078 pairs at 50% vs 10%)

### Parallel Processing
- **Multi-threaded matching**: Uses up to 8 parallel workers for feature matching
- **Thread-safe inference**: PyTorch inference is thread-safe, allowing efficient parallelization
- **GPU acceleration**: Automatically uses GPU (CUDA or Apple Silicon MPS) when available

### Track Generation
- **Feature linking**: Links matched features across multiple images to form tracks
- **Track validation**: Filters tracks based on geometric consistency
- **Point cloud reconstruction**: Performs 3D triangulation from validated tracks

## Image Filename Format

Images should follow the format:
```
{h3_cell_index}_{flight_number}_{image_number}.jpg
```

Example: `8928d89ac57ffff_172550_0001.jpg`

## Output Formats

### Match Files

Matches are saved in JSON format with the following structure:
```json
{
  "image0": "image1.jpg",
  "image1": "image2.jpg",
  "num_matches": 285,
  "matches": [[0, 1], [2, 3], ...],
  "match_confidence": [0.95, 0.87, ...]
}
```

### Track Files

Tracks link features across multiple images:
```json
{
  "track_id": 0,
  "length": 5,
  "images": ["img1.jpg", "img2.jpg", ...],
  "features": [("img1.jpg", 0), ("img2.jpg", 1), ...]
}
```

### MetaShape JSON Format

The exported JSON file follows MetaShape's match file format:
- `cameras`: Dictionary of images with their features
- `matches`: List of match pairs between images with confidence scores
- `metadata`: Summary statistics

## Architecture

The module is organized into the `matcher/` package with the following structure:

### Core Modules
- `matcher/h3_utils.py`: H3 cell utilities and image filename parsing
- `matcher/lightglue_matcher.py`: LightGlue-based feature extraction and matching
- `matcher/metashape_export.py`: Export to MetaShape-compatible formats
- `matcher/track_analysis.py`: Track computation and analysis
- `matcher/point_cloud_reconstruction.py`: 3D point cloud reconstruction
- `matcher/robust_reconstruction.py`: Robust reconstruction with bundle adjustment
- `matcher/epipolar_validation.py`: Epipolar geometry validation (currently disabled)
- `matcher/image_utils.py`: Image processing utilities
- `matcher/ply_export.py`: PLY file export for 3D visualization

### Consolidated Modules
- `matcher/visualization.py`: All visualization functions consolidated:
  - `visualize_features_and_matches()`: Feature and match visualization
  - `visualize_feature_distribution()`: Feature statistics
  - `visualize_single_track()`: Single track visualization
  - `visualize_tracks_on_image_triplets()`: Track visualization on image triplets
  - `create_multiple_track_visualizations()`: Batch track visualization
  - `visualize_overlap_counts()`: Footprint overlap statistics
  - `visualize_camera_poses()`: Camera pose visualization
  - `visualize_footprints_2d()`: 2D footprint visualization

- `matcher/cameras.py`: Camera pose extraction and PLY export:
  - `extract_camera_pose()`: Extract pose from single image
  - `extract_poses_from_directory()`: Extract poses from directory
  - `export_camera_poses_to_ply()`: Export camera poses to PLY format
  - `compute_footprint_from_exif()`: Compute footprint from EXIF data

- `matcher/utils.py`: Utility functions:
  - `compute_all_overlaps()`: Compute footprint overlaps for all image pairs
  - `compute_overlap_percentage()`: Calculate overlap between two footprints
  - `compute_footprint_polygon()`: Compute footprint polygon from EXIF
  - `debug_track()`: Debug track matching
  - `detect_scanlines()`: Detect scanlines from camera poses
  - `verify_scanline_straightness()`: Verify scanline quality

### Main Script
- `matcher.py`: Main matching script (renamed from `test_tiepoint_matcher.py`)

## Dependencies

- `torch`: PyTorch for deep learning models
- `lightglue`: LightGlue feature matcher
- `h3`: H3 geospatial indexing
- `numpy`: Numerical computations
- `matplotlib`: Visualizations
- `pillow`: Image processing
- `opencv-python`: Computer vision utilities
- `tqdm`: Progress bars

## Performance

Typical performance on a dataset with 155 images:
- **Feature extraction**: ~2-3 minutes (cached after first run)
- **Matching**: ~7-10 minutes for 1,161 pairs with 50% overlap threshold (cached after first run)
- **Track computation**: ~1-2 seconds
- **Point cloud reconstruction**: ~10-20 seconds
- **Total time (with cache)**: ~20-30 seconds

## Notes

- **Match confidences**: LightGlue may not always return confidence scores, so the `match_confidence` field may be empty
- **Epipolar validation**: Currently disabled - using unfiltered matches only
- **Filtered matches**: No longer generated - only unfiltered matches are saved
- **Track length 2**: Tracks with only 2 images are still valid and included in results

## Future Work

- **INTER-cell matching**: Match features across neighboring H3 cells
- **Tiepoint graph**: Build and maintain a graph structure for incremental updates
- **Bundle adjustment integration**: Full bundle adjustment pipeline
- **Additional feature extractors**: Support for DISK, SIFT, ALIKED extractors
- **Epipolar validation**: Re-enable with improved filtering options
