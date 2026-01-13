"""
Test script for tiepoint matcher on Qualicum Beach data.
"""

import sys
import json
import numpy as np
try:
    import h3
    HAS_H3 = True
except ImportError:
    HAS_H3 = False
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from matcher import (
    find_central_cell,
    get_cell_images,
    LightGlueMatcher,
    export_to_metashape,
    visualize_features_and_matches,
    visualize_feature_distribution
)
from matcher.image_utils import (
    downsample_images_batch,
    save_feature_coordinates,
    load_feature_coordinates,
    save_features_full,
    load_features_full
)
from matcher.track_analysis import (
    compute_tracks,
    plot_track_length_histogram
)
from matcher.point_cloud_reconstruction import reconstruct_point_cloud
from matcher.robust_reconstruction import robust_reconstruct_with_bundle_adjustment
from matcher.ply_export import export_point_cloud_to_ply
from tqdm import tqdm


def load_footprint_overlaps(overlaps_file: str = "outputs/footprint_overlaps.json", 
                           min_overlap_threshold: float = 50.0) -> dict:
    """
    Load footprint overlaps and create a lookup dictionary.
    
    Args:
        overlaps_file: Path to footprint_overlaps.json
        min_overlap_threshold: Minimum overlap percentage to consider (default: 10%)
    
    Returns:
        Dictionary mapping (image1, image2) -> overlap_percentage for pairs above threshold
    """
    try:
        with open(overlaps_file, 'r') as f:
            data = json.load(f)
        
        overlaps_dict = {}
        for overlap in data['overlaps']:
            if overlap['overlap_percentage'] >= min_overlap_threshold:
                img1 = overlap['image1']
                img2 = overlap['image2']
                # Store both directions for easy lookup
                overlaps_dict[(img1, img2)] = overlap['overlap_percentage']
                overlaps_dict[(img2, img1)] = overlap['overlap_percentage']
        
        print(f"   Loaded {len(overlaps_dict) // 2} overlapping pairs (>= {min_overlap_threshold}% overlap)")
        return overlaps_dict
    except FileNotFoundError:
        print(f"   Warning: {overlaps_file} not found. Matching all pairs.")
        return {}
    except Exception as e:
        print(f"   Warning: Error loading overlaps: {e}. Matching all pairs.")
        return {}


def main(min_overlap_threshold: float = 50.0, enable_epipolar_filtering: bool = True, mode: str = 'ortho', resolution: float = 1.0, cell_ids_file: Optional[str] = None, output_dir_override: Optional[str] = None, image_dir_override: Optional[str] = None):
    """
    Main matching pipeline.
    
    Args:
        min_overlap_threshold: Minimum footprint overlap percentage (default: 50.0)
        enable_epipolar_filtering: Whether to enable epipolar filtering (default: True)
        mode: Processing mode - 'ortho' for nadir-only or 'hybrid' for nadir+oblique (default: 'ortho')
        resolution: Image resolution scale factor (default: 1.0 for full resolution). 
                    Use 0.125, 0.25, or 0.5 for downsampled versions.
    """
    # Paths based on mode (can be overridden)
    if image_dir_override:
        image_dir = Path(image_dir_override)
    elif mode == 'hybrid':
        image_dir = Path("/Users/mauriciohessflores/Documents/Code/Data/hybrid_flight_plan/images")
    else:  # ortho mode
        image_dir = Path("/Users/mauriciohessflores/Documents/Code/MyCode/research-qualicum_beach_gcp_analysis/input/images")
    
    if output_dir_override:
        output_dir = Path(output_dir_override)
    elif mode == 'hybrid':
        output_dir = Path("outputs_hybrid")
    else:  # ortho mode
        output_dir = Path("outputs")
    
    if mode == 'hybrid':
        print("=" * 60)
        print("Tiepoint Matcher - Hybrid Mode (Nadir + Oblique)")
        print("=" * 60)
    else:
        print("=" * 60)
        print("Tiepoint Matcher Test - Qualicum Beach Data (Ortho Mode)")
        print("=" * 60)
    
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Tiepoint Matcher Test - Qualicum Beach Data")
    print("=" * 60)
    
    # Step 1: Get all H3 cells from images
    print("\n1. Parsing image filenames to extract H3 cells...")
    cell_to_images = get_cell_images(str(image_dir))
    all_cell_ids = list(cell_to_images.keys())
    print(f"   Found {len(all_cell_ids)} H3 cells:")
    for cell_id in sorted(all_cell_ids):
        num_images = len(cell_to_images[cell_id])
        print(f"     {cell_id}: {num_images} images")
    
    # Save cell analysis
    cell_analysis = {
        'total_cells': len(all_cell_ids),
        'cells': {cell_id: {'num_images': len(images)} 
                 for cell_id, images in cell_to_images.items()}
    }
    cell_analysis_path = output_dir / "cell_analysis.json"
    with open(cell_analysis_path, 'w') as f:
        json.dump(cell_analysis, f, indent=2)
    print(f"   Saved cell analysis to: {cell_analysis_path}")
    
    # Determine which cells to process
    if cell_ids_file and Path(cell_ids_file).exists():
        # Load cell IDs from file
        with open(cell_ids_file, 'r') as f:
            cell_ids_data = json.load(f)
        if isinstance(cell_ids_data, list):
            cell_ids = cell_ids_data
        elif isinstance(cell_ids_data, dict) and 'cell_ids' in cell_ids_data:
            cell_ids = cell_ids_data['cell_ids']
        else:
            cell_ids = [cell_ids_data] if isinstance(cell_ids_data, str) else all_cell_ids
        
        # Filter to only cells that exist
        cell_ids = [cid for cid in cell_ids if cid in cell_to_images]
        print(f"   Processing {len(cell_ids)} cells from file: {cell_ids}")
    else:
        # Use original single-cell logic
        cell_ids = None
    
    # Check if we should use multi-cell processing
    if cell_ids and len(cell_ids) > 1:
        # Multi-cell processing
        print("\n" + "=" * 60)
        print("MULTI-CELL MODE: Processing intra-cell and inter-cell")
        print("=" * 60)
        
        from matcher.multi_cell_processor import process_multi_cell
        
        # Get camera parameters from defaults (can be made configurable)
        focal_length_mm = 8.8
        sensor_width_mm = 13.2
        sensor_height_mm = 9.9
        max_features = 200
        
        graph = process_multi_cell(
            cell_ids=cell_ids,
            image_dir=image_dir,
            output_dir=output_dir,
            min_overlap_threshold=min_overlap_threshold,
            enable_epipolar_filtering=enable_epipolar_filtering,
            max_features=max_features,
            resolution=resolution,
            focal_length_mm=focal_length_mm,
            sensor_width_mm=sensor_width_mm,
            sensor_height_mm=sensor_height_mm,
            mode=mode
        )
        
        print("\n" + "=" * 60)
        print("Multi-cell processing complete!")
        print("=" * 60)
        print(f"  Tiepoint graph saved to: {output_dir / 'tiepoint_graph.json'}")
        print(f"  Linked tracks saved to: {output_dir / 'linked_tracks.json'}")
        return
    
    # Single-cell processing (original logic)
    # Step 2: Find central cell
    print("\n2. Finding central cell (completely surrounded by 6 neighbors)...")
    if cell_ids and len(cell_ids) == 1:
        central_cell = cell_ids[0]
        central_cell_info = {
            'central_cell': central_cell,
            'is_central': False,
            'note': 'Single cell specified'
        }
    else:
        central_cell = find_central_cell(all_cell_ids)
        
        if central_cell is None:
            print("   WARNING: No central cell found. Using first cell for testing.")
            central_cell = all_cell_ids[0]
            central_cell_info = {
                'central_cell': central_cell,
                'is_central': False,
                'note': 'No cell with 6 neighbors found, using first cell'
            }
        else:
            print(f"   Found central cell: {central_cell}")
            if HAS_H3:
                neighbors = list(h3.grid_ring(central_cell, k=1))
            else:
                neighbors = []
            central_cell_info = {
                'central_cell': central_cell,
                'is_central': True,
                'neighbors': neighbors
            }
    
    if central_cell is None:
        print("   WARNING: No central cell found. Using first cell for testing.")
        central_cell = cell_ids[0]
        central_cell_info = {
            'central_cell': central_cell,
            'is_central': False,
            'note': 'No cell with 6 neighbors found, using first cell'
        }
    else:
        print(f"   Found central cell: {central_cell}")
        if HAS_H3:
            neighbors = list(h3.grid_ring(central_cell, k=1))
        else:
            neighbors = []
        central_cell_info = {
            'central_cell': central_cell,
            'is_central': True,
            'neighbors': neighbors
        }
    
    # Save central cell info
    central_cell_path = output_dir / "central_cell_info.json"
    with open(central_cell_path, 'w') as f:
        json.dump(central_cell_info, f, indent=2)
    print(f"   Saved central cell info to: {central_cell_path}")
    
    # Step 3: Get images for central cell
    print(f"\n3. Processing central cell: {central_cell}")
    cell_images = cell_to_images[central_cell]
    print(f"   Found {len(cell_images)} images in cell {central_cell}")
    
    # Step 3.5: Extract camera poses FIRST (REQUIRED before footprint overlap computation)
    # This MUST happen before Step 3.6 (compute_footprint_overlap) and Step 5.5 (matching)
    print("\n3.5. Extracting camera poses from images...")
    camera_poses_file = output_dir / "camera_poses.json"
    
    if not camera_poses_file.exists():
        print("   Camera poses file not found, extracting from images...")
        from matcher.cameras import extract_camera_pose
        
        camera_poses = []
        for img_path in tqdm(cell_images, desc="Extracting camera poses"):
            pose = extract_camera_pose(img_path)
            if pose:
                camera_poses.append(pose)
        
        # Save camera poses
        camera_poses_file.parent.mkdir(parents=True, exist_ok=True)
        with open(camera_poses_file, 'w') as f:
            json.dump(camera_poses, f, indent=2)
        print(f"   Saved {len(camera_poses)} camera poses to: {camera_poses_file}")
        print(f"   Images with GPS: {sum(1 for p in camera_poses if p.get('gps'))}")
        print(f"   Images with DJI orientation: {sum(1 for p in camera_poses if p.get('dji_orientation'))}")
    else:
        print(f"   Camera poses file already exists: {camera_poses_file}")
    
    # Step 3.6: Compute footprint overlaps (REQUIRED before matching)
    # This MUST happen after Step 3.5 (extract_camera_poses) and before Step 5.5 (matching)
    print("\n3.6. Computing footprint overlaps...")
    from matcher.utils import compute_all_overlaps
    footprint_overlaps_file = output_dir / "footprint_overlaps.json"
    
    # Verify camera_poses.json exists (required for footprint overlap computation)
    if not camera_poses_file.exists():
        raise FileNotFoundError(
            f"Camera poses file not found: {camera_poses_file}\n"
            f"  Step 3.5 (extract_camera_poses) must complete before Step 3.6 (compute_footprint_overlap)."
        )
    
    if not footprint_overlaps_file.exists():
        print(f"   Computing footprint overlaps (will filter pairs with < {min_overlap_threshold}% overlap)...")
        csv_file = str(footprint_overlaps_file).replace('.json', '.csv')
        compute_all_overlaps(
            image_dir="",  # Not used, kept for compatibility
            output_dir="",  # Not used, kept for compatibility
            camera_poses_file=str(camera_poses_file),
            output_json_file=str(footprint_overlaps_file),
            output_csv_file=csv_file,
            focal_length_mm=8.8,
            sensor_width_mm=13.2,
            sensor_height_mm=9.9
        )
        print(f"   Saved footprint overlaps to: {footprint_overlaps_file}")
    else:
        print(f"   Footprint overlaps file already exists: {footprint_overlaps_file}")
    
    # Step 3.6.5: Visualize camera poses and footprints
    print("\n3.6.5. Creating camera pose and footprint visualizations...")
    from matcher.visualization import visualize_camera_poses, visualize_footprints_2d
    
    # Create visualization directory
    viz_dir = output_dir / "visualization"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize camera poses
    camera_poses_viz = viz_dir / "camera_poses.png"
    visualize_camera_poses(
        poses_file=str(camera_poses_file),
        output_file=str(camera_poses_viz)
    )
    
    # Visualize all footprints
    footprints_viz = viz_dir / "footprints_2d.png"
    visualize_footprints_2d(
        camera_poses_file=str(camera_poses_file),
        output_file=str(footprints_viz),
        focal_length_mm=8.8,
        sensor_width_mm=13.2,
        sensor_height_mm=9.9,
        show_image_numbers=True,
        alpha=0.3
    )
    
    # Visualize nadir footprints only (rectangular)
    footprints_nadir_viz = viz_dir / "footprints_2d_nadir.png"
    visualize_footprints_2d(
        camera_poses_file=str(camera_poses_file),
        output_file=str(footprints_nadir_viz),
        focal_length_mm=8.8,
        sensor_width_mm=13.2,
        sensor_height_mm=9.9,
        show_image_numbers=True,
        alpha=0.3,
        filter_pitch=-90.0,
        filter_pitch_tolerance=1.0
    )
    
    # Visualize oblique footprints only (trapezoidal)
    footprints_oblique_viz = viz_dir / "footprints_2d_oblique.png"
    visualize_footprints_2d(
        camera_poses_file=str(camera_poses_file),
        output_file=str(footprints_oblique_viz),
        focal_length_mm=8.8,
        sensor_width_mm=13.2,
        sensor_height_mm=9.9,
        show_image_numbers=True,
        alpha=0.3,
        filter_oblique_only=True,
        filter_pitch_tolerance=1.0
    )
    
    # Step 3.7: Prepare images at requested resolution
    print(f"\n3.7. Preparing images at resolution scale {resolution}...")
    from matcher.utils import get_image_dimensions, create_downsampled_images
    
    if resolution == 1.0:
        # Full resolution: use images directly
        print("   Using full-resolution images")
        processed_images_dir = image_dir
        processed_images = [str(img_path) for img_path in cell_images]
    else:
        # Downsampled: create or use existing downsampled images
        resolution_str = str(resolution).replace('.', '_')  # e.g., 0.25 -> 0_25
        processed_images_dir = Path("inputs") / f"images_{resolution_str}"
        
        if processed_images_dir.exists() and any(processed_images_dir.glob('*.jpg')):
            print(f"   Using existing downsampled images in {processed_images_dir}")
            # Map original images to downsampled images
            processed_images = []
            for img_path in cell_images:
                img_name = Path(img_path).name
                downsampled_path = processed_images_dir / img_name
                if downsampled_path.exists():
                    processed_images.append(str(downsampled_path))
                else:
                    print(f"   Warning: Downsampled image not found: {downsampled_path}")
        else:
            print(f"   Creating downsampled images in {processed_images_dir}...")
            processed_images = create_downsampled_images(cell_images, processed_images_dir, resolution)
    
    print(f"   Found {len(processed_images)} images at resolution {resolution}")
    
    # Read actual image dimensions from the first image
    if processed_images:
        processed_width, processed_height = get_image_dimensions(processed_images[0])
        print(f"   Image dimensions (read from images): {processed_width}x{processed_height}")
    else:
        raise ValueError("No images found to read dimensions from")
    
    # Use processed_width/height for all subsequent operations
    quarter_res_width = processed_width
    quarter_res_height = processed_height
    quarter_res_images = processed_images
    quarter_res_dir = processed_images_dir
    
    # Export camera poses to 3D PLY (red dots for positions, blue arrows for orientations)
    print("\n3.7.5. Exporting camera poses to 3D PLY...")
    from matcher.cameras import export_camera_poses_to_ply
    
    camera_poses_ply = output_dir / "camera_poses_3d.ply"
    export_camera_poses_to_ply(
        camera_poses_file=str(camera_poses_file),
        output_file=str(camera_poses_ply),
        image_width=processed_width,
        image_height=processed_height,
        sphere_radius=5.0,
        arrow_length=20.0,
        draw_footprints=False  # Don't draw footprints in 3D PLY, just cameras
    )
    print(f"   Saved camera poses 3D PLY to: {camera_poses_ply}")
    
    # Step 4: Initialize matcher
    print("\n4. Initializing LightGlue matcher...")
    max_features = 200  # Top features per image for matching
    matcher = LightGlueMatcher(extractor_type='superpoint', device=None, 
                               max_features_per_image=max_features)
    print(f"   Using device: {matcher.device}")
    print(f"   Using extractor: {matcher.extractor_type}")
    print(f"   Max features per image: {max_features}")
    
    # Check for cached features
    features_cache_path = output_dir / f"features_cache_cell_{central_cell}.json"
    use_cache = features_cache_path.exists()
    if use_cache:
        print(f"   Found cached features at: {features_cache_path}")
    
    # Step 5: Extract features (with caching)
    print("\n5. Extracting features from images...")
    script_start_time = time.time()
    feature_time = 0
    
    # Check for cached features
    features_cache_path = output_dir / f"features_cache_cell_{central_cell}.json"
    use_cache = features_cache_path.exists()
    
    if use_cache:
        print(f"   Loading features from cache: {features_cache_path}")
        try:
            cached_features = load_features_full(str(features_cache_path))
            
            # Verify all images are in cache
            quarter_res_image_names = {Path(img).name for img in quarter_res_images}
            cached_image_names = {Path(data['image_path']).name for data in cached_features.values()}
            
            if quarter_res_image_names == cached_image_names:
                print("   All images found in cache, using cached features")
                # Filter cached features to top N based on scores (if needed)
                print(f"   Filtering cached features to top {max_features} per image...")
                filtered_features = {}
                for img_path, feat_data in cached_features.items():
                    keypoints = np.array(feat_data['keypoints'])
                    scores = np.array(feat_data['scores'])
                    descriptors = np.array(feat_data.get('descriptors', []))
                    
                    if len(keypoints) > max_features:
                        # Sort by score and take top N
                        top_indices = np.argsort(scores)[::-1][:max_features]
                        top_indices = np.sort(top_indices)  # Keep original order
                        
                        filtered_features[img_path] = {
                            'keypoints': keypoints[top_indices],
                            'scores': scores[top_indices],
                            'descriptors': descriptors[top_indices] if len(descriptors) > 0 else [],
                            'image_path': feat_data.get('image_path', img_path),
                            # Note: tensor versions will be recreated in match_features if needed
                            # but match indices will be relative to the filtered feature set (0 to max_features-1)
                        }
                    else:
                        filtered_features[img_path] = feat_data
                
                # Reconstruct match_results structure from cache
                match_results = {
                    'features': filtered_features,
                    'matches': []  # Will be computed in matching step
                }
                feature_time = 0  # No time spent extracting
            else:
                print(f"   Cache incomplete (found {len(cached_image_names)}, need {len(quarter_res_image_names)}), re-extracting...")
                use_cache = False
        except Exception as e:
            print(f"   Error loading cache: {e}, re-extracting...")
            use_cache = False
    
    if not use_cache:
        # Extract features from quarter-resolution images
        print(f"   Extracting features from {len(quarter_res_images)} quarter-resolution images...")
        feature_start = time.time()
        
        # Extract features only (no matching yet)
        features = {}
        for idx, image_path in enumerate(tqdm(quarter_res_images, desc="Extracting features")):
            image_name = Path(image_path).name
            if (idx + 1) % 20 == 0:
                print(f"\n[{idx+1}/{len(quarter_res_images)}] Processing {image_name}...")
            feat_dict = matcher.extract_features(image_path)
            features[image_path] = feat_dict
            
            # Output feature coordinates for progress tracking
            num_features = len(feat_dict['keypoints'])
            if (idx + 1) % 20 == 0:
                print(f"  Found {num_features} features")
                if num_features > 0:
                    kpts = feat_dict['keypoints']
                    print(f"  First feature: ({kpts[0][0]:.1f}, {kpts[0][1]:.1f})")
                    if num_features > 1:
                        print(f"  Last feature: ({kpts[-1][0]:.1f}, {kpts[-1][1]:.1f})")
        
        feature_time = time.time() - feature_start
        print(f"   Feature extraction completed in {feature_time:.1f} seconds")
        
        # Save features cache
        save_features_full(features, str(features_cache_path))
        
        match_results = {
            'features': features,
            'matches': []  # Will be computed in matching step
        }
    
    # Step 5.5: Match features (with caching)
    # NOTE: Camera poses (Step 3.5) and footprint overlaps (Step 3.6) MUST be computed BEFORE matching
    print("\n5.5. Matching features across image pairs...")
    
    # Verify that required files exist before matching
    camera_poses_file = output_dir / "camera_poses.json"
    footprint_overlaps_file = output_dir / "footprint_overlaps.json"
    
    if not camera_poses_file.exists():
        raise FileNotFoundError(
            f"Camera poses file not found: {camera_poses_file}\n"
            f"  Camera poses must be extracted (Step 3.5) before matching can proceed."
        )
    
    if not footprint_overlaps_file.exists():
        raise FileNotFoundError(
            f"Footprint overlaps file not found: {footprint_overlaps_file}\n"
            f"  Footprint overlaps must be computed (Step 3.6) before matching can proceed."
        )
    
    # Check for cached matches
    matches_cache_path = output_dir / f"matches_cache_cell_{central_cell}.json"
    use_matches_cache = matches_cache_path.exists()
    
    # Match all pairs with sufficient footprint overlap
    max_pairs = None  # None = match all pairs that pass overlap filter
    
    if use_matches_cache and len(match_results['matches']) == 0:
        print(f"   Loading matches from cache: {matches_cache_path}")
        try:
            with open(matches_cache_path, 'r') as f:
                matches_data = json.load(f)
            match_results['matches'] = matches_data
            print(f"   Loaded {len(matches_data)} match pairs from cache")
            match_time = 0
        except Exception as e:
            print(f"   Error loading matches cache: {e}, re-computing...")
            use_matches_cache = False
    
    if not use_matches_cache or len(match_results['matches']) == 0:
        # Need to do matching
        # Load footprint overlaps to filter pairs (already computed in Step 3.6)
        print(f"   Loading footprint overlaps (minimum threshold: {min_overlap_threshold}%)...")
        footprint_overlaps = load_footprint_overlaps(
            overlaps_file=str(footprint_overlaps_file),
            min_overlap_threshold=min_overlap_threshold
        )
        if footprint_overlaps:
            print(f"   Footprint overlap filtering ENABLED - will match only pairs with >= {min_overlap_threshold}% overlap")
        else:
            raise ValueError(
                f"No footprint overlaps found! Check {footprint_overlaps_file}.\n"
                f"  Make sure Step 3.6 (compute_footprint_overlap) completed successfully."
            )
        
        # Scanline analysis no longer used - footprint overlap filtering is more robust
        
        # Count pairs that actually pass the overlap filter
        if footprint_overlaps:
            pairs_passing_filter = len(footprint_overlaps)
            print(f"   Found {pairs_passing_filter} image pairs with >= {min_overlap_threshold}% overlap")
        else:
            print(f"   WARNING: No footprint overlaps loaded - will match ALL pairs!")
            pairs_passing_filter = len(quarter_res_images) * (len(quarter_res_images) - 1) // 2
        
        total_pairs = len(quarter_res_images) * (len(quarter_res_images) - 1) // 2
        pairs_to_match = min(max_pairs, pairs_passing_filter) if max_pairs else pairs_passing_filter
        print(f"   Total possible pairs: {total_pairs}")
        print(f"   Pairs with >= {min_overlap_threshold}% overlap: {pairs_passing_filter}")
        print(f"   Matching {pairs_to_match} image pairs (filtered by footprint overlap >= {min_overlap_threshold}%)...")
        print(f"   Estimated time: ~{pairs_to_match * 0.5 / 60:.1f} minutes (assuming ~0.5 sec/pair)")
        match_start = time.time()
        
        # Generate pairs with priorities, filtered by footprint overlap
        image_list = list(quarter_res_images)
        pairs_with_priority = []
        
        for i in range(len(image_list)):
            for j in range(i + 1, len(image_list)):
                img0_path = image_list[i]
                img1_path = image_list[j]
                
                # Get image names (without quarter_ prefix for lookup)
                img0_name = Path(img0_path).name
                img1_name = Path(img1_path).name
                # Remove quarter_ prefix if present
                img0_name_clean = img0_name.replace('quarter_', '')
                img1_name_clean = img1_name.replace('quarter_', '')
                
                # Check if this pair has sufficient footprint overlap
                if footprint_overlaps:
                    overlap_key = (img0_name_clean, img1_name_clean)
                    if overlap_key not in footprint_overlaps:
                        # Skip this pair - insufficient overlap
                        continue
                    overlap_pct = footprint_overlaps[overlap_key]
                else:
                    # No overlaps loaded, match all pairs
                    overlap_pct = 100.0
                
                # Add all pairs with sufficient overlap
                pairs_with_priority.append((i, j, img0_path, img1_path))
        
        # Sort pairs (for consistency, though we're matching all pairs)
        pairs_with_priority.sort()
        
        # Select top priority pairs up to max_pairs (None = all pairs)
        if max_pairs:
            pairs_with_priority = pairs_with_priority[:max_pairs]
        
        if footprint_overlaps:
            print(f"   Selected {len(pairs_with_priority)} pairs (filtered by >= {min_overlap_threshold}% overlap)")
        else:
            print(f"   WARNING: No footprint overlaps loaded - selected {len(pairs_with_priority)} pairs (ALL pairs, no filtering)")
        
        pairs = [(i, j) for i, j, _, _ in pairs_with_priority]
        
        # Parallel matching function
        def match_pair(args):
            """Worker function for parallel matching."""
            i, j, img0_path, img1_path, feats0, feats1 = args
            try:
                match_result = matcher.match_features(feats0, feats1)
                return {
                    'image0': img0_path,
                    'image1': img1_path,
                    'matches': match_result['matches'].tolist(),
                    'match_confidence': match_result['match_confidence'].tolist() if len(match_result['match_confidence']) > 0 else [],
                    'num_matches': match_result['num_matches']
                }
            except Exception as e:
                print(f"   Error matching {Path(img0_path).name} <-> {Path(img1_path).name}: {e}")
                return {
                    'image0': img0_path,
                    'image1': img1_path,
                    'matches': [],
                    'match_confidence': [],
                    'num_matches': 0
                }
        
        # Prepare arguments for parallel processing
        import os
        import torch
        # MPS (Metal) on macOS has issues with threading - use fewer workers or sequential
        if torch.backends.mps.is_available():
            num_workers = 1  # MPS doesn't work well with threading
            print(f"   Using sequential matching (MPS device detected - threading disabled)")
        else:
            num_workers = min(os.cpu_count() or 4, len(pairs), 4)  # Limit to 4 workers max
            print(f"   Using {num_workers} parallel workers for matching...")
        
        match_args = [
            (i, j, image_list[i], image_list[j], 
             match_results['features'][image_list[i]], 
             match_results['features'][image_list[j]])
            for i, j in pairs
        ]
        
        # Use ThreadPoolExecutor for parallel matching (PyTorch inference is thread-safe)
        matches = []
        print(f"   Starting matching of {len(match_args)} pairs...")
        match_start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Use tqdm with file=sys.stdout to ensure it goes to log
            results = list(tqdm(
                executor.map(match_pair, match_args),
                total=len(match_args),
                desc="Matching pairs",
                file=sys.stdout,
                ncols=100,
                mininterval=1.0,  # Update at least once per second
                maxinterval=5.0   # But not more than every 5 seconds
            ))
            matches.extend(results)
        
        elapsed = time.time() - match_start_time
        print(f"   Matched {len(matches)} pairs in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        if len(matches) > 0:
            avg_time = elapsed / len(matches)
            print(f"   Average time per pair: {avg_time:.2f} seconds")
        
        match_results['matches'] = matches
        match_time = time.time() - match_start
        print(f"   Matching completed in {match_time:.1f} seconds")
        
        # Save matches cache (with match confidences)
        matches_cache_data = []
        for match_dict in matches:
            matches_cache_data.append({
                'image0': match_dict['image0'],
                'image1': match_dict['image1'],
                'num_matches': match_dict['num_matches'],
                'matches': match_dict['matches'] if isinstance(match_dict['matches'], list) else match_dict['matches'].tolist(),
                'match_confidence': match_dict.get('match_confidence', []) if isinstance(match_dict.get('match_confidence', []), list) else match_dict.get('match_confidence', []).tolist()
            })
        with open(matches_cache_path, 'w') as f:
            json.dump(matches_cache_data, f, indent=2)
        print(f"   Saved matches cache (with confidences) to: {matches_cache_path}")
    
    print(f"\n   Results:")
    print(f"     Images processed: {len(match_results['features'])}")
    print(f"     Image pairs matched: {len(match_results['matches'])}")
    total_matches = sum(m['num_matches'] for m in match_results['matches'])
    print(f"     Total matches: {total_matches}")
    if match_results['matches']:
        avg_matches = total_matches / len(match_results['matches'])
        print(f"     Average matches per pair: {avg_matches:.1f}")
    
    # Save feature extraction results (FULL data including descriptors for matching)
    print("\n   Saving feature extraction results (full data with descriptors)...")
    features_output = output_dir / f"features_cell_{central_cell}.json"
    features_data = {}
    for img_path, feat_dict in match_results['features'].items():
        img_name = Path(img_path).name
        features_data[img_name] = {
            'image_path': img_path,
            'num_keypoints': len(feat_dict['keypoints']),
            'keypoints': feat_dict['keypoints'].tolist(),
            'scores': feat_dict['scores'].tolist(),
            'descriptors': feat_dict.get('descriptors', []).tolist() if 'descriptors' in feat_dict and feat_dict['descriptors'] is not None else []
        }
    with open(features_output, 'w') as f:
        json.dump(features_data, f, indent=2)
    print(f"     Saved full features (with descriptors) to: {features_output}")
    
    # Save match results (summary, not full cache) - UNFILTERED
    print("   Saving match results (unfiltered - original matches)...")
    matches_output_unfiltered = output_dir / f"matches_unfiltered_cell_{central_cell}.json"
    matches_data_unfiltered = []
    for match_dict in match_results['matches']:
        matches_data_unfiltered.append({
            'image0': Path(match_dict['image0']).name,
            'image1': Path(match_dict['image1']).name,
            'num_matches': match_dict['num_matches'],
            'matches': match_dict['matches'] if isinstance(match_dict['matches'], list) else match_dict['matches'].tolist(),
            'match_confidence': match_dict.get('match_confidence', []) if isinstance(match_dict.get('match_confidence', []), list) else match_dict.get('match_confidence', []).tolist()
        })
    with open(matches_output_unfiltered, 'w') as f:
        json.dump(matches_data_unfiltered, f, indent=2)
    print(f"     Saved unfiltered matches to: {matches_output_unfiltered}")
    
    # Step 5.6: Epipolar filtering (optional)
    matches_output_filtered = None
    if enable_epipolar_filtering:
        print("\n5.6. Filtering matches using robust epipolar geometry...")
        from matcher.epipolar_validation import filter_matches_epipolar_robust
        
        matches_output_filtered = output_dir / f"matches_filtered_cell_{central_cell}.json"
        
        if not matches_output_filtered.exists():
            epipolar_stats = filter_matches_epipolar_robust(
                matches_file=str(matches_output_unfiltered),
                features_file=str(features_output),
                output_file=str(matches_output_filtered),
                image_width=quarter_res_width,
                image_height=quarter_res_height,
                ransac_threshold=0.5,
                confidence=0.999,
                max_iters=2000
            )
            print(f"   Saved filtered matches to: {matches_output_filtered}")
            print(f"   Epipolar filtering stats: {epipolar_stats}")
        else:
            print(f"   Filtered matches file already exists: {matches_output_filtered}")
    else:
        print("\n5.6. Epipolar filtering disabled (using unfiltered matches)")
    
    # Save statistics
    stats_output = output_dir / f"statistics_cell_{central_cell}.json"
    stats = {
        'cell_id': central_cell,
        'num_images': len(match_results['features']),
        'num_pairs': len(match_results['matches']),
        'total_matches': int(total_matches),
        'avg_matches_per_pair': float(avg_matches) if match_results['matches'] else 0.0,
        'min_matches': int(min(m['num_matches'] for m in match_results['matches'])) if match_results['matches'] else 0,
        'max_matches': int(max(m['num_matches'] for m in match_results['matches'])) if match_results['matches'] else 0,
        'features_per_image': {
            Path(img_path).name: len(feat_dict['keypoints'])
            for img_path, feat_dict in match_results['features'].items()
        }
    }
    with open(stats_output, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"     Saved statistics to: {stats_output}")
    
    # Step 6: Export to MetaShape format
    print("\n6. Exporting to MetaShape format...")
    metashape_output = output_dir / f"tiepoints_cell_{central_cell}_metashape.json"
    export_to_metashape(
        match_results,
        str(metashape_output),
        image_base_path=str(image_dir)
    )
    print(f"   Exported unfiltered tiepoints to: {metashape_output}")
    
    # Also export filtered matches and tracks to MetaShape if available
    if enable_epipolar_filtering and matches_output_filtered and matches_output_filtered.exists():
        print("   Exporting filtered matches and tracks to MetaShape format...")
        # Create filtered match_results structure
        with open(matches_output_filtered, 'r') as f:
            filtered_matches_data = json.load(f)
        
        # Convert filtered matches to match_results format
        filtered_match_results = {
            'features': match_results['features'],  # Same features
            'matches': []
        }
        
        # Convert filtered matches to the format expected by export_to_metashape
        for match_dict in filtered_matches_data:
            img0_name = match_dict['image0']
            img1_name = match_dict['image1']
            
            # Find full paths in features
            img0_path = None
            img1_path = None
            for feat_path in match_results['features'].keys():
                feat_name = Path(feat_path).name
                if feat_name == img0_name or feat_path.endswith(img0_name):
                    img0_path = feat_path
                if feat_name == img1_name or feat_path.endswith(img1_name):
                    img1_path = feat_path
            
            if img0_path and img1_path:
                filtered_match_results['matches'].append({
                    'image0': img0_path,
                    'image1': img1_path,
                    'matches': np.array(match_dict['matches']),
                    'num_matches': match_dict['num_matches'],
                    'match_confidence': match_dict.get('match_confidence', [])
                })
        
        metashape_output_filtered = output_dir / f"tiepoints_filtered_cell_{central_cell}_metashape.json"
        export_to_metashape(
            filtered_match_results,
            str(metashape_output_filtered),
            image_base_path=str(image_dir)
        )
        print(f"   Exported filtered tiepoints to: {metashape_output_filtered}")
    
    # Step 7: Create visualizations
    print("\n7. Creating visualizations...")
    
    # Visualization 1: Feature and match visualization (top pairs)
    viz_output = output_dir / f"matches_visualization_cell_{central_cell}.png"
    # Use quarter_res_images instead of cell_images since matches use quarter-resolution paths
    visualize_features_and_matches(
        match_results,
        quarter_res_images[:20] if len(quarter_res_images) >= 20 else quarter_res_images,  # Visualize first 20 images
        str(viz_output),
        max_pairs=10
    )
    print(f"   Saved match visualization to: {viz_output}")
    
    # Visualization 2: Feature distribution
    dist_output = output_dir / f"feature_distribution_cell_{central_cell}.png"
    visualize_feature_distribution(
        match_results,
        str(dist_output)
    )
    print(f"   Saved feature distribution to: {dist_output}")
    
    # Visualization 2.5: Overlap counts per image
    print("\n   Creating overlap counts visualization...")
    from matcher.visualization import visualize_overlap_counts
    overlap_counts_output = output_dir / f"overlap_counts_per_image.png"
    visualize_overlap_counts(
        overlaps_file=str(output_dir / "footprint_overlaps.json"),
        min_overlap_threshold=min_overlap_threshold,
        output_file=str(overlap_counts_output)
    )
    print(f"   Saved overlap counts to: {overlap_counts_output}")
    
    # Step 8: Compute tracks and create histogram
    print("\n8. Computing feature tracks...")
    # Get max overlap count to limit track length
    max_overlap_count = 0
    if Path(output_dir / "footprint_overlaps.json").exists():
        with open(output_dir / "footprint_overlaps.json", 'r') as f:
            overlaps_data = json.load(f)
        overlap_counts = {}
        for overlap in overlaps_data.get('overlaps', []):
            if overlap.get('overlap_percentage', 0) >= min_overlap_threshold:
                img1 = overlap['image1']
                img2 = overlap['image2']
                overlap_counts[img1] = overlap_counts.get(img1, 0) + 1
                overlap_counts[img2] = overlap_counts.get(img2, 0) + 1
        max_overlap_count = max(overlap_counts.values()) if overlap_counts else 0
        print(f"   Maximum overlaps per image: {max_overlap_count}")
        print(f"   Limiting track length to {max_overlap_count} (max overlap count)")
    
    # Determine which matches to use for tracks
    matches_for_tracks = match_results['matches']
    tracks_output_filtered = None
    use_filtered = False
    
    # If filtered matches exist, use them; otherwise use unfiltered
    if enable_epipolar_filtering and matches_output_filtered and matches_output_filtered.exists():
        print("   Using filtered matches for track computation...")
        with open(matches_output_filtered, 'r') as f:
            filtered_matches_data = json.load(f)
        
        # Convert filtered matches to the format expected by compute_tracks
        # Need to map image names back to full paths
        filtered_matches_for_tracks = []
        for match_dict in filtered_matches_data:
            # Find full paths for images
            img0_name = match_dict['image0']
            img1_name = match_dict['image1']
            
            # Try to find full paths in features
            img0_path = None
            img1_path = None
            for feat_path in match_results['features'].keys():
                feat_name = Path(feat_path).name
                if feat_name == img0_name or feat_path.endswith(img0_name):
                    img0_path = feat_path
                if feat_name == img1_name or feat_path.endswith(img1_name):
                    img1_path = feat_path
            
            if img0_path and img1_path:
                filtered_matches_for_tracks.append({
                    'image0': img0_path,
                    'image1': img1_path,
                    'matches': np.array(match_dict['matches']),
                    'num_matches': match_dict['num_matches'],
                    'match_confidence': match_dict.get('match_confidence', [])
                })
        
        if len(filtered_matches_for_tracks) > 0:
            matches_for_tracks = filtered_matches_for_tracks
            tracks_output_filtered = output_dir / f"tracks_filtered_cell_{central_cell}.json"
            use_filtered = True
            print(f"   Using {len(matches_for_tracks)} filtered match pairs")
        else:
            print("   Warning: Could not map filtered matches to full paths, using unfiltered matches")
    else:
        print("   Using unfiltered matches for track computation...")
        use_filtered = False
    
    tracks_data = compute_tracks(
        matches_for_tracks, 
        match_results['features'],
        max_track_length=max_overlap_count if max_overlap_count > 0 else None
    )
    print(f"   Found {tracks_data['num_tracks']} tracks")
    print(f"   Total features in tracks: {tracks_data['total_features_in_tracks']}")
    
    # Save tracks (with full feature information)
    tracks_output = output_dir / f"tracks_cell_{central_cell}.json"
    if use_filtered and tracks_output_filtered:
        # Save filtered tracks to separate file
        tracks_output = tracks_output_filtered
        print(f"   Saving filtered tracks to: {tracks_output}")
        
        # Also compute and save unfiltered tracks for comparison
        print("   Also computing unfiltered tracks for comparison...")
        tracks_data_unfiltered = compute_tracks(
            match_results['matches'], 
            match_results['features'],
            max_track_length=max_overlap_count if max_overlap_count > 0 else None
        )
        tracks_output_unfiltered = output_dir / f"tracks_cell_{central_cell}.json"
        tracks_json_unfiltered = {
            'num_tracks': tracks_data_unfiltered['num_tracks'],
            'total_features_in_tracks': tracks_data_unfiltered['total_features_in_tracks'],
            'tracks': [
                {
                    'track_id': t['track_id'],
                    'length': t['length'],
                    'num_images': len(t['images']),
                    'images': list(t['images']),
                    'features': [(img, idx) for img, idx in t['features']]
                }
                for t in tracks_data_unfiltered['tracks']
            ]
        }
        with open(tracks_output_unfiltered, 'w') as f:
            json.dump(tracks_json_unfiltered, f, indent=2)
        print(f"   Saved unfiltered tracks to: {tracks_output_unfiltered}")
    else:
        print(f"   Saving unfiltered tracks to: {tracks_output}")
    
    tracks_json = {
        'num_tracks': tracks_data['num_tracks'],
        'total_features_in_tracks': tracks_data['total_features_in_tracks'],
        'tracks': [
            {
                'track_id': t['track_id'],
                'length': t['length'],
                'num_images': len(t['images']),
                'images': list(t['images']),
                'features': [(img, idx) for img, idx in t['features']]  # Save feature indices too
            }
            for t in tracks_data['tracks']
        ]
    }
    with open(tracks_output, 'w') as f:
        json.dump(tracks_json, f, indent=2)
    print(f"   Saved tracks to: {tracks_output}")
    
    # Create track length histogram
    histogram_output = output_dir / f"track_length_histogram_cell_{central_cell}.png"
    # Use unfiltered tracks for the main histogram if filtering was used
    if use_filtered and 'tracks_data_unfiltered' in locals():
        plot_track_length_histogram(tracks_data_unfiltered, str(histogram_output))
    else:
        plot_track_length_histogram(tracks_data, str(histogram_output))
    print(f"   Saved track length histogram to: {histogram_output}")
    
    # Create filtered track length histogram if filtering was used
    if use_filtered and tracks_output_filtered and tracks_output_filtered.exists():
        histogram_output_filtered = output_dir / f"track_length_histogram_filtered_cell_{central_cell}.png"
        plot_track_length_histogram(tracks_data, str(histogram_output_filtered))  # tracks_data is filtered when use_filtered is True
        print(f"   Saved filtered track length histogram to: {histogram_output_filtered}")
    
    # Determine which tracks file to use for main visualizations
    # When filtering is enabled, use unfiltered tracks for main visualizations
    # and filtered tracks for filtered visualizations
    tracks_file_for_main_viz = tracks_output
    if use_filtered and tracks_output_filtered:
        # Use unfiltered tracks for main visualizations
        tracks_file_for_main_viz = output_dir / f"tracks_cell_{central_cell}.json"
    
    # Create track visualizations for 5 random tracks (unfiltered)
    print("\n   Creating track visualizations (unfiltered)...")
    from matcher.visualization import create_multiple_track_visualizations, visualize_tracks_on_image_triplets
    create_multiple_track_visualizations(
        tracks_file=str(tracks_file_for_main_viz),
        features_file=str(features_cache_path),
        matches_file=str(matches_output_unfiltered),
        image_dir=str(quarter_res_dir),  # Use quarter-resolution image directory
        output_dir=str(output_dir / "visualization"),  # Save to outputs/visualization/
        num_tracks=5
    )
    
    # Create track visualizations on image triplets (unfiltered)
    print("\n   Creating track visualizations on image triplets (unfiltered)...")
    # Load features data
    with open(features_cache_path, 'r') as f:
        features_data = json.load(f)
    
    # Define triplets to visualize (as integers for the function)
    triplets = [
        (1, 2, 3),      # 0001, 0002, 0003
        (51, 52, 53),   # 0051, 0052, 0053
        (101, 102, 103), # 0101, 0102, 0103
        (151, 152, 153)  # 0151, 0152, 0153
    ]
    
    triplet_output_dir = output_dir / "visualize_tracks_on_images"
    triplet_output_dir.mkdir(parents=True, exist_ok=True)
    
    for img1_num, img2_num, img3_num in triplets:
        # Create output filename
        output_filename = f"tracks_{img1_num:04d}_{img2_num:04d}_{img3_num:04d}.png"
        output_path = triplet_output_dir / output_filename
        
        # Create visualization for all tracks
        visualize_tracks_on_image_triplets(
            image_dir=str(quarter_res_dir),
            features_data=features_data,
            tracks_file=str(tracks_file_for_main_viz),
            img_nums=(img1_num, img2_num, img3_num),
            output_path=str(output_path),
            only_three_image_tracks=False
        )
        
        # Also create visualization for three-image tracks only
        output_filename_three = f"tracks_{img1_num:04d}_{img2_num:04d}_{img3_num:04d}_three_only.png"
        output_path_three = triplet_output_dir / output_filename_three
        
        visualize_tracks_on_image_triplets(
            image_dir=str(quarter_res_dir),
            features_data=features_data,
            tracks_file=str(tracks_file_for_main_viz),
            img_nums=(img1_num, img2_num, img3_num),
            output_path=str(output_path_three),
            only_three_image_tracks=True
        )
    
    # Create filtered track visualizations if epipolar filtering was used
    if use_filtered and tracks_output_filtered and tracks_output_filtered.exists():
        print("\n   Creating filtered track visualizations...")
        
        # Create filtered track visualizations for 5 random tracks
        create_multiple_track_visualizations(
            tracks_file=str(tracks_output_filtered),
            features_file=str(features_cache_path),
            matches_file=str(matches_output_filtered) if matches_output_filtered and matches_output_filtered.exists() else str(matches_output_unfiltered),
            image_dir=str(quarter_res_dir),
            output_dir=str(output_dir / "visualization" / "filtered"),  # Save to outputs/visualization/filtered/
            num_tracks=5
        )
        
        # Create filtered track visualizations on image triplets
        print("\n   Creating filtered track visualizations on image triplets...")
        filtered_triplet_output_dir = output_dir / "visualize_tracks_on_images" / "filtered"
        filtered_triplet_output_dir.mkdir(parents=True, exist_ok=True)
        
        for img1_num, img2_num, img3_num in triplets:
            # Create output filename with _filtered suffix
            output_filename = f"tracks_{img1_num:04d}_{img2_num:04d}_{img3_num:04d}_filtered.png"
            output_path = filtered_triplet_output_dir / output_filename
            
            # Create visualization for all filtered tracks
            visualize_tracks_on_image_triplets(
                image_dir=str(quarter_res_dir),
                features_data=features_data,
                tracks_file=str(tracks_output_filtered),
                img_nums=(img1_num, img2_num, img3_num),
                output_path=str(output_path),
                only_three_image_tracks=False
            )
            
            # Also create visualization for three-image filtered tracks only
            output_filename_three = f"tracks_{img1_num:04d}_{img2_num:04d}_{img3_num:04d}_filtered_three_only.png"
            output_path_three = filtered_triplet_output_dir / output_filename_three
            
            visualize_tracks_on_image_triplets(
                image_dir=str(quarter_res_dir),
                features_data=features_data,
                tracks_file=str(tracks_output_filtered),
                img_nums=(img1_num, img2_num, img3_num),
                output_path=str(output_path_three),
                only_three_image_tracks=True
            )
    
    # Step 8.5: Reconstruct point cloud from tracks
    print("\n8.5. Reconstructing point cloud from tracks...")
    poses_file = camera_poses_file
    
    if poses_file.exists():
        try:
            with open(poses_file, 'r') as f:
                camera_poses_list = json.load(f)
            
            # Convert to dictionary keyed by image name
            # Camera poses use original image names (e.g., "8928d89ac57ffff_172550_0001.jpg")
            # but tracks/features use quarter-resolution names (e.g., "quarter_8928d89ac57ffff_172550_0001.jpg")
            camera_poses_dict = {}
            feature_keys = list(match_results['features'].keys())
            
            for p in camera_poses_list:
                img_name = p['image_name']
                # Build quarter-resolution name
                quarter_name = f"quarter_{img_name}"
                
                # Check if this quarter name exists in features
                if quarter_name in feature_keys:
                    camera_poses_dict[quarter_name] = p
                else:
                    # Fallback: try to find by matching the base name
                    img_base = Path(img_name).name
                    found = False
                    for feat_key in feature_keys:
                        if isinstance(feat_key, str) and Path(feat_key).name == img_base:
                            camera_poses_dict[feat_key] = p
                            found = True
                            break
                    if not found:
                        # Last resort: use original name
                        camera_poses_dict[img_name] = p
                        print(f"   Warning: Could not find matching feature key for {img_name}")
            
            # Determine origin from first pose
            first_pose = next(iter(camera_poses_dict.values()))
            if first_pose.get('gps'):
                origin_lat = first_pose['gps']['latitude']
                origin_lon = first_pose['gps']['longitude']
                
                print(f"   Using origin: ({origin_lat:.6f}, {origin_lon:.6f})")
                print(f"   Image dimensions: {quarter_res_width}x{quarter_res_height}")
                
                # Use filtered tracks if available, otherwise use unfiltered
                if use_filtered and tracks_output_filtered and tracks_output_filtered.exists():
                    print("   Loading filtered tracks for point cloud reconstruction...")
                    with open(tracks_output_filtered, 'r') as f:
                        filtered_tracks_data = json.load(f)
                    tracks_for_reconstruction = filtered_tracks_data['tracks']
                    
                    # Use filtered matches
                    if matches_output_filtered and matches_output_filtered.exists():
                        with open(matches_output_filtered, 'r') as f:
                            filtered_matches_data = json.load(f)
                        # Convert filtered matches to the format expected by reconstruction
                        matches_for_reconstruction = []
                        for match_dict in filtered_matches_data:
                            img0_path = match_dict['image0']
                            img1_path = match_dict['image1']
                            
                            # Find full paths in features
                            img0_full = None
                            img1_full = None
                            for feat_path in match_results['features'].keys():
                                if Path(feat_path).name == Path(img0_path).name or feat_path.endswith(Path(img0_path).name):
                                    img0_full = feat_path
                                if Path(feat_path).name == Path(img1_path).name or feat_path.endswith(Path(img1_path).name):
                                    img1_full = feat_path
                            
                            if img0_full and img1_full:
                                matches_for_reconstruction.append({
                                    'image0': img0_full,
                                    'image1': img1_full,
                                    'matches': np.array(match_dict['matches']),
                                    'num_matches': match_dict['num_matches'],
                                    'match_confidence': match_dict.get('match_confidence', [])
                                })
                        print(f"   Using {len(tracks_for_reconstruction)} filtered tracks and {len(matches_for_reconstruction)} filtered match pairs")
                    else:
                        matches_for_reconstruction = matches_for_tracks
                        print(f"   Using {len(tracks_for_reconstruction)} filtered tracks with unfiltered matches")
                else:
                    tracks_for_reconstruction = tracks_data['tracks']
                    matches_for_reconstruction = matches_for_tracks
                    print("   Using unfiltered tracks for point cloud reconstruction")
                
                # Use direct triangulation from matches (simpler and more reliable)
                from matcher.triangulate_from_matches import triangulate_from_matches
                from matcher.ply_export import export_point_cloud_to_ply
                from matcher.dji_exif_parser import get_camera_rotation_from_dji_orientation
                
                # Build camera poses as (R, t) tuples
                # Map camera poses to match the image names used in matches/features
                camera_poses_rt = {}
                feature_keys = set(match_results['features'].keys())
                
                for img_name, pose_dict in camera_poses_dict.items():
                    if not pose_dict.get('gps'):
                        continue
                    
                    # Get position
                    lat = pose_dict['gps']['latitude']
                    lon = pose_dict['gps']['longitude']
                    alt = pose_dict['gps']['altitude']
                    
                    lat_diff = lat - origin_lat
                    lon_diff = lon - origin_lon
                    x = lon_diff * 111000.0 * np.cos(np.radians(origin_lat))
                    y = lat_diff * 111000.0
                    z = alt
                    t = np.array([x, y, z])
                    
                    # Get rotation
                    if pose_dict.get('dji_orientation'):
                        R = get_camera_rotation_from_dji_orientation(pose_dict['dji_orientation'])
                    else:
                        # Default to identity (will be refined by bundle adjustment)
                        R = np.eye(3, dtype=np.float64)
                    
                    # Map to feature key name (try to find matching feature key)
                    pose_key = None
                    img_base = Path(img_name).name
                    for feat_key in feature_keys:
                        feat_base = Path(feat_key).name
                        if feat_base == img_base or feat_base.replace('quarter_', '') == img_base:
                            pose_key = feat_key
                            break
                    
                    # If no match found, use original name
                    if pose_key is None:
                        pose_key = img_name
                    
                    camera_poses_rt[pose_key] = (R, t)
                
                # Estimate camera intrinsics
                from matcher.robust_reconstruction import estimate_camera_intrinsics
                K = estimate_camera_intrinsics(
                    quarter_res_width,
                    quarter_res_height,
                    focal_length_mm=8.8,
                    sensor_width_mm=13.2
                )
                
                # Triangulate directly from matches
                print("   Triangulating 3D points directly from matches...")
                points_3d_list, triang_stats = triangulate_from_matches(
                    matches_for_reconstruction,
                    match_results['features'],
                    camera_poses_rt,
                    K,
                    max_reprojection_error=2.0
                )
                
                print(f"   Triangulated {len(points_3d_list)} 3D points from {len(matches_for_reconstruction)} match pairs")
                print(f"   Mean reprojection error: {triang_stats.get('mean_reprojection_error', 0):.3f} pixels")
                
                # Convert to point cloud format
                point_cloud = points_3d_list
                points_3d = np.array([p['point_3d'] for p in points_3d_list])
                
                pc_stats = {
                    'triangulation': triang_stats,
                    'num_points': len(points_3d_list)
                }
                
                camera_poses_refined = camera_poses_rt  # No refinement for now
                
                print(f"   Reconstructed {len(points_3d)} 3D points from {len(matches_for_reconstruction)} match pairs")
                print(f"   Mean reprojection error: {pc_stats.get('triangulation', {}).get('mean_reprojection_error', 0):.3f} pixels")
                
                # Save point cloud to JSON
                point_cloud_json = output_dir / f"point_cloud_cell_{central_cell}.json"
                point_cloud_data = {
                    'num_points': len(point_cloud),
                    'points': point_cloud,
                    'stats': pc_stats
                }
                with open(point_cloud_json, 'w') as f:
                    json.dump(point_cloud_data, f, indent=2)
                print(f"   Saved point cloud JSON to: {point_cloud_json}")
                
                # Export to PLY
                point_cloud_ply = output_dir / f"point_cloud_cell_{central_cell}.ply"
                export_point_cloud_to_ply(
                    str(point_cloud_json),
                    str(point_cloud_ply),
                    color_by_error=True
                )
                print(f"   Saved point cloud PLY to: {point_cloud_ply}")
            else:
                print("   Warning: No GPS data in camera poses, skipping point cloud reconstruction")
        except Exception as e:
            print(f"   Warning: Error during point cloud reconstruction: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"   Warning: Camera poses file not found: {poses_file}")
        print("   Run extract_camera_poses.py first to enable point cloud reconstruction")
    
    # Step 8.6: Export debug files (footprints and camera poses to CSV)
    print("\n8.6. Exporting debug files...")
    from matcher.debug_export import export_footprints_to_csv, export_camera_poses_to_csv
    
    footprints_csv = output_dir / f"footprints_cell_{central_cell}.csv"
    export_footprints_to_csv(
        camera_poses_file=str(camera_poses_file),
        output_csv=str(footprints_csv),
        focal_length_mm=focal_length_mm,
        sensor_width_mm=sensor_width_mm,
        sensor_height_mm=sensor_height_mm
    )
    print(f"   Exported footprints to: {footprints_csv}")
    
    camera_poses_csv = output_dir / f"camera_poses_cell_{central_cell}.csv"
    export_camera_poses_to_csv(
        camera_poses_file=str(camera_poses_file),
        output_csv=str(camera_poses_csv)
    )
    print(f"   Exported camera poses to: {camera_poses_csv}")
    
    # Step 9: Performance summary
    print("\n9. Performance Summary:")
    total_time = time.time() - script_start_time
    print(f"   Total processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"   Feature extraction time: {feature_time:.1f} seconds")
    if 'match_time' in locals():
        print(f"   Matching time: {match_time:.1f} seconds")
    print(f"   Images processed: {len(match_results['features'])}")
    print(f"   Image pairs matched: {len(match_results['matches'])}")
    print(f"   Total matches: {total_matches}")
    print(f"   Tracks computed: {tracks_data['num_tracks']}")
    
    # Save performance summary
    perf_output = output_dir / f"performance_summary_cell_{central_cell}.json"
    perf_data = {
        'total_time_seconds': total_time,
        'feature_extraction_time_seconds': feature_time,
        'matching_time_seconds': match_time if 'match_time' in locals() else 0,
        'num_images': len(match_results['features']),
        'num_pairs': len(match_results['matches']),
        'total_matches': int(total_matches),
        'num_tracks': tracks_data['num_tracks'],
        'device': str(matcher.device),
        'max_features_per_image': max_features,
        'used_quarter_resolution': (resolution < 1.0),
        'resolution': resolution
    }
    with open(perf_output, 'w') as f:
        json.dump(perf_data, f, indent=2)
    print(f"   Saved performance summary to: {perf_output}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
    print(f"\nAll output files saved to: {output_dir}")
    print(f"\nIntermediate results:")
    print(f"  - Cell analysis: {cell_analysis_path}")
    print(f"  - Central cell info: {central_cell_path}")
    print(f"  - Features: {features_output}")
    print(f"  - Matches (unfiltered): {matches_output_unfiltered}")
    print(f"  - Statistics: {stats_output}")
    print(f"\nFinal outputs:")
    print(f"  - MetaShape tiepoints: {metashape_output}")
    print(f"  - Match visualization: {viz_output}")
    print(f"  - Feature distribution: {dist_output}")
    print(f"  - Track length histogram: {histogram_output}")
    print(f"  - Performance summary: {perf_output}")
    print(f"  - Tracks: {tracks_output}")


if __name__ == "__main__":
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Tiepoint matcher with footprint overlap filtering')
    parser.add_argument('--mode', type=str, default='ortho', choices=['ortho', 'hybrid'],
                       help='Processing mode: "ortho" for nadir-only, "hybrid" for nadir+oblique (default: ortho)')
    parser.add_argument('--resolution', type=float, default=1.0,
                       help='Image resolution scale factor: 1.0 for full resolution, 0.5 for half, 0.25 for quarter, etc. (default: 1.0)')
    parser.add_argument('--min-overlap', type=float, default=50.0,
                       help='Minimum footprint overlap percentage to consider for matching (default: 50.0)')
    parser.add_argument('--no-epipolar-filtering', action='store_true',
                       help='Disable epipolar geometry filtering (default: filtering enabled)')
    parser.add_argument('--cell-ids-file', type=str, default=None,
                       help='Path to JSON file with list of cell IDs to process. If multiple cells, enables multi-cell processing with intra and inter-cell matching.')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Override output directory (default: outputs/ for ortho, outputs_hybrid/ for hybrid)')
    parser.add_argument('--image-dir', type=str, default=None,
                       help='Override image directory (default: based on mode)')
    args = parser.parse_args()
    
    enable_epipolar_filtering = not args.no_epipolar_filtering
    mode = args.mode
    
    # Set up output directory for logs based on mode
    if mode == 'hybrid':
        output_dir = Path("outputs_hybrid")
    else:
        output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    log_file = output_dir / "run_log.txt"
    
    # Redirect both stdout and stderr to both console and log file
    # This ensures tqdm progress bars (which write to stderr) are captured
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
        def isatty(self):
            # Return False so tqdm doesn't try to use terminal features
            return False
    
    with open(log_file, 'w') as f:
        f.write(f"Run started at {datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n")
        f.flush()
        
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        # Create Tee instances for both stdout and stderr
        stdout_tee = Tee(original_stdout, f)
        stderr_tee = Tee(original_stderr, f)
        
        sys.stdout = stdout_tee
        sys.stderr = stderr_tee
        
        try:
            main(min_overlap_threshold=args.min_overlap, enable_epipolar_filtering=enable_epipolar_filtering, mode=mode, resolution=args.resolution, cell_ids_file=args.cell_ids_file, output_dir_override=args.output_dir, image_dir_override=args.image_dir)
        except Exception as e:
            import traceback
            error_msg = f"\nERROR: {str(e)}\n{traceback.format_exc()}\n"
            print(error_msg, file=sys.stderr)
            f.write(error_msg)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            f.write(f"\nRun completed at {datetime.now().isoformat()}\n")
            f.flush()
