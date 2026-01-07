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
from tiepoint_matcher import (
    find_central_cell,
    get_cell_images,
    LightGlueMatcher,
    export_to_metashape,
    visualize_features_and_matches,
    visualize_feature_distribution
)
from tiepoint_matcher.image_utils import (
    downsample_images_batch,
    save_feature_coordinates,
    load_feature_coordinates,
    save_features_full,
    load_features_full
)
from tiepoint_matcher.track_analysis import (
    compute_tracks,
    plot_track_length_histogram
)
from tiepoint_matcher.point_cloud_reconstruction import reconstruct_point_cloud
from tiepoint_matcher.robust_reconstruction import robust_reconstruct_with_bundle_adjustment
from tiepoint_matcher.ply_export import export_point_cloud_to_ply
from scanline_matching import ScanlineMatcher, load_scanlines_from_analysis, parse_image_number
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


def main(min_overlap_threshold: float = 50.0):
    # Paths
    image_dir = Path("/Users/mauriciohessflores/Documents/Code/MyCode/research-qualicum_beach_gcp_analysis/input/images")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Tiepoint Matcher Test - Qualicum Beach Data")
    print("=" * 60)
    
    # Step 1: Get all H3 cells from images
    print("\n1. Parsing image filenames to extract H3 cells...")
    cell_to_images = get_cell_images(str(image_dir))
    cell_ids = list(cell_to_images.keys())
    print(f"   Found {len(cell_ids)} H3 cells:")
    for cell_id in sorted(cell_ids):
        num_images = len(cell_to_images[cell_id])
        print(f"     {cell_id}: {num_images} images")
    
    # Save cell analysis
    cell_analysis = {
        'total_cells': len(cell_ids),
        'cells': {cell_id: {'num_images': len(images)} 
                 for cell_id, images in cell_to_images.items()}
    }
    cell_analysis_path = output_dir / "cell_analysis.json"
    with open(cell_analysis_path, 'w') as f:
        json.dump(cell_analysis, f, indent=2)
    print(f"   Saved cell analysis to: {cell_analysis_path}")
    
    # Step 2: Find central cell
    print("\n2. Finding central cell (completely surrounded by 6 neighbors)...")
    central_cell = find_central_cell(cell_ids)
    
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
    
    # Step 3.5: Extract camera poses FIRST (needed for footprint overlap computation)
    print("\n3.5. Extracting camera poses from images...")
    camera_poses_file = output_dir / "camera_poses.json"
    
    if not camera_poses_file.exists():
        print("   Camera poses file not found, extracting from images...")
        from extract_camera_poses import extract_camera_pose
        
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
    
    # Step 3.6: Compute footprint overlaps (needed for matching)
    print("\n3.6. Computing footprint overlaps...")
    from compute_footprint_overlap import compute_all_overlaps
    footprint_overlaps_file = output_dir / "footprint_overlaps.json"
    
    if not footprint_overlaps_file.exists():
        print(f"   Computing footprint overlaps (will filter pairs with < {min_overlap_threshold}% overlap)...")
        compute_all_overlaps(
            camera_poses_file=str(camera_poses_file),
            output_file=str(footprint_overlaps_file),
            focal_length_mm=8.8,
            sensor_width_mm=13.2,
            sensor_height_mm=9.9
        )
        print(f"   Saved footprint overlaps to: {footprint_overlaps_file}")
    else:
        print(f"   Footprint overlaps file already exists: {footprint_overlaps_file}")
    
    # Step 3.7: Use existing quarter-resolution images from inputs/
    print("\n3.7. Using quarter-resolution images from inputs/quarter_resolution_images...")
    quarter_res_dir = Path("inputs/quarter_resolution_images")
    
    if not quarter_res_dir.exists():
        raise FileNotFoundError(f"Quarter-resolution images directory not found: {quarter_res_dir}")
    
    # Map original images to quarter-resolution images
    quarter_res_images = []
    for img_path in cell_images:
        img_name = Path(img_path).name
        quarter_res_path = quarter_res_dir / f"quarter_{img_name}"
        if quarter_res_path.exists():
            quarter_res_images.append(str(quarter_res_path))
        else:
            print(f"   Warning: Quarter-resolution image not found: {quarter_res_path}")
    
    print(f"   Found {len(quarter_res_images)} quarter-resolution images")
    
    # Step 4: Initialize matcher
    print("\n4. Initializing LightGlue matcher...")
    max_features = 100  # Reduced from 500 to save memory
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
                import numpy as np
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
                            'image_path': feat_data.get('image_path', img_path)
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
    
    # Step 5.4: Compute footprint overlaps BEFORE matching
    # This must happen AFTER camera poses are extracted (which happens later in the script)
    # So we'll compute it right before matching, after camera_poses.json is created
    
    # Step 5.5: Match features (with caching)
    print("\n5.5. Matching features across image pairs...")
    
    # Check for cached matches
    matches_cache_path = output_dir / f"matches_cache_cell_{central_cell}.json"
    use_matches_cache = matches_cache_path.exists()
    
    # Use scanline-based matching - will select high-priority pairs
    # For debugging: match ALL pairs with sufficient overlap (no limit)
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
        # Load footprint overlaps to filter pairs (should already be computed in Step 3.6)
        footprint_overlaps_file = output_dir / "footprint_overlaps.json"
        print(f"   Loading footprint overlaps (minimum threshold: {min_overlap_threshold}%)...")
        footprint_overlaps = load_footprint_overlaps(
            overlaps_file=str(footprint_overlaps_file),
            min_overlap_threshold=min_overlap_threshold
        )
        if footprint_overlaps:
            print(f"   Footprint overlap filtering ENABLED - will match only pairs with >= {min_overlap_threshold}% overlap")
        else:
            raise ValueError(f"No footprint overlaps found! Check {footprint_overlaps_file}. Make sure Step 3.6 completed successfully.")
        
        # Load scanlines for intelligent matching (DISABLED FOR DEBUGGING)
        print("   Loading scanline information for improved matching...")
        try:
            scanlines = load_scanlines_from_analysis()
            scanline_matcher = ScanlineMatcher(scanlines)
            print(f"   Loaded {len(scanlines)} scanlines")
            print("   Note: Scanline filtering DISABLED for debugging - matching all pairs that pass overlap filter")
            scanline_matcher = None  # Disable scanline filtering temporarily
        except Exception as e:
            print(f"   Warning: Could not load scanlines: {e}, using default matching")
            scanline_matcher = None
        
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
                
                # Extract image numbers
                img0_num = parse_image_number(img0_name)
                img1_num = parse_image_number(img1_name)
                
                # For debugging: match all pairs (no scanline filtering)
                # Just add all pairs with default priority
                pairs_with_priority.append((i, j, 5, img0_path, img1_path))
        
        # For debugging: match all pairs (no sorting or limiting)
        # Sort by priority (lower = better) - but we're matching all pairs anyway
        pairs_with_priority.sort(key=lambda x: x[2])
        
        # Select top priority pairs up to max_pairs (None = all pairs)
        if max_pairs:
            pairs_with_priority = pairs_with_priority[:max_pairs]
        
        if footprint_overlaps:
            print(f"   Selected {len(pairs_with_priority)} pairs (filtered by >= {min_overlap_threshold}% overlap)")
        else:
            print(f"   WARNING: No footprint overlaps loaded - selected {len(pairs_with_priority)} pairs (ALL pairs, no filtering)")
        
        pairs = [(i, j) for i, j, _, _, _ in pairs_with_priority]
        
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
        num_workers = min(os.cpu_count() or 4, len(pairs), 8)  # Limit to 8 workers max to avoid memory issues
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
    
    # Epipolar validation disabled - using unfiltered matches only
    
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
    print(f"   Exported to: {metashape_output}")
    
    # Step 7: Create visualizations
    print("\n7. Creating visualizations...")
    
    # Visualization 1: Feature and match visualization (top pairs)
    viz_output = output_dir / f"matches_visualization_cell_{central_cell}.png"
    visualize_features_and_matches(
        match_results,
        cell_images[:20],  # Visualize first 20 images
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
    from visualize_overlap_counts import visualize_overlap_counts
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
    
    tracks_data = compute_tracks(
        match_results['matches'], 
        match_results['features'],
        max_track_length=max_overlap_count if max_overlap_count > 0 else None
    )
    print(f"   Found {tracks_data['num_tracks']} tracks")
    print(f"   Total features in tracks: {tracks_data['total_features_in_tracks']}")
    
    # Save tracks (with full feature information)
    tracks_output = output_dir / f"tracks_cell_{central_cell}.json"
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
    plot_track_length_histogram(tracks_data, str(histogram_output))
    
    # Create track visualizations for 5 random tracks
    print("\n   Creating track visualizations...")
    from create_track_visualizations import create_multiple_track_visualizations
    create_multiple_track_visualizations(
        tracks_file=str(tracks_output),
        features_file=str(features_cache_path),
        matches_file=str(matches_output_unfiltered),
        image_dir=str(image_dir),
        output_dir=str(output_dir / "visualization"),  # Save to outputs/visualization/
        num_tracks=5
    )
    
    # Step 8.5: Reconstruct point cloud from tracks (DISABLED FOR NOW)
    print("\n8.5. Point cloud reconstruction disabled for now")
    # poses_file = output_dir / "camera_poses.json"
    # 
    # if poses_file.exists():
    #     try:
    #         with open(poses_file, 'r') as f:
    #                 #             camera_poses_list = json.load(f)
    #         
    #         # Convert to dictionary keyed by image name
    #             #         camera_poses_dict = {p['image_name']: p for p in camera_poses_list}
    #         
    #         # Determine origin from first pose
    #         first_pose = next(iter(camera_poses_dict.values()))
    #         if first_pose.get('gps'):
    #             origin_lat = first_pose['gps']['latitude']
    #             origin_lon = first_pose['gps']['longitude']
    #             
    #             # (rest of point cloud code would go here - commented out for now)
    #             
    #             # Use robust reconstruction with bundle adjustment
    #             # Set filter_outliers=False to see all points and check if they form a ground plane
    #             # camera_poses_refined, points_3d, point_cloud, pc_stats = robust_reconstruct_with_bundle_adjustment(
    #             #     tracks_data['tracks'],
    #             #     match_results['matches'],
    #             #     match_results['features'],
    #             #     camera_poses_dict,
    #             #     origin_lat,
    #             #     origin_lon,
    #             #     image_width=4000,
    #             #     image_height=3000,
    #             #     max_reprojection_error=10.0,  # Increased threshold since we're not filtering
    #             #     use_bundle_adjustment=False,  # Disable for now to see raw triangulation
    #             #     filter_outliers=False  # Don't filter - use all tracks
    #             # )
    #             
    #             # ... (rest of point cloud code commented out)
    #         else:
    #             print("   Warning: No GPS data in camera poses, skipping point cloud reconstruction")
    #     except Exception as e:
    #         print(f"   Warning: Error during point cloud reconstruction: {e}")
    #         import traceback
    #         traceback.print_exc()
    # else:
    #     print(f"   Warning: Camera poses file not found: {poses_file}")
    #     print("   Run extract_camera_poses.py first to enable point cloud reconstruction")
    
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
        'used_quarter_resolution': True
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
    parser.add_argument('--min-overlap', type=float, default=50.0,
                       help='Minimum footprint overlap percentage to consider for matching (default: 50.0)')
    args = parser.parse_args()
    
    # Set up output directory for logs
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
            main(min_overlap_threshold=args.min_overlap)
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
