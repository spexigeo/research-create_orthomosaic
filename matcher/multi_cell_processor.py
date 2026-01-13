"""
Multi-cell processor for computing intra-cell and inter-cell features, matches, and tracks.
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
from tqdm import tqdm

from .tiepoint_graph import TiepointGraph
from .h3_utils import get_cell_images, parse_image_filename
from .track_analysis import compute_tracks


def process_single_cell(
    cell_id: str,
    cell_images: List[str],
    image_dir: Path,
    output_dir: Path,
    min_overlap_threshold: float,
    enable_epipolar_filtering: bool,
    max_features: int,
    resolution: float,
    focal_length_mm: float,
    sensor_width_mm: float,
    sensor_height_mm: float,
    mode: str
) -> Dict:
    """
    Process a single cell (intra-cell matching).
    
    This is essentially a wrapper around the existing matcher functionality
    for a single cell.
    
    Returns:
        Dictionary with keys: 'features', 'matches', 'tracks', 'cell_id'
    """
    from .cameras import extract_poses_from_directory
    from .utils import compute_all_overlaps
    from .lightglue_matcher import LightGlueMatcher
    from .epipolar_validation import filter_matches_epipolar_robust
    
    print(f"\n{'='*60}")
    print(f"Processing INTRA-cell for cell: {cell_id}")
    print(f"{'='*60}")
    print(f"  Images: {len(cell_images)}")
    
    cell_output_dir = output_dir / f"cell_{cell_id}"
    cell_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Extract camera poses
    print(f"\n  Step 1: Extracting camera poses...")
    camera_poses_file = cell_output_dir / "camera_poses.json"
    if not camera_poses_file.exists():
        # Create temporary directory with just this cell's images for pose extraction
        temp_image_dir = cell_output_dir / "temp_images"
        temp_image_dir.mkdir(exist_ok=True)
        for img_path in cell_images:
            import shutil
            shutil.copy2(img_path, temp_image_dir / Path(img_path).name)
        
        extract_poses_from_directory(
            image_dir=str(temp_image_dir),
            output_file=str(camera_poses_file)
        )
        
        # Clean up temp directory
        import shutil
        shutil.rmtree(temp_image_dir, ignore_errors=True)
        print(f"    Saved camera poses")
    else:
        print(f"    Using existing camera poses")
    
    with open(camera_poses_file, 'r') as f:
        camera_poses = json.load(f)
    
    # Step 2: Compute footprint overlaps
    print(f"\n  Step 2: Computing footprint overlaps...")
    footprint_overlaps_file = cell_output_dir / "footprint_overlaps.json"
    if not footprint_overlaps_file.exists():
        compute_all_overlaps(
            image_dir=str(image_dir),
            output_dir=str(cell_output_dir),
            camera_poses_file=str(camera_poses_file),
            output_json_file=str(footprint_overlaps_file),
            output_csv_file=str(footprint_overlaps_file).replace('.json', '.csv'),
            focal_length_mm=focal_length_mm,
            sensor_width_mm=sensor_width_mm,
            sensor_height_mm=sensor_height_mm
        )
        print(f"    Saved footprint overlaps")
    else:
        print(f"    Using existing footprint overlaps")
    
    # Step 3: Extract features
    print(f"\n  Step 3: Extracting features...")
    features_file = cell_output_dir / f"features_cell_{cell_id}.json"
    features_cache_file = cell_output_dir / f"features_cache_cell_{cell_id}.json"
    
    if features_cache_file.exists():
        print(f"    Loading features from cache")
        with open(features_cache_file, 'r') as f:
            features_data_raw = json.load(f)
        # Convert loaded features back to numpy arrays for matching
        features_data = {}
        for img_path, feat_dict in features_data_raw.items():
            features_data[img_path] = {}
            for key, value in feat_dict.items():
                if value is None:
                    features_data[img_path][key] = None
                elif isinstance(value, list):
                    # Convert lists back to numpy arrays (they were originally arrays or tensors)
                    features_data[img_path][key] = np.array(value)
                else:
                    features_data[img_path][key] = value
    else:
        print(f"    Extracting features (this may take a while)...")
        matcher = LightGlueMatcher(device='mps', max_features_per_image=max_features)
        
        # Determine image paths based on resolution
        if resolution < 1.0:
            resolution_str = str(resolution).replace('.', '_')
            processed_dir = image_dir.parent / f"images_{resolution_str}"
            processed_images = [str(processed_dir / Path(img).name) for img in cell_images]
        else:
            processed_images = cell_images
        
        features_data = {}
        for img_path in tqdm(processed_images, desc="  Extracting"):
            if Path(img_path).exists():
                features = matcher.extract_features(img_path)
                if features:
                    # Only save essential data: keypoints, descriptors, scores
                    # Skip tensor versions and image_tensor to save space
                    features_serializable = {
                        'keypoints': features['keypoints'].tolist() if isinstance(features['keypoints'], np.ndarray) else features['keypoints'],
                        'descriptors': features['descriptors'].tolist() if isinstance(features['descriptors'], np.ndarray) else features['descriptors'],
                        'scores': features['scores'].tolist() if isinstance(features['scores'], np.ndarray) else features['scores'],
                        'image_path': features.get('image_path', img_path)
                    }
                    # Verify we're not saving too many features
                    num_features = len(features_serializable['keypoints'])
                    if num_features > max_features:
                        print(f"    Warning: Image {Path(img_path).name} has {num_features} features, expected max {max_features}")
                    features_data[img_path] = features_serializable
        
        # Save features
        with open(features_cache_file, 'w') as f:
            json.dump(features_data, f, indent=2)
        print(f"    Saved features for {len(features_data)} images")
    
    # Step 4: Match images
    print(f"\n  Step 4: Matching images...")
    matches_file = cell_output_dir / f"matches_unfiltered_cell_{cell_id}.json"
    
    if matches_file.exists():
        print(f"    Loading matches from cache")
        with open(matches_file, 'r') as f:
            matches_data = json.load(f)
    else:
        print(f"    Computing matches...")
        # Load footprint overlaps
        with open(footprint_overlaps_file, 'r') as f:
            overlaps_data = json.load(f)
        
        # Handle different formats: list of dicts or dict with 'overlaps' key
        if isinstance(overlaps_data, dict):
            if 'overlaps' in overlaps_data:
                overlaps_list = overlaps_data['overlaps']
            else:
                # Try to find the actual overlaps list
                overlaps_list = [v for v in overlaps_data.values() if isinstance(v, list)]
                if overlaps_list:
                    overlaps_list = overlaps_list[0]
                else:
                    overlaps_list = []
        else:
            overlaps_list = overlaps_data
        
        # Filter pairs by overlap threshold
        pairs_to_match = []
        for overlap in overlaps_list:
            if isinstance(overlap, dict) and overlap.get('overlap_percentage', 0) >= min_overlap_threshold:
                # Handle both 'image1'/'image2' and 'image0'/'image1' formats
                img1 = overlap.get('image1') or overlap.get('image0')
                img2 = overlap.get('image2') or overlap.get('image1')
                if img1 and img2:
                    pairs_to_match.append((img1, img2))
        
        print(f"    Matching {len(pairs_to_match)} image pairs...")
        matcher = LightGlueMatcher(device='mps', max_features_per_image=max_features)
        
        matches_data = []
        for img0, img1 in tqdm(pairs_to_match, desc="  Matching"):
            # Find full paths
            img0_path = None
            img1_path = None
            for feat_path in features_data.keys():
                if Path(feat_path).name == Path(img0).name or img0 in feat_path:
                    img0_path = feat_path
                if Path(feat_path).name == Path(img1).name or img1 in feat_path:
                    img1_path = feat_path
            
            if img0_path and img1_path and img0_path in features_data and img1_path in features_data:
                matches = matcher.match_features(
                    features_data[img0_path],
                    features_data[img1_path]
                )
                if matches and len(matches['matches']) > 0:
                    matches_data.append({
                        'image0': img0_path,
                        'image1': img1_path,
                        'matches': matches['matches'].tolist(),
                        'num_matches': len(matches['matches']),
                        'match_confidence': matches.get('match_confidence', []).tolist() if 'match_confidence' in matches else []
                    })
        
        # Save matches
        with open(matches_file, 'w') as f:
            json.dump(matches_data, f, indent=2)
        print(f"    Saved {len(matches_data)} match pairs")
    
    # Step 5: Epipolar filtering (optional)
    matches_filtered_file = cell_output_dir / f"matches_filtered_cell_{cell_id}.json"
    if enable_epipolar_filtering and not matches_filtered_file.exists():
        print(f"\n  Step 5: Epipolar filtering...")
        # Get image dimensions from first image
        from .utils import get_image_dimensions
        first_img = list(features_data.keys())[0]
        img_width, img_height = get_image_dimensions(Path(first_img))
        
        filter_matches_epipolar_robust(
            matches_file=str(matches_file),
            features_file=str(features_cache_file),
            output_file=str(matches_filtered_file),
            image_width=img_width,
            image_height=img_height
        )
        print(f"    Saved filtered matches")
    
    # Step 6: Compute tracks
    print(f"\n  Step 6: Computing tracks...")
    tracks_file = cell_output_dir / f"tracks_cell_{cell_id}.json"
    
    # Use filtered matches if available
    matches_for_tracks = matches_data
    if enable_epipolar_filtering and matches_filtered_file.exists():
        with open(matches_filtered_file, 'r') as f:
            filtered_matches = json.load(f)
        # Convert to format expected by compute_tracks
        matches_for_tracks = []
        for match_dict in filtered_matches:
            img0_name = match_dict['image0']
            img1_name = match_dict['image1']
            
            # Find full paths
            img0_path = None
            img1_path = None
            for feat_path in features_data.keys():
                if Path(feat_path).name == Path(img0_name).name or img0_name in feat_path:
                    img0_path = feat_path
                if Path(feat_path).name == Path(img1_name).name or img1_name in feat_path:
                    img1_path = feat_path
            
            if img0_path and img1_path:
                matches_for_tracks.append({
                    'image0': img0_path,
                    'image1': img1_path,
                    'matches': np.array(match_dict['matches']),
                    'num_matches': match_dict['num_matches'],
                    'match_confidence': match_dict.get('match_confidence', [])
                })
    
    tracks_data = compute_tracks(matches_for_tracks, features_data)
    
    # Convert sets to lists for JSON serialization
    tracks_data_serializable = {
        'num_tracks': tracks_data['num_tracks'],
        'total_features_in_tracks': tracks_data['total_features_in_tracks'],
        'tracks': []
    }
    for track in tracks_data['tracks']:
        track_serializable = {
            'track_id': track['track_id'],
            'features': track['features'],
            'length': track['length'],
            'images': list(track['images']) if isinstance(track['images'], set) else track['images']
        }
        tracks_data_serializable['tracks'].append(track_serializable)
    
    # Save tracks
    with open(tracks_file, 'w') as f:
        json.dump(tracks_data_serializable, f, indent=2)
    print(f"    Saved {tracks_data_serializable['num_tracks']} tracks")
    
    return {
        'cell_id': cell_id,
        'features': features_data,
        'matches': matches_data,
        'tracks': tracks_data_serializable,  # Use serializable version with sets converted to lists
        'camera_poses': camera_poses,
        'footprint_overlaps': json.load(open(footprint_overlaps_file)) if footprint_overlaps_file.exists() else []
    }


def process_inter_cell_pair(
    cell1_id: str,
    cell2_id: str,
    cell1_images: List[str],
    cell2_images: List[str],
    image_dir: Path,
    output_dir: Path,
    min_overlap_threshold: float,
    enable_epipolar_filtering: bool,
    max_features: int,
    resolution: float,
    focal_length_mm: float,
    sensor_width_mm: float,
    sensor_height_mm: float
) -> Dict:
    """
    Process inter-cell matching between two cells.
    
    Returns:
        Dictionary with keys: 'matches', 'cell1_id', 'cell2_id'
    """
    print(f"\n{'='*60}")
    print(f"Processing INTER-cell between: {cell1_id} <-> {cell2_id}")
    print(f"{'='*60}")
    print(f"  Cell {cell1_id}: {len(cell1_images)} images")
    print(f"  Cell {cell2_id}: {len(cell2_images)} images")
    
    inter_output_dir = output_dir / f"inter_{cell1_id}_{cell2_id}"
    inter_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load features from both cells
    print(f"\n  Step 1: Loading features from both cells...")
    cell1_features_file = output_dir / f"cell_{cell1_id}" / f"features_cache_cell_{cell1_id}.json"
    cell2_features_file = output_dir / f"cell_{cell2_id}" / f"features_cache_cell_{cell2_id}.json"
    
    if not cell1_features_file.exists() or not cell2_features_file.exists():
        print(f"    ERROR: Cell features not found. Run intra-cell processing first.")
        return {'matches': [], 'cell1_id': cell1_id, 'cell2_id': cell2_id}
    
    with open(cell1_features_file, 'r') as f:
        cell1_features = json.load(f)
    with open(cell2_features_file, 'r') as f:
        cell2_features = json.load(f)
    
    print(f"    Cell {cell1_id}: {len(cell1_features)} images with features")
    print(f"    Cell {cell2_id}: {len(cell2_features)} images with features")
    
    # Step 2: Load camera poses and compute footprint overlaps between cells
    print(f"\n  Step 2: Computing footprint overlaps between cells...")
    cell1_poses_file = output_dir / f"cell_{cell1_id}" / "camera_poses.json"
    cell2_poses_file = output_dir / f"cell_{cell2_id}" / "camera_poses.json"
    
    if not cell1_poses_file.exists() or not cell2_poses_file.exists():
        print(f"    ERROR: Camera poses not found. Run intra-cell processing first.")
        return {'matches': [], 'cell1_id': cell1_id, 'cell2_id': cell2_id}
    
    with open(cell1_poses_file, 'r') as f:
        cell1_poses = json.load(f)
    with open(cell2_poses_file, 'r') as f:
        cell2_poses = json.load(f)
    
    # Compute overlaps between cells
    from .utils import compute_footprint_overlap, compute_footprint_polygon
    
    # Find origin from first pose
    origin_lat = None
    origin_lon = None
    for pose in cell1_poses + cell2_poses:
        if pose.get('gps'):
            origin_lat = pose['gps']['latitude']
            origin_lon = pose['gps']['longitude']
            break
    
    if origin_lat is None:
        print(f"    ERROR: No GPS data found")
        return {'matches': [], 'cell1_id': cell1_id, 'cell2_id': cell2_id}
    
    # Compute overlaps
    overlaps = []
    for pose1 in cell1_poses:
        if not pose1.get('gps') or not pose1.get('dji_orientation'):
            continue
        
        img1_name = pose1['image_name']
        poly1, _ = compute_footprint_polygon(
            pose1['gps'],
            pose1['dji_orientation'],
            focal_length_mm,
            sensor_width_mm,
            sensor_height_mm,
            origin_lat,
            origin_lon
        )
        
        if poly1 is None:
            continue
        
        for pose2 in cell2_poses:
            if not pose2.get('gps') or not pose2.get('dji_orientation'):
                continue
            
            img2_name = pose2['image_name']
            poly2, _ = compute_footprint_polygon(
                pose2['gps'],
                pose2['dji_orientation'],
                focal_length_mm,
                sensor_width_mm,
                sensor_height_mm,
                origin_lat,
                origin_lon
            )
            
            if poly2 is None:
                continue
            
            overlap_pct = compute_footprint_overlap(poly1, poly2)
            if overlap_pct >= min_overlap_threshold:
                overlaps.append({
                    'image1': img1_name,
                    'image2': img2_name,
                    'overlap_percentage': overlap_pct
                })
    
    print(f"    Found {len(overlaps)} image pairs with >= {min_overlap_threshold}% overlap")
    
    # Step 3: Match overlapping image pairs
    print(f"\n  Step 3: Matching inter-cell image pairs...")
    matches_file = inter_output_dir / f"matches_inter_{cell1_id}_{cell2_id}.json"
    
    if matches_file.exists():
        print(f"    Loading matches from cache")
        with open(matches_file, 'r') as f:
            matches_data = json.load(f)
    else:
        print(f"    Computing matches for {len(overlaps)} pairs...")
        from .lightglue_matcher import LightGlueMatcher
        matcher = LightGlueMatcher(device='mps')
        
        matches_data = []
        for overlap in tqdm(overlaps, desc="  Matching"):
            img1_name = overlap['image1']
            img2_name = overlap['image2']
            
            # Find feature data
            feat1 = None
            feat2 = None
            for feat_path, feat_data in cell1_features.items():
                if Path(feat_path).name == img1_name or img1_name in feat_path:
                    feat1 = feat_data
                    break
            for feat_path, feat_data in cell2_features.items():
                if Path(feat_path).name == img2_name or img2_name in feat_path:
                    feat2 = feat_data
                    break
            
            if feat1 and feat2:
                matches = matcher.match_features(feat1, feat2)
                if matches and len(matches['matches']) > 0:
                    matches_data.append({
                        'image0': img1_name,
                        'image1': img2_name,
                        'matches': matches['matches'].tolist(),
                        'num_matches': len(matches['matches']),
                        'match_confidence': matches.get('match_confidence', []).tolist() if 'match_confidence' in matches else []
                    })
        
        # Save matches
        with open(matches_file, 'w') as f:
            json.dump(matches_data, f, indent=2)
        print(f"    Saved {len(matches_data)} inter-cell match pairs")
    
    return {
        'matches': matches_data,
        'cell1_id': cell1_id,
        'cell2_id': cell2_id,
        'overlaps': overlaps
    }


def process_multi_cell(
    cell_ids: List[str],
    image_dir: Path,
    output_dir: Path,
    min_overlap_threshold: float = 50.0,
    enable_epipolar_filtering: bool = True,
    max_features: int = 200,
    resolution: float = 1.0,
    focal_length_mm: float = 8.8,
    sensor_width_mm: float = 13.2,
    sensor_height_mm: float = 9.9,
    mode: str = 'ortho'
) -> TiepointGraph:
    """
    Process multiple cells with intra-cell and inter-cell matching.
    
    Args:
        cell_ids: List of cell IDs to process
        image_dir: Directory containing images
        output_dir: Output directory
        min_overlap_threshold: Minimum footprint overlap percentage
        enable_epipolar_filtering: Whether to enable epipolar filtering
        max_features: Maximum features per image
        resolution: Image resolution scale factor
        focal_length_mm: Focal length in millimeters
        sensor_width_mm: Sensor width in millimeters
        sensor_height_mm: Sensor height in millimeters
        mode: Processing mode ('ortho' or 'hybrid')
    
    Returns:
        TiepointGraph with all features, matches, and tracks
    """
    print("=" * 60)
    print("Multi-Cell Tiepoint Matcher")
    print("=" * 60)
    print(f"Processing {len(cell_ids)} cells: {cell_ids}")
    
    # Get images for each cell
    cell_to_images = get_cell_images(str(image_dir))
    
    # Initialize tiepoint graph
    graph = TiepointGraph()
    
    # Store cell images
    for cell_id in cell_ids:
        if cell_id in cell_to_images:
            graph.cell_images[cell_id] = cell_to_images[cell_id]
        else:
            print(f"WARNING: Cell {cell_id} not found in images")
    
    # Step 1: Process each cell (intra-cell)
    print("\n" + "=" * 60)
    print("STEP 1: Processing INTRA-cell features, matches, and tracks")
    print("=" * 60)
    
    intra_results = {}
    for cell_id in cell_ids:
        if cell_id not in cell_to_images:
            continue
        
        cell_images = cell_to_images[cell_id]
        result = process_single_cell(
            cell_id=cell_id,
            cell_images=cell_images,
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
        intra_results[cell_id] = result
        
        # Add to graph
        for img_path, feat_data in result['features'].items():
            img_name = Path(img_path).name
            keypoints = feat_data['keypoints']
            for idx in range(len(keypoints)):
                graph.add_feature(img_name, idx, {
                    'keypoint': keypoints[idx].tolist() if isinstance(keypoints[idx], np.ndarray) else keypoints[idx],
                    'score': feat_data['scores'][idx] if 'scores' in feat_data else 0.0
                }, cell_id)
        
        for match_dict in result['matches']:
            img0 = Path(match_dict['image0']).name
            img1 = Path(match_dict['image1']).name
            matches = np.array(match_dict['matches'])
            for match in matches:
                feat0_idx, feat1_idx = int(match[0]), int(match[1])
                graph.add_match(
                    img0, feat0_idx, img1, feat1_idx,
                    {'confidence': match_dict.get('match_confidence', [])[int(match[0])] if 'match_confidence' in match_dict else 0.0},
                    cell_id, cell_id
                )
        
        for track in result['tracks']['tracks']:
            graph.add_track(track, {cell_id})
    
    # Step 2: Process inter-cell pairs
    if len(cell_ids) > 1:
        print("\n" + "=" * 60)
        print("STEP 2: Processing INTER-cell matches")
        print("=" * 60)
        
        inter_results = {}
        for i, cell1_id in enumerate(cell_ids):
            for cell2_id in cell_ids[i+1:]:
                if cell1_id not in cell_to_images or cell2_id not in cell_to_images:
                    continue
                
                result = process_inter_cell_pair(
                    cell1_id=cell1_id,
                    cell2_id=cell2_id,
                    cell1_images=cell_to_images[cell1_id],
                    cell2_images=cell_to_images[cell2_id],
                    image_dir=image_dir,
                    output_dir=output_dir,
                    min_overlap_threshold=min_overlap_threshold,
                    enable_epipolar_filtering=enable_epipolar_filtering,
                    max_features=max_features,
                    resolution=resolution,
                    focal_length_mm=focal_length_mm,
                    sensor_width_mm=sensor_width_mm,
                    sensor_height_mm=sensor_height_mm
                )
                inter_results[(cell1_id, cell2_id)] = result
                
                # Add inter-cell matches to graph
                for match_dict in result['matches']:
                    img0 = match_dict['image0']
                    img1 = match_dict['image1']
                    matches = np.array(match_dict['matches'])
                    for match in matches:
                        feat0_idx, feat1_idx = int(match[0]), int(match[1])
                        graph.add_match(
                            img0, feat0_idx, img1, feat1_idx,
                            {'confidence': match_dict.get('match_confidence', [])[int(match[0])] if 'match_confidence' in match_dict else 0.0},
                            cell1_id, cell2_id
                        )
    
    # Step 3: Link tracks across cells
    if len(cell_ids) > 1:
        print("\n" + "=" * 60)
        print("STEP 3: Linking tracks across cells")
        print("=" * 60)
        
        linked_tracks = graph.link_tracks_across_cells()
        print(f"  Original tracks: {linked_tracks['num_original_tracks']}")
        print(f"  Linked tracks: {linked_tracks['num_linked_tracks']}")
        
        # Convert sets to lists for JSON serialization
        linked_tracks_serializable = {
            'num_linked_tracks': linked_tracks['num_linked_tracks'],
            'num_original_tracks': linked_tracks['num_original_tracks'],
            'linked_tracks': []
        }
        for track in linked_tracks['linked_tracks']:
            track_serializable = {
                'linked_track_id': track['linked_track_id'],
                'component_track_ids': track['component_track_ids'],
                'features': track['features'],
                'length': track['length'],
                'images': list(track['images']) if isinstance(track['images'], set) else track['images'],
                'cells': track['cells']
            }
            linked_tracks_serializable['linked_tracks'].append(track_serializable)
        
        # Save linked tracks
        linked_tracks_file = output_dir / "linked_tracks.json"
        with open(linked_tracks_file, 'w') as f:
            json.dump(linked_tracks_serializable, f, indent=2)
        print(f"  Saved linked tracks to: {linked_tracks_file}")
    
    # Save tiepoint graph
    graph_file = output_dir / "tiepoint_graph.json"
    graph.save(str(graph_file))
    print(f"\n  Saved tiepoint graph to: {graph_file}")
    
    return graph
