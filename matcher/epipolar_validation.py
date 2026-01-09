"""
Epipolar geometry validation for filtering outlier matches.
Uses camera poses from EXIF metadata to validate feature matches.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pyproj


def gps_to_ecef(lat: float, lon: float, alt: float) -> np.ndarray:
    """Convert GPS coordinates (lat, lon, alt) to ECEF (Earth-Centered, Earth-Fixed) coordinates."""
    # WGS84 ellipsoid parameters
    a = 6378137.0  # Semi-major axis (meters)
    e2 = 0.00669437999014  # First eccentricity squared
    
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    # Compute N (radius of curvature in prime vertical)
    sin_lat = np.sin(lat_rad)
    N = a / np.sqrt(1 - e2 * sin_lat**2)
    
    # Compute ECEF coordinates
    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - e2) + alt) * sin_lat
    
    return np.array([x, y, z])


def gps_to_local_utm(lat: float, lon: float, alt: float, 
                     origin_lat: float, origin_lon: float) -> np.ndarray:
    """Convert GPS to local UTM coordinates relative to an origin."""
    # Convert origin to UTM
    utm_origin = pyproj.Proj(proj='utm', zone=int((origin_lon + 180) / 6) + 1, ellps='WGS84')
    lonlat = pyproj.Proj(proj='latlong', ellps='WGS84')
    
    # Convert origin
    origin_x, origin_y = pyproj.transform(lonlat, utm_origin, origin_lon, origin_lat)
    
    # Convert point
    point_x, point_y = pyproj.transform(lonlat, utm_origin, lon, lat)
    
    # Return relative to origin
    return np.array([point_x - origin_x, point_y - origin_y, alt])


def estimate_camera_intrinsics(image_width: int = 1000, image_height: int = 750,
                               sensor_width_mm: float = 13.2, focal_length_mm: float = 8.8) -> np.ndarray:
    """
    Estimate camera intrinsics matrix.
    
    For DJI FC3682 (typical drone camera):
    - Sensor width: ~13.2mm
    - Focal length: ~8.8mm
    """
    # Focal length in pixels
    focal_length_px = (focal_length_mm / sensor_width_mm) * image_width
    
    # Principal point (assume center of image)
    cx = image_width / 2.0
    cy = image_height / 2.0
    
    K = np.array([
        [focal_length_px, 0, cx],
        [0, focal_length_px, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    return K


def compute_relative_pose(pose1: Dict, pose2: Dict, origin_lat: float, origin_lon: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute relative pose between two cameras.
    
    Returns:
        R: 3x3 rotation matrix (from camera1 to camera2)
        t: 3x1 translation vector (from camera1 to camera2)
    """
    if not pose1.get('gps') or not pose2.get('gps'):
        return None, None
    
    # Convert GPS to local coordinates
    pos1 = gps_to_local_utm(
        pose1['gps']['latitude'],
        pose1['gps']['longitude'],
        pose1['gps']['altitude'],
        origin_lat, origin_lon
    )
    
    pos2 = gps_to_local_utm(
        pose2['gps']['latitude'],
        pose2['gps']['longitude'],
        pose2['gps']['altitude'],
        origin_lat, origin_lon
    )
    
    # Translation vector
    t = pos2 - pos1
    
    # For now, assume cameras are roughly nadir (pointing down)
    # In a full implementation, we'd use orientation data from EXIF
    # For simplicity, assume identity rotation (cameras aligned)
    R = np.eye(3, dtype=np.float64)
    
    return R, t


def compute_essential_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compute essential matrix from relative pose."""
    # Skew-symmetric matrix for translation
    t_skew = np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])
    
    # Essential matrix: E = [t]_x * R
    E = t_skew @ R
    
    return E


def compute_sampson_distance(F: np.ndarray, pt1: np.ndarray, pt2: np.ndarray) -> float:
    """
    Compute Sampson distance for epipolar constraint.
    
    Args:
        F: Fundamental matrix (3x3)
        pt1: Point in image 1 (x, y)
        pt2: Point in image 2 (x, y)
    
    Returns:
        Sampson distance (squared)
    """
    # Convert to homogeneous coordinates
    p1 = np.array([pt1[0], pt1[1], 1.0])
    p2 = np.array([pt2[0], pt2[1], 1.0])
    
    # Epipolar line in image 2: l2 = F * p1
    l2 = F @ p1
    
    # Distance from point to epipolar line
    numerator = (p2.T @ F @ p1)**2
    denominator = l2[0]**2 + l2[1]**2
    
    if denominator < 1e-10:
        return np.inf
    
    sampson_dist_sq = numerator / denominator
    
    return sampson_dist_sq


def validate_matches_epipolar(
    matches: List[Dict],
    features: Dict,
    camera_poses: Dict[str, Dict],
    image_width: int = 1000,
    image_height: int = 750,
    threshold: float = 2.0,
    origin_lat: Optional[float] = None,
    origin_lon: Optional[float] = None
) -> Tuple[List[Dict], Dict]:
    """
    Validate matches using epipolar geometry.
    
    Args:
        matches: List of match dictionaries
        features: Dictionary mapping image_path to feature dict
        camera_poses: Dictionary mapping image_name to pose dict
        image_width: Image width in pixels
        image_height: Image height in pixels
        threshold: Sampson distance threshold (pixels)
        origin_lat: Origin latitude for local coordinate system
        origin_lon: Origin longitude for local coordinate system
    
    Returns:
        filtered_matches: List of validated matches
        stats: Dictionary with validation statistics
    """
    # Estimate camera intrinsics
    K = estimate_camera_intrinsics(image_width, image_height)
    K_inv = np.linalg.inv(K)
    
    # Determine origin from first pose if not provided
    if origin_lat is None or origin_lon is None:
        first_pose = next(iter(camera_poses.values()))
        if first_pose.get('gps'):
            origin_lat = first_pose['gps']['latitude']
            origin_lon = first_pose['gps']['longitude']
        else:
            print("Warning: No GPS data available, cannot validate epipolar geometry")
            return matches, {'total_matches': len(matches), 'valid_matches': len(matches), 'invalid_matches': 0}
    
    filtered_matches = []
    total_matches = 0
    valid_matches = 0
    invalid_matches = 0
    
    for match_dict in matches:
        img0_path = match_dict['image0']
        img1_path = match_dict['image1']
        
        img0_name = Path(img0_path).name
        img1_name = Path(img1_path).name
        
        # Remove quarter_ prefix if present for pose lookup
        img0_name_clean = img0_name.replace('quarter_', '')
        img1_name_clean = img1_name.replace('quarter_', '')
        
        # Get camera poses
        pose0 = camera_poses.get(img0_name_clean) or camera_poses.get(img0_name)
        pose1 = camera_poses.get(img1_name_clean) or camera_poses.get(img1_name)
        
        if not pose0 or not pose1:
            # Skip if poses not available
            filtered_matches.append(match_dict)
            continue
        
        if not pose0.get('gps') or not pose1.get('gps'):
            # Skip if GPS not available
            filtered_matches.append(match_dict)
            continue
        
        # Get features
        if img0_path not in features or img1_path not in features:
            filtered_matches.append(match_dict)
            continue
        
        feat0 = features[img0_path]
        feat1 = features[img1_path]
        
        keypoints0 = feat0['keypoints']
        keypoints1 = feat1['keypoints']
        
        # Compute relative pose
        R, t = compute_relative_pose(pose0, pose1, origin_lat, origin_lon)
        if R is None or t is None:
            filtered_matches.append(match_dict)
            continue
        
        # Compute essential matrix
        E = compute_essential_matrix(R, t)
        
        # Compute fundamental matrix: F = K_inv^T * E * K_inv
        F = K_inv.T @ E @ K_inv
        
        # Validate each match
        match_indices = np.array(match_dict['matches'])
        valid_match_indices = []
        
        for match_idx in match_indices:
            idx0, idx1 = int(match_idx[0]), int(match_idx[1])
            
            if idx0 >= len(keypoints0) or idx1 >= len(keypoints1):
                continue
            
            pt0 = keypoints0[idx0]
            pt1 = keypoints1[idx1]
            
            # Compute Sampson distance
            sampson_dist_sq = compute_sampson_distance(F, pt0, pt1)
            sampson_dist = np.sqrt(sampson_dist_sq)
            
            if sampson_dist <= threshold:
                valid_match_indices.append(match_idx)
                valid_matches += 1
            else:
                invalid_matches += 1
            
            total_matches += 1
        
        # Only keep matches with at least some valid correspondences
        if len(valid_match_indices) > 0:
            filtered_match_dict = match_dict.copy()
            filtered_match_dict['matches'] = np.array(valid_match_indices)
            filtered_match_dict['num_matches'] = len(valid_match_indices)
            filtered_matches.append(filtered_match_dict)
    
    stats = {
        'total_matches': total_matches,
        'valid_matches': valid_matches,
        'invalid_matches': invalid_matches,
        'validation_rate': valid_matches / total_matches if total_matches > 0 else 0.0
    }
    
    return filtered_matches, stats


def filter_matches_epipolar_robust(
    matches_file: str,
    features_file: str,
    output_file: str,
    image_width: int = 1000,
    image_height: int = 750,
    ransac_threshold: float = 0.5,
    confidence: float = 0.999,
    max_iters: int = 2000
) -> Dict:
    """
    Filter matches using robust epipolar geometry estimation (RANSAC).
    
    For each image pair in matches_file, compute fundamental matrix using RANSAC
    and keep only inlier matches.
    
    Args:
        matches_file: Path to unfiltered matches JSON file
        features_file: Path to features JSON file
        output_file: Path to save filtered matches JSON file
        image_width: Image width in pixels
        image_height: Image height in pixels
        ransac_threshold: RANSAC threshold in pixels (default: 1.0)
        confidence: RANSAC confidence (default: 0.999)
        max_iters: Maximum RANSAC iterations (default: 2000)
    
    Returns:
        Dictionary with filtering statistics
    """
    import json
    
    # Load matches and features
    with open(matches_file, 'r') as f:
        matches_data = json.load(f)
    
    with open(features_file, 'r') as f:
        features_data = json.load(f)
    
    # Estimate camera intrinsics
    K = estimate_camera_intrinsics(image_width, image_height)
    
    filtered_matches = []
    total_matches_before = 0
    total_matches_after = 0
    pairs_processed = 0
    pairs_with_inliers = 0
    
    print(f"Filtering matches using robust epipolar geometry (RANSAC)...")
    print(f"  RANSAC threshold: {ransac_threshold} pixels")
    print(f"  Confidence: {confidence}")
    print(f"  Max iterations: {max_iters}")
    
    for match_dict in matches_data:
        img0_path = match_dict['image0']
        img1_path = match_dict['image1']
        match_indices = np.array(match_dict['matches'])
        
        # Get image names for feature lookup
        img0_name = Path(img0_path).name
        img1_name = Path(img1_path).name
        
        # Try to find features (handle both with and without quarter_ prefix)
        feat0 = None
        feat1 = None
        
        # Try exact match first
        if img0_path in features_data:
            feat0 = features_data[img0_path]
        elif img0_name in features_data:
            feat0 = features_data[img0_name]
        else:
            # Try without quarter_ prefix
            img0_name_clean = img0_name.replace('quarter_', '')
            for key in features_data.keys():
                if key.endswith(img0_name_clean):
                    feat0 = features_data[key]
                    break
        
        if img1_path in features_data:
            feat1 = features_data[img1_path]
        elif img1_name in features_data:
            feat1 = features_data[img1_name]
        else:
            img1_name_clean = img1_name.replace('quarter_', '')
            for key in features_data.keys():
                if key.endswith(img1_name_clean):
                    feat1 = features_data[key]
                    break
        
        if feat0 is None or feat1 is None:
            # Skip if features not found
            filtered_matches.append(match_dict)
            continue
        
        # Get keypoints
        if isinstance(feat0, dict):
            keypoints0 = np.array(feat0.get('keypoints', []))
        else:
            keypoints0 = np.array(feat0.get('keypoints', []))
        
        if isinstance(feat1, dict):
            keypoints1 = np.array(feat1.get('keypoints', []))
        else:
            keypoints1 = np.array(feat1.get('keypoints', []))
        
        if len(keypoints0) == 0 or len(keypoints1) == 0:
            filtered_matches.append(match_dict)
            continue
        
        # Extract matched points
        if len(match_indices) < 8:
            # Need at least 8 points for fundamental matrix estimation
            filtered_matches.append(match_dict)
            continue
        
        pts0 = []
        pts1 = []
        
        for match_idx in match_indices:
            idx0, idx1 = int(match_idx[0]), int(match_idx[1])
            if idx0 < len(keypoints0) and idx1 < len(keypoints1):
                pts0.append(keypoints0[idx0])
                pts1.append(keypoints1[idx1])
        
        if len(pts0) < 8:
            filtered_matches.append(match_dict)
            continue
        
        pts0 = np.array(pts0, dtype=np.float32)
        pts1 = np.array(pts1, dtype=np.float32)
        
        total_matches_before += len(pts0)
        pairs_processed += 1
        
        # Compute fundamental matrix using RANSAC
        try:
            F, mask = cv2.findFundamentalMat(
                pts0, pts1,
                method=cv2.FM_RANSAC,
                ransacReprojThreshold=ransac_threshold,
                confidence=confidence,
                maxIters=max_iters
            )
            
            if F is None or mask is None:
                # RANSAC failed, keep original matches
                filtered_matches.append(match_dict)
                continue
            
            # Count inliers
            inlier_mask = mask.ravel() == 1
            num_inliers = np.sum(inlier_mask)
            
            # Require at least 8 inliers AND at least 50% of matches to be inliers
            min_inlier_ratio = 0.5
            if num_inliers < 8 or (num_inliers / len(pts0)) < min_inlier_ratio:
                # Too few inliers or too low inlier ratio, skip this pair
                continue
            
            # Filter matches to keep only inliers
            inlier_indices = match_indices[inlier_mask]
            
            filtered_match_dict = match_dict.copy()
            filtered_match_dict['matches'] = inlier_indices.tolist()
            filtered_match_dict['num_matches'] = len(inlier_indices)
            
            # Filter match_confidence if present
            if 'match_confidence' in match_dict and match_dict['match_confidence']:
                if isinstance(match_dict['match_confidence'], list):
                    filtered_match_dict['match_confidence'] = [
                        match_dict['match_confidence'][i] 
                        for i in range(len(match_dict['match_confidence'])) 
                        if inlier_mask[i]
                    ]
                else:
                    # It's a numpy array
                    filtered_match_dict['match_confidence'] = np.array(match_dict['match_confidence'])[inlier_mask].tolist()
            
            filtered_matches.append(filtered_match_dict)
            total_matches_after += num_inliers
            pairs_with_inliers += 1
            
        except Exception as e:
            # If RANSAC fails, keep original matches
            print(f"Warning: RANSAC failed for {img0_name} <-> {img1_name}: {e}")
            filtered_matches.append(match_dict)
    
    # Save filtered matches
    with open(output_file, 'w') as f:
        json.dump(filtered_matches, f, indent=2)
    
    stats = {
        'total_pairs': len(matches_data),
        'pairs_processed': pairs_processed,
        'pairs_with_inliers': pairs_with_inliers,
        'total_matches_before': total_matches_before,
        'total_matches_after': total_matches_after,
        'filtering_rate': total_matches_after / total_matches_before if total_matches_before > 0 else 0.0,
        'pairs_kept': len(filtered_matches)
    }
    
    print(f"  Processed {pairs_processed} pairs")
    print(f"  Pairs with inliers: {pairs_with_inliers}")
    print(f"  Total matches before: {total_matches_before}")
    print(f"  Total matches after: {total_matches_after}")
    print(f"  Filtering rate: {stats['filtering_rate']:.2%}")
    
    return stats
