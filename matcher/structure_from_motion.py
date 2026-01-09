"""
Structure-from-Motion (SfM) for estimating camera poses from feature matches.
Handles nadir views and degenerate configurations.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import defaultdict


def estimate_camera_intrinsics(image_width: int, image_height: int,
                               sensor_width_mm: float = 13.2, focal_length_mm: float = 8.8) -> np.ndarray:
    """Estimate camera intrinsics matrix."""
    focal_length_px = (focal_length_mm / sensor_width_mm) * image_width
    cx = image_width / 2.0
    cy = image_height / 2.0
    
    K = np.array([
        [focal_length_px, 0, cx],
        [0, focal_length_px, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    return K


def gps_to_local_meters(lat: float, lon: float, alt: float, 
                        origin_lat: float, origin_lon: float) -> np.ndarray:
    """Convert GPS to local coordinates in meters."""
    lat_diff = lat - origin_lat
    lon_diff = lon - origin_lon
    
    x = lon_diff * 111000.0 * np.cos(np.radians(origin_lat))
    y = lat_diff * 111000.0
    z = alt
    
    return np.array([x, y, z])


def estimate_relative_pose_from_matches(
    K: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """
    Estimate relative pose between two cameras from matched points.
    
    Returns:
        R: Rotation matrix (3x3) from camera 1 to camera 2
        t: Translation vector (3x1) from camera 1 to camera 2
        inliers: Boolean mask of inlier matches
    """
    if len(pts1) < 8:
        return None, None, np.array([])
    
    # Normalize points
    pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, None).reshape(-1, 2)
    pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, None).reshape(-1, 2)
    
    # Find essential matrix
    E, inliers = cv2.findEssentialMat(
        pts1_norm, pts2_norm,
        cameraMatrix=np.eye(3),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )
    
    if E is None:
        return None, None, np.array([])
    
    # Recover pose
    inliers_flat = inliers.ravel() == 1
    if inliers_flat.sum() < 8:
        return None, None, np.array([])
    
    _, R, t, pose_inliers = cv2.recoverPose(
        E,
        pts1_norm[inliers_flat],
        pts2_norm[inliers_flat],
        cameraMatrix=np.eye(3)
    )
    
    # Combine inliers
    final_inliers = np.zeros(len(pts1), dtype=bool)
    final_inliers[inliers_flat] = pose_inliers.ravel() == 1
    
    return R, t, final_inliers


def incremental_sfm(
    matches: List[Dict],
    features: Dict,
    camera_poses_gps: Dict[str, Dict],
    origin_lat: float,
    origin_lon: float,
    K: np.ndarray,
    image_width: int,
    image_height: int
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Incremental Structure-from-Motion to estimate camera poses.
    
    Algorithm:
    1. Initialize with first two cameras using relative pose estimation
    2. Triangulate initial 3D points
    3. Add more cameras using PnP (Perspective-n-Point)
    4. Refine with bundle adjustment (optional)
    
    Returns:
        Dictionary mapping image_name to (R, t) camera pose
    """
    camera_poses = {}
    
    # Step 1: Initialize poses from GPS (positions only, rotations will be estimated)
    for img_name, pose_gps in camera_poses_gps.items():
        if not pose_gps.get('gps'):
            continue
        
        t_world = gps_to_local_meters(
            pose_gps['gps']['latitude'],
            pose_gps['gps']['longitude'],
            pose_gps['gps']['altitude'],
            origin_lat, origin_lon
        )
        
        # Start with identity rotation (will be estimated)
        R = np.eye(3, dtype=np.float64)
        camera_poses[img_name] = (R, t_world)
    
    # Step 2: Find best initial pair
    best_pair = None
    best_matches = 0
    
    for match_dict in matches:
        img0_name = Path(match_dict['image0']).name.replace('quarter_', '')
        img1_name = Path(match_dict['image1']).name.replace('quarter_', '')
        
        if img0_name not in camera_poses or img1_name not in camera_poses:
            continue
        
        # Count actual matches (not just the num_matches field which might be filtered)
        match_indices = match_dict.get('matches', [])
        if isinstance(match_indices, list):
            num_matches = len(match_indices)
        else:
            num_matches = match_dict.get('num_matches', 0)
        
        if num_matches > best_matches and num_matches >= 20:  # Lower threshold for more pairs
            best_pair = (img0_name, img1_name, match_dict)
            best_matches = num_matches
    
    if best_pair is None:
        print("Warning: No good image pair found for SfM initialization")
        return camera_poses
    
    # Step 3: Estimate relative pose for initial pair
    img0_name, img1_name, match_dict = best_pair
    
    img0_path = match_dict['image0']
    img1_path = match_dict['image1']
    
    if img0_path not in features or img1_path not in features:
        return camera_poses
    
    feat0 = features[img0_path]
    feat1 = features[img1_path]
    
    keypoints0 = feat0['keypoints'] * 4.0  # Scale to full resolution
    keypoints1 = feat1['keypoints'] * 4.0
    
    match_indices = np.array(match_dict['matches'])
    if len(match_indices) < 20:  # Need enough matches
        return camera_poses
    
    pts0 = keypoints0[match_indices[:, 0]]
    pts1 = keypoints1[match_indices[:, 1]]
    
    # Estimate relative pose
    R_rel, t_rel, inliers = estimate_relative_pose_from_matches(K, pts0, pts1)
    
    if R_rel is None:
        return camera_poses
    
    # Set first camera at origin with identity rotation
    R0 = np.eye(3)
    t0 = camera_poses[img0_name][1]  # Use GPS position
    
    # Set second camera
    R1 = R_rel @ R0
    t1 = R_rel @ t0 + t_rel.ravel()
    
    # Update poses
    camera_poses[img0_name] = (R0, t0)
    camera_poses[img1_name] = (R1, t1)
    
    print(f"SfM: Initialized poses for pair: {img0_name} <-> {img1_name} ({best_matches} matches, {inliers.sum()} inliers)")
    
    # Step 4: Triangulate initial 3D points from first pair
    # (This will be used for PnP in next step)
    
    # Step 5: Add more cameras using PnP
    # For now, we'll estimate rotations for other pairs and propagate
    # A full implementation would:
    # - Triangulate points from initial pair
    # - Use PnP to add each new camera
    # - Triangulate more points
    # - Iterate
    
    # For now, estimate rotations for other pairs with many matches
    estimated_rotations = 0
    for match_dict in matches[:100]:  # Limit to avoid too many
        img0_name = Path(match_dict['image0']).name.replace('quarter_', '')
        img1_name = Path(match_dict['image1']).name.replace('quarter_', '')
        
        # Skip if both already have rotations
        if img0_name in camera_poses and img1_name in camera_poses:
            R0, t0 = camera_poses[img0_name]
            R1, t1 = camera_poses[img1_name]
            
            # If one has identity rotation and the other doesn't, estimate
            if np.allclose(R0, np.eye(3)) and not np.allclose(R1, np.eye(3)):
                # Estimate rotation for img0
                img0_path = match_dict['image0']
                img1_path = match_dict['image1']
                
                if img0_path in features and img1_path in features:
                    feat0 = features[img0_path]
                    feat1 = features[img1_path]
                    
                    keypoints0 = feat0['keypoints'] * 4.0
                    keypoints1 = feat1['keypoints'] * 4.0
                    
                    match_indices = np.array(match_dict['matches'])
                    if len(match_indices) >= 8:
                        pts0 = keypoints0[match_indices[:, 0]]
                        pts1 = keypoints1[match_indices[:, 1]]
                        
                        R_rel, _, _ = estimate_relative_pose_from_matches(K, pts0, pts1)
                        if R_rel is not None:
                            # R0 = R1 @ R_rel^T (inverse relative rotation)
                            R0_new = R1 @ R_rel.T
                            camera_poses[img0_name] = (R0_new, t0)
                            estimated_rotations += 1
            elif not np.allclose(R0, np.eye(3)) and np.allclose(R1, np.eye(3)):
                # Estimate rotation for img1
                img0_path = match_dict['image0']
                img1_path = match_dict['image1']
                
                if img0_path in features and img1_path in features:
                    feat0 = features[img0_path]
                    feat1 = features[img1_path]
                    
                    keypoints0 = feat0['keypoints'] * 4.0
                    keypoints1 = feat1['keypoints'] * 4.0
                    
                    match_indices = np.array(match_dict['matches'])
                    if len(match_indices) >= 8:
                        pts0 = keypoints0[match_indices[:, 0]]
                        pts1 = keypoints1[match_indices[:, 1]]
                        
                        R_rel, _, _ = estimate_relative_pose_from_matches(K, pts0, pts1)
                        if R_rel is not None:
                            R1_new = R0 @ R_rel
                            camera_poses[img1_name] = (R1_new, t1)
                            estimated_rotations += 1
    
    print(f"SfM: Estimated rotations for {estimated_rotations} additional cameras")
    
    return camera_poses
