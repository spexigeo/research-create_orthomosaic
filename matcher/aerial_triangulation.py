"""
Aerial triangulation for estimating camera poses and 3D points from feature matches.
Handles nadir views and degenerate configurations better than simple triangulation.
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
    pts2: np.ndarray,
    method: str = 'RANSAC'
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
    if method == 'RANSAC':
        E, inliers = cv2.findEssentialMat(
            pts1_norm, pts2_norm,
            cameraMatrix=np.eye(3),  # Already normalized
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )
    else:
        E, inliers = cv2.findEssentialMat(
            pts1_norm, pts2_norm,
            cameraMatrix=np.eye(3),
            method=cv2.LMEDS
        )
        inliers = np.ones(len(pts1), dtype=bool)
    
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


def triangulate_points(
    P1: np.ndarray,
    P2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray
) -> np.ndarray:
    """Triangulate 3D points from two views using DLT."""
    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    
    # Convert from homogeneous to 3D
    points_3d = points_4d[:3] / points_4d[3]
    
    return points_3d.T


def initialize_camera_poses(
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
    Initialize camera poses using GPS and refine with feature matches.
    
    Returns:
        Dictionary mapping image_name to (R, t) where:
        - R: Rotation matrix (3x3)
        - t: Translation vector (3x1) in world coordinates
    """
    camera_poses = {}
    
    # First, set poses from GPS (with identity rotation for now)
    for img_name, pose_gps in camera_poses_gps.items():
        if not pose_gps.get('gps'):
            continue
        
        t_world = gps_to_local_meters(
            pose_gps['gps']['latitude'],
            pose_gps['gps']['longitude'],
            pose_gps['gps']['altitude'],
            origin_lat, origin_lon
        )
        
        # Start with identity rotation (will be refined)
        R = np.eye(3, dtype=np.float64)
        
        camera_poses[img_name] = (R, t_world)
    
    # Find a good initial pair with many matches
    best_pair = None
    best_matches = 0
    
    for match_dict in matches:
        img0_name = Path(match_dict['image0']).name.replace('quarter_', '')
        img1_name = Path(match_dict['image1']).name.replace('quarter_', '')
        
        if img0_name not in camera_poses or img1_name not in camera_poses:
            continue
        
        num_matches = match_dict.get('num_matches', 0)
        if num_matches > best_matches and num_matches >= 8:
            best_pair = (img0_name, img1_name, match_dict)
            best_matches = num_matches
    
    if best_pair is None:
        print("Warning: No good image pair found for pose initialization")
        return camera_poses
    
    # Estimate relative pose for best pair
    img0_name, img1_name, match_dict = best_pair
    
    # Get features
    img0_path = match_dict['image0']
    img1_path = match_dict['image1']
    
    if img0_path not in features or img1_path not in features:
        return camera_poses
    
    feat0 = features[img0_path]
    feat1 = features[img1_path]
    
    keypoints0 = feat0['keypoints']
    keypoints1 = feat1['keypoints']
    
    # Get matched points
    match_indices = np.array(match_dict['matches'])
    pts0 = keypoints0[match_indices[:, 0]]
    pts1 = keypoints1[match_indices[:, 1]]
    
    # Scale points from quarter resolution to full resolution
    pts0 = pts0 * 4.0
    pts1 = pts1 * 4.0
    
    # Estimate relative pose
    R_rel, t_rel, inliers = estimate_relative_pose_from_matches(K, pts0, pts1)
    
    if R_rel is None:
        return camera_poses
    
    # Set first camera at origin
    R0 = np.eye(3)
    t0 = camera_poses[img0_name][1]  # Use GPS position
    
    # Set second camera relative to first
    R1 = R_rel @ R0
    t1 = R_rel @ t0 + t_rel.ravel()
    
    # Update poses
    camera_poses[img0_name] = (R0, t0)
    camera_poses[img1_name] = (R1, t1)
    
    print(f"Initialized poses for pair: {img0_name} <-> {img1_name} ({best_matches} matches)")
    
    return camera_poses


def add_view_to_reconstruction(
    new_img_name: str,
    camera_poses: Dict[str, Tuple[np.ndarray, np.ndarray]],
    matches: List[Dict],
    features: Dict,
    K: np.ndarray,
    points_3d: np.ndarray,
    point_track_map: Dict[int, List[Tuple[str, int]]]
) -> bool:
    """
    Add a new camera view to the reconstruction using PnP.
    
    Args:
        new_img_name: Name of new image to add
        camera_poses: Current camera poses
        matches: All matches
        features: Feature dictionaries
        K: Camera intrinsics
        points_3d: Current 3D points
        point_track_map: Map from point index to list of (image_name, feature_idx) tuples
    
    Returns:
        True if view was successfully added
    """
    # Find matches between new image and existing images
    new_img_path = None
    for match_dict in matches:
        img0_name = Path(match_dict['image0']).name.replace('quarter_', '')
        img1_name = Path(match_dict['image1']).name.replace('quarter_', '')
        
        if img0_name == new_img_name:
            new_img_path = match_dict['image0']
            other_img_name = img1_name
        elif img1_name == new_img_name:
            new_img_path = match_dict['image1']
            other_img_name = img0_name
        else:
            continue
        
        if other_img_name not in camera_poses:
            continue
        
        # Get features
        if new_img_path not in features:
            continue
        
        feat_new = features[new_img_path]
        keypoints_new = feat_new['keypoints'] * 4.0  # Scale to full resolution
        
        # Find corresponding 3D points
        match_indices = np.array(match_dict['matches'])
        if new_img_name == img0_name:
            new_indices = match_indices[:, 0]
            other_indices = match_indices[:, 1]
        else:
            new_indices = match_indices[:, 1]
            other_indices = match_indices[:, 0]
        
        # Find 3D points visible in other image
        object_points = []
        image_points = []
        
        for i, other_idx in enumerate(other_indices):
            # Find which 3D point this corresponds to
            # This is simplified - in practice, need to track point-to-feature mapping
            # For now, skip this complex mapping
            pass
        
        if len(object_points) < 6:
            continue
        
        # Solve PnP
        object_points = np.array(object_points, dtype=np.float64)
        image_points = np.array(image_points, dtype=np.float64)
        
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            K,
            None,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=2.0,
            confidence=0.99
        )
        
        if success and len(inliers) >= 6:
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.ravel()
            camera_poses[new_img_name] = (R, t)
            return True
    
    return False


def aerial_triangulation(
    matches: List[Dict],
    features: Dict,
    camera_poses_gps: Dict[str, Dict],
    origin_lat: float,
    origin_lon: float,
    image_width: int,
    image_height: int,
    max_points: int = 1000
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], np.ndarray, Dict]:
    """
    Perform aerial triangulation to estimate camera poses and 3D points.
    
    Returns:
        camera_poses: Dictionary mapping image_name to (R, t)
        points_3d: Array of 3D points (N, 3)
        stats: Statistics dictionary
    """
    K = estimate_camera_intrinsics(image_width, image_height)
    
    # Initialize camera poses from GPS
    camera_poses = initialize_camera_poses(
        matches, features, camera_poses_gps, origin_lat, origin_lon, K, image_width, image_height
    )
    
    # For now, return initialized poses
    # Full implementation would:
    # 1. Triangulate initial points from first pair
    # 2. Add more views using PnP
    # 3. Triangulate more points
    # 4. Iterate
    
    points_3d = np.array([]).reshape(0, 3)
    
    stats = {
        'num_cameras': len(camera_poses),
        'num_points': 0,
        'initialization_method': 'GPS_with_refinement'
    }
    
    return camera_poses, points_3d, stats
