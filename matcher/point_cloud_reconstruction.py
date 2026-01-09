"""
Point cloud reconstruction from feature tracks using triangulation.
Uses camera poses and robust methods to filter outliers.
"""

import numpy as np
import cv2
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from scipy.optimize import least_squares
from collections import defaultdict


def gps_to_local_utm(lat: float, lon: float, alt: float, 
                     origin_lat: float, origin_lon: float) -> np.ndarray:
    """Convert GPS to local coordinates in meters relative to an origin.
    
    Uses simple approximation: 1 degree latitude ≈ 111km, longitude depends on latitude.
    More accurate than UTM for small areas and avoids zone issues.
    """
    # Simple approximation (good for small areas)
    # 1 degree latitude ≈ 111,000 meters
    # 1 degree longitude ≈ 111,000 * cos(latitude) meters
    
    lat_diff = lat - origin_lat
    lon_diff = lon - origin_lon
    
    # Convert to meters
    x = lon_diff * 111000.0 * np.cos(np.radians(origin_lat))
    y = lat_diff * 111000.0
    z = alt
    
    return np.array([x, y, z])


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


def build_camera_matrices(poses: List[Dict], origin_lat: float, origin_lon: float,
                          K: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Build camera projection matrices from poses.
    
    Returns:
        List of (P, R, t) tuples where P is projection matrix, R is rotation, t is translation
    """
    camera_matrices = []
    
    for pose in poses:
        if not pose.get('gps'):
            camera_matrices.append(None)
            continue
        
        # Convert GPS to local coordinates
        t = gps_to_local_utm(
            pose['gps']['latitude'],
            pose['gps']['longitude'],
            pose['gps']['altitude'],
            origin_lat, origin_lon
        )
        
        # For now, assume identity rotation (nadir view)
        # In full implementation, use orientation from EXIF
        R = np.eye(3, dtype=np.float64)
        
        # Build projection matrix: P = K [R | t]
        Rt = np.hstack([R, t.reshape(3, 1)])
        P = K @ Rt
        
        camera_matrices.append((P, R, t))
    
    return camera_matrices


def linear_triangulation(P1: np.ndarray, P2: np.ndarray, 
                        pt1: np.ndarray, pt2: np.ndarray) -> np.ndarray:
    """
    Linear triangulation using DLT (Direct Linear Transform).
    
    Args:
        P1, P2: Camera projection matrices (3x4)
        pt1, pt2: Image points (x, y)
    
    Returns:
        3D point in homogeneous coordinates (4,)
    """
    # Build system of equations
    A = np.zeros((4, 4))
    
    # From first camera
    A[0] = pt1[0] * P1[2] - P1[0]
    A[1] = pt1[1] * P1[2] - P1[1]
    
    # From second camera
    A[2] = pt2[0] * P2[2] - P2[0]
    A[3] = pt2[1] * P2[2] - P2[1]
    
    # Solve using SVD
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    
    # Normalize to get 3D point
    if abs(X[3]) > 1e-10:
        X = X / X[3]
    
    return X[:3]  # Return 3D point (x, y, z)


def reproject_point(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Reproject 3D point to image coordinates."""
    X_hom = np.append(X, 1.0)
    x_hom = P @ X_hom
    if abs(x_hom[2]) > 1e-10:
        x_hom = x_hom / x_hom[2]
    return x_hom[:2]


def compute_reprojection_error(P: np.ndarray, X: np.ndarray, pt: np.ndarray) -> float:
    """Compute reprojection error for a 3D point."""
    pt_reproj = reproject_point(P, X)
    error = np.linalg.norm(pt - pt_reproj)
    return error


def triangulate_track_robust(
    track: Dict,
    features: Dict,
    camera_poses: Dict[str, Dict],
    camera_matrices: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    K: np.ndarray,
    max_reprojection_error: float = 2.0,
    min_views: int = 2
) -> Optional[Dict]:
    """
    Triangulate a 3D point from a feature track using robust methods.
    
    Args:
        track: Track dictionary with 'features' list
        features: Dictionary mapping image_path to feature dict
        camera_poses: Dictionary mapping image_name to pose dict
        camera_matrices: Dictionary mapping image_name to (P, R, t)
        K: Camera intrinsics matrix
        max_reprojection_error: Maximum allowed reprojection error (pixels)
        min_views: Minimum number of views required
    
    Returns:
        Dictionary with 3D point and statistics, or None if triangulation fails
    """
    track_features = track['features']
    
    if len(track_features) < min_views:
        return None
    
    # Collect valid views (images with poses and features)
    views = []
    for img_name, feat_idx in track_features:
        # Remove quarter_ prefix if present
        img_name_clean = img_name.replace('quarter_', '')
        
        pose = camera_poses.get(img_name_clean)
        cam_matrix = camera_matrices.get(img_name_clean)
        
        if pose is None or cam_matrix is None:
            continue
        
        # Find feature in features dict (try both with and without quarter_ prefix)
        feat_dict = None
        for key in features.keys():
            if Path(key).name == img_name or Path(key).name == img_name_clean:
                feat_dict = features[key]
                break
        
        if feat_dict is None:
            continue
        
        keypoints = feat_dict['keypoints']
        if feat_idx >= len(keypoints):
            continue
        
        pt = keypoints[feat_idx]
        views.append({
            'image_name': img_name_clean,
            'point': pt,
            'P': cam_matrix[0],  # Projection matrix
            'R': cam_matrix[1],  # Rotation
            't': cam_matrix[2]   # Translation
        })
    
    if len(views) < min_views:
        return None
    
    # Try all pairs for initial triangulation
    best_X = None
    best_error = np.inf
    best_pair = None
    
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            P1 = views[i]['P']
            P2 = views[j]['P']
            pt1 = views[i]['point']
            pt2 = views[j]['point']
            
            # Triangulate
            X = linear_triangulation(P1, P2, pt1, pt2)
            
            # Check if point is valid and within reasonable range
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                continue
            
            # Filter out points that are clearly degenerate (too far from cameras)
            # For a drone at ~100m altitude, points should be within ~500m horizontally
            # and below the camera (negative Z in camera frame, but we're in world frame)
            # In world frame with cameras at ~100m altitude, ground points should be at ~0m Z
            point_distance = np.linalg.norm(X)
            if point_distance > 1000.0:  # Filter points more than 1km away
                continue
            
            # Compute reprojection error for all views
            total_error = 0.0
            valid_views = 0
            
            for view in views:
                error = compute_reprojection_error(view['P'], X, view['point'])
                if error <= max_reprojection_error:
                    total_error += error
                    valid_views += 1
            
            # Prefer triangulations with more inliers
            if valid_views >= min_views:
                avg_error = total_error / valid_views if valid_views > 0 else np.inf
                if avg_error < best_error:
                    best_error = avg_error
                    best_X = X
                    best_pair = (i, j)
    
    if best_X is None:
        return None
    
    # Refine using all inlier views
    inlier_views = []
    for view in views:
        error = compute_reprojection_error(view['P'], best_X, view['point'])
        if error <= max_reprojection_error:
            inlier_views.append(view)
    
    if len(inlier_views) < min_views:
        return None
    
    # Compute final statistics
    reprojection_errors = []
    for view in inlier_views:
        error = compute_reprojection_error(view['P'], best_X, view['point'])
        reprojection_errors.append(error)
    
    # Final check: ensure point is within reasonable range
    point_distance = np.linalg.norm(best_X)
    if point_distance > 1000.0:  # Filter points more than 1km away
        return None
    
    return {
        'point_3d': best_X.tolist(),
        'num_views': len(inlier_views),
        'reprojection_error_mean': np.mean(reprojection_errors),
        'reprojection_error_max': np.max(reprojection_errors),
        'reprojection_error_std': np.std(reprojection_errors),
        'inlier_views': [v['image_name'] for v in inlier_views],
        'distance_from_origin': float(point_distance)
    }


def reconstruct_point_cloud(
    tracks: List[Dict],
    features: Dict,
    camera_poses: Dict[str, Dict],
    origin_lat: float,
    origin_lon: float,
    image_width: int,
    image_height: int,
    max_reprojection_error: float = 2.0,
    min_views: int = 2
) -> Tuple[List[Dict], Dict]:
    """
    Reconstruct point cloud from feature tracks.
    
    Args:
        tracks: List of track dictionaries
        features: Dictionary mapping image_path to feature dict
        camera_poses: Dictionary mapping image_name to pose dict
        origin_lat: Origin latitude for local coordinate system
        origin_lon: Origin longitude for local coordinate system
        image_width: Image width in pixels
        image_height: Image height in pixels
        max_reprojection_error: Maximum allowed reprojection error (pixels)
        min_views: Minimum number of views required
    
    Returns:
        point_cloud: List of 3D points with statistics
        stats: Dictionary with reconstruction statistics
    """
    # Estimate camera intrinsics
    K = estimate_camera_intrinsics(image_width, image_height)
    
    # Build camera matrices for all images
    camera_matrices = {}
    for img_name, pose in camera_poses.items():
        if not pose.get('gps'):
            continue
        
        t = gps_to_local_utm(
            pose['gps']['latitude'],
            pose['gps']['longitude'],
            pose['gps']['altitude'],
            origin_lat, origin_lon
        )
        
        R = np.eye(3, dtype=np.float64)  # Assume nadir view
        Rt = np.hstack([R, t.reshape(3, 1)])
        P = K @ Rt
        
        camera_matrices[img_name] = (P, R, t)
    
    # Triangulate each track
    point_cloud = []
    successful_triangulations = 0
    failed_triangulations = 0
    
    for track in tracks:
        result = triangulate_track_robust(
            track, features, camera_poses, camera_matrices, K,
            max_reprojection_error, min_views
        )
        
        if result is not None:
            result['track_id'] = track['track_id']
            point_cloud.append(result)
            successful_triangulations += 1
        else:
            failed_triangulations += 1
    
    # Compute statistics
    if point_cloud:
        reproj_errors = [p['reprojection_error_mean'] for p in point_cloud]
        num_views = [p['num_views'] for p in point_cloud]
        
        stats = {
            'total_tracks': len(tracks),
            'successful_triangulations': successful_triangulations,
            'failed_triangulations': failed_triangulations,
            'success_rate': successful_triangulations / len(tracks) if len(tracks) > 0 else 0.0,
            'mean_reprojection_error': np.mean(reproj_errors),
            'max_reprojection_error': np.max(reproj_errors),
            'mean_num_views': np.mean(num_views),
            'min_num_views': np.min(num_views),
            'max_num_views': np.max(num_views)
        }
    else:
        stats = {
            'total_tracks': len(tracks),
            'successful_triangulations': 0,
            'failed_triangulations': len(tracks),
            'success_rate': 0.0
        }
    
    return point_cloud, stats
