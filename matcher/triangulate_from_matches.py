"""
Triangulate 3D points directly from matches (not tracks).
This is simpler and more direct than using tracks.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2


def triangulate_from_matches(
    matches: List[Dict],
    features: Dict,
    camera_poses: Dict[str, Tuple[np.ndarray, np.ndarray]],  # (R, t) tuples
    K: np.ndarray,
    max_reprojection_error: float = 2.0
) -> Tuple[List[Dict], Dict]:
    """
    Triangulate 3D points directly from matches.
    
    Args:
        matches: List of match dictionaries with 'image0', 'image1', 'matches'
        features: Dictionary mapping image_path to feature dict
        camera_poses: Dictionary mapping image_name to (R, t) tuple
        K: Camera intrinsics matrix
        max_reprojection_error: Maximum allowed reprojection error (pixels)
    
    Returns:
        Tuple of (points_3d_list, stats_dict)
    """
    points_3d_list = []
    stats = {
        'total_matches_processed': 0,
        'successful_triangulations': 0,
        'failed_triangulations': 0,
        'reprojection_errors': []
    }
    
    # Build projection matrices for all cameras
    camera_matrices = {}
    for img_name, (R, t) in camera_poses.items():
        Rt = np.hstack([R, t.reshape(3, 1)])
        P = K @ Rt
        camera_matrices[img_name] = P
    
    # Process each match pair
    for match_dict in matches:
        img0_path = match_dict['image0']
        img1_path = match_dict['image1']
        
        # Get image names (handle quarter_ prefix)
        img0_name = Path(img0_path).name.replace('quarter_', '')
        img1_name = Path(img1_path).name.replace('quarter_', '')
        
        # Find matching camera poses (try multiple name variations)
        P0 = None
        P1 = None
        img0_key = None
        img1_key = None
        
        for key in camera_matrices.keys():
            key_base = Path(key).name if isinstance(key, str) else str(key)
            # Try exact match, base name match, and with/without quarter_ prefix
            if (key_base == img0_name or key == img0_name or 
                key_base == Path(img0_path).name or key == img0_path or
                key_base.replace('quarter_', '') == img0_name):
                P0 = camera_matrices[key]
                img0_key = key
            if (key_base == img1_name or key == img1_name or 
                key_base == Path(img1_path).name or key == img1_path or
                key_base.replace('quarter_', '') == img1_name):
                P1 = camera_matrices[key]
                img1_key = key
        
        if P0 is None or P1 is None:
            continue
        
        # Get features
        feat0 = None
        feat1 = None
        
        for key in features.keys():
            key_base = Path(key).name
            if key_base == img0_name or key == img0_path or key_base == Path(img0_path).name:
                feat0 = features[key]
            if key_base == img1_name or key == img1_path or key_base == Path(img1_path).name:
                feat1 = features[key]
        
        if feat0 is None or feat1 is None:
            continue
        
        keypoints0 = feat0['keypoints']
        keypoints1 = feat1['keypoints']
        
        # Get match indices
        match_indices = np.array(match_dict['matches'])
        stats['total_matches_processed'] += len(match_indices)
        
        # Triangulate each matched point pair
        for match_idx in match_indices:
            idx0, idx1 = match_idx[0], match_idx[1]
            
            if idx0 >= len(keypoints0) or idx1 >= len(keypoints1):
                continue
            
            pt0 = keypoints0[idx0]
            pt1 = keypoints1[idx1]
            
            # Triangulate using OpenCV
            points_4d = cv2.triangulatePoints(
                P0, P1,
                pt0.reshape(1, 1, 2).astype(np.float32),
                pt1.reshape(1, 1, 2).astype(np.float32)
            )
            
            # Convert from homogeneous to 3D
            X = (points_4d[:3] / points_4d[3]).reshape(3)
            
            # Check validity
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                stats['failed_triangulations'] += 1
                continue
            
            # Compute reprojection errors
            # Project back to both images
            R0, t0 = camera_poses[img0_key]
            R1, t1 = camera_poses[img1_key]
            pt0_proj = project_point(K, R0, t0, X)
            pt1_proj = project_point(K, R1, t1, X)
            
            error0 = np.linalg.norm(pt0 - pt0_proj)
            error1 = np.linalg.norm(pt1 - pt1_proj)
            max_error = max(error0, error1)
            
            if max_error > max_reprojection_error:
                stats['failed_triangulations'] += 1
                continue
            
            # Valid point
            points_3d_list.append({
                'point_3d': X.tolist(),
                'reprojection_error_mean': (error0 + error1) / 2.0,
                'reprojection_error_max': max_error,
                'views': [
                    {'image': img0_key, 'feature_idx': int(idx0), 'error': float(error0)},
                    {'image': img1_key, 'feature_idx': int(idx1), 'error': float(error1)}
                ]
            })
            
            stats['successful_triangulations'] += 1
            stats['reprojection_errors'].append((error0 + error1) / 2.0)
    
    if stats['reprojection_errors']:
        stats['mean_reprojection_error'] = np.mean(stats['reprojection_errors'])
        stats['max_reprojection_error'] = np.max(stats['reprojection_errors'])
    else:
        stats['mean_reprojection_error'] = 0.0
        stats['max_reprojection_error'] = 0.0
    
    return points_3d_list, stats


def project_point(K: np.ndarray, R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Project 3D point to image coordinates."""
    X_hom = np.append(X, 1.0)
    P = K @ np.hstack([R, t.reshape(3, 1)])
    x_hom = P @ X_hom
    return x_hom[:2] / x_hom[2]
