"""
Robust 3D reconstruction using aerial triangulation and bundle adjustment.
Handles nadir views and degenerate configurations.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import defaultdict

from .aerial_triangulation import (
    estimate_camera_intrinsics,
    gps_to_local_meters,
    initialize_camera_poses,
    triangulate_points
)
from .bundle_adjustment import bundle_adjust
from .dji_exif_parser import extract_dji_orientation, get_camera_rotation_from_dji_orientation
from .structure_from_motion import incremental_sfm


def estimate_camera_rotation_from_matches(
    K: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray
) -> Optional[np.ndarray]:
    """
    Estimate relative rotation between two cameras given their positions.
    Uses essential matrix estimation.
    """
    if len(pts1) < 8:
        return None
    
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
        return None
    
    # Recover rotation
    inliers_flat = inliers.ravel() == 1
    if inliers_flat.sum() < 8:
        return None
    
    _, R, _, _ = cv2.recoverPose(
        E,
        pts1_norm[inliers_flat],
        pts2_norm[inliers_flat],
        cameraMatrix=np.eye(3)
    )
    
    return R


def robust_reconstruct_with_bundle_adjustment(
    tracks: List[Dict],
    matches: List[Dict],
    features: Dict,
    camera_poses_gps: Dict[str, Dict],
    origin_lat: float,
    origin_lon: float,
    image_width: int,
    image_height: int,
    max_reprojection_error: float = 2.0,
    use_bundle_adjustment: bool = True,
    filter_outliers: bool = False
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], np.ndarray, List[Dict], Dict]:
    """
    Robust 3D reconstruction using aerial triangulation and bundle adjustment.
    
    Returns:
        camera_poses: Dictionary mapping image_name to (R, t)
        points_3d: Array of 3D points (N, 3)
        point_cloud: List of point dictionaries with statistics
        stats: Statistics dictionary
    """
    K = estimate_camera_intrinsics(image_width, image_height)
    
    # Step 1: Initialize camera poses from GPS and EXIF/SfM
    print("  Step 1: Initializing camera poses from GPS...")
    camera_poses = {}
    exif_orientations = {}
    exif_success = 0
    
    # Try to extract orientations from EXIF
    print("  Step 1a: Extracting camera orientations from EXIF metadata...")
    for img_name, pose_gps in camera_poses_gps.items():
        if not pose_gps.get('gps'):
            continue
        
        t_world = gps_to_local_meters(
            pose_gps['gps']['latitude'],
            pose_gps['gps']['longitude'],
            pose_gps['gps']['altitude'],
            origin_lat, origin_lon
        )
        
        # Try to get orientation from EXIF (check if already extracted)
        dji_orientation = pose_gps.get('dji_orientation')
        if not dji_orientation:
            # Try to extract it now
            image_path = pose_gps.get('image_path')
            if image_path:
                dji_orientation = extract_dji_orientation(image_path)
        
        if dji_orientation:
            # Use gimbal angles (camera orientation) if available
            R = get_camera_rotation_from_dji_orientation(dji_orientation)
            camera_poses[img_name] = (R, t_world)
            exif_orientations[img_name] = dji_orientation
            exif_success += 1
        else:
            # Start with identity rotation (will be estimated via SfM)
            R = np.eye(3, dtype=np.float64)
            camera_poses[img_name] = (R, t_world)
    
    print(f"    Initialized {len(camera_poses)} camera poses")
    print(f"    Extracted {exif_success} orientations from EXIF")
    
    # Step 1b: Use Structure-from-Motion to estimate remaining rotations
    if exif_success < len(camera_poses):
        print("  Step 1b: Estimating camera rotations using Structure-from-Motion...")
        sfm_poses = incremental_sfm(
            matches, features, camera_poses_gps, origin_lat, origin_lon, K, image_width, image_height
        )
        
        # Update poses with SfM estimates (only for cameras without EXIF orientation)
        sfm_updates = 0
        for img_name in camera_poses.keys():
            if img_name not in exif_orientations and img_name in sfm_poses:
                R_sfm, t_sfm = sfm_poses[img_name]
                # Only update rotation if SfM found a non-identity rotation
                if not np.allclose(R_sfm, np.eye(3)):
                    _, t_gps = camera_poses[img_name]
                    camera_poses[img_name] = (R_sfm, t_gps)  # Keep GPS position
                    sfm_updates += 1
        
        print(f"    Updated {sfm_updates} camera rotations using SfM")
    
    # Step 2: Graph-based rotation propagation
    print("  Step 2: Propagating camera rotations through match graph...")
    
    # Build graph of image connections
    from collections import deque
    image_graph = defaultdict(list)
    
    for match_dict in matches:
        img0_name = Path(match_dict['image0']).name.replace('quarter_', '')
        img1_name = Path(match_dict['image1']).name.replace('quarter_', '')
        
        if img0_name not in camera_poses or img1_name not in camera_poses:
            continue
        
        # Count matches
        match_indices = match_dict.get('matches', [])
        if isinstance(match_indices, list):
            num_matches = len(match_indices)
        else:
            num_matches = match_dict.get('num_matches', 0)
        
        if num_matches >= 20:  # Need enough matches for reliable estimation
            image_graph[img0_name].append((img1_name, match_dict))
            image_graph[img1_name].append((img0_name, match_dict))
    
    # BFS propagation from cameras with known rotations
    queue = deque()
    visited = set()
    
    # Start with cameras that have non-identity rotations
    for img_name, (R, t) in camera_poses.items():
        if not np.allclose(R, np.eye(3)):
            queue.append((img_name, R))
            visited.add(img_name)
    
    rotations_estimated = 0
    max_iterations = 500
    iteration = 0
    
    while queue and iteration < max_iterations:
        iteration += 1
        current_img, R_current = queue.popleft()
        
        # Process neighbors
        for neighbor_img, match_dict in image_graph.get(current_img, []):
            if neighbor_img in visited:
                continue
            
            R_neighbor, t_neighbor = camera_poses[neighbor_img]
            
            # If neighbor has identity rotation, estimate it
            if np.allclose(R_neighbor, np.eye(3)):
                # Get features and matches
                img0_path = match_dict['image0']
                img1_path = match_dict['image1']
                
                img0_name_clean = Path(img0_path).name.replace('quarter_', '')
                img1_name_clean = Path(img1_path).name.replace('quarter_', '')
                
                if img0_path not in features or img1_path not in features:
                    continue
                
                feat0 = features[img0_path]
                feat1 = features[img1_path]
                
                keypoints0 = feat0['keypoints'] * 4.0
                keypoints1 = feat1['keypoints'] * 4.0
                
                match_indices = np.array(match_dict['matches'])
                if len(match_indices) < 20:
                    continue
                
                # Determine which is current and which is neighbor
                if img0_name_clean == current_img:
                    pts_current = keypoints0[match_indices[:, 0]]
                    pts_neighbor = keypoints1[match_indices[:, 1]]
                else:
                    pts_current = keypoints1[match_indices[:, 1]]
                    pts_neighbor = keypoints0[match_indices[:, 0]]
                
                # Estimate relative rotation
                R_rel = estimate_camera_rotation_from_matches(K, pts_current, pts_neighbor, 
                                                               camera_poses[current_img][1], t_neighbor)
                
                if R_rel is not None:
                    # R_neighbor = R_current @ R_rel
                    R_neighbor_new = R_current @ R_rel
                    camera_poses[neighbor_img] = (R_neighbor_new, t_neighbor)
                    queue.append((neighbor_img, R_neighbor_new))
                    visited.add(neighbor_img)
                    rotations_estimated += 1
    
    print(f"    Propagated rotations to {rotations_estimated} cameras via graph")
    
    # Step 3: Triangulate 3D points from tracks
    print("  Step 3: Triangulating 3D points from feature tracks...")
    
    # Build projection matrices
    camera_matrices = {}
    for img_name, (R, t) in camera_poses.items():
        Rt = np.hstack([R, t.reshape(3, 1)])
        P = K @ Rt
        camera_matrices[img_name] = (P, R, t)
    
    # Triangulate points from tracks
    points_3d_list = []
    observations = []  # For bundle adjustment
    point_track_map = {}  # Map point index to track info
    
    point_idx = 0
    
    for track in tracks:
        track_features = track['features']
        
        # Collect views for this track
        views = []
        for img_name, feat_idx in track_features:
            img_name_clean = img_name.replace('quarter_', '')
            
            if img_name_clean not in camera_matrices:
                continue
            
            # Find feature in features dict
            feat_dict = None
            for key in features.keys():
                if Path(key).name == img_name or Path(key).name == img_name_clean:
                    feat_dict = features[key]
                    break
            
            if feat_dict is None:
                continue
            
            keypoints = feat_dict['keypoints'] * 4.0  # Scale to full resolution
            if feat_idx >= len(keypoints):
                continue
            
            pt = keypoints[feat_idx]
            P, R, t = camera_matrices[img_name_clean]
            
            views.append({
                'image_name': img_name_clean,
                'point': pt,
                'P': P,
                'R': R,
                't': t
            })
        
        if len(views) < 2:
            continue
        
        # Triangulate from first two views
        P1 = views[0]['P']
        P2 = views[1]['P']
        pt1 = views[0]['point']
        pt2 = views[1]['point']
        
        X = triangulate_points(P1, P2, pt1.reshape(1, 2), pt2.reshape(1, 2))
        X = X[0]  # Get single point
        
        # Only filter out NaN/Inf points, keep everything else
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            continue
        
        # Don't filter by distance - keep all points to see structure
        
        # Compute reprojection errors for all views
        reproj_errors = []
        for view in views:
            pt_proj = project_point(K, view['R'], view['t'], X)
            error = float(np.linalg.norm(view['point'] - pt_proj))
            reproj_errors.append(error)
        
        # Only filter if filter_outliers is True
        if filter_outliers:
            # Check if any reprojection error exceeds threshold
            if any(e > max_reprojection_error for e in reproj_errors):
                continue
            
            # Filter by distance
            point_distance = np.linalg.norm(X)
            if point_distance > 1000.0:
                continue
        
        # Add point
        points_3d_list.append(X)
        point_track_map[point_idx] = {
            'track_id': track['track_id'],
            'views': [(v['image_name'], v['point']) for v in views]
        }
        
        # Add observations for bundle adjustment
        for view in views:
            observations.append({
                'image_name': view['image_name'],
                'point_idx': point_idx,
                'point_2d': view['point']
            })
        
        point_idx += 1
    
    if len(points_3d_list) == 0:
        return camera_poses, np.array([]).reshape(0, 3), [], {'status': 'no_points'}
    
    points_3d = np.array(points_3d_list)
    print(f"    Triangulated {len(points_3d)} 3D points from {len(tracks)} tracks")
    
    # Step 4: Bundle adjustment
    if use_bundle_adjustment and len(observations) > 0:
        print("  Step 4: Running bundle adjustment...")
        refined_poses, refined_points, ba_stats = bundle_adjust(
            camera_poses,
            points_3d,
            observations,
            K,
            fix_cameras=True,  # Keep GPS positions fixed
            max_iterations=50
        )
        
        camera_poses = refined_poses
        points_3d = refined_points
        
        print(f"    Bundle adjustment: RMSE = {ba_stats.get('final_cost', 0):.2f} pixels")
    else:
        ba_stats = {'status': 'skipped'}
    
    # Build point cloud output
    point_cloud = []
    for i, X in enumerate(points_3d):
        track_info = point_track_map.get(i, {})
        
        # Compute final reprojection errors
        reproj_errors = []
        for img_name, pt_2d in track_info.get('views', []):
            if img_name in camera_poses:
                R, t = camera_poses[img_name]
                error = np.linalg.norm(pt_2d - project_point(K, R, t, X))
                reproj_errors.append(error)
        
        if len(reproj_errors) > 0:
            point_cloud.append({
                'point_3d': X.tolist(),
                'track_id': track_info.get('track_id', -1),
                'num_views': len(reproj_errors),
                'reprojection_error_mean': float(np.mean(reproj_errors)),
                'reprojection_error_max': float(np.max(reproj_errors)),
                'distance_from_origin': float(np.linalg.norm(X))
            })
    
    stats = {
        'num_cameras': len(camera_poses),
        'num_points': len(points_3d),
        'num_observations': len(observations),
        'bundle_adjustment': ba_stats
    }
    
    return camera_poses, points_3d, point_cloud, stats


def project_point(K: np.ndarray, R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Project 3D point to image coordinates."""
    X = np.array(X).reshape(3, 1) if X.ndim == 1 else X.reshape(3, 1)
    X_cam = R @ X + t.reshape(3, 1)
    x_hom = K @ X_cam
    x_hom_val = float(x_hom[2])
    if abs(x_hom_val) > 1e-10:
        x_hom = x_hom / x_hom_val
    return x_hom[:2].ravel()
