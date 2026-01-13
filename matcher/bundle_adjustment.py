"""
Bundle adjustment for refining camera poses and 3D points.
Minimizes reprojection error across all views.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import least_squares
from collections import defaultdict


def rotation_matrix_to_angle_axis(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to angle-axis representation."""
    import cv2
    rvec, _ = cv2.Rodrigues(R)
    return rvec.ravel()


def angle_axis_to_rotation_matrix(rvec: np.ndarray) -> np.ndarray:
    """Convert angle-axis to rotation matrix."""
    import cv2
    R, _ = cv2.Rodrigues(rvec)
    return R


def project_point(K: np.ndarray, R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Project 3D point to image coordinates."""
    # World to camera
    X_cam = R @ X + t.reshape(3, 1)
    
    # Project
    x_hom = K @ X_cam
    if abs(x_hom[2]) > 1e-10:
        x_hom = x_hom / x_hom[2]
    
    return x_hom[:2]


def compute_reprojection_error(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    X: np.ndarray,
    x_obs: np.ndarray
) -> float:
    """Compute reprojection error for a single point."""
    x_proj = project_point(K, R, t, X)
    error = np.linalg.norm(x_obs - x_proj)
    return error


def bundle_adjustment_residuals(
    params: np.ndarray,
    n_cameras: int,
    n_points: int,
    camera_indices: np.ndarray,
    point_indices: np.ndarray,
    points_2d: np.ndarray,
    K: np.ndarray,
    fixed_cameras: Optional[List[int]] = None
) -> np.ndarray:
    """
    Compute residuals for bundle adjustment.
    
    Args:
        params: Flattened parameters [rvecs, tvecs, points_3d]
        n_cameras: Number of cameras
        n_points: Number of 3D points
        camera_indices: Which camera each observation belongs to
        point_indices: Which 3D point each observation corresponds to
        points_2d: Observed 2D points (N, 2)
        K: Camera intrinsics
        fixed_cameras: List of camera indices to keep fixed (use GPS positions)
    
    Returns:
        Residuals (2*N,)
    """
    # Unpack parameters
    param_idx = 0
    
    # Camera parameters: 6 per camera (3 for rotation, 3 for translation)
    camera_params = params[:n_cameras * 6].reshape(n_cameras, 6)
    
    # 3D points: 3 per point
    points_3d = params[n_cameras * 6:].reshape(n_points, 3)
    
    residuals = []
    
    for i in range(len(points_2d)):
        cam_idx = camera_indices[i]
        pt_idx = point_indices[i]
        
        # Get camera pose
        rvec = camera_params[cam_idx, :3]
        tvec = camera_params[cam_idx, 3:6]
        
        R = angle_axis_to_rotation_matrix(rvec)
        
        # Get 3D point
        X = points_3d[pt_idx]
        
        # Project
        x_proj = project_point(K, R, tvec, X)
        
        # Compute residual
        x_obs = points_2d[i]
        residual = (x_obs - x_proj).ravel()
        
        residuals.extend(residual)
    
    return np.array(residuals)


def bundle_adjust(
    camera_poses: Dict[str, Tuple[np.ndarray, np.ndarray]],
    points_3d: np.ndarray,
    observations: List[Dict],
    K: np.ndarray,
    fix_cameras: bool = True,
    max_iterations: int = 100
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], np.ndarray, Dict]:
    """
    Perform bundle adjustment to refine camera poses and 3D points.
    
    Args:
        camera_poses: Dictionary mapping image_name to (R, t)
        points_3d: 3D points (N, 3)
        observations: List of dicts with keys:
            - 'image_name': str
            - 'point_idx': int
            - 'point_2d': np.ndarray (2,)
        K: Camera intrinsics
        fix_cameras: If True, keep camera positions fixed (only optimize rotation and points)
        max_iterations: Maximum iterations for optimization
    
    Returns:
        refined_poses: Refined camera poses
        refined_points: Refined 3D points (N, 3)
        stats: Statistics dictionary
    """
    if len(points_3d) == 0 or len(observations) == 0:
        return camera_poses, points_3d, {'status': 'no_data'}
    
    # Build index maps
    image_to_idx = {img_name: idx for idx, img_name in enumerate(camera_poses.keys())}
    
    # Prepare data for optimization
    camera_indices = []
    point_indices = []
    points_2d_list = []
    
    for obs in observations:
        img_name = obs['image_name']
        if img_name not in image_to_idx:
            continue
        
        camera_indices.append(image_to_idx[img_name])
        point_indices.append(obs['point_idx'])
        points_2d_list.append(obs['point_2d'])
    
    if len(camera_indices) < 6:
        return camera_poses, points_3d, {'status': 'insufficient_observations'}
    
    camera_indices = np.array(camera_indices)
    point_indices = np.array(point_indices)
    points_2d = np.array(points_2d_list)
    
    n_cameras = len(camera_poses)
    n_points = len(points_3d)
    
    # Initialize parameters
    params = []
    
    # Camera parameters
    for img_name, (R, t) in camera_poses.items():
        rvec = rotation_matrix_to_angle_axis(R)
        if fix_cameras:
            # Keep translation fixed, only optimize rotation
            params.extend(rvec)
            params.extend(t)  # Will be marked as fixed in bounds
        else:
            params.extend(rvec)
            params.extend(t)
    
    # 3D points
    params.extend(points_3d.ravel())
    
    params = np.array(params, dtype=np.float64)
    
    # Set bounds (fix camera translations if requested)
    if fix_cameras:
        # For fixed cameras, we'll use a mask approach instead of bounds
        # Set very tight bounds for translations (effectively fixed)
        bounds_lower = []
        bounds_upper = []
        
        for i in range(n_cameras):
            # Rotation: -pi to pi for each axis
            bounds_lower.extend([-np.pi, -np.pi, -np.pi])
            bounds_upper.extend([np.pi, np.pi, np.pi])
            
            # Translation: very tight bounds (effectively fixed)
            _, t = list(camera_poses.values())[i]
            bounds_lower.extend([t[0] - 0.01, t[1] - 0.01, t[2] - 0.01])
            bounds_upper.extend([t[0] + 0.01, t[1] + 0.01, t[2] + 0.01])
        
        # Points: allow to vary freely
        bounds_lower.extend([-np.inf] * (n_points * 3))
        bounds_upper.extend([np.inf] * (n_points * 3))
        
        bounds = (np.array(bounds_lower), np.array(bounds_upper))
    else:
        bounds = None
    
    # Compute initial reprojection errors
    initial_residuals = bundle_adjustment_residuals(
        params, n_cameras, n_points, camera_indices, point_indices, points_2d, K
    )
    initial_rmse = np.sqrt(np.mean(initial_residuals**2))
    initial_errors = initial_residuals.reshape(-1, 2)
    initial_errors_per_obs = np.linalg.norm(initial_errors, axis=1)
    
    print(f"    Initial reprojection error:")
    print(f"      RMSE: {initial_rmse:.4f} pixels")
    print(f"      Mean: {np.mean(initial_errors_per_obs):.4f} pixels")
    print(f"      Median: {np.median(initial_errors_per_obs):.4f} pixels")
    print(f"      Max: {np.max(initial_errors_per_obs):.4f} pixels")
    print(f"      Min: {np.min(initial_errors_per_obs):.4f} pixels")
    print(f"      Std: {np.std(initial_errors_per_obs):.4f} pixels")
    
    # Store initial poses for comparison
    initial_poses = {}
    for img_name, (R, t) in camera_poses.items():
        initial_poses[img_name] = (R.copy(), t.copy())
    
    # Optimize
    try:
        # For now, skip bundle adjustment if we have too few observations
        # The optimization can be unstable with small datasets
        if len(observations) < 20:
            return camera_poses, points_3d, {'status': 'skipped', 'reason': 'insufficient_observations'}
        
        print(f"    Running optimization (max {max_iterations} iterations)...")
        print(f"    Iteration progress:")
        
        result = least_squares(
            bundle_adjustment_residuals,
            params,
            args=(n_cameras, n_points, camera_indices, point_indices, points_2d, K),
            bounds=bounds if fix_cameras and bounds is not None else None,
            max_nfev=max_iterations * len(params),
            verbose=2  # Show detailed iteration output
        )
        
        # Extract refined parameters
        refined_params = result.x
        
        # Extract camera poses
        refined_poses = {}
        param_idx = 0
        for img_name in camera_poses.keys():
            rvec = refined_params[param_idx:param_idx+3]
            param_idx += 3
            t = refined_params[param_idx:param_idx+3]
            param_idx += 3
            
            R = angle_axis_to_rotation_matrix(rvec)
            refined_poses[img_name] = (R, t)
        
        # Extract 3D points
        refined_points = refined_params[param_idx:].reshape(n_points, 3)
        
        # Compute final reprojection errors
        final_residuals = result.fun
        final_rmse = np.sqrt(np.mean(final_residuals**2))
        final_errors = final_residuals.reshape(-1, 2)
        final_errors_per_obs = np.linalg.norm(final_errors, axis=1)
        
        print(f"\n    Final reprojection error:")
        print(f"      RMSE: {final_rmse:.4f} pixels")
        print(f"      Mean: {np.mean(final_errors_per_obs):.4f} pixels")
        print(f"      Median: {np.median(final_errors_per_obs):.4f} pixels")
        print(f"      Max: {np.max(final_errors_per_obs):.4f} pixels")
        print(f"      Min: {np.min(final_errors_per_obs):.4f} pixels")
        print(f"      Std: {np.std(final_errors_per_obs):.4f} pixels")
        print(f"    Improvement: {initial_rmse - final_rmse:.4f} pixels ({100 * (initial_rmse - final_rmse) / initial_rmse:.2f}% reduction)")
        
        # Compare poses before and after
        print(f"\n    Camera pose changes:")
        rotation_changes = []
        translation_changes = []
        
        for img_name in camera_poses.keys():
            R_initial, t_initial = initial_poses[img_name]
            R_final, t_final = refined_poses[img_name]
            
            # Compute rotation change (angle between rotation matrices)
            R_diff = R_final @ R_initial.T
            trace = np.trace(R_diff)
            angle_change = np.arccos(np.clip((trace - 1) / 2, -1, 1))
            rotation_changes.append(np.degrees(angle_change))
            
            # Compute translation change
            t_change = np.linalg.norm(t_final - t_initial)
            translation_changes.append(t_change)
        
        if rotation_changes:
            print(f"      Rotation changes:")
            print(f"        Mean: {np.mean(rotation_changes):.4f} degrees")
            print(f"        Max: {np.max(rotation_changes):.4f} degrees")
            print(f"        Median: {np.median(rotation_changes):.4f} degrees")
        
        if translation_changes:
            print(f"      Translation changes:")
            print(f"        Mean: {np.mean(translation_changes):.4f} meters")
            print(f"        Max: {np.max(translation_changes):.4f} meters")
            print(f"        Median: {np.median(translation_changes):.4f} meters")
        
        # Compute point changes
        point_changes = np.linalg.norm(refined_points - points_3d, axis=1)
        print(f"    Point 3D changes:")
        print(f"      Mean: {np.mean(point_changes):.4f} meters")
        print(f"      Max: {np.max(point_changes):.4f} meters")
        print(f"      Median: {np.median(point_changes):.4f} meters")
        
        stats = {
            'status': 'success',
            'initial_rmse': initial_rmse,
            'initial_mean_error': float(np.mean(initial_errors_per_obs)),
            'initial_median_error': float(np.median(initial_errors_per_obs)),
            'initial_max_error': float(np.max(initial_errors_per_obs)),
            'initial_std_error': float(np.std(initial_errors_per_obs)),
            'final_rmse': final_rmse,
            'final_mean_error': float(np.mean(final_errors_per_obs)),
            'final_median_error': float(np.median(final_errors_per_obs)),
            'final_max_error': float(np.max(final_errors_per_obs)),
            'final_std_error': float(np.std(final_errors_per_obs)),
            'improvement_rmse': float(initial_rmse - final_rmse),
            'improvement_percent': float(100 * (initial_rmse - final_rmse) / initial_rmse),
            'iterations': result.nfev,
            'function_evaluations': result.nfev,
            'n_observations': len(observations),
            'n_cameras': n_cameras,
            'n_points': n_points,
            'rotation_changes_deg': {
                'mean': float(np.mean(rotation_changes)) if rotation_changes else 0.0,
                'max': float(np.max(rotation_changes)) if rotation_changes else 0.0,
                'median': float(np.median(rotation_changes)) if rotation_changes else 0.0
            },
            'translation_changes_m': {
                'mean': float(np.mean(translation_changes)) if translation_changes else 0.0,
                'max': float(np.max(translation_changes)) if translation_changes else 0.0,
                'median': float(np.median(translation_changes)) if translation_changes else 0.0
            },
            'point_changes_m': {
                'mean': float(np.mean(point_changes)),
                'max': float(np.max(point_changes)),
                'median': float(np.median(point_changes))
            },
            'optimization_status': result.status,
            'optimization_message': result.message
        }
        
        return refined_poses, refined_points, stats
        
    except Exception as e:
        print(f"Bundle adjustment failed: {e}")
        return camera_poses, points_3d, {'status': 'failed', 'error': str(e)}
