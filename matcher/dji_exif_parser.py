"""
Parse DJI-specific EXIF metadata to extract camera orientation (pitch/roll/yaw).
Uses exifread and exiftool to extract gimbal angles from XMP and MakerNote.
"""

import subprocess
import json
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
import exifread


def extract_dji_orientation_exifread(image_path: str) -> Optional[Dict[str, float]]:
    """
    Extract camera orientation from DJI image using exifread.
    
    Returns:
        Dictionary with 'pitch', 'roll', 'yaw' in degrees, or None if not found
    """
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f, details=True)
        
        orientation = {}
        
        # Check MakerNote for pitch/roll/yaw (these are flight angles, not gimbal)
        if 'MakerNote Pitch' in tags:
            pitch_val = tags['MakerNote Pitch']
            if hasattr(pitch_val, 'values') and len(pitch_val.values) > 0:
                orientation['flight_pitch'] = float(pitch_val.values[0])
            else:
                try:
                    orientation['flight_pitch'] = float(str(pitch_val))
                except:
                    pass
        
        if 'MakerNote Roll' in tags:
            roll_val = tags['MakerNote Roll']
            if hasattr(roll_val, 'values') and len(roll_val.values) > 0:
                orientation['flight_roll'] = float(roll_val.values[0])
            else:
                try:
                    orientation['flight_roll'] = float(str(roll_val))
                except:
                    pass
        
        if 'MakerNote Yaw' in tags:
            yaw_val = tags['MakerNote Yaw']
            if hasattr(yaw_val, 'values') and len(yaw_val.values) > 0:
                orientation['flight_yaw'] = float(yaw_val.values[0])
            else:
                try:
                    orientation['flight_yaw'] = float(str(yaw_val))
                except:
                    pass
        
        return orientation if orientation else None
        
    except Exception as e:
        return None


def extract_dji_orientation_exiftool(image_path: str) -> Optional[Dict[str, float]]:
    """
    Extract camera orientation from DJI image using exiftool.
    Exiftool can parse XMP data which contains gimbal angles.
    
    Returns:
        Dictionary with 'gimbal_pitch', 'gimbal_roll', 'gimbal_yaw' in degrees
    """
    try:
        # Use exiftool to extract XMP and MakerNote data
        result = subprocess.run(
            ['exiftool', '-j', '-GimbalPitchDegree', '-GimbalRollDegree', 
             '-GimbalYawDegree', '-FlightPitchDegree', '-FlightRollDegree', 
             '-FlightYawDegree', str(image_path)],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            return None
        
        if not result.stdout.strip():
            return None
        
        data = json.loads(result.stdout)
        if not data or len(data) == 0:
            return None
        
        exif_data = data[0]
        orientation = {}
        
        # Extract gimbal angles (camera orientation) - check all possible tag formats
        for key in exif_data.keys():
            value = exif_data[key]
            
            # XMP tags
            if 'GimbalPitchDegree' in key and 'gimbal_pitch' not in orientation:
                try:
                    orientation['gimbal_pitch'] = float(value)
                except (ValueError, TypeError):
                    pass
            
            if 'GimbalRollDegree' in key and 'gimbal_roll' not in orientation:
                try:
                    orientation['gimbal_roll'] = float(value)
                except (ValueError, TypeError):
                    pass
            
            if 'GimbalYawDegree' in key and 'gimbal_yaw' not in orientation:
                try:
                    orientation['gimbal_yaw'] = float(value)
                except (ValueError, TypeError):
                    pass
            
            # Flight angles (aircraft orientation)
            if 'FlightPitchDegree' in key and 'flight_pitch' not in orientation:
                try:
                    orientation['flight_pitch'] = float(value)
                except (ValueError, TypeError):
                    pass
            
            if 'FlightRollDegree' in key and 'flight_roll' not in orientation:
                try:
                    orientation['flight_roll'] = float(value)
                except (ValueError, TypeError):
                    pass
            
            if 'FlightYawDegree' in key and 'flight_yaw' not in orientation:
                try:
                    orientation['flight_yaw'] = float(value)
                except (ValueError, TypeError):
                    pass
        
        return orientation if orientation else None
        
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, json.JSONDecodeError, KeyError, ValueError) as e:
        return None
    except Exception:
        return None


def extract_dji_orientation(image_path: str) -> Optional[Dict[str, float]]:
    """
    Extract camera orientation from DJI image.
    Tries exiftool first (for XMP gimbal data), then falls back to exifread.
    
    Returns:
        Dictionary with orientation angles in degrees:
        - 'gimbal_pitch', 'gimbal_roll', 'gimbal_yaw' (camera gimbal orientation)
        - 'flight_pitch', 'flight_roll', 'flight_yaw' (aircraft orientation)
    """
    # Try exiftool first (better XMP support)
    orientation = extract_dji_orientation_exiftool(image_path)
    
    if orientation and 'gimbal_pitch' in orientation:
        return orientation
    
    # Fall back to exifread (MakerNote only)
    orientation_exifread = extract_dji_orientation_exifread(image_path)
    
    if orientation_exifread:
        # Merge with exiftool results if available
        if orientation:
            orientation.update(orientation_exifread)
        else:
            orientation = orientation_exifread
    
    return orientation


def euler_to_rotation_matrix(pitch: float, roll: float, yaw: float, 
                             order: str = 'ZYX') -> np.ndarray:
    """
    Convert Euler angles to rotation matrix for DJI gimbal.
    
    DJI gimbal convention:
    - Pitch: rotation around X-axis (-90° = nadir, pointing straight down)
    - Roll: rotation around Z-axis (camera roll)
    - Yaw: rotation around Y-axis (camera heading/azimuth)
    
    Coordinate system:
    - World: X=East, Y=North, Z=Up
    - Camera: X=right, Y=down, Z=forward (into image, pointing down for nadir)
    
    For nadir (pitch=-90°): camera Z should point down (world -Z)
    
    Args:
        pitch: Gimbal pitch angle (degrees, -90° = nadir)
        roll: Gimbal roll angle (degrees)
        yaw: Gimbal yaw angle (degrees, camera heading)
    
    Returns:
        3x3 rotation matrix (world to camera)
    """
    # Convert to radians
    pitch_rad = np.radians(pitch)
    roll_rad = np.radians(roll)
    yaw_rad = np.radians(yaw)
    
    # Rotation matrices
    cos_p = np.cos(pitch_rad)
    sin_p = np.sin(pitch_rad)
    cos_r = np.cos(roll_rad)
    sin_r = np.sin(roll_rad)
    cos_y = np.cos(yaw_rad)
    sin_y = np.sin(yaw_rad)
    
    # For DJI gimbal, the pitch rotation is around Y-axis
    # Pitch = -90° means camera points straight down (camera Z = -world Z)
    R_pitch = np.array([
        [cos_p, 0, sin_p],
        [0, 1, 0],
        [-sin_p, 0, cos_p]
    ])
    
    # Yaw: rotation around Z-axis (camera heading/azimuth in horizontal plane)
    R_yaw = np.array([
        [cos_y, -sin_y, 0],
        [sin_y, cos_y, 0],
        [0, 0, 1]
    ])
    
    # Roll: rotation around X-axis (camera roll)
    R_roll = np.array([
        [1, 0, 0],
        [0, cos_r, -sin_r],
        [0, sin_r, cos_r]
    ])
    
    # Apply rotations in order: yaw (heading) -> pitch (gimbal) -> roll
    # This gives: R = R_roll @ R_pitch @ R_yaw
    R = R_roll @ R_pitch @ R_yaw
    
    return R


def get_camera_rotation_from_dji_orientation(orientation: Dict[str, float]) -> np.ndarray:
    """
    Get camera rotation matrix from DJI orientation data.
    
    Uses gimbal angles if available (preferred), otherwise uses flight angles.
    
    For nadir views (pitch ≈ -90°), uses a direct rotation matrix.
    For other angles, uses Euler angle conversion.
    
    Args:
        orientation: Dictionary with gimbal or flight angles
    
    Returns:
        3x3 rotation matrix (world to camera)
    """
    # Prefer gimbal angles (camera orientation)
    if 'gimbal_pitch' in orientation:
        pitch = orientation['gimbal_pitch']
        roll = orientation.get('gimbal_roll', 0.0)
        yaw = orientation.get('gimbal_yaw', 0.0)
    elif 'flight_pitch' in orientation:
        # Use flight angles if gimbal not available
        pitch = orientation['flight_pitch']
        roll = orientation.get('flight_roll', 0.0)
        yaw = orientation.get('flight_yaw', 0.0)
    else:
        # Default to nadir (camera pointing down)
        return np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ], dtype=np.float64)
    
    # For nadir views (pitch close to -90°), use direct rotation
    if abs(pitch + 90.0) < 5.0:  # Within 5° of nadir
        # Base nadir rotation: camera Z points down (world -Z)
        R_nadir = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ], dtype=np.float64)
        
        # Apply yaw rotation around Z-axis (camera heading)
        if abs(yaw) > 0.1:
            yaw_rad = np.radians(yaw)
            cos_y = np.cos(yaw_rad)
            sin_y = np.sin(yaw_rad)
            R_yaw = np.array([
                [cos_y, -sin_y, 0],
                [sin_y, cos_y, 0],
                [0, 0, 1]
            ])
            # Rotate camera X and Y by yaw
            R = R_nadir @ R_yaw.T
        else:
            R = R_nadir
        
        # Apply roll rotation around camera Z-axis
        if abs(roll) > 0.1:
            roll_rad = np.radians(roll)
            cos_r = np.cos(roll_rad)
            sin_r = np.sin(roll_rad)
            R_roll = np.array([
                [cos_r, -sin_r, 0],
                [sin_r, cos_r, 0],
                [0, 0, 1]
            ])
            R = R @ R_roll
        
        return R
    else:
        # For non-nadir views, use Euler angle conversion
        return euler_to_rotation_matrix(pitch, roll, yaw, order='ZYX')
