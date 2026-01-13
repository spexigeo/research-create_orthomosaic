"""
Debug export functions for footprints and camera poses.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List
from shapely.geometry import Polygon
import numpy as np

from .utils import compute_footprint_polygon


def export_footprints_to_csv(
    camera_poses_file: str,
    output_csv: str,
    focal_length_mm: float = 8.8,
    sensor_width_mm: float = 13.2,
    sensor_height_mm: float = 9.9
):
    """
    Export footprints to CSV with corner coordinates and shape type.
    
    Args:
        camera_poses_file: Path to camera_poses.json
        output_csv: Path to output CSV file
        focal_length_mm: Focal length in millimeters
        sensor_width_mm: Sensor width in millimeters
        sensor_height_mm: Sensor height in millimeters
    """
    # Load camera poses
    with open(camera_poses_file, 'r') as f:
        poses = json.load(f)
    
    # Find origin from first pose with GPS
    origin_lat = None
    origin_lon = None
    for pose in poses:
        if pose.get('gps'):
            origin_lat = pose['gps']['latitude']
            origin_lon = pose['gps']['longitude']
            break
    
    if origin_lat is None:
        raise ValueError("No GPS data found in camera poses")
    
    # Compute footprints
    rows = []
    
    for pose in poses:
        if not pose.get('gps'):
            continue
        
        image_name = pose.get('image_name', 'unknown')
        
        # Get pitch to determine shape type
        pitch = -90.0  # Default to nadir
        if pose.get('dji_orientation'):
            pitch = pose['dji_orientation'].get('gimbal_pitch', -90.0)
        
        # Determine shape type
        if abs(pitch + 90.0) < 1.0:  # Within 1° of nadir
            shape_type = 'rectangular'
        else:
            shape_type = 'trapezoidal'
        
        # Use dji_orientation if available, otherwise use default nadir orientation
        if not pose.get('dji_orientation'):
            pose['dji_orientation'] = {'gimbal_pitch': -90.0, 'gimbal_roll': 0.0, 'gimbal_yaw': 0.0}
        
        # Compute footprint
        footprint_poly, _ = compute_footprint_polygon(
            gps=pose['gps'],
            dji_orientation=pose['dji_orientation'],
            focal_length_mm=focal_length_mm,
            sensor_width_mm=sensor_width_mm,
            sensor_height_mm=sensor_height_mm,
            origin_lat=origin_lat,
            origin_lon=origin_lon
        )
        
        if footprint_poly is None:
            continue
        
        # Extract corner coordinates
        if isinstance(footprint_poly, Polygon):
            corners = list(footprint_poly.exterior.coords[:-1])  # Exclude closing point
        else:
            corners = list(footprint_poly) if hasattr(footprint_poly, '__iter__') else []
        
        # Ensure we have 4 corners (complete if needed)
        if len(corners) == 3:
            # Complete to 4 corners
            p1, p2, p3 = corners
            v13 = np.array(p3) - np.array(p1)
            p4 = np.array(p2) + v13
            corners.append(p4.tolist())
        elif len(corners) < 3:
            continue
        elif len(corners) > 4:
            corners = corners[:4]
        
        # Extract corner coordinates
        corner1_x, corner1_y = corners[0] if len(corners) > 0 else (None, None)
        corner2_x, corner2_y = corners[1] if len(corners) > 1 else (None, None)
        corner3_x, corner3_y = corners[2] if len(corners) > 2 else (None, None)
        corner4_x, corner4_y = corners[3] if len(corners) > 3 else (None, None)
        
        # Calculate footprint area in square meters
        if isinstance(footprint_poly, Polygon):
            area_m2 = footprint_poly.area
        else:
            # If not a Polygon, create one from corners to calculate area
            try:
                poly_from_corners = Polygon(corners)
                area_m2 = poly_from_corners.area
            except:
                area_m2 = None
        
        # Get GPS coordinates
        gps_lat = pose['gps']['latitude']
        gps_lon = pose['gps']['longitude']
        gps_alt = pose['gps']['altitude']
        
        # Get orientation
        roll = pose['dji_orientation'].get('gimbal_roll', 0.0)
        pitch_actual = pose['dji_orientation'].get('gimbal_pitch', -90.0)
        yaw = pose['dji_orientation'].get('gimbal_yaw', 0.0)
        if abs(yaw) < 0.1:
            yaw = pose['dji_orientation'].get('flight_yaw', 0.0)
        
        rows.append({
            'image_name': image_name,
            'shape_type': shape_type,
            'pitch_degrees': pitch_actual,
            'roll_degrees': roll,
            'yaw_degrees': yaw,
            'gps_latitude': gps_lat,
            'gps_longitude': gps_lon,
            'gps_altitude': gps_alt,
            'corner1_x': corner1_x,
            'corner1_y': corner1_y,
            'corner2_x': corner2_x,
            'corner2_y': corner2_y,
            'corner3_x': corner3_x,
            'corner3_y': corner3_y,
            'corner4_x': corner4_x,
            'corner4_y': corner4_y,
            'num_corners': len(corners),
            'area_m2': area_m2
        })
    
    # Write to CSV
    if rows:
        fieldnames = [
            'image_name', 'shape_type', 'pitch_degrees', 'roll_degrees', 'yaw_degrees',
            'gps_latitude', 'gps_longitude', 'gps_altitude',
            'corner1_x', 'corner1_y',
            'corner2_x', 'corner2_y',
            'corner3_x', 'corner3_y',
            'corner4_x', 'corner4_y',
            'num_corners', 'area_m2'
        ]
        
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"Exported {len(rows)} footprints to {output_csv}")
    else:
        print(f"Warning: No footprints to export")


def export_camera_poses_to_csv(
    camera_poses_file: str,
    output_csv: str
):
    """
    Export camera poses to CSV for debugging.
    
    Args:
        camera_poses_file: Path to camera_poses.json
        output_csv: Path to output CSV file
    """
    # Load camera poses
    with open(camera_poses_file, 'r') as f:
        poses = json.load(f)
    
    # Find origin from first pose with GPS
    origin_lat = None
    origin_lon = None
    for pose in poses:
        if pose.get('gps'):
            origin_lat = pose['gps']['latitude']
            origin_lon = pose['gps']['longitude']
            break
    
    rows = []
    
    for pose in poses:
        image_name = pose.get('image_name', 'unknown')
        
        # GPS coordinates
        if pose.get('gps'):
            gps_lat = pose['gps']['latitude']
            gps_lon = pose['gps']['longitude']
            gps_alt = pose['gps']['altitude']
            
            # Convert to local meters
            if origin_lat is not None and origin_lon is not None:
                lat_diff = gps_lat - origin_lat
                lon_diff = gps_lon - origin_lon
                local_x = lon_diff * 111000.0 * np.cos(np.radians(origin_lat))
                local_y = lat_diff * 111000.0
                local_z = gps_alt
            else:
                local_x = None
                local_y = None
                local_z = None
        else:
            gps_lat = None
            gps_lon = None
            gps_alt = None
            local_x = None
            local_y = None
            local_z = None
        
        # Orientation
        if pose.get('dji_orientation'):
            roll = pose['dji_orientation'].get('gimbal_roll', 0.0)
            pitch = pose['dji_orientation'].get('gimbal_pitch', -90.0)
            yaw = pose['dji_orientation'].get('gimbal_yaw', 0.0)
            if abs(yaw) < 0.1:
                yaw = pose['dji_orientation'].get('flight_yaw', 0.0)
        else:
            roll = None
            pitch = None
            yaw = None
        
        rows.append({
            'image_name': image_name,
            'gps_latitude': gps_lat,
            'gps_longitude': gps_lon,
            'gps_altitude': gps_alt,
            'local_x': local_x,
            'local_y': local_y,
            'local_z': local_z,
            'roll_degrees': roll,
            'pitch_degrees': pitch,
            'yaw_degrees': yaw
        })
    
    # Write to CSV
    if rows:
        fieldnames = [
            'image_name',
            'gps_latitude', 'gps_longitude', 'gps_altitude',
            'local_x', 'local_y', 'local_z',
            'roll_degrees', 'pitch_degrees', 'yaw_degrees'
        ]
        
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"Exported {len(rows)} camera poses to {output_csv}")
    else:
        print(f"Warning: No camera poses to export")
