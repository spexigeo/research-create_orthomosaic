"""
Create a 2D visualization of image footprints to show overlap and rotation.
"""

import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import sys

# Import footprint calculation from the inputs module
sys.path.insert(0, str(Path(__file__).parent / 'inputs'))
from main import Footprint


def calculate_fov_from_focal_length(focal_length_mm: float, sensor_dimension_mm: float) -> float:
    """Calculate field of view in degrees from focal length and sensor dimension."""
    fov_rad = 2 * math.atan(sensor_dimension_mm / (2 * focal_length_mm))
    return math.degrees(fov_rad)


def compute_footprint_polygon(
    gps: dict,
    dji_orientation: dict,
    focal_length_mm: float,
    sensor_width_mm: float,
    sensor_height_mm: float,
    origin_lat: float,
    origin_lon: float
) -> np.ndarray:
    """
    Compute image footprint polygon in local meters.
    
    Returns:
        Array of 4 corner points (4, 2) in local meters (x, y)
    """
    # Calculate FOV
    fov_h = calculate_fov_from_focal_length(focal_length_mm, sensor_width_mm)
    fov_v = calculate_fov_from_focal_length(focal_length_mm, sensor_height_mm)
    
    # Get camera parameters
    altitude = gps['altitude']
    latitude = gps['latitude']
    longitude = gps['longitude']
    
    # Get orientation (use gimbal angles)
    # Note: The Footprint class adjusts yaw by -90 degrees internally
    roll = dji_orientation.get('gimbal_roll', 0.0)
    pitch = dji_orientation.get('gimbal_pitch', -90.0)
    # Try flight_yaw if gimbal_yaw is not available or is 0
    yaw = dji_orientation.get('gimbal_yaw', 0.0)
    if abs(yaw) < 0.1:  # If gimbal_yaw is essentially 0, try flight_yaw
        yaw = dji_orientation.get('flight_yaw', 0.0)
    
    # Calculate footprint using Footprint class
    footprint_feature = Footprint.get_bounding_polygon(
        fov_h=fov_h,
        fov_v=fov_v,
        altitude=altitude,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        latitude=latitude,
        longitude=longitude
    )
    
    # Extract polygon coordinates from GeoJSON
    if footprint_feature.geometry is None:
        return None
    
    polygon_coords = footprint_feature.geometry.coordinates[0]
    
    # Convert lat/lon to local meters
    corners_2d = []
    for lon, lat in polygon_coords:
        if len(corners_2d) < 4:
            lat_diff = lat - origin_lat
            lon_diff = lon - origin_lon
            
            x = lon_diff * 111000.0 * math.cos(math.radians(origin_lat))
            y = lat_diff * 111000.0
            
            corners_2d.append([x, y])
    
    # Ensure we have exactly 4 corners
    if len(corners_2d) >= 4:
        return np.array(corners_2d[:4])
    elif len(corners_2d) > 0:
        while len(corners_2d) < 4:
            corners_2d.append(corners_2d[-1])
        return np.array(corners_2d[:4])
    
    return None


def visualize_footprints_2d(
    camera_poses_file: str = "outputs/camera_poses.json",
    output_file: str = "test_visualization/footprints_2d.png",
    focal_length_mm: float = 8.8,
    sensor_width_mm: float = 13.2,
    sensor_height_mm: float = 9.9,
    show_image_numbers: bool = True,
    alpha: float = 0.3
):
    """
    Create a 2D visualization of image footprints.
    
    Args:
        camera_poses_file: Path to camera_poses.json
        output_file: Path to output PNG file
        focal_length_mm: Focal length in millimeters
        sensor_width_mm: Sensor width in millimeters
        sensor_height_mm: Sensor height in millimeters
        show_image_numbers: Whether to show image numbers on footprints
        alpha: Transparency of footprint polygons (0-1)
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
    
    # Compute all footprints
    footprints = []
    image_names = []
    
    for pose in poses:
        if not pose.get('gps') or not pose.get('dji_orientation'):
            continue
        
        footprint_poly = compute_footprint_polygon(
            gps=pose['gps'],
            dji_orientation=pose['dji_orientation'],
            focal_length_mm=focal_length_mm,
            sensor_width_mm=sensor_width_mm,
            sensor_height_mm=sensor_height_mm,
            origin_lat=origin_lat,
            origin_lon=origin_lon
        )
        
        if footprint_poly is not None:
            footprints.append(footprint_poly)
            image_names.append(pose['image_name'])
    
    if not footprints:
        raise ValueError("No valid footprints computed")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    
    # Determine plot bounds
    all_x = []
    all_y = []
    for fp in footprints:
        all_x.extend(fp[:, 0])
        all_y.extend(fp[:, 1])
    
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    
    # Add padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    padding = max(x_range, y_range) * 0.05
    
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (meters, relative to origin)', fontsize=12)
    ax.set_ylabel('Y (meters, relative to origin)', fontsize=12)
    ax.set_title('Image Footprints (2D View)\nShowing overlap and rotation', fontsize=14, fontweight='bold')
    
    # Draw footprints with color gradient
    num_footprints = len(footprints)
    colors = plt.cm.viridis(np.linspace(0, 1, num_footprints))
    
    for i, (footprint, img_name) in enumerate(zip(footprints, image_names)):
        # Create polygon patch
        polygon = patches.Polygon(
            footprint,
            closed=True,
            edgecolor='black',
            linewidth=0.5,
            facecolor=colors[i],
            alpha=alpha
        )
        ax.add_patch(polygon)
        
        # Add image number label at centroid
        if show_image_numbers:
            centroid = footprint.mean(axis=0)
            # Extract image number from filename (e.g., "0001" from "8928d89ac57ffff_172550_0001.jpg")
            try:
                parts = img_name.split('_')
                if len(parts) >= 3:
                    img_num = parts[-1].replace('.jpg', '')
                    ax.text(centroid[0], centroid[1], img_num, 
                           fontsize=6, ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))
            except:
                pass
    
    # Add legend showing color progression
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=num_footprints-1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Image Sequence')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved footprint visualization to: {output_file}")
    print(f"  Total footprints: {num_footprints}")
    print(f"  X range: [{x_min:.1f}, {x_max:.1f}] meters")
    print(f"  Y range: [{y_min:.1f}, {y_max:.1f}] meters")
    
    # Analyze rotation
    print("\nAnalyzing footprint rotation:")
    rotations = []
    for i, footprint in enumerate(footprints[:20]):  # Check first 20
        # Calculate angle of first edge (from corner 0 to corner 1)
        edge = footprint[1] - footprint[0]
        angle_rad = math.atan2(edge[1], edge[0])
        angle_deg = math.degrees(angle_rad)
        rotations.append(angle_deg)
        if i < 5:
            print(f"  Image {i+1}: rotation = {angle_deg:.1f}°")
    
    if rotations:
        rotation_std = np.std(rotations)
        print(f"  Rotation std (first 20): {rotation_std:.1f}°")
        if rotation_std < 1.0:
            print("  ⚠️  Warning: Very low rotation variation - footprints may be grid-aligned")
        else:
            print("  ✓ Good rotation variation detected")


if __name__ == "__main__":
    visualize_footprints_2d()
