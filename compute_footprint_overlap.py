"""
Compute overlap percentage between image footprints.
Outputs a file showing matching percentage for each image pair.
"""

import json
import math
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Tuple

# Import footprint calculation from the inputs module
# Try multiple possible paths for the inputs directory
possible_paths = [
    Path(__file__).parent / 'inputs',  # ./inputs (same directory as script)
    Path(__file__).parent.parent / 'inputs',  # ../inputs
]
FOOTPRINT_AVAILABLE = False
Footprint = None
for inputs_path in possible_paths:
    if inputs_path.exists() and (inputs_path / 'main.py').exists():
        sys.path.insert(0, str(inputs_path))
        try:
            from main import Footprint
            FOOTPRINT_AVAILABLE = True
            break
        except ImportError as e:
            continue

if not FOOTPRINT_AVAILABLE:
    # Fallback: Use a simple footprint calculation without the Footprint class
    # This is a simplified version that computes footprints directly
    print("Warning: Footprint class not available. Using fallback footprint calculation.")
    
    def compute_footprint_simple(gps, dji_orientation, focal_length_mm, sensor_width_mm, sensor_height_mm, origin_lat, origin_lon):
        """Simple footprint calculation without Footprint class."""
        # Calculate FOV
        fov_h = calculate_fov_from_focal_length(focal_length_mm, sensor_width_mm)
        fov_v = calculate_fov_from_focal_length(focal_length_mm, sensor_height_mm)
        
        # Get camera parameters
        altitude = gps['altitude']
        latitude = gps['latitude']
        longitude = gps['longitude']
        
        # Get orientation
        roll = dji_orientation.get('gimbal_roll', 0.0)
        pitch = dji_orientation.get('gimbal_pitch', -90.0)
        yaw = dji_orientation.get('gimbal_yaw', 0.0)
        if abs(yaw) < 0.1:
            yaw = dji_orientation.get('flight_yaw', 0.0)
        
        # Simple footprint calculation: project image corners to ground
        # This is a simplified version - for accurate footprints, use the Footprint class
        # Calculate ground coverage based on FOV and altitude
        ground_width = 2 * altitude * math.tan(math.radians(fov_h / 2))
        ground_height = 2 * altitude * math.tan(math.radians(fov_v / 2))
        
        # Convert to local meters
        lat_diff = latitude - origin_lat
        lon_diff = longitude - origin_lon
        x_center = lon_diff * 111000.0 * math.cos(math.radians(origin_lat))
        y_center = lat_diff * 111000.0
        
        # Create a simple rectangle (will be rotated by yaw)
        corners = [
            [-ground_width/2, -ground_height/2],
            [ground_width/2, -ground_height/2],
            [ground_width/2, ground_height/2],
            [-ground_width/2, ground_height/2]
        ]
        
        # Rotate by yaw
        cos_yaw = math.cos(math.radians(yaw))
        sin_yaw = math.sin(math.radians(yaw))
        rotated_corners = []
        for corner in corners:
            x_rot = corner[0] * cos_yaw - corner[1] * sin_yaw
            y_rot = corner[0] * sin_yaw + corner[1] * cos_yaw
            rotated_corners.append([x_center + x_rot, y_center + y_rot])
        
        return Polygon(rotated_corners)
    
    # Replace Footprint.get_bounding_polygon with our simple version
    class Footprint:
        @staticmethod
        def get_bounding_polygon(*args, **kwargs):
            # This won't be called if we use compute_footprint_simple directly
            pass

try:
    from shapely.geometry import Polygon
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print("Warning: shapely not available. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shapely"])
    from shapely.geometry import Polygon
    SHAPELY_AVAILABLE = True


def calculate_fov_from_focal_length(focal_length_mm: float, sensor_dimension_mm: float) -> float:
    """Calculate field of view in degrees from focal length and sensor dimension."""
    fov_rad = 2 * math.atan(sensor_dimension_mm / (2 * focal_length_mm))
    return math.degrees(fov_rad)
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
) -> Tuple[Polygon, str]:
    """
    Compute image footprint polygon in local meters.
    
    Returns:
        Tuple of (Polygon object, image_name) or (None, image_name) if calculation fails
    """
    # Calculate FOV
    fov_h = calculate_fov_from_focal_length(focal_length_mm, sensor_width_mm)
    fov_v = calculate_fov_from_focal_length(focal_length_mm, sensor_height_mm)
    
    # Get camera parameters
    altitude = gps['altitude']
    latitude = gps['latitude']
    longitude = gps['longitude']
    
    # Get orientation (use flight_yaw if gimbal_yaw is 0)
    roll = dji_orientation.get('gimbal_roll', 0.0)
    pitch = dji_orientation.get('gimbal_pitch', -90.0)
    yaw = dji_orientation.get('gimbal_yaw', 0.0)
    if abs(yaw) < 0.1:
        yaw = dji_orientation.get('flight_yaw', 0.0)
    
    # Calculate footprint
    if FOOTPRINT_AVAILABLE:
        # Use Footprint class if available
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
            return None, None
        
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
    else:
        # Use simple footprint calculation
        footprint_poly = compute_footprint_simple(
            gps, dji_orientation, focal_length_mm, sensor_width_mm, sensor_height_mm,
            origin_lat, origin_lon
        )
        if footprint_poly is None:
            return None, None
        corners_2d = list(footprint_poly.exterior.coords[:-1])  # Exclude closing point
    
    # Ensure we have exactly 4 corners
    if len(corners_2d) >= 4:
        corners_array = np.array(corners_2d[:4])
    elif len(corners_2d) > 0:
        while len(corners_2d) < 4:
            corners_2d.append(corners_2d[-1])
        corners_array = np.array(corners_2d[:4])
    else:
        return None, None
    
    # Create Shapely polygon
    try:
        polygon = Polygon(corners_array)
        if not polygon.is_valid:
            # Try to fix invalid polygon
            polygon = polygon.buffer(0)
        return polygon, None
    except Exception as e:
        return None, None


def compute_overlap_percentage(poly1: Polygon, poly2: Polygon) -> float:
    """
    Compute overlap percentage between two polygons.
    
    Returns:
        Overlap percentage (0-100), where percentage is based on the area of the first polygon
    """
    if poly1 is None or poly2 is None:
        return 0.0
    
    try:
        # Calculate intersection
        intersection = poly1.intersection(poly2)
        
        if intersection.is_empty:
            return 0.0
        
        # Calculate overlap as percentage of first polygon's area
        area1 = poly1.area
        if area1 == 0:
            return 0.0
        
        overlap_area = intersection.area
        overlap_percentage = (overlap_area / area1) * 100.0
        
        return overlap_percentage
    except Exception as e:
        return 0.0


def compute_all_overlaps(
    camera_poses_file: str = "outputs/camera_poses.json",
    output_file: str = "outputs/footprint_overlaps.json",
    focal_length_mm: float = 8.8,
    sensor_width_mm: float = 13.2,
    sensor_height_mm: float = 9.9
):
    """
    Compute overlap percentages for all image pairs based on footprints.
    
    Args:
        camera_poses_file: Path to camera_poses.json
        output_file: Path to output JSON file
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
    
    # Compute all footprints
    print("Computing footprints for all images...")
    footprints = []
    image_names = []
    
    for pose in poses:
        if not pose.get('gps') or not pose.get('dji_orientation'):
            footprints.append(None)
            image_names.append(pose.get('image_name', 'unknown'))
            continue
        
        footprint_poly, _ = compute_footprint_polygon(
            gps=pose['gps'],
            dji_orientation=pose['dji_orientation'],
            focal_length_mm=focal_length_mm,
            sensor_width_mm=sensor_width_mm,
            sensor_height_mm=sensor_height_mm,
            origin_lat=origin_lat,
            origin_lon=origin_lon
        )
        
        footprints.append(footprint_poly)
        image_names.append(pose['image_name'])
    
    print(f"Computed {sum(1 for f in footprints if f is not None)} valid footprints out of {len(footprints)} images")
    
    # Compute overlaps for all pairs
    print("Computing overlaps for all image pairs...")
    overlaps = []
    total_pairs = len(footprints) * (len(footprints) - 1) // 2
    processed = 0
    
    for i in range(len(footprints)):
        if footprints[i] is None:
            continue
        
        for j in range(i + 1, len(footprints)):
            if footprints[j] is None:
                continue
            
            overlap_pct = compute_overlap_percentage(footprints[i], footprints[j])
            
            # Round to 2 decimal places
            overlap_pct_rounded = round(overlap_pct, 2)
            
            # Only include overlaps > 0 (filter out zero overlaps after rounding)
            if overlap_pct_rounded > 0:
                overlaps.append({
                    'image1': image_names[i],
                    'image2': image_names[j],
                    'overlap_percentage': overlap_pct_rounded
                })
            
            processed += 1
            if processed % 1000 == 0:
                print(f"  Processed {processed}/{total_pairs} pairs...")
    
    # Sort by overlap percentage (descending)
    overlaps.sort(key=lambda x: x['overlap_percentage'], reverse=True)
    
    # Save to JSON file
    output_data = {
        'total_images': len(footprints),
        'valid_footprints': sum(1 for f in footprints if f is not None),
        'total_overlapping_pairs': len(overlaps),
        'overlaps': overlaps
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSaved overlap data to: {output_file}")
    print(f"  Total images: {len(footprints)}")
    print(f"  Valid footprints: {sum(1 for f in footprints if f is not None)}")
    print(f"  Overlapping pairs (non-zero): {len(overlaps)}")
    print(f"  Max overlap: {max(o['overlap_percentage'] for o in overlaps) if overlaps else 0:.2f}%")
    print(f"  Min overlap: {min(o['overlap_percentage'] for o in overlaps) if overlaps else 0:.2f}%")
    print(f"  Mean overlap: {np.mean([o['overlap_percentage'] for o in overlaps]) if overlaps else 0:.2f}%")
    
    # Also create a CSV version for easier viewing (save to same directory as JSON)
    csv_file = str(Path(output_file).with_suffix('.csv'))
    with open(csv_file, 'w') as f:
        f.write('image1,image2,overlap_percentage\n')
        for overlap in overlaps:
            f.write(f"{overlap['image1']},{overlap['image2']},{overlap['overlap_percentage']}\n")
    
    print(f"  Also saved CSV version to: {csv_file}")


if __name__ == "__main__":
    compute_all_overlaps()
