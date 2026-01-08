"""
Utility functions for debugging tracks and computing footprint overlaps.
"""

import json
import math
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional

# Import footprint calculation from the inputs module
# Try multiple possible paths for the inputs directory
possible_paths = [
    Path(__file__).parent.parent / 'inputs',  # ../inputs
    Path(__file__).parent.parent.parent / 'inputs',  # ../../inputs
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
    print("Warning: Footprint class not available. Using fallback footprint calculation.")
    
    def compute_footprint_simple(gps, dji_orientation, focal_length_mm, sensor_width_mm, sensor_height_mm, origin_lat, origin_lon):
        """Simple footprint calculation without Footprint class."""
        from shapely.geometry import Polygon
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


def compute_footprint_polygon(
    gps: dict,
    dji_orientation: dict,
    focal_length_mm: float,
    sensor_width_mm: float,
    sensor_height_mm: float,
    origin_lat: float,
    origin_lon: float
) -> Tuple[Optional[Polygon], Optional[str]]:
    """
    Compute image footprint polygon in local meters.
    
    Returns:
        Tuple of (Polygon object, image_name) or (None, None) if calculation fails
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
    image_dir: str,
    output_dir: str,
    camera_poses_file: str,
    output_json_file: str = "outputs/footprint_overlaps.json",
    output_csv_file: str = "outputs/footprint_overlaps.csv",
    focal_length_mm: float = 8.8,
    sensor_width_mm: float = 13.2,
    sensor_height_mm: float = 9.9
):
    """
    Compute overlap percentages for all image pairs based on footprints.
    
    Args:
        image_dir: Directory containing images (unused, kept for compatibility)
        output_dir: Output directory (unused, kept for compatibility)
        camera_poses_file: Path to camera_poses.json
        output_json_file: Path to output JSON file
        output_csv_file: Path to output CSV file
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
    
    Path(output_json_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSaved overlap data to: {output_json_file}")
    print(f"  Total images: {len(footprints)}")
    print(f"  Valid footprints: {sum(1 for f in footprints if f is not None)}")
    print(f"  Overlapping pairs (non-zero): {len(overlaps)}")
    print(f"  Max overlap: {max(o['overlap_percentage'] for o in overlaps) if overlaps else 0:.2f}%")
    print(f"  Min overlap: {min(o['overlap_percentage'] for o in overlaps) if overlaps else 0:.2f}%")
    print(f"  Mean overlap: {np.mean([o['overlap_percentage'] for o in overlaps]) if overlaps else 0:.2f}%")
    
    # Also create a CSV version for easier viewing
    Path(output_csv_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv_file, 'w') as f:
        f.write('image1,image2,overlap_percentage\n')
        for overlap in overlaps:
            f.write(f"{overlap['image1']},{overlap['image2']},{overlap['overlap_percentage']}\n")
    
    print(f"  Also saved CSV version to: {output_csv_file}")


def debug_track(tracks_file: str, matches_file: str, features_file: str, track_id: int = 11):
    """
    Debug script to verify track matches.
    
    Args:
        tracks_file: Path to tracks JSON file
        matches_file: Path to matches JSON file
        features_file: Path to features JSON file
        track_id: ID of track to debug (default: 11)
    """
    # Load data
    tracks = json.load(open(tracks_file))
    matches = json.load(open(matches_file))
    features = json.load(open(features_file))

    # Get specified track
    track_data = [t for t in tracks['tracks'] if t['track_id'] == track_id]
    if not track_data:
        print(f"Track {track_id} not found!")
        return
    
    track = track_data[0]
    print(f"Track {track_id}: {track['length']} features")
    print(f"First 5 features: {track['features'][:5]}\n")

    # Check matches between consecutive images
    for i in range(len(track['features']) - 1):
        img1, idx1 = track['features'][i]
        img2, idx2 = track['features'][i + 1]
        
        print(f"Checking: {img1} feature {idx1} -> {img2} feature {idx2}")
        
        # Find match between these two images
        img1_name = Path(img1).name if '/' in img1 else img1
        img2_name = Path(img2).name if '/' in img2 else img2
        
        found_match = False
        for match in matches:
            match_img0 = Path(match['image0']).name if '/' in match['image0'] else match['image0']
            match_img1 = Path(match['image1']).name if '/' in match['image1'] else match['image1']
            
            if (match_img0 == img1_name and match_img1 == img2_name) or \
               (match_img0 == img2_name and match_img1 == img1_name):
                matches_list = np.array(match['matches'])
                
                # Check direction
                if match_img0 == img2_name:
                    # Need to swap
                    matches_list = matches_list[:, [1, 0]]
                
                # Check if (idx1, idx2) is in matches
                for m in matches_list:
                    if int(m[0]) == idx1 and int(m[1]) == idx2:
                        found_match = True
                        print(f"  ✓ MATCH FOUND")
                        break
                
                if not found_match:
                    print(f"  ✗ NO DIRECT MATCH")
                    # Show what feature idx1 matches to in img2
                    matches_from_idx1 = [m for m in matches_list if int(m[0]) == idx1]
                    if matches_from_idx1:
                        print(f"    Feature {idx1} in {img1} matches to: {[int(m[1]) for m in matches_from_idx1[:5]]}")
                    # Show what matches to idx2 in img2
                    matches_to_idx2 = [m for m in matches_list if int(m[1]) == idx2]
                    if matches_to_idx2:
                        print(f"    Feature {idx2} in {img2} is matched from: {[int(m[0]) for m in matches_to_idx2[:5]]}")
                break
        
        if not found_match:
            print(f"  ✗ NO MATCH DATA FOUND between these images")
        print()


def parse_image_number(image_name: str) -> int:
    """Extract image number from filename."""
    import re
    match = re.search(r'_(\d{4})\.jpg$', image_name)
    return int(match.group(1)) if match else -1


def calculate_heading(lat1, lon1, lat2, lon2):
    """Calculate heading (bearing) between two GPS points in degrees."""
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    
    y = np.sin(dlon) * np.cos(lat2_rad)
    x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
    
    heading = np.degrees(np.arctan2(y, x))
    return (heading + 360) % 360  # Normalize to 0-360


def detect_scanlines(poses_file: str = "outputs/camera_poses.json") -> Dict:
    """
    Detect scanlines by analyzing camera positions and headings.
    
    Returns:
        Dictionary with scanline assignments: {scanline_id: [image_numbers]}
    """
    # Load poses
    with open(poses_file, 'r') as f:
        poses = json.load(f)
    
    # Filter to poses with GPS
    poses_with_gps = [p for p in poses if p.get('gps') is not None]
    
    # Sort by image number
    poses_with_gps.sort(key=lambda p: parse_image_number(p['image_name']))
    
    # Extract data
    image_numbers = [parse_image_number(p['image_name']) for p in poses_with_gps]
    lats = np.array([p['gps']['latitude'] for p in poses_with_gps])
    lons = np.array([p['gps']['longitude'] for p in poses_with_gps])
    
    # Calculate headings between consecutive images
    headings = []
    distances = []
    for i in range(len(lats) - 1):
        heading = calculate_heading(lats[i], lons[i], lats[i+1], lons[i+1])
        headings.append(heading)
        
        # Distance in meters
        lat_diff = (lats[i+1] - lats[i]) * 111320
        lon_diff = (lons[i+1] - lons[i]) * 111320 * np.cos(np.radians(lats[i]))
        dist = np.sqrt(lat_diff**2 + lon_diff**2)
        distances.append(dist)
    
    # Detect scanline boundaries by finding large heading changes
    # A scanline turn typically involves a heading change > 90 degrees
    heading_changes = []
    for i in range(len(headings) - 1):
        change = abs(headings[i+1] - headings[i])
        if change > 180:
            change = 360 - change
        heading_changes.append(change)
    
    # Find scanline boundaries (large heading changes)
    # Use a higher threshold and require sustained direction changes
    threshold = 120  # degrees - more conservative
    boundaries = [0]  # Start with first image
    
    # Look for sustained direction changes (not just single spikes)
    for i in range(len(heading_changes) - 2):
        # Check if this is a sustained turn (current and next change are both large)
        if heading_changes[i] > threshold and heading_changes[i+1] > threshold:
            boundaries.append(i + 2)  # Image after the turn completes
        # Or if it's a very large single turn (180 degrees = U-turn)
        elif heading_changes[i] > 170:
            boundaries.append(i + 1)
    
    boundaries.append(len(image_numbers))  # End with last image
    
    # Remove duplicate boundaries
    boundaries = sorted(list(set(boundaries)))
    
    # Group images into scanlines
    scanlines = {}
    for scanline_id, start_idx in enumerate(boundaries[:-1], 1):
        end_idx = boundaries[scanline_id] if scanline_id < len(boundaries) - 1 else len(image_numbers)
        scanline_images = image_numbers[start_idx:end_idx]
        scanlines[scanline_id] = scanline_images
    
    return scanlines


def verify_scanline_straightness(poses_file: str = "outputs/camera_poses.json", 
                                 scanlines: Dict = None) -> Dict:
    """
    Verify that images in each scanline lie approximately on a straight line.
    
    Returns:
        Dictionary with scanline quality metrics
    """
    if scanlines is None:
        scanlines = detect_scanlines(poses_file)
    
    # Load poses
    with open(poses_file, 'r') as f:
        poses = json.load(f)
    
    poses_dict = {parse_image_number(p['image_name']): p for p in poses if p.get('gps')}
    
    results = {}
    for scanline_id, image_nums in scanlines.items():
        if len(image_nums) < 2:
            continue
        
        # Get positions for this scanline
        positions = []
        for img_num in image_nums:
            if img_num in poses_dict:
                p = poses_dict[img_num]
                positions.append([p['gps']['latitude'], p['gps']['longitude']])
        
        if len(positions) < 2:
            continue
        
        positions = np.array(positions)
        
        # Fit a line to the positions (using PCA to find principal direction)
        # Center the data
        center = positions.mean(axis=0)
        centered = positions - center
        
        # PCA to find principal direction
        if len(centered) > 1:
            cov = np.cov(centered.T)
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            principal_dir = eigenvecs[:, np.argmax(eigenvals)]
            
            # Project points onto the line
            projections = np.dot(centered, principal_dir)
            projected_points = center + np.outer(projections, principal_dir)
            
            # Calculate RMS distance from line
            distances = np.linalg.norm(positions - projected_points, axis=1)
            rms_error = np.sqrt(np.mean(distances**2))
            
            # Convert to meters (rough approximation)
            rms_error_m = rms_error * 111320
            
            results[scanline_id] = {
                'image_numbers': image_nums,
                'num_images': len(image_nums),
                'rms_error_meters': rms_error_m,
                'start_image': min(image_nums),
                'end_image': max(image_nums)
            }
    
    return results
