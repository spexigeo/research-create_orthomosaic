"""
Extract camera pose information from image EXIF metadata and export to PLY format.
"""
import json
import math
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import numpy as np
from typing import Dict, List, Optional, Tuple
import sys


def get_decimal_from_dms(dms, ref):
    """Convert degrees, minutes, seconds to decimal degrees."""
    degrees = dms[0]
    minutes = dms[1] / 60.0
    seconds = dms[2] / 3600.0
    
    if ref in ['S', 'W']:
        return -(degrees + minutes + seconds)
    return degrees + minutes + seconds


def extract_gps_coordinates(exif_data) -> Optional[Tuple[float, float, float]]:
    """Extract GPS coordinates (lat, lon, altitude) from EXIF data."""
    if 34853 not in exif_data:  # GPSInfo tag
        return None
    
    gps_info = exif_data[34853]
    
    # GPS info might be stored as a dict or as a reference
    # Try to convert to dict if needed
    if not isinstance(gps_info, dict):
        # Sometimes it's stored as a reference - try to get the actual data
        try:
            # Access via _get_ifd method if available
            if hasattr(exif_data, '_get_ifd'):
                gps_info = exif_data._get_ifd(34853)
            else:
                return None
        except:
            return None
    
    if not isinstance(gps_info, dict):
        return None
    
    # Get latitude
    if 2 not in gps_info or 1 not in gps_info:  # GPSLatitude and GPSLatitudeRef
        return None
    
    lat_dms = gps_info[2]
    lat_ref = gps_info[1]
    lat = get_decimal_from_dms(lat_dms, lat_ref)
    
    # Get longitude
    if 4 not in gps_info or 3 not in gps_info:  # GPSLongitude and GPSLongitudeRef
        return None
    
    lon_dms = gps_info[4]
    lon_ref = gps_info[3]
    lon = get_decimal_from_dms(lon_dms, lon_ref)
    
    # Get altitude
    altitude = None
    if 6 in gps_info:  # GPSAltitude
        altitude = float(gps_info[6])
        if 5 in gps_info and gps_info[5] == 1:  # GPSAltitudeRef (1 = below sea level)
            altitude = -altitude
    
    return (lat, lon, altitude if altitude is not None else 0.0)


def extract_orientation(exif_data) -> Optional[Dict[str, float]]:
    """Extract camera orientation from EXIF data."""
    orientation = {}
    
    # Try to get orientation from various EXIF tags
    # These tags may vary by camera manufacturer
    
    # Common orientation tags
    orientation_tags = {
        'CameraRoll': (0x9206,),  # Orientation
        'CameraPitch': None,  # May not be in standard EXIF
        'CameraYaw': None,   # May not be in standard EXIF
    }
    
    # Check for orientation (rotation)
    if 274 in exif_data:  # Orientation tag
        orientation['image_orientation'] = exif_data[274]
    
    # Check for custom tags that might contain pose info
    # Some drones store this in custom tags
    for tag_id, value in exif_data.items():
        tag_name = TAGS.get(tag_id, f'Unknown_{tag_id}')
        if isinstance(tag_name, str) and any(keyword in tag_name.lower() for keyword in ['yaw', 'pitch', 'roll', 'gimbal', 'attitude', 'orientation']):
            orientation[tag_name] = value
    
    return orientation if orientation else None


def extract_camera_pose(image_path: str) -> Dict:
    """
    Extract camera pose information from image EXIF metadata.
    
    Returns:
        Dictionary with pose information:
        - image_path: Path to image
        - gps: (lat, lon, altitude) tuple if available
        - orientation: Dictionary with orientation info if available
        - exif_tags: All relevant EXIF tags
    """
    result = {
        'image_path': str(image_path),
        'image_name': Path(image_path).name,
        'gps': None,
        'orientation': None,
        'exif_available': False
    }
    
    try:
        img = Image.open(image_path)
        # Use _getexif() for better GPS support
        exif = img._getexif() if hasattr(img, '_getexif') else img.getexif()
        
        if not exif:
            return result
        
        result['exif_available'] = True
        
        # Extract GPS coordinates
        gps = extract_gps_coordinates(exif)
        if gps:
            result['gps'] = {
                'latitude': gps[0],
                'longitude': gps[1],
                'altitude': gps[2]
            }
        
        # Extract orientation (basic EXIF)
        orientation = extract_orientation(exif)
        if orientation:
            result['orientation'] = orientation
        
        # Try to extract DJI-specific orientation using exifread/exiftool
        try:
            from .dji_exif_parser import extract_dji_orientation
            dji_orientation = extract_dji_orientation(str(image_path))
            if dji_orientation:
                result['dji_orientation'] = dji_orientation
        except Exception:
            pass  # DJI orientation extraction is optional
        
        # Store some useful EXIF tags
        useful_tags = {}
        for tag_id, value in exif.items():
            tag_name = TAGS.get(tag_id, f'Tag_{tag_id}')
            # Store datetime, camera model, etc.
            if tag_name in ['DateTime', 'DateTimeOriginal', 'Make', 'Model', 'Software']:
                useful_tags[tag_name] = str(value)
        
        result['exif_tags'] = useful_tags
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def extract_poses_from_directory(image_dir: str, pattern: str = "*.jpg", 
                                  output_file: Optional[str] = None,
                                  export_ply: bool = True,
                                  ply_output_file: Optional[str] = None) -> List[Dict]:
    """
    Extract camera poses from all images in a directory.
    
    Args:
        image_dir: Directory containing images
        pattern: File pattern to match (default: "*.jpg")
        output_file: Optional path to save camera_poses.json
        export_ply: Whether to export PLY file (default: True)
        ply_output_file: Optional path for PLY output (default: outputs/camera_poses_3d.ply)
    
    Returns:
        List of camera pose dictionaries
    """
    image_dir_path = Path(image_dir)
    image_files = sorted(image_dir_path.glob(pattern))
    
    poses = []
    for img_path in image_files:
        pose = extract_camera_pose(str(img_path))
        poses.append(pose)
    
    # Save to JSON if output_file is provided
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(poses, f, indent=2)
        print(f"Saved camera poses to: {output_file}")
    
    # Export to PLY if requested
    if export_ply:
        if not output_file:
            # Default output file if not provided
            output_file = "outputs/camera_poses.json"
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(poses, f, indent=2)
        
        if not ply_output_file:
            ply_output_file = str(Path(output_file).parent / "camera_poses_3d.ply")
        
        # Read image dimensions from first image
        from matcher.utils import get_image_dimensions
        if image_files:
            image_width, image_height = get_image_dimensions(str(image_files[0]))
        else:
            raise ValueError("No images found to read dimensions from")
        
        export_camera_poses_to_ply(str(output_file), ply_output_file, image_width, image_height)
    
    return poses


if __name__ == "__main__":
    import sys
    
    image_dir = "/Users/mauriciohessflores/Documents/Code/MyCode/research-qualicum_beach_gcp_analysis/input/images"
    output_file = "outputs/camera_poses.json"
    
    print("Extracting camera poses from images...")
    poses = extract_poses_from_directory(image_dir)
    
    # Filter to only images from the central cell for now
    central_cell = "8928d89ac57ffff"
    cell_poses = [p for p in poses if central_cell in p['image_name']]
    
    print(f"Extracted poses from {len(cell_poses)} images")
    print(f"Images with GPS: {sum(1 for p in cell_poses if p['gps'] is not None)}")
    print(f"Images with orientation: {sum(1 for p in cell_poses if p['orientation'] is not None)}")
    
    # Save to JSON
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(cell_poses, f, indent=2)
    
    print(f"Saved camera poses to: {output_file}")
    
    # Print sample
    if cell_poses:
        print("\nSample pose data:")
        sample = cell_poses[0]
        print(json.dumps(sample, indent=2))
def compute_footprint_from_exif(
    gps: dict,
    dji_orientation: dict,
    focal_length_mm: float,
    sensor_width_mm: float,
    sensor_height_mm: float,
    origin_lat: float,
    origin_lon: float,
    origin_alt: float
) -> Optional[np.ndarray]:
    """
    Compute image footprint using the Footprint class from inputs/main.py.
    
    Args:
        gps: Dictionary with 'latitude', 'longitude', 'altitude'
        dji_orientation: Dictionary with 'gimbal_pitch', 'gimbal_roll', 'gimbal_yaw'
        focal_length_mm: Focal length in millimeters
        sensor_width_mm: Sensor width in millimeters
        sensor_height_mm: Sensor height in millimeters
        origin_lat: Origin latitude for coordinate conversion
        origin_lon: Origin longitude for coordinate conversion
        origin_alt: Origin altitude for coordinate conversion
    
    Returns:
        Array of 4 corner points in local meters (4, 3) or None if calculation fails
    """
    try:
        # Calculate FOV from focal length and sensor dimensions
        fov_h = calculate_fov_from_focal_length(focal_length_mm, sensor_width_mm)
        fov_v = calculate_fov_from_focal_length(focal_length_mm, sensor_height_mm)
        
        # Get camera parameters
        altitude = gps['altitude']
        latitude = gps['latitude']
        longitude = gps['longitude']
        
        # Get orientation (use gimbal angles)
        # Note: Use flight_yaw if gimbal_yaw is 0 (gimbal may be locked to 0)
        roll = dji_orientation.get('gimbal_roll', 0.0)
        pitch = dji_orientation.get('gimbal_pitch', -90.0)  # Default to nadir
        yaw = dji_orientation.get('gimbal_yaw', 0.0)
        if abs(yaw) < 0.1:  # If gimbal_yaw is essentially 0, use flight_yaw
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
        
        # Convert lat/lon to local meters (relative to origin)
        corners_3d = []
        for lon, lat in polygon_coords:
            # Skip the closing point if it's a duplicate
            if len(corners_3d) > 0 and len(corners_3d) < 4:
                # Convert to local meters
                lat_diff = lat - origin_lat
                lon_diff = lon - origin_lon
                
                x = lon_diff * 111000.0 * math.cos(math.radians(origin_lat))
                y = lat_diff * 111000.0
                z = 0.0  # Ground plane
                
                corners_3d.append([x, y, z])
        
        # Ensure we have exactly 4 corners
        if len(corners_3d) >= 4:
            return np.array(corners_3d[:4])
        elif len(corners_3d) > 0:
            # If we have fewer than 4, pad with the last point
            while len(corners_3d) < 4:
                corners_3d.append(corners_3d[-1])
            return np.array(corners_3d[:4])
        
        return None
        
    except Exception as e:
        print(f"Warning: Error computing footprint from EXIF: {e}")
        return None


def project_image_corners_to_ground(
    camera_center: np.ndarray,
    camera_rotation: np.ndarray,
    image_width: int,
    image_height: int,
    focal_length_px: float,
    ground_z: float = 0.0
) -> Optional[np.ndarray]:
    """
    Project image corners onto the ground plane (fallback method).
    
    Args:
        camera_center: Camera position in world coordinates (3D)
        camera_rotation: Camera rotation matrix (world to camera) (3x3)
        image_width: Image width in pixels
        image_height: Image height in pixels
        focal_length_px: Focal length in pixels
        ground_z: Z coordinate of ground plane (default: 0.0)
    
    Returns:
        Array of 4 corner points on ground plane (4, 3) or None if projection fails
    """
    # Camera intrinsics
    cx = image_width / 2.0
    cy = image_height / 2.0
    
    # Image corners in pixel coordinates (normalized)
    corners_px = np.array([
        [0, 0],                    # Top-left
        [image_width, 0],          # Top-right
        [image_width, image_height],  # Bottom-right
        [0, image_height]         # Bottom-left
    ], dtype=np.float64)
    
    # Convert pixel coordinates to normalized camera coordinates
    # In camera frame: X = (u - cx) / fx, Y = (v - cy) / fy, Z = 1
    # Then normalize to get ray direction
    corners_3d_world = []
    
    # R is world-to-camera, so R^T is camera-to-world
    R_cam_to_world = camera_rotation.T
    
    for u, v in corners_px:
        # Normalized camera coordinates (direction vector in camera frame)
        X_cam = (u - cx) / focal_length_px
        Y_cam = (v - cy) / focal_length_px
        Z_cam = 1.0
        
        # Normalize to get unit direction vector in camera frame
        ray_dir_cam = np.array([X_cam, Y_cam, Z_cam])
        ray_dir_cam = ray_dir_cam / np.linalg.norm(ray_dir_cam)
        
        # Transform ray direction to world frame
        ray_dir_world = R_cam_to_world @ ray_dir_cam
        
        # Intersect ray with ground plane (z = ground_z)
        # Ray: P(t) = camera_center + t * ray_dir_world
        # Ground: z = ground_z
        # Solve: camera_center[2] + t * ray_dir_world[2] = ground_z
        if abs(ray_dir_world[2]) < 1e-6:
            # Ray is parallel to ground plane, skip this corner
            continue
        
        t = (ground_z - camera_center[2]) / ray_dir_world[2]
        
        if t < 0:
            # Ray points away from ground (upward), skip
            continue
        
        # Calculate intersection point
        intersection = camera_center + t * ray_dir_world
        corners_3d_world.append(intersection)
    
    if len(corners_3d_world) < 4:
        # Not all corners could be projected
        return None
    
    return np.array(corners_3d_world)


def create_polygon_outline(
    corners: np.ndarray,
    line_width: float = 0.5,
    line_height: float = 0.1
) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Create a thick outline around a polygon by creating a thin box along each edge.
    
    Args:
        corners: Array of corner points (N, 3) - should be 4 corners for a quad
        line_width: Width of the outline line
        line_height: Height of the outline (thickness above ground)
    
    Returns:
        vertices: Array of vertices for the outline
        faces: List of face indices
    """
    if len(corners) != 4:
        return np.array([]), []
    
    vertices = []
    faces = []
    
    # Create outline for each edge
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        
        # Edge direction
        edge_dir = p2 - p1
        edge_length = np.linalg.norm(edge_dir)
        if edge_length < 1e-6:
            continue
        edge_dir = edge_dir / edge_length
        
        # Perpendicular direction (in XY plane)
        perp = np.array([-edge_dir[1], edge_dir[0], 0])
        perp_norm = np.linalg.norm(perp)
        if perp_norm < 1e-6:
            perp = np.array([1, 0, 0])
        else:
            perp = perp / perp_norm
        
        # Create a thin box along the edge
        half_width = line_width / 2.0
        
        # Four corners of the box cross-section at ground level
        v1 = p1 + perp * half_width
        v2 = p1 - perp * half_width
        v3 = p2 - perp * half_width
        v4 = p2 + perp * half_width
        
        # Current vertex offset (before adding new vertices)
        base_idx = len(vertices)
        
        # Add vertices at ground level
        vertices.append(v1)
        vertices.append(v2)
        vertices.append(v3)
        vertices.append(v4)
        
        # Add vertices at top level
        v1_top = v1 + np.array([0, 0, line_height])
        v2_top = v2 + np.array([0, 0, line_height])
        v3_top = v3 + np.array([0, 0, line_height])
        v4_top = v4 + np.array([0, 0, line_height])
        
        vertices.append(v1_top)
        vertices.append(v2_top)
        vertices.append(v3_top)
        vertices.append(v4_top)
        
        # Create faces for the box (bottom, top, 4 sides)
        # Bottom face (two triangles)
        faces.append([base_idx, base_idx + 1, base_idx + 2])
        faces.append([base_idx, base_idx + 2, base_idx + 3])
        
        # Top face (two triangles)
        faces.append([base_idx + 4, base_idx + 7, base_idx + 6])
        faces.append([base_idx + 4, base_idx + 6, base_idx + 5])
        
        # Side faces (4 sides, 2 triangles each)
        # Side 1: v1-v2 edge
        faces.append([base_idx, base_idx + 4, base_idx + 5])
        faces.append([base_idx, base_idx + 5, base_idx + 1])
        
        # Side 2: v2-v3 edge
        faces.append([base_idx + 1, base_idx + 5, base_idx + 6])
        faces.append([base_idx + 1, base_idx + 6, base_idx + 2])
        
        # Side 3: v3-v4 edge
        faces.append([base_idx + 2, base_idx + 6, base_idx + 7])
        faces.append([base_idx + 2, base_idx + 7, base_idx + 3])
        
        # Side 4: v4-v1 edge
        faces.append([base_idx + 3, base_idx + 7, base_idx + 4])
        faces.append([base_idx + 3, base_idx + 4, base_idx])
    
    return np.array(vertices), faces


def create_sphere(center: np.ndarray, radius: float, num_segments: int = 16) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Create a sphere mesh for PLY export.
    
    Args:
        center: Center point of sphere (3D)
        radius: Radius of sphere
        num_segments: Number of segments for sphere (higher = smoother)
    
    Returns:
        Tuple of (vertices, faces) where vertices is (N, 3) array and faces is list of triangle indices
    """
    vertices = []
    faces = []
    
    # Generate sphere vertices using spherical coordinates
    for i in range(num_segments + 1):  # Latitude (theta)
        theta = math.pi * i / num_segments  # 0 to pi
        for j in range(num_segments + 1):  # Longitude (phi)
            phi = 2 * math.pi * j / num_segments  # 0 to 2*pi
            
            # Convert to Cartesian coordinates
            x = radius * math.sin(theta) * math.cos(phi)
            y = radius * math.sin(theta) * math.sin(phi)
            z = radius * math.cos(theta)
            
            vertices.append([center[0] + x, center[1] + y, center[2] + z])
    
    # Generate faces (triangles)
    for i in range(num_segments):
        for j in range(num_segments):
            # Current quad indices
            v1 = i * (num_segments + 1) + j
            v2 = i * (num_segments + 1) + (j + 1)
            v3 = (i + 1) * (num_segments + 1) + (j + 1)
            v4 = (i + 1) * (num_segments + 1) + j
            
            # Two triangles per quad
            faces.append([v1, v2, v3])
            faces.append([v1, v3, v4])
    
    return np.array(vertices), faces


def create_arrow(start: np.ndarray, direction: np.ndarray, length: float, 
                 shaft_radius: float = 0.1, head_length: float = 0.3, 
                 head_radius: float = 0.2, num_segments: int = 8) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Create an arrow mesh for PLY export.
    
    Args:
        start: Start point (3D)
        direction: Direction vector (normalized)
        length: Total arrow length
        shaft_radius: Radius of arrow shaft
        head_length: Length of arrow head
        head_radius: Radius of arrow head base
        num_segments: Number of segments for circular cross-section
    
    Returns:
        vertices: Array of vertices (N, 3)
        faces: List of face indices (triangles)
    """
    vertices = []
    faces = []
    
    # Normalize direction
    direction = direction / np.linalg.norm(direction)
    
    # Shaft length
    shaft_length = length - head_length
    
    # Create perpendicular vectors for cross-section
    if abs(direction[2]) < 0.9:
        perp1 = np.cross(direction, np.array([0, 0, 1]))
    else:
        perp1 = np.cross(direction, np.array([1, 0, 0]))
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)
    perp2 = perp2 / np.linalg.norm(perp2)
    
    vertex_offset = len(vertices)
    
    # Shaft vertices
    for i in range(num_segments + 1):
        theta = 2 * np.pi * i / num_segments
        cross_vec = perp1 * np.cos(theta) + perp2 * np.sin(theta)
        
        # Start of shaft
        v_start = start + cross_vec * shaft_radius
        vertices.append(v_start)
        
        # End of shaft
        v_end = start + direction * shaft_length + cross_vec * shaft_radius
        vertices.append(v_end)
    
    # Shaft faces
    for i in range(num_segments):
        v1 = vertex_offset + i * 2
        v2 = vertex_offset + (i + 1) * 2
        v3 = vertex_offset + i * 2 + 1
        v4 = vertex_offset + (i + 1) * 2 + 1
        
        faces.append([v1, v2, v3])
        faces.append([v2, v4, v3])
    
    # Arrow head
    head_start = start + direction * shaft_length
    head_tip = start + direction * length
    
    vertex_offset_head = len(vertices)
    
    # Head base vertices
    for i in range(num_segments):
        theta = 2 * np.pi * i / num_segments
        cross_vec = perp1 * np.cos(theta) + perp2 * np.sin(theta)
        v_base = head_start + cross_vec * head_radius
        vertices.append(v_base)
    
    # Head tip
    vertices.append(head_tip)
    
    # Head faces
    tip_vertex = len(vertices) - 1
    for i in range(num_segments):
        v1 = vertex_offset_head + i
        v2 = vertex_offset_head + ((i + 1) % num_segments)
        faces.append([v1, v2, tip_vertex])
    
    return np.array(vertices), faces


def export_camera_poses_to_ply(
    camera_poses_file: str,
    output_file: str,
    image_width: int,
    image_height: int,
    sphere_radius: float = 5.0,
    arrow_length: float = 20.0,
    arrow_shaft_radius: float = 0.5,
    arrow_head_length: float = 5.0,
    arrow_head_radius: float = 2.0,
    focal_length_mm: float = 8.8,
    sensor_width_mm: float = 13.2,
    draw_footprints: bool = True
):
    """
    Export camera poses to PLY format for 3D visualization.
    
    Args:
        camera_poses_file: Path to camera_poses.json file
        output_file: Path to output PLY file
        sphere_radius: Radius of camera center spheres
        arrow_length: Length of orientation arrows
        arrow_shaft_radius: Radius of arrow shaft
        arrow_head_length: Length of arrow head
        arrow_head_radius: Radius of arrow head base
    """
    # Load camera poses
    with open(camera_poses_file, 'r') as f:
        poses = json.load(f)
    
    # Find first pose with GPS to use as origin
    origin_lat = None
    origin_lon = None
    origin_alt = None
    for pose in poses:
        if pose.get('gps'):
            origin_lat = pose['gps']['latitude']
            origin_lon = pose['gps']['longitude']
            origin_alt = pose['gps']['altitude']
            break
    
    if origin_lat is None:
        raise ValueError("No GPS data found in camera poses")
    
    all_vertices = []
    all_faces = []
    all_colors = []
    
    # Color scheme
    camera_color = [255, 0, 0]  # Red for camera centers
    arrow_color = [0, 0, 255]   # Blue for orientation arrows
    ground_color = [200, 200, 200]  # Light gray for ground plane
    footprint_color = [0, 255, 0]  # Green for image footprints
    footprint_outline_color = [255, 0, 0]  # Red for footprint outlines
    
    # Calculate focal length in pixels
    focal_length_px = (focal_length_mm / sensor_width_mm) * image_width
    
    # Calculate sensor height from aspect ratio
    sensor_height_mm = sensor_width_mm * (image_height / image_width)
    
    vertex_count = 0
    
    # First, collect all camera positions to determine ground plane extent
    camera_positions = []
    for pose in poses:
        if not pose.get('gps'):
            continue
        
        lat = pose['gps']['latitude']
        lon = pose['gps']['longitude']
        alt = pose['gps']['altitude']
        
        lat_diff = lat - origin_lat
        lon_diff = lon - origin_lon
        
        x = lon_diff * 111000.0 * np.cos(np.radians(origin_lat))
        y = lat_diff * 111000.0
        z = alt - origin_alt
        
        camera_positions.append((x, y, z))
    
    if not camera_positions:
        raise ValueError("No camera positions found")
    
    # Calculate ground plane extent with some padding
    x_coords = [p[0] for p in camera_positions]
    y_coords = [p[1] for p in camera_positions]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Add padding (10% of range)
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    x_min -= x_padding
    x_max += x_padding
    y_min -= y_padding
    y_max += y_padding
    
    # Create ground plane at z=0 (relative to origin altitude)
    # Create a grid of vertices for the ground plane
    ground_resolution = 20  # Number of grid points per side
    ground_vertices = []
    ground_faces = []
    
    for i in range(ground_resolution + 1):
        for j in range(ground_resolution + 1):
            x = x_min + (x_max - x_min) * i / ground_resolution
            y = y_min + (y_max - y_min) * j / ground_resolution
            z = 0.0  # Ground plane at z=0
            ground_vertices.append([x, y, z])
    
    # Create faces (quads) for the ground plane
    for i in range(ground_resolution):
        for j in range(ground_resolution):
            v1 = i * (ground_resolution + 1) + j
            v2 = i * (ground_resolution + 1) + (j + 1)
            v3 = (i + 1) * (ground_resolution + 1) + (j + 1)
            v4 = (i + 1) * (ground_resolution + 1) + j
            
            # Two triangles per quad
            ground_faces.append([v1, v2, v3])
            ground_faces.append([v1, v3, v4])
    
    # Add ground plane to vertices and faces
    for vertex in ground_vertices:
        all_vertices.append(vertex)
        all_colors.append(ground_color)
    
    for face in ground_faces:
        all_faces.append(face)
    
    vertex_count = len(ground_vertices)
    
    print(f'Created ground plane at z=0:')
    print(f'  X range: [{x_min:.2f}, {x_max:.2f}] meters')
    print(f'  Y range: [{y_min:.2f}, {y_max:.2f}] meters')
    print(f'  Grid resolution: {ground_resolution}x{ground_resolution}')
    print()
    
    # Find first pose with GPS to use as origin
    origin_lat = None
    origin_lon = None
    origin_alt = None
    for pose in poses:
        if pose.get('gps'):
            origin_lat = pose['gps']['latitude']
            origin_lon = pose['gps']['longitude']
            origin_alt = pose['gps']['altitude']
            break
    
    if origin_lat is None:
        raise ValueError("No GPS data found in camera poses")
    
    # Process each camera (skip position collection since we already did it)
    camera_idx = 0
    for i, pose in enumerate(poses):
        if not pose.get('gps'):
            continue
        
        # Get camera position
        lat = pose['gps']['latitude']
        lon = pose['gps']['longitude']
        alt = pose['gps']['altitude']
        
        # Convert to local coordinates (same method as used in reconstruction)
        # Simple lat/lon to meters conversion for small areas
        # Use pre-calculated position
        if camera_idx < len(camera_positions):
            x, y, z = camera_positions[camera_idx]
            camera_center = np.array([x, y, z])
            camera_idx += 1
        else:
            # Fallback calculation
            lat_diff = lat - origin_lat
            lon_diff = lon - origin_lon
            
            x = lon_diff * 111000.0 * np.cos(np.radians(origin_lat))
            y = lat_diff * 111000.0
            z = alt - origin_alt
            camera_center = np.array([x, y, z])
        
        # Get camera orientation
        camera_direction = None
        
        # Try to get rotation from DJI orientation
        if pose.get('dji_orientation'):
            from .dji_exif_parser import get_camera_rotation_from_dji_orientation
            R = get_camera_rotation_from_dji_orientation(pose['dji_orientation'])
            # R is world-to-camera, so R^T is camera-to-world
            # Camera Z-axis (forward, into image) in camera coordinates is [0, 0, 1]
            # Transform to world coordinates: R^T @ [0, 0, 1]
            # For all cameras looking at ground, this should point downward (negative Z in world)
            camera_direction = R.T @ np.array([0, 0, 1])
            
            # Ensure direction points downward (toward ground plane at z=0)
            # If Z component is positive, the direction is pointing upward, so negate it
            if camera_direction[2] > 0:
                camera_direction = -camera_direction
        else:
            # Default: point down (nadir)
            camera_direction = np.array([0, 0, -1])
        
        # Create sphere for camera center
        sphere_vertices, sphere_faces = create_sphere(camera_center, sphere_radius)
        
        for vertex in sphere_vertices:
            all_vertices.append(vertex)
            all_colors.append(camera_color)
        
        for face in sphere_faces:
            all_faces.append([f + vertex_count for f in face])
        
        vertex_count += len(sphere_vertices)
        
        # Create arrow for camera orientation
        arrow_vertices, arrow_faces = create_arrow(
            camera_center,
            camera_direction,
            arrow_length,
            arrow_shaft_radius,
            arrow_head_length,
            arrow_head_radius
        )
        
        for vertex in arrow_vertices:
            all_vertices.append(vertex)
            all_colors.append(arrow_color)
        
        for face in arrow_faces:
            all_faces.append([f + vertex_count for f in face])
        
        vertex_count += len(arrow_vertices)
        
        # Create image footprint on ground plane
        if draw_footprints:
            # Try to compute footprint from EXIF using Footprint class
            footprint_corners = None
            if pose.get('gps') and pose.get('dji_orientation'):
                footprint_corners = compute_footprint_from_exif(
                    gps=pose['gps'],
                    dji_orientation=pose['dji_orientation'],
                    focal_length_mm=focal_length_mm,
                    sensor_width_mm=sensor_width_mm,
                    sensor_height_mm=sensor_height_mm,
                    origin_lat=origin_lat,
                    origin_lon=origin_lon,
                    origin_alt=origin_alt
                )
            
            # Fallback to projection method if EXIF-based calculation fails
            if footprint_corners is None:
                # For footprint projection, use the typical camera height above ground
                # All cameras are approximately at origin_alt (~100m) above sea level
                # The ground plane in visualization is at z=0 (relative to origin_alt)
                # So camera height above ground plane = approximately origin_alt
                # Use origin_alt as the camera height for all cameras to get consistent footprint sizes
                camera_height_above_ground = origin_alt  # ~100m for all cameras
                
                camera_center_for_projection = np.array([camera_center[0], camera_center[1], camera_height_above_ground])
                
                footprint_corners = project_image_corners_to_ground(
                    camera_center_for_projection,
                    R,
                    image_width,
                    image_height,
                    focal_length_px,
                    ground_z=0.0  # Ground plane is at z=0 (relative)
                )
            
            if footprint_corners is not None and len(footprint_corners) == 4:
                # Add footprint vertices (green polygon)
                footprint_start_idx = vertex_count
                for corner in footprint_corners:
                    all_vertices.append(corner)
                    all_colors.append(footprint_color)
                
                # Create footprint face (quad as two triangles)
                # Order: top-left, top-right, bottom-right, bottom-left
                v0 = footprint_start_idx
                v1 = footprint_start_idx + 1
                v2 = footprint_start_idx + 2
                v3 = footprint_start_idx + 3
                
                # Two triangles: (v0, v1, v2) and (v0, v2, v3)
                all_faces.append([v0, v1, v2])
                all_faces.append([v0, v2, v3])
                
                vertex_count += len(footprint_corners)
                
                # Create red outline around footprint
                outline_vertices, outline_faces = create_polygon_outline(
                    footprint_corners,
                    line_width=1.0,  # 1 meter wide outline
                    line_height=0.5  # 0.5 meters high (slightly above ground)
                )
                
                if len(outline_vertices) > 0:
                    outline_start_idx = vertex_count
                    for vertex in outline_vertices:
                        all_vertices.append(vertex)
                        all_colors.append(footprint_outline_color)
                    
                    for face in outline_faces:
                        all_faces.append([f + outline_start_idx for f in face])
                    
                    vertex_count += len(outline_vertices)
    
    # Write PLY file
    all_vertices = np.array(all_vertices)
    all_colors = np.array(all_colors)
    
    with open(output_file, 'w') as f:
        # PLY header
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(all_vertices)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write(f'element face {len(all_faces)}\n')
        f.write('property list uchar int vertex_indices\n')
        f.write('end_header\n')
        
        # Write vertices
        for i, vertex in enumerate(all_vertices):
            color = all_colors[i]
            f.write(f'{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f} {color[0]} {color[1]} {color[2]}\n')
        
        # Write faces
        for face in all_faces:
            f.write(f'{len(face)} {" ".join(map(str, face))}\n')
    
    # Calculate coordinate ranges
    all_vertices_array = np.array(all_vertices)
    x_range = [all_vertices_array[:, 0].min(), all_vertices_array[:, 0].max()]
    y_range = [all_vertices_array[:, 1].min(), all_vertices_array[:, 1].max()]
    z_range = [all_vertices_array[:, 2].min(), all_vertices_array[:, 2].max()]
    
    print(f'Exported {len(poses)} camera poses to PLY file: {output_file}')
    print(f'  Total vertices: {len(all_vertices)}')
    print(f'  Total faces: {len(all_faces)}')
    print(f'  Ground plane: Gray polygon at z=0 (relative to origin altitude)')
    print(f'  Camera centers: Red spheres (radius={sphere_radius}m)')
    print(f'  Camera orientations: Blue arrows (length={arrow_length}m)')
    if draw_footprints:
        print(f'  Image footprints: Green polygons on ground plane')
        print(f'  Footprint outlines: Red borders around each footprint')
    print(f'  Coordinate ranges:')
    print(f'    X: [{x_range[0]:.2f}, {x_range[1]:.2f}] meters')
    print(f'    Y: [{y_range[0]:.2f}, {y_range[1]:.2f}] meters')
    print(f'    Z: [{z_range[0]:.2f}, {z_range[1]:.2f}] meters (altitude variation)')
