"""
Extract camera pose information from image EXIF metadata.
"""
import json
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import numpy as np
from typing import Dict, List, Optional, Tuple


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
            from tiepoint_matcher.dji_exif_parser import extract_dji_orientation
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


def extract_poses_from_directory(image_dir: str, pattern: str = "*.jpg") -> List[Dict]:
    """Extract camera poses from all images in a directory."""
    image_dir_path = Path(image_dir)
    image_files = sorted(image_dir_path.glob(pattern))
    
    poses = []
    for img_path in image_files:
        pose = extract_camera_pose(str(img_path))
        poses.append(pose)
    
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
