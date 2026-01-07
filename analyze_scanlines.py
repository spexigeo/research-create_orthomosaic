"""
Analyze scanlines and verify user-provided scanline assignments.
"""
import json
import numpy as np
from pathlib import Path
import re
from typing import List, Dict


def parse_image_number(image_name: str) -> int:
    """Extract image number from filename."""
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
    return (heading + 360) % 360


def analyze_scanline_straightness(poses_file: str, scanline_ranges: List[tuple]) -> Dict:
    """
    Analyze how well images in each scanline lie on a straight line.
    
    Args:
        poses_file: Path to camera poses JSON
        scanline_ranges: List of (start_image, end_image) tuples for each scanline
    """
    # Load poses
    with open(poses_file, 'r') as f:
        poses = json.load(f)
    
    poses_dict = {parse_image_number(p['image_name']): p for p in poses if p.get('gps')}
    
    results = {}
    for scanline_id, (start_img, end_img) in enumerate(scanline_ranges, 1):
        image_nums = list(range(start_img, end_img + 1))
        
        # Get positions for this scanline
        positions = []
        valid_images = []
        for img_num in image_nums:
            if img_num in poses_dict:
                p = poses_dict[img_num]
                positions.append([p['gps']['latitude'], p['gps']['longitude']])
                valid_images.append(img_num)
        
        if len(positions) < 2:
            results[scanline_id] = {
                'image_numbers': valid_images,
                'num_images': len(valid_images),
                'rms_error_meters': None,
                'avg_heading_std': None,
                'note': 'Too few images'
            }
            continue
        
        positions = np.array(positions)
        
        # Calculate headings between consecutive images
        headings = []
        for i in range(len(positions) - 1):
            heading = calculate_heading(positions[i][0], positions[i][1],
                                       positions[i+1][0], positions[i+1][1])
            headings.append(heading)
        
        # Normalize headings to handle wrap-around
        headings = np.array(headings)
        headings_centered = headings - headings.mean()
        headings_centered = (headings_centered + 180) % 360 - 180
        
        # Fit a line to the positions (using PCA)
        center = positions.mean(axis=0)
        centered = positions - center
        
        cov = np.cov(centered.T)
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        principal_dir = eigenvecs[:, np.argmax(eigenvals)]
        
        # Project points onto the line
        projections = np.dot(centered, principal_dir)
        projected_points = center + np.outer(projections, principal_dir)
        
        # Calculate RMS distance from line
        distances = np.linalg.norm(positions - projected_points, axis=1)
        rms_error = np.sqrt(np.mean(distances**2))
        rms_error_m = rms_error * 111320  # Convert to meters
        
        # Heading consistency (standard deviation)
        heading_std = np.std(headings_centered)
        
        results[scanline_id] = {
            'image_numbers': valid_images,
            'num_images': len(valid_images),
            'rms_error_meters': rms_error_m,
            'avg_heading': float(np.mean(headings)),
            'heading_std': float(heading_std),
            'start_image': min(valid_images),
            'end_image': max(valid_images)
        }
    
    return results


def auto_detect_scanlines_by_heading(poses_file: str, min_scanline_length: int = 5) -> Dict:
    """
    Detect scanlines by finding groups of images with consistent headings.
    """
    # Load poses
    with open(poses_file, 'r') as f:
        poses = json.load(f)
    
    poses_with_gps = [p for p in poses if p.get('gps') is not None]
    poses_with_gps.sort(key=lambda p: parse_image_number(p['image_name']))
    
    image_numbers = [parse_image_number(p['image_name']) for p in poses_with_gps]
    lats = np.array([p['gps']['latitude'] for p in poses_with_gps])
    lons = np.array([p['gps']['longitude'] for p in poses_with_gps])
    
    # Calculate headings
    headings = []
    for i in range(len(lats) - 1):
        heading = calculate_heading(lats[i], lons[i], lats[i+1], lons[i+1])
        headings.append(heading)
    
    headings = np.array(headings)
    
    # Group images with consistent headings
    scanlines = {}
    current_scanline = [image_numbers[0]]
    current_heading = headings[0] if len(headings) > 0 else None
    scanline_id = 1
    
    for i in range(1, len(image_numbers) - 1):
        if i < len(headings):
            heading = headings[i]
            heading_diff = abs(heading - current_heading)
            if heading_diff > 180:
                heading_diff = 360 - heading_diff
            
            # If heading change is small, continue current scanline
            if heading_diff < 30:  # 30 degree threshold
                current_scanline.append(image_numbers[i])
                current_heading = heading
            else:
                # Start new scanline
                if len(current_scanline) >= min_scanline_length:
                    scanlines[scanline_id] = current_scanline
                    scanline_id += 1
                current_scanline = [image_numbers[i]]
                current_heading = heading
        else:
            current_scanline.append(image_numbers[i])
    
    # Add last scanline
    if len(current_scanline) >= min_scanline_length:
        scanlines[scanline_id] = current_scanline
    
    return scanlines


if __name__ == "__main__":
    # User-provided scanline ranges
    user_scanlines = [
        (1, 14),    # scanline 1
        (15, 29),   # scanline 2
        (30, 48),   # scanline 3
        (49, 68),   # scanline 4
        (69, 91),   # scanline 5
        (92, 111),  # scanline 6
        (112, 129), # scanline 7
        (130, 144), # scanline 8
        (145, 155), # scanline 9 (assuming 155 is the last)
    ]
    
    print("Analyzing user-provided scanline assignments...")
    results = analyze_scanline_straightness("outputs/camera_poses.json", user_scanlines)
    
    print("\nScanline Analysis Results:")
    for scanline_id, metrics in sorted(results.items()):
        if metrics['rms_error_meters'] is not None:
            print(f"  Scanline {scanline_id}: Images {metrics['start_image']:04d}-{metrics['end_image']:04d} "
                  f"({metrics['num_images']} images)")
            print(f"    RMS error from straight line: {metrics['rms_error_meters']:.2f} m")
            print(f"    Heading consistency (std): {metrics['heading_std']:.1f}°")
            print(f"    Average heading: {metrics['avg_heading']:.1f}°")
        else:
            print(f"  Scanline {scanline_id}: {metrics['note']}")
    
    print("\n" + "="*60)
    print("Auto-detecting scanlines by heading consistency...")
    auto_scanlines = auto_detect_scanlines_by_heading("outputs/camera_poses.json", min_scanline_length=5)
    
    print(f"\nAuto-detected {len(auto_scanlines)} scanlines:")
    for scanline_id, image_nums in sorted(auto_scanlines.items()):
        print(f"  Scanline {scanline_id}: {min(image_nums):04d}-{max(image_nums):04d} ({len(image_nums)} images)")
    
    # Save both results
    output_data = {
        'user_scanlines': {str(k): v for k, v in results.items()},
        'auto_detected_scanlines': {str(k): v for k, v in auto_scanlines.items()}
    }
    
    with open("outputs/scanlines_analysis.json", 'w') as f:
        json.dump(output_data, f, indent=2)
    print("\nSaved analysis to: outputs/scanlines_analysis.json")
